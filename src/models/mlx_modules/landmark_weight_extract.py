"""Extract weights for the ConvNeXt-Tiny landmark model from landmark.onnx.

This ONNX export stores some parameters as named initializers (Conv weights,
dwconv, biases, intra-block LN, gamma, fc_*, norm.*, norm_s3.*) but other
parameters as anonymous `onnx::MatMul_NNN` / `onnx::Mul_NNN` / `onnx::Add_NNN`
constants. We walk the graph to recover the missing assignments:

  * pwconv1 / pwconv2 weights (anonymous MatMul constants) -> per block
  * stem LayerNorm gamma/beta and inter-stage LayerNorm gamma/beta
    (anonymous Mul/Add constants) -> per downsample stage
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import mlx.core as mx
import mlx.utils as mu


_BLOCK_RE = re.compile(r"^(stages\.\d+\.\d+)\.dwconv\.weight$")


def _block_pwconv_map(model) -> Dict[str, Tuple[str, str]]:
    """For each block stages.X.Y, return (pwconv1_const_name, pwconv2_const_name)."""
    in_to_nodes: Dict[str, list] = {}
    for n in model.graph.node:
        for inp in n.input:
            in_to_nodes.setdefault(inp, []).append(n)

    blocks = {}
    for n in model.graph.node:
        if n.op_type != "Conv":
            continue
        m = _BLOCK_RE.match(n.input[1] if len(n.input) > 1 else "")
        if not m:
            continue
        block_key = m.group(1)
        seen = set()
        queue = list(n.output)
        matmuls = []
        while queue and len(matmuls) < 2:
            next_q = []
            for q in queue:
                for child in in_to_nodes.get(q, []):
                    if id(child) in seen:
                        continue
                    seen.add(id(child))
                    if child.op_type == "MatMul":
                        matmuls.append(child)
                        if len(matmuls) >= 2:
                            break
                    next_q.extend(child.output)
            queue = next_q
        if len(matmuls) >= 2:
            # MatMul stores W as input[1] (rhs of x @ W).
            blocks[block_key] = (matmuls[0].input[1], matmuls[1].input[1])
    return blocks


def _walk_to_ln_tail_after(model, start_out: str,
                            init_names: set) -> Tuple[str, str] | None:
    """Walk forward from start_out following the unique-child chain and find
    the next (Mul, Add) pair where Mul has a constant input and Add has a
    constant input. Returns (gamma_name, beta_name) or None.

    The Mul/Add tail of a LayerNorm decomposition: y = (norm * gamma) + beta.
    Each constant can sit in either input slot of the Mul/Add (commutative).
    """
    in_to_nodes: Dict[str, list] = {}
    for n in model.graph.node:
        for inp in n.input:
            in_to_nodes.setdefault(inp, []).append(n)

    cur = start_out
    for _ in range(40):
        children = in_to_nodes.get(cur, [])
        if not children:
            return None
        c = children[0]
        if c.op_type == "Mul":
            const_in = next((x for x in c.input if x in init_names), None)
            if const_in is not None:
                # Look at the Mul's child for the Add tail.
                next_children = in_to_nodes.get(c.output[0], [])
                if next_children and next_children[0].op_type == "Add":
                    nc = next_children[0]
                    beta_in = next((x for x in nc.input if x in init_names), None)
                    if beta_in is not None:
                        return const_in, beta_in
        cur = c.output[0]
    return None


def _walk_back_to_ln_tail_before(model, target_out: str,
                                  init_names: set) -> Tuple[str, str] | None:
    """Walk backwards from a node consuming target_out — find the Mul/Add LN
    tail whose Add output IS target_out."""
    out_to_node = {o: n for n in model.graph.node for o in n.output}
    add_node = out_to_node.get(target_out)
    if add_node is None or add_node.op_type != "Add":
        return None
    beta = next((x for x in add_node.input if x in init_names), None)
    other = next((x for x in add_node.input if x not in init_names), None)
    if beta is None or other is None:
        return None
    mul_node = out_to_node.get(other)
    if mul_node is None or mul_node.op_type != "Mul":
        return None
    gamma = next((x for x in mul_node.input if x in init_names), None)
    if gamma is None:
        return None
    return gamma, beta


def load_landmark_from_onnx(model, onnx_path: str) -> None:
    import onnx

    onnx_model = onnx.load(onnx_path)
    init_map = {init.name: onnx.numpy_helper.to_array(init) for init in onnx_model.graph.initializer}
    init_names = set(init_map.keys())
    block_pw = _block_pwconv_map(onnx_model)

    new_state: Dict[str, mx.array] = {}

    def put(key: str, arr: np.ndarray):
        new_state[key] = mx.array(np.ascontiguousarray(arr.astype(np.float32)))

    # ---- Stem (downsample_layers.0) ----
    stem_w = init_map["downsample_layers.0.0.weight"]  # PyTorch (96, 3, 4, 4)
    stem_b = init_map["downsample_layers.0.0.bias"]
    put("downsample_layers.0.0.weight", np.transpose(stem_w, (0, 2, 3, 1)))
    put("downsample_layers.0.0.bias", stem_b)
    # Stem LayerNorm: walk forward from stem-conv output 'x0' to find the LN tail.
    stem_conv = next(n for n in onnx_model.graph.node if n.op_type == "Conv"
                     and len(n.input) > 1 and n.input[1] == "downsample_layers.0.0.weight")
    stem_ln = _walk_to_ln_tail_after(onnx_model, stem_conv.output[0], init_names)
    if stem_ln is None:
        raise ValueError("could not locate stem LayerNorm tail")
    g_const, b_const = stem_ln
    g = init_map[g_const].reshape(-1)  # collapse any (96, 1, 1) shape
    b = init_map[b_const].reshape(-1)
    put("downsample_layers.0.1.weight", g)
    put("downsample_layers.0.1.bias", b)

    # ---- Inter-stage downsamples (downsample_layers.{1,2,3}) ----
    for i in range(1, 4):
        wkey = f"downsample_layers.{i}.1.weight"
        bkey = f"downsample_layers.{i}.1.bias"
        put(wkey, np.transpose(init_map[wkey], (0, 2, 3, 1)))
        put(bkey, init_map[bkey])
        # Find the LN tail that feeds this 2x2 conv.
        conv_node = next(n for n in onnx_model.graph.node if n.op_type == "Conv"
                         and len(n.input) > 1 and n.input[1] == wkey)
        ln_pair = _walk_back_to_ln_tail_before(onnx_model, conv_node.input[0], init_names)
        if ln_pair is None:
            raise ValueError(f"could not locate LN tail for downsample_layers.{i}.0")
        g_const, b_const = ln_pair
        g = init_map[g_const].reshape(-1)
        b = init_map[b_const].reshape(-1)
        put(f"downsample_layers.{i}.0.weight", g)
        put(f"downsample_layers.{i}.0.bias", b)

    # ---- Per-block params ----
    for stage_idx in range(4):
        for blk_idx in range(model.DEPTHS[stage_idx]):
            base = f"stages.{stage_idx}.{blk_idx}"
            put(f"{base}.dwconv.weight", np.transpose(init_map[f"{base}.dwconv.weight"], (0, 2, 3, 1)))
            put(f"{base}.dwconv.bias", init_map[f"{base}.dwconv.bias"])
            put(f"{base}.norm.weight", init_map[f"{base}.norm.weight"])
            put(f"{base}.norm.bias", init_map[f"{base}.norm.bias"])
            put(f"{base}.gamma", init_map[f"{base}.gamma"])
            w1, w2 = block_pw[base]
            # ONNX MatMul stores W as (in, out). nn.Linear expects (out, in).
            put(f"{base}.pwconv1.weight", init_map[w1].T)
            put(f"{base}.pwconv1.bias", init_map[f"{base}.pwconv1.bias"])
            put(f"{base}.pwconv2.weight", init_map[w2].T)
            put(f"{base}.pwconv2.bias", init_map[f"{base}.pwconv2.bias"])

    # ---- Final norms + heads ----
    for k in ("norm.weight", "norm.bias", "norm_s3.weight", "norm_s3.bias"):
        put(k, init_map[k])
    for head in ("fc_coeff", "fc_lmk", "fc_pts"):
        put(f"{head}.weight", init_map[f"{head}.weight"])
        put(f"{head}.bias", init_map[f"{head}.bias"])

    expected = {k for k, _ in mu.tree_flatten(model.parameters())}
    missing = expected - set(new_state.keys())
    extra = set(new_state.keys()) - expected
    if missing or extra:
        raise ValueError(
            f"landmark state mismatch.\n  missing: {sorted(missing)[:8]}\n  extra: {sorted(extra)[:8]}"
        )
    model.update(mu.tree_unflatten(list(new_state.items())))


def load_landmark_from_npz(model, npz_path: str) -> None:
    with np.load(npz_path) as data:
        new_state = {
            key: mx.array(np.ascontiguousarray(data[key].astype(np.float32)))
            for key in data.files
        }

    expected = {k for k, _ in mu.tree_flatten(model.parameters())}
    missing = expected - set(new_state.keys())
    extra = set(new_state.keys()) - expected
    if missing or extra:
        raise ValueError(
            f"landmark state mismatch.\n  missing: {sorted(missing)[:8]}\n  extra: {sorted(extra)[:8]}"
        )
    model.update(mu.tree_unflatten(list(new_state.items())))
