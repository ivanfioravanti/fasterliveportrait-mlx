#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export source checkpoint containers into runtime MLX npz weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

import mlx.utils as mu
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mlx_modules.landmark import LandmarkModel
from src.models.mlx_modules.landmark_weight_extract import load_landmark_from_onnx
from src.models.mlx_modules.weight_convert import save_converted_npz
from src.models.mlx_joyvasa_audio_model import export_mlx_joyvasa_audio_from_pytorch_checkpoint
from src.models.mlx_joyvasa_motion_model import export_mlx_joyvasa_motion_from_pytorch_checkpoint


_OFFICIAL_KEY_BY_MODULE = {
    "stitching": "retarget_shoulder",
    "stitching_retarget": "retarget_shoulder",
    "retarget_shoulder": "retarget_shoulder",
    "stitching_eye": "retarget_eye",
    "stitching_eye_retarget": "retarget_eye",
    "eye": "retarget_eye",
    "retarget_eye": "retarget_eye",
    "stitching_lip": "retarget_mouth",
    "stitching_lip_retarget": "retarget_mouth",
    "lip": "retarget_mouth",
    "mouth": "retarget_mouth",
    "retarget_mouth": "retarget_mouth",
}

_LAYER_RE = re.compile(r"(?:^|\.)(\d+)\.weight$")


def _layer_index(name: str) -> int:
    match = _LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot infer MLP layer index from {name}")
    return int(match.group(1))


def _load_stitching_from_onnx(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    tensors: Dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init).astype(np.float32)
        for init in model.graph.initializer
    }
    weight_names = sorted(
        (name for name in tensors if name.endswith(".weight")),
        key=_layer_index,
    )
    layers = []
    for weight_name in weight_names:
        bias_name = weight_name.replace(".weight", ".bias")
        if bias_name not in tensors:
            raise ValueError(f"Missing bias tensor {bias_name} in {path}")
        layers.append((tensors[weight_name], tensors[bias_name]))
    if not layers:
        raise ValueError(f"No MLP weights found in {path}")
    return layers


def _load_stitching_from_torch(path: str, module_name: str | None) -> List[Tuple[np.ndarray, np.ndarray]]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if module_name is not None and module_name in _OFFICIAL_KEY_BY_MODULE:
        checkpoint = checkpoint[_OFFICIAL_KEY_BY_MODULE[module_name]]
    weight_names = sorted(
        (name for name in checkpoint if name.endswith(".weight")),
        key=_layer_index,
    )
    layers = []
    for weight_name in weight_names:
        bias_name = weight_name.replace(".weight", ".bias")
        if bias_name not in checkpoint:
            raise ValueError(f"Missing bias tensor {bias_name} in {path}")
        weight = checkpoint[weight_name].detach().cpu().numpy().astype(np.float32)
        bias = checkpoint[bias_name].detach().cpu().numpy().astype(np.float32)
        layers.append((weight, bias))
    if not layers:
        raise ValueError(f"No MLP weights found in {path}")
    return layers


APPEARANCE_KEY_RENAMES = ((r"resblocks_3d\.3dr(\d+)\.", r"resblocks_3d.\1."),)

HUMAN_CORE = (
    ("liveportrait_torch/warping_module.pth", "liveportrait_mlx/warping_module.npz", {}),
    ("liveportrait_torch/spade_generator.pth", "liveportrait_mlx/spade_generator.npz", {}),
    (
        "liveportrait_torch/motion_extractor.pth",
        "liveportrait_mlx/motion_extractor.npz",
        {"dropping_prefix": "detector."},
    ),
    (
        "liveportrait_torch/appearance_feature_extractor.pth",
        "liveportrait_mlx/appearance_feature_extractor.npz",
        {"key_renames": APPEARANCE_KEY_RENAMES},
    ),
)

ANIMAL_CORE = (
    (
        "liveportrait_animals/base_models_v1.1/warping_module.pth",
        "liveportrait_animal_mlx/base_models_v1.1/warping_module.npz",
        {},
    ),
    (
        "liveportrait_animals/base_models_v1.1/spade_generator.pth",
        "liveportrait_animal_mlx/base_models_v1.1/spade_generator.npz",
        {},
    ),
    (
        "liveportrait_animals/base_models_v1.1/motion_extractor.pth",
        "liveportrait_animal_mlx/base_models_v1.1/motion_extractor.npz",
        {"dropping_prefix": "detector."},
    ),
    (
        "liveportrait_animals/base_models_v1.1/appearance_feature_extractor.pth",
        "liveportrait_animal_mlx/base_models_v1.1/appearance_feature_extractor.npz",
        {"key_renames": APPEARANCE_KEY_RENAMES},
    ),
)

HUMAN_STITCHING = (
    ("stitching", "retarget_shoulder", "liveportrait_onnx/stitching.onnx"),
    ("stitching_eye", "retarget_eye", "liveportrait_onnx/stitching_eye.onnx"),
    ("stitching_lip", "retarget_mouth", "liveportrait_onnx/stitching_lip.onnx"),
)

OFFICIAL_STITCHING = "liveportrait_torch/stitching_retargeting_module.pth"
JOYVASA_MOTION_SRC = "JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"
JOYVASA_MOTION_DST = "JoyVASA/motion_generator/motion_generator_hubert_chinese_mlx.npz"
JOYVASA_AUDIO_ENCODER_SRC = "chinese-hubert-base"
JOYVASA_AUDIO_DST = "JoyVASA/audio_encoder/hubert_chinese_mlx.npz"


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    print(f"wrote {path}")


def export_landmark(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"skip missing {src}")
        return
    model = LandmarkModel()
    load_landmark_from_onnx(model, str(src))
    payload = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in mu.tree_flatten(model.parameters())
    }
    _save_npz(dst, payload)


def export_stitching_layers(layers, dst: Path) -> None:
    payload = {}
    for idx, (weight, bias) in enumerate(layers):
        payload[f"layers.{idx}.weight"] = weight.astype(np.float32)
        payload[f"layers.{idx}.bias"] = bias.astype(np.float32)
    _save_npz(dst, payload)


def export_stitching(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"skip missing {src}")
        return
    export_stitching_layers(_load_stitching_from_onnx(str(src)), dst)


def export_core_models(checkpoints_dir: Path, entries) -> None:
    for rel_src, rel_dst, options in entries:
        src = checkpoints_dir / rel_src
        dst = checkpoints_dir / rel_dst
        if not src.exists():
            print(f"skip missing {src}")
            continue
        save_converted_npz(str(src), str(dst), **options)
        print(f"wrote {dst}")


def export_joyvasa_motion(checkpoints_dir: Path, src_rel: str, dst_rel: str) -> None:
    src = checkpoints_dir / src_rel
    dst = checkpoints_dir / dst_rel
    if not src.exists():
        print(f"skip missing {src}")
        return
    export_mlx_joyvasa_motion_from_pytorch_checkpoint(src, dst)
    print(f"wrote {dst}")


def export_joyvasa_audio(checkpoints_dir: Path, motion_src_rel: str, audio_src_rel: str, dst_rel: str) -> None:
    motion_src = checkpoints_dir / motion_src_rel
    audio_src = checkpoints_dir / audio_src_rel
    dst = checkpoints_dir / dst_rel
    if not motion_src.exists():
        print(f"skip missing {motion_src}")
        return
    if not audio_src.exists():
        print(f"skip missing {audio_src}")
        return
    export_mlx_joyvasa_audio_from_pytorch_checkpoint(motion_src, audio_src, dst)
    print(f"wrote {dst}")


def export_human_stitching(checkpoints_dir: Path, dst_dir: Path, source: str, official_path: Path) -> None:
    use_official = source in ("auto", "torch") and official_path.exists()
    if source == "torch" and not official_path.exists():
        raise FileNotFoundError(f"Official stitching checkpoint not found: {official_path}")

    for name, module_name, rel_onnx in HUMAN_STITCHING:
        dst = dst_dir / f"{name}.npz"
        if use_official:
            export_stitching_layers(_load_stitching_from_torch(str(official_path), module_name), dst)
        else:
            export_stitching(checkpoints_dir / rel_onnx, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint containers to MLX npz weights.")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="checkpoint root")
    parser.add_argument(
        "--stitching-source",
        choices=("auto", "torch", "onnx"),
        default="auto",
        help="source for human stitching weights; auto prefers the official combined PyTorch checkpoint",
    )
    parser.add_argument(
        "--official-stitching-path",
        default=None,
        help="path to official stitching_retargeting_module.pth",
    )
    parser.add_argument("--include-animal", action="store_true", help="also export animal core MLX weights")
    parser.add_argument("--include-joyvasa", action="store_true", help="also export JoyVASA MLX motion/audio weights")
    parser.add_argument("--joyvasa-motion-src", default=JOYVASA_MOTION_SRC, help="JoyVASA PyTorch motion checkpoint")
    parser.add_argument("--joyvasa-motion-dst", default=JOYVASA_MOTION_DST, help="JoyVASA MLX motion npz output")
    parser.add_argument("--joyvasa-audio-src", default=JOYVASA_AUDIO_ENCODER_SRC, help="JoyVASA HuBERT encoder directory")
    parser.add_argument("--joyvasa-audio-dst", default=JOYVASA_AUDIO_DST, help="JoyVASA MLX audio npz output")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    human_out = checkpoints_dir / "liveportrait_mlx"
    official_stitching_path = Path(args.official_stitching_path) if args.official_stitching_path else (
        checkpoints_dir / OFFICIAL_STITCHING
    )

    export_core_models(checkpoints_dir, HUMAN_CORE)
    export_landmark(
        checkpoints_dir / "liveportrait_onnx" / "landmark.onnx",
        human_out / "landmark.npz",
    )
    export_human_stitching(
        checkpoints_dir,
        human_out,
        args.stitching_source,
        official_stitching_path,
    )
    if args.include_animal:
        export_core_models(checkpoints_dir, ANIMAL_CORE)
    if args.include_joyvasa:
        export_joyvasa_motion(checkpoints_dir, args.joyvasa_motion_src, args.joyvasa_motion_dst)
        export_joyvasa_audio(checkpoints_dir, args.joyvasa_motion_src, args.joyvasa_audio_src, args.joyvasa_audio_dst)


if __name__ == "__main__":
    main()
