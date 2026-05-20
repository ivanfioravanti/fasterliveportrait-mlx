#!/usr/bin/env python3
"""Benchmark the experimental Metal 4 TensorOps matmul helper."""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import mlx.core as mx

_TENSOR_OPS_PATH = ROOT / "src" / "models" / "mlx_modules" / "tensor_ops.py"
_SPEC = importlib.util.spec_from_file_location("mlx_tensor_ops", _TENSOR_OPS_PATH)
_TENSOR_OPS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TENSOR_OPS)
tensorops_matmul = _TENSOR_OPS.tensorops_matmul


def _dtype(name: str):
    if name in ("fp16", "float16", "half"):
        return mx.float16
    if name in ("bf16", "bfloat16"):
        return mx.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _time(fn, warmup: int, iters: int):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    values = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        mx.eval(out)
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    dtype = _dtype(args.dtype)
    shapes = [
        (64, 32, 32),
        (256, 256, 256),
        (1024, 512, 256),
        (4096, 512, 64),
        # DenseMotion mask head would like M=65536, K=48706, N=22, but a full
        # im2col materialization would be multi-GB. This shape is a tile-sized
        # proxy for the M/N aspect ratio.
        (4096, 2048, 32),
    ]

    print(f"dtype={dtype}")
    for m, k, n in shapes:
        a = mx.random.normal((m, k)).astype(dtype)
        b = mx.random.normal((k, n)).astype(dtype)
        mx.eval(a, b)

        ref = a.astype(mx.float32) @ b.astype(mx.float32)
        got = tensorops_matmul(a, b)
        mx.eval(ref, got)
        max_diff = float(mx.max(mx.abs(ref - got)))
        mean_diff = float(mx.mean(mx.abs(ref - got)))

        native = _time(lambda: a.astype(mx.float32) @ b.astype(mx.float32), args.warmup, args.iters)
        metal4 = _time(lambda: tensorops_matmul(a, b), args.warmup, args.iters)
        print(
            f"{m}x{k} @ {k}x{n}: "
            f"native p50={statistics.median(native):.3f} ms, "
            f"tensorops p50={statistics.median(metal4):.3f} ms, "
            f"diff max={max_diff:.6g}, mean={mean_diff:.6g}"
        )


if __name__ == "__main__":
    main()
