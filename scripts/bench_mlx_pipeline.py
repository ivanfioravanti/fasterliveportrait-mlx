#!/usr/bin/env python3
"""Benchmark FasterLivePortrait-MLX inference components.

Run with uv so the same project environment is used as the app:

    uv run python scripts/bench_mlx_pipeline.py
    uv run python scripts/bench_mlx_pipeline.py --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from itertools import cycle

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.mlx_profiles import MLX_PROFILE_CHOICES, apply_mlx_profile


PROFILE_ENV_KEYS = (
    "FLP_MLX_2D_MASK",
    "FLP_MLX_MASK_BACKEND",
    "FLP_MLX_COMPRESS_BACKEND",
    "FLP_MLX_COMPILE_HOURGLASS",
    "FLP_MLX_COMPILE_SPADE",
    "FLP_MLX_COMPILE_WARPING",
    "FLP_MLX_COMPILE_MOTION",
    "FLP_MLX_COMPILE_APPEARANCE",
    "FLP_MLX_CACHE_SOURCE_GAUSSIAN",
    "FLP_MLX_FUSED_UINT8",
    "FLP_MLX_FUSED_DEFORMATION",
    "FLP_MLX_FUSED_SPARSE_SAMPLE",
    "FLP_MLX_FUSED_HOURGLASS_INPUT",
    "FLP_MLX_SKIP_OCCLUSION",
    "FLP_MLX_CONV3D_BACKEND",
    "FLP_MLX_GS3D_GATHER",
    "FLP_MLX_WARP_OUT_BACKEND",
    "FLP_MLX_WARP_FOURTH_BACKEND",
    "FLP_MLX_SPADE_BF16_NATIVE_NORM",
    "FLP_MLX_SPADE_SHORTCUT_BACKEND",
    "FLP_MLX_SPADE_SHORTCUT_MIN_OUT",
    "FLP_MLX_TEMPORAL_WARP_INTERVAL",
    "FLP_MLX_TEMPORAL_WARP_THRESHOLD",
)

SWEEP_VARIANTS = {
    "base": {},
    "warp_out_standard": {"FLP_MLX_WARP_OUT_BACKEND": "standard"},
    "fused_deformation": {"FLP_MLX_FUSED_DEFORMATION": "1"},
    "fused_sparse_sample": {"FLP_MLX_FUSED_SPARSE_SAMPLE": "1"},
    "fused_hourglass_input": {"FLP_MLX_FUSED_HOURGLASS_INPUT": "1"},
    "warp_out_direct": {"FLP_MLX_WARP_OUT_BACKEND": "direct"},
    "warp_out_c4": {"FLP_MLX_WARP_OUT_BACKEND": "c4"},
    "mask_packed2d_stack": {"FLP_MLX_MASK_BACKEND": "packed2d_stack"},
    "compress_tensorops": {"FLP_MLX_COMPRESS_BACKEND": "tensorops"},
    "fourth_tensorops": {"FLP_MLX_WARP_FOURTH_BACKEND": "tensorops"},
    "mask_tensorops": {"FLP_MLX_MASK_BACKEND": "tensorops"},
}


def parse_set(values: list[str]) -> dict[str, str]:
    settings = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--set expects KEY=VALUE, got {value!r}")
        key, val = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set expects KEY=VALUE, got {value!r}")
        settings[key] = val.strip()
    return settings


def dtype_from_name(name: str):
    import mlx.core as mx

    if name in ("fp32", "float32"):
        return mx.float32
    if name in ("fp16", "float16", "half"):
        return mx.float16
    if name in ("bf16", "bfloat16"):
        return mx.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def time_call(name: str, fn, *, warmup: int, iters: int) -> dict[str, float | str]:
    import mlx.core as mx

    for _ in range(warmup):
        out = fn()
        if isinstance(out, (list, tuple)):
            mx.eval(*[x for x in out if isinstance(x, mx.array)])
        elif isinstance(out, mx.array):
            mx.eval(out)

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        if isinstance(out, (list, tuple)):
            mx.eval(*[x for x in out if isinstance(x, mx.array)])
        elif isinstance(out, mx.array):
            mx.eval(out)
        samples.append((time.perf_counter() - start) * 1000.0)

    p50_ms = statistics.median(samples)
    mean_ms = statistics.mean(samples)
    return {
        "name": name,
        "p50_ms": p50_ms,
        "mean_ms": mean_ms,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_fps": 1000.0 / p50_ms if p50_ms > 0 else float("inf"),
        "mean_fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


def active_settings() -> dict[str, str]:
    return {key: os.environ[key] for key in PROFILE_ENV_KEYS if key in os.environ}


def print_result(result: dict[str, float | str]) -> None:
    print(
        f"{result['name']}: "
        f"p50={result['p50_ms']:.2f} ms, "
        f"p50_fps={result['p50_fps']:.2f}, "
        f"mean={result['mean_ms']:.2f} ms, "
        f"mean_fps={result['mean_fps']:.2f}, "
        f"min={result['min_ms']:.2f} ms, "
        f"max={result['max_ms']:.2f} ms"
    )


def run_sweep(args: argparse.Namespace) -> None:
    print("variant,p50_ms,p50_fps,mean_ms,mean_fps,min_ms,max_ms")
    for name, settings in SWEEP_VARIANTS.items():
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--profile",
            args.profile,
            "--component",
            "warping",
            "--dtype",
            args.dtype,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--seed",
            str(args.seed),
            "--json",
        ]
        for key, value in settings.items():
            cmd.extend(["--set", f"{key}={value}"])
        for value in args.set:
            cmd.extend(["--set", value])
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.returncode:
            last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"
            print(f"{name},ERROR,{last}")
            continue
        lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        payload = json.loads(lines[-1])
        result = payload["results"][0]
        print(
            f"{name},"
            f"{result['p50_ms']:.2f},"
            f"{result['p50_fps']:.2f},"
            f"{result['mean_ms']:.2f},"
            f"{result['mean_fps']:.2f},"
            f"{result['min_ms']:.2f},"
            f"{result['max_ms']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=MLX_PROFILE_CHOICES, default="quality")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument(
        "--component",
        choices=("all", "motion", "appearance", "warping", "split", "full-mx"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--kp-jitter",
        type=float,
        default=0.0,
        help="Per-frame driving keypoint random-walk step scale; source keypoints stay fixed.",
    )
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--set", action="append", default=[], help="Override an FLP_MLX_* setting as KEY=VALUE")
    parser.add_argument("--sweep", action="store_true", help="Benchmark selected import-time MLX flag variants")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON payload")
    args = parser.parse_args()

    apply_mlx_profile(args.profile)
    for key, value in parse_set(args.set).items():
        os.environ[key] = value

    if args.sweep:
        run_sweep(args)
        return

    import mlx.core as mx
    from src.models import (
        MlxAppearanceFeatureExtractorModel,
        MlxMotionExtractorModel,
        MlxWarpingSpadeModel,
    )
    from src.models.mlx_modules.image_kernels import image_to_uint8

    rng = np.random.default_rng(args.seed)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    kp = (rng.normal(size=(1, 21, 3)).astype(np.float32) * 0.05)
    if args.kp_jitter > 0:
        n_steps = max(args.sequence_length - 1, 1)
        steps = rng.normal(size=(n_steps, 1, 21, 3)).astype(np.float32) * args.kp_jitter
        kp_seq = np.cumsum(np.concatenate([kp[None], steps], axis=0), axis=0).astype(np.float32)
    else:
        kp_seq = None
    dtype = dtype_from_name(args.dtype)

    results = []
    motion = None
    appearance = None
    warping_spade = None
    f_s = None

    if args.component in ("all", "motion"):
        motion = MlxMotionExtractorModel(
            dtype=args.dtype,
            model_path="./checkpoints/liveportrait_torch/motion_extractor.pth",
        )
        results.append(time_call("motion", lambda: motion.predict(img), warmup=args.warmup, iters=args.iters))

    if args.component in ("all", "appearance", "warping", "split", "full-mx"):
        appearance = MlxAppearanceFeatureExtractorModel(
            dtype=args.dtype,
            model_path="./checkpoints/liveportrait_torch/appearance_feature_extractor.pth",
        )
        if args.component in ("all", "appearance"):
            results.append(
                time_call("appearance", lambda: appearance.predict(img), warmup=args.warmup, iters=args.iters)
            )
        f_s = appearance.predict(img)

    if args.component in ("all", "warping", "split", "full-mx"):
        warping_spade = MlxWarpingSpadeModel(
            dtype=args.dtype,
            model_path=[
                "./checkpoints/liveportrait_torch/warping_module.pth",
                "./checkpoints/liveportrait_torch/spade_generator.pth",
            ],
        )
        if args.component in ("all", "warping"):
            if kp_seq is None:
                predict_warping = lambda: warping_spade.predict(f_s, kp, kp, return_numpy=True, return_uint8=True)
            else:
                kp_iter = cycle(kp_seq)

                def predict_warping():
                    return warping_spade.predict(f_s, kp, next(kp_iter), return_numpy=True, return_uint8=True)

            results.append(
                time_call(
                    "warping_spade_numpy_uint8",
                    predict_warping,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
        if args.component == "full-mx":
            if kp_seq is None:
                predict_warping_mx = lambda: warping_spade.predict(f_s, kp, kp, return_mx=True, return_uint8=True)
            else:
                kp_iter = cycle(kp_seq)

                def predict_warping_mx():
                    return warping_spade.predict(f_s, kp, next(kp_iter), return_mx=True, return_uint8=True)

            results.append(
                time_call(
                    "warping_spade_mx_uint8",
                    predict_warping_mx,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
        if args.component == "split":
            f_mx = warping_spade._feature_to_mx(f_s)
            kp_s_mx = warping_spade._source_kp_to_mx(kp)
            kp_d_mx = warping_spade._to_mx(kp)
            mx.eval(f_mx, kp_s_mx, kp_d_mx)
            results.append(
                time_call(
                    "warping_only",
                    lambda: warping_spade._fresh_warping_out(f_mx, kp_d_mx, kp_s_mx),
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
            warped_feature = warping_spade._fresh_warping_out(f_mx, kp_d_mx, kp_s_mx)
            mx.eval(warped_feature)
            results.append(
                time_call(
                    "spade_only",
                    lambda: warping_spade._spade_fn(warped_feature),
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
            results.append(
                time_call(
                    "spade_uint8",
                    lambda: image_to_uint8(warping_spade._spade_fn(warped_feature)),
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )

    payload = {
        "profile": args.profile,
        "dtype": str(dtype),
        "settings": active_settings(),
        "results": results,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print("Active MLX settings:")
        for key, value in payload["settings"].items():
            print(f"  {key}={value}")
        for result in results:
            print_result(result)


if __name__ == "__main__":
    main()
