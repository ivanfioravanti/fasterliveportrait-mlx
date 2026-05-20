#!/usr/bin/env python3
"""Run and track MLX performance experiments.

This script intentionally launches each variant in a fresh Python process
because many FLP_MLX_* switches are read at module import time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    name: str
    profile: str = "quality"
    dtype: str = "bf16"
    settings: tuple[tuple[str, str], ...] = ()
    exact_candidate: bool = True


EXPERIMENTS = (
    Experiment("quality_baseline"),
    Experiment("warp_out_standard", settings=(("FLP_MLX_WARP_OUT_BACKEND", "standard"),)),
    Experiment("fused_deformation", settings=(("FLP_MLX_FUSED_DEFORMATION", "1"),)),
    Experiment("fused_sparse_sample", settings=(("FLP_MLX_FUSED_SPARSE_SAMPLE", "1"),)),
    Experiment("fused_hourglass_input", settings=(("FLP_MLX_FUSED_HOURGLASS_INPUT", "1"),)),
    Experiment("compile_warping", settings=(("FLP_MLX_COMPILE_WARPING", "1"),)),
    Experiment("compile_hourglass_off", settings=(("FLP_MLX_COMPILE_HOURGLASS", "0"),)),
    Experiment("compile_spade_off", settings=(("FLP_MLX_COMPILE_SPADE", "0"),)),
    Experiment("mask_backend_2d", settings=(("FLP_MLX_MASK_BACKEND", "2d"),)),
    Experiment("mask_backend_packed2d_stack", settings=(("FLP_MLX_MASK_BACKEND", "packed2d_stack"),)),
    Experiment("mask_backend_native", settings=(("FLP_MLX_MASK_BACKEND", "native"),)),
    Experiment("mask_backend_tensorops", settings=(("FLP_MLX_MASK_BACKEND", "tensorops"),)),
    Experiment("compress_tensorops", settings=(("FLP_MLX_COMPRESS_BACKEND", "tensorops"),)),
    Experiment("fourth_tensorops", settings=(("FLP_MLX_WARP_FOURTH_BACKEND", "tensorops"),)),
    Experiment("spade_shortcut_tensorops", settings=(("FLP_MLX_SPADE_SHORTCUT_BACKEND", "tensorops"),)),
    Experiment("spade_fp32_norm", settings=(("FLP_MLX_SPADE_BF16_NATIVE_NORM", "0"),)),
    Experiment("dtype_fp16", dtype="fp16"),
    Experiment("speed_profile", profile="speed", exact_candidate=False),
    Experiment("turbo_profile", profile="turbo", exact_candidate=False),
    Experiment("ultra_profile", profile="ultra", exact_candidate=False),
    Experiment(
        "turbo_interval4",
        profile="turbo",
        settings=(("FLP_MLX_TEMPORAL_WARP_INTERVAL", "4"),),
        exact_candidate=False,
    ),
    Experiment(
        "turbo_skip_occlusion",
        profile="turbo",
        settings=(("FLP_MLX_SKIP_OCCLUSION", "1"),),
        exact_candidate=False,
    ),
)


OUTPUT_CODE = r"""
import os
import sys
import numpy as np

from src.utils.mlx_profiles import apply_mlx_profile

profile = sys.argv[1]
dtype = sys.argv[2]
input_path = sys.argv[3]
output_path = sys.argv[4]
settings = json.loads(sys.argv[5])

apply_mlx_profile(profile)
for key, value in settings.items():
    os.environ[key] = value

from src.models import MlxAppearanceFeatureExtractorModel, MlxWarpingSpadeModel

inp = np.load(input_path)
img = inp["img"]
kp = inp["kp"]
app = MlxAppearanceFeatureExtractorModel(
    dtype=dtype,
    model_path="./checkpoints/liveportrait_torch/appearance_feature_extractor.pth",
)
warp = MlxWarpingSpadeModel(
    dtype=dtype,
    model_path=[
        "./checkpoints/liveportrait_torch/warping_module.pth",
        "./checkpoints/liveportrait_torch/spade_generator.pth",
    ],
)
f_s = app.predict(img)
out = warp.predict(f_s, kp, kp, return_numpy=True, return_uint8=False)
np.save(output_path, out.astype(np.float32, copy=False))
"""


def settings_dict(exp: Experiment) -> dict[str, str]:
    return dict(exp.settings)


def run_timing(exp: Experiment, warmup: int, iters: int, seed: int, kp_jitter: float, sequence_length: int) -> dict:
    cmd = [
        sys.executable,
        "scripts/bench_mlx_pipeline.py",
        "--profile",
        exp.profile,
        "--dtype",
        exp.dtype,
        "--component",
        "warping",
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--seed",
        str(seed),
        "--kp-jitter",
        str(kp_jitter),
        "--sequence-length",
        str(sequence_length),
        "--json",
    ]
    for key, value in exp.settings:
        cmd.extend(["--set", f"{key}={value}"])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])["results"][0]


def run_output(exp: Experiment, input_path: Path, output_path: Path) -> np.ndarray:
    code = "import json\n" + OUTPUT_CODE
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            exp.profile,
            exp.dtype,
            str(input_path),
            str(output_path),
            json.dumps(settings_dict(exp)),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return np.load(output_path)


def print_table(rows: list[dict], *, markdown: bool) -> None:
    headers = [
        "experiment",
        "class",
        "p50_ms",
        "p50_fps",
        "mean_ms",
        "mean_fps",
        "max_diff",
        "mean_diff",
        "viable",
    ]
    if markdown:
        print("| " + " | ".join(headers) + " |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            print(
                "| "
                + " | ".join(str(row[h]) for h in headers)
                + " |"
            )
    else:
        print(",".join(headers))
        for row in rows:
            print(",".join(str(row[h]) for h in headers))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--kp-jitter", type=float, default=0.0)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    temp_dir = Path(tempfile.mkdtemp(prefix="flp_mlx_experiments_"))
    input_path = temp_dir / "inputs.npz"
    np.savez(
        input_path,
        img=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        kp=(rng.normal(size=(1, 21, 3)).astype(np.float32) * 0.05),
    )

    baseline_exp = EXPERIMENTS[0]
    baseline_output = run_output(baseline_exp, input_path, temp_dir / "baseline.npy")

    rows = []
    for exp in EXPERIMENTS:
        try:
            timing = run_timing(exp, args.warmup, args.iters, args.seed, args.kp_jitter, args.sequence_length)
            if exp.exact_candidate:
                out = baseline_output if exp is baseline_exp else run_output(exp, input_path, temp_dir / f"{exp.name}.npy")
                diff = np.abs(baseline_output - out)
                max_diff = float(diff.max())
                mean_diff = float(diff.mean())
                viable = (max_diff <= 1e-4 and timing["p50_ms"] < rows[0]["_p50"]) if rows else True
                cls = "exact"
            else:
                max_diff = ""
                mean_diff = ""
                viable = timing["p50_ms"] < rows[0]["_p50"]
                cls = "approx"
            rows.append(
                {
                    "experiment": exp.name,
                    "class": cls,
                    "p50_ms": f"{timing['p50_ms']:.2f}",
                    "p50_fps": f"{timing['p50_fps']:.2f}",
                    "mean_ms": f"{timing['mean_ms']:.2f}",
                    "mean_fps": f"{timing['mean_fps']:.2f}",
                    "max_diff": "" if max_diff == "" else f"{max_diff:.6g}",
                    "mean_diff": "" if mean_diff == "" else f"{mean_diff:.6g}",
                    "viable": "yes" if viable else "no",
                    "_p50": timing["p50_ms"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "experiment": exp.name,
                    "class": "error",
                    "p50_ms": "",
                    "p50_fps": "",
                    "mean_ms": "",
                    "mean_fps": "",
                    "max_diff": "",
                    "mean_diff": "",
                    "viable": f"error: {exc}",
                    "_p50": float("inf"),
                }
            )

    printable = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    print_table(printable, markdown=args.markdown)


if __name__ == "__main__":
    main()
