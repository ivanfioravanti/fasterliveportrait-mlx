#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publish converted MLX runtime weights to Hugging Face Hub.

This script uploads permissively licensed MLX runtime weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi, upload_folder


DEFAULT_REPO_ID = "ivanfioravanti/FasterLivePortrait-MLX-weights"


@dataclass(frozen=True)
class WeightFile:
    local_path: str
    repo_path: str
    source: str


WEIGHT_FILES = (
    WeightFile("liveportrait_mlx/warping_module.npz", "liveportrait_mlx/warping_module.npz", "KlingTeam/LivePortrait"),
    WeightFile("liveportrait_mlx/spade_generator.npz", "liveportrait_mlx/spade_generator.npz", "KlingTeam/LivePortrait"),
    WeightFile("liveportrait_mlx/motion_extractor.npz", "liveportrait_mlx/motion_extractor.npz", "KlingTeam/LivePortrait"),
    WeightFile(
        "liveportrait_mlx/appearance_feature_extractor.npz",
        "liveportrait_mlx/appearance_feature_extractor.npz",
        "KlingTeam/LivePortrait",
    ),
    WeightFile("liveportrait_mlx/landmark.npz", "liveportrait_mlx/landmark.npz", "warmshao/FasterLivePortrait"),
    WeightFile("liveportrait_mlx/stitching.npz", "liveportrait_mlx/stitching.npz", "KlingTeam/LivePortrait"),
    WeightFile(
        "liveportrait_mlx/stitching_eye.npz",
        "liveportrait_mlx/stitching_eye.npz",
        "KlingTeam/LivePortrait",
    ),
    WeightFile(
        "liveportrait_mlx/stitching_lip.npz",
        "liveportrait_mlx/stitching_lip.npz",
        "KlingTeam/LivePortrait",
    ),
    WeightFile(
        "liveportrait_animal_mlx/base_models_v1.1/warping_module.npz",
        "liveportrait_animal_mlx/base_models_v1.1/warping_module.npz",
        "KlingTeam/LivePortrait",
    ),
    WeightFile(
        "liveportrait_animal_mlx/base_models_v1.1/spade_generator.npz",
        "liveportrait_animal_mlx/base_models_v1.1/spade_generator.npz",
        "KlingTeam/LivePortrait",
    ),
    WeightFile(
        "liveportrait_animal_mlx/base_models_v1.1/motion_extractor.npz",
        "liveportrait_animal_mlx/base_models_v1.1/motion_extractor.npz",
        "KlingTeam/LivePortrait",
    ),
    WeightFile(
        "liveportrait_animal_mlx/base_models_v1.1/appearance_feature_extractor.npz",
        "liveportrait_animal_mlx/base_models_v1.1/appearance_feature_extractor.npz",
        "KlingTeam/LivePortrait",
    ),
    WeightFile(
        "JoyVASA/motion_generator/motion_generator_hubert_chinese_mlx.npz",
        "JoyVASA/motion_generator/motion_generator_hubert_chinese_mlx.npz",
        "jdh-algo/JoyVASA",
    ),
    WeightFile(
        "JoyVASA/audio_encoder/hubert_chinese_mlx.npz",
        "JoyVASA/audio_encoder/hubert_chinese_mlx.npz",
        "jdh-algo/JoyVASA + TencentGameMate/chinese-hubert-base",
    ),
    WeightFile(
        "JoyVASA/motion_template/motion_template.pkl",
        "JoyVASA/motion_template/motion_template.pkl",
        "jdh-algo/JoyVASA",
    ),
)


MODEL_CARD = """---
license: mit
tags:
- mlx
- liveportrait
- image-to-video
- apple-silicon
base_model:
- KlingTeam/LivePortrait
- warmshao/FasterLivePortrait
- jdh-algo/JoyVASA
- TencentGameMate/chinese-hubert-base
---

# FasterLivePortrait-MLX Weights

Converted MLX `.npz` runtime weights for
[FasterLivePortrait-MLX](https://github.com/ivanfioravanti/fasterliveportrait-mlx).

These files are converted from permissively licensed source checkpoints:

- [KlingTeam/LivePortrait](https://huggingface.co/KlingTeam/LivePortrait), MIT
- [warmshao/FasterLivePortrait](https://huggingface.co/warmshao/FasterLivePortrait), MIT
- [jdh-algo/JoyVASA](https://huggingface.co/jdh-algo/JoyVASA), MIT
- [TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base), MIT

## Included

- Human LivePortrait core MLX weights
- Human landmark MLX weights
- Human stitching / eye / lip retargeting MLX weights
- Animal LivePortrait v1.1 core MLX weights
- JoyVASA MLX audio-to-motion weights for the configured Chinese HuBERT path
- JoyVASA motion template runtime asset

## Not Included

This repository also does **not** include the original JoyVASA PyTorch
checkpoint or the original Transformers HuBERT directory. Those are conversion
inputs only.

## Use

```bash
uv run python scripts/download_mlx_weights.py --repo-id {repo_id}
```

## Conversion

The weights were produced with:

```bash
uv run --group convert python scripts/export_mlx_weights.py --include-animal
```

JoyVASA weights were produced with:

```bash
uv run --group convert python scripts/export_mlx_weights.py --include-joyvasa
```

Converted tensors are derivative model weights and inherit the obligations of
the original model licenses.
"""


def validate_files(checkpoints_dir: Path) -> list[tuple[WeightFile, Path]]:
    resolved = []
    missing = []
    for weight_file in WEIGHT_FILES:
        local_path = checkpoints_dir / weight_file.local_path
        if local_path.exists():
            resolved.append((weight_file, local_path))
        else:
            missing.append(str(local_path))
    if missing:
        missing_lines = "\n  ".join(missing)
        raise FileNotFoundError(f"missing converted MLX weights:\n  {missing_lines}")
    return resolved


def stage_repo(stage_dir: Path, files: list[tuple[WeightFile, Path]], repo_id: str) -> None:
    for weight_file, local_path in files:
        dst = stage_dir / weight_file.repo_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)
    (stage_dir / "README.md").write_text(MODEL_CARD.format(repo_id=repo_id), encoding="utf-8")


def print_manifest(files: list[tuple[WeightFile, Path]]) -> None:
    total = 0
    for weight_file, local_path in files:
        size = local_path.stat().st_size
        total += size
        print(f"{weight_file.repo_path}\t{size / (1024 * 1024):.1f} MiB\tfrom {weight_file.source}")
    print(f"total\t{total / (1024 * 1024):.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish converted FasterLivePortrait-MLX weights to HF Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="target Hugging Face model repo")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="local checkpoint root")
    parser.add_argument("--private", action="store_true", help="create the HF repo as private")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the upload manifest only")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    files = validate_files(checkpoints_dir)
    print_manifest(files)
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flp_mlx_weights_") as tmpdir:
        stage_dir = Path(tmpdir)
        stage_repo(stage_dir, files, args.repo_id)
        upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(stage_dir),
            commit_message="Upload FasterLivePortrait-MLX runtime weights",
        )
    print(f"uploaded {args.repo_id}")


if __name__ == "__main__":
    main()
