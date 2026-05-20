#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download runtime assets for the MLX port.

The default download path uses a permissive MLX weights repo containing only
converted LivePortrait runtime weights. XPose remains optional because its
license is non-commercial research only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import urllib.request

from huggingface_hub import snapshot_download


DEFAULT_MLX_REPO = "ivanfioravanti/FasterLivePortrait-MLX-weights"
MEDIAPIPE_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

MLX_ALLOW_PATTERNS = (
    "liveportrait_mlx/*.npz",
    "liveportrait_animal_mlx/base_models_v1.1/*.npz",
)

XPOSE_ALLOW_PATTERNS = (
    "liveportrait_animals/xpose.pth",
)

ANIMAL_EMBEDDING_PATTERNS = (
    "liveportrait_animal_onnx/clip_embedding_9.pkl",
    "liveportrait_animal_onnx/clip_embedding_68.pkl",
)


def download_mlx_weights(repo_id: str, checkpoints_dir: Path, revision: str | None) -> None:
    print(f"downloading MLX weights from {repo_id} -> {checkpoints_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=checkpoints_dir,
        allow_patterns=list(MLX_ALLOW_PATTERNS),
    )


def download_mediapipe(checkpoints_dir: Path) -> None:
    out_path = checkpoints_dir / "mediapipe" / "face_landmarker.task"
    if out_path.exists():
        print(f"already exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading MediaPipe face landmarker -> {out_path}")
    urllib.request.urlretrieve(MEDIAPIPE_FACE_LANDMARKER_URL, out_path)


def download_xpose(checkpoints_dir: Path) -> None:
    print(
        "downloading XPose assets. Note: XPose is licensed for non-commercial "
        "research use only; it is intentionally not included in the MLX weights repo."
    )
    snapshot_download(
        repo_id="KlingTeam/LivePortrait",
        repo_type="model",
        local_dir=checkpoints_dir,
        allow_patterns=list(XPOSE_ALLOW_PATTERNS),
    )
    snapshot_download(
        repo_id="warmshao/FasterLivePortrait",
        repo_type="model",
        local_dir=checkpoints_dir,
        allow_patterns=list(ANIMAL_EMBEDDING_PATTERNS),
    )

    embedding_dir = checkpoints_dir / "liveportrait_animals" / "clip_embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        ("clip_embedding_9.pkl", "clip_embedding_9.pkl"),
        ("clip_embedding_68.pkl", "clip_embedding_68.pkl"),
    ):
        src = checkpoints_dir / "liveportrait_animal_onnx" / src_name
        dst = embedding_dir / dst_name
        if not src.exists():
            raise FileNotFoundError(f"missing downloaded XPose embedding: {src}")
        shutil.copy2(src, dst)
        print(f"wrote {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FasterLivePortrait-MLX runtime assets.")
    parser.add_argument("--repo-id", default=DEFAULT_MLX_REPO, help="HF repo containing converted MLX .npz weights")
    parser.add_argument("--revision", default=None, help="optional HF revision for the MLX weights repo")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="local checkpoint root")
    parser.add_argument(
        "--skip-mediapipe",
        action="store_true",
        help="do not download Google's MediaPipe face landmarker task model",
    )
    parser.add_argument(
        "--include-animal-xpose",
        action="store_true",
        help="also download XPose assets for animal mode; XPose is non-commercial research only",
    )
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    try:
        download_mlx_weights(args.repo_id, checkpoints_dir, args.revision)
        if not args.skip_mediapipe:
            download_mediapipe(checkpoints_dir)
        if args.include_animal_xpose:
            download_xpose(checkpoints_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
