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

JOYVASA_REPO = "jdh-algo/JoyVASA"
JOYVASA_ALLOW_PATTERNS = (
    "motion_generator/motion_generator_hubert_chinese.pt",
    "motion_template/motion_template.pkl",
)

CHINESE_HUBERT_REPO = "TencentGameMate/chinese-hubert-base"
CHINESE_HUBERT_ALLOW_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
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


def download_joyvasa(checkpoints_dir: Path) -> None:
    print(
        "downloading JoyVASA assets. Note: JoyVASA audio/text driving is "
        "experimental in this MLX release."
    )
    snapshot_download(
        repo_id=JOYVASA_REPO,
        repo_type="model",
        local_dir=checkpoints_dir / "JoyVASA",
        allow_patterns=list(JOYVASA_ALLOW_PATTERNS),
    )
    snapshot_download(
        repo_id=CHINESE_HUBERT_REPO,
        repo_type="model",
        local_dir=checkpoints_dir / "chinese-hubert-base",
        allow_patterns=list(CHINESE_HUBERT_ALLOW_PATTERNS),
    )
    from src.models.mlx_joyvasa_audio_model import export_mlx_joyvasa_audio_from_pytorch_checkpoint
    from src.models.mlx_joyvasa_motion_model import export_mlx_joyvasa_motion_from_pytorch_checkpoint

    src = checkpoints_dir / "JoyVASA" / "motion_generator" / "motion_generator_hubert_chinese.pt"
    motion_dst = checkpoints_dir / "JoyVASA" / "motion_generator" / "motion_generator_hubert_chinese_mlx.npz"
    audio_dst = checkpoints_dir / "JoyVASA" / "audio_encoder" / "hubert_chinese_mlx.npz"
    audio_src = checkpoints_dir / "chinese-hubert-base"
    if not motion_dst.exists():
        print(f"exporting JoyVASA MLX motion weights -> {motion_dst}")
        export_mlx_joyvasa_motion_from_pytorch_checkpoint(src, motion_dst)
    else:
        print(f"already exists: {motion_dst}")
    if not audio_dst.exists():
        print(f"exporting JoyVASA MLX HuBERT audio weights -> {audio_dst}")
        export_mlx_joyvasa_audio_from_pytorch_checkpoint(src, audio_src, audio_dst)
    else:
        print(f"already exists: {audio_dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FasterLivePortrait-MLX runtime assets.")
    parser.add_argument("--repo-id", default=DEFAULT_MLX_REPO, help="HF repo containing converted MLX .npz weights")
    parser.add_argument("--revision", default=None, help="optional HF revision for the MLX weights repo")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="local checkpoint root")
    parser.add_argument(
        "--skip-mlx-weights",
        action="store_true",
        help="do not download LivePortrait MLX .npz runtime weights",
    )
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
    parser.add_argument(
        "--include-joyvasa",
        action="store_true",
        help="also download experimental JoyVASA audio-to-motion assets for audio/text driving",
    )
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    try:
        if not args.skip_mlx_weights:
            download_mlx_weights(args.repo_id, checkpoints_dir, args.revision)
        if not args.skip_mediapipe:
            download_mediapipe(checkpoints_dir)
        if args.include_animal_xpose:
            download_xpose(checkpoints_dir)
        if args.include_joyvasa:
            download_joyvasa(checkpoints_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
