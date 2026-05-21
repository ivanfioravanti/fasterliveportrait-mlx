"""Runtime asset resolution for Hugging Face cached model snapshots."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .utils.tqdm_utils import configure_tqdm_single_process

configure_tqdm_single_process()


DEFAULT_MLX_WEIGHTS_REPO = "ivanfioravanti/FasterLivePortrait-MLX-weights"

MLX_ALLOW_PATTERNS = (
    "liveportrait_mlx/*.npz",
    "liveportrait_animal_mlx/base_models_v1.1/*.npz",
    "liveportrait_animal_mlx/retargeting_models/*.npz",
    "JoyVASA/motion_generator/motion_generator_hubert_chinese_mlx.npz",
    "JoyVASA/audio_encoder/hubert_chinese_mlx.npz",
    "JoyVASA/motion_template/motion_template.pkl",
)


def get_local_checkpoints_dir() -> Path | None:
    """Return an explicit local checkpoint override, if the user requested one."""
    explicit = os.environ.get("FLIP_CHECKPOINT_DIR")
    if not explicit:
        return None
    return Path(explicit).expanduser()


def runtime_repo_id() -> str:
    return os.environ.get("FLIP_MLX_WEIGHTS_REPO", DEFAULT_MLX_WEIGHTS_REPO)


def runtime_revision() -> str | None:
    return os.environ.get("FLIP_MLX_WEIGHTS_REVISION")


def _log(log, message: str) -> None:
    if log is not None:
        log.info(message)
    else:
        print(message)


def download_mlx_snapshot(*, log=None) -> Path:
    repo_id = runtime_repo_id()
    revision = runtime_revision()
    _log(log, f"resolve MLX runtime weights from Hugging Face Hub: {repo_id}")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=list(MLX_ALLOW_PATTERNS),
    )
    return Path(snapshot_path)


def download_mlx_to_local(checkpoints_dir: Path, *, log=None) -> None:
    _log(log, f"download MLX runtime weights -> {checkpoints_dir}")
    snapshot_download(
        repo_id=runtime_repo_id(),
        repo_type="model",
        revision=runtime_revision(),
        local_dir=checkpoints_dir,
        allow_patterns=list(MLX_ALLOW_PATTERNS),
    )


def resolve_checkpoint_path(path: str, checkpoint_root: Path | None) -> str:
    if not path:
        return path

    expanded = os.path.expanduser(os.fspath(path))
    if os.path.isabs(expanded):
        return expanded

    normalized = expanded.replace("\\", "/")
    for prefix in ("./checkpoints", "checkpoints"):
        if normalized == prefix:
            return str(checkpoint_root) if checkpoint_root is not None else expanded
        prefix_with_slash = f"{prefix}/"
        if normalized.startswith(prefix_with_slash):
            rel_path = normalized[len(prefix_with_slash):]
            return str(checkpoint_root / rel_path) if checkpoint_root is not None else expanded

    return expanded


def resolve_runtime_config(
    infer_cfg,
    *,
    checkpoint_root: Path | None,
):
    """Mutate a loaded OmegaConf config to point at resolved runtime asset files."""
    for group_name in ("models", "animal_models"):
        group = infer_cfg.get(group_name, None)
        if group is None:
            continue
        for model_name in group:
            model_path = group[model_name].get("model_path", None)
            if isinstance(model_path, str):
                group[model_name].model_path = resolve_checkpoint_path(model_path, checkpoint_root)
            elif model_path is not None:
                for index, item in enumerate(model_path):
                    model_path[index] = resolve_checkpoint_path(item, checkpoint_root)

    joyvasa_models = infer_cfg.get("joyvasa_models", None)
    if joyvasa_models is not None:
        for key in ("motion_mlx_model_path", "audio_mlx_model_path", "motion_template_path"):
            if joyvasa_models.get(key, None):
                joyvasa_models[key] = resolve_checkpoint_path(joyvasa_models[key], checkpoint_root)

    return infer_cfg


def required_runtime_paths(infer_cfg) -> list[str]:
    required: list[str] = []

    for group_name in ("models", "animal_models"):
        group = infer_cfg.get(group_name, None)
        if group is None:
            continue
        for model_name in group:
            model_path = group[model_name].get("model_path", None)
            if isinstance(model_path, str):
                required.append(model_path)
            elif model_path is not None:
                required.extend(str(path) for path in model_path)

    joyvasa_models = infer_cfg.get("joyvasa_models", None)
    if joyvasa_models is not None:
        for key in ("motion_mlx_model_path", "audio_mlx_model_path", "motion_template_path"):
            if joyvasa_models.get(key, None):
                required.append(joyvasa_models[key])

    return required


def missing_runtime_paths(infer_cfg) -> list[str]:
    return [path for path in required_runtime_paths(infer_cfg) if not os.path.exists(path)]


def ensure_runtime_assets(infer_cfg, *, log=None):
    local_checkpoints_dir = get_local_checkpoints_dir()
    if local_checkpoints_dir is not None:
        resolve_runtime_config(infer_cfg, checkpoint_root=local_checkpoints_dir)
        missing = missing_runtime_paths(infer_cfg)
        if missing:
            _log(log, f"missing runtime assets under FLIP_CHECKPOINT_DIR={local_checkpoints_dir}")
            download_mlx_to_local(local_checkpoints_dir, log=log)

        missing = missing_runtime_paths(infer_cfg)
        if missing:
            raise RuntimeError(
                "Downloaded runtime assets are incomplete under FLIP_CHECKPOINT_DIR="
                f"{local_checkpoints_dir}. Missing: {', '.join(missing)}"
            )
        return infer_cfg

    mlx_snapshot = download_mlx_snapshot(log=log)
    resolve_runtime_config(infer_cfg, checkpoint_root=mlx_snapshot)

    missing = missing_runtime_paths(infer_cfg)
    if missing:
        raise RuntimeError(
            "Downloaded runtime assets are incomplete. Missing: "
            + ", ".join(missing)
        )

    return infer_cfg
