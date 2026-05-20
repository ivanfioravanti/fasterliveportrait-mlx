"""Named MLX runtime profiles for command-line use.

The lower-level MLX tuning flags are intentionally still environment variables
because several modules read them at import time. This module gives run.py a
small, stable CLI surface and applies the chosen values before those modules
are imported.
"""

from __future__ import annotations

import os


_BASE_PROFILE = {
    "FLP_MLX_2D_MASK": "1",
    "FLP_MLX_MASK_BACKEND": "packed2d",
    "FLP_MLX_COMPRESS_BACKEND": "native",
    "FLP_MLX_COMPILE_HOURGLASS": "1",
    "FLP_MLX_COMPILE_SPADE": "1",
    "FLP_MLX_COMPILE_WARPING": "0",
    "FLP_MLX_COMPILE_MOTION": "1",
    "FLP_MLX_COMPILE_APPEARANCE": "1",
    "FLP_MLX_CACHE_SOURCE_GAUSSIAN": "1",
    "FLP_MLX_FUSED_UINT8": "1",
    "FLP_MLX_FUSED_DEFORMATION": "0",
    "FLP_MLX_FUSED_SPARSE_SAMPLE": "0",
    "FLP_MLX_FUSED_HOURGLASS_INPUT": "0",
    "FLP_MLX_SKIP_OCCLUSION": "0",
    "FLP_MLX_CONV3D_BACKEND": "native",
    "FLP_MLX_GS3D_GATHER": "0",
    "FLP_MLX_WARP_OUT_BACKEND": "direct",
    "FLP_MLX_WARP_FOURTH_BACKEND": "native",
    "FLP_MLX_SPADE_BF16_NATIVE_NORM": "1",
    "FLP_MLX_SPADE_SHORTCUT_BACKEND": "native",
    "FLP_MLX_SPADE_SHORTCUT_MIN_OUT": "128",
    "FLP_MLX_TEMPORAL_WARP_INTERVAL": "1",
    "FLP_MLX_TEMPORAL_WARP_THRESHOLD": "0",
}


MLX_PROFILES = {
    "quality": {
        "description": "Highest-fidelity MLX path; no temporal warp reuse.",
        "settings": dict(_BASE_PROFILE),
    },
    "speed": {
        "description": "Faster realtime MLX path; reuses warping every other frame with adaptive refresh.",
        "settings": {
            **_BASE_PROFILE,
            "FLP_MLX_CONV3D_BACKEND": "auto",
            "FLP_MLX_FUSED_DEFORMATION": "1",
            "FLP_MLX_TEMPORAL_WARP_INTERVAL": "2",
            "FLP_MLX_TEMPORAL_WARP_THRESHOLD": "0.0015",
        },
    },
    "turbo": {
        "description": "Aggressive realtime MLX path; reuses warping for up to three frames.",
        "settings": {
            **_BASE_PROFILE,
            "FLP_MLX_CONV3D_BACKEND": "auto",
            "FLP_MLX_FUSED_DEFORMATION": "1",
            "FLP_MLX_TEMPORAL_WARP_INTERVAL": "3",
            "FLP_MLX_TEMPORAL_WARP_THRESHOLD": "0.0015",
        },
    },
    "ultra": {
        "description": "Highest-throughput realtime MLX path; reuses warping for up to eight frames with a larger refresh threshold.",
        "settings": {
            **_BASE_PROFILE,
            "FLP_MLX_CONV3D_BACKEND": "auto",
            "FLP_MLX_FUSED_DEFORMATION": "1",
            "FLP_MLX_TEMPORAL_WARP_INTERVAL": "8",
            "FLP_MLX_TEMPORAL_WARP_THRESHOLD": "0.02",
        },
    },
    "custom": {
        "description": "Do not set MLX tuning values; use explicit environment/config values.",
        "settings": {},
    },
}


MLX_PROFILE_CHOICES = tuple(MLX_PROFILES)


def apply_mlx_profile(profile: str, *, overwrite: bool = True) -> dict[str, str]:
    if profile not in MLX_PROFILES:
        choices = ", ".join(MLX_PROFILE_CHOICES)
        raise ValueError(f"unknown MLX profile {profile!r}; expected one of: {choices}")

    settings = MLX_PROFILES[profile]["settings"]
    for key, value in settings.items():
        if overwrite or key not in os.environ:
            os.environ[key] = value
    return dict(settings)


def describe_mlx_profile(profile: str) -> str:
    return MLX_PROFILES[profile]["description"]
