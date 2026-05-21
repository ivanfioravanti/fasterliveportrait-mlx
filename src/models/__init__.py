# -*- coding: utf-8 -*-
"""Lazy model exports for the MLX runtime."""

from __future__ import annotations

import importlib


_MODEL_MODULES = {
    "MlxAnimalFaceAnalysisModel": ".mlx_animal_face_analysis_model",
    "MlxFaceAnalysisModel": ".mlx_face_analysis_model",
    "MlxWarpingSpadeModel": ".mlx_warping_spade_model",
    "MlxMotionExtractorModel": ".mlx_motion_extractor_model",
    "MlxAppearanceFeatureExtractorModel": ".mlx_appearance_feature_extractor_model",
    "MlxLandmarkModel": ".mlx_landmark_model",
    "MlxStitchingModel": ".mlx_stitching_model",
}

__all__ = sorted(_MODEL_MODULES)


def __getattr__(name):
    module_name = _MODEL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
