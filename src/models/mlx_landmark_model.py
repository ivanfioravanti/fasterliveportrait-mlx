# -*- coding: utf-8 -*-
"""MLX landmark model wrapper for exported runtime npz weights."""

from __future__ import annotations

import cv2
import numpy as np
import mlx.core as mx

from src.utils.crop import crop_image, _transform_pts
from .mlx_modules.landmark import LandmarkModel


class MlxLandmarkModel:
    def __init__(self, **kwargs):
        self.predict_type = "mlx"
        self.dsize = 224
        from .mlx_warping_spade_model import _resolve_dtype, _cast_params
        self.dtype = _resolve_dtype(kwargs)

        self.model = LandmarkModel()
        model_path = str(kwargs["model_path"])
        if not model_path.endswith(".npz"):
            raise ValueError(
                f"MlxLandmarkModel requires exported .npz weights, got {model_path}. "
                "Run scripts/export_mlx_weights.py first."
            )
        from .mlx_modules.landmark_weight_extract import load_landmark_from_npz

        load_landmark_from_npz(self.model, model_path)
        self.model.eval()
        _cast_params(self.model, self.dtype)
        mx.eval(self.model.parameters())

    def _input(self, *data):
        if len(data) > 1:
            img_rgb, lmk = data
        else:
            img_rgb, lmk = data[0], None
        if lmk is not None:
            crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
            img_crop_rgb = crop_dct["img_crop"]
        else:
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {"M_c2o": np.array([[scale, 0., 0.],
                                            [0., scale, 0.],
                                            [0., 0., 1.]], dtype=np.float32)}
        x = (img_crop_rgb.astype(np.float32) / 255.0)[None, ...]  # (1, H, W, 3) NHWC
        return x, crop_dct

    def predict(self, *data):
        x_np, crop_dct = self._input(*data)
        x_mlx = mx.array(x_np).astype(self.dtype)
        out = self.model(x_mlx)
        pts = np.array(out["pts"].astype(mx.float32), dtype=np.float32)
        lmk = pts.reshape(-1, 2) * self.dsize
        lmk = _transform_pts(lmk, M=crop_dct["M_c2o"])
        return lmk

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
