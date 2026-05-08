# -*- coding: utf-8 -*-
"""MLX implementation of warping_module + spade_generator combined inference.
Drop-in replacement for the ONNX/PyTorch WarpingSpadeModel."""

from __future__ import annotations

import numpy as np
import torch
import mlx.core as mx
import mlx.utils as mu

from .mlx_modules.warping_network import WarpingNetwork
from .mlx_modules.spade_generator import SPADEDecoder
from .mlx_modules.weight_convert import load_into_model
from .predictor import get_default_device


WARPING_PARAMS = dict(
    num_kp=21,
    block_expansion=64,
    max_features=512,
    num_down_blocks=2,
    reshape_channel=32,
    estimate_occlusion_map=True,
    dense_motion_params=dict(
        block_expansion=32,
        max_features=1024,
        num_blocks=5,
        reshape_depth=16,
        compress=4,
    ),
)

SPADE_PARAMS = dict(
    upscale=2,
    block_expansion=64,
    max_features=512,
    num_down_blocks=2,
)


_DTYPE_MAP = {
    "fp32": mx.float32, "float32": mx.float32,
    "fp16": mx.float16, "float16": mx.float16, "half": mx.float16,
    "bf16": mx.bfloat16, "bfloat16": mx.bfloat16,
}


def _resolve_dtype(kwargs):
    dt = kwargs.get("dtype")
    if dt is not None:
        return _DTYPE_MAP[dt.lower()]
    if kwargs.get("use_bf16"):
        return mx.bfloat16
    if kwargs.get("use_fp16"):
        return mx.float16
    return mx.float32


def _cast_params(model, dtype):
    if dtype == mx.float32:
        return
    flat = dict(mu.tree_flatten(model.parameters()))
    model.update(mu.tree_unflatten([(k, v.astype(dtype)) for k, v in flat.items()]))


class MlxWarpingSpadeModel:
    """warping_module + spade_generator running on MLX (Apple Silicon)."""

    def __init__(self, **kwargs):
        self.predict_type = "mlx"
        self.device = get_default_device()
        if isinstance(self.device, int):
            self.device = torch.device(f"cuda:{self.device}")

        self.dtype = _resolve_dtype(kwargs)

        warping_path, spade_path = kwargs["model_path"]

        self.warping = WarpingNetwork(**WARPING_PARAMS)
        load_into_model(self.warping, warping_path, strict=True)
        self.warping.eval()

        self.spade = SPADEDecoder(**SPADE_PARAMS)
        load_into_model(self.spade, spade_path, strict=True)
        self.spade.eval()

        _cast_params(self.warping, self.dtype)
        _cast_params(self.spade, self.dtype)

        mx.eval(self.warping.parameters())
        mx.eval(self.spade.parameters())

    def _to_mx(self, x):
        if isinstance(x, mx.array):
            return x.astype(self.dtype)
        return mx.array(np.asarray(x, dtype=np.float32)).astype(self.dtype)

    def predict(self, *data):
        # data = (f_s NCDHW numpy, kp_source numpy, kp_driving numpy)
        f_s_pt, kp_s, kp_d = data
        # f_s comes in as PyTorch NCDHW (1, 32, 16, 64, 64). Convert to NDHWC.
        f_s_arr = np.asarray(f_s_pt)
        if f_s_arr.ndim == 5 and f_s_arr.shape[1] == 32 and f_s_arr.shape[2] == 16:
            f_s_arr = np.transpose(f_s_arr, (0, 2, 3, 4, 1))  # NDHWC
        f_s_mx = self._to_mx(f_s_arr)
        kp_s_mx = self._to_mx(kp_s)
        kp_d_mx = self._to_mx(kp_d)

        wout = self.warping(f_s_mx, kp_d_mx, kp_s_mx)
        img = self.spade(wout["out"])  # (N, H, W, 3) in [0, 1]
        if img.dtype != mx.float32:
            img = img.astype(mx.float32)
        mx.eval(img)

        # MLX (N, H, W, 3) -> NumPy fp32 -> PyTorch tensor on self.device, in [0, 255]
        img_np = np.array(img, dtype=np.float32)[0]  # (H, W, 3)
        np.clip(img_np, 0.0, 1.0, out=img_np)
        img_np *= 255.0
        return torch.from_numpy(img_np).to(device=self.device)

    def __del__(self):
        for attr in ("warping", "spade"):
            if hasattr(self, attr):
                delattr(self, attr)
