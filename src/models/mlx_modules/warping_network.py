"""WarpingNetwork (W): warps the appearance feature volume with the dense
motion field predicted by DenseMotionNetwork."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .util import SameBlock2d
from .dense_motion import DenseMotionNetwork
from .grid_sample import grid_sample_3d


class WarpingNetwork(nn.Module):
    def __init__(
        self,
        num_kp: int,
        block_expansion: int,
        max_features: int,
        num_down_blocks: int,
        reshape_channel: int,
        estimate_occlusion_map: bool = True,
        dense_motion_params: dict | None = None,
        flag_use_occlusion_map: bool = True,
    ):
        super().__init__()
        self.flag_use_occlusion_map = flag_use_occlusion_map

        if dense_motion_params is None:
            self.dense_motion_network = None
        else:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp,
                feature_channel=reshape_channel,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params,
            )

        self.third = SameBlock2d(
            max_features,
            block_expansion * (2 ** num_down_blocks),
            kernel_size=3,
            padding=1,
            lrelu=True,
        )
        self.fourth = nn.Conv2d(
            in_channels=block_expansion * (2 ** num_down_blocks),
            out_channels=block_expansion * (2 ** num_down_blocks),
            kernel_size=1,
            stride=1,
        )

    def __call__(self, feature_3d, kp_driving, kp_source):
        # feature_3d: (N, D, H, W, C)
        dense_motion = self.dense_motion_network(
            feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source
        )
        occlusion_map = dense_motion.get("occlusion_map")
        deformation = dense_motion["deformation"]  # (N, D, H, W, 3)

        out = grid_sample_3d(feature_3d, deformation, align_corners=False)
        # PyTorch view collapses (C, D) into one channel axis with C outer, D inner.
        # In NDHWC, we transpose so (D, C) move adjacent at the end with C outer:
        n, d, h, w, c = out.shape
        # (N, D, H, W, C) -> (N, H, W, C, D) -> (N, H, W, C*D)
        out = mx.transpose(out, (0, 2, 3, 4, 1)).reshape(n, h, w, c * d)
        out = self.third(out)
        out = self.fourth(out)
        if self.flag_use_occlusion_map and occlusion_map is not None:
            out = out * occlusion_map

        return {
            "occlusion_map": occlusion_map,
            "deformation": deformation,
            "out": out,
        }
