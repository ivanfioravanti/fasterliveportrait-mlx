"""DenseMotionNetwork: predicts the dense motion field used by WarpingNetwork.

Layout convention in this module is NDHWC (and NKDHWC for keypoint-indexed
tensors), the natural MLX channel-last layout.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .util import Hourglass, BatchNorm3d, make_coordinate_grid, kp2gaussian
from .grid_sample import grid_sample_3d


class DenseMotionNetwork(nn.Module):
    def __init__(
        self,
        block_expansion: int,
        num_blocks: int,
        max_features: int,
        num_kp: int,
        feature_channel: int,
        reshape_depth: int,
        compress: int,
        estimate_occlusion_map: bool = True,
    ):
        super().__init__()
        in_features = (num_kp + 1) * (compress + 1)
        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=in_features,
            max_features=max_features,
            num_blocks=num_blocks,
        )
        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)
        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress)
        self.num_kp = num_kp
        self.flag_estimate_occlusion_map = estimate_occlusion_map

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(
                self.hourglass.out_filters * reshape_depth, 1, kernel_size=7, padding=3
            )
        else:
            self.occlusion = None

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        # feature: (N, D, H, W, _) -- only shape used.
        n, d, h, w, _ = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), dtype=kp_source.dtype)
        identity_grid = identity_grid.reshape(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving.reshape(n, self.num_kp, 1, 1, 1, 3)
        driving_to_source = coordinate_grid + kp_source.reshape(n, self.num_kp, 1, 1, 1, 3)
        # (N, 1, D, H, W, 3) for the identity branch
        identity_grid_b = mx.broadcast_to(identity_grid, (n, 1, d, h, w, 3))
        return mx.concatenate([identity_grid_b, driving_to_source], axis=1)

    def create_deformed_feature(self, feature, sparse_motions):
        # feature: (N, D, H, W, C). Sparse_motions: (N, K+1, D, H, W, 3)
        n, d, h, w, c = feature.shape
        kpp1 = self.num_kp + 1
        feat_rep = mx.broadcast_to(feature[:, None, ...], (n, kpp1, d, h, w, c))
        feat_rep = feat_rep.reshape(n * kpp1, d, h, w, c)
        sm = sparse_motions.reshape(n * kpp1, d, h, w, 3)
        out = grid_sample_3d(feat_rep, sm, align_corners=False)
        return out.reshape(n, kpp1, d, h, w, c)

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        # feature: (N, K+1, D, H, W, C). spatial_size = (D, H, W)
        spatial_size = feature.shape[2:5]
        g_d = kp2gaussian(kp_driving, spatial_size, kp_variance=0.01)  # (N, K, D, H, W)
        g_s = kp2gaussian(kp_source, spatial_size, kp_variance=0.01)
        heatmap = g_d - g_s
        n = heatmap.shape[0]
        zeros = mx.zeros((n, 1, *spatial_size), dtype=heatmap.dtype)
        heatmap = mx.concatenate([zeros, heatmap], axis=1)  # (N, K+1, D, H, W)
        return heatmap[..., None]  # (N, K+1, D, H, W, 1)

    def __call__(self, feature, kp_driving, kp_source):
        # feature: (N, D, H, W, feature_channel)
        n, d, h, w, _ = feature.shape

        # Compress + BN3d + ReLU
        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = nn.relu(feature)

        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        x = mx.concatenate([heatmap, deformed_feature], axis=-1)  # (N, K+1, D, H, W, C+1)
        # Flatten the (K+1) axis into C: PyTorch did view(N, (K+1)*C, D, H, W). In NDHWC
        # we move (K+1, C) into a single C axis at the end:
        # (N, K+1, D, H, W, C+1) -> (N, D, H, W, K+1, C+1) -> (N, D, H, W, (K+1)*(C+1))
        x = mx.transpose(x, (0, 2, 3, 4, 1, 5))
        kpp1 = x.shape[-2]
        cc = x.shape[-1]
        x = x.reshape(n, d, h, w, kpp1 * cc)

        prediction = self.hourglass(x)  # (N, D, H, W, hourglass_out)
        mask_logits = self.mask(prediction)  # (N, D, H, W, K+1)
        mask = mx.softmax(mask_logits, axis=-1)
        # Take mask in shape (N, K+1, D, H, W, 1) for broadcasting against sparse_motion
        mask_perm = mx.transpose(mask, (0, 4, 1, 2, 3))[..., None]
        # sparse_motion: (N, K+1, D, H, W, 3)
        deformation = mx.sum(sparse_motion * mask_perm, axis=1)  # (N, D, H, W, 3)

        out = {"mask": mask, "deformation": deformation}

        if self.flag_estimate_occlusion_map:
            # PyTorch did `prediction.view(B, F*D, H, W)` on a (B, F, D, H, W)
            # tensor, flattening (F, D) with F outer / D inner. In NDHWC we
            # transpose to (N, H, W, F, D) before the channel reshape so the
            # last axis carries the same f*D + d ordering that the occlusion
            # conv expects.
            n_, d_, h_, w_, f_ = prediction.shape
            pred2d = mx.transpose(prediction, (0, 2, 3, 4, 1)).reshape(n_, h_, w_, f_ * d_)
            occ = mx.sigmoid(self.occlusion(pred2d))  # (N, H, W, 1)
            out["occlusion_map"] = occ

        return out
