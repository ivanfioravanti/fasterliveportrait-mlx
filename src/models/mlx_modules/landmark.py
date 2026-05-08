"""ConvNeXt-Tiny + LayerScale + 3 heads, mirroring the FasterLivePortrait
landmark.onnx architecture.

Outputs (1, 3, 224, 224) image -> dict with:
  coeff: (1, 214)   # 3D pose / expression coefficients
  lmk  : (1, 262)   # landmark coords (131 points x 2)
  pts  : (1, 406)   # additional points (203 x 2), uses concat of stages 2+3 features
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class _LSBlock(nn.Module):
    """ConvNeXt v1 block: dwconv -> LN -> pwconv1 -> GELU -> pwconv2 -> LayerScale."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.ones((dim,))

    def __call__(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return residual + x


class LandmarkModel(nn.Module):
    DEPTHS = (3, 3, 9, 3)
    DIMS = (96, 192, 384, 768)

    def __init__(self):
        super().__init__()
        in_chans = 3

        self.downsample_layers = [
            [nn.Conv2d(in_chans, self.DIMS[0], kernel_size=4, stride=4),
             nn.LayerNorm(self.DIMS[0], eps=1e-6)],
        ]
        for i in range(3):
            self.downsample_layers.append([
                nn.LayerNorm(self.DIMS[i], eps=1e-6),
                nn.Conv2d(self.DIMS[i], self.DIMS[i + 1], kernel_size=2, stride=2),
            ])

        self.stages = [
            [_LSBlock(self.DIMS[i]) for _ in range(self.DEPTHS[i])]
            for i in range(4)
        ]

        # Final norm on stage-3 features (768)
        self.norm = nn.LayerNorm(self.DIMS[-1], eps=1e-6)
        # Auxiliary norm on stage-2 features (384) used by fc_pts
        self.norm_s3 = nn.LayerNorm(self.DIMS[-2], eps=1e-6)

        # Heads
        self.fc_coeff = nn.Linear(self.DIMS[-1], 214)
        self.fc_lmk = nn.Linear(self.DIMS[-1], 262)
        self.fc_pts = nn.Linear(self.DIMS[-1] + self.DIMS[-2], 406)

    def __call__(self, x: mx.array):
        for stage_idx in range(4):
            for layer in self.downsample_layers[stage_idx]:
                x = layer(x)
            for blk in self.stages[stage_idx]:
                x = blk(x)
            if stage_idx == 2:
                # Auxiliary feature: GAP + LayerNorm on stage-2 output
                aux = mx.mean(x, axis=(1, 2))  # (N, 384)
                aux = self.norm_s3(aux)

        x = mx.mean(x, axis=(1, 2))  # (N, 768)
        x = self.norm(x)

        coeff = self.fc_coeff(x)
        lmk = self.fc_lmk(x)
        # fc_pts expects (1152) = (384) ++ (768) — aux first, then main.
        pts = self.fc_pts(mx.concatenate([aux, x], axis=-1))
        return {"coeff": coeff, "lmk": lmk, "pts": pts}
