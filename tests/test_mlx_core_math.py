import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest


def _np_grid_sample_3d(input_np, grid_np, *, align_corners=False):
    n, d, h, w, c = input_np.shape
    _, do, ho, wo, _ = grid_np.shape
    out = np.zeros((n, do, ho, wo, c), dtype=np.float32)

    for ni in range(n):
        for zi_out in range(do):
            for yi_out in range(ho):
                for xi_out in range(wo):
                    gx, gy, gz = grid_np[ni, zi_out, yi_out, xi_out]
                    if align_corners:
                        x = (gx + 1.0) * (w - 1) * 0.5
                        y = (gy + 1.0) * (h - 1) * 0.5
                        z = (gz + 1.0) * (d - 1) * 0.5
                    else:
                        x = ((gx + 1.0) * w - 1.0) * 0.5
                        y = ((gy + 1.0) * h - 1.0) * 0.5
                        z = ((gz + 1.0) * d - 1.0) * 0.5

                    x0 = int(np.floor(x))
                    y0 = int(np.floor(y))
                    z0 = int(np.floor(z))
                    wx = x - x0
                    wy = y - y0
                    wz = z - z0

                    for dz, wz_weight in ((0, 1.0 - wz), (1, wz)):
                        zi = z0 + dz
                        if zi < 0 or zi >= d:
                            continue
                        for dy, wy_weight in ((0, 1.0 - wy), (1, wy)):
                            yi = y0 + dy
                            if yi < 0 or yi >= h:
                                continue
                            for dx, wx_weight in ((0, 1.0 - wx), (1, wx)):
                                xi = x0 + dx
                                if xi < 0 or xi >= w:
                                    continue
                                out[ni, zi_out, yi_out, xi_out] += (
                                    input_np[ni, zi, yi, xi] * wz_weight * wy_weight * wx_weight
                                )
    return out


def _sample_inputs(channels=4):
    input_np = np.linspace(-0.7, 0.9, 1 * 3 * 4 * 5 * channels, dtype=np.float32).reshape(
        1, 3, 4, 5, channels
    )
    grid_np = np.array(
        [
            [
                [
                    [[-1.2, -1.0, -0.7], [-0.45, 0.1, 0.25], [1.1, 0.8, -0.2]],
                    [[-0.8, 0.7, 1.0], [0.2, -0.35, 0.45], [0.95, 1.1, 1.2]],
                ],
                [
                    [[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                    [[-0.25, 0.5, -0.5], [0.4, -0.7, 0.8], [1.25, -1.1, 0.3]],
                ],
            ]
        ],
        dtype=np.float32,
    )
    return input_np, grid_np


@pytest.mark.parametrize("align_corners", (False, True))
def test_grid_sample_gather_matches_numpy_reference(align_corners):
    from src.models.mlx_modules.grid_sample import _grid_sample_3d_gather

    input_np, grid_np = _sample_inputs()
    expected = _np_grid_sample_3d(input_np, grid_np, align_corners=align_corners)

    actual = _grid_sample_3d_gather(mx.array(input_np), mx.array(grid_np), align_corners=align_corners)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal grid kernels require MLX Metal")
def test_grid_sample_metal_and_to_2d_variants_match_reference_layout():
    from src.models.mlx_modules.grid_sample import (
        grid_sample_3d,
        grid_sample_3d_sparse_motions,
        grid_sample_3d_to_2d_channels,
        grid_sample_3d_to_2d_channels_c4,
    )

    input_np, grid_np = _sample_inputs(channels=4)
    expected = _np_grid_sample_3d(input_np, grid_np, align_corners=False)
    expected_2d = expected.transpose(0, 2, 3, 4, 1).reshape(
        expected.shape[0],
        expected.shape[2],
        expected.shape[3],
        expected.shape[4] * expected.shape[1],
    )

    input_mx = mx.array(input_np)
    grid_mx = mx.array(grid_np)
    np.testing.assert_allclose(np.array(grid_sample_3d(input_mx, grid_mx)), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.array(grid_sample_3d_to_2d_channels(input_mx, grid_mx)),
        expected_2d,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(grid_sample_3d_to_2d_channels_c4(input_mx, grid_mx)),
        expected_2d,
        rtol=1e-6,
        atol=1e-6,
    )

    sparse_grid_np = np.stack([grid_np[0], grid_np[0] * 0.5], axis=0)[None]
    sparse_expected = np.stack(
        [
            _np_grid_sample_3d(input_np, sparse_grid_np[:, k], align_corners=False)
            for k in range(sparse_grid_np.shape[1])
        ],
        axis=1,
    )
    sparse_actual = grid_sample_3d_sparse_motions(input_mx, mx.array(sparse_grid_np))
    np.testing.assert_allclose(np.array(sparse_actual), sparse_expected, rtol=1e-6, atol=1e-6)


def test_pixel_shuffle_nhwc_matches_pytorch_channel_order():
    from src.models.mlx_modules.spade_generator import _pixel_shuffle_nhwc

    x = np.arange(1 * 2 * 3 * 8, dtype=np.float32).reshape(1, 2, 3, 8)
    expected = np.empty((1, 4, 6, 2), dtype=np.float32)
    for n in range(x.shape[0]):
        for h in range(x.shape[1]):
            for w in range(x.shape[2]):
                for oc in range(2):
                    for rh in range(2):
                        for rw in range(2):
                            expected[n, h * 2 + rh, w * 2 + rw, oc] = x[n, h, w, oc * 4 + rh * 2 + rw]

    actual = _pixel_shuffle_nhwc(mx.array(x), 2)
    np.testing.assert_array_equal(np.array(actual), expected)


def test_conv3d_via_2d_matches_native_mlx_conv3d():
    from src.models.mlx_modules.util import conv3d_via_2d

    rng = np.random.default_rng(7)
    x_np = rng.normal(size=(1, 4, 5, 4, 3)).astype(np.float32)
    weight_np = rng.normal(size=(2, 3, 3, 3, 3)).astype(np.float32) * 0.1
    bias_np = rng.normal(size=(2,)).astype(np.float32) * 0.1

    x = mx.array(x_np)
    weight = mx.array(weight_np)
    bias = mx.array(bias_np)
    expected = mx.conv3d(x, weight, padding=1) + bias
    actual = conv3d_via_2d(x, weight, bias, padding=1)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)


def test_human_runtime_imports_do_not_import_torch():
    code = (
        "import sys;"
        "import src.utils.utils;"
        "from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline;"
        "from src.models.mlx_warping_spade_model import MlxWarpingSpadeModel;"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_landmark_state_validation_rejects_shape_mismatch():
    from src.models.mlx_modules.landmark_weight_extract import _validate_landmark_state

    class TinyModel:
        def parameters(self):
            return {"a": mx.zeros((2, 3)), "b": mx.zeros((1,))}

    good_state = {"a": mx.ones((2, 3)), "b": mx.ones((1,))}
    _validate_landmark_state(TinyModel(), good_state)

    bad_state = {"a": mx.ones((3, 2)), "b": mx.ones((1,))}
    with pytest.raises(ValueError, match="bad shapes"):
        _validate_landmark_state(TinyModel(), bad_state)
