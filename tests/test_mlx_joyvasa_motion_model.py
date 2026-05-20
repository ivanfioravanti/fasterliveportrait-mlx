import sys
from argparse import Namespace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_f
from transformers import HubertConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_torch_motion_model(
    *,
    target="sample",
    use_indicator=False,
    n_diff_steps=4,
    guiding_conditions="audio,",
    cfg_mode="incremental",
):
    from src.models.JoyVASA.dit_talking_head import DenoisingNetwork, DiffusionSchedule, DitTalkingHead

    torch_model = DitTalkingHead.__new__(DitTalkingHead)
    torch_nn.Module.__init__(torch_model)
    torch_model.target = target
    torch_model.architecture = "decoder"
    torch_model.motion_feat_dim = 4
    torch_model.fps = 25
    torch_model.n_motions = 3
    torch_model.n_prev_motions = 2
    torch_model.feature_dim = 8
    torch_model.cfg_mode = cfg_mode
    guiding_conditions = guiding_conditions.split(",") if guiding_conditions else []
    torch_model.guiding_conditions = [cond for cond in guiding_conditions if cond in ["audio"]]
    torch_model.start_motion_feat = torch_nn.Parameter(torch.randn(1, 2, 4))
    torch_model.start_audio_feat = torch_nn.Parameter(torch.randn(1, 2, 8))
    if "audio" in torch_model.guiding_conditions:
        torch_model.null_audio_feat = torch_nn.Parameter(torch.randn(1, 1, 8))
    torch_model.denoising_net = DenoisingNetwork(
        device="cpu",
        motion_feat_dim=4,
        use_indicator=use_indicator,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        n_prev_motions=2,
        n_motions=3,
        n_diff_steps=n_diff_steps,
    )
    torch_model.diffusion_sched = DiffusionSchedule(n_diff_steps, "cosine")
    return torch_model.eval()


def _tiny_hubert_config():
    config = HubertConfig(
        conv_dim=(4, 4),
        conv_kernel=(3, 3),
        conv_stride=(2, 2),
        conv_bias=False,
        feat_extract_norm="group",
        feat_proj_layer_norm=True,
        feat_proj_dropout=0.0,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
        layer_norm_eps=1e-5,
    )
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize("mode", ("linear", "quadratic", "sigmoid", "cosine"))
def test_mlx_diffusion_schedule_matches_torch(mode):
    from src.models.JoyVASA.dit_talking_head import DiffusionSchedule
    from src.models.mlx_joyvasa_motion_model import MlxDiffusionSchedule

    torch_schedule = DiffusionSchedule(num_steps=12, mode=mode)
    mlx_schedule = MlxDiffusionSchedule(num_steps=12, mode=mode)

    for attr in ("betas", "alphas", "alpha_bars", "sigmas_flex", "sigmas_inflex"):
        torch_values = getattr(torch_schedule, attr).detach().cpu().numpy()
        mlx_values = np.array(getattr(mlx_schedule, attr))
        np.testing.assert_allclose(mlx_values, torch_values, rtol=2e-4, atol=2e-6)

    t = np.array([1, 5, 12], dtype=np.int64)
    torch_sigmas = torch_schedule.get_sigmas(torch.from_numpy(t), flexibility=0.35).detach().cpu().numpy()
    mlx_sigmas = np.array(mlx_schedule.get_sigmas(mx.array(t), flexibility=0.35))
    np.testing.assert_allclose(mlx_sigmas, torch_sigmas, rtol=2e-4, atol=2e-6)


def test_mlx_positional_encoding_matches_torch():
    from src.models.JoyVASA.common import PositionalEncoding
    from src.models.mlx_joyvasa_motion_model import MlxPositionalEncoding

    x = np.arange(2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8) / 100.0
    torch_pe = PositionalEncoding(d_model=8, dropout=0.0, max_len=16).eval()
    mlx_pe = MlxPositionalEncoding(d_model=8, max_len=16)

    with torch.no_grad():
        torch_out = torch_pe(torch.from_numpy(x)).numpy()
    mlx_out = np.array(mlx_pe(mx.array(x)))

    np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("t", "s", "frame_width", "expansion"),
    (
        (5, 5, 1, 0),
        (5, 10, 2, 1),
        (7, 9, 1, 2),
    ),
)
def test_mlx_enc_dec_mask_matches_torch(t, s, frame_width, expansion):
    from src.models.JoyVASA.common import enc_dec_mask
    from src.models.mlx_joyvasa_motion_model import mlx_enc_dec_mask

    torch_mask = enc_dec_mask(t, s, frame_width=frame_width, expansion=expansion, device="cpu").numpy()
    mlx_mask = np.array(mlx_enc_dec_mask(t, s, frame_width=frame_width, expansion=expansion))

    np.testing.assert_array_equal(mlx_mask, torch_mask)


@pytest.mark.parametrize("audio_len", (640, 719, 720, 721))
def test_mlx_pad_audio_matches_torch(audio_len):
    from src.models.JoyVASA.common import pad_audio
    from src.models.mlx_joyvasa_motion_model import mlx_pad_audio

    audio = np.linspace(-1.0, 1.0, audio_len, dtype=np.float32).reshape(1, audio_len)

    torch_out = pad_audio(torch.from_numpy(audio)).numpy()
    mlx_out = np.array(mlx_pad_audio(mx.array(audio)))

    np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-6, atol=1e-7)


def test_mlx_hubert_model_matches_torch():
    from src.models.JoyVASA.hubert import HubertModel
    from src.models.mlx_joyvasa_audio_model import MlxHubertConfig, MlxHubertModel

    torch.manual_seed(101)
    rng = np.random.default_rng(101)
    config = _tiny_hubert_config()
    torch_model = HubertModel(config).eval()
    mlx_model = MlxHubertModel(MlxHubertConfig.from_dict(config.to_dict()))
    mlx_model.load_pytorch_state_dict(torch_model.state_dict())

    audio = rng.normal(size=(2, 64)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(audio), output_fps=25, frame_num=6).last_hidden_state.numpy()
    mlx_out = np.array(mlx_model(mx.array(audio), output_fps=25, frame_num=6))

    np.testing.assert_allclose(mlx_out, torch_out, rtol=2e-3, atol=2e-3)


def test_mlx_joyvasa_audio_feature_extractor_matches_torch():
    from src.models.JoyVASA.hubert import HubertModel
    from src.models.mlx_joyvasa_audio_model import MlxHubertConfig, MlxJoyVASAAudioFeatureExtractor

    torch.manual_seed(103)
    rng = np.random.default_rng(103)
    config = _tiny_hubert_config()
    torch_hubert = HubertModel(config).eval()
    torch_audio_feature_map = torch_nn.Linear(config.hidden_size, 5).eval()
    mlx_model = MlxJoyVASAAudioFeatureExtractor(
        MlxHubertConfig.from_dict(config.to_dict()),
        feature_dim=5,
        fps=25,
        n_motions=4,
    )
    mlx_model.load_pytorch_state_dicts(
        torch_hubert.state_dict(),
        {
            "audio_feature_map.weight": torch_audio_feature_map.weight,
            "audio_feature_map.bias": torch_audio_feature_map.bias,
        },
    )

    audio = rng.normal(size=(1, 64)).astype(np.float32)

    with torch.no_grad():
        hidden_states = torch_hubert(torch.from_numpy(audio), output_fps=25, frame_num=8).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = torch_f.interpolate(hidden_states, size=4, align_corners=False, mode="linear")
        hidden_states = hidden_states.transpose(1, 2)
        torch_out = torch_audio_feature_map(hidden_states).numpy()
    mlx_out = np.array(mlx_model(mx.array(audio), frame_num=4))

    np.testing.assert_allclose(mlx_out, torch_out, rtol=2e-3, atol=2e-3)


def test_mlx_joyvasa_audio_feature_extractor_pad_matches_torch():
    from src.models.JoyVASA.common import pad_audio
    from src.models.JoyVASA.hubert import HubertModel
    from src.models.mlx_joyvasa_audio_model import MlxHubertConfig, MlxJoyVASAAudioFeatureExtractor

    torch.manual_seed(107)
    rng = np.random.default_rng(107)
    config = _tiny_hubert_config()
    torch_hubert = HubertModel(config).eval()
    torch_audio_feature_map = torch_nn.Linear(config.hidden_size, 5).eval()
    mlx_model = MlxJoyVASAAudioFeatureExtractor(
        MlxHubertConfig.from_dict(config.to_dict()),
        feature_dim=5,
        fps=25,
        n_motions=4,
    )
    mlx_model.load_pytorch_state_dicts(
        torch_hubert.state_dict(),
        {
            "audio_feature_map.weight": torch_audio_feature_map.weight,
            "audio_feature_map.bias": torch_audio_feature_map.bias,
        },
    )

    audio = rng.normal(size=(1, 63)).astype(np.float32)

    with torch.no_grad():
        torch_audio = pad_audio(torch.from_numpy(audio))
        hidden_states = torch_hubert(torch_audio, output_fps=25, frame_num=8).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = torch_f.interpolate(hidden_states, size=4, align_corners=False, mode="linear")
        hidden_states = hidden_states.transpose(1, 2)
        torch_out = torch_audio_feature_map(hidden_states).numpy()
    mlx_out = np.array(mlx_model.extract_audio_feature(mx.array(audio), frame_num=4))

    np.testing.assert_allclose(mlx_out, torch_out, rtol=2e-3, atol=2e-3)


def test_mlx_joyvasa_audio_npz_roundtrip(tmp_path):
    from src.models.JoyVASA.hubert import HubertModel
    from src.models.mlx_joyvasa_audio_model import (
        MlxHubertConfig,
        MlxJoyVASAAudioFeatureExtractor,
        load_mlx_joyvasa_audio_npz,
        save_mlx_joyvasa_audio_npz,
    )

    torch.manual_seed(109)
    rng = np.random.default_rng(109)
    config = _tiny_hubert_config()
    torch_hubert = HubertModel(config).eval()
    torch_audio_feature_map = torch_nn.Linear(config.hidden_size, 5).eval()
    mlx_model = MlxJoyVASAAudioFeatureExtractor(
        MlxHubertConfig.from_dict(config.to_dict()),
        feature_dim=5,
        fps=25,
        n_motions=4,
    )
    mlx_model.load_pytorch_state_dicts(
        torch_hubert.state_dict(),
        {
            "audio_feature_map.weight": torch_audio_feature_map.weight,
            "audio_feature_map.bias": torch_audio_feature_map.bias,
        },
    )

    path = tmp_path / "joyvasa_audio.npz"
    save_mlx_joyvasa_audio_npz(path, mlx_model, pad_mode="replicate")
    loaded = load_mlx_joyvasa_audio_npz(path)
    audio = rng.normal(size=(1, 64)).astype(np.float32)

    np.testing.assert_allclose(
        np.array(loaded(mx.array(audio), frame_num=4)),
        np.array(mlx_model(mx.array(audio), frame_num=4)),
        rtol=1e-6,
        atol=1e-7,
    )
    assert loaded.pad_mode == "replicate"


def test_mlx_transformer_decoder_layer_matches_torch():
    from src.models.JoyVASA.dit_talking_head import DenoisingNetwork
    from src.models.mlx_joyvasa_motion_model import MlxDenoisingNetwork

    torch.manual_seed(0)
    rng = np.random.default_rng(7)
    torch_model = DenoisingNetwork(
        device="cpu",
        motion_feat_dim=4,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        n_prev_motions=2,
        n_motions=3,
        n_diff_steps=6,
    ).eval()
    mlx_model = MlxDenoisingNetwork(
        motion_feat_dim=4,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        n_prev_motions=2,
        n_motions=3,
        n_diff_steps=6,
    )
    mlx_model.load_pytorch_state_dict(torch_model.state_dict())

    memory = rng.normal(size=(1, 5, 8)).astype(np.float32)
    target = rng.normal(size=(1, 5, 8)).astype(np.float32)

    with torch.no_grad():
        torch_layer = torch_model.transformer.layers[0]
        torch_out = torch_layer(
            torch.from_numpy(target),
            torch.from_numpy(memory),
            memory_mask=torch_model.alignment_mask,
        ).numpy()
    mlx_out = np.array(mlx_model.layers[0](mx.array(target), mx.array(memory), mlx_model.alignment_mask))

    np.testing.assert_allclose(mlx_out, torch_out, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("use_indicator", (False, True))
def test_mlx_denoising_network_matches_torch(use_indicator):
    from src.models.JoyVASA.dit_talking_head import DenoisingNetwork
    from src.models.mlx_joyvasa_motion_model import MlxDenoisingNetwork

    torch.manual_seed(1)
    rng = np.random.default_rng(11)
    torch_model = DenoisingNetwork(
        device="cpu",
        motion_feat_dim=4,
        use_indicator=use_indicator,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        n_prev_motions=2,
        n_motions=3,
        n_diff_steps=6,
    ).eval()
    mlx_model = MlxDenoisingNetwork(
        motion_feat_dim=4,
        use_indicator=use_indicator,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        n_prev_motions=2,
        n_motions=3,
        n_diff_steps=6,
    )
    mlx_model.load_pytorch_state_dict(torch_model.state_dict())

    motion = rng.normal(size=(2, 3, 4)).astype(np.float32)
    audio = rng.normal(size=(2, 3, 8)).astype(np.float32)
    prev_motion = rng.normal(size=(2, 2, 4)).astype(np.float32)
    prev_audio = rng.normal(size=(2, 2, 8)).astype(np.float32)
    step = np.array([2, 5], dtype=np.int64)
    indicator = rng.integers(0, 2, size=(2, 3)).astype(np.float32) if use_indicator else None

    with torch.no_grad():
        torch_out = torch_model(
            torch.from_numpy(motion),
            torch.from_numpy(audio),
            torch.from_numpy(prev_motion),
            torch.from_numpy(prev_audio),
            torch.from_numpy(step),
            torch.from_numpy(indicator) if indicator is not None else None,
        ).numpy()
    mlx_out = np.array(
        mlx_model(
            mx.array(motion),
            mx.array(audio),
            mx.array(prev_motion),
            mx.array(prev_audio),
            mx.array(step),
            mx.array(indicator) if indicator is not None else None,
        )
    )

    np.testing.assert_allclose(mlx_out, torch_out, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("target", ("sample", "noise"))
def test_mlx_motion_sampler_matches_torch_without_cfg(target, monkeypatch):
    import src.models.JoyVASA.dit_talking_head as torch_dit
    from src.models.mlx_joyvasa_motion_model import MlxJoyVASAMotionModel

    torch.manual_seed(2)
    rng = np.random.default_rng(23)
    torch_model = _make_torch_motion_model(
        target=target,
        use_indicator=False,
        n_diff_steps=3,
        guiding_conditions="",
    )
    mlx_model = MlxJoyVASAMotionModel(
        target=target,
        motion_feat_dim=4,
        n_motions=3,
        n_prev_motions=2,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        n_diff_steps=3,
        diff_schedule="cosine",
        guiding_conditions="",
    )
    mlx_model.load_pytorch_state_dict(torch_model.state_dict())

    audio = rng.normal(size=(2, 3, 8)).astype(np.float32)
    prev_motion = rng.normal(size=(2, 2, 4)).astype(np.float32)
    prev_audio = rng.normal(size=(2, 2, 8)).astype(np.float32)
    motion_at_t = rng.normal(size=(2, 3, 4)).astype(np.float32)
    noise_by_step = {
        3: rng.normal(size=(2, 3, 4)).astype(np.float32),
        2: rng.normal(size=(2, 3, 4)).astype(np.float32),
    }
    noise_iter = iter([torch.from_numpy(noise_by_step[3]), torch.from_numpy(noise_by_step[2])])

    monkeypatch.setattr(torch, "randn_like", lambda value: next(noise_iter).to(dtype=value.dtype))
    monkeypatch.setattr(torch_dit, "tqdm", lambda values: values)

    with torch.no_grad():
        torch_motion, torch_noise, torch_audio = torch_model.sample(
            torch.from_numpy(audio),
            torch.from_numpy(prev_motion),
            torch.from_numpy(prev_audio),
            torch.from_numpy(motion_at_t),
            cfg_cond=[],
            dynamic_threshold=0,
        )
    mlx_motion, mlx_noise, mlx_audio = mlx_model.sample(
        mx.array(audio),
        mx.array(prev_motion),
        mx.array(prev_audio),
        mx.array(motion_at_t),
        cfg_cond=[],
        dynamic_threshold=0,
        noise_by_step=noise_by_step,
    )

    np.testing.assert_allclose(np.array(mlx_motion), torch_motion.numpy(), rtol=6e-3, atol=6e-3)
    np.testing.assert_allclose(np.array(mlx_noise), torch_noise.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.array(mlx_audio), torch_audio.numpy(), rtol=1e-6, atol=1e-7)


def test_mlx_motion_sampler_matches_torch_with_audio_cfg(monkeypatch):
    import src.models.JoyVASA.dit_talking_head as torch_dit
    from src.models.mlx_joyvasa_motion_model import MlxJoyVASAMotionModel

    torch.manual_seed(3)
    rng = np.random.default_rng(29)
    torch_model = _make_torch_motion_model(
        target="sample",
        use_indicator=True,
        n_diff_steps=3,
        guiding_conditions="audio,",
        cfg_mode="incremental",
    )
    mlx_model = MlxJoyVASAMotionModel(
        target="sample",
        motion_feat_dim=4,
        n_motions=3,
        n_prev_motions=2,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        n_diff_steps=3,
        diff_schedule="cosine",
        cfg_mode="incremental",
        guiding_conditions="audio,",
        use_indicator=True,
    )
    mlx_model.load_pytorch_state_dict(torch_model.state_dict())

    audio = rng.normal(size=(1, 3, 8)).astype(np.float32)
    prev_motion = rng.normal(size=(1, 2, 4)).astype(np.float32)
    prev_audio = rng.normal(size=(1, 2, 8)).astype(np.float32)
    motion_at_t = rng.normal(size=(1, 3, 4)).astype(np.float32)
    indicator = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    noise_by_step = {
        3: rng.normal(size=(1, 3, 4)).astype(np.float32),
        2: rng.normal(size=(1, 3, 4)).astype(np.float32),
    }
    noise_iter = iter([torch.from_numpy(noise_by_step[3]), torch.from_numpy(noise_by_step[2])])

    monkeypatch.setattr(torch, "randn_like", lambda value: next(noise_iter).to(dtype=value.dtype))
    monkeypatch.setattr(torch_dit, "tqdm", lambda values: values)

    with torch.no_grad():
        torch_motion, torch_noise, torch_audio = torch_model.sample(
            torch.from_numpy(audio),
            torch.from_numpy(prev_motion),
            torch.from_numpy(prev_audio),
            torch.from_numpy(motion_at_t),
            indicator=torch.from_numpy(indicator),
            cfg_mode="incremental",
            cfg_cond=["audio"],
            cfg_scale=1.2,
            dynamic_threshold=0,
        )
    mlx_motion, mlx_noise, mlx_audio = mlx_model.sample(
        mx.array(audio),
        mx.array(prev_motion),
        mx.array(prev_audio),
        mx.array(motion_at_t),
        indicator=mx.array(indicator),
        cfg_mode="incremental",
        cfg_cond=["audio"],
        cfg_scale=1.2,
        dynamic_threshold=0,
        noise_by_step=noise_by_step,
    )

    np.testing.assert_allclose(np.array(mlx_motion), torch_motion.numpy(), rtol=6e-3, atol=6e-3)
    np.testing.assert_allclose(np.array(mlx_noise), torch_noise.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.array(mlx_audio), torch_audio.numpy(), rtol=1e-6, atol=1e-7)


def test_exported_mlx_motion_npz_reloads_and_matches_torch(tmp_path, monkeypatch):
    import src.models.JoyVASA.dit_talking_head as torch_dit
    from src.models.mlx_joyvasa_motion_model import (
        export_mlx_joyvasa_motion_from_pytorch_checkpoint,
        load_mlx_joyvasa_motion_npz,
    )

    torch.manual_seed(5)
    rng = np.random.default_rng(31)
    torch_model = _make_torch_motion_model(
        target="sample",
        use_indicator=True,
        n_diff_steps=3,
        guiding_conditions="audio,",
        cfg_mode="independent",
    )
    checkpoint_path = tmp_path / "motion_generator.pt"
    weights_path = tmp_path / "motion_generator_mlx.npz"
    args = Namespace(
        target="sample",
        architecture="decoder",
        motion_feat_dim=4,
        fps=25,
        n_motions=3,
        n_prev_motions=2,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        no_use_learnable_pe=True,
        n_diff_steps=3,
        diff_schedule="cosine",
        use_indicator=True,
    )
    torch.save({"args": args, "model": torch_model.state_dict()}, checkpoint_path)

    export_mlx_joyvasa_motion_from_pytorch_checkpoint(
        checkpoint_path,
        weights_path,
        cfg_mode="independent",
        guiding_conditions="audio,",
    )
    mlx_model = load_mlx_joyvasa_motion_npz(weights_path)

    assert mlx_model.n_motions == 3
    assert mlx_model.n_prev_motions == 2
    assert mlx_model.feature_dim == 8
    assert mlx_model.cfg_mode == "independent"
    assert mlx_model.guiding_conditions == ["audio"]
    assert mlx_model.denoising_net.use_indicator is True

    audio = rng.normal(size=(1, 3, 8)).astype(np.float32)
    prev_motion = rng.normal(size=(1, 2, 4)).astype(np.float32)
    prev_audio = rng.normal(size=(1, 2, 8)).astype(np.float32)
    motion_at_t = rng.normal(size=(1, 3, 4)).astype(np.float32)
    indicator = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    noise_by_step = {
        3: rng.normal(size=(1, 3, 4)).astype(np.float32),
        2: rng.normal(size=(1, 3, 4)).astype(np.float32),
    }
    noise_iter = iter([torch.from_numpy(noise_by_step[3]), torch.from_numpy(noise_by_step[2])])

    monkeypatch.setattr(torch, "randn_like", lambda value: next(noise_iter).to(dtype=value.dtype))
    monkeypatch.setattr(torch_dit, "tqdm", lambda values: values)

    with torch.no_grad():
        torch_motion, torch_noise, torch_audio = torch_model.sample(
            torch.from_numpy(audio),
            torch.from_numpy(prev_motion),
            torch.from_numpy(prev_audio),
            torch.from_numpy(motion_at_t),
            indicator=torch.from_numpy(indicator),
            cfg_mode="independent",
            cfg_cond=["audio"],
            cfg_scale=1.2,
            dynamic_threshold=0,
        )
    mlx_motion, mlx_noise, mlx_audio = mlx_model.sample(
        mx.array(audio),
        mx.array(prev_motion),
        mx.array(prev_audio),
        mx.array(motion_at_t),
        indicator=mx.array(indicator),
        cfg_mode="independent",
        cfg_cond=["audio"],
        cfg_scale=1.2,
        dynamic_threshold=0,
        noise_by_step=noise_by_step,
    )

    np.testing.assert_allclose(np.array(mlx_motion), torch_motion.numpy(), rtol=6e-3, atol=6e-3)
    np.testing.assert_allclose(np.array(mlx_noise), torch_noise.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.array(mlx_audio), torch_audio.numpy(), rtol=1e-6, atol=1e-7)


def test_pipeline_mlx_motion_bridge_matches_direct_sampler():
    from src.models.mlx_joyvasa_motion_model import MlxJoyVASAMotionModel
    from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

    torch.manual_seed(7)
    rng = np.random.default_rng(37)
    torch_model = _make_torch_motion_model(
        target="sample",
        use_indicator=True,
        n_diff_steps=1,
        guiding_conditions="audio,",
        cfg_mode="incremental",
    )
    mlx_model = MlxJoyVASAMotionModel(
        target="sample",
        motion_feat_dim=4,
        n_motions=3,
        n_prev_motions=2,
        feature_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        align_mask_width=2,
        n_diff_steps=1,
        diff_schedule="cosine",
        cfg_mode="incremental",
        guiding_conditions="audio,",
        use_indicator=True,
    )
    mlx_model.load_pytorch_state_dict(torch_model.state_dict())

    pipe = JoyVASAAudio2MotionPipeline.__new__(JoyVASAAudio2MotionPipeline)
    pipe.motion_generator = mlx_model
    pipe.cfg_mode = "incremental"
    pipe.cfg_cond = ["audio"]
    pipe.cfg_scale = 1.2

    audio = rng.normal(size=(1, 3, 8)).astype(np.float32)
    prev_motion = rng.normal(size=(1, 2, 4)).astype(np.float32)
    prev_audio = rng.normal(size=(1, 2, 8)).astype(np.float32)
    motion_at_t = rng.normal(size=(1, 3, 4)).astype(np.float32)
    indicator = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)

    bridge_motion, bridge_noise, bridge_audio = pipe._sample_mlx_motion_from_audio_feature(
        torch.from_numpy(audio),
        prev_motion_feat=torch.from_numpy(prev_motion),
        prev_audio_feat=torch.from_numpy(prev_audio),
        motion_at_T=torch.from_numpy(motion_at_t),
        indicator=torch.from_numpy(indicator),
    )
    direct_motion, direct_noise, direct_audio = mlx_model.sample(
        mx.array(audio),
        mx.array(prev_motion),
        mx.array(prev_audio),
        mx.array(motion_at_t),
        indicator=mx.array(indicator),
        cfg_mode="incremental",
        cfg_cond=["audio"],
        cfg_scale=1.2,
        dynamic_threshold=0,
    )

    np.testing.assert_allclose(np.array(bridge_motion), np.array(direct_motion), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.array(bridge_noise), np.array(direct_noise), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.array(bridge_audio), np.array(direct_audio), rtol=1e-6, atol=1e-7)
