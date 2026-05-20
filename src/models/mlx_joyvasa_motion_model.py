from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _to_mx(value, dtype=mx.float32):
    if isinstance(value, mx.array):
        return value.astype(dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return mx.array(np.asarray(value)).astype(dtype)


def _to_mx_index(value):
    if isinstance(value, mx.array):
        return value.astype(mx.int32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return mx.array(np.asarray(value, dtype=np.int32))


def _gelu(x):
    return nn.gelu(x)


class MlxDiffusionSchedule:
    def __init__(self, num_steps, mode="linear", beta_1=1e-4, beta_T=0.02, s=0.008):
        if mode == "linear":
            betas = mx.linspace(beta_1, beta_T, num_steps)
        elif mode == "quadratic":
            betas = mx.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == "sigmoid":
            betas = mx.sigmoid(mx.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1
        elif mode == "cosine":
            steps = num_steps + 1
            x = mx.linspace(0, num_steps, steps)
            alpha_bars = mx.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = mx.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown diffusion schedule {mode}!")

        betas = mx.concatenate([mx.zeros((1,), dtype=mx.float32), betas.astype(mx.float32)])
        alphas = 1 - betas
        alpha_bars = mx.exp(mx.cumsum(mx.log(alphas), axis=0))
        sigmas_flex = mx.sqrt(betas)
        sigmas_inflex = mx.zeros_like(sigmas_flex)
        inflex_values = ((1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])) * betas[1:]
        sigmas_inflex = mx.concatenate([sigmas_inflex[:1], mx.sqrt(inflex_values)], axis=0)

        self.num_steps = num_steps
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sigmas_flex = sigmas_flex
        self.sigmas_inflex = sigmas_inflex

    def get_sigmas(self, t, flexibility=0):
        if not 0 <= flexibility <= 1:
            raise ValueError("flexibility must be in [0, 1]")
        return self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)


class MlxPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super().__init__()
        position = mx.arange(max_len, dtype=mx.float32)[:, None]
        div_term = mx.exp(mx.arange(0, d_model, 2, dtype=mx.float32) * (-math.log(10000.0) / d_model))
        pe = mx.zeros((max_len, d_model), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self.pe = mx.expand_dims(pe, axis=0)

    def __call__(self, x):
        return x + self.pe[:, x.shape[1], :]


def mlx_enc_dec_mask(t, s, frame_width=2, expansion=0):
    rows = []
    for i in range(t):
        row = np.ones((s,), dtype=bool)
        row[max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = False
        rows.append(row)
    return mx.array(np.stack(rows, axis=0))


def mlx_pad_audio(audio, audio_unit=320, pad_threshold=80):
    _, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = _reflect_pad_audio(audio, reflect_len)
            audio = _reflect_pad_audio(audio, reflect_len)
        if replicate_len > 0:
            left = audio[:, :1]
            right = audio[:, -1:]
            audio = mx.concatenate([left, audio, right], axis=1)
    return audio


def _reflect_pad_audio(audio, pad):
    _, audio_len = audio.shape
    left_idx = mx.arange(pad, 0, -1)
    right_idx = mx.arange(audio_len - 2, audio_len - pad - 2, -1)
    left = mx.take(audio, left_idx, axis=1)
    right = mx.take(audio, right_idx, axis=1)
    return mx.concatenate([left, audio, right], axis=1)


class MlxTransformerDecoderLayer(nn.Module):
    def __init__(self, feature_dim=512, n_heads=8, mlp_ratio=4):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(feature_dim, n_heads, bias=True)
        self.multihead_attn = nn.MultiHeadAttention(feature_dim, n_heads, bias=True)
        self.linear1 = nn.Linear(feature_dim, mlp_ratio * feature_dim)
        self.linear2 = nn.Linear(mlp_ratio * feature_dim, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)

    def __call__(self, x, memory, memory_mask=None):
        y = self.self_attn(x, x, x)
        x = self.norm1(x + y)
        y = self.multihead_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + y)
        y = self.linear2(_gelu(self.linear1(x)))
        return self.norm3(x + y)


class MlxDenoisingNetwork(nn.Module):
    def __init__(
        self,
        motion_feat_dim=76,
        use_indicator=None,
        architecture="decoder",
        feature_dim=512,
        n_heads=8,
        n_layers=8,
        mlp_ratio=4,
        align_mask_width=1,
        no_use_learnable_pe=True,
        n_prev_motions=10,
        n_motions=100,
        n_diff_steps=500,
    ):
        super().__init__()
        if architecture != "decoder":
            raise ValueError(f"Unknown architecture: {architecture}")

        self.motion_feat_dim = motion_feat_dim
        self.use_indicator = use_indicator
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        self.use_learnable_pe = not no_use_learnable_pe
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions

        self.TE = MlxPositionalEncoding(feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = [
            nn.Linear(feature_dim, feature_dim),
            nn.Linear(feature_dim, feature_dim),
        ]

        if self.use_learnable_pe:
            self.PE = mx.zeros((1, 1 + n_prev_motions + n_motions, feature_dim), dtype=mx.float32)
        else:
            self.PE = MlxPositionalEncoding(feature_dim)

        input_dim = motion_feat_dim + (1 if use_indicator else 0)
        self.feature_proj = nn.Linear(input_dim, feature_dim)
        self.layers = [MlxTransformerDecoderLayer(feature_dim, n_heads, mlp_ratio) for _ in range(n_layers)]
        if align_mask_width > 0:
            motion_len = n_prev_motions + n_motions
            bool_mask = mlx_enc_dec_mask(motion_len, motion_len, frame_width=1, expansion=align_mask_width - 1)
            self.alignment_mask = mx.where(bool_mask, mx.array(-1e9, dtype=mx.float32), mx.array(0.0, dtype=mx.float32))
        else:
            self.alignment_mask = None
        self.motion_dec = [
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Linear(feature_dim // 2, motion_feat_dim),
        ]

    def __call__(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        motion_feat = motion_feat.astype(audio_feat.dtype)
        step = _to_mx_index(step)
        diff_step_embedding = mx.take(self.TE.pe[0], step, axis=0)
        diff_step_embedding = self.diff_step_map[1](_gelu(self.diff_step_map[0](diff_step_embedding)))
        diff_step_embedding = mx.expand_dims(diff_step_embedding, axis=1)

        if indicator is not None:
            zeros = mx.zeros((indicator.shape[0], self.n_prev_motions), dtype=indicator.dtype)
            indicator = mx.concatenate([zeros, indicator], axis=1)
            indicator = mx.expand_dims(indicator, axis=-1)

        feats_in = mx.concatenate([prev_motion_feat, motion_feat], axis=1)
        if self.use_indicator:
            feats_in = mx.concatenate([feats_in, indicator], axis=-1)
        feats_in = self.feature_proj(feats_in)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE + diff_step_embedding
        else:
            feats_in = self.PE(feats_in) + diff_step_embedding

        audio_feat_in = mx.concatenate([prev_audio_feat, audio_feat], axis=1)
        for layer in self.layers:
            feats_in = layer(feats_in, audio_feat_in, self.alignment_mask)

        return self.motion_dec[1](_gelu(self.motion_dec[0](feats_in)))

    def load_pytorch_state_dict(self, state_dict, prefix=""):
        def get(name):
            return _to_mx(state_dict[prefix + name])

        self.diff_step_map[0].weight = get("diff_step_map.0.weight")
        self.diff_step_map[0].bias = get("diff_step_map.0.bias")
        self.diff_step_map[1].weight = get("diff_step_map.2.weight")
        self.diff_step_map[1].bias = get("diff_step_map.2.bias")

        if self.use_learnable_pe:
            self.PE = get("PE")

        self.feature_proj.weight = get("feature_proj.weight")
        self.feature_proj.bias = get("feature_proj.bias")

        for idx, layer in enumerate(self.layers):
            layer_prefix = f"transformer.layers.{idx}."
            _load_mha(layer.self_attn, state_dict, prefix + layer_prefix + "self_attn.")
            _load_mha(layer.multihead_attn, state_dict, prefix + layer_prefix + "multihead_attn.")
            _load_linear(layer.linear1, state_dict, prefix + layer_prefix + "linear1.")
            _load_linear(layer.linear2, state_dict, prefix + layer_prefix + "linear2.")
            _load_layer_norm(layer.norm1, state_dict, prefix + layer_prefix + "norm1.")
            _load_layer_norm(layer.norm2, state_dict, prefix + layer_prefix + "norm2.")
            _load_layer_norm(layer.norm3, state_dict, prefix + layer_prefix + "norm3.")

        self.motion_dec[0].weight = get("motion_dec.0.weight")
        self.motion_dec[0].bias = get("motion_dec.0.bias")
        self.motion_dec[1].weight = get("motion_dec.2.weight")
        self.motion_dec[1].bias = get("motion_dec.2.bias")
        mx.eval(self.parameters())


def _load_linear(module, state_dict, prefix):
    module.weight = _to_mx(state_dict[prefix + "weight"])
    module.bias = _to_mx(state_dict[prefix + "bias"])


def _load_layer_norm(module, state_dict, prefix):
    module.weight = _to_mx(state_dict[prefix + "weight"])
    module.bias = _to_mx(state_dict[prefix + "bias"])


def _load_mha(module, state_dict, prefix):
    in_proj_weight = _to_mx(state_dict[prefix + "in_proj_weight"])
    in_proj_bias = _to_mx(state_dict[prefix + "in_proj_bias"])
    q_w, k_w, v_w = mx.split(in_proj_weight, 3, axis=0)
    q_b, k_b, v_b = mx.split(in_proj_bias, 3, axis=0)

    module.query_proj.weight = q_w
    module.query_proj.bias = q_b
    module.key_proj.weight = k_w
    module.key_proj.bias = k_b
    module.value_proj.weight = v_w
    module.value_proj.bias = v_b
    module.out_proj.weight = _to_mx(state_dict[prefix + "out_proj.weight"])
    module.out_proj.bias = _to_mx(state_dict[prefix + "out_proj.bias"])


class MlxJoyVASAMotionModel(nn.Module):
    def __init__(
        self,
        target="sample",
        architecture="decoder",
        motion_feat_dim=76,
        fps=25,
        n_motions=100,
        n_prev_motions=10,
        feature_dim=512,
        n_heads=8,
        n_layers=8,
        mlp_ratio=4,
        align_mask_width=1,
        no_use_learnable_pe=True,
        n_diff_steps=500,
        diff_schedule="cosine",
        cfg_mode="incremental",
        guiding_conditions="audio,",
        use_indicator=None,
    ):
        super().__init__()
        if architecture != "decoder":
            raise ValueError(f"Unknown architecture {architecture}!")

        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim
        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.feature_dim = feature_dim
        self.cfg_mode = cfg_mode
        self.guiding_conditions = _normalize_cfg_conditions(guiding_conditions)

        self.start_motion_feat = mx.random.normal((1, n_prev_motions, motion_feat_dim))
        self.start_audio_feat = mx.random.normal((1, n_prev_motions, feature_dim))
        if "audio" in self.guiding_conditions:
            self.null_audio_feat = mx.random.normal((1, 1, feature_dim))

        self.denoising_net = MlxDenoisingNetwork(
            motion_feat_dim=motion_feat_dim,
            use_indicator=use_indicator,
            architecture=architecture,
            feature_dim=feature_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            no_use_learnable_pe=no_use_learnable_pe,
            n_prev_motions=n_prev_motions,
            n_motions=n_motions,
            n_diff_steps=n_diff_steps,
        )
        self.diffusion_sched = MlxDiffusionSchedule(n_diff_steps, diff_schedule)

    def sample(
        self,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        motion_at_T=None,
        indicator=None,
        cfg_mode=None,
        cfg_cond=None,
        cfg_scale=1.15,
        flexibility=0,
        dynamic_threshold=None,
        ret_traj=False,
        noise_by_step=None,
    ):
        if dynamic_threshold:
            raise NotImplementedError("dynamic_threshold is not ported to MLX yet")

        if audio_or_feat.ndim == 2:
            raise NotImplementedError("Raw-audio JoyVASA feature extraction is not ported to MLX yet")
        if audio_or_feat.ndim != 3:
            raise ValueError(f"Incorrect audio input shape {audio_or_feat.shape}")
        if audio_or_feat.shape[1] != self.n_motions:
            raise ValueError(f"Incorrect audio feature length {audio_or_feat.shape[1]}")

        batch_size = audio_or_feat.shape[0]
        audio_feat = audio_or_feat

        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        cfg_cond = self.guiding_conditions if cfg_cond is None else _normalize_cfg_conditions(cfg_cond)
        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if cfg_cond:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ["audio"].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []

        if prev_motion_feat is None:
            prev_motion_feat = mx.broadcast_to(
                self.start_motion_feat.astype(audio_feat.dtype),
                (batch_size, self.n_prev_motions, self.motion_feat_dim),
            )
        if prev_audio_feat is None:
            prev_audio_feat = mx.broadcast_to(
                self.start_audio_feat.astype(audio_feat.dtype),
                (batch_size, self.n_prev_motions, self.feature_dim),
            )
        if motion_at_T is None:
            motion_at_T = mx.random.normal((batch_size, self.n_motions, self.motion_feat_dim), dtype=audio_feat.dtype)

        if "audio" in cfg_cond:
            audio_feat_null = mx.broadcast_to(
                self.null_audio_feat.astype(audio_feat.dtype),
                (batch_size, self.n_motions, self.feature_dim),
            )
        else:
            audio_feat_null = audio_feat

        audio_feat_in = [audio_feat_null]
        for cond in cfg_cond:
            if cond == "audio":
                audio_feat_in.append(audio_feat)

        n_entries = len(audio_feat_in)
        audio_feat_in = mx.concatenate(audio_feat_in, axis=0)
        prev_motion_feat_in = _repeat_batch(prev_motion_feat, n_entries)
        prev_audio_feat_in = _repeat_batch(prev_audio_feat, n_entries)
        indicator_in = _repeat_batch(indicator, n_entries) if indicator is not None else None

        motion_at_t = motion_at_T
        traj = {self.diffusion_sched.num_steps: motion_at_t} if ret_traj else None
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            if t > 1:
                z = _noise_for_step(t, motion_at_T, noise_by_step)
            else:
                z = mx.zeros_like(motion_at_T)

            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_in = _repeat_batch(motion_at_t, n_entries)
            step_in = mx.full((batch_size * n_entries,), t, dtype=mx.int32)
            results = self.denoising_net(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_in,
            )
            results = mx.split(results, n_entries, axis=0)

            target_theta = results[0][:, -self.n_motions:]
            for i in range(0, n_entries - 1):
                scale = cfg_scale[i]
                if cfg_mode == "independent":
                    target_theta = target_theta + scale * (
                        results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:]
                    )
                elif cfg_mode == "incremental":
                    target_theta = target_theta + scale * (
                        results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:]
                    )
                else:
                    raise NotImplementedError(f"Unknown cfg_mode {cfg_mode}")

            if self.target == "noise":
                c0 = 1 / mx.sqrt(alpha)
                c1 = (1 - alpha) / mx.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == "sample":
                c0 = (1 - alpha_bar_prev) * mx.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * mx.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError(f"Unknown target type: {self.target}")

            motion_at_t = motion_next
            mx.eval(motion_at_t)
            if ret_traj:
                traj[t - 1] = motion_at_t

        if ret_traj:
            return traj, motion_at_T, audio_feat
        return motion_at_t, motion_at_T, audio_feat

    def load_pytorch_state_dict(self, state_dict, prefix=""):
        def has(name):
            return prefix + name in state_dict

        def get(name):
            return _to_mx(state_dict[prefix + name])

        self.start_motion_feat = get("start_motion_feat")
        self.start_audio_feat = get("start_audio_feat")
        if has("null_audio_feat"):
            self.null_audio_feat = get("null_audio_feat")

        schedule_prefix = prefix + "diffusion_sched."
        if any(key.startswith(schedule_prefix) for key in state_dict):
            self.diffusion_sched.betas = _to_mx(state_dict[schedule_prefix + "betas"])
            self.diffusion_sched.alphas = _to_mx(state_dict[schedule_prefix + "alphas"])
            self.diffusion_sched.alpha_bars = _to_mx(state_dict[schedule_prefix + "alpha_bars"])
            self.diffusion_sched.sigmas_flex = _to_mx(state_dict[schedule_prefix + "sigmas_flex"])
            self.diffusion_sched.sigmas_inflex = _to_mx(state_dict[schedule_prefix + "sigmas_inflex"])
            self.diffusion_sched.num_steps = int(self.diffusion_sched.betas.shape[0] - 1)

        denoising_prefix = prefix + "denoising_net."
        if any(key.startswith(denoising_prefix) for key in state_dict):
            self.denoising_net.load_pytorch_state_dict(state_dict, denoising_prefix)
        else:
            self.denoising_net.load_pytorch_state_dict(state_dict, prefix)
        mx.eval(self.parameters())


def _normalize_cfg_conditions(cfg_cond):
    if cfg_cond is None:
        return []
    if isinstance(cfg_cond, str):
        cfg_cond = cfg_cond.split(",")
    return [cond for cond in cfg_cond if cond in ["audio"]]


def _repeat_batch(value, n_entries):
    if n_entries == 1:
        return value
    return mx.concatenate([value] * n_entries, axis=0)


def _noise_for_step(t, like, noise_by_step):
    if noise_by_step is None:
        return mx.random.normal(like.shape, dtype=like.dtype)
    return _to_mx(noise_by_step[t], dtype=like.dtype)
