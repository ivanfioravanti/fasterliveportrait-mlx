from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mu
import numpy as np


def _to_mx(value, dtype=mx.float32):
    if isinstance(value, mx.array):
        return value.astype(dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return mx.array(np.asarray(value)).astype(dtype)


def _torch_conv1d_weight_to_mlx(weight):
    weight = _to_mx(weight)
    return weight.transpose(0, 2, 1)


_METADATA_PREFIX = "__joyvasa_audio_mlx__."
_FEAT_EXTRACT_NORM_TO_ID = {"group": 0, "layer": 1}
_ID_TO_FEAT_EXTRACT_NORM = {value: key for key, value in _FEAT_EXTRACT_NORM_TO_ID.items()}
_PAD_MODE_TO_ID = {"zero": 0, "replicate": 1}
_ID_TO_PAD_MODE = {value: key for key, value in _PAD_MODE_TO_ID.items()}


@dataclass(frozen=True)
class MlxHubertConfig:
    conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = False
    feat_extract_norm: str = "group"
    feat_proj_layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16

    @classmethod
    def from_dict(cls, config):
        return cls(
            conv_dim=tuple(config.get("conv_dim", cls.conv_dim)),
            conv_kernel=tuple(config.get("conv_kernel", cls.conv_kernel)),
            conv_stride=tuple(config.get("conv_stride", cls.conv_stride)),
            conv_bias=bool(config.get("conv_bias", cls.conv_bias)),
            feat_extract_norm=config.get("feat_extract_norm", cls.feat_extract_norm),
            feat_proj_layer_norm=bool(config.get("feat_proj_layer_norm", cls.feat_proj_layer_norm)),
            layer_norm_eps=float(config.get("layer_norm_eps", cls.layer_norm_eps)),
            hidden_size=int(config.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(config.get("intermediate_size", cls.intermediate_size)),
            num_attention_heads=int(config.get("num_attention_heads", cls.num_attention_heads)),
            num_hidden_layers=int(config.get("num_hidden_layers", cls.num_hidden_layers)),
            num_conv_pos_embeddings=int(config.get("num_conv_pos_embeddings", cls.num_conv_pos_embeddings)),
            num_conv_pos_embedding_groups=int(
                config.get("num_conv_pos_embedding_groups", cls.num_conv_pos_embedding_groups)
            ),
        )


class MlxHubertConvLayer(nn.Module):
    def __init__(self, config: MlxHubertConfig, layer_id: int):
        super().__init__()
        in_channels = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_channels = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = None
        if config.feat_extract_norm == "group" and layer_id == 0:
            self.layer_norm = nn.GroupNorm(
                num_groups=out_channels,
                dims=out_channels,
                affine=True,
                pytorch_compatible=True,
            )
        elif config.feat_extract_norm == "layer":
            self.layer_norm = nn.LayerNorm(out_channels, eps=config.layer_norm_eps)

    def __call__(self, x):
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return nn.gelu(x)


class MlxHubertFeatureProjection(nn.Module):
    def __init__(self, config: MlxHubertConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.feat_proj_layer_norm = config.feat_proj_layer_norm

    def __call__(self, x):
        if self.feat_proj_layer_norm:
            x = self.layer_norm(x)
        return self.projection(x)


class MlxHubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config: MlxHubertConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.num_pad_remove = 1 if config.num_conv_pos_embeddings % 2 == 0 else 0

    def __call__(self, x):
        x = self.conv(x)
        if self.num_pad_remove > 0:
            x = x[:, :-self.num_pad_remove, :]
        return nn.gelu(x)


class MlxHubertAttention(nn.Module):
    def __init__(self, config: MlxHubertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(x)


class MlxHubertFeedForward(nn.Module):
    def __init__(self, config: MlxHubertConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        return self.output_dense(nn.gelu(self.intermediate_dense(x)))


class MlxHubertEncoderLayer(nn.Module):
    def __init__(self, config: MlxHubertConfig):
        super().__init__()
        self.attention = MlxHubertAttention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = MlxHubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x):
        x = self.layer_norm(x + self.attention(x))
        return self.final_layer_norm(x + self.feed_forward(x))


class MlxHubertEncoder(nn.Module):
    def __init__(self, config: MlxHubertConfig):
        super().__init__()
        self.pos_conv_embed = MlxHubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [MlxHubertEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x):
        x = self.layer_norm(x + self.pos_conv_embed(x))
        for layer in self.layers:
            x = layer(x)
        return x


class MlxHubertModel(nn.Module):
    def __init__(self, config: MlxHubertConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = [
            MlxHubertConvLayer(config, layer_id) for layer_id in range(len(config.conv_dim))
        ]
        self.feature_projection = MlxHubertFeatureProjection(config)
        self.encoder = MlxHubertEncoder(config)

    def __call__(self, input_values, output_fps=25, frame_num=None):
        x = mx.expand_dims(input_values, axis=-1)
        for conv_layer in self.feature_extractor:
            x = conv_layer(x)
        if frame_num is not None:
            feature_len = round(frame_num * 50 / output_fps)
            x = x[:, :feature_len, :]
        x = mlx_linear_interpolate_1d(x, output_len=frame_num or int(x.shape[1] * output_fps / 50))
        x = self.feature_projection(x)
        return self.encoder(x)

    def load_pytorch_state_dict(self, state_dict, prefix=""):
        for idx, layer in enumerate(self.feature_extractor):
            layer.conv.weight = _torch_conv1d_weight_to_mlx(
                state_dict[prefix + f"feature_extractor.conv_layers.{idx}.conv.weight"]
            )
            conv_bias_key = prefix + f"feature_extractor.conv_layers.{idx}.conv.bias"
            if conv_bias_key in state_dict:
                layer.conv.bias = _to_mx(state_dict[conv_bias_key])
            if layer.layer_norm is not None:
                norm_prefix = prefix + f"feature_extractor.conv_layers.{idx}.layer_norm."
                layer.layer_norm.weight = _to_mx(state_dict[norm_prefix + "weight"])
                layer.layer_norm.bias = _to_mx(state_dict[norm_prefix + "bias"])

        _load_layer_norm(self.feature_projection.layer_norm, state_dict, prefix + "feature_projection.layer_norm.")
        _load_linear(self.feature_projection.projection, state_dict, prefix + "feature_projection.projection.")

        _load_pos_conv(self.encoder.pos_conv_embed.conv, state_dict, prefix + "encoder.pos_conv_embed.conv.")
        _load_layer_norm(self.encoder.layer_norm, state_dict, prefix + "encoder.layer_norm.")

        for idx, layer in enumerate(self.encoder.layers):
            layer_prefix = prefix + f"encoder.layers.{idx}."
            _load_hubert_attention(layer.attention, state_dict, layer_prefix + "attention.")
            _load_layer_norm(layer.layer_norm, state_dict, layer_prefix + "layer_norm.")
            _load_linear(layer.feed_forward.intermediate_dense, state_dict, layer_prefix + "feed_forward.intermediate_dense.")
            _load_linear(layer.feed_forward.output_dense, state_dict, layer_prefix + "feed_forward.output_dense.")
            _load_layer_norm(layer.final_layer_norm, state_dict, layer_prefix + "final_layer_norm.")
        mx.eval(self.parameters())


class MlxJoyVASAAudioFeatureExtractor(nn.Module):
    def __init__(self, hubert_config: MlxHubertConfig, feature_dim: int, fps: int, n_motions: int):
        super().__init__()
        self.hubert = MlxHubertModel(hubert_config)
        self.audio_feature_map = nn.Linear(hubert_config.hidden_size, feature_dim)
        self.fps = fps
        self.n_motions = n_motions

    def __call__(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        hidden_states = self.hubert(audio, self.fps, frame_num=frame_num * 2)
        hidden_states = mlx_linear_interpolate_1d(hidden_states, output_len=frame_num)
        return self.audio_feature_map(hidden_states)

    def extract_audio_feature(self, audio, frame_num=None):
        from .mlx_joyvasa_motion_model import mlx_pad_audio

        return self(mlx_pad_audio(_to_mx(audio)), frame_num=frame_num)

    def load_pytorch_state_dicts(self, hubert_state_dict, motion_state_dict):
        self.hubert.load_pytorch_state_dict(hubert_state_dict)
        _load_linear(self.audio_feature_map, motion_state_dict, "audio_feature_map.")
        mx.eval(self.parameters())


def mlx_linear_interpolate_1d(x, output_len):
    input_len = x.shape[1]
    if input_len == output_len:
        return x
    positions = (mx.arange(output_len, dtype=mx.float32) + 0.5) * (input_len / output_len) - 0.5
    positions = mx.clip(positions, 0, input_len - 1)
    left_idx = positions.astype(mx.int32)
    right_idx = mx.minimum(left_idx + 1, input_len - 1)
    weight = mx.expand_dims(positions - left_idx.astype(mx.float32), axis=(0, 2))
    left = mx.take(x, left_idx, axis=1)
    right = mx.take(x, right_idx, axis=1)
    return left * (1 - weight) + right * weight


def _load_linear(module, state_dict, prefix):
    module.weight = _to_mx(state_dict[prefix + "weight"])
    module.bias = _to_mx(state_dict[prefix + "bias"])


def _load_layer_norm(module, state_dict, prefix):
    module.weight = _to_mx(state_dict[prefix + "weight"])
    module.bias = _to_mx(state_dict[prefix + "bias"])


def _load_pos_conv(module, state_dict, prefix):
    bias_key = prefix + "bias"
    if bias_key in state_dict:
        module.bias = _to_mx(state_dict[bias_key])
    g_key = prefix + "parametrizations.weight.original0"
    v_key = prefix + "parametrizations.weight.original1"
    if g_key in state_dict:
        g = _to_mx(state_dict[g_key])
        v = _to_mx(state_dict[v_key])
        norm = mx.sqrt(mx.sum(v * v, axis=(0, 1), keepdims=True))
        weight = v * (g / norm)
    else:
        weight = _to_mx(state_dict[prefix + "weight"])
    module.weight = _torch_conv1d_weight_to_mlx(weight)


def _load_hubert_attention(module, state_dict, prefix):
    _load_linear(module.k_proj, state_dict, prefix + "k_proj.")
    _load_linear(module.v_proj, state_dict, prefix + "v_proj.")
    _load_linear(module.q_proj, state_dict, prefix + "q_proj.")
    _load_linear(module.out_proj, state_dict, prefix + "out_proj.")


def save_mlx_joyvasa_audio_npz(path, model, *, pad_mode="zero"):
    payload = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in mu.tree_flatten(model.parameters())
    }
    payload.update(_metadata_payload(model, pad_mode=pad_mode))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_mlx_joyvasa_audio_npz(path, *, dtype=mx.float32, strict=True):
    state = mx.load(str(path))
    config, feature_dim, fps, n_motions, pad_mode = _config_from_loaded_state(state)
    model = MlxJoyVASAAudioFeatureExtractor(
        config,
        feature_dim=feature_dim,
        fps=fps,
        n_motions=n_motions,
    )
    params = {
        key: value.astype(dtype)
        for key, value in state.items()
        if not key.startswith(_METADATA_PREFIX)
    }

    expected_keys = {key for key, _ in mu.tree_flatten(model.parameters())}
    if strict:
        missing = expected_keys - set(params)
        unexpected = set(params) - expected_keys
        if missing or unexpected:
            raise ValueError(
                "JoyVASA MLX audio state_dict mismatch.\n"
                f"  missing in checkpoint: {sorted(missing)[:8]}\n"
                f"  unexpected in checkpoint: {sorted(unexpected)[:8]}"
            )

    model.update(mu.tree_unflatten(list(params.items())))
    model.pad_mode = pad_mode
    mx.eval(model.parameters())
    return model


def export_mlx_joyvasa_audio_from_pytorch_checkpoint(
    checkpoint_path,
    audio_encoder_path,
    out_path,
):
    import torch
    from transformers import HubertConfig

    from .JoyVASA.helper import NullableArgs
    from .JoyVASA.hubert import HubertModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = NullableArgs(checkpoint["args"])
    audio_model = getattr(args, "audio_model", "")
    if audio_model not in ("hubert", "hubert_zh", "hubert_zh_ori"):
        raise NotImplementedError(f"JoyVASA MLX audio export only supports HuBERT models, got {audio_model!r}")

    hubert_config = HubertConfig.from_pretrained(audio_encoder_path)
    hubert_config._attn_implementation = "eager"
    torch_hubert = HubertModel.from_pretrained(audio_encoder_path, attn_implementation="eager").eval()
    model = MlxJoyVASAAudioFeatureExtractor(
        MlxHubertConfig.from_dict(hubert_config.to_dict()),
        feature_dim=int(getattr(args, "feature_dim")),
        fps=int(getattr(args, "fps")),
        n_motions=int(getattr(args, "n_motions")),
    )
    model.load_pytorch_state_dicts(torch_hubert.state_dict(), checkpoint["model"])
    save_mlx_joyvasa_audio_npz(out_path, model, pad_mode=getattr(args, "pad_mode", "zero"))


def _metadata_payload(model, *, pad_mode):
    config = model.hubert.config
    if config.feat_extract_norm not in _FEAT_EXTRACT_NORM_TO_ID:
        raise ValueError(f"Unsupported HuBERT feature extract norm: {config.feat_extract_norm}")
    if pad_mode not in _PAD_MODE_TO_ID:
        raise ValueError(f"Unsupported JoyVASA pad mode: {pad_mode}")

    return {
        _metadata_key("format_version"): np.array(1, dtype=np.int32),
        _metadata_key("feature_dim"): np.array(model.audio_feature_map.weight.shape[0], dtype=np.int32),
        _metadata_key("fps"): np.array(model.fps, dtype=np.int32),
        _metadata_key("n_motions"): np.array(model.n_motions, dtype=np.int32),
        _metadata_key("pad_mode"): np.array(_PAD_MODE_TO_ID[pad_mode], dtype=np.int32),
        _metadata_key("conv_dim"): np.asarray(config.conv_dim, dtype=np.int32),
        _metadata_key("conv_kernel"): np.asarray(config.conv_kernel, dtype=np.int32),
        _metadata_key("conv_stride"): np.asarray(config.conv_stride, dtype=np.int32),
        _metadata_key("conv_bias"): np.array(config.conv_bias, dtype=np.int32),
        _metadata_key("feat_extract_norm"): np.array(_FEAT_EXTRACT_NORM_TO_ID[config.feat_extract_norm], dtype=np.int32),
        _metadata_key("feat_proj_layer_norm"): np.array(config.feat_proj_layer_norm, dtype=np.int32),
        _metadata_key("layer_norm_eps"): np.array(config.layer_norm_eps, dtype=np.float32),
        _metadata_key("hidden_size"): np.array(config.hidden_size, dtype=np.int32),
        _metadata_key("intermediate_size"): np.array(config.intermediate_size, dtype=np.int32),
        _metadata_key("num_attention_heads"): np.array(config.num_attention_heads, dtype=np.int32),
        _metadata_key("num_hidden_layers"): np.array(config.num_hidden_layers, dtype=np.int32),
        _metadata_key("num_conv_pos_embeddings"): np.array(config.num_conv_pos_embeddings, dtype=np.int32),
        _metadata_key("num_conv_pos_embedding_groups"): np.array(config.num_conv_pos_embedding_groups, dtype=np.int32),
    }


def _config_from_loaded_state(state):
    format_version = _metadata_int(state, "format_version")
    if format_version != 1:
        raise ValueError(f"Unsupported JoyVASA MLX audio weight format version: {format_version}")

    config = MlxHubertConfig(
        conv_dim=tuple(_metadata_array(state, "conv_dim").astype(np.int32).tolist()),
        conv_kernel=tuple(_metadata_array(state, "conv_kernel").astype(np.int32).tolist()),
        conv_stride=tuple(_metadata_array(state, "conv_stride").astype(np.int32).tolist()),
        conv_bias=bool(_metadata_int(state, "conv_bias")),
        feat_extract_norm=_ID_TO_FEAT_EXTRACT_NORM[_metadata_int(state, "feat_extract_norm")],
        feat_proj_layer_norm=bool(_metadata_int(state, "feat_proj_layer_norm")),
        layer_norm_eps=float(np.asarray(state[_metadata_key("layer_norm_eps")]).item()),
        hidden_size=_metadata_int(state, "hidden_size"),
        intermediate_size=_metadata_int(state, "intermediate_size"),
        num_attention_heads=_metadata_int(state, "num_attention_heads"),
        num_hidden_layers=_metadata_int(state, "num_hidden_layers"),
        num_conv_pos_embeddings=_metadata_int(state, "num_conv_pos_embeddings"),
        num_conv_pos_embedding_groups=_metadata_int(state, "num_conv_pos_embedding_groups"),
    )
    return (
        config,
        _metadata_int(state, "feature_dim"),
        _metadata_int(state, "fps"),
        _metadata_int(state, "n_motions"),
        _ID_TO_PAD_MODE[_metadata_int(state, "pad_mode")],
    )


def _metadata_key(name):
    return f"{_METADATA_PREFIX}{name}"


def _metadata_int(state, name):
    return int(np.asarray(state[_metadata_key(name)]).item())


def _metadata_array(state, name):
    return np.asarray(state[_metadata_key(name)])
