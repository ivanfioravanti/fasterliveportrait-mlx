# -*- coding: utf-8 -*-
# @Time    : 2024/12/15
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: joyvasa_audio_to_motion_pipeline.py

import math

import mlx.core as mx
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pickle
from tqdm import tqdm
import pathlib
import os
import soundfile as sf
from scipy.signal import resample_poly

from ..models.JoyVASA.dit_talking_head import DitTalkingHead
from ..models.JoyVASA.common import pad_audio
from ..models.JoyVASA.helper import NullableArgs
from ..models.mlx_joyvasa_motion_model import load_mlx_joyvasa_motion_npz
from ..utils import utils


class JoyVASAAudioFeatureExtractor(nn.Module):
    def __init__(self, *, audio_model, audio_encoder_path, feature_dim, fps, n_motions):
        super().__init__()
        self.audio_model = audio_model
        self.fps = fps
        self.n_motions = n_motions
        self.audio_encoder = self._load_audio_encoder(audio_model, audio_encoder_path)
        self.audio_feature_map = nn.Linear(768, feature_dim)

    def _load_audio_encoder(self, audio_model, audio_encoder_path):
        if audio_model in ("wav2vec2", "wav2vec2_ori"):
            if audio_model == "wav2vec2":
                print("using wav2vec2 audio encoder ...")
            from ..models.JoyVASA.wav2vec2 import Wav2Vec2Model
            audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path, attn_implementation="eager")
        elif audio_model in ("hubert", "hubert_zh", "hubert_zh_ori"):
            if audio_model == "hubert_zh":
                print("using hubert chinese")
            elif audio_model == "hubert_zh_ori":
                print("using hubert chinese ori")
            from ..models.JoyVASA.hubert import HubertModel
            audio_encoder = HubertModel.from_pretrained(audio_encoder_path, attn_implementation="eager")
        else:
            raise ValueError(f"Unknown audio model {audio_model}!")

        audio_encoder.feature_extractor._freeze_parameters()
        if audio_model in ("wav2vec2", "hubert", "hubert_zh"):
            frozen_layers = [0, 1]
            for name, param in audio_encoder.named_parameters():
                if name.startswith("feature_projection"):
                    param.requires_grad = False
                if name.startswith("encoder.layers"):
                    layer = int(name.split(".")[2])
                    if layer in frozen_layers:
                        param.requires_grad = False
        return audio_encoder

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        hidden_states = self.audio_encoder(
            pad_audio(audio),
            self.fps,
            frame_num=frame_num * 2,
        ).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode="linear")
        hidden_states = hidden_states.transpose(1, 2)
        return self.audio_feature_map(hidden_states)


class JoyVASAAudio2MotionPipeline:
    """
    JoyVASA 声音生成LivePortrait Motion
    """

    def __init__(self, **kwargs):
        self.device, self.dtype = utils.get_opt_device_dtype()
        # Check if the operating system is Windows
        if os.name == 'nt':
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        motion_model_path = kwargs.get("motion_model_path", "")
        motion_mlx_model_path = kwargs.get("motion_mlx_model_path", "")
        audio_model_path = kwargs.get("audio_model_path", "")
        motion_template_path = kwargs.get("motion_template_path", "")
        # JoyVASA checkpoints store argparse.Namespace metadata alongside tensors.
        # PyTorch 2.6+ defaults torch.load(weights_only=True), which rejects that
        # metadata, so this experimental path must opt into full checkpoint loading.
        model_data = torch.load(motion_model_path, map_location="cpu", weights_only=False)
        model_args = NullableArgs(model_data['args'])
        if motion_mlx_model_path:
            if not os.path.exists(motion_mlx_model_path):
                raise FileNotFoundError(
                    f"JoyVASA MLX motion weights not found: {motion_mlx_model_path}. "
                    "Run `uv run python scripts/export_mlx_weights.py --include-joyvasa`."
                )
            model = JoyVASAAudioFeatureExtractor(
                audio_model=model_args.audio_model,
                audio_encoder_path=audio_model_path,
                feature_dim=model_args.feature_dim,
                fps=model_args.fps,
                n_motions=model_args.n_motions,
            )
            audio_state = {
                key: value
                for key, value in model_data["model"].items()
                if key.startswith("audio_feature_map.")
            }
            model.load_state_dict(audio_state, strict=False)
            model.to(self.device, dtype=self.dtype)
            model.eval()
            self.motion_generator = load_mlx_joyvasa_motion_npz(
                motion_mlx_model_path,
                cfg_mode=kwargs.get("cfg_mode", "incremental"),
            )
            self.motion_backend = "mlx"
        else:
            model = DitTalkingHead(motion_feat_dim=model_args.motion_feat_dim,
                                   device=self.device,
                                   n_motions=model_args.n_motions,
                                   n_prev_motions=model_args.n_prev_motions,
                                   feature_dim=model_args.feature_dim,
                                   audio_model=model_args.audio_model,
                                   n_diff_steps=model_args.n_diff_steps,
                                   audio_encoder_path=audio_model_path)
            model_data['model'].pop('denoising_net.TE.pe', None)
            model.load_state_dict(model_data['model'], strict=False)
            model.to(self.device, dtype=self.dtype)
            model.eval()
            self.motion_generator = model
            self.motion_backend = "torch"

        # Restore the original PosixPath if it was changed
        if os.name == 'nt':
            pathlib.PosixPath = temp

        self.audio_feature_extractor = model
        self.n_motions = model_args.n_motions
        self.n_prev_motions = model_args.n_prev_motions
        self.fps = model_args.fps
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = model_args.pad_mode
        self.use_indicator = (
            bool(self.motion_generator.denoising_net.use_indicator)
            if self.motion_backend == "mlx"
            else model_args.use_indicator
        )
        self.cfg_mode = kwargs.get("cfg_mode", "incremental")
        self.cfg_cond = kwargs.get("cfg_cond", None)
        self.cfg_scale = kwargs.get("cfg_scale", 2.8)
        with open(motion_template_path, 'rb') as fin:
            self.templete_dict = pickle.load(fin)

    def _to_mx(self, value):
        if value is None:
            return None
        if isinstance(value, mx.array):
            return value
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return mx.array(np.asarray(value)).astype(mx.float32)

    def _sample_mlx_motion(
        self,
        audio_in,
        prev_motion_feat=None,
        prev_audio_feat=None,
        motion_at_T=None,
        indicator=None,
    ):
        audio_feat = self.audio_feature_extractor.extract_audio_feature(audio_in)
        return self._sample_mlx_motion_from_audio_feature(
            audio_feat,
            prev_motion_feat=prev_motion_feat,
            prev_audio_feat=prev_audio_feat,
            motion_at_T=motion_at_T,
            indicator=indicator,
        )

    def _sample_mlx_motion_from_audio_feature(
        self,
        audio_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        motion_at_T=None,
        indicator=None,
    ):
        return self.motion_generator.sample(
            self._to_mx(audio_feat),
            self._to_mx(prev_motion_feat),
            self._to_mx(prev_audio_feat),
            self._to_mx(motion_at_T),
            indicator=self._to_mx(indicator),
            cfg_mode=self.cfg_mode,
            cfg_cond=self.cfg_cond,
            cfg_scale=self.cfg_scale,
            dynamic_threshold=0,
        )

    @torch.inference_mode()
    def gen_motion_sequence(self, audio_path, **kwargs):
        # preprocess audio
        audio_np, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio_np = audio_np.mean(axis=1)
        if sample_rate != 16000:
            gcd = math.gcd(sample_rate, 16000)
            audio_np = resample_poly(audio_np, 16000 // gcd, sample_rate // gcd)
        audio = torch.from_numpy(audio_np).to(self.device, dtype=self.dtype)
        # audio = F.pad(audio, (1280, 640), "constant", 0)
        # audio_mean, audio_std = torch.mean(audio), torch.std(audio)
        # audio = (audio - audio_mean) / (audio_std + 1e-5)

        # crop audio into n_subdivision according to n_motions
        clip_len = int(len(audio) / 16000 * self.fps)
        stride = self.n_motions
        if clip_len <= self.n_motions:
            n_subdivision = 1
        else:
            n_subdivision = math.ceil(clip_len / stride)

        # padding
        n_padding_audio_samples = self.n_audio_samples * n_subdivision - len(audio)
        n_padding_frames = math.ceil(n_padding_audio_samples / self.audio_unit)
        if n_padding_audio_samples > 0:
            if self.pad_mode == 'zero':
                padding_value = 0
            elif self.pad_mode == 'replicate':
                padding_value = audio[-1]
            else:
                raise ValueError(f'Unknown pad mode: {self.pad_mode}')
            audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)

        # generate motions
        coef_list = []
        prev_motion_feat = None
        prev_audio_feat = None
        noise = None
        for i in range(0, n_subdivision):
            start_idx = i * stride
            end_idx = start_idx + self.n_motions
            indicator = torch.ones((1, self.n_motions)).to(self.device) if self.use_indicator else None
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0
            audio_in = audio[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)].unsqueeze(0)

            if self.motion_backend == "mlx":
                motion_feat, noise, prev_audio_feat = self._sample_mlx_motion(
                    audio_in,
                    prev_motion_feat=prev_motion_feat,
                    prev_audio_feat=prev_audio_feat,
                    motion_at_T=noise,
                    indicator=indicator,
                )
                prev_motion_feat = motion_feat[:, -self.n_prev_motions:]
                prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]
                motion_coef = motion_feat
                if i == n_subdivision - 1 and n_padding_frames > 0:
                    motion_coef = motion_coef[:, :-n_padding_frames]
                coef_list.append(np.asarray(motion_coef, dtype=np.float32))
            else:
                if i == 0:
                    motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,
                                                                                       indicator=indicator,
                                                                                       cfg_mode=self.cfg_mode,
                                                                                       cfg_cond=self.cfg_cond,
                                                                                       cfg_scale=self.cfg_scale,
                                                                                       dynamic_threshold=0)
                else:
                    motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,
                                                                                       prev_motion_feat.to(self.dtype),
                                                                                       prev_audio_feat.to(self.dtype),
                                                                                       noise.to(self.dtype),
                                                                                       indicator=indicator,
                                                                                       cfg_mode=self.cfg_mode,
                                                                                       cfg_cond=self.cfg_cond,
                                                                                       cfg_scale=self.cfg_scale,
                                                                                       dynamic_threshold=0)
                prev_motion_feat = motion_feat[:, -self.n_prev_motions:].clone()
                prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]

                motion_coef = motion_feat
                if i == n_subdivision - 1 and n_padding_frames > 0:
                    motion_coef = motion_coef[:, :-n_padding_frames]  # delete padded frames
                coef_list.append(motion_coef)

        if self.motion_backend == "mlx":
            motion_coef = np.concatenate(coef_list, axis=1).squeeze().astype(np.float32)
        else:
            motion_coef = torch.cat(coef_list, dim=1).squeeze().cpu().numpy().astype(np.float32)
        motion_list = []
        for idx in tqdm(range(motion_coef.shape[0]), total=motion_coef.shape[0]):
            exp = motion_coef[idx][:63] * self.templete_dict["std_exp"] + self.templete_dict["mean_exp"]
            scale = motion_coef[idx][63:64] * (
                    self.templete_dict["max_scale"] - self.templete_dict["min_scale"]) + self.templete_dict[
                        "min_scale"]
            t = motion_coef[idx][64:67] * (self.templete_dict["max_t"] - self.templete_dict["min_t"]) + \
                self.templete_dict["min_t"]
            pitch = motion_coef[idx][67:68] * (
                    self.templete_dict["max_pitch"] - self.templete_dict["min_pitch"]) + self.templete_dict[
                        "min_pitch"]
            yaw = motion_coef[idx][68:69] * (self.templete_dict["max_yaw"] - self.templete_dict["min_yaw"]) + \
                  self.templete_dict["min_yaw"]
            roll = motion_coef[idx][69:70] * (self.templete_dict["max_roll"] - self.templete_dict["min_roll"]) + \
                   self.templete_dict["min_roll"]

            R = utils.get_rotation_matrix(pitch, yaw, roll)
            R = R.reshape(1, 3, 3).astype(np.float32)

            exp = exp.reshape(1, 21, 3).astype(np.float32)
            scale = scale.reshape(1, 1).astype(np.float32)
            t = t.reshape(1, 3).astype(np.float32)
            pitch = pitch.reshape(1, 1).astype(np.float32)
            yaw = yaw.reshape(1, 1).astype(np.float32)
            roll = roll.reshape(1, 1).astype(np.float32)

            motion_list.append({"exp": exp, "scale": scale, "R": R, "t": t, "pitch": pitch, "yaw": yaw, "roll": roll})
        tgt_motion = {'n_frames': motion_coef.shape[0], 'output_fps': self.fps, 'motion': motion_list, 'c_eyes_lst': [],
                      'c_lip_lst': []}
        return tgt_motion
