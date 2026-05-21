# -*- coding: utf-8 -*-
# @Time    : 2024/12/15
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: joyvasa_audio_to_motion_pipeline.py

import math

import mlx.core as mx
import numpy as np
import pickle
from tqdm import tqdm
import os
import soundfile as sf
from scipy.signal import resample_poly

from ..models.mlx_joyvasa_audio_model import load_mlx_joyvasa_audio_npz
from ..models.mlx_joyvasa_motion_model import load_mlx_joyvasa_motion_npz
from ..utils.tqdm_utils import configure_tqdm_single_process
from ..utils import utils

configure_tqdm_single_process()


class JoyVASAAudio2MotionPipeline:
    """
    JoyVASA 声音生成LivePortrait Motion
    """

    def __init__(self, **kwargs):
        motion_mlx_model_path = kwargs.get("motion_mlx_model_path", "")
        audio_mlx_model_path = kwargs.get("audio_mlx_model_path", "")
        motion_template_path = kwargs.get("motion_template_path", "")

        if not motion_mlx_model_path:
            raise ValueError("JoyVASA runtime requires motion_mlx_model_path")
        if not audio_mlx_model_path:
            raise ValueError("JoyVASA runtime requires audio_mlx_model_path")
        if not os.path.exists(motion_mlx_model_path):
            raise FileNotFoundError(
                f"JoyVASA MLX motion weights not found: {motion_mlx_model_path}. "
                "Run `uv run python scripts/export_mlx_weights.py --include-joyvasa`."
            )
        if not os.path.exists(audio_mlx_model_path):
            raise FileNotFoundError(
                f"JoyVASA MLX audio weights not found: {audio_mlx_model_path}. "
                "Run `uv run python scripts/export_mlx_weights.py --include-joyvasa`."
            )

        self.motion_generator = load_mlx_joyvasa_motion_npz(
            motion_mlx_model_path,
            cfg_mode=kwargs.get("cfg_mode", "incremental"),
        )
        model = load_mlx_joyvasa_audio_npz(audio_mlx_model_path)
        self.motion_backend = "mlx"
        self.audio_backend = "mlx"

        self.audio_feature_extractor = model
        self.n_motions = self.motion_generator.n_motions
        self.n_prev_motions = self.motion_generator.n_prev_motions
        self.fps = self.motion_generator.fps
        if model.n_motions != self.n_motions:
            raise ValueError(
                f"JoyVASA audio/motion n_motions mismatch: {model.n_motions} != {self.n_motions}"
            )
        if model.audio_feature_map.weight.shape[0] != self.motion_generator.feature_dim:
            raise ValueError(
                "JoyVASA audio/motion feature_dim mismatch: "
                f"{model.audio_feature_map.weight.shape[0]} != {self.motion_generator.feature_dim}"
            )
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = model.pad_mode
        self.use_indicator = bool(self.motion_generator.denoising_net.use_indicator)
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

    def gen_motion_sequence(self, audio_path, **kwargs):
        # preprocess audio
        audio_np, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio_np = audio_np.mean(axis=1)
        if sample_rate != 16000:
            gcd = math.gcd(sample_rate, 16000)
            audio_np = resample_poly(audio_np, 16000 // gcd, sample_rate // gcd)
        audio = audio_np.astype(np.float32)
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
            audio = np.pad(
                audio,
                (0, n_padding_audio_samples),
                mode="constant",
                constant_values=float(padding_value),
            )

        # generate motions
        coef_list = []
        prev_motion_feat = None
        prev_audio_feat = None
        noise = None
        for i in range(0, n_subdivision):
            start_idx = i * stride
            end_idx = start_idx + self.n_motions
            indicator = np.ones((1, self.n_motions), dtype=np.float32) if self.use_indicator else None
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0
            audio_slice = audio[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)]
            audio_in = audio_slice[None, :]

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

        motion_coef = np.concatenate(coef_list, axis=1).squeeze().astype(np.float32)
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
