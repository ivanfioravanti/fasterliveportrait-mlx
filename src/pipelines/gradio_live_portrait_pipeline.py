# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
import gradio as gr
import cv2
import datetime
import gc
import os
import time
from tqdm import tqdm
import pickle
import numpy as np
import mlx.core as mx
from .faster_live_portrait_pipeline import FasterLivePortraitPipeline
from .joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline
from .mlx_audio_tts import MLX_AUDIO_KOKORO_MODEL, MLXAudioTextToSpeech
from ..utils.utils import video_has_audio
from ..utils.utils import resize_to_limit, prepare_paste_back, transform_keypoint
from ..utils.crop import crop_image, paste_back_numpy, _transform_pts
from ..utils.ffmpeg_utils import run_ffmpeg
from ..utils.tqdm_utils import configure_tqdm_single_process
from src.utils import utils
from src.utils.mlx_profiles import MLX_PROFILE_CHOICES, apply_mlx_profile
import platform

configure_tqdm_single_process()

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"


def mux_audio(video_path, audio_path, output_path, duration=None, fps=None):
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-b:v",
        "10M",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
    ]
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    if fps is not None:
        cmd.extend(["-r", str(fps)])
    cmd.append(output_path)
    run_ffmpeg(cmd)


def update_progress(progress, value, desc):
    if progress is not None:
        if value is None:
            value = 0
        progress(value, desc=desc)


def update_frame_progress(progress, frame_index, total_frames, desc_prefix):
    if progress is None or total_frames <= 0:
        return
    if frame_index == 0 or (frame_index + 1) % max(1, total_frames // 100) == 0 or frame_index + 1 == total_frames:
        progress((frame_index + 1, total_frames), desc=f"{desc_prefix} {frame_index + 1}/{total_frames}")


class GradioLivePortraitPipeline(FasterLivePortraitPipeline):
    _ANIMAL_EYE_TARGET_MAX = 0.6
    _ANIMAL_LIP_TARGET_SCALE = 0.375
    _ANIMAL_LIP_TARGET_MAX = 0.30
    _MLX_BUFFER_RELEASE_INTERVAL = 96

    def __init__(self, cfg, **kwargs):
        super(GradioLivePortraitPipeline, self).__init__(cfg, **kwargs)
        self.joyvasa_pipe = None
        self.text_to_speech = None
        self._realtime_settings_key = None
        self._realtime_frame_index = 0
        self._realtime_last_warning = 0.0
        self._realtime_prepare_retry_key = None
        self._realtime_last_prepare_attempt = 0.0
        self.mlx_profile = "quality"

    def set_mlx_profile(self, mlx_profile: str | None):
        mlx_profile = mlx_profile or "quality"
        if mlx_profile not in MLX_PROFILE_CHOICES:
            mlx_profile = "quality"
        # Idempotent on the hot path: realtime webcam calls this per frame.
        # apply_mlx_profile() writes os.environ, and macOS setenv() is not
        # thread-safe — racing with getenv() in gradio's upload-handler
        # thread corrupts the environ array and segfaults the process.
        if mlx_profile == self.mlx_profile and getattr(self, "model_dict", None):
            return None
        settings = apply_mlx_profile(mlx_profile)
        self.mlx_profile = mlx_profile

        warping_spade = self.model_dict.get("warping_spade") if hasattr(self, "model_dict") else None
        if warping_spade is not None:
            interval = int(settings.get("FLP_MLX_TEMPORAL_WARP_INTERVAL",
                                        os.environ.get("FLP_MLX_TEMPORAL_WARP_INTERVAL", "1")))
            threshold = float(settings.get("FLP_MLX_TEMPORAL_WARP_THRESHOLD",
                                           os.environ.get("FLP_MLX_TEMPORAL_WARP_THRESHOLD", "0")))
            if hasattr(warping_spade, "_temporal_warp_interval"):
                warping_spade._temporal_warp_interval = max(1, interval)
            if hasattr(warping_spade, "_temporal_warp_threshold"):
                warping_spade._temporal_warp_threshold = threshold
            if hasattr(warping_spade, "_fused_uint8"):
                warping_spade._fused_uint8 = os.environ.get("FLP_MLX_FUSED_UINT8", "1") == "1"
            if hasattr(warping_spade, "reset_temporal_cache"):
                warping_spade.reset_temporal_cache()
        return settings

    def _release_mlx_memory(self):
        """Release MLX device buffers and drop references to cached mx arrays held
        inside models (especially warping_spade). This must be called around heavy
        inference paths because Apple unified memory pressure after only 2-3 runs
        surfaces as native segfaults instead of clean OOM (the root cause of the
        web-ui server dying after a few Generate/Retarget usages).

        The prior mitigation (gc+clear only inside execute_video) was incomplete.
        """
        # Drop pinned mx.array refs from warping_spade caches so they can be freed.
        ws = None
        if hasattr(self, "model_dict"):
            ws = self.model_dict.get("warping_spade")
        if ws is not None:
            for attr in (
                "_feature_cache_key",
                "_feature_cache_value",
                "_kp_source_cache_key",
                "_kp_source_cache_value",
                "_temporal_warp_value",
                "_temporal_warp_prev_kp_np",
            ):
                if hasattr(ws, attr):
                    setattr(ws, attr, None)
            if hasattr(ws, "reset_temporal_cache"):
                try:
                    ws.reset_temporal_cache()
                except Exception:
                    pass

        # JoyVASA models (if loaded) - drop any large internal state they may hold.
        jpipe = getattr(self, "joyvasa_pipe", None)
        if jpipe is not None:
            for mname in ("motion_model", "audio_model", "model"):
                m = getattr(jpipe, mname, None)
                if m is not None:
                    # Best-effort: clear common cache attrs if present on the MLX model wrappers.
                    for attr in ("_cache", "_feature_cache", "_model_cache"):
                        if hasattr(m, attr):
                            setattr(m, attr, None)

        self._release_mlx_buffers()

    def _release_mlx_buffers(self):
        """Drop unattached MLX pool memory without clearing live model caches."""
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    @classmethod
    def _calibrate_animal_retarget_ratios(cls, input_eye_ratio: float, input_lip_ratio: float) -> tuple[float, float]:
        eye_ratio = float(np.clip(float(input_eye_ratio), 0.0, cls._ANIMAL_EYE_TARGET_MAX))
        lip_ratio = float(np.clip(float(input_lip_ratio), 0.0, 1.0) * cls._ANIMAL_LIP_TARGET_SCALE)
        lip_ratio = float(np.clip(lip_ratio, 0.0, cls._ANIMAL_LIP_TARGET_MAX))
        return eye_ratio, lip_ratio

    def _ensure_source_prepared(self, source_path, **kwargs):
        should_prepare = (
            self.source_path != source_path
            or kwargs.get("update_ret", False)
            or not self.src_imgs
            or not self.src_infos
        )
        if not should_prepare:
            return

        self.init_vars(**kwargs)
        ret = self.prepare_source(source_path, **kwargs)
        if not ret or not self.src_imgs or not self.src_infos:
            reason = getattr(self, "prepare_source_error", None) or "Error in processing source."
            raise gr.Error(f"{reason} Source: {source_path}", duration=5)

    @staticmethod
    def _video_outputs(video_path, video_path_concat):
        return (
            gr.update(visible=True, value=video_path),
            gr.update(visible=True, value=video_path_concat),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=""),
        )

    @staticmethod
    def _image_outputs(image_path, image_path_concat):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_concat = cv2.imread(image_path_concat, cv2.IMREAD_COLOR)
        if image is not None:
            image_path = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_concat is not None:
            image_path_concat = cv2.cvtColor(image_concat, cv2.COLOR_BGR2RGB)
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=True, value=image_path),
            gr.update(visible=True, value=image_path_concat),
            gr.update(visible=False, value=""),
        )

    def _require_joyvasa_assets(self):
        required_paths = [
            self.cfg.joyvasa_models.motion_mlx_model_path,
            self.cfg.joyvasa_models.audio_mlx_model_path,
            self.cfg.joyvasa_models.motion_template_path,
        ]
        missing = [path for path in required_paths if not os.path.exists(path)]
        if missing:
            raise gr.Error(
                "JoyVASA audio driving assets are not installed. Missing: "
                + ", ".join(missing)
                + ". Text driving uses MLX-audio for TTS, but still needs JoyVASA to convert audio into motion. "
                + "The default startup downloads these from the Hugging Face Hub cache. "
                + "If you set FLIP_CHECKPOINT_DIR, prefetch them with: "
                + "`uv run python scripts/download_mlx_weights.py --skip-mlx-weights "
                + "--include-joyvasa --checkpoints-dir $FLIP_CHECKPOINT_DIR`.",
                duration=8,
            )

    @staticmethod
    def _has_value(value):
        return value is not None and str(value) not in ("", "None")

    @classmethod
    def _select_source_path(cls, input_source_image_path, input_source_video_path, source_mode):
        selected_source_mode = source_mode if source_mode in ("Image", "Video") else None
        if selected_source_mode is None:
            selected_source_mode = "Video" if cls._has_value(input_source_video_path) else "Image"
        return input_source_video_path if selected_source_mode == "Video" else input_source_image_path

    @staticmethod
    def _realtime_key(source_path, flag_is_animal, args_user, mlx_profile):
        return (
            str(source_path),
            bool(flag_is_animal),
            str(mlx_profile or "quality"),
            tuple((key, str(args_user[key])) for key in sorted(args_user)),
        )

    def execute_realtime_frame(
            self,
            webcam_frame=None,
            input_source_image_path=None,
            input_source_video_path=None,
            source_mode=None,
            driving_mode=None,
            flag_relative_input=True,
            flag_do_crop_input=True,
            flag_remap_input=True,
            driving_multiplier=1.0,
            flag_stitching=True,
            flag_crop_driving_video_input=True,
            flag_video_editing_head_rotation=False,
            flag_is_animal=False,
            animation_region="all",
            scale=2.3,
            vx_ratio=0.0,
            vy_ratio=-0.125,
            scale_crop_driving_video=2.2,
            vx_ratio_crop_driving_video=0.0,
            vy_ratio_crop_driving_video=-0.1,
            driving_smooth_observation_variance=1e-7,
            cfg_scale=4.0,
            mlx_profile="quality",
    ):
        if webcam_frame is None:
            return None
        if driving_mode is not None and driving_mode != "Webcam":
            return gr.update()

        input_source_path = self._select_source_path(
            input_source_image_path,
            input_source_video_path,
            source_mode,
        )
        if not self._has_value(input_source_path):
            now = time.time()
            if now - self._realtime_last_warning > 3.0:
                gr.Warning("Waiting for the source image/video to finish loading.", duration=2)
                self._realtime_last_warning = now
            return gr.update()

        args_user = {
            'source': input_source_path,
            'driving': '0',
            'flag_relative_motion': flag_relative_input,
            'flag_do_crop': flag_do_crop_input,
            'flag_pasteback': flag_remap_input,
            'driving_multiplier': driving_multiplier,
            'flag_stitching': flag_stitching,
            'flag_crop_driving_video': flag_crop_driving_video_input,
            'flag_video_editing_head_rotation': flag_video_editing_head_rotation,
            'src_scale': scale,
            'src_vx_ratio': vx_ratio,
            'src_vy_ratio': vy_ratio,
            'dri_scale': scale_crop_driving_video,
            'dri_vx_ratio': vx_ratio_crop_driving_video,
            'dri_vy_ratio': vy_ratio_crop_driving_video,
            'driving_smooth_observation_variance': driving_smooth_observation_variance,
            'animation_region': animation_region,
            'cfg_scale': cfg_scale,
        }
        self.set_mlx_profile(mlx_profile)
        settings_key = self._realtime_key(input_source_path, flag_is_animal, args_user, mlx_profile)
        settings_changed = settings_key != self._realtime_settings_key

        if flag_is_animal != self.is_animal:
            self.init_models(is_animal=flag_is_animal)
            settings_changed = True

        update_ret = False
        if settings_changed:
            update_ret = self.update_cfg(args_user)
            self._realtime_frame_index = 0

        now = time.time()
        if (
                self._realtime_prepare_retry_key == settings_key
                and now - self._realtime_last_prepare_attempt < 0.75
        ):
            return gr.update()

        try:
            self._realtime_last_prepare_attempt = now
            self._ensure_source_prepared(
                str(input_source_path),
                update_ret=update_ret or settings_changed,
                realtime=True,
            )
        except gr.Error as exc:
            now = time.time()
            self._realtime_prepare_retry_key = settings_key
            if now - self._realtime_last_warning > 3.0:
                gr.Warning(str(exc), duration=2)
                self._realtime_last_warning = now
            return gr.update()

        self._realtime_prepare_retry_key = None
        if settings_changed:
            self._realtime_settings_key = settings_key
            self._release_mlx_memory()

        frame_rgb = np.asarray(webcam_frame)
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
            return None
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        if frame_rgb.shape[2] > 3:
            frame_rgb = frame_rgb[:, :, :3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        first_frame = self._realtime_frame_index == 0
        source_index = min(self._realtime_frame_index, len(self.src_imgs) - 1) if self.is_source_video else 0
        out_crop = self.run(
            frame_bgr,
            self.src_imgs[source_index],
            self.src_infos[source_index],
            first_frame=first_frame,
            realtime=True,
        )[1]
        self._realtime_frame_index += 1

        if out_crop is None:
            now = time.time()
            if now - self._realtime_last_warning > 3.0:
                gr.Warning("No face detected in the webcam frame.", duration=2)
                self._realtime_last_warning = now
            return None

        return gr.update(visible=True, value=out_crop)

    def execute_video(
            self,
            input_source_image_path=None,
            input_source_video_path=None,
            input_driving_video_path=None,
            input_driving_image_path=None,
            input_driving_pickle_path=None,
            input_driving_audio_path=None,
            input_driving_text=None,
            source_mode=None,
            driving_mode=None,
            flag_relative_input=True,
            flag_do_crop_input=True,
            flag_remap_input=True,
            driving_multiplier=1.0,
            flag_stitching=True,
            flag_crop_driving_video_input=True,
            flag_video_editing_head_rotation=False,
            flag_is_animal=False,
            animation_region="all",
            scale=2.3,
            vx_ratio=0.0,
            vy_ratio=-0.125,
            scale_crop_driving_video=2.2,
            vx_ratio_crop_driving_video=0.0,
            vy_ratio_crop_driving_video=-0.1,
            driving_smooth_observation_variance=1e-7,
            cfg_scale=4.0,
            voice_name='af',
            mlx_profile="quality",
            progress=gr.Progress(),
    ):
        """ for video driven potrait animation
        """
        self._release_mlx_memory()
        try:
            return self._execute_video_impl(
                input_source_image_path=input_source_image_path,
                input_source_video_path=input_source_video_path,
                input_driving_video_path=input_driving_video_path,
                input_driving_image_path=input_driving_image_path,
                input_driving_pickle_path=input_driving_pickle_path,
                input_driving_audio_path=input_driving_audio_path,
                input_driving_text=input_driving_text,
                source_mode=source_mode,
                driving_mode=driving_mode,
                flag_relative_input=flag_relative_input,
                flag_do_crop_input=flag_do_crop_input,
                flag_remap_input=flag_remap_input,
                driving_multiplier=driving_multiplier,
                flag_stitching=flag_stitching,
                flag_crop_driving_video_input=flag_crop_driving_video_input,
                flag_video_editing_head_rotation=flag_video_editing_head_rotation,
                flag_is_animal=flag_is_animal,
                animation_region=animation_region,
                scale=scale,
                vx_ratio=vx_ratio,
                vy_ratio=vy_ratio,
                scale_crop_driving_video=scale_crop_driving_video,
                vx_ratio_crop_driving_video=vx_ratio_crop_driving_video,
                vy_ratio_crop_driving_video=vy_ratio_crop_driving_video,
                driving_smooth_observation_variance=driving_smooth_observation_variance,
                cfg_scale=cfg_scale,
                voice_name=voice_name,
                mlx_profile=mlx_profile,
                progress=progress,
            )
        finally:
            self._release_mlx_memory()

    def _execute_video_impl(
            self,
            input_source_image_path=None,
            input_source_video_path=None,
            input_driving_video_path=None,
            input_driving_image_path=None,
            input_driving_pickle_path=None,
            input_driving_audio_path=None,
            input_driving_text=None,
            source_mode=None,
            driving_mode=None,
            flag_relative_input=True,
            flag_do_crop_input=True,
            flag_remap_input=True,
            driving_multiplier=1.0,
            flag_stitching=True,
            flag_crop_driving_video_input=True,
            flag_video_editing_head_rotation=False,
            flag_is_animal=False,
            animation_region="all",
            scale=2.3,
            vx_ratio=0.0,
            vy_ratio=-0.125,
            scale_crop_driving_video=2.2,
            vx_ratio_crop_driving_video=0.0,
            vy_ratio_crop_driving_video=-0.1,
            driving_smooth_observation_variance=1e-7,
            cfg_scale=4.0,
            voice_name='af',
            mlx_profile="quality",
            progress=gr.Progress(),
    ):
        input_source_path = self._select_source_path(input_source_image_path, input_source_video_path, source_mode)
        self.set_mlx_profile(mlx_profile)

        driving_values = {
            "Video": input_driving_video_path,
            "Image": input_driving_image_path,
            "Pickle": input_driving_pickle_path,
            "Audio": input_driving_audio_path,
            "Text": input_driving_text,
            "Webcam": "0",
        }
        selected_driving_mode = driving_mode if driving_mode in driving_values else None
        if selected_driving_mode is None:
            for mode, value in (
                ("Video", input_driving_video_path),
                ("Image", input_driving_image_path),
                ("Pickle", input_driving_pickle_path),
                ("Audio", input_driving_audio_path),
            ):
                if self._has_value(value):
                    selected_driving_mode = mode
                    break
            if selected_driving_mode is None and self._has_value(input_driving_text):
                selected_driving_mode = "Text"

        input_driving_path = driving_values[selected_driving_mode] if selected_driving_mode else None
        if selected_driving_mode == "Webcam":
            raise gr.Error(
                "Webcam driving streams live from the webcam panel. Use the browser camera controls instead of Generate.",
                duration=5,
            )
        if selected_driving_mode != "Text" and self._has_value(input_driving_path):
            input_driving_path = str(input_driving_path)

        if flag_is_animal != self.is_animal:
            self.init_models(is_animal=flag_is_animal)

        if input_source_path and input_driving_path:
            args_user = {
                'source': input_source_path,
                'driving': input_driving_path,
                'flag_relative_motion': flag_relative_input,
                'flag_do_crop': flag_do_crop_input,
                'flag_pasteback': flag_remap_input,
                'driving_multiplier': driving_multiplier,
                'flag_stitching': flag_stitching,
                'flag_crop_driving_video': flag_crop_driving_video_input,
                'flag_video_editing_head_rotation': flag_video_editing_head_rotation,
                'src_scale': scale,
                'src_vx_ratio': vx_ratio,
                'src_vy_ratio': vy_ratio,
                'dri_scale': scale_crop_driving_video,
                'dri_vx_ratio': vx_ratio_crop_driving_video,
                'dri_vy_ratio': vy_ratio_crop_driving_video,
                'driving_smooth_observation_variance': driving_smooth_observation_variance,
                'animation_region': animation_region,
                'cfg_scale': cfg_scale
            }
            # update config from user input
            update_ret = self.update_cfg(args_user)
            update_progress(progress, None, "Preparing source")
            if selected_driving_mode == 'Video':
                # video driven animation
                video_path, video_path_concat, total_time = self.run_video_driving(input_driving_path,
                                                                                   input_source_path,
                                                                                   update_ret=update_ret,
                                                                                   progress=progress)
                update_progress(progress, 1.0, "Ready")
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return self._video_outputs(video_path, video_path_concat)
            elif selected_driving_mode == 'Pickle':
                # pickle driven animation
                video_path, video_path_concat, total_time = self.run_pickle_driving(input_driving_path,
                                                                                    input_source_path,
                                                                                    update_ret=update_ret,
                                                                                    progress=progress)
                update_progress(progress, 1.0, "Ready")
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return self._video_outputs(video_path, video_path_concat)
            elif selected_driving_mode == 'Audio':
                # audio driven animation
                video_path, video_path_concat, total_time = self.run_audio_driving(input_driving_path,
                                                                                   input_source_path,
                                                                                   update_ret=update_ret,
                                                                                   progress=progress)
                update_progress(progress, 1.0, "Ready")
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return self._video_outputs(video_path, video_path_concat)
            elif selected_driving_mode == 'Text':
                # Text driven animation
                video_path, video_path_concat, total_time = self.run_text_driving(input_driving_path,
                                                                                  voice_name,
                                                                                  input_source_path,
                                                                                  update_ret=update_ret,
                                                                                  progress=progress)
                update_progress(progress, 1.0, "Ready")
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return self._video_outputs(video_path, video_path_concat)
            else:
                # video driven animation
                image_path, image_path_concat, total_time = self.run_image_driving(input_driving_path,
                                                                                   input_source_path,
                                                                                   update_ret=update_ret,
                                                                                   progress=progress)
                update_progress(progress, 1.0, "Ready")
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return self._image_outputs(image_path, image_path_concat)
        else:
            raise gr.Error("The input source portrait or driving video hasn't been prepared yet 💥!", duration=5)

    def run_image_driving(self, driving_image_path, source_path, **kwargs):
        progress = kwargs.get("progress")
        update_progress(progress, 0, "Preparing source")
        self._ensure_source_prepared(source_path, **kwargs)
        update_progress(progress, 0.25, "Reading driving image")

        driving_image = cv2.imread(driving_image_path)
        save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

        image_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(source_path)}-{os.path.basename(driving_image_path)}-crop.jpg")
        image_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(source_path)}-{os.path.basename(driving_image_path)}-org.jpg")

        t0 = time.time()
        update_progress(progress, 0.5, "Rendering image")
        dri_crop, out_crop, out_org = self.run(driving_image, self.src_imgs[0], self.src_infos[0],
                                               first_frame=True)[:3]

        update_progress(progress, 0.8, "Writing image results")
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_crop_path, out_crop)
        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_org_path, out_org)
        total_time = time.time() - t0

        return image_org_path, image_crop_path, total_time

    def run_video_driving(self, driving_video_path, source_path, **kwargs):
        t00 = time.time()
        progress = kwargs.get("progress")

        update_progress(progress, 0, "Preparing source")
        self._ensure_source_prepared(source_path, **kwargs)
        update_progress(progress, 0, "Opening driving video")

        vcap = cv2.VideoCapture(driving_video_path)
        if self.is_source_video:
            duration, fps = utils.get_video_info(self.source_path)
            fps = int(fps)
        else:
            fps = int(vcap.get(cv2.CAP_PROP_FPS))

        dframe = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.is_source_video:
            max_frame = min(dframe, len(self.src_imgs))
        else:
            max_frame = dframe
        h, w = self.src_imgs[0].shape[:2]
        save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

        # render output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(source_path)}-{os.path.basename(driving_video_path)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(source_path)}-{os.path.basename(driving_video_path)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

        infer_times = []
        print(f"render {max_frame} frames to {vsave_crop_path} and {vsave_org_path}", flush=True)
        frame_iter = range(max_frame)
        if progress is None:
            frame_iter = tqdm(frame_iter, desc="Rendering frames", unit="frame")
        for i in frame_iter:
            ret, frame = vcap.read()
            if not ret:
                break
            t0 = time.time()
            first_frame = i == 0
            if self.is_source_video:
                dri_crop, out_crop, out_org = self.run(frame, self.src_imgs[i], self.src_infos[i],
                                                       first_frame=first_frame)[:3]
            else:
                dri_crop, out_crop, out_org = self.run(frame, self.src_imgs[0], self.src_infos[0],
                                                       first_frame=first_frame)[:3]
            if out_crop is None:
                print(f"no face in driving frame:{i}")
                continue
            infer_times.append(time.time() - t0)
            dri_crop = cv2.resize(dri_crop, (512, 512))
            out_crop = np.concatenate([dri_crop, out_crop], axis=1)
            out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
            update_frame_progress(progress, i, max_frame, "Rendering frames")
            if progress is not None and i > 0 and (i + 1) % self._MLX_BUFFER_RELEASE_INTERVAL == 0:
                self._release_mlx_buffers()
        update_progress(progress, 0.98, "Finalizing video files")
        print("finalizing video writers", flush=True)
        vcap.release()
        vout_crop.release()
        vout_org.release()

        if video_has_audio(driving_video_path):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            update_progress(progress, 0.98, "Muxing audio into crop video")
            print(f"mux audio: {vsave_crop_path_new}", flush=True)
            if self.is_source_video:
                duration, fps = utils.get_video_info(vsave_crop_path)
                mux_audio(vsave_crop_path, driving_video_path, vsave_crop_path_new, duration=duration, fps=fps)
                update_progress(progress, 0.99, "Muxing audio into pasteback video")
                print(f"mux audio: {vsave_org_path_new}", flush=True)
                mux_audio(vsave_org_path, driving_video_path, vsave_org_path_new, duration=duration, fps=fps)
            else:
                mux_audio(vsave_crop_path, driving_video_path, vsave_crop_path_new)
                update_progress(progress, 0.99, "Muxing audio into pasteback video")
                print(f"mux audio: {vsave_org_path_new}", flush=True)
                mux_audio(vsave_org_path, driving_video_path, vsave_org_path_new)

            return vsave_org_path_new, vsave_crop_path_new, time.time() - t00
        else:
            return vsave_org_path, vsave_crop_path, time.time() - t00

    def run_pickle_driving(self, driving_pickle_path, source_path, **kwargs):
        t00 = time.time()
        progress = kwargs.get("progress")

        update_progress(progress, 0, "Preparing source")
        self._ensure_source_prepared(source_path, **kwargs)
        update_progress(progress, 0, "Loading driving motion")

        with open(driving_pickle_path, "rb") as fin:
            dri_motion_infos = pickle.load(fin)

        if self.is_source_video:
            duration, fps = utils.get_video_info(self.source_path)
            fps = int(fps)
        else:
            fps = int(dri_motion_infos["output_fps"])

        motion_lst = dri_motion_infos["motion"]
        c_eyes_lst = dri_motion_infos["c_eyes_lst"] if "c_eyes_lst" in dri_motion_infos else dri_motion_infos[
            "c_d_eyes_lst"]
        c_lip_lst = dri_motion_infos["c_lip_lst"] if "c_lip_lst" in dri_motion_infos else dri_motion_infos[
            "c_d_lip_lst"]
        dframe = len(motion_lst)

        if self.is_source_video:
            max_frame = min(dframe, len(self.src_imgs))
        else:
            max_frame = dframe
        h, w = self.src_imgs[0].shape[:2]
        save_dir = kwargs.get("save_dir", f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)

        # render output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(source_path)}-{os.path.basename(driving_pickle_path)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(source_path)}-{os.path.basename(driving_pickle_path)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

        infer_times = []
        print(f"render {max_frame} pickle frames to {vsave_crop_path} and {vsave_org_path}", flush=True)
        frame_iter = range(max_frame)
        if progress is None:
            frame_iter = tqdm(frame_iter, desc="Rendering pickle frames", unit="frame")
        for frame_ind in frame_iter:
            t0 = time.time()
            first_frame = frame_ind == 0
            dri_motion_info_ = [motion_lst[frame_ind]]
            if c_eyes_lst:
                dri_motion_info_.append(c_eyes_lst[frame_ind])
            else:
                dri_motion_info_.append(None)
            if c_lip_lst:
                dri_motion_info_.append(c_lip_lst[frame_ind])
            else:
                dri_motion_info_.append(None)
            if self.is_source_video:
                out_crop, out_org = self.run_with_pkl(dri_motion_info_, self.src_imgs[frame_ind],
                                                      self.src_infos[frame_ind],
                                                      first_frame=first_frame)[:3]
            else:
                out_crop, out_org = self.run_with_pkl(dri_motion_info_, self.src_imgs[0], self.src_infos[0],
                                                      first_frame=first_frame)[:3]
            if out_crop is None:
                print(f"no face in driving frame:{frame_ind}")
                continue
            infer_times.append(time.time() - t0)
            out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
            update_frame_progress(progress, frame_ind, max_frame, "Rendering pickle frames")
            if progress is not None and frame_ind > 0 and (frame_ind + 1) % self._MLX_BUFFER_RELEASE_INTERVAL == 0:
                self._release_mlx_buffers()
        update_progress(progress, 0.98, "Finalizing video files")
        total_time = time.time() - t00
        vout_crop.release()
        vout_org.release()

        return vsave_org_path, vsave_crop_path, total_time

    def run_audio_driving(self, driving_audio_path, source_path, **kwargs):
        t00 = time.time()
        progress = kwargs.get("progress")

        update_progress(progress, 0, "Preparing source")
        self._ensure_source_prepared(source_path, **kwargs)
        save_dir = kwargs.get("save_dir", f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)

        if self.joyvasa_pipe is None:
            self._require_joyvasa_assets()
            self.joyvasa_pipe = JoyVASAAudio2MotionPipeline(motion_mlx_model_path=self.cfg.joyvasa_models.motion_mlx_model_path,
                                                            audio_mlx_model_path=self.cfg.joyvasa_models.audio_mlx_model_path,
                                                            motion_template_path=self.cfg.joyvasa_models.motion_template_path,
                                                            cfg_mode=self.cfg.infer_params.cfg_mode,
                                                            cfg_scale=self.cfg.infer_params.cfg_scale
                                                            )
        t01 = time.time()
        update_progress(progress, 0, "Generating audio motion")
        dri_motion_infos = self.joyvasa_pipe.gen_motion_sequence(driving_audio_path)
        gr.Info(f"JoyVASA cost time:{time.time() - t01}", duration=2)
        motion_pickle_path = os.path.join(save_dir,
                                          f"{os.path.basename(source_path)}-{os.path.basename(driving_audio_path)}.pkl")
        with open(motion_pickle_path, "wb") as fw:
            pickle.dump(dri_motion_infos, fw)

        vsave_org_path, vsave_crop_path, total_time = self.run_pickle_driving(motion_pickle_path, source_path,
                                                                              save_dir=save_dir,
                                                                              progress=progress)

        vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
        vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"

        duration, fps = utils.get_video_info(vsave_crop_path)
        update_progress(progress, 0.98, "Muxing audio into crop video")
        mux_audio(vsave_crop_path, driving_audio_path, vsave_crop_path_new, duration=duration, fps=fps)
        update_progress(progress, 0.99, "Muxing audio into pasteback video")
        mux_audio(vsave_org_path, driving_audio_path, vsave_org_path_new, duration=duration, fps=fps)

        return vsave_org_path_new, vsave_crop_path_new, time.time() - t00

    def run_text_driving(self, driving_text, voice_name, source_path, **kwargs):
        progress = kwargs.get("progress")
        update_progress(progress, 0, "Preparing source")
        self._ensure_source_prepared(source_path, **kwargs)
        save_dir = kwargs.get("save_dir", f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)
        tts_cfg = self.cfg.get("text_to_speech", {})
        model_id = tts_cfg.get("model_id", MLX_AUDIO_KOKORO_MODEL)
        speed = float(tts_cfg.get("speed", 1.0))
        voice_name = voice_name or tts_cfg.get("default_voice", "af_heart")
        if self.text_to_speech is None or self.text_to_speech.model_id != model_id or self.text_to_speech.speed != speed:
            self.text_to_speech = MLXAudioTextToSpeech(model_id=model_id, speed=speed)

        audio_save_path = os.path.join(save_dir, f"mlx-audio-{voice_name}.wav")
        try:
            update_progress(progress, 0, "Synthesizing driving audio")
            audio_save_path, sample_rate = self.text_to_speech.synthesize_to_file(
                driving_text,
                voice_name,
                audio_save_path,
            )
        except RuntimeError as exc:
            raise gr.Error(f"MLX-audio text driving failed: {exc}", duration=5) from exc
        print(f"save audio to: {audio_save_path} ({sample_rate} Hz)")
        vsave_org_path, vsave_crop_path, total_time = self.run_audio_driving(audio_save_path, source_path,
                                                                             save_dir=save_dir,
                                                                             progress=progress)

        return vsave_org_path, vsave_crop_path, total_time

    def execute_image(self, input_eye_ratio: float, input_lip_ratio: float, input_image, flag_do_crop=True):
        """ for single image retargeting
        """
        if input_image is None:
            gr.Warning("Upload or select a retargeting input image first.", duration=5)
            return None, None
        if input_eye_ratio is None or input_lip_ratio is None:
            gr.Warning("Set valid eye and lip retargeting ratios first.", duration=5)
            return None, None

        self._release_mlx_memory()
        try:
            # disposable feature
            f_s_user, x_s_user, source_lmk_user, crop_M_c2o, mask_ori, img_rgb = \
                self.prepare_retargeting(input_image, flag_do_crop)
            if self.is_animal:
                input_eye_ratio, input_lip_ratio = self._calibrate_animal_retarget_ratios(
                    input_eye_ratio,
                    input_lip_ratio,
                )

            # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
            combined_eye_ratio_tensor = self.calc_combined_eye_ratio([[input_eye_ratio]], source_lmk_user)
            eyes_delta = self.retarget_eye(x_s_user, combined_eye_ratio_tensor)
            # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
            combined_lip_ratio_tensor = self.calc_combined_lip_ratio([[input_lip_ratio]], source_lmk_user)
            lip_delta = self.retarget_lip(x_s_user, combined_lip_ratio_tensor)
            num_kp = x_s_user.shape[1]
            # default: use x_s
            x_d_new = x_s_user + eyes_delta.reshape(-1, num_kp, 3) + lip_delta.reshape(-1, num_kp, 3)
            # D(W(f_s; x_s, x′_d))
            out = self.model_dict["warping_spade"].predict(
                f_s_user,
                x_s_user,
                x_d_new,
                return_numpy=True,
                return_uint8=True,
            )
            out_to_ori_blend = paste_back_numpy(out, crop_M_c2o, img_rgb, mask_ori)
            gr.Info("Run successfully!", duration=2)
            return out, out_to_ori_blend
        finally:
            self._release_mlx_memory()

    def prepare_retargeting(self, input_image, flag_do_crop=True):
        """ for single image retargeting
        """
        if input_image is not None:
            ######## process source portrait ########
            img_bgr = cv2.imread(input_image, cv2.IMREAD_COLOR)
            img_bgr = resize_to_limit(img_bgr, self.cfg.infer_params.source_max_dim,
                                      self.cfg.infer_params.source_division)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            face_analysis = self.model_dict["face_analysis"]
            dense_src_faces = None
            if self.is_animal and hasattr(face_analysis, "predict_dense"):
                src_faces = face_analysis.predict(img_bgr)
                dense_src_faces = face_analysis.predict_dense(img_bgr)
                if len(src_faces) == 0 or len(dense_src_faces) == 0:
                    raise gr.Error("No dense animal face landmarks detected for retargeting 💥!", duration=5)
            elif self.is_animal:
                raise gr.Error("Animal face retargeting needs dense MLX landmarks 💥!", duration=5)
            else:
                src_faces = face_analysis.predict(img_bgr)

            if len(src_faces) == 0:
                raise gr.Error("No face detect in image 💥!", duration=5)
            src_faces = src_faces[:1]
            crop_infos = []
            for i in range(len(src_faces)):
                # NOTE: temporarily only pick the first face, to support multiple face in the future
                lmk = src_faces[i]
                # crop the face
                ret_dct = crop_image(
                    img_rgb,  # ndarray
                    lmk,  # 106x2 or Nx2
                    dsize=self.cfg.crop_params.src_dsize,
                    scale=self.cfg.crop_params.src_scale,
                    vx_ratio=self.cfg.crop_params.src_vx_ratio,
                    vy_ratio=self.cfg.crop_params.src_vy_ratio,
                    flag_do_rot=bool(self.cfg.infer_params.get("flag_do_rot", True)),
                    borderMode=cv2.BORDER_REPLICATE,
                )

                if self.is_animal:
                    lmk = _transform_pts(dense_src_faces[i], ret_dct["M_o2c"])
                else:
                    lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                ret_dct["lmk_crop"] = lmk
                ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize

                # update a 256x256 version for network input
                ret_dct["img_crop_256x256"] = cv2.resize(
                    ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
                )
                ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize
                crop_infos.append(ret_dct)
            crop_info = crop_infos[0]
            if flag_do_crop:
                I_s = crop_info['img_crop_256x256'].copy()
            else:
                I_s = img_rgb.copy()
            pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(I_s)
            f_s_user = self.model_dict["app_feat_extractor"].predict(I_s)
            x_s_user = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
            source_lmk_user = crop_info['lmk_crop']
            crop_M_c2o = crop_info['M_c2o']
            mask_ori = prepare_paste_back(self.mask_crop, crop_info['M_c2o'],
                                          dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            return f_s_user, x_s_user, source_lmk_user, crop_M_c2o, mask_ori, img_rgb
        else:
            # when press the clear button, go here
            raise gr.Error("The retargeting input hasn't been prepared yet 💥!", duration=5)
