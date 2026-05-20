from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


MLX_AUDIO_KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"

MLX_AUDIO_KOKORO_VOICES = (
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)

_VOICE_LANG_CODES = {
    "af": "a",
    "am": "a",
    "bf": "b",
    "bm": "b",
    "ef": "e",
    "em": "e",
    "ff": "f",
    "hf": "h",
    "hm": "h",
    "if": "i",
    "im": "i",
    "jf": "j",
    "jm": "j",
    "pf": "p",
    "pm": "p",
    "zf": "z",
    "zm": "z",
}


def kokoro_lang_code_for_voice(voice_name: str) -> str:
    return _VOICE_LANG_CODES.get(voice_name.split("_", 1)[0], "a")


class MLXAudioTextToSpeech:
    def __init__(self, model_id: str = MLX_AUDIO_KOKORO_MODEL, speed: float = 1.0):
        self.model_id = model_id
        self.speed = speed
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from mlx_audio.tts.utils import load_model
            except ImportError as exc:
                raise RuntimeError("MLX-audio is not installed. Run `uv sync` before using text driving.") from exc
            self._model = load_model(self.model_id)
        return self._model

    def synthesize_to_file(self, text: str, voice_name: str, output_path: str | Path) -> tuple[str, int]:
        model = self._load_model()
        lang_code = kokoro_lang_code_for_voice(voice_name)

        audio_segments = []
        sample_rate = int(getattr(model, "sample_rate", 24000))
        for result in model.generate(
            text=text,
            voice=voice_name,
            speed=self.speed,
            lang_code=lang_code,
            split_pattern=r"\n+",
        ):
            if result.audio is not None:
                audio_segments.append(np.asarray(result.audio))
            sample_rate = int(getattr(result, "sample_rate", sample_rate))

        if not audio_segments:
            raise RuntimeError("MLX-audio did not generate any audio.")

        audio = np.concatenate(audio_segments)
        output_path = str(output_path)
        sf.write(output_path, audio, sample_rate)
        return output_path, sample_rate
