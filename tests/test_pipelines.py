# -*- coding: utf-8 -*-
# @Time    : 2024/12/15
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: test_pipelines.py
import os
import pickle
from pathlib import Path
import sys

try:
    import pytest
except ModuleNotFoundError:
    pytest = None

sys.path.append(".")

JOYVASA_REQUIRED_ASSETS = (
    Path("checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"),
    Path("checkpoints/chinese-hubert-base"),
    Path("checkpoints/JoyVASA/motion_template/motion_template.pkl"),
    Path("assets/examples/driving/a-01.wav"),
)


def skip_manual(reason):
    if pytest is not None:
        pytest.skip(reason)
    raise SystemExit(reason)


def test_joyvasa_pipeline():
    if os.environ.get("FLP_RUN_JOYVASA_TEST") != "1":
        skip_manual("JoyVASA audio pipeline is experimental; set FLP_RUN_JOYVASA_TEST=1 to run manually.")

    missing_assets = [str(path) for path in JOYVASA_REQUIRED_ASSETS if not path.exists()]
    if missing_assets:
        skip_manual("missing JoyVASA assets: " + ", ".join(missing_assets))

    from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

    pipe = JoyVASAAudio2MotionPipeline(
        motion_model_path="checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt",
        audio_model_path="checkpoints/chinese-hubert-base",
        motion_template_path="checkpoints/JoyVASA/motion_template/motion_template.pkl")

    audio_path = "assets/examples/driving/a-01.wav"
    motion_data = pipe.gen_motion_sequence(audio_path)
    with open("assets/examples/driving/d1-joyvasa.pkl", "wb") as fw:
        pickle.dump(motion_data, fw)


if __name__ == '__main__':
    if os.environ.get("FLP_RUN_JOYVASA_TEST") != "1":
        print("Set FLP_RUN_JOYVASA_TEST=1 to run the experimental JoyVASA test manually.")
        raise SystemExit(0)
    test_joyvasa_pipeline()
