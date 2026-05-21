import subprocess
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "mlx_infer.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HUMAN_RUNTIME_FILES = (
    ROOT / "assets/examples/source/s10.jpg",
    ROOT / "assets/examples/source/s12.jpg",
    ROOT / "checkpoints/liveportrait_mlx/warping_module.npz",
    ROOT / "checkpoints/liveportrait_mlx/spade_generator.npz",
    ROOT / "checkpoints/liveportrait_mlx/motion_extractor.npz",
    ROOT / "checkpoints/liveportrait_mlx/appearance_feature_extractor.npz",
    ROOT / "checkpoints/liveportrait_mlx/landmark.npz",
    ROOT / "checkpoints/liveportrait_mlx/stitching.npz",
    ROOT / "checkpoints/liveportrait_mlx/stitching_eye.npz",
    ROOT / "checkpoints/liveportrait_mlx/stitching_lip.npz",
)

ANIMAL_RUNTIME_FILES = HUMAN_RUNTIME_FILES + (
    ROOT / "assets/examples/source/s39.jpg",
    ROOT / "checkpoints/liveportrait_animal_mlx/base_models_v1.1/warping_module.npz",
    ROOT / "checkpoints/liveportrait_animal_mlx/base_models_v1.1/spade_generator.npz",
    ROOT / "checkpoints/liveportrait_animal_mlx/base_models_v1.1/motion_extractor.npz",
    ROOT / "checkpoints/liveportrait_animal_mlx/base_models_v1.1/appearance_feature_extractor.npz",
)


def _require_files(paths):
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        pytest.skip("missing local runtime assets: " + ", ".join(missing))


def _assert_non_black_rgb(image, *, label):
    assert image is not None, label
    assert image.shape == (512, 512, 3)
    assert image.dtype.name == "uint8"
    assert int(image.max()) > 16, label
    assert float(image.mean()) > 1.0, label


def test_mlx_config_has_no_runtime_ort_models():
    cfg = OmegaConf.load(CFG_PATH)
    allowed_model_names = {
        "MlxAnimalFaceAnalysisModel",
        "MlxFaceAnalysisModel",
        "MlxAppearanceFeatureExtractorModel",
        "MlxLandmarkModel",
        "MlxMotionExtractorModel",
        "MlxStitchingModel",
        "MlxWarpingSpadeModel",
    }

    for group_name in ("models", "animal_models"):
        group = cfg[group_name]
        for model_name, model_cfg in group.items():
            assert model_cfg["name"] in allowed_model_names, (group_name, model_name, model_cfg["name"])
            assert model_cfg["predict_type"] != "ort", (group_name, model_name)
            model_paths = model_cfg.get("model_path")
            if isinstance(model_paths, str):
                model_paths = [model_paths]
            for model_path in model_paths or []:
                if model_cfg["name"].startswith("Mlx"):
                    assert model_path.endswith(".npz"), (group_name, model_name, model_path)

    assert cfg.animal_models.warping_spade.dtype == "bf16"
    assert cfg.animal_models.motion_extractor.dtype == "bf16"
    assert cfg.animal_models.app_feat_extractor.dtype == "bf16"
    assert cfg.models.face_analysis.name == "MlxFaceAnalysisModel"
    assert cfg.animal_models.face_analysis.name == "MlxAnimalFaceAnalysisModel"
    assert "animal_xpose" not in cfg

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime_dependencies = {dep.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0] for dep in pyproject["project"]["dependencies"]}
    assert "onnxruntime" not in runtime_dependencies
    assert "onnx" not in runtime_dependencies
    assert "kokoro" not in runtime_dependencies
    assert "insightface" not in runtime_dependencies
    assert "mediapipe" not in runtime_dependencies
    assert "torch" not in runtime_dependencies
    assert "torchvision" not in runtime_dependencies
    assert "transformers" not in runtime_dependencies
    assert "mlx-audio" in runtime_dependencies
    assert "torchgeometry" not in runtime_dependencies
    assert "onnx" in pyproject["dependency-groups"]["convert"]
    assert not (ROOT / "requirements.txt").exists()
    assert not (ROOT / "requirements_macos.txt").exists()
    assert not (ROOT / "requirements_convert.txt").exists()
    assert not (ROOT / "configs" / "onnx_infer.yaml").exists()
    assert not (ROOT / "configs" / "onnx_mp_infer.yaml").exists()


def test_importing_mlx_runtime_does_not_import_legacy_runtime_deps():
    code = (
        "import json, sys;"
        "import src.models;"
        "from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline;"
        "print(json.dumps({name: name in sys.modules for name in ['mediapipe', 'onnxruntime', 'torch']}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == '{"mediapipe": false, "onnxruntime": false, "torch": false}'


def test_mlx_face_analysis_predicts_landmarks_without_mediapipe():
    _require_files((ROOT / "assets/examples/source/s10.jpg", ROOT / "checkpoints/liveportrait_mlx/landmark.npz"))

    from src.models.mlx_face_analysis_model import MlxFaceAnalysisModel

    image = cv2.imread(str(ROOT / "assets/examples/source/s10.jpg"), cv2.IMREAD_COLOR)
    assert image is not None

    model = MlxFaceAnalysisModel(model_path=str(ROOT / "checkpoints/liveportrait_mlx/landmark.npz"), dtype="fp32")
    faces = model.predict(image)

    assert "mediapipe" not in sys.modules
    assert len(faces) == 1
    assert faces[0].shape == (203, 2)
    assert faces[0].dtype == np.float32
    assert np.isfinite(faces[0]).all()


def test_mlx_animal_face_analysis_fallback_landmarks_match_crop_contract():
    from src.models.mlx_animal_face_analysis_model import MlxAnimalFaceAnalysisModel

    landmarks = MlxAnimalFaceAnalysisModel._landmarks_from_bbox((10, 20, 100, 80))

    assert landmarks.shape == (9, 2)
    assert landmarks.dtype == np.float32
    assert np.isfinite(landmarks).all()
    assert np.all(landmarks[:, 0] >= 10)
    assert np.all(landmarks[:, 0] <= 110)
    assert np.all(landmarks[:, 1] >= 20)
    assert np.all(landmarks[:, 1] <= 100)


def test_mlx_animal_face_analysis_predicts_landmarks_without_xpose():
    _require_files((ROOT / "assets/examples/source/s39.jpg", ROOT / "checkpoints/liveportrait_mlx/landmark.npz"))

    from src.models.mlx_animal_face_analysis_model import MlxAnimalFaceAnalysisModel

    image = cv2.imread(str(ROOT / "assets/examples/source/s39.jpg"), cv2.IMREAD_COLOR)
    assert image is not None

    model = MlxAnimalFaceAnalysisModel(model_path=str(ROOT / "checkpoints/liveportrait_mlx/landmark.npz"), dtype="fp32")
    faces = model.predict(image)

    assert len(faces) == 1
    assert faces[0].shape[1] == 2
    assert faces[0].shape[0] in {9, 203}
    assert faces[0].dtype == np.float32
    assert np.isfinite(faces[0]).all()


def test_mlx_audio_kokoro_voice_language_mapping():
    from src.pipelines.mlx_audio_tts import kokoro_lang_code_for_voice

    assert kokoro_lang_code_for_voice("af_heart") == "a"
    assert kokoro_lang_code_for_voice("bf_alice") == "b"
    assert kokoro_lang_code_for_voice("jf_alpha") == "j"
    assert kokoro_lang_code_for_voice("zf_xiaobei") == "z"


def test_numpy_paste_back_does_not_require_torchgeometry():
    import numpy as np

    from src.utils.crop import paste_back_numpy

    crop = np.full((4, 4, 3), 255, dtype=np.uint8)
    ori = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4, 1), dtype=np.float32)
    transform = np.eye(3, dtype=np.float32)

    blended = paste_back_numpy(crop, transform, ori, mask)
    assert blended.dtype == np.uint8
    assert blended.shape == ori.shape
    assert int(blended.min()) == 255


@pytest.mark.parametrize(
    ("source_name", "driving_name", "is_animal", "required_files"),
    (
        ("s10.jpg", "s12.jpg", False, HUMAN_RUNTIME_FILES),
        ("s39.jpg", "s10.jpg", True, ANIMAL_RUNTIME_FILES),
    ),
)
def test_mlx_one_frame_render_is_not_black(source_name, driving_name, is_animal, required_files):
    _require_files(required_files)

    from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

    cfg = OmegaConf.load(CFG_PATH)
    pipe = FasterLivePortraitPipeline(cfg, is_animal=is_animal)
    source_path = ROOT / "assets/examples/source" / source_name
    driving_path = ROOT / "assets/examples/source" / driving_name

    assert pipe.prepare_source(str(source_path), realtime=True)
    frame = cv2.imread(str(driving_path), cv2.IMREAD_COLOR)
    assert frame is not None

    _, out_crop, _, _ = pipe.run(
        frame,
        pipe.src_imgs[0],
        pipe.src_infos[0],
        realtime=True,
        first_frame=True,
    )
    _assert_non_black_rgb(out_crop, label=f"is_animal={is_animal}")


def test_mlx_retargeting_smoke_changes_output():
    _require_files(HUMAN_RUNTIME_FILES)

    from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline

    cfg = OmegaConf.load(CFG_PATH)
    pipe = GradioLivePortraitPipeline(cfg, is_animal=False)
    source_path = ROOT / "assets/examples/source/s9.jpg"

    closed_crop, closed_paste = pipe.execute_image(0.0, 0.0, str(source_path), True)
    open_crop, open_paste = pipe.execute_image(0.8, 0.8, str(source_path), True)

    _assert_non_black_rgb(closed_crop, label="retargeting closed crop")
    _assert_non_black_rgb(open_crop, label="retargeting open crop")
    assert closed_paste.shape == open_paste.shape
    assert closed_paste.dtype == np.uint8
    assert open_paste.dtype == np.uint8

    paste_delta = np.mean(
        np.abs(closed_paste.astype(np.int16) - open_paste.astype(np.int16))
    )
    assert paste_delta > 1.0
