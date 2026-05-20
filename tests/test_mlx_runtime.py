import subprocess
import sys
import tomllib
from pathlib import Path

import cv2
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
    ROOT / "checkpoints/mediapipe/face_landmarker.task",
)

ANIMAL_RUNTIME_FILES = HUMAN_RUNTIME_FILES + (
    ROOT / "assets/examples/source/s39.jpg",
    ROOT / "checkpoints/liveportrait_animal_mlx/base_models_v1.1/warping_module.npz",
    ROOT / "checkpoints/liveportrait_animal_mlx/base_models_v1.1/spade_generator.npz",
    ROOT / "checkpoints/liveportrait_animal_mlx/base_models_v1.1/motion_extractor.npz",
    ROOT / "checkpoints/liveportrait_animal_mlx/base_models_v1.1/appearance_feature_extractor.npz",
    ROOT / "checkpoints/liveportrait_animals/xpose.pth",
    ROOT / "checkpoints/liveportrait_animals/clip_embedding/clip_embedding_9.pkl",
    ROOT / "checkpoints/liveportrait_animals/clip_embedding/clip_embedding_68.pkl",
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
        "MediaPipeFaceModel",
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

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime_dependencies = {dep.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0] for dep in pyproject["project"]["dependencies"]}
    assert "onnxruntime" not in runtime_dependencies
    assert "onnx" not in runtime_dependencies
    assert "onnx" in pyproject["dependency-groups"]["convert"]
    assert not (ROOT / "requirements.txt").exists()
    assert not (ROOT / "requirements_macos.txt").exists()
    assert not (ROOT / "requirements_convert.txt").exists()
    assert not (ROOT / "configs" / "onnx_infer.yaml").exists()
    assert not (ROOT / "configs" / "onnx_mp_infer.yaml").exists()


def test_importing_mlx_runtime_does_not_import_onnxruntime():
    code = (
        "import sys;"
        "import src.models;"
        "from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline;"
        "print('onnxruntime' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


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
