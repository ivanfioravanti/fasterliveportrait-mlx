import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

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
    ROOT / "checkpoints/liveportrait_animal_mlx/retargeting_models/stitching.npz",
    ROOT / "checkpoints/liveportrait_animal_mlx/retargeting_models/stitching_eye.npz",
    ROOT / "checkpoints/liveportrait_animal_mlx/retargeting_models/stitching_lip.npz",
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
    assert cfg.models.face_analysis.max_num_faces == 3
    assert cfg.models.face_analysis.max_detection_candidates >= cfg.models.face_analysis.max_num_faces
    assert cfg.models.face_analysis.nms_iou_threshold == pytest.approx(0.35)
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


def test_mlx_face_detector_dedupes_overlapping_cascade_rects():
    from src.models.mlx_face_analysis_model import MlxFaceAnalysisModel

    class FakeCascade:
        def __init__(self, rects):
            self.rects = np.asarray(rects, dtype=np.int32)

        def detectMultiScale(self, *_args, **_kwargs):
            return self.rects

    model = object.__new__(MlxFaceAnalysisModel)
    model.face_cascades = [
        FakeCascade([[20, 30, 100, 100], [150, 40, 60, 60]]),
        FakeCascade([[22, 32, 98, 98], [151, 41, 59, 59]]),
    ]
    model.max_num_faces = 3
    model.max_detection_candidates = 12
    model.min_face_size = 24.0
    model.max_detector_dim = 960
    model.cascade_scale_factor = 1.05
    model.cascade_min_neighbors = 3
    model.nms_iou_threshold = 0.35
    model.cascade_output_scale = 1.0

    seeds = model._detect_face_seeds(np.zeros((256, 256, 3), dtype=np.uint8))

    assert len(seeds) == 2
    assert [round(seed_size) for _seed, seed_size in seeds] == [100, 60]


def test_multiface_crop_preview_keeps_fixed_writer_frame():
    from src.pipelines.faster_live_portrait_pipeline import _compose_face_preview

    crops = [np.full((512, 512, 3), value, dtype=np.uint8) for value in (40, 120, 220)]

    preview = _compose_face_preview(crops)

    assert preview.shape == (512, 512, 3)
    assert preview.dtype == np.uint8
    assert int(preview[64, 64, 0]) == 40
    assert int(preview[64, 320, 0]) == 120
    assert int(preview[320, 64, 0]) == 220


def test_runtime_config_resolves_checkpoint_paths_to_hf_snapshots(tmp_path, monkeypatch):
    monkeypatch.delenv("FLIP_CHECKPOINT_DIR", raising=False)

    from src.runtime_assets import get_local_checkpoints_dir, resolve_runtime_config

    cfg = OmegaConf.load(CFG_PATH)
    mlx_snapshot = tmp_path / "hub" / "models--ivanfioravanti--FasterLivePortrait-MLX-weights" / "snapshots" / "abc"

    assert get_local_checkpoints_dir() is None
    resolve_runtime_config(cfg, checkpoint_root=mlx_snapshot)

    assert cfg.models.motion_extractor.model_path == str(mlx_snapshot / "liveportrait_mlx" / "motion_extractor.npz")
    assert cfg.models.face_analysis.model_path == str(mlx_snapshot / "liveportrait_mlx" / "landmark.npz")
    assert cfg.animal_models.warping_spade.model_path[0] == str(
        mlx_snapshot / "liveportrait_animal_mlx" / "base_models_v1.1" / "warping_module.npz"
    )
    assert cfg.animal_models.stitching.model_path == str(
        mlx_snapshot / "liveportrait_animal_mlx" / "retargeting_models" / "stitching.npz"
    )
    assert cfg.animal_models.face_analysis.model_path == str(mlx_snapshot / "liveportrait_mlx" / "landmark.npz")
    assert cfg.joyvasa_models.motion_template_path == str(
        mlx_snapshot / "JoyVASA" / "motion_template" / "motion_template.pkl"
    )


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


def test_tqdm_uses_thread_lock_without_multiprocessing_semaphore():
    import importlib

    from tqdm import tqdm

    importlib.import_module("src.pipelines.faster_live_portrait_pipeline")

    lock = tqdm.get_lock()
    assert "multiprocessing" not in type(lock).__module__
    assert not hasattr(lock, "mp_lock")


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
    bbox_size = np.max(faces[0], axis=0) - np.min(faces[0], axis=0)
    assert float(max(bbox_size)) < 0.45 * max(image.shape[:2])


def test_mlx_animal_face_analysis_fallback_landmarks_match_crop_contract():
    from src.utils.crop import parse_rect_from_landmark
    from src.models.mlx_animal_face_analysis_model import MlxAnimalFaceAnalysisModel

    landmarks = MlxAnimalFaceAnalysisModel._landmarks_from_bbox((10, 20, 100, 80))

    assert landmarks.shape == (9, 2)
    assert landmarks.dtype == np.float32
    assert np.isfinite(landmarks).all()
    assert np.all(landmarks[:, 0] >= 10)
    assert np.all(landmarks[:, 0] <= 110)
    assert np.all(landmarks[:, 1] >= 20)
    assert np.all(landmarks[:, 1] <= 100)
    _, size, _ = parse_rect_from_landmark(landmarks, scale=2.3, vy_ratio=-0.125)
    assert 150 < float(size[0]) < 170


def test_mlx_animal_face_analysis_prefers_cat_cascade(monkeypatch):
    from src.models.mlx_animal_face_analysis_model import MlxAnimalFaceAnalysisModel

    model = MlxAnimalFaceAnalysisModel(enable_mlx_bootstrap=False, enable_cat_cascade=False)
    dense_landmarks = np.zeros((203, 2), dtype=np.float32)
    sparse_landmarks = np.ones((9, 2), dtype=np.float32)

    class Bootstrap:
        def predict(self, _image):
            return [dense_landmarks]

    model.bootstrap = Bootstrap()
    monkeypatch.setattr(model, "_predict_cat_cascade", lambda _image: [sparse_landmarks])

    faces = model.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0] is sparse_landmarks


def test_mlx_animal_face_analysis_predict_dense_uses_bootstrap():
    from src.models.mlx_animal_face_analysis_model import MlxAnimalFaceAnalysisModel

    model = MlxAnimalFaceAnalysisModel(enable_mlx_bootstrap=False, enable_cat_cascade=False)
    dense_landmarks = np.zeros((203, 2), dtype=np.float32)

    class Bootstrap:
        def predict(self, _image):
            return [dense_landmarks]

    model.bootstrap = Bootstrap()

    faces = model.predict_dense(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0] is dense_landmarks


def test_animal_retarget_ratios_are_calibrated_to_stable_range():
    from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline

    assert GradioLivePortraitPipeline._calibrate_animal_retarget_ratios(0.8, 0.8) == pytest.approx((0.6, 0.3))
    assert GradioLivePortraitPipeline._calibrate_animal_retarget_ratios(0.4, 0.4) == pytest.approx((0.4, 0.15))
    assert GradioLivePortraitPipeline._calibrate_animal_retarget_ratios(-1.0, -1.0) == pytest.approx((0.0, 0.0))


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


def test_reference_mlx_profile_disables_experimental_paths(monkeypatch):
    from src.utils.mlx_profiles import MLX_PROFILE_CHOICES, apply_mlx_profile

    assert "reference" in MLX_PROFILE_CHOICES
    expected = {
        "FLP_MLX_MASK_BACKEND": "native",
        "FLP_MLX_COMPILE_HOURGLASS": "0",
        "FLP_MLX_COMPILE_SPADE": "0",
        "FLP_MLX_COMPILE_MOTION": "0",
        "FLP_MLX_COMPILE_APPEARANCE": "0",
        "FLP_MLX_FUSED_UINT8": "0",
        "FLP_MLX_FUSED_DEFORMATION": "0",
        "FLP_MLX_FUSED_SPARSE_SAMPLE": "0",
        "FLP_MLX_FUSED_HOURGLASS_INPUT": "0",
        "FLP_MLX_CONV3D_BACKEND": "native",
        "FLP_MLX_GS3D_GATHER": "1",
        "FLP_MLX_WARP_OUT_BACKEND": "standard",
        "FLP_MLX_TEMPORAL_WARP_INTERVAL": "1",
        "FLP_MLX_TEMPORAL_WARP_THRESHOLD": "0",
    }
    for key in expected:
        monkeypatch.setenv(key, "sentinel")

    settings = apply_mlx_profile("reference")

    for key, value in expected.items():
        assert settings[key] == value
        assert os.environ[key] == value


def test_run_cli_overrides_cover_web_settings():
    from run import apply_cli_overrides

    cfg = OmegaConf.load(CFG_PATH)
    args = SimpleNamespace(
        relative_motion=True,
        do_crop=False,
        stitching=False,
        crop_driving_video=True,
        video_editing_head_rotation=True,
        driving_multiplier=1.25,
        animation_region="pose",
        driving_smooth_observation_variance=1e-6,
        cfg_scale=3.5,
        src_scale=2.4,
        src_vx_ratio=0.1,
        src_vy_ratio=-0.2,
        dri_scale=2.6,
        dri_vx_ratio=-0.1,
        dri_vy_ratio=0.2,
    )

    apply_cli_overrides(cfg, args)

    assert cfg.infer_params.flag_relative_motion is True
    assert cfg.infer_params.flag_do_crop is False
    assert cfg.infer_params.flag_stitching is False
    assert cfg.infer_params.flag_crop_driving_video is True
    assert cfg.infer_params.flag_video_editing_head_rotation is True
    assert cfg.infer_params.driving_multiplier == 1.25
    assert cfg.infer_params.animation_region == "pose"
    assert cfg.infer_params.driving_smooth_observation_variance == 1e-6
    assert cfg.infer_params.cfg_scale == 3.5
    assert cfg.crop_params.src_scale == 2.4
    assert cfg.crop_params.src_vx_ratio == 0.1
    assert cfg.crop_params.src_vy_ratio == -0.2
    assert cfg.crop_params.dri_scale == 2.6
    assert cfg.crop_params.dri_vx_ratio == -0.1
    assert cfg.crop_params.dri_vy_ratio == 0.2


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
    assert float(np.mean(np.all(closed_crop < 4, axis=2))) < 0.01
    assert float(np.mean(np.all(open_crop < 4, axis=2))) < 0.01
    assert closed_paste.shape == open_paste.shape
    assert closed_paste.dtype == np.uint8
    assert open_paste.dtype == np.uint8

    paste_delta = np.mean(
        np.abs(closed_paste.astype(np.int16) - open_paste.astype(np.int16))
    )
    assert paste_delta > 1.0


def test_mlx_retargeting_missing_input_is_noop():
    from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline

    pipe = object.__new__(GradioLivePortraitPipeline)

    assert pipe.execute_image(0.0, 0.0, None, True) == (None, None)
