# FasterLivePortrait-MLX: Bring Portraits to Life in Real Time

Apple MLX port of FasterLivePortrait for Apple Silicon.

[GitHub repository](https://github.com/ivanfioravanti/fasterliveportrait-mlx) | [MLX runtime weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights)

This repository is [ivanfioravanti/fasterliveportrait-mlx](https://github.com/ivanfioravanti/fasterliveportrait-mlx). Runtime `.npz` weights are hosted on Hugging Face at [ivanfioravanti/FasterLivePortrait-MLX-weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights).

This project is derived from [warmshao/FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait), which is based on [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait).

### Acknowledgements

Thanks to the authors of [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) and [LivePortrait](https://github.com/KwaiVGI/LivePortrait) for releasing the original code, model assets, and research this MLX port builds on.

### Status

Supported release surface:

- Human image, video, and camera driving on Apple Silicon using the MLX LivePortrait core.
- Animal image and video driving using the MLX LivePortrait animal core with MLX animal face analysis.
- MLX human face analysis and landmarks using the exported landmark `.npz` checkpoint.
- Optional conversion and publishing tools for exporting MLX `.npz` weights from source checkpoints.

- MLX implementations for the main human pipeline models:
  - warping module
  - SPADE generator
  - motion extractor
  - appearance feature extractor
  - landmark model
  - stitching and retargeting MLPs
- The default human runtime path is MLX-only for LivePortrait core models, face analysis, landmarks, stitching, and retargeting. ONNX is kept only as an optional conversion-time dependency for exporting MLX `.npz` weights.
- Human and driving face analysis use the MLX landmark checkpoint as a bootstrap/refiner. MediaPipe is no longer part of the default config.
- Animal source analysis uses `MlxAnimalFaceAnalysisModel`: MLX landmark bootstrap first, then a no-PyTorch cat-face cascade fallback for crop landmarks when the bootstrap cannot lock on. XPose is no longer part of the default config.
- Human MLX landmark and stitching weights load from exported `.npz` files at runtime.
- Animal base models are configured for official LivePortrait animal v1.1 weights.
- Runtime profiles are available for exact and faster approximate realtime paths.
- FastAPI and Gradio audio/text driving are present but experimental for this MLX release.

Long-term target: make the user-facing runtime MLX-only so the model stack can later move to
`mlx-swift`. This branch moves text-to-speech onto MLX-audio and ports the configured
JoyVASA HuBERT/audio-to-motion path to MLX runtime weights. Treat audio/text driving as
experimental until it has the same breadth of coverage as the image/video paths. See
[MLX-Only Runtime Migration](docs/mlx_only_migration.md) for the migration order.

### Setup

Install ffmpeg, then create the project environment with uv:

```shell
uv sync
```

Runtime weights are resolved through the normal Hugging Face Hub cache by default.
On startup, the Gradio, CLI, and API entrypoints call `snapshot_download` for
[ivanfioravanti/FasterLivePortrait-MLX-weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights)
and load files from the returned `$HF_HOME/hub/models--.../snapshots/...` directory.
To prefetch the runtime assets manually:

```shell
uv run python scripts/download_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights
```

This downloads the human MLX weights, animal LivePortrait core MLX weights, JoyVASA MLX
audio-to-motion weights, and the JoyVASA motion template into the Hugging Face Hub cache.

Gradio text driving uses [MLX-audio](https://github.com/Blaizzy/mlx-audio) with the default
`mlx-community/Kokoro-82M-bf16` model. The model and selected voice are downloaded lazily from
Hugging Face on first use; `checkpoints/Kokoro-82M` is no longer required.

Full text and audio-driven animation still requires the experimental JoyVASA assets listed in
`configs/mlx_infer.yaml` because JoyVASA converts the generated or uploaded audio into LivePortrait
motion. The configured HuBERT audio encoder and JoyVASA diffusion/motion model run through MLX
`.npz` weights downloaded from the MLX weights repo or exported locally from the trusted source
checkpoints.
The original JoyVASA `.pt` checkpoint and Transformers HuBERT directory are conversion inputs only;
they are not part of the Gradio runtime config.

To regenerate those experimental audio-driving weights locally instead of downloading them from the
MLX weights repo:

```shell
uv run python scripts/download_mlx_weights.py \
  --skip-mlx-weights \
  --include-joyvasa \
  --checkpoints-dir ./checkpoints
```

That local conversion command writes:

- `checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese_mlx.npz`
- `checkpoints/JoyVASA/audio_encoder/hubert_chinese_mlx.npz`
- `checkpoints/JoyVASA/motion_template/motion_template.pkl`

Alternatively, export the MLX weights locally from the original checkpoints:

```shell
uv run hf download warmshao/FasterLivePortrait --local-dir ./checkpoints
```

Download the official combined stitching and retargeting checkpoint:

```shell
curl -L https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth \
  -o checkpoints/liveportrait_torch/stitching_retargeting_module.pth
```

For animal mode, download the official v1.1 base models:

```shell
uv run hf download KlingTeam/LivePortrait \
  liveportrait_animals/base_models_v1.1/appearance_feature_extractor.pth \
  liveportrait_animals/base_models_v1.1/motion_extractor.pth \
  liveportrait_animals/base_models_v1.1/spade_generator.pth \
  liveportrait_animals/base_models_v1.1/warping_module.pth \
  --local-dir ./checkpoints
```

Export the MLX runtime weights after the source checkpoints are in place:

```shell
uv run --group convert python scripts/export_mlx_weights.py --include-animal --include-joyvasa
```

The default runtime weights repo is [ivanfioravanti/FasterLivePortrait-MLX-weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights). To dry-run or publish converted permissive MLX weights to that repo, run:

```shell
uv run python scripts/publish_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights \
  --dry-run

uv run python scripts/publish_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights
```

The publisher uploads converted permissive MLX runtime weights, including JoyVASA MLX
audio-to-motion weights. It does not upload the original JoyVASA PyTorch checkpoint or the
original Transformers HuBERT directory.

### Run With A Video

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s10.jpg \
  --dri_video assets/examples/driving/d14.mp4 \
  --mlx-profile quality
```

### Run With Camera

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s10.jpg \
  --dri_video 0 \
  --realtime \
  --mlx-profile turbo
```

Press `q` in the render window to exit.

### Quality Controls

The MLX config follows the official LivePortrait quality defaults where they help this fork:

- `MlxFaceAnalysisModel` runs the exported MLX landmark checkpoint on the full frame, then refines once on a face crop. It expects a visible human face in the source/driving frame; use Animal mode for animal inputs.
- `MlxAnimalFaceAnalysisModel` prefers a packaged cat-face cascade for cat-like source crops and emits the 9-point animal crop layout used by the animal pipeline, with the MLX landmark bootstrap kept as a fallback when the cascade cannot lock on.
- `driving_option: expression-friendly` scales motion using source/driving keypoint geometry.
- `flag_stabilize_driving_crop: true` keeps webcam framing centered without smile-driven zoom jitter.
- `flag_lock_driving_crop_scale: true` locks the camera crop zoom after the first detected frame.
- `flag_lock_driving_motion_scale: true` prevents the motion extractor's scale estimate from zooming the portrait.

Camera crop zoom is controlled by `crop_params.dri_scale`: lower values zoom in, higher values zoom out.

You can override the first two from the CLI:

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s10.jpg \
  --dri_video 0 \
  --realtime \
  --det-thresh 0.15 \
  --driving-option expression-friendly
```

### MLX Profiles

- `quality`: highest-fidelity default MLX path.
- `reference`: validation-oriented path with fusions, compile wrappers, and temporal reuse disabled.
- `speed`: moderate realtime reuse.
- `turbo`: aggressive realtime path, reusing warping for up to three frames.
- `ultra`: highest-throughput experimental path, reusing warping for up to eight frames.
- `custom`: keep explicit `FLP_MLX_*` environment values.

Example:

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s10.jpg \
  --dri_video 0 \
  --realtime \
  --mlx-profile ultra
```

### Benchmarks

Component benchmark:

```shell
uv run python scripts/bench_mlx_pipeline.py \
  --profile quality \
  --component warping \
  --warmup 6 \
  --iters 42
```

Experiment sweep:

```shell
uv run python scripts/bench_mlx_experiments.py \
  --warmup 2 \
  --iters 4 \
  --markdown
```

The benchmark scripts print timing in milliseconds and FPS.

### API

The FastAPI entrypoint is experimental in this release. On startup it reads `configs/mlx_infer.yaml` and resolves runtime assets from [ivanfioravanti/FasterLivePortrait-MLX-weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights) through the Hugging Face Hub cache. To prefetch the same assets, run:

```shell
uv run python scripts/download_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights
```

Set `HF_HOME` to move the Hugging Face cache. Set `FLIP_CHECKPOINT_DIR` only when you want an explicit local checkpoint tree instead of the Hub cache. Set `FLIP_MLX_WEIGHTS_REPO` or `FLIP_MLX_WEIGHTS_REVISION` to override the default MLX weights repo or revision.

### Tests

Run the MLX runtime regression checks:

```shell
uv run --group dev pytest tests/test_mlx_runtime.py -q
```

These tests verify the default config stays MLX-only, importing the runtime does not load the legacy fallback runtime, and human/animal one-frame renders are non-black when local checkpoints are present.

Run the JoyVASA MLX mathematical parity checks:

```shell
uv run --group dev pytest tests/test_mlx_joyvasa_motion_model.py -q
```

The asset-heavy JoyVASA end-to-end pipeline check is intentionally skipped in normal pytest runs. To run that manual experimental check, install its checkpoints and set `FLP_RUN_JOYVASA_TEST=1`.

For a lightweight pre-push Python sanity check over the active MLX runtime paths, run:

```shell
uv run --group dev pyflakes \
  run.py api.py webui.py \
  src/pipelines \
  src/models/mlx_face_analysis_model.py \
  src/models/mlx_joyvasa_audio_model.py \
  src/models/mlx_joyvasa_motion_model.py \
  scripts/download_mlx_weights.py scripts/export_mlx_weights.py \
  tests/test_mlx_runtime.py tests/test_mlx_joyvasa_motion_model.py
```

`pyflakes` is a development-only check for issues such as undefined names and stale imports. It is not part of the runtime dependency set.

### License

- Code: MIT, preserving the upstream license and copyright notices.
- Models: model files are subject to their respective licenses. Check the original model sources before redistribution or commercial use.
