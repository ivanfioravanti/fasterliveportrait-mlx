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
- Animal image and video driving using the MLX LivePortrait animal core with XPose PyTorch landmarks.
- MediaPipe face landmarks for human source and driving frames.
- Optional conversion and publishing tools for exporting MLX `.npz` weights from source checkpoints.

- MLX implementations for the main human pipeline models:
  - warping module
  - SPADE generator
  - motion extractor
  - appearance feature extractor
  - landmark model
  - stitching and retargeting MLPs
- The runtime path is MLX-only for LivePortrait core models; ONNX is kept only as an optional conversion-time dependency for exporting MLX `.npz` weights.
- Human and driving face analysis use MediaPipe.
- Human MLX landmark and stitching weights load from exported `.npz` files at runtime.
- Animal base models are configured for official LivePortrait animal v1.1 weights.
- Runtime profiles are available for exact and faster approximate realtime paths.
- FastAPI, Gradio audio driving, MLX-audio text driving, and JoyVASA are present but experimental for this MLX release.

Long-term target: make the user-facing runtime MLX-only so the model stack can later move to
`mlx-swift`. This branch moves text-to-speech onto MLX-audio, but full Text and Audio driving still
passes through JoyVASA's PyTorch/Transformers audio-to-motion stack. Treat JoyVASA support here as a
temporary compatibility path, not the final architecture. See
[MLX-Only Runtime Migration](docs/mlx_only_migration.md) for the migration order.

### Setup

Install ffmpeg, then create the project environment with uv:

```shell
uv sync
```

If this checkout does not already contain model checkpoints, download the converted MLX runtime weights from [ivanfioravanti/FasterLivePortrait-MLX-weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights):

```shell
uv run python scripts/download_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights
```

This downloads the human MLX weights, animal LivePortrait core MLX weights, and the MediaPipe face landmarker.

Gradio text driving uses [MLX-audio](https://github.com/Blaizzy/mlx-audio) with the default
`mlx-community/Kokoro-82M-bf16` model. The model and selected voice are downloaded lazily from
Hugging Face on first use; `checkpoints/Kokoro-82M` is no longer required.

Full text and audio-driven animation still requires the experimental JoyVASA checkpoints listed in
`configs/mlx_infer.yaml` because JoyVASA converts the generated or uploaded audio into LivePortrait
motion.

To install those temporary experimental audio-driving assets:

```shell
uv run python scripts/download_mlx_weights.py \
  --skip-mlx-weights \
  --skip-mediapipe \
  --include-joyvasa
```

For animal mode, XPose is still required for animal landmark detection. XPose is licensed for non-commercial research use only, so it is not included in the MLX weights repo:

```shell
uv run python scripts/download_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights \
  --include-animal-xpose
```

Alternatively, export the MLX weights locally from the original checkpoints:

```shell
uv run hf download warmshao/FasterLivePortrait --local-dir ./checkpoints
```

Download the official combined stitching and retargeting checkpoint:

```shell
curl -L https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth \
  -o checkpoints/liveportrait_torch/stitching_retargeting_module.pth
```

For animal mode, download the official v1.1 base models and XPose checkpoint:

```shell
uv run hf download KlingTeam/LivePortrait \
  liveportrait_animals/base_models_v1.1/appearance_feature_extractor.pth \
  liveportrait_animals/base_models_v1.1/motion_extractor.pth \
  liveportrait_animals/base_models_v1.1/spade_generator.pth \
  liveportrait_animals/base_models_v1.1/warping_module.pth \
  liveportrait_animals/xpose.pth \
  --local-dir ./checkpoints
```

Download the cached XPose text embeddings used by this fork:

```shell
uv run hf download warmshao/FasterLivePortrait \
  liveportrait_animal_onnx/clip_embedding_9.pkl \
  liveportrait_animal_onnx/clip_embedding_68.pkl \
  --local-dir ./checkpoints
```

Move those cached embeddings to the MLX animal checkpoint layout:

```shell
mkdir -p checkpoints/liveportrait_animals/clip_embedding
cp checkpoints/liveportrait_animal_onnx/clip_embedding_9.pkl checkpoints/liveportrait_animals/clip_embedding/clip_embedding_9.pkl
cp checkpoints/liveportrait_animal_onnx/clip_embedding_68.pkl checkpoints/liveportrait_animals/clip_embedding/clip_embedding_68.pkl
```

Download the MediaPipe face landmarker used by the human MLX config:

```shell
mkdir -p checkpoints/mediapipe
curl -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -o checkpoints/mediapipe/face_landmarker.task
```

Export the MLX runtime weights after the source checkpoints are in place:

```shell
uv run --group convert python scripts/export_mlx_weights.py --include-animal
```

The default runtime weights repo is [ivanfioravanti/FasterLivePortrait-MLX-weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights). To dry-run or publish converted permissive MLX weights to that repo, run:

```shell
uv run python scripts/publish_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights \
  --dry-run

uv run python scripts/publish_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights
```

The publisher intentionally uploads only LivePortrait-derived `.npz` weights. It does not upload XPose.

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

- `det_thresh: 0.15` improves face detection recall for webcams and lower-light video.
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

- `quality`: exact baseline path.
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

The FastAPI entrypoint is experimental in this release. On startup it reads `configs/mlx_infer.yaml` and, when runtime assets are missing, downloads from [ivanfioravanti/FasterLivePortrait-MLX-weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights) by calling:

```shell
uv run python scripts/download_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights \
  --include-animal-xpose
```

Set `FLIP_CHECKPOINT_DIR` to use a checkpoint directory outside the repo. Set `FLIP_MLX_WEIGHTS_REPO` or `FLIP_MLX_WEIGHTS_REVISION` to override the default MLX weights repo or revision.

### Tests

Run the MLX runtime regression checks:

```shell
uv run --group dev pytest tests/test_mlx_runtime.py -q
```

These tests verify the default config stays MLX-only, importing the runtime does not load the legacy fallback runtime, and human/animal one-frame renders are non-black when local checkpoints are present.

The JoyVASA audio pipeline is intentionally skipped in normal pytest runs. To run that manual experimental check, install its checkpoints and set `FLP_RUN_JOYVASA_TEST=1`.

For a lightweight pre-push Python sanity check, run:

```shell
uv run --group dev pyflakes src scripts tests run.py api.py webui.py
```

`pyflakes` is a development-only check for issues such as undefined names and stale imports. It is not part of the runtime dependency set.

### License

- Code: MIT, preserving the upstream license and copyright notices.
- Models: model files are subject to their respective licenses. Check the original model sources before redistribution or commercial use.
