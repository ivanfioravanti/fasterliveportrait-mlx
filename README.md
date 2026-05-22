# FasterLivePortrait-MLX

Apple MLX port of FasterLivePortrait for Apple Silicon.

[GitHub repository](https://github.com/ivanfioravanti/fasterliveportrait-mlx) | [MLX runtime weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights)

This project is derived from [warmshao/FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait), which is based on [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait).

## What Works

- Human image, video, and camera driving on Apple Silicon through MLX.
- Human multi-face source images in the regular Generate flow: up to 3 detected source faces are animated by one driving face.
- Animal image and video driving through the MLX LivePortrait animal core.
- MLX face analysis and landmarks using exported `.npz` weights.
- Stitching and eye/lip retargeting MLPs through MLX `.npz` weights.
- Runtime profiles for quality, validation, and faster realtime paths.
- Gradio web UI, CLI rendering, and an experimental FastAPI entrypoint.
- Experimental audio and text driving through MLX-audio plus JoyVASA MLX weights.

The default runtime path is MLX-only for the configured LivePortrait models, face analysis, landmarks, stitching, and retargeting. ONNX, PyTorch, and Transformers are only used by optional conversion/dev workflows.

## Requirements

- Apple Silicon Mac.
- macOS with Python 3.11 or newer.
- `ffmpeg`.
- `uv`.

Install system tools with Homebrew:

```shell
brew install ffmpeg uv
```

Then install the Python environment:

```shell
uv sync
```

## Quick Start: Web UI

Start Gradio:

```shell
uv run python webui.py
```

Open:

```text
http://127.0.0.1:9870
```

On first startup, the app downloads MLX runtime weights from [ivanfioravanti/FasterLivePortrait-MLX-weights](https://huggingface.co/ivanfioravanti/FasterLivePortrait-MLX-weights) into the normal Hugging Face cache.

To expose the UI on your local network:

```shell
uv run python webui.py --host_ip 0.0.0.0 --port 9870
```

## Web UI Usage

For the standard image-to-video flow:

1. Set **Source Input** to `Image`.
2. Upload a source portrait.
3. Set **Driving Input** to `Video`, `Webcam`, `Image`, `Pickle`, `Audio`, or `Text`.
4. Keep **is_animal** unchecked for human portraits.
5. Click **Generate**.

For animal mode:

1. Upload an animal source image or video.
2. Check **is_animal**.
3. Use a driving video/image/pickle input.
4. Click **Generate**.

For multi-face human sources:

1. Set **Source Input** to `Image`.
2. Upload [assets/examples/source/s_multi.jpg](assets/examples/source/s_multi.jpg), or another image with multiple visible human faces.
3. Keep **is_animal** unchecked.
4. Use any normal driving input and click **Generate**.

The original-space output animates all detected source faces, up to the configured limit of 3. The crop-preview output remains a fixed `512x512` tiled preview so the existing web/video writers stay compatible.

Current multi-face limitations:

- One driving face drives all detected source faces.
- Realtime webcam source mode remains single-face.
- Retargeting remains single-face.
- If only some faces animate, use clearer, more frontal, larger source faces or reduce visual overlap.

## CLI Examples

Run human image-to-video:

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s10.jpg \
  --dri_video assets/examples/driving/d14.mp4 \
  --paste-back \
  --mlx-profile quality
```

Run the multi-face sample:

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s_multi.jpg \
  --dri_video assets/examples/driving/d14.mp4 \
  --paste-back \
  --mlx-profile quality
```

Run realtime camera driving:

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s10.jpg \
  --dri_video 0 \
  --realtime \
  --paste-back \
  --mlx-profile turbo
```

Press `q` in the render window to exit.

Run animal mode:

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s39.jpg \
  --dri_video assets/examples/driving/d14.mp4 \
  --animal \
  --paste-back \
  --mlx-profile quality
```

Reuse a saved driving-motion pickle:

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s10.jpg \
  --dri_video results/<run>/<driving>.mp4.pkl \
  --paste-back \
  --mlx-profile quality
```

CLI outputs are written under `results/<timestamp>/`.

## Audio And Text Driving

Audio and text driving are experimental in this MLX release.

- Text driving uses [MLX-audio](https://github.com/Blaizzy/mlx-audio) with the default `mlx-community/Kokoro-82M-bf16` voice model.
- Text and audio driving use the configured JoyVASA MLX audio-to-motion weights.
- Kokoro voice models are downloaded lazily from Hugging Face on first use.
- JoyVASA runtime `.npz` weights are part of the MLX weights snapshot.

Use these modes from the web UI by selecting **Driving Input** as `Audio` or `Text`.

## Runtime Weights

By default, Gradio, CLI, and API entrypoints call `snapshot_download` for:

```text
ivanfioravanti/FasterLivePortrait-MLX-weights
```

The downloaded Hugging Face snapshot provides:

- Human LivePortrait MLX weights.
- Human landmark, stitching, and retargeting MLX weights.
- Animal LivePortrait v1.1 MLX weights.
- JoyVASA MLX audio encoder and motion generator weights.
- JoyVASA motion template.

To prefetch the runtime assets manually:

```shell
uv run python scripts/download_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights
```

Useful environment variables:

- `HF_HOME`: move the Hugging Face cache.
- `FLIP_MLX_WEIGHTS_REPO`: override the runtime weights repo.
- `FLIP_MLX_WEIGHTS_REVISION`: pin a model repo revision.
- `FLIP_CHECKPOINT_DIR`: use an explicit local checkpoint tree instead of the Hub cache.

Example local checkpoint tree:

```shell
export FLIP_CHECKPOINT_DIR="$PWD/checkpoints"
uv run python scripts/download_mlx_weights.py \
  --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights \
  --checkpoints-dir "$FLIP_CHECKPOINT_DIR"
```

## Optional: Export Weights Locally

Most users should use the hosted MLX runtime weights. Use local export only if you need to regenerate `.npz` files from original checkpoints.

Download the original FasterLivePortrait checkpoint set:

```shell
uv run hf download warmshao/FasterLivePortrait --local-dir ./checkpoints
```

Download the official combined stitching and retargeting checkpoint:

```shell
curl -L https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth \
  -o checkpoints/liveportrait_torch/stitching_retargeting_module.pth
```

For animal mode, download the official LivePortrait animal v1.1 base models:

```shell
uv run hf download KlingTeam/LivePortrait \
  liveportrait_animals/base_models_v1.1/appearance_feature_extractor.pth \
  liveportrait_animals/base_models_v1.1/motion_extractor.pth \
  liveportrait_animals/base_models_v1.1/spade_generator.pth \
  liveportrait_animals/base_models_v1.1/warping_module.pth \
  --local-dir ./checkpoints
```

Export MLX runtime weights:

```shell
uv run --group convert python scripts/export_mlx_weights.py \
  --include-animal \
  --include-joyvasa
```

To regenerate only the experimental JoyVASA MLX assets:

```shell
uv run python scripts/download_mlx_weights.py \
  --skip-mlx-weights \
  --include-joyvasa \
  --checkpoints-dir ./checkpoints
```

Converted JoyVASA outputs:

- `checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese_mlx.npz`
- `checkpoints/JoyVASA/audio_encoder/hubert_chinese_mlx.npz`
- `checkpoints/JoyVASA/motion_template/motion_template.pkl`

## MLX Profiles

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
  --paste-back \
  --mlx-profile ultra
```

## Quality Controls

The MLX config follows the official LivePortrait quality defaults where they help this fork:

- `MlxFaceAnalysisModel` uses OpenCV face seeds plus the exported MLX landmark checkpoint. Human source images can return up to 3 detected faces.
- `MlxAnimalFaceAnalysisModel` prefers packaged cat-face cascades for cat-like crops and keeps the MLX landmark bootstrap as fallback.
- `driving_option: expression-friendly` scales motion using source/driving keypoint geometry.
- `flag_stabilize_driving_crop: true` keeps webcam framing centered without smile-driven zoom jitter.
- `flag_lock_driving_crop_scale: true` locks camera crop zoom after the first detected frame.
- `flag_lock_driving_motion_scale: true` prevents motion-extractor scale from zooming the portrait.

Camera crop zoom is controlled by `crop_params.dri_scale`: lower values zoom in, higher values zoom out.

Common CLI overrides:

```shell
uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image assets/examples/source/s10.jpg \
  --dri_video assets/examples/driving/d14.mp4 \
  --paste-back \
  --relative-motion \
  --stitching \
  --crop-driving-video \
  --driving-option expression-friendly \
  --animation-region all \
  --src-scale 2.3 \
  --dri-scale 2.2
```

## API

The FastAPI entrypoint is experimental. Start it with:

```shell
uv run python api.py
```

Default API URL:

```text
http://127.0.0.1:9871
```

The main endpoint is:

```text
POST /predict/
```

It accepts multipart form uploads for `source_image`, `driving_video` or `driving_pickle`, plus the same core animation flags used by the web UI. It returns a zip containing generated outputs.

## Benchmarks

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

## Tests

Run the MLX runtime regression checks:

```shell
uv run --group dev pytest tests/test_mlx_runtime.py -q
```

These tests verify the default config stays MLX-only, importing the runtime does not load legacy fallback runtimes, and human/animal one-frame renders are non-black when local checkpoints are present.

Run the JoyVASA MLX mathematical parity checks:

```shell
uv run --group dev pytest tests/test_mlx_joyvasa_motion_model.py -q
```

The asset-heavy JoyVASA end-to-end pipeline check is skipped in normal pytest runs. To run that manual experimental check, install its checkpoints and set:

```shell
export FLP_RUN_JOYVASA_TEST=1
```

For a lightweight pre-push Python sanity check:

```shell
uv run --group dev pyflakes \
  run.py api.py webui.py \
  src/pipelines \
  src/models/mlx_face_analysis_model.py \
  src/models/mlx_animal_face_analysis_model.py \
  src/models/mlx_joyvasa_audio_model.py \
  src/models/mlx_joyvasa_motion_model.py \
  scripts/download_mlx_weights.py scripts/export_mlx_weights.py \
  tests/test_mlx_runtime.py tests/test_mlx_joyvasa_motion_model.py
```

## Migration Notes

The long-term target is a user-facing MLX-only runtime so the model stack can later move to `mlx-swift`. The configured image/video runtime is already MLX-only. Audio/text driving is present through MLX weights but remains experimental until it has the same coverage as image/video.

See [MLX-Only Runtime Migration](docs/mlx_only_migration.md) for the migration order and remaining details.

## Acknowledgements

Thanks to the authors of [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) and [LivePortrait](https://github.com/KwaiVGI/LivePortrait) for releasing the original code, model assets, and research this MLX port builds on.

## License

- Code: MIT, preserving the upstream license and copyright notices.
- Models: model files are subject to their respective licenses. Check the original model sources before redistribution or commercial use.
