# MLX-Only Runtime Migration

The long-term goal is a runtime that can be expressed in MLX and later moved to
`mlx-swift`. The current Python repo should therefore treat every PyTorch,
Transformers, or Python-only model path as a temporary compatibility layer.

## Current Runtime Surface

- MLX: human and animal LivePortrait core models, human face-analysis bootstrap,
  animal source face analysis, human landmark model, stitching, retargeting,
  MLX-audio Kokoro text-to-speech, JoyVASA HuBERT audio features, and JoyVASA
  diffusion/motion generation.
- Conversion-only PyTorch: export scripts that read source PyTorch/Transformers
  checkpoints and write MLX `.npz` runtime weights.

## Migration Status

The original runtime migration order is complete for the configured Python runtime:

1. LivePortrait human and animal core models run through MLX `.npz` weights.
2. JoyVASA audio-to-motion runs through MLX for the configured HuBERT audio feature
   extractor and diffusion/motion generator.
3. Human face analysis no longer uses MediaPipe in the default runtime.
4. Animal source analysis no longer uses XPose in the default runtime.

Remaining work is no longer a blocker for the configured runtime. It is about
coverage hardening, optional upstream API parity, and future `mlx-swift` mapping.

## Remaining Migration Order

1. Keep the configured runtime MLX-only and avoid adding hidden PyTorch, ONNX,
   Transformers, MediaPipe, or XPose fallbacks back to that path.
2. Broaden JoyVASA coverage only if additional upstream JoyVASA audio encoders or
   sampling options are needed. The configured runtime already uses the MLX audio
   feature extractor before calling the MLX sampler; direct raw-audio input inside
   `MlxJoyVASAMotionModel.sample()` and dynamic thresholding are intentionally not
   required by the current pipeline.
3. Map the model graph, preprocessing contracts, and postprocessing contracts onto
   `mlx-swift`.

## Human Face Analysis MLX Progress

- Replaced the default MediaPipe human face-analysis config with `MlxFaceAnalysisModel`.
- `MlxFaceAnalysisModel` uses the exported MLX landmark checkpoint as a full-frame
  bootstrap and performs one refinement pass on the landmark crop.
- This is a face bootstrap/refiner, not a general multi-object detector; animal
  inputs still need Animal mode so the animal LivePortrait core is selected.
- Updated full-frame landmark inference to letterbox instead of stretching non-square
  images, so source/driving landmark coordinates map back into the original image
  geometry.
- MediaPipe runtime code and downloads have been removed.

## Animal Landmark MLX Progress

- Removed XPose from the default animal runtime path.
- Animal source preparation now uses `MlxAnimalFaceAnalysisModel` to define the
  source crop, then runs the MLX animal LivePortrait core models.
- `MlxAnimalFaceAnalysisModel` tries OpenCV's packaged cat-face cascade first
  for cat-like sources and emits the 9-point crop-landmark contract used by the
  animal pipeline. The MLX landmark bootstrap remains a fallback for images
  where the cat cascade cannot lock on.
- XPose runtime code and downloads have been removed.

## JoyVASA MLX Progress

- Started the JoyVASA motion port with MLX equivalents for the diffusion schedule,
  positional encoding, audio padding, alignment mask, Transformer decoder layer, and
  denoising network.
- Added the MLX reverse diffusion sampler for already-computed JoyVASA audio features,
  including the non-CFG path, audio classifier-free guidance path, indicator handling,
  and both `sample` and `noise` target update equations.
- Added `.npz` export/load support for the JoyVASA MLX motion model. The export path
  reads the trusted PyTorch checkpoint once and writes runtime weights that can be
  loaded without importing torch.
- Integrated the exported MLX JoyVASA motion model into the audio/text driving pipeline
  and removed the PyTorch JoyVASA runtime fallback.
- Added `.npz` export/load support for the configured JoyVASA HuBERT audio feature path,
  including the audio feature projection and runtime metadata.
- Integrated the exported MLX HuBERT/audio feature path into the pipeline so configured
  JoyVASA audio/text driving uses MLX for both audio features and diffusion/motion.
- Added PyTorch parity tests in `tests/test_mlx_joyvasa_motion_model.py` for each
  migrated part. These tests use fixed inputs and copied PyTorch weights so future
  migration steps can catch mathematical drift before replacing runtime paths.
- Export command: `uv run python scripts/export_mlx_weights.py --include-joyvasa`.
- Remaining JoyVASA work: broaden coverage beyond the configured Chinese HuBERT checkpoint
  if other upstream JoyVASA audio encoders are needed.

## Compatibility Policy

Experimental compatibility paths may remain in Python while the MLX port is being
completed, but release docs should identify them clearly. The default supported surface
should not depend on ONNX, TensorRT, or hidden PyTorch fallbacks.
