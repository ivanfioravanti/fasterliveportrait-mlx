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

## Migration Order

1. Keep the LivePortrait core MLX-only and avoid adding PyTorch back to that path.
2. Port or replace JoyVASA audio-to-motion:
   - export HuBERT/audio encoder weights into an MLX-compatible format,
   - port the JoyVASA diffusion/Transformer blocks to MLX,
   - verify generated motion sequences match the PyTorch baseline closely enough.
3. Replace MediaPipe human face analysis with an MLX-compatible bootstrap/refiner.
4. Once the Python runtime is MLX-only, map the model graph and preprocessing contracts
   onto `mlx-swift`.

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
- `MlxAnimalFaceAnalysisModel` tries the MLX landmark bootstrap first and falls
  back to OpenCV's packaged cat-face cascade to emit the 9-point crop-landmark
  contract when the MLX bootstrap cannot lock on.
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
