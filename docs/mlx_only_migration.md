# MLX-Only Runtime Migration

The long-term goal is a runtime that can be expressed in MLX and later moved to
`mlx-swift`. The current Python repo should therefore treat every PyTorch,
Transformers, or Python-only model path as a temporary compatibility layer.

## Current Runtime Surface

- MLX: human and animal LivePortrait core models, human landmark model, stitching,
  retargeting, MLX-audio Kokoro text-to-speech, and JoyVASA diffusion/motion generation.
- Native/non-MLX: MediaPipe human face landmarks.
- PyTorch compatibility: XPose animal landmarks and JoyVASA HuBERT audio feature extraction.
- Conversion-only PyTorch: export scripts that read source `.pth` checkpoints and write
  MLX `.npz` runtime weights.

## Migration Order

1. Keep the LivePortrait core MLX-only and avoid adding PyTorch back to that path.
2. Port or replace JoyVASA audio-to-motion:
   - export HuBERT/audio encoder weights into an MLX-compatible format,
   - port the JoyVASA diffusion/Transformer blocks to MLX,
   - verify generated motion sequences match the PyTorch baseline closely enough.
3. Replace XPose animal landmarks with an MLX-compatible detector, or isolate it behind
   an optional animal-only compatibility layer until an MLX model is available.
4. Once the Python runtime is MLX-only, map the model graph and preprocessing contracts
   onto `mlx-swift`.

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
- Integrated the exported MLX JoyVASA motion model into the audio/text driving pipeline.
  The temporary PyTorch portion is now limited to HuBERT feature extraction.
- Added PyTorch parity tests in `tests/test_mlx_joyvasa_motion_model.py` for each
  migrated part. These tests use fixed inputs and copied PyTorch weights so future
  migration steps can catch mathematical drift before replacing runtime paths.
- Export command: `uv run python scripts/export_mlx_weights.py --include-joyvasa`.
- Remaining JoyVASA work: port the audio encoder feature path.

## Compatibility Policy

Experimental compatibility paths may remain in Python while the MLX port is being
completed, but release docs should identify them clearly. The default supported surface
should not depend on ONNX, TensorRT, or hidden PyTorch fallbacks.
