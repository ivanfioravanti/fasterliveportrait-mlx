# MLX-Only Runtime Migration

The long-term goal is a runtime that can be expressed in MLX and later moved to
`mlx-swift`. The current Python repo should therefore treat every PyTorch,
Transformers, or Python-only model path as a temporary compatibility layer.

## Current Runtime Surface

- MLX: human and animal LivePortrait core models, human landmark model, stitching,
  retargeting, and MLX-audio Kokoro text-to-speech.
- Native/non-MLX: MediaPipe human face landmarks.
- PyTorch compatibility: XPose animal landmarks and JoyVASA audio-to-motion.
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

## Compatibility Policy

Experimental compatibility paths may remain in Python while the MLX port is being
completed, but release docs should identify them clearly. The default supported surface
should not depend on ONNX, TensorRT, or hidden PyTorch fallbacks.
