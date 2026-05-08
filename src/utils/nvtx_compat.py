"""No-op nvtx shim for non-CUDA platforms.

PyTorch's `torch.cuda.nvtx` is importable on macOS but raises at the first
range_push/range_pop call. Use this shim instead of importing nvtx directly.
"""

import torch


class _NoopNvtx:
    @staticmethod
    def range_push(msg=""):
        pass

    @staticmethod
    def range_pop():
        pass


if torch.cuda.is_available():
    from torch.cuda import nvtx as _nvtx
    nvtx = _nvtx
else:
    nvtx = _NoopNvtx()
