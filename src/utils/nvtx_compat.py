"""No-op nvtx shim for the MLX runtime."""


class _NoopNvtx:
    @staticmethod
    def range_push(msg=""):
        pass

    @staticmethod
    def range_pop():
        pass


nvtx = _NoopNvtx()
