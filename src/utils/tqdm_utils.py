"""tqdm configuration helpers."""

from __future__ import annotations

import threading

from tqdm import tqdm


def configure_tqdm_single_process() -> None:
    """Avoid tqdm's default multiprocessing lock in this local runtime."""
    tqdm.set_lock(threading.RLock())
