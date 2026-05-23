"""tqdm configuration helpers."""

from __future__ import annotations

import threading

from tqdm import tqdm


def configure_tqdm_single_process() -> None:
    """Avoid tqdm's default multiprocessing lock in this local runtime."""
    tqdm.set_lock(threading.RLock())
    # Each tqdm bar starts a background monitor thread. Under Gradio's thread
    # pool that accumulates across long video renders and correlates with native
    # segfaults in logs/webui_faults.log.
    tqdm.monitor_interval = 0
