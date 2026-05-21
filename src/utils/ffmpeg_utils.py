"""Small ffmpeg process helpers."""

from __future__ import annotations

import subprocess
import os


def run_ffmpeg(cmd: list[str], timeout: float | None = None) -> None:
    """Run ffmpeg quietly, preserving stderr when the command fails."""
    if not cmd:
        raise ValueError("ffmpeg command is empty")
    if timeout is None:
        timeout_raw = os.environ.get("FLP_FFMPEG_TIMEOUT", "600")
        timeout = float(timeout_raw) if timeout_raw else None
        if timeout is not None and timeout <= 0:
            timeout = None
    quiet_cmd = [
        cmd[0],
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        *cmd[1:],
    ]
    try:
        subprocess.run(
            quiet_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stderr or exc.stdout or "").strip()
        details = f": {output}" if output else ""
        timeout_text = f"{timeout:g}s" if timeout is not None else "unknown timeout"
        raise TimeoutError(f"ffmpeg timed out after {timeout_text}{details}") from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stderr or exc.stdout or "").strip()
        details = f": {output}" if output else ""
        raise RuntimeError(f"ffmpeg failed with exit code {exc.returncode}{details}") from exc
