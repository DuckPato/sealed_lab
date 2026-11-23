"""Worker processes end with the process that started them.

A pool worker normally ends when its executor shuts down, but if the app dies (a native crash, the task manager) its workers
stayed alive, each holding a few hundred MB, and every later crash added more. `watch_parent` (a pool `initializer`) starts a
thread in the worker that waits on the parent process and ends the worker when it is gone.
"""
from __future__ import annotations

import multiprocessing
import os
import threading


def _wait_and_exit(parent) -> None:
    parent.join()
    os._exit(1)


def process_alive(pid: int) -> bool:
    """Whether process `pid` is still running (True when it cannot be told)."""
    try:
        if os.name == "nt":
            import ctypes

            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x00100000, False, int(pid))          # SYNCHRONIZE
            if not h:
                return False
            try:
                return k32.WaitForSingleObject(h, 0) == 0x102         # WAIT_TIMEOUT: not signalled = alive
            finally:
                k32.CloseHandle(h)
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001 - never cancel a calculation over a failed probe
        return True


def watch_parent() -> None:
    parent = multiprocessing.parent_process()
    if parent is not None:
        threading.Thread(target=_wait_and_exit, args=(parent,), name="watch-parent", daemon=True).start()
