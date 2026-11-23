"""Concurrent tasks of one calculation: a thread per independent model fit, all heavy numerics in one shared process pool.

The validation of the optional components, the deployed combination and the final fit are independent fits of the same
kind (cross-validated chains of penalised GLMs).  Run one after the other each used only the 5 cores of its own
cross-validation, most of the machine idling.  Here every fit is driven by a coordinator thread of this process; the
cross-validation folds (and every other `glm.run_parallel` call) of all of them share ONE process pool (`glm.shared_pool`),
and the main-process parts (L-BFGS on sparse matrices, dense Hessians) release the GIL in their numpy / scipy kernels, so the
coordinators overlap too.  Nothing about the numerics changes: every task computes exactly what it computed before.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import threading
import time
from concurrent.futures import Future
from typing import Callable

from sealed_lab.model import memory
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)


class CombinedEvent:
    """Read-only `threading.Event` look-alike: set when ANY of the given events is set (callers only use `is_set`)."""

    def __init__(self, *events: threading.Event | None):
        self._events = [e for e in events if e is not None]

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


_NULL = _NullContext()


class _Slot:
    """Semaphore place + main-process memory reservation."""

    def __init__(self, sem, nbytes: int, cancel):
        self._sem, self._nbytes, self._cancel = sem, int(nbytes), cancel
        self._held = 0

    def __enter__(self):
        b = memory.current()
        self._held = b.main.acquire(self._nbytes, self._cancel) if b is not None else 0      # memory first: waiting here holds no place
        try:
            self._sem.acquire()
        except BaseException:
            if b is not None:
                b.main.release(self._held)
            raise
        return self

    def __exit__(self, *a):
        self._sem.release()
        b = memory.current()
        if b is not None and self._held:
            b.main.release(self._held)
        self._held = 0
        return False


class Runner:
    """Thread-per-task executor with a shared abort flag.

    `spawn(fn, ...)` returns a Future; an exception in any task (or an external cancel) sets `self.cancel` so the other tasks
    stop at their next `_check`, and `result()` re-raises the FIRST real error rather than the Cancelled it caused in the
    others.  `slot()` bounds how many tasks are in their heavy phase at once (memory: every fit holds its design matrices)."""

    def __init__(self, external_cancel: threading.Event | None = None, max_active: int = 8):
        self._external = external_cancel
        self._abort = threading.Event()
        self.cancel = CombinedEvent(self._abort, external_cancel)
        self._sem = threading.BoundedSemaphore(max(1, int(max_active)))
        self._threads: list[threading.Thread] = []
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def slot(self, nbytes: int = 0):
        """Context manager for the heavy phase of a task: one of `max_active` places and `nbytes` of the main-process memory
        budget (`memory.Budget`; waits while the estimates of the running tasks leave no room)."""
        return _Slot(self._sem, nbytes, self.cancel)

    def spawn(self, fn: Callable, *args, **kwargs) -> Future:
        fut: Future = Future()

        def run() -> None:
            try:
                try:
                    res = fn(*args, **kwargs)
                except BaseException as e:  # noqa: BLE001
                    if not memory.is_memory_error(e):
                        raise
                    # out of memory with others running: run the whole task again once the other tasks have left the
                    # main-process budget, alone (the task's own reservations are admitted inside this section)
                    log.warning("task %s ran out of memory; retrying it alone", getattr(fn, "__name__", fn))
                    b = memory.current()
                    ctx = b.main.exclusive(self.cancel) if b is not None else _NULL
                    with ctx:
                        res = fn(*args, **kwargs)
            except BaseException as e:  # noqa: BLE001 - handed to result()
                if not isinstance(e, Cancelled):             # (a task may also be cancelled alone, through a private flag)
                    with self._lock:
                        if self._error is None:
                            self._error = e
                    self._abort.set()
                fut.set_exception(e)
            else:
                fut.set_result(res)

        t = threading.Thread(target=run, daemon=True, name="sealed-lab-task")
        self._threads.append(t)
        t.start()
        return fut

    def result(self, fut: Future):
        """Wait for a task (polling the external cancel flag) and return its result."""
        while not fut.done():
            cf.wait([fut], timeout=0.25)
            if not fut.done() and self._external is not None and self._external.is_set():
                self._abort.set()
                raise Cancelled()
        try:
            return fut.result()
        except BaseException:                                        # the task failed
            if self._error is not None:
                raise self._error
            raise

    def close(self, timeout: float = 30.0) -> None:
        """Wait for the task threads that are still winding down after an abort."""
        end = time.time() + timeout
        for t in self._threads:
            t.join(max(0.0, end - time.time()))

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._abort.set()                                        # the other tasks stop at their next check
        self.close()


class InlineRunner:
    """Same interface, runs every task immediately in the calling thread (tests, direct calls, one-core machines)."""

    def __init__(self, cancel: threading.Event | None = None):
        self.cancel = cancel
        self._sem = threading.BoundedSemaphore(1 << 20)

    def slot(self, nbytes: int = 0):
        return _NULL                                  # sequential: nothing runs next to it

    def spawn(self, fn: Callable, *args, **kwargs) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut

    def result(self, fut: Future):
        return fut.result()

    def close(self, timeout: float = 0.0) -> None:
        return None

    def __enter__(self) -> "InlineRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class ProgressMux:
    """Combine the progress of `n` concurrent tasks into one monotone fraction in [lo, hi] (mean of the tasks' fractions)."""

    def __init__(self, progress: Callable[[float, str], None] | None, lo: float, hi: float, n: int):
        self._progress = progress
        self._lo, self._hi = lo, hi
        self._frac = [0.0] * max(1, n)
        self._lock = threading.Lock()
        self._msg = ""

    def slot(self, i: int) -> Callable[[float, str], None]:
        def prog(f: float, m: str) -> None:
            if self._progress is None:
                return
            with self._lock:
                self._frac[i] = min(max(f, self._frac[i]), 1.0)
                if m:
                    self._msg = m
                total = self._lo + (self._hi - self._lo) * sum(self._frac) / len(self._frac)
                msg = self._msg
            self._progress(total, msg)
        return prog

    def done(self, i: int) -> None:
        self.slot(i)(1.0, "")
