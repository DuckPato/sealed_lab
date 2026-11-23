"""Memory budget of a calculation: concurrent fits are admitted only while their estimated footprint fits in RAM.

Running the cross-validated chains, component tests, deployed test and final fit at the same time multiplies the peak memory
(every fit holds its design matrices, and every pool worker loads and copies one).  Two `MemoryGate`s share a budget of a
fraction of the RAM that is available when the calculation starts: one for the coordinator threads of the main process and one
for the tasks of the worker pool.  A request is admitted when the already admitted requests plus its own estimate fit in
the gate's budget; a request that does not fit waits, and a request is ALWAYS admitted when its gate is empty, so the work
degrades to sequential steps instead of deadlocking or failing (an estimate larger than the whole budget just runs alone).

`MemoryError` (also numpy's `_ArrayMemoryError`) raised by a task is handled by the callers: see `parallel.Runner` and
`glm.run_parallel`, which retry such a task once, alone, after the other tasks have finished.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from contextlib import contextmanager

from sealed_lab.types import Cancelled

GB = 1 << 30
MB = 1 << 20


COMMIT_RESERVE = 1 * GB     # commit left to the rest of the system


def memory_status() -> tuple[int, int]:
    """(available bytes, total physical bytes); (huge, huge) where it cannot be read.

    On Windows "available" is the smaller of the free RAM and the free commit (RAM + page file not yet promised to any process,
    minus COMMIT_RESERVE): with other programs open the commit can run out while RAM still looks free, and an allocation that
    cannot be committed crashed the process natively (an access violation in polars' allocator)."""
    try:
        if sys.platform == "win32":
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(min(ms.ullAvailPhys, max(ms.ullAvailPageFile - COMMIT_RESERVE, 0))), int(ms.ullTotalPhys)
        elif os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo") as fh:
                for line in fh:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0]) * 1024
            return info.get("MemAvailable", info["MemFree"]), info["MemTotal"]
    except Exception:  # noqa: BLE001 - never fail a calculation over a memory probe
        pass
    return 1 << 62, 1 << 62


def available_bytes() -> int:
    return memory_status()[0]


class MemoryGate:
    """Counting budget in bytes. `acquire(n)` blocks while `used + n` would exceed the budget (unless nothing is admitted).

    `exclusive()` waits until nothing else is admitted and then keeps every other thread out until it ends (the thread that
    holds it is admitted freely, so the work it runs can still reserve memory)."""

    def __init__(self, budget: int, name: str = "memory", floor: int = 1 * GB):
        self.budget = int(budget)
        self.name = name
        self.floor = int(floor)           # keep this much physical RAM free whatever the estimates say
        self.used = 0
        self.peak = 0
        self.waits = 0
        self.exclusive_runs = 0
        self._excl_owner: int | None = None
        self._excl_waiting = 0
        self._cv = threading.Condition()

    def _fits(self, n: int) -> bool:
        me = threading.get_ident()
        if self._excl_owner is not None:
            return self._excl_owner == me                        # inside an exclusive section only its owner is admitted
        if self._excl_waiting:
            return False                                          # let the exclusive request drain the gate first
        if self.used == 0:
            return True                                           # alone: always admitted (sequential fallback)
        if self.used + n > self.budget:
            return False
        return memory_status()[0] - n >= self.floor              # the machine itself is short of RAM (other programs)

    def acquire(self, nbytes: int, cancel=None) -> int:
        """Reserve `nbytes`; returns the amount to hand back to `release` (0 for the owner of an exclusive section)."""
        n = max(int(nbytes), 0)
        with self._cv:
            if self._excl_owner == threading.get_ident():
                return 0
            if not self._fits(n):
                self.waits += 1
                while not self._fits(n):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    self._cv.wait(0.25)
            self.used += n
            self.peak = max(self.peak, self.used)
        return n

    def try_acquire(self, nbytes: int) -> int | None:
        n = max(int(nbytes), 0)
        with self._cv:
            if self._excl_owner == threading.get_ident():
                return 0
            if not self._fits(n):
                return None
            self.used += n
            self.peak = max(self.peak, self.used)
            return n

    def release(self, nbytes: int) -> None:
        with self._cv:
            self.used = max(self.used - int(nbytes), 0)
            self._cv.notify_all()

    @contextmanager
    def reserve(self, nbytes: int, cancel=None):
        n = self.acquire(nbytes, cancel)
        try:
            yield n
        finally:
            self.release(n)

    @contextmanager
    def exclusive(self, cancel=None):
        """Run alone (as far as this gate is concerned): wait for the admitted work to finish, then block the others."""
        me = threading.get_ident()
        with self._cv:
            reentrant = self._excl_owner == me
        if reentrant:
            yield
            return
        with self._cv:
            self._excl_waiting += 1
            try:
                while self.used > 0 or self._excl_owner is not None:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    self._cv.wait(0.25)
            finally:
                self._excl_waiting -= 1
            self._excl_owner = me
            self.exclusive_runs += 1
        try:
            yield
        finally:
            with self._cv:
                self._excl_owner = None
                self._cv.notify_all()


class Budget:
    """The two gates of one calculation (main-process fits, worker-pool tasks) plus the figures for the log."""

    def __init__(self, fraction: float = 0.6, main_share: float = 0.4, available: int | None = None):
        avail, total = memory_status()
        self.available = int(avail if available is None else available)
        self.total = int(total)
        total_budget = int(self.available * fraction)
        self.main = MemoryGate(int(total_budget * main_share), "main")
        self.pool = MemoryGate(total_budget - int(total_budget * main_share), "pool")

    def summary(self) -> str:
        return (f"memory budget {(self.main.budget + self.pool.budget) / GB:.1f} GB of {self.available / GB:.1f} GB available "
                f"(main {self.main.budget / GB:.1f}, pool {self.pool.budget / GB:.1f}); peak admitted main {self.main.peak / GB:.1f} GB, "
                f"pool {self.pool.peak / GB:.1f} GB; waits {self.main.waits}/{self.pool.waits}")


_current: Budget | None = None


def current() -> Budget | None:
    return _current


@contextmanager
def budget(fraction: float | None = None, available: int | None = None):
    """Install a `Budget` for the duration of a calculation (nested use keeps the outer one). The fraction of the available RAM
    defaults to 0.6; the environment variable SEALED_LAB_MEMORY_FRACTION overrides it."""
    global _current
    if _current is not None:
        yield _current
        return
    if fraction is None:
        try:
            fraction = float(os.environ.get("SEALED_LAB_MEMORY_FRACTION", "0.6"))
        except ValueError:
            fraction = 0.6
    b = Budget(fraction, available=available)
    _current = b
    try:
        yield b
    finally:
        _current = None


# ---- footprint estimates, calibrated on the largest set (SOS: 647k draft, 42k Bo1, 18k Bo3 games; 20+ concurrent fits) with
#      tracemalloc inside the worker tasks and the process working sets of a full Calculate
WORKER_BASE = 150 * MB            # an idle pool worker (interpreter, numpy, scipy, polars)
CV_FACTOR = 5.6                   # CV fold task: traced peak <= 4.3 x its problem file, + 25% for allocator overhead
BOOT_FACTOR = 4.5                 # bootstrap replicate (matchup layer: 3.5 x the pooled problem file)
MAIN_BYTES_PER_GAME = 1500        # coordinator running a chain (stage arrays ~460 B/game x 3 for fit temporaries / copies)
OPP_BYTES_PER_GAME = 2500         # matchup layer: pooled interaction matrix (~45 non-zeros x 12 B per game, x ~4)


def estimate_chain_bytes(n_games: int, pairs: bool = False) -> int:
    """Main-process footprint of one cross-validated chain over `n_games` games of all its stages."""
    return int(n_games * MAIN_BYTES_PER_GAME * (1.3 if pairs else 1.0))


def estimate_opp_bytes(n_games: int) -> int:
    """Main-process footprint of the card x opponent layer (pooled interaction problem) over `n_games` games."""
    return int(n_games * OPP_BYTES_PER_GAME)


def is_memory_error(e: BaseException) -> bool:
    """MemoryError, numpy's _ArrayMemoryError (a MemoryError subclass) - also when it comes back from a worker process."""
    if isinstance(e, MemoryError):
        return True
    cause = e.__cause__
    while cause is not None:
        if isinstance(cause, MemoryError) or "ArrayMemoryError" in type(cause).__name__:
            return True
        cause = cause.__cause__
    return "ArrayMemoryError" in type(e).__name__ or "Unable to allocate" in str(e)


def peak_rss_mb() -> float:
    """Peak working set of this process (Windows; 0 elsewhere): used by the tests / measurements."""
    try:
        if sys.platform == "win32":
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / MB
        else:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0
