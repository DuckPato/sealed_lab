"""QThread workers: the GUI never calls heavy API functions on the UI thread."""
from __future__ import annotations

import logging
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal

log = logging.getLogger(__name__)

ERROR_LOG_BYTES = 2_000_000
ERROR_LOG_KEEP = 3
_error_lock = threading.Lock()


def error_log_path():
    from sealed_lab.config import app_home

    return app_home() / "logs" / "errors.log"


def record_error(operation: str, text: str) -> None:
    """Append a failure (operation + full traceback) to APP_HOME/logs/errors.log and the logger at ERROR level.

    errors.log rotates at ~2 MB, keeping errors.1.log .. errors.3.log; writing it must never raise.
    """
    log.error("%s failed\n%s", operation or "task", text)
    try:
        path = error_log_path()
        with _error_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > ERROR_LOG_BYTES:
                for i in range(ERROR_LOG_KEEP - 1, 0, -1):
                    older = path.with_name(f"errors.{i}.log")
                    if older.exists():
                        older.replace(path.with_name(f"errors.{i + 1}.log"))
                path.replace(path.with_name("errors.1.log"))
            stamp = datetime.now().isoformat(sep=" ", timespec="seconds")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"===== {stamp}  {operation or 'task'}\n{text.rstrip()}\n\n")
    except Exception:  # noqa: BLE001 - the error log must never become a second error
        pass


def is_cancelled_exc(exc: BaseException) -> bool:
    """True for sealed_lab.types.Cancelled (matched by name so fake/real both work)."""
    return any(c.__name__ == "Cancelled" for c in type(exc).__mro__)


class _QtLogHandler(logging.Handler):
    def __init__(self, emit: Callable[[str], None]):
        super().__init__(level=logging.INFO)
        self._emit_fn = emit
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            self._emit_fn(self.format(record))
        except Exception:
            pass


class Worker(QThread):
    """Runs ``fn(progress=cb, cancel=event)`` in a thread.

    Signals: progress(float, str), log(str), result(object), error(str traceback), cancelled().
    """

    progress = Signal(float, str)
    log = Signal(str)
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, fn: Callable[..., Any], parent: QObject | None = None, capture_logging: bool = True,
                 label: str = ""):
        super().__init__(parent)
        self._fn = fn
        self.label = label or getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "") or "task"
        self.cancel_event = threading.Event()
        self._last_msg = ""
        self._last_emit = 0.0
        self._capture = capture_logging

    def cancel(self) -> None:
        self.cancel_event.set()

    # called from the worker thread by the API
    def _on_progress(self, frac: float, msg: str = "") -> None:
        now = time.monotonic()
        msg = msg or ""
        changed = msg != self._last_msg
        if changed or frac >= 1.0 or now - self._last_emit >= 0.05:
            self._last_emit = now
            self.progress.emit(float(frac), msg)
        if changed and msg:
            self._last_msg = msg
            self.log.emit(msg)

    def run(self) -> None:  # noqa: D401 - QThread API
        handler = None
        root = logging.getLogger("sealed_lab")
        if self._capture:
            handler = _QtLogHandler(self.log.emit)
            root.addHandler(handler)
            if root.level == logging.NOTSET or root.level > logging.INFO:
                root.setLevel(logging.INFO)
        try:
            res = self._fn(progress=self._on_progress, cancel=self.cancel_event)
        except BaseException as exc:  # noqa: BLE001 - report everything to the UI
            if is_cancelled_exc(exc) or (self.cancel_event.is_set() and isinstance(exc, KeyboardInterrupt)):
                self.cancelled.emit()
            else:
                tb = traceback.format_exc()
                record_error(self.label, tb)
                self.error.emit(tb)
        else:
            if self.cancel_event.is_set():
                # finished anyway; still deliver the result
                pass
            self.result.emit(res)
        finally:
            if handler is not None:
                root.removeHandler(handler)


class TaskController(QObject):
    """Owns at most one Worker at a time and re-emits its signals.

    ``started``/``ended`` bracket a run; ``ended`` fires after result/error/cancelled.
    """

    started = Signal()
    ended = Signal()
    progress = Signal(float, str)
    log = Signal(str)
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._worker: Worker | None = None

    @property
    def running(self) -> bool:
        return self._worker is not None

    def run(self, fn: Callable[..., Any], label: str = "") -> bool:
        """Start ``fn`` unless a run is active; ``label`` (e.g. "calculate HOB") names it in errors.log."""
        if self._worker is not None:
            return False
        w = Worker(fn, self, label=label)
        self._worker = w
        w.progress.connect(self.progress)
        w.log.connect(self.log)
        w.result.connect(self.result)
        w.error.connect(self.error)
        w.cancelled.connect(self.cancelled)
        w.finished.connect(self._on_finished)
        self.started.emit()
        w.start()
        return True

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def wait(self, ms: int = 5000) -> None:
        w = self._worker
        if w is not None:
            w.cancel()
            w.wait(ms)

    def _on_finished(self) -> None:
        w, self._worker = self._worker, None
        if w is not None:
            w.deleteLater()
        self.ended.emit()


def fetch_card_info(api: Any, set_code: str) -> dict[int, dict] | None:
    """Optional ``api.card_info`` -> {card_id: {name, mana_value, is_creature, is_land, colors}} or None."""
    fn = getattr(api, "card_info", None)
    if fn is None:
        return None
    try:
        info = fn(set_code)
    except AttributeError:
        return None
    except Exception:  # grouping is cosmetic; never fail a build because of it
        log.exception("card_info failed")
        return None
    return dict(info) if info else None
