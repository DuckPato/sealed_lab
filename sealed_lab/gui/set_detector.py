"""Automatic set detection for the text pasted into the Pool tab.

Watches the paste box: a pause after the text changed (debounce) runs ``api.detect_set`` on a worker thread and emits the answer
(`detected`).  The tab turns that into "Set: HOB — detected (240/240 cards matched)" or, when the answer is ambiguous, an inline
chooser.  `newPaste` fires when the text changed so much that it counts as a different pool (the tab then forgets a set the user
chose by hand); `cleared` when the box was emptied.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QPlainTextEdit

from .workers import TaskController

DEBOUNCE_MS = 600
NEW_PASTE_OVERLAP = 0.5   # a text change keeping fewer of the previous lines than this counts as a new paste


class SetDetector(QObject):
    detected = Signal(dict)       # every answer of api.detect_set
    newPaste = Signal()
    cleared = Signal()

    def __init__(self, api: Any, text: QPlainTextEdit, parent: QObject | None = None, debounce_ms: int = DEBOUNCE_MS):
        super().__init__(parent)
        self.api, self.text = api, text
        self.last: dict | None = None
        self._lines: set[str] = set()
        self._rerun = False
        self._snapshot = ""
        self.task = TaskController(self)
        self.task.result.connect(self._on_result)
        self.task.ended.connect(self._on_ended)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(debounce_ms)
        self.timer.timeout.connect(self.run)
        text.textChanged.connect(self._on_text)

    # ------------------------------------------------------------------
    def idle(self) -> bool:
        return not self.task.running and not self.timer.isActive()

    def _on_text(self) -> None:
        t = self.text.toPlainText()
        lines = {ln.strip() for ln in t.splitlines() if ln.strip()}
        prev, self._lines = self._lines, lines
        if not lines:
            self.last = None
            self.timer.stop()
            self.cleared.emit()
            return
        if len(lines & prev) < NEW_PASTE_OVERLAP * len(lines):      # (almost) a different text: a new paste or import
            self.newPaste.emit()
        self.timer.start()

    def run(self) -> None:
        """Detect now (normally called by the debounce timer)."""
        fn = getattr(self.api, "detect_set", None)
        text = self.text.toPlainText()
        if fn is None or not text.strip():
            return
        if self.task.running:
            self._rerun = True
            return
        self._snapshot = text

        def job(progress, cancel):
            return fn(text)
        self.task.run(job, label="detect set")

    def _on_ended(self) -> None:
        if self._rerun:
            self._rerun = False
            self.timer.start()

    def _on_result(self, res: object) -> None:
        if not isinstance(res, dict) or self._snapshot != self.text.toPlainText():
            return                       # stale answer: a newer one is on its way
        self.last = res
        self.detected.emit(res)

    def wait(self, ms: int = 5000) -> None:
        self.timer.stop()
        self.task.wait(ms)
