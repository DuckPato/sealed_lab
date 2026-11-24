"""The line under a built deck on the Pool tab: which archetype of the Report tab the deck belongs to, and a link to it.

Plain observed description (`api.deck_archetype`); the builder does not use archetypes.  The backend call runs on a background thread
(latest request wins) so neither the build nor an edit of the deck waits for it.
"""
from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from .deckbuilder import _Evaluator
from .widgets import muted, sized_font


class ArchetypeLine(QWidget):
    openRequested = Signal(str, str, str)            # set code, format, colour pair key

    def __init__(self, api: Any, set_code: str, fmt: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.api, self.set_code, self.fmt = api, set_code, fmt
        self.result: dict | None = None
        self._key: tuple | None = None
        self._seq = self._shown = 0
        self._ev = _Evaluator(lambda deck: self.api.deck_archetype(self.set_code, self.fmt, deck), label=lambda: "archetype line")
        self._ev.done.connect(self._on_done)
        self._ev.failed.connect(lambda _s, _m: self._show(None))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 2)
        lay.setSpacing(8)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setFont(sized_font(self.font(), -1))
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        muted(self.label)
        self.link = QPushButton("See in Report")
        self.link.setFlat(True)
        self.link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.link.setToolTip("Open the Report tab at this colour pair")
        self.link.clicked.connect(self._open)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.link, 0, Qt.AlignmentFlag.AlignTop)
        self.setVisible(False)

    def request(self, deck: dict[int, int]) -> None:
        key = tuple(sorted((int(k), int(v)) for k, v in deck.items() if v > 0))
        if key == self._key:
            return
        self._key = key
        self._seq += 1
        self._ev.submit(self._seq, dict(key))

    def shutdown(self) -> None:
        """The page is being discarded: later answers are ignored."""
        self._seq += 1

    def _on_done(self, seq: int, res: object) -> None:
        if seq < self._seq:
            return                                    # a newer deck is on its way
        self._shown = seq
        self._show(res if isinstance(res, dict) else None)

    def _show(self, res: dict | None) -> None:
        self.result = res if res and res.get("available") and res.get("text") else None
        r = self.result
        self.setVisible(r is not None)
        if r is None:
            return
        self.label.setText(r["text"])
        tip = [f"{r['label']}: {r['range']}." if r.get("range") else ""]
        s = r.get("spectrum") or {}
        if s.get("end") in ("a", "b") and s.get("range"):
            tip.append(f"{s['label']} end: {s['range']}.")
        tip.append("Observed in 17lands data, adjusted for pilot skill. The builder does not use archetypes and did not choose a style.")
        self.label.setToolTip(" ".join(t for t in tip if t))

    def _open(self) -> None:
        if self.result:
            self.openRequested.emit(self.set_code, self.fmt, str(self.result["key"]))

    def wait_idle(self, ms: int = 3000) -> None:
        """Block until the latest request is answered (tests)."""
        end = time.monotonic() + ms / 1000
        while time.monotonic() < end:
            QApplication.processEvents()
            if self._shown == self._seq:
                return
            time.sleep(0.01)
