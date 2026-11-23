"""Side-panel widgets of the deck builder that explain a deck: the "Why this deck" waterfall and the suggested swaps."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from . import helptext as H
from .widgets import muted, semantic, sized_font


class WaterfallChart(QWidget):
    """Horizontal waterfall: average build -> card steps -> structure steps -> this deck, in game win rate (%).

    Every step is a floating bar from the win rate reached so far to the one after the step (green up, red down); the
    first and last rows are the win rates of the average real build and of this deck.  Hover a row for the details."""

    ROW = 21                    # least row height; the real one grows with the font

    def row_h(self) -> int:
        return max(self.ROW, self.fontMetrics().lineSpacing() + 7)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.steps: list[Any] = []
        self._hover = -1
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.row_h() * 2 + 8)
        self.setToolTip(H.WATERFALL_TIP)

    def set_explanation(self, expl: Any | None) -> None:
        self.steps = list(expl.steps) if expl is not None else []
        self.setFixedHeight(max(1, len(self.steps)) * self.row_h() + 8)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(300, max(1, len(self.steps)) * self.row_h() + 8)

    def clipped_texts(self) -> list[tuple[str, float, float]]:
        """Audit hook: a step label elided to the room it has is fine (its row tooltip shows the whole label); the value column must fit."""
        out: list[tuple[str, float, float]] = []
        if not self.steps:
            return out
        fm = self.fontMetrics()
        label_w, px0, pw, _lo, _hi = self._layout()
        for k, s in enumerate(self.steps):
            if fm.horizontalAdvance(s.label) > label_w - 8 and s.label not in self.tip(k):
                out.append((s.label, float(fm.horizontalAdvance(s.label)), label_w - 8))
        need = label_w + fm.horizontalAdvance("+00.00") + 10 + 50
        if need > self.width() + 1:
            out.append(("waterfall values", need, float(self.width())))
        return out

    # -------------------------------------------------------------- geometry
    def _layout(self) -> tuple[float, float, float, float, float]:
        """(label width, x of the plot start, plot width, lo %, hi %)"""
        fm = self.fontMetrics()
        label_w = min(max([fm.horizontalAdvance(s.label) for s in self.steps] + [60]) + 8, self.width() * 0.56)
        val_w = fm.horizontalAdvance("+00.00") + 10
        px0 = label_w
        pw = max(50.0, self.width() - label_w - val_w)
        lv = [100.0 * s.wr_after for s in self.steps] or [50.0]
        lo, hi = min(lv), max(lv)
        pad = max((hi - lo) * 0.06, 0.15)
        return label_w, px0, pw, lo - pad, hi + pad

    def _x(self, wr_pct: float, px0: float, pw: float, lo: float, hi: float) -> float:
        return px0 + (wr_pct - lo) / max(hi - lo, 1e-9) * pw

    def row_at(self, y: float) -> int:
        k = int((y - 4) // self.row_h())
        return k if 0 <= k < len(self.steps) else -1

    def mouseMoveEvent(self, event) -> None:
        k = self.row_at(event.position().y())
        if k != self._hover:
            self._hover = k
            self.update()
            self.setToolTip(self.tip(k))

    def leaveEvent(self, event) -> None:
        self._hover = -1
        self.update()

    def tip(self, k: int) -> str:
        if not (0 <= k < len(self.steps)):
            return H.WATERFALL_TIP
        s = self.steps[k]
        return H.waterfall_row_tip(s.label, s.kind, s.d_pp, s.d_logit, s.wr_after, list(s.detail or []))

    # -------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.steps:
            p.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "—")
            return
        pal, sem = self.palette(), semantic()
        fm = p.fontMetrics()
        label_w, px0, pw, lo, hi = self._layout()
        val_w = fm.horizontalAdvance("+00.00") + 10
        fx = lambda v: self._x(v, px0, pw, lo, hi)  # noqa: E731
        text = pal.color(QPalette.ColorRole.WindowText)
        mute = pal.color(QPalette.ColorRole.PlaceholderText)
        prev = 100.0 * self.steps[0].wr_after
        for i, s in enumerate(self.steps):
            y = 4 + i * self.row_h()
            row = QRectF(0, y, self.width(), self.row_h())
            if i == self._hover:
                p.fillRect(row, QColor(128, 128, 128, 36))
            edge = s.kind in ("start", "end")
            p.setPen(text if edge or s.kind != "structure" else text)
            lab = fm.elidedText(s.label, Qt.TextElideMode.ElideRight, int(label_w - 8))
            f = p.font()
            f.setBold(edge)
            p.setFont(f)
            p.drawText(QRectF(2, y, label_w - 6, self.row_h()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, lab)
            f.setBold(False)
            p.setFont(f)
            cur = 100.0 * s.wr_after
            p.setPen(Qt.PenStyle.NoPen)
            by, bh = y + 4, self.row_h() - 8
            if edge:
                p.setBrush(QBrush(sem["accent"] if s.kind == "end" else mute))
                x0 = px0
                p.drawRoundedRect(QRectF(x0, by, max(fx(cur) - x0, 2), bh), 2.5, 2.5)
                p.setPen(text)
                p.drawText(QRectF(fx(cur) + 4, y, fm.horizontalAdvance(f"{cur:.1f}%") + 6, self.row_h()),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{cur:.1f}%")
            else:
                a, b = fx(min(prev, cur)), fx(max(prev, cur))
                col = sem["good"] if s.d_pp >= 0 else sem["bad"]
                p.setBrush(QBrush(col))
                p.drawRoundedRect(QRectF(a, by, max(b - a, 1.5), bh), 2, 2)
                p.setPen(QPen(pal.color(QPalette.ColorRole.Mid), 1, Qt.PenStyle.DotLine))
                p.drawLine(QPointF(fx(prev), y - 1), QPointF(fx(prev), by))
                p.setPen(text)
                p.drawText(QRectF(self.width() - val_w, y, val_w - 2, self.row_h()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                           f"{s.d_pp:+.2f}")
                prev = cur
            if edge:
                prev = cur


class SuggestedSwaps(QWidget):
    """Up to three suggested swaps with gain, P(better) and an Apply button each."""

    applyRequested = Signal(int, int)       # card to remove, card to add

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(4)
        self.note = muted(QLabel(""))
        self.note.setWordWrap(True)
        self._lay.addWidget(self.note)
        self.rows: list[tuple[QWidget, QLabel, QToolButton]] = []

    def clear_rows(self) -> None:
        for w, _lab, _btn in self.rows:
            self._lay.removeWidget(w)
            w.deleteLater()
        self.rows = []

    def set_swaps(self, items: list[tuple[int, int, str, float, float | None]], note: str = "",
                  n_refits: int | None = None) -> None:
        """items: (out_id, in_id, label, gain_pp, p_better); `n_refits` = number of bootstrap refits behind p_better."""
        self.clear_rows()
        self.note.setText(note)
        self.note.setVisible(bool(note))
        for out_id, in_id, label, gain, p in items:
            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)
            lab = QLabel()
            lab.setWordWrap(True)
            col = semantic()["good"].name() if gain > 0 else semantic()["bad"].name()
            ptxt = ""
            if p is not None:
                ptxt = f" · P(better) {p * 100:.0f}%"
                if n_refits:
                    ptxt += f" ({round(p * n_refits)}/{n_refits} refits)"
            lab.setText(f"{label}<br><span style='color:{col}'><b>{gain:+.2f} pp</b></span>{ptxt}")
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setToolTip(H.swap_tip(gain, p, n_refits))
            btn = QToolButton()
            btn.setText("Apply")
            btn.setToolTip("Make this swap in the deck (one Undo step)")
            btn.clicked.connect(lambda _=False, o=out_id, i=in_id: self.applyRequested.emit(o, i))
            hl.addWidget(lab, 1)
            hl.addWidget(btn, 0, Qt.AlignmentFlag.AlignTop)
            self._lay.addWidget(w)
            self.rows.append((w, lab, btn))
