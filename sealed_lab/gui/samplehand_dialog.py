"""Non-modal "Sample hand" dialog of the deck builder: deal 7 cards from the deck as edited, mulligan (London), draw."""
from __future__ import annotations

import random
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sealed_lab.deck import samplehand as S

from . import helptext as H
from .widgets import muted

CARD_W = 150
CARD_H = 210


class _HandCard(QToolButton):
    """One card of the hand: the image (the builder's ImageStore), or its name while the image is missing."""

    def __init__(self, card: S.SCard, store: Any, set_code: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.card, self.store, self.set_code = card, store, set_code
        self.setFixedSize(CARD_W + 8, CARD_H + 8)
        self.setIconSize(QSize(CARD_W, CARD_H))
        self.setCheckable(False)
        self.setToolTip(card.name)
        self.setStyleSheet("QToolButton{border:2px solid transparent;border-radius:8px;padding:2px;}"
                           "QToolButton:checked{border:2px solid #d9822b;background:rgba(217,130,43,60);}")
        self.refresh_image()

    def refresh_image(self) -> None:
        pm = self.store.pixmap(self.set_code, self.card.name, CARD_W, self.devicePixelRatioF()) if self.store else None
        if pm is not None:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.setIcon(QIcon(pm))
            self.setText("")
        else:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.setIcon(QIcon())
            self.setText(self.card.name)

    @property
    def shows_image(self) -> bool:
        return self.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly


class SampleHandDialog(QDialog):
    """Opening hand + mulligans + draws for the deck currently in `builder` (read again at every "New hand")."""

    def __init__(self, builder: Any, parent: QWidget | None = None, rng: random.Random | None = None):
        super().__init__(parent or builder)
        self.builder = builder
        self.setWindowTitle("Sample hand")
        self.setModal(False)
        self.rng = rng or random.Random()
        fmt = builder.fmt if builder.fmt in ("bo1", "bo3") else "bo3"
        self.deal = S.Deal(self._cards(), self.rng, fmt, True)
        self.widgets: list[_HandCard] = []
        self._build_ui(fmt)
        builder.store.ready.connect(self._images_ready)
        self.new_hand()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, fmt: str) -> None:
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_new = QPushButton("New hand")
        self.btn_new.setToolTip(H.SAMPLE_HAND_TIP)
        self.btn_new.clicked.connect(lambda: self.new_hand())
        top.addWidget(self.btn_new)
        self.rb_play, self.rb_draw = QRadioButton("On the play"), QRadioButton("On the draw")
        self.rb_play.setChecked(True)
        self.rb_bo1, self.rb_bo3 = QRadioButton("Bo1 (smoothed)"), QRadioButton("Bo3 (plain shuffle)")
        (self.rb_bo1 if fmt == "bo1" else self.rb_bo3).setChecked(True)
        self.rb_bo1.setToolTip(H.SAMPLE_HAND_BO1_TIP)
        self.rb_bo3.setToolTip(H.SAMPLE_HAND_BO3_TIP)
        self._g1, self._g2 = QButtonGroup(self), QButtonGroup(self)
        for rb in (self.rb_play, self.rb_draw):
            self._g1.addButton(rb)
        for rb in (self.rb_bo1, self.rb_bo3):
            self._g2.addButton(rb)
        for rb in (self.rb_play, self.rb_draw):
            top.addSpacing(8)
            top.addWidget(rb)
        top.addSpacing(16)
        top.addWidget(self.rb_bo1)
        top.addWidget(self.rb_bo3)
        top.addStretch(1)
        lay.addLayout(top)
        for rb in (self.rb_play, self.rb_draw, self.rb_bo1, self.rb_bo3):
            rb.toggled.connect(self._options_changed)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.hand_host = QWidget()
        self.hand_lay = QHBoxLayout(self.hand_host)
        self.hand_lay.setContentsMargins(0, 0, 0, 0)
        self.hand_lay.setSpacing(6)
        self.hand_lay.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.hand_host)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(CARD_H + 40)
        self.scroll.setMinimumWidth(7 * (CARD_W + 14))
        lay.addWidget(self.scroll, 1)

        row = QHBoxLayout()
        self.btn_mull = QPushButton("Mulligan")
        self.btn_mull.setToolTip(H.SAMPLE_HAND_MULL_TIP)
        self.btn_mull.clicked.connect(self.mulligan)
        self.btn_confirm = QPushButton("Put on the bottom")
        self.btn_confirm.setToolTip(H.SAMPLE_HAND_MULL_TIP)
        self.btn_confirm.clicked.connect(self.confirm_bottom)
        self.btn_draw = QPushButton("Draw")
        self.btn_draw.setToolTip(H.SAMPLE_HAND_DRAW_TIP)
        self.btn_draw.clicked.connect(self.draw)
        for b in (self.btn_mull, self.btn_confirm, self.btn_draw):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setToolTip(H.SAMPLE_HAND_SUMMARY_TIP)
        lay.addWidget(self.summary)
        self.note = muted(QLabel(H.SAMPLE_HAND_NOTE))
        self.note.setWordWrap(True)
        lay.addWidget(self.note)

    # ------------------------------------------------------------------ state
    def _cards(self) -> list[S.SCard]:
        b = self.builder
        return S.library_from_deck(b.deck, b.info)

    def _options_changed(self, checked: bool = True) -> None:
        if checked:                                 # not the radio button that was just switched off
            self.new_hand()

    def new_hand(self) -> None:
        self.deal.fmt = "bo1" if self.rb_bo1.isChecked() else "bo3"
        self.deal.on_play = self.rb_play.isChecked()
        self.deal.new_hand(self._cards())
        self._show()

    def mulligan(self) -> None:
        if self.deal.mulligan():
            self._show()

    def confirm_bottom(self) -> None:
        idx = [i for i, w in enumerate(self.widgets) if w.isChecked()]
        if self.deal.bottom(idx):
            self._show()

    def draw(self) -> None:
        if self.deal.can_draw():
            self.deal.draw()
            self._show()

    def _images_ready(self, _name: str = "") -> None:
        for w in self.widgets:
            w.refresh_image()

    def _show(self) -> None:
        d = self.deal
        for w in self.widgets:
            self.hand_lay.removeWidget(w)
            w.deleteLater()
        self.widgets = []
        for i, c in enumerate(d.hand):
            w = _HandCard(c, self.builder.store, self.builder.set_code)
            w.setCheckable(d.to_bottom > 0)
            w.toggled.connect(lambda _c=False: self._update_buttons())
            self.hand_lay.insertWidget(i, w)
            self.widgets.append(w)
        side = "on the play" if d.on_play else "on the draw"
        n = len(d.hand)
        if d.to_bottom:
            self.status.setText(f"Mulligan to {7 - d.mulligans}: click {d.to_bottom} card{'s' if d.to_bottom > 1 else ''} "
                                f"to put on the bottom ({side}).")
        elif d.turn == 0:
            self.status.setText((f"Mulligan to {7 - d.mulligans} — " if d.mulligans else "Opening hand — ") + f"{n} cards, {side}.")
        else:
            self.status.setText(f"Turn {d.turn} — {side} ({n} cards in hand).")
        self._update_buttons()
        lands, src, ok = S.castable(d.hand)
        spells = [c for c in d.hand if not c.is_land]
        srcs = " ".join(f"{k}{v}" for k, v in src.items() if v)
        txt = f"{lands} land{'s' if lands != 1 else ''} in hand" + (f" (sources: {srcs})" if srcs else "")
        if spells:
            txt += f"; castable now (by land count and colors): {', '.join(c.name for c in ok) if ok else 'none'}"
        self.summary.setText(txt)

    def _update_buttons(self) -> None:
        d = self.deal
        picking = d.to_bottom > 0
        self.btn_confirm.setVisible(picking)
        self.btn_confirm.setEnabled(picking and sum(w.isChecked() for w in self.widgets) == d.to_bottom)
        self.btn_mull.setEnabled(d.can_mulligan())
        self.btn_mull.setText(f"Mulligan to {6 - d.mulligans}" if d.can_mulligan() else "Mulligan")
        self.btn_draw.setEnabled(d.can_draw())
        self.btn_draw.setText("Start turn 1" if d.turn == 0 and d.on_play else "Draw")
