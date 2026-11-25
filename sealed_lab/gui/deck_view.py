"""Deck presentation: grouped card table, metrics header, mana stats, breakdown (shared by Pool and Score tabs)."""
from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sealed_lab.types import CardLine, DeckEval

from . import helptext as H
from .widgets import (
    BASIC_NAMES,
    BreakdownBars,
    ColorPips,
    FullTextTipDelegate,
    MetricCard,
    WarningBox,
    color_title,
    mono_font,
    muted,
    pct,
    pct_interval,
    remember_splitter,
    row_height,
    semantic,
    signed,
    sized_font,
    temp_flash,
)

log = logging.getLogger(__name__)

CREATURES, SPELLS, LANDS = "Creatures", "Non-creature spells", "Lands"
CAST_WARN = 0.65         # castable on time below this is flagged (same threshold as the deck builder's chip)

MANA_STATS = [
    ("p_screw", "Mana screw", "P(fewer than 3 lands in play at the end of turn 3)"),
    ("p_flood", "Flood", "P(9 or more lands among the first 16 cards seen)"),
    ("p_color_screw", "Color screw", "Share of spells the mana could pay for but the colors could not (turns 2-6)"),
    ("mana_spent", "Mana spent", "Mana spent on turns 1-7 as a share of the maximum (28)"),
    ("mull_rate", "Mulligan rate", "P(the opening seven is mulliganed)"),
]


def fetch_context(api: Any, set_code: str, fmt: str, deck: DeckEval | None) -> dict:
    """Optional extras for explaining a deck: {"reference_wr": win rate of the average real deck of the format (from the
    waterfall's starting point) or None, "smoothing": the mana simulation models Arena's Bo1 hand smoother}.  Never raises."""
    out: dict = {"reference_wr": None, "smoothing": False, "level": None}
    cn = getattr(api, "calibration_note", None)
    if cn is not None:
        try:
            out["level"] = cn(set_code, fmt)
        except Exception:  # noqa: BLE001 - cosmetic
            log.debug("calibration_note failed", exc_info=True)
    fn = getattr(api, "explain_deck", None)
    if fn is not None and deck is not None:
        try:
            counts: dict[int, int] = {}
            for c in deck.cards:
                if c.card_id is not None:
                    counts[c.card_id] = counts.get(c.card_id, 0) + c.count
            if counts:
                out["reference_wr"] = float(fn(set_code, counts, fmt).start_wr)
        except Exception:  # noqa: BLE001 - the reference is cosmetic
            log.debug("explain_deck failed", exc_info=True)
    mc = getattr(api, "model_components", None)
    if mc is not None:
        try:
            ms = (mc(set_code) or {}).get("manasim") or {}
            out["smoothing"] = bool(ms.get("applied")) and int(ms.get("version") or 1) >= 2
        except Exception:  # noqa: BLE001
            log.debug("model_components failed", exc_info=True)
    return out


# --------------------------------------------------------------------------- grouping (pure, unit-testable)

def classify_card(line: CardLine, info: dict[int, dict] | None) -> tuple[str, float]:
    """Return (group, mana_value) for a card line."""
    d = info.get(line.card_id) if (info and line.card_id is not None) else None
    if d is None:
        if line.name in BASIC_NAMES:
            return LANDS, 0.0
        return SPELLS, 99.0
    mv = float(d.get("mana_value") or 0.0)
    if d.get("is_land"):
        return LANDS, mv
    if d.get("is_creature"):
        return CREATURES, mv
    return SPELLS, mv


def group_cards(cards: list[CardLine], info: dict[int, dict] | None) -> list[tuple[str, list[CardLine]]]:
    """Group into Creatures / Non-creature spells / Lands sorted by mana value (name tiebreak).

    Without ``info`` returns one unnamed group sorted by value (best first, basics last).
    """
    if not info:
        def flat_key(c: CardLine):
            return (c.value_pp is None, -(c.value_pp or 0.0), c.name)
        return [("", sorted(cards, key=flat_key))]
    buckets: dict[str, list[tuple[float, CardLine]]] = {CREATURES: [], SPELLS: [], LANDS: []}
    for c in cards:
        g, mv = classify_card(c, info)
        buckets[g].append((mv, c))
    out: list[tuple[str, list[CardLine]]] = []
    for g in (CREATURES, SPELLS, LANDS):
        items = buckets[g]
        if not items:
            continue
        if g == LANDS:
            items.sort(key=lambda t: (t[1].name in BASIC_NAMES, t[1].name))
        else:
            items.sort(key=lambda t: (t[0], t[1].name))
        out.append((g, [c for _, c in items]))
    return out


# --------------------------------------------------------------------------- card table

class CardTable(QTableWidget):
    COLS = ["Count", "Name", "Value (pp)", "Castable", "Notes"]
    HEADER_TIPS = [
        "Copies of the card in the deck.",
        "Card name.",
        "Value in percentage points of game win rate: how much the deck's win rate falls when one copy is taken out "
        "(positive = the card helps), everything else equal.  With the castability layer the copy counts as its value x "
        "castable + the dead-card penalty x (1 - castable).  'Low data' = few games, the value leans on the prior.",
        "Chance, in the mana simulation, that this deck's lands can pay for the card on time (colors included).  "
        "Below 65% the card is flagged: the mana struggles to cast it.",
        "Warnings about the card.",
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(24)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setFont(mono_font())
        self.setItemDelegate(FullTextTipDelegate(self))
        self.verticalHeader().setDefaultSectionSize(row_height(self.font()))
        self.horizontalHeader().setHighlightSections(False)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._user_cols: set[int] = set()            # columns the user dragged keep their width; the others fit their content
        self._fitting = False
        hh.sectionResized.connect(self._on_col_resized)
        self._fit_columns()
        for k, tp in enumerate(self.HEADER_TIPS):
            self.horizontalHeaderItem(k).setToolTip(H.tip(self.COLS[k], tp))
        self.setMinimumWidth(320)
        self.n_card_rows = 0
        self.group_rows: list[str] = []

    def _on_col_resized(self, c: int, _old: int, _new: int) -> None:
        if not self._fitting:
            self._user_cols.add(c)

    def _fit_columns(self) -> None:
        """Count / value / castable / notes columns as wide as their content and their header text need (the name column takes the rest and
        elides with a tooltip)."""
        hfm = self.horizontalHeader().fontMetrics()
        self._fitting = True
        try:
            for c in (0, 2, 3, 4):
                if c in self._user_cols:
                    continue
                head = hfm.horizontalAdvance(self.COLS[c]) + 24
                self.setColumnWidth(c, max(self.sizeHintForColumn(c) + 8, head, 46))
        finally:
            self._fitting = False

    def set_cards(self, cards: list[CardLine], info: dict[int, dict] | None) -> None:
        sem = semantic()
        pal = QApplication.palette()
        self.setRowCount(0)
        self.group_rows = []
        self.n_card_rows = 0
        groups = group_cards(cards, info)
        total_rows = sum(len(g) + (1 if name else 0) for name, g in groups)
        self.setRowCount(total_rows)
        r = 0
        bold = sized_font(self.font(), 0, bold=True)
        for name, lines in groups:
            if name:
                n = sum(c.count for c in lines)
                it = QTableWidgetItem(f"{name} ({n})")
                it.setFont(bold)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                it.setBackground(pal.color(QPalette.ColorRole.AlternateBase).darker(112))
                self.setItem(r, 0, it)
                self.setSpan(r, 0, 1, len(self.COLS))
                self.group_rows.append(name)
                r += 1
            for c in lines:
                self._fill_row(r, c, sem, pal)
                self.n_card_rows += 1
                r += 1
        self.clearSelection()
        self._fit_columns()

    def _fill_row(self, r: int, c: CardLine, sem, pal) -> None:
        cnt = QTableWidgetItem(f"{c.count}×")
        cnt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        nm = QTableWidgetItem(c.name)
        val = QTableWidgetItem("" if c.value_pp is None else f"{c.value_pp:+.2f}")
        val.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if c.value_pp is not None and abs(c.value_pp) >= 0.005:
            val.setForeground(QBrush(sem["good"] if c.value_pp > 0 else sem["bad"]))
        parts = []
        if c.low_data:
            parts.append("⚠ low data")
        if c.note:
            parts.append(c.note)
        cast = QTableWidgetItem("" if c.castable is None else f"{c.castable * 100:.0f}%")
        cast.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if c.castable is not None:
            cast.setToolTip(H.castable_tip(c.castable))
            if c.castable < CAST_WARN:
                cast.setForeground(QBrush(sem["warn"]))
                cast.setText(f"⚠ {c.castable * 100:.0f}%")
        note = QTableWidgetItem(", ".join(parts))
        if c.low_data or c.note == "unmatched":
            note.setForeground(QBrush(sem["warn"]))
        elif c.note:
            note.setForeground(pal.color(QPalette.ColorRole.PlaceholderText))
        if c.card_id is None and c.name not in BASIC_NAMES:
            f = QFont(nm.font() if nm.font() else self.font())
            f.setItalic(True)
            nm.setFont(f)
            nm.setToolTip("Card not recognised by the model")
        if c.low_data:
            note.setToolTip("Few games in the data: the value estimate is dominated by the prior")
        if c.value_pp is not None:
            val.setToolTip(H.tip("Value", H.value_badge_tip(c.value_pp, c.low_data, True, castable=c.castable)))
        for col, it in enumerate((cnt, nm, val, cast, note)):
            self.setItem(r, col, it)

    def rows_text(self) -> list[tuple[str, ...]]:
        """Plain-text view of non-header rows (used by tests)."""
        out = []
        for r in range(self.rowCount()):
            if self.columnSpan(r, 0) > 1:
                continue
            out.append(tuple(self.item(r, c).text() if self.item(r, c) else "" for c in range(len(self.COLS))))
        return out


# --------------------------------------------------------------------------- deck view

class _SideColumn(QWidget):
    """The right-hand column of a DeckView; tells its owner when its width changes (a splitter drag), see DeckView._fit_side_height."""

    resized = Signal()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()


class DeckView(QWidget):
    """One deck: header (title + metric cards), card table, mana stats, breakdown, warnings, export button."""

    errorOccurred = Signal(str, str)  # short, details

    def __init__(self, exporter: Callable[[DeckEval], str] | None = None, show_metrics: bool = True,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._exporter = exporter
        self._deck: DeckEval | None = None
        self._last_args: tuple | None = None
        self.level_note: dict | None = None      # api.calibration_note: absolute-level error of the model (Game win rate tooltip)
        self.show_metrics = show_metrics

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # header --------------------------------------------------------------
        self.header = QWidget()
        hl = QVBoxLayout(self.header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        title_row = QHBoxLayout()
        self.pips = ColorPips()
        self.title = QLabel()
        self.title.setFont(sized_font(self.font(), 4, bold=True))
        self.diff_label = muted(QLabel())
        title_row.addWidget(self.pips)
        title_row.addSpacing(6)
        title_row.addWidget(self.title)
        title_row.addSpacing(12)
        title_row.addWidget(self.diff_label)
        title_row.addStretch(1)
        hl.addLayout(title_row)
        self.metrics = QWidget()
        cards_row = QHBoxLayout(self.metrics)
        cards_row.setContentsMargins(0, 0, 0, 0)
        cards_row.setSpacing(10)
        self.m_game = MetricCard("Game win rate (avg pilot)")
        self.m_match = MetricCard("Match win rate")
        self.m_score = MetricCard("Score (0–100)")
        self.m_pbest = MetricCard("P(best)")
        for m in (self.m_game, self.m_match, self.m_score, self.m_pbest):
            cards_row.addWidget(m, 1)
        self.m_pbest.set_tip(H.pbest_tip())
        hl.addWidget(self.metrics)
        self.support_box = WarningBox("Estimate less reliable: this deck is outside what real decks look like")
        self.support_box.setToolTip(H.support_tip())
        hl.addWidget(self.support_box)
        root.addWidget(self.header)
        self.metrics.setVisible(show_metrics)

        # body ----------------------------------------------------------------
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.table = CardTable()
        split.addWidget(self.table)

        side = _SideColumn()
        side.resized.connect(self._fit_side_height)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 0, 0, 0)
        sl.setSpacing(10)

        mana_box = QGroupBox("Mana simulation")
        form = QFormLayout(mana_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._mana_labels: dict[str, QLabel] = {}
        self._mana_names: dict[str, QLabel] = {}
        for key, label, tip in MANA_STATS:
            lab = QLabel(label)
            val = QLabel("—")
            val.setFont(mono_font())
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._mana_labels[key] = val
            self._mana_names[key] = lab
            form.addRow(lab, val)
        self.mana_note = muted(QLabel(""))
        self.mana_note.setWordWrap(True)
        self.mana_note.setFont(sized_font(self.font(), -1))
        form.addRow(self.mana_note)
        self._mana_tips("bo3", False)
        sl.addWidget(mana_box)
        mana_box.setToolTip(H.tip("Mana simulation", "Monte-Carlo simulation of this exact deck's opening hands and draws: how "
                                                     "often the mana fails.  Hover each line for what it means."))

        br_box = QGroupBox("Win-rate breakdown (percentage points)")
        br_box.setToolTip(H.BREAKDOWN_TITLE_TIP)
        bl = QVBoxLayout(br_box)
        self.bars = BreakdownBars()
        bl.addWidget(self.bars)
        self.bars_note = muted(QLabel("Cards = sum of card values; Structure = curve, creature and land counts; "
                                      "Mana = simulated screw/flood/color effects.  Bars are pp of game win rate, applied "
                                      "one after the other from 50%; hover a bar for the log-odds."))
        self.bars_note.setWordWrap(True)
        self.bars_note.setFont(sized_font(self.font(), -1))
        bl.addWidget(self.bars_note)
        sl.addWidget(br_box)

        self.warnings = WarningBox("Warnings")
        sl.addWidget(self.warnings)

        btn_row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy Arena export")
        self.copy_btn.clicked.connect(self.copy_export)
        self.copy_msg = muted(QLabel(""))
        btn_row.addWidget(self.copy_btn)
        btn_row.addWidget(self.copy_msg, 1)
        sl.addLayout(btn_row)
        sl.addStretch(1)
        split.addWidget(side)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)
        m = sl.contentsMargins()                                          # ... and its box titles (a wider system font, e.g. Linux)
        titles = max((QFontMetrics(g.font()).horizontalAdvance(g.title()) + 30 for g in side.findChildren(QGroupBox) if g.title()), default=0)
        side.setMinimumWidth(max(260, side.minimumSizeHint().width(), titles + m.left() + m.right()))   # never narrower than its charts and labels need
        self.table.setMinimumWidth(max(320, 36 * self.table.fontMetrics().horizontalAdvance("0")))
        split.setSizes([620, 460])
        remember_splitter(split, "deckview/h")
        self._split, self._side = split, side
        split.splitterMoved.connect(lambda *_a: self._fit_side_height())
        root.addWidget(split, 1)

    def _fit_side_height(self) -> None:
        """A splitter does not do height-for-width: when the side column gets narrow its wrapped notes need more lines than the minimum
        height assumed, so the splitter is given the height the column needs at its real width."""
        side = getattr(self, "_side", None)
        if side is not None and side.layout() is not None and side.layout().hasHeightForWidth() and side.width() > 0:
            self._split.setMinimumHeight(side.heightForWidth(side.width()))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_side_height()

    # ---------------------------------------------------------------- API
    def _mana_tips(self, fmt: str, smoothing: bool) -> None:
        for key, lab in self._mana_names.items():
            tp = H.mana_tip(key, fmt, smoothing)
            lab.setToolTip(tp)
            self._mana_labels[key].setToolTip(tp)
        self.mana_note.setText(H.mana_regime_note(fmt, smoothing).replace("<b>", "").replace("</b>", ""))

    def set_deck(self, deck: DeckEval, info: dict[int, dict] | None, fmt: str = "bo3",
                 p_best: float | None = None, diff: int | None = None, rank: int | None = None,
                 family: dict | None = None, reference_wr: float | None = None, smoothing: bool = False) -> None:
        self._deck = deck
        self._last_args = (deck, info, fmt, p_best, diff, rank, family, reference_wr, smoothing)
        self._mana_tips(fmt, smoothing)
        self.pips.set_colors(deck.colors, deck.splash)
        self.title.setText(color_title(deck.colors, deck.splash))
        if rank is None:
            self.diff_label.setText("")
        elif rank == 0:
            self.diff_label.setText("Top family" if family else "Best build")
        else:
            self.diff_label.setText(f"{diff if diff is not None else '?'} nonland cards different from #1")
        sup = list(getattr(deck, "support", None) or [])
        sub = pct_interval(deck.game_wr_lo, deck.game_wr_hi)
        if reference_wr is not None:
            sub += f" · avg real deck {pct(reference_wr)}"
        if sup:
            sub += " · ⚠ less reliable"
        self.m_game.set(pct(deck.game_wr), sub, tip=H.game_wr_tip(fmt, reference_wr, deck.game_wr, bool(sup), level=self.level_note),
                        sub_role="warn" if sup else None)
        if deck.match_wr is None:
            self.m_match.set("—", "Bo1: no match", tip=H.match_wr_tip(fmt))
        else:
            self.m_match.set(pct(deck.match_wr), "Bo3: p²(3−2p)", tip=H.match_wr_tip(fmt))
        self.m_score.set(f"{deck.score:.0f}", f"percentile · 80%: {deck.score_lo:.0f}–{deck.score_hi:.0f}", tip=H.score_tip(fmt))
        self.support_box.set_lines(sup)
        self.m_pbest.setVisible(p_best is not None)
        if p_best is not None:
            if family and rank:
                self.m_pbest.set(pct(p_best, 0), f"family · gap {-family['gap_pp']:+.1f} pp "
                                 f"[{-family['gap_hi_pp']:+.1f}, {-family['gap_lo_pp']:+.1f}]")
            else:
                self.m_pbest.set(pct(p_best, 0), "family · bootstrap refits" if family else "bootstrap draws")
        self.table.set_cards(deck.cards, info)
        for key, _label, _tip in MANA_STATS:
            v = deck.mana.get(key)
            self._mana_labels[key].setText("—" if v is None else f"{v * 100:.1f}%")
        self.bars.set_items(deck.breakdown, deck.game_wr)
        self.warnings.set_lines(list(deck.warnings))

    def deck(self) -> DeckEval | None:
        return self._deck

    def copy_export(self) -> None:
        if self._deck is None or self._exporter is None:
            return
        try:
            text = self._exporter(self._deck)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.errorOccurred.emit(f"Could not build the Arena export: {exc}", traceback.format_exc())
            return
        QApplication.clipboard().setText(text)
        temp_flash(self.copy_msg, "Copied to clipboard")

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.ApplicationPaletteChange and self._last_args is not None:
            a = self._last_args
            self.set_deck(*a[:3], p_best=a[3], diff=a[4], rank=a[5], family=a[6], reference_wr=a[7], smoothing=a[8])
