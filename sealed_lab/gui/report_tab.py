"""The "Report" tab: the written summary of a set first, then plain-language facts about the set and its Sealed format (computed by
`sealed_lab.report`, cached per model).

For a Magic player without statistics: a sentence first, the number second, and a "How sure?" badge (Strong / Likely / Weak) on
everything that is an estimate.  The tab only READS the report (a background task asks `api.set_report`, which uses the cache).
The Summary section (written_tab.SummaryPanel) is the first section: a click on one of its sentences scrolls to the section that holds the
figure and flashes it; a list on the left jumps to a section and shows the one in view.  A card name shows the card in a floating preview.
"""
from __future__ import annotations

import logging
import traceback
from itertools import combinations
from typing import Any

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QLinearGradient, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sealed_lab.cheatsheet import POINTS_PER_LOGIT

from . import cross_view, export_ui
from . import helptext as H
from .deckbuilder import ImageStore
from .pool_tab import default_format
from .report_explorer import CardExplorer
from .widgets import (
    BusyStrip,
    ColorPips,
    ElidedCombo,
    Collapsible,
    ElidedLabel,
    FlowLayout,
    FloatingCardPreview,
    MetricCard,
    muted,
    new_settings,
    semantic,
    short_error,
    sized_font,
)
from .workers import TaskController
from .written_tab import SummaryPanel, fact_target

log = logging.getLogger(__name__)
FORMATS = ("bo1", "bo3")
PAIRS = ["".join(p) for p in combinations("WUBRG", 2)]       # WU WB WR WG UB UR UG BR BG RG (the report's own order of colors)
R_COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
DECK_SPELLS = 23                    # a deck's non-land cards: the "For your pool" value of a color pair adds up your best 23 cards in it
POOL_PAIRS = 4                      # color pairs listed in "For your pool"
POOL_ROW_WIDTH = 640                # px: the card rows of "For your pool" (a name far from its figures reads badly)
POOL_BEST = 8                       # best cards of the pool listed
POOL_HINT = "Load a pool for this set in the Deck builder to see what applies to you."
FLASH_MS = 1500                     # how long a section stays highlighted after a click on a sentence of the summary
NAV_WIDTH = 176                     # px: the list of sections on the left
NAV_LINE = 28                       # px below the top of the page: the section whose top is above this line is "in view"
SETTLE_MS = 120                     # a jump is made once more after this long (the text above it may change its height when laid out)
FOLD_TOP = 3                        # rows kept visible when every row of a table is Weak
ALL = "ALL"                         # the combo entry "All sets": the cross-set lessons (sealed_lab.cross_set)
ALL_LABEL = "All sets"
BADGE_TEXT = {"strong": "Strong", "likely": "Likely", "weak": "Weak"}
BADGE_TIP = H.tip(
    "How sure?",
    "<b>Strong</b>: the result is clear; it would be very surprising if it were just luck.<br>"
    "<b>Likely</b>: probably real, but give it some room.<br>"
    "<b>Weak</b>: could easily be chance: do not act on it.",
    "Hover a badge for the range the true value probably lies in.  In a table the Weak rows are dimmed and folded under \"Show N more with no clear difference\".")
POINTS_TIP = H.tip("Points of win rate",
                   "A \"point\" is one percentage point of the chance to win a game, near 50%.  +1.0 means winning about 51 games in "
                   "100 instead of 50.")
GROUP_ORDER = ["W", "U", "B", "R", "G", "gold", "colorless", "land"]
GROUP_TITLE = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green", "gold": "Multicolor", "colorless": "Artifacts & colorless",
               "land": "Lands"}


def pts(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def rng_text(r: dict | None, digits: int = 0) -> str:
    """"16–17" for a {lo, hi} range of a deck-shape metric (one number when both ends round to the same)."""
    if not r:
        return "–"
    lo, hi = round(r["lo"], digits), round(r["hi"], digits)
    f = (lambda x: f"{x:.{digits}f}")
    return f(lo) if lo == hi else f"{f(lo)}–{f(hi)}"


def _rgba(c: QColor, a: int) -> str:
    return f"rgba({c.red()},{c.green()},{c.blue()},{a})"


class Badge(QLabel):
    """The "How sure?" pill: Strong / Likely / Weak."""

    def __init__(self, level: str, tip: str = "", parent: QWidget | None = None):
        super().__init__(BADGE_TEXT.get(level, "Weak"), parent)
        acc = semantic()["accent"]
        pal = QApplication.palette()
        if level == "strong":
            css = f"background: {_rgba(acc, 90)}; border: 1px solid {acc.name()};"
        elif level == "likely":
            css = f"background: {_rgba(acc, 36)}; border: 1px solid {_rgba(acc, 150)};"
        else:
            css = f"background: transparent; border: 1px solid {pal.color(QPalette.ColorRole.Mid).name()};"
        self.setStyleSheet(f"QLabel {{ {css} border-radius: 6px; padding: 0px 7px; }}")
        self.setFont(sized_font(self.font(), -1))
        if level == "weak":
            muted(self)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setToolTip(H.tip("How sure?", tip, "<b>Strong</b>: clear.  <b>Likely</b>: probably real.  <b>Weak</b>: could easily be chance.") if tip else BADGE_TIP)


class EffectBar(QWidget):
    """A bar from zero to an effect (green up, red down) with a thin line for the range it probably lies in; pale when it is Weak."""

    def __init__(self, effect: float, lo: float, hi: float, scale: float, badge: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.effect, self.lo, self.hi, self.scale, self.badge = effect, lo, hi, max(scale, 0.5), badge
        self.setMinimumWidth(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(max(18, round(self.fontMetrics().height() * 1.1)))

    def sizeHint(self) -> QSize:
        return QSize(220, self.height())

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = w / 2.0
        k = (w / 2.0 - 3) / self.scale
        pal = self.palette()
        p.setPen(QPen(pal.color(QPalette.ColorRole.Mid), 1))
        p.drawLine(int(mid), 1, int(mid), h - 2)
        col = QColor(semantic()["good"] if self.effect >= 0 else semantic()["bad"])
        col.setAlpha(230 if self.badge != "weak" else 110)
        x0, x1 = sorted((mid, mid + max(-self.scale, min(self.scale, self.effect)) * k))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(col))
        bh = max(6, h - 8)
        p.drawRoundedRect(int(x0), (h - bh) // 2, max(2, int(x1 - x0)), bh, 2, 2)
        p.setPen(QPen(pal.color(QPalette.ColorRole.WindowText), 1))
        a = mid + max(-self.scale, min(self.scale, self.lo)) * k
        b = mid + max(-self.scale, min(self.scale, self.hi)) * k
        p.drawLine(int(a), h // 2, int(b), h // 2)
        p.drawLine(int(a), h // 2 - 3, int(a), h // 2 + 3)
        p.drawLine(int(b), h // 2 - 3, int(b), h // 2 + 3)


class CardLink(ElidedLabel):
    """A card name that shows the card in the floating preview when hovered or clicked."""

    shown = Signal(str, str)           # name, caption (HTML)
    left = Signal()                    # the pointer left the name

    def __init__(self, name: str, caption: str, parent: QWidget | None = None):
        super().__init__(name, parent=parent, min_chars=8)
        self.name, self.caption = name, caption
        self._owned, self._tip = 0, ""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        f = self.font()
        f.setBold(True)
        self.setFont(f)

    def set_owned(self, n: int) -> None:
        """Mark the card as one you own (a star before the name and "In your pool (2 copies)" in the tooltip); 0 removes the mark."""
        n = int(n or 0)
        if n == self._owned:
            return
        self._owned = n
        self.setText(("\u2605 " if n else "") + self.name)
        self._compose_tip()

    def owned(self) -> int:
        return self._owned

    def setToolTip(self, tip: str) -> None:
        self._tip = tip
        self._compose_tip()

    def _compose_tip(self) -> None:
        own = f"In your pool ({self._owned} cop{'y' if self._owned == 1 else 'ies'})" if self._owned else ""
        super().setToolTip("\n".join(t for t in (own, self._tip) if t))

    def enterEvent(self, event) -> None:
        self.shown.emit(self.name, self.caption)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.left.emit()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self.shown.emit(self.name, self.caption)
        super().mousePressEvent(event)


class SpectrumBar(QWidget):
    """A thin rounded bar that fades from one end style to the other (no text: the end labels sit above it)."""

    def __init__(self, tip: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(12)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if tip:
            self.setToolTip(tip)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        acc = QColor(semantic()["accent"])
        mid = self.palette().color(QPalette.ColorRole.Mid)
        g = QLinearGradient(0, 0, self.width(), 0)
        g.setColorAt(0.0, mid)
        g.setColorAt(1.0, acc)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        r = self.rect().adjusted(0, 2, 0, -2)
        p.drawRoundedRect(r, r.height() / 2.0, r.height() / 2.0)


class Chip(QLabel):
    """A short fact in a rounded outline ("17 lands"); `accent` = a shape word of an archetype ("lower curve")."""

    def __init__(self, text: str, tip: str = "", accent: bool = False, parent: QWidget | None = None):
        super().__init__(text, parent)
        acc = semantic()["accent"]
        pal = QApplication.palette()
        edge = _rgba(acc, 150) if accent else pal.color(QPalette.ColorRole.Mid).name()
        fill = _rgba(acc, 36) if accent else "transparent"
        self.setStyleSheet(f"QLabel {{ background: {fill}; border: 1px solid {edge}; border-radius: 7px; padding: 1px 8px; }}")
        self.setFont(sized_font(self.font(), -1))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if tip:
            self.setToolTip(tip)


class Section(QFrame):
    """A titled panel: heading, lead sentences, content, and a one-line note on whether the builder uses it."""

    def __init__(self, title: str, parent: QWidget | None = None, key: str = "", short: str = ""):
        super().__init__(parent)
        self.key, self.title, self.short = key, title, short or title      # `key` / `short`: the Report tab's list of sections
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setProperty("card", True)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(14, 10, 14, 12)
        self.lay.setSpacing(8)
        head = QLabel(title)
        head.setFont(sized_font(self.font(), 3, bold=True))
        self.lay.addWidget(head)
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(lambda: self._set_flash(False))

    def _set_flash(self, on: bool) -> None:
        self.setProperty("flash", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def flash(self, ms: int = FLASH_MS) -> None:
        """Highlight the section (accent border) for `ms` milliseconds."""
        self._set_flash(True)
        self._flash_timer.start(ms)

    def is_flashing(self) -> bool:
        return bool(self.property("flash"))

    def sentence(self, text: str, bullet: bool = True) -> QLabel:
        lab = QLabel(("•  " if bullet else "") + text)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lay.addWidget(lab)
        return lab

    def note(self, text: str) -> QLabel:
        lab = muted(QLabel(text))
        lab.setWordWrap(True)
        lab.setFont(sized_font(self.font(), -1))
        self.lay.addWidget(lab)
        return lab


def wrap_widget(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


class ReportTab(QWidget):
    logMessage = Signal(str)
    statusMessage = Signal(str)
    errorOccurred = Signal(str, str)
    FLASH_MS = FLASH_MS

    def __init__(self, api: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self.statuses: dict[str, Any] = {}
        self.calculating: set[str] = set()
        self.reports: dict[tuple[str, str], dict] = {}          # (set, format) -> report; dropped when the set's model is rebuilt
        self.built: dict[str, str] = {}                          # set -> built_at the cached reports belong to
        self.current: dict | None = None
        self._task_key: tuple[str, str] | None = None            # (set, format) the running report task was started for
        self._pool_req: tuple[str, str] | None = None            # (set, built_at) the running pool-data task was started for
        self._pool_failed: tuple[str, str] | None = None         # (set, built_at) whose card table failed to load: not retried
        self.cross: dict | None = None                           # api.cross_set result (the "All sets" view); dropped when a model is rebuilt
        self._loaded_key: tuple[str, str] | None = None
        self._fmt_user: dict[str, str] = {}                      # set -> format the user picked
        self._rendering = False
        self._open: set[str] = set()                             # color pairs whose archetype block the user opened
        self._pair_widgets: dict[str, tuple[QWidget, QToolButton]] = {}   # color pair -> (its archetype block, its open/close button)
        self._pending_focus: dict | None = None                  # {key: (set, fmt), section, detail, pair}: scroll to it once that report is shown
        self._secs: dict[str, Section] = {}                      # section key -> section, in page order (the list on the left)
        self._details: dict[str, Collapsible] = {}               # "More detail" key -> its block
        self._fold_open: set[str] = set()                        # tables whose Weak rows the user unfolded (this session only)
        self._programmatic = False                               # the scroll bar is being moved by a jump (the list keeps what was clicked)
        self._user_scrolls = 0                                   # how often the user scrolled (a jump is not corrected after that)
        self.summary: SummaryPanel | None = None                 # the written summary on screen (its section is the first one)
        self._pool: tuple[str, dict[int, int]] = ("", {})        # the Deck builder's pool: set code, {card_id: copies}
        self._best: tuple[str, str, str] = ("", "", "")          # the builder's best deck from it: set, format, main colors
        self._pool_info: dict[str, dict] = {}                    # set -> card_info (for the pool section and the marks)
        self._pool_vals: dict[tuple[str, str], dict] = {}        # (set, format) -> card_values
        self._owned: dict[str, int] = {}                         # card name -> copies, for the report on screen ({} when the pool is of another set)
        self._card_links: list[CardLink] = []                    # the card names on screen that can carry the "in your pool" mark
        self._pool_sec: Section | None = None                    # the "For your pool" section on screen
        self.pool_hint: QLabel | None = None                     # the line under the Summary that asks for a pool
        self.pool_summary: dict = {}                             # what the pool section lists (pairs, good, watch, best), for tests
        self.explorer: CardExplorer | None = None
        self._cheat_path = ""
        self.pool_task = TaskController(self)                    # card table + card values for the pool section (the values load the model)
        self.pool_task.result.connect(self._on_pool_data)
        self.pool_task.error.connect(self._on_pool_error)
        self.pool_task.ended.connect(self._ensure_pool_data)
        self.cheat_task = TaskController(self)                   # the cheat sheet page (reads the report cache and the model)
        self.cheat_task.result.connect(self._on_cheat_result)
        self.cheat_task.error.connect(self._on_cheat_error)
        self.cheat_task.ended.connect(self._update_controls)
        self.task = TaskController(self)
        self.task.started.connect(self._on_started)
        self.task.progress.connect(lambda f, m: self.strip.update_progress(f, m))
        self.task.result.connect(self._on_result)
        self.task.error.connect(self._on_error)
        self.task.cancelled.connect(lambda: self.strip.end("Cancelled"))
        self.task.ended.connect(self._on_ended)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("Report")
        title.setFont(sized_font(self.font(), 3, bold=True))
        head.addWidget(title)
        head.addSpacing(10)
        head.addWidget(QLabel("Set"))
        self.set_combo = ElidedCombo()
        self.set_combo.setToolTip("Sets with a calculated model")
        self.set_combo.activated.connect(self._on_set_chosen)
        head.addWidget(self.set_combo)
        head.addSpacing(10)
        self.fmt_bo1, self.fmt_bo3 = QRadioButton("Best of 1"), QRadioButton("Best of 3")
        self.fmt_bo1.setToolTip(H.FORMAT_BO1_LABEL)
        self.fmt_bo3.setToolTip(H.FORMAT_BO3_LABEL)
        self.fmt_group = QButtonGroup(self)
        for r in (self.fmt_bo1, self.fmt_bo3):
            self.fmt_group.addButton(r)
            head.addWidget(r)
        self.fmt_bo3.setChecked(True)
        self.fmt_bo1.clicked.connect(lambda: self._on_format_clicked("bo1"))
        self.fmt_bo3.clicked.connect(lambda: self._on_format_clicked("bo3"))
        head.addStretch(1)
        self.refresh_btn = QPushButton("Recalculate report")
        self.refresh_btn.setToolTip("Compute the report again from the data (normally not needed: it is kept until the model is rebuilt)")
        self.refresh_btn.clicked.connect(lambda: self.load(refresh=True))
        head.addWidget(self.refresh_btn)
        self.cheat_btn = export_ui.make_cheat_button(self)
        head.addWidget(self.cheat_btn)
        self.export_btn = export_ui.make_button(self)
        head.addWidget(self.export_btn)
        root.addLayout(head)
        self.strip = BusyStrip()
        self.strip.cancelRequested.connect(self.task.cancel)
        root.addWidget(self.strip)

        main = QHBoxLayout()
        main.setSpacing(8)
        self.nav = QListWidget()                                  # the sections, to jump to one; follows the scrolling
        self.nav.setObjectName("reportNav")
        self.nav.setFixedWidth(NAV_WIDTH)
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.nav.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.nav.setToolTip("Jump to a section")
        self.nav.itemClicked.connect(self._on_nav_clicked)
        main.addWidget(self.nav)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 8, 0)
        self.body_lay.setSpacing(10)
        self.scroll.setWidget(self.body)
        main.addWidget(self.scroll, 1)
        root.addLayout(main, 1)
        self.hover_popup = FloatingCardPreview(ImageStore.for_api(api), self)       # a card name hovered: the card beside the pointer
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scrolled)
        self.scroll.verticalScrollBar().actionTriggered.connect(lambda _a: setattr(self, "_user_scrolls", self._user_scrolls + 1))   # (wheel, drag, keys)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        muted(self.message)
        self.body_lay.addWidget(self.message)
        self.body_lay.addStretch(1)
        self._show_message("Choose a set that has a calculated model.")
        self._update_controls()

    # ------------------------------------------------------------------ state
    def fmt(self) -> str:
        return "bo1" if self.fmt_bo1.isChecked() else "bo3"

    def set_code(self) -> str:
        return str(self.set_combo.currentData() or "")

    def key(self) -> tuple[str, str] | None:
        sc = self.set_code()
        return (sc, self.fmt()) if sc else None

    def _show_message(self, text: str) -> None:
        self.scroll.setMinimumWidth(0)
        self._clear_sections()
        self.message.setText(text)
        self.message.setVisible(True)
        self.body_lay.addStretch(1)
        self._update_controls()

    def _update_controls(self) -> None:
        st = self.statuses.get(self.set_code())
        have = set(getattr(st, "formats", []) or []) if st is not None else set()
        for f, r in (("bo1", self.fmt_bo1), ("bo3", self.fmt_bo3)):
            r.setEnabled(f in have)                              # (for "All sets" both stay disabled: the tables show both formats)
        every = self.set_code() == ALL
        self.refresh_btn.setEnabled(bool(self.set_code()) and not every and not self.task.running and self.set_code() not in self.calculating)
        export_ui.set_enabled(self.export_btn, bool(self.set_code()), not every)
        shown = self.current is not None and self._loaded_key == self.key() and not self.current.get("too_few") and not self.message.isVisible()
        export_ui.set_cheat_enabled(self.cheat_btn, shown and not self.cheat_task.running, every, shown)

    # ------------------------------------------------------------------ sets
    def refresh_sets(self, statuses: list) -> None:
        """The Sets tab's list changed: keep the sets that have a model; a rebuilt model drops that set's cached reports."""
        self.statuses = {st.set_code: st for st in statuses if st.has_model}
        if {c: st.built_at for c, st in self.statuses.items()} != self.built:
            self.cross = None                                    # the lessons were made from other models
            if self._loaded_key and self._loaded_key[0] == ALL:
                self._loaded_key = None
        for code, st in self.statuses.items():
            if self.built.get(code) not in (None, st.built_at):
                for k in [k for k in self.reports if k[0] == code]:
                    del self.reports[k]
                self._pool_info.pop(code, None)                  # (the card values belong to the old model)
                for k in [k for k in self._pool_vals if k[0] == code]:
                    del self._pool_vals[k]
                if self._loaded_key and self._loaded_key[0] == code:
                    self._loaded_key = None
            self.built[code] = st.built_at
        prev = self.set_code() or str(new_settings().value("report/set", "") or "")
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        if self.statuses:
            self.set_combo.addItem(ALL_LABEL, ALL)             # "All sets" is first, but never the default
        for code in self.statuses:                             # (the Sets tab's order: newest release first)
            self.set_combo.addItem(code, code)
        self.set_combo.setCurrentIndex((self.set_combo.findData(prev) if self.set_combo.findData(prev) >= 0 else (1 if self.statuses else -1)))
        self.set_combo.blockSignals(False)
        self._pick_format()
        if not self.statuses:
            self.current, self._loaded_key = None, None
            self._show_message("No set has a calculated model yet.  Calculate one on the Sets tab.")
        self._update_controls()
        if self.isVisible():
            self.load()

    def set_calculating(self, sets: set[str]) -> None:
        self.calculating = {s.upper() for s in sets}
        self._update_controls()
        if self.set_code() in self.calculating:
            self._loaded_key = None
            self._show_message(f"{self.set_code()} is being recalculated: its report is available again when the calculation finishes.")
        elif self.isVisible():
            self.load()

    def _pick_format(self) -> None:
        sc = self.set_code()
        st = self.statuses.get(sc)
        if st is None:
            return
        want = self._fmt_user.get(sc) or str(new_settings().value(f"report/fmt/{sc}", "") or "") or default_format(st) or "bo3"
        if want not in st.formats:
            want = default_format(st) or "bo3"
        (self.fmt_bo1 if want == "bo1" else self.fmt_bo3).setChecked(True)

    def _on_set_chosen(self, _i: int) -> None:
        self._apply_set()

    def _apply_set(self) -> None:
        new_settings().setValue("report/set", self.set_code())
        self._pick_format()
        self._update_controls()
        if self.isVisible():
            self.load()

    def select_set(self, code: str) -> None:
        """Choose a set programmatically (no `activated` signal); the report loads when this tab is shown."""
        i = self.set_combo.findData(code)
        if i < 0 or i == self.set_combo.currentIndex():
            return
        self.set_combo.blockSignals(True)
        self.set_combo.setCurrentIndex(i)
        self.set_combo.blockSignals(False)
        self._apply_set()

    def show_pair(self, code: str, fmt: str, key: str) -> None:
        """Open the archetype block of color pair `key` for set `code` and format `fmt` (the Deck builder's link): select the set and format,
        load the report if needed, open the block and scroll to it."""
        self.select_set(code)
        if self.set_code() != code:
            return
        r = self.fmt_bo1 if fmt == "bo1" else self.fmt_bo3
        if r.isEnabled() and not r.isChecked():
            r.setChecked(True)
            self._fmt_user[code] = fmt
            new_settings().setValue(f"report/fmt/{code}", fmt)
        self._pending_focus = dict(key=(code, self.fmt()), section="archetypes", detail=None, pair=key)
        self.load()
        QTimer.singleShot(0, self._apply_pending_focus)

    # ------------------------------------------------------------------ jumping to a section
    def focus_section(self, section: str, fmt: str | None = None, detail: str | None = None, pair: str | None = None) -> None:
        """Scroll to a section (and flash it).  `fmt`: show that format's report first; `detail`: open that "More detail" block;
        `pair`: open that color pair's archetype block and scroll to it."""
        code = self.set_code()
        if not code:
            return
        changed = False
        if code != ALL and fmt in FORMATS and fmt != self.fmt():
            r = self.fmt_bo1 if fmt == "bo1" else self.fmt_bo3
            if r.isEnabled():
                r.setChecked(True)
                self._fmt_user[code] = fmt
                new_settings().setValue(f"report/fmt/{code}", fmt)
                changed = True
        self._pending_focus = dict(key=(ALL, "") if code == ALL else (code, self.fmt()), section=section, detail=detail, pair=pair)
        if changed:
            self.load()                                              # (render() applies the jump once that report is shown)
        QTimer.singleShot(0, self._apply_pending_focus)

    def _apply_pending_focus(self) -> None:
        p = self._pending_focus
        if p is None:
            return
        shown = self._loaded_key == p["key"] and (self.key() == p["key"] or p["key"][0] == ALL == self.set_code())
        if not shown:
            return                                                   # that report is not on screen yet (render() calls this again)
        self._pending_focus = None
        sec = self._secs.get(p["section"])
        if sec is None:
            return
        target: QWidget = sec
        if p.get("detail") in self._details:
            self._details[p["detail"]].set_expanded(True)
        hit = self._pair_widgets.get(p.get("pair") or "")
        if hit is not None:
            hit[1].setChecked(True)
            target = hit[0]

        def go() -> None:
            try:
                self._scroll_to(target)
                self._select_nav(sec.key)
                sec.flash(self.FLASH_MS)
                gen = self._user_scrolls
                QTimer.singleShot(SETTLE_MS, lambda: gen == self._user_scrolls and self._scroll_to(target))     # (the page may still be re-laid out)
            except RuntimeError:                                     # (the report was drawn again in between)
                pass

        QTimer.singleShot(0, go)

    def _on_fact_clicked(self, ids: list) -> None:
        """A sentence of the summary was clicked: jump to the section with its figures (the first fact id that has one)."""
        for fid in ids:
            t = fact_target(str(fid))
            if t is None:
                continue
            fmt, section, detail, pair = t
            if section == "xset":
                section = self._lesson_section(str(fid))
                if section is None:
                    continue
            elif self.set_code() == ALL:
                continue
            self.focus_section(section, fmt, detail, pair)
            return

    def _lesson_section(self, fid: str) -> str | None:
        """The cross-set section (`cross_L<n>`) of an XSET fact id."""
        for les in (self.cross or {}).get("lessons", []):
            for it in les.get("items", []):
                if any((t or {}).get("id") == fid for t in (it.get("levels") or {}).values()):
                    return f"cross_{les['id']}"
            if any(g.get("id") == fid for g in les.get("glance", [])):
                return f"cross_{les['id']}"
        return None

    def _scroll_to(self, w: QWidget, margin: int = 6) -> None:
        self.body_lay.activate()
        try:
            if self.summary is not None and self.summary.isVisibleTo(self.body):                    # (the summary text gets its height once it has its width)
                self.summary.fit()
                self.body_lay.activate()
        except RuntimeError:                                           # (no summary on screen)
            pass
        y = w.mapTo(self.body, QPoint(0, 0)).y()
        self._programmatic = True
        try:
            self.scroll.verticalScrollBar().setValue(max(0, y - margin))
        finally:
            self._programmatic = False

    def _on_nav_clicked(self, item: QListWidgetItem) -> None:
        key = str(item.data(Qt.ItemDataRole.UserRole))
        sec = self._secs.get(key)
        if sec is not None:
            self._scroll_to(sec)
            self._select_nav(key)

    def _select_nav(self, key: str) -> None:
        self.nav.blockSignals(True)
        for r in range(self.nav.count()):
            if self.nav.item(r).data(Qt.ItemDataRole.UserRole) == key:
                self.nav.setCurrentRow(r)
                break
        self.nav.blockSignals(False)

    def _on_scrolled(self, _v: int) -> None:
        self.hover_popup.hide()
        if self._programmatic or not self._secs:
            return
        bar = self.scroll.verticalScrollBar()
        cur = next(iter(self._secs))
        if bar.maximum() > 0 and bar.value() >= bar.maximum() - 2:
            cur = list(self._secs)[-1]                               # (the last sections are too short to ever reach the top)
        else:
            for k, sec in self._secs.items():
                if sec.mapTo(self.body, QPoint(0, 0)).y() <= bar.value() + NAV_LINE:
                    cur = k
        self._select_nav(cur)

    def _fill_nav(self, secs: list) -> None:
        self._secs = {s_.key: s_ for s_ in secs if s_.key}
        self.nav.blockSignals(True)
        self.nav.clear()
        for s_ in self._secs.values():
            it = QListWidgetItem(s_.short)
            it.setData(Qt.ItemDataRole.UserRole, s_.key)
            it.setToolTip(s_.title)
            self.nav.addItem(it)
        self.nav.blockSignals(False)
        self.nav.setVisible(bool(self._secs))
        if self._secs:
            self._select_nav(next(iter(self._secs)))
            QTimer.singleShot(0, lambda: self._on_scrolled(0))

    def _on_format_clicked(self, f: str) -> None:
        sc = self.set_code()
        if sc:
            self._fmt_user[sc] = f
            new_settings().setValue(f"report/fmt/{sc}", f)
        self.load()

    # ------------------------------------------------------------------ loading
    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self.load)
        QTimer.singleShot(0, self._refresh_summary)

    def _refresh_summary(self) -> None:
        """The written report may have been installed (or replaced) since the page was drawn: read it again when the tab is shown."""
        try:
            if self.summary is not None and self.set_code() and self._loaded_key is not None:
                self.summary.load(self.set_code())
        except RuntimeError:                                         # (the page was drawn again in between)
            pass

    def load(self, refresh: bool = False) -> None:
        """Show the report of the chosen set and format: from memory, else ask the backend in the background (its disk cache makes
        that instant after the first time)."""
        k = self.key()
        if k is None or k[0] in self.calculating:
            return
        if k[0] == ALL:
            self._load_cross()
            return
        if not refresh and k in self.reports:
            self.current = self.reports[k]
            if self._loaded_key != k:
                self._loaded_key = k
                self.render()
            return
        if self.task.running:
            return                                           # `_on_ended` starts the wanted one
        if refresh:
            self.reports.pop(k, None)
        sc, fm = k
        self._task_key = k
        self.task.run(lambda progress, cancel: self.api.set_report(sc, fm, refresh=refresh, progress=progress, cancel=cancel),
                      label=f"report {sc} {fm}")

    def _load_cross(self) -> None:
        if self.cross is not None:
            if self._loaded_key != (ALL, ""):
                self.render_cross()
            return
        if self.task.running:
            return                                           # `_on_ended` starts the wanted one
        self._task_key = (ALL, "")
        self.task.run(lambda progress, cancel: self.api.cross_set(progress=progress), label="cross-set lessons")

    def render_cross(self) -> None:
        """The "All sets" view: one section per lesson (gui/cross_view.py)."""
        if self.cross is None:
            return
        self._rendering = True
        try:
            self.current = None
            self._loaded_key = (ALL, "")
            self._clear_sections()
            self._pair_widgets = {}
            self.message.setVisible(False)
            secs: list[Section] = [self._summary_section(hint=False)]
            for w in cross_view.build_sections(self.cross, self.font(), Section):
                w.key = w.objectName()
                w.short = "All sets" if w.key == "cross_intro" else w.title.replace("  ", " ")
                secs.append(w)
            for w in secs:
                self.body_lay.addWidget(w)
            self.body_lay.addStretch(1)
            self._fill_nav(secs)
            self.scroll.setMinimumWidth(0)
            QTimer.singleShot(0, self._apply_pending_focus)
            QTimer.singleShot(0, self._update_controls)
        finally:
            self._rendering = False

    def _on_started(self) -> None:
        self.strip.begin("Reading the report…")
        self._update_controls()

    def _on_result(self, rep: object) -> None:
        if not isinstance(rep, dict):
            return
        if "lessons" in rep:
            self.cross = rep
            self.strip.end("Ready")
            return
        k = (rep["set_code"], rep["fmt"])
        self.reports[k] = rep
        self.strip.end("Report ready" if not rep.get("cached") else "Ready")

    def _on_error(self, tb: str) -> None:
        if "SetBusy" in tb:
            self.strip.end("Waiting for the calculation")
            return
        self.strip.end("Failed")
        failed = self._task_key
        if failed is None or failed == self.key() or (failed == (ALL, "") and self.set_code() == ALL):
            self._show_message("The report could not be computed (see the error message).")
        self.errorOccurred.emit(short_error(tb), tb)

    def _on_ended(self) -> None:
        self._update_controls()
        k = self.key()
        if k is not None and k[0] == ALL:
            if self.cross is not None and self._loaded_key != (ALL, "") and self.isVisible():
                self.render_cross()
            return
        if k is not None and k in self.reports and self._loaded_key != k:
            self.load()
        elif k is not None and k not in self.reports and k[0] not in self.calculating and self.isVisible() and self._loaded_key != k:
            if not self.message.text().startswith("The report could not"):
                self.load()

    # ------------------------------------------------------------------ cheat sheet
    def start_cheat_sheet(self, code: str, fmt: str, path: str) -> bool:
        """Make the cheat sheet page of `code` and `fmt` in the background and save it as `path` (the Cheat sheet button asks for the path)."""
        if self.cheat_task.running:
            return False
        self._cheat_path = path
        api = self.api
        ok = self.cheat_task.run(lambda progress, cancel: api.cheat_sheet(code, fmt), label=f"cheat sheet {code} {fmt}")
        self._update_controls()
        return ok

    def _on_cheat_result(self, page: object) -> None:
        from pathlib import Path

        path = Path(self._cheat_path)
        try:
            path.write_text(str(page), encoding="utf-8")
        except OSError as e:
            self.errorOccurred.emit(f"Could not save the cheat sheet: {e}", "")
            return
        msg = f"Saved the cheat sheet to {path}"
        self.logMessage.emit(msg)
        self.statusMessage.emit(msg)
        export_ui.open_file(str(path))

    def _on_cheat_error(self, tb: str) -> None:
        if "SetBusy" in tb:
            self.errorOccurred.emit("The set is being recalculated: try the cheat sheet again when it has finished.", tb)
            return
        self.errorOccurred.emit(short_error(tb), tb)

    def wait(self, ms: int = 8000) -> None:
        self.task.wait(ms)
        self.pool_task.wait(ms)
        self.cheat_task.wait(ms)

    # ------------------------------------------------------------------ rendering
    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and not self._rendering:
            if self.current is not None:
                QTimer.singleShot(0, self.render)            # badges / bars / the summary's links carry theme colors
            elif self._loaded_key == (ALL, ""):
                QTimer.singleShot(0, self.render_cross)

    def _clear_sections(self) -> None:
        """Remove every section (the message label stays as the first item)."""
        self._secs, self._details = {}, {}
        self._card_links, self._pool_sec, self.pool_hint, self.explorer = [], None, None, None
        self.nav.clear()
        self.nav.setVisible(False)
        while self.body_lay.count() > 1:
            it = self.body_lay.takeAt(1)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.hide()                                         # (a queued "show if not hidden" from the layout must not show it as a window)
                w.deleteLater()

    def render(self) -> None:
        """Draw the report on screen (from `self.current`), then what depends on the pool of the Deck builder."""
        self._render_report()
        self._pool_changed()
        self._update_controls()

    def _render_report(self) -> None:
        rep = self.current
        if rep is None:
            return
        self._owned = self._owned_names()
        self._rendering = True
        try:
            self._clear_sections()
            self._pair_widgets = {}
            if rep.get("too_few"):
                self.message.setText("Not enough games in this format yet for a report.")
                self.message.setVisible(True)
                summary = self._summary_section()
                self.body_lay.addWidget(summary)
                self.body_lay.addStretch(1)
                self._fill_nav([summary])
                return
            self.message.setVisible(False)
            summary = self._summary_section()
            parts = [summary, self._intro(rep), self._glance(rep), self._pairs(rep), self._cards(rep)]
            if rep.get("archetypes", {}).get("pairs"):
                parts.append(self._archetypes(rep))
            if rep.get("shape", {}).get("rows"):
                parts.append(self._shape(rep))
            if rep.get("evolution"):
                parts.append(self._evolution(rep))
            if rep.get("compare"):
                parts.append(self._compare(rep))
            if rep.get("data_notes"):
                parts.append(self._data_notes(rep))
            for w in parts:
                self.body_lay.addWidget(w)
            self.body_lay.addStretch(1)
            self._fill_nav([w for w in parts if isinstance(w, Section)])
            QTimer.singleShot(0, self._fit_min_width)           # (the rows' minimum sizes are only known once they are laid out)
            QTimer.singleShot(0, self._apply_pending_focus)
        finally:
            self._rendering = False

    def _fit_min_width(self) -> None:
        """The column never gets narrower than its rows need (a splitter drag would otherwise squash the grids): scroll bar + frame included."""
        if self.current is None or self.message.isVisible():
            return
        self.body_lay.activate()
        self.scroll.setMinimumWidth(self.body.minimumSizeHint().width() + self.scroll.verticalScrollBar().sizeHint().width()
                                    + 2 * self.scroll.frameWidth() + 6)

    def _summary_section(self, hint: bool = True) -> Section:
        """The written report as the first section (it reads `api.written_report`; the text covers both formats).  `hint`: the line under it that
        asks for a pool (shown while the report on screen has none of the Deck builder's pool)."""
        sec = Section("Summary", key="summary", short="Summary")
        panel = SummaryPanel(self.api)
        panel.factClicked.connect(self._on_fact_clicked)
        panel.cardHovered.connect(self._on_card_hovered)
        panel.errorOccurred.connect(self.errorOccurred)
        self.summary = panel
        sec.lay.addWidget(panel)
        panel.set_owned(self._owned, render=False)
        panel.load(self.set_code())
        self.pool_hint = None
        if hint:
            self.pool_hint = muted(QLabel(POOL_HINT))
            self.pool_hint.setObjectName("poolHint")
            self.pool_hint.setWordWrap(True)
            self.pool_hint.setFont(sized_font(self.font(), -1))
            sec.lay.addWidget(self.pool_hint)
        return sec

    # ------------------------------------------------------------------ "For your pool"
    def set_pool(self, code: str, pool: dict) -> None:
        """The Deck builder's pool changed ({card_id: copies} of a set; ("", {}) = none): show or drop the "For your pool" section and the
        marks of the cards you own, live (the report itself is not recomputed)."""
        self._pool = ((code or "").upper(), {int(k): int(v) for k, v in (pool or {}).items() if int(v) > 0})
        self._pool_changed()

    def set_best_deck(self, code: str, fmt: str, colors: str) -> None:
        """The builder's best deck from the pool (main colors, e.g. "WU"); ("", "", "") = none."""
        self._best = ((code or "").upper(), fmt or "", colors or "")
        self._apply_pool()

    def _pool_active(self) -> bool:
        """The shown report is of the pool's set and there is a pool."""
        code = self._pool[0]
        return bool(code and self._pool[1]) and code != ALL and self.current is not None and self._loaded_key == (code, self.fmt()) \
            and not self.current.get("too_few")

    def _pool_changed(self) -> None:
        self._ensure_pool_data()
        self._apply_pool()

    def _ensure_pool_data(self) -> None:
        """The card table and card values the pool section needs (background: the values load the model)."""
        if not self._pool_active() or self._loaded_key is None:
            return
        sc, fmt = self._loaded_key
        need_info, need_vals = sc not in self._pool_info, (sc, fmt) not in self._pool_vals
        if not (need_info or need_vals) or self.pool_task.running:
            return
        req = (sc, str(self.built.get(sc) or ""))
        if need_info and req == self._pool_failed:
            return                                               # card_info failed for this model: no endless retry
        self._pool_req = req
        api = self.api

        def job(progress, cancel):
            info = dict(api.card_info(sc)) if need_info else None
            vals = None
            if need_vals:
                try:
                    vals = dict(api.card_values(sc, fmt) or {})
                except Exception:  # noqa: BLE001 - the section then shows what it can without values
                    log.exception("card values of %s %s", sc, fmt)
                    vals = {}
            return sc, fmt, info, vals

        self.pool_task.run(job, label=f"pool values {sc} {fmt}")

    def _on_pool_data(self, payload: object) -> None:
        sc, fmt, info, vals = payload  # type: ignore[misc]
        if info is not None:
            self._pool_info[sc] = info
        if vals is not None:
            self._pool_vals[(sc, fmt)] = vals
        self._apply_pool()

    def _on_pool_error(self, tb: str) -> None:
        self._pool_failed = self._pool_req
        log.error("pool data failed\n%s", tb)

    def _owned_names(self) -> dict[str, int]:
        """{card name: copies} of the pool for the shown report ({} when it does not apply)."""
        if not self._pool_active() or self._loaded_key is None:
            return {}
        info = self._pool_info.get(self._loaded_key[0]) or {}
        out: dict[str, int] = {}
        for cid, n in self._pool[1].items():
            d = info.get(cid)
            if d is not None and not d.get("is_basic"):
                out[d["name"]] = out.get(d["name"], 0) + n
        return out

    def _apply_pool(self) -> None:
        """Marks on the cards you own, the "For your pool" section and the hint under the Summary, from the current state (cheap)."""
        if self._rendering or self.current is None or self._loaded_key is None or self._loaded_key[0] == ALL:
            return
        anchor = self._scroll_anchor()          # the stars rewrap text above the reader too: keep the section they look at in place
        active = self._pool_active()
        self._owned = self._owned_names()
        for link in list(self._card_links):
            try:
                link.set_owned(self._owned.get(link.name, 0))
            except RuntimeError:                                        # (deleted with its section)
                self._card_links.remove(link)
        for part in (getattr(self, "explorer", None), self.summary):
            try:
                if part is not None:
                    part.set_owned(self._owned)
            except RuntimeError:
                pass
        info = self._pool_info.get(self._loaded_key[0]) if active else None
        vals = self._pool_vals.get(self._loaded_key) if active else None
        self._set_pool_section(self._pool_section(self.current, info, vals) if (info is not None and vals is not None) else None)
        if self.pool_hint is not None:
            try:
                self.pool_hint.setVisible(not active)
            except RuntimeError:
                self.pool_hint = None
        self._restore_anchor(anchor)

    def _scroll_anchor(self) -> "tuple[QWidget, int] | None":
        """(the first section whose bottom is below the top of the view, its offset from the view's top); None at the top of the page.
        The "For your pool" section is never the anchor (it is replaced)."""
        v = self.scroll.verticalScrollBar().value()
        if v <= 0:
            return None
        for i in range(self.body_lay.count()):
            w = self.body_lay.itemAt(i).widget()
            if w is None or w is self._pool_sec or not w.isVisible():
                continue
            if w.y() + w.height() > v:
                return w, w.y() - v
        return None

    def _restore_anchor(self, anchor: "tuple[QWidget, int] | None") -> None:
        """Scroll so the anchor section is where it was on screen; again once the posted relayouts of rewrapped labels have run."""
        if anchor is None:
            return
        w, off = anchor

        def apply() -> None:
            try:
                self.body_lay.activate()
                y = w.y()
            except RuntimeError:                                        # (deleted meanwhile)
                return
            self._programmatic = True
            try:
                self.scroll.verticalScrollBar().setValue(max(0, y - off))
            finally:
                self._programmatic = False

        apply()
        QTimer.singleShot(0, apply)
        QTimer.singleShot(40, apply)

    def _set_pool_section(self, sec: "Section | None") -> None:
        """Replace the "For your pool" section (right after the Summary) by `sec` (None: remove it), keeping the nav and the scroll place."""
        old = self._pool_sec
        summary = self._secs.get("summary")
        if summary is None or (old is None and sec is None):
            return
        bar = self.scroll.verticalScrollBar()
        keep = bar.value()
        shift = 0
        if old is not None:
            shift -= old.height() + self.body_lay.spacing()
            self.body_lay.removeWidget(old)
            old.setParent(None)
            old.hide()
            old.deleteLater()
        self._pool_sec = sec
        if sec is not None:
            self.body_lay.insertWidget(self.body_lay.indexOf(summary) + 1, sec)
            self.body_lay.activate()
            h = sec.sizeHint().height()
            if sec.hasHeightForWidth() and summary.width() > 0:      # wrapped text: the height at the page's real width
                h = max(h, sec.heightForWidth(summary.width()), sec.minimumSizeHint().height())
            shift += h + self.body_lay.spacing()
        row = self.nav.currentRow()
        cur = str(self.nav.item(row).data(Qt.ItemDataRole.UserRole)) if row >= 0 else ""
        secs = [self.body_lay.itemAt(i).widget() for i in range(self.body_lay.count())]
        self._fill_nav([w for w in secs if isinstance(w, Section)])
        if cur in self._secs:
            self._select_nav(cur)
        if keep > 0:
            self.body_lay.activate()
            self._programmatic = True
            try:
                bar.setValue(max(0, keep + shift))
            finally:
                self._programmatic = False
        QTimer.singleShot(0, self._fit_min_width)

    def _pool_rows(self, info: dict, vals: dict) -> list[dict]:
        """The non-land, non-basic cards of the pool: {cid, name, copies, colors, rarity, pts (points per copy, None without a value), low}."""
        out = []
        for cid, n in self._pool[1].items():
            d = info.get(cid)
            if d is None or d.get("is_land") or d.get("is_basic"):
                continue
            v = vals.get(cid)
            out.append(dict(cid=cid, name=d["name"], copies=int(n), colors="".join(c for c in "WUBRG" if c in (d.get("colors") or "")),
                            rarity=d.get("rarity") or "", pts=None if v is None else float(v["beta"]) * POINTS_PER_LOGIT, low=bool(v and v.get("low_data"))))
        return out

    @staticmethod
    def pair_support(rows: list[dict]) -> list[dict]:
        """For each color pair: {key, n (cards of the pool castable in it: colors within the pair, colorless included), value (points: the sum of
        the per-copy values of the best DECK_SPELLS of them)}; best value first."""
        res = []
        for pair in PAIRS:
            ok = [r for r in rows if set(r["colors"]) <= set(pair)]
            vs = sorted((r["pts"] for r in ok if r["pts"] is not None for _ in range(r["copies"])), reverse=True)[:DECK_SPELLS]
            res.append(dict(key=pair, n=sum(r["copies"] for r in ok), value=float(sum(vs))))
        res.sort(key=lambda p: (-p["value"], -p["n"], p["key"]))
        return res

    def _best_pair_label(self, rep: dict, labels: dict[str, str]) -> str:
        code, fmt, colors = self._best
        if not colors or code != rep["set_code"]:
            return ""
        key = colors[:2]
        name = labels.get(key) or "-".join(R_COLOR_NAMES.get(c, c) for c in colors)
        return name + ("" if fmt == rep["fmt"] or not fmt else f" (in {'Best of 1' if fmt == 'bo1' else 'Best of 3'})")

    def _pool_card_row(self, name: str, copies: int, tag: str, value: float | None, badge: str = "", rng: str = "", tip: str = "") -> QWidget:
        w = QWidget()
        w.setMaximumWidth(POOL_ROW_WIDTH)
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 1, 0, 1)
        h.setSpacing(6)
        link = CardLink(name, f"<b>{name}</b>" + (f"<br>{tip}" if tip else ""))
        link.shown.connect(self._show_card)
        link.left.connect(self.hover_popup.hide)
        link.setToolTip(tip)
        h.addWidget(link, 1)
        if copies > 1:
            h.addWidget(muted(QLabel(f"×{copies}")))
        if tag:
            h.addWidget(muted(QLabel(tag)))
        if value is not None:
            h.addWidget(self._num_label(pts(value), True, "good" if value >= 0 else "bad"))
        if badge:
            h.addWidget(Badge(badge, f"{name}: {rng}."))
        return w

    def _pool_section(self, rep: dict, info: dict, vals: dict) -> Section:
        """"For your pool": the color pairs the pool supports, your cards named in this report and your best cards (per-copy values in points: the
        model's log-odds effect x 25, the same conversion as the report's card values)."""
        sec = Section("For your pool", key="pool", short="Your pool")
        rows = self._pool_rows(info, vals)
        n_spells = sum(r["copies"] for r in rows)
        sec.sentence(f"{n_spells} non-land cards of your pool are in this set's card table.  Everything below comes from the figures of this report and the "
                     f"model's card values; it changes when you change the pool in the Deck builder.", bullet=False)
        labels = {r["key"]: r["label"] for r in rep["pairs"]["rows"] if len(r.get("key", "")) == 2}
        set_rows = {r["key"]: r for r in rep["pairs"]["rows"] if len(r.get("key", "")) == 2 and not r.get("hidden")}
        support = self.pair_support(rows)
        # a. color pairs
        h = QLabel("<b>Color pairs your pool supports</b>")
        h.setTextFormat(Qt.TextFormat.RichText)
        sec.lay.addWidget(h)
        best = self._best_pair_label(rep, labels)
        if best:
            lab = QLabel(f"The builder's best deck from this pool is <b>{best}</b>.")
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setObjectName("poolBest")
            sec.lay.addWidget(lab)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        self._grid_header(grid, [("Colors", ""), ("Playable cards", "Non-land cards of your pool that can be cast with only these two colors (colorless cards included)"),
                                 (f"Best {DECK_SPELLS} vs average cards", f"The per-copy values of your {DECK_SPELLS} best cards in these colors, added up: "
                                  f"0 = as good as {DECK_SPELLS} average cards of the set.  Compare pairs with it; a negative total is common.  " + POINTS_TIP),
                                 ("Whole set (points)", "How decks of this pair do against an average deck in the whole set.  " + POINTS_TIP), ("How sure?", BADGE_TIP)],
                          right=(1, 2, 3))
        shown = support[:POOL_PAIRS]
        for i, p in enumerate(shown, start=1):
            r = set_rows.get(p["key"])
            cell = QWidget()
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(6)
            pips = ColorPips()
            pips.set_colors(p["key"], "")
            cl.addWidget(pips)
            cl.addWidget(ElidedLabel(labels.get(p["key"], p["key"]), min_chars=8), 1)
            grid.addWidget(cell, i, 0)
            grid.addWidget(self._num_label(str(p["n"])), i, 1)
            grid.addWidget(self._num_label(pts(p["value"]), True, "good" if p["value"] >= 0 else "bad"), i, 2)
            if r is not None:
                grid.addWidget(self._num_label(pts(r["effect_pp"])), i, 3)
                grid.addWidget(Badge(r["badge"], f"{r['label']}: {r['range']}."), i, 4)
            else:
                grid.addWidget(self._num_label("–"), i, 3)
        sec.lay.addWidget(wrap_widget(grid))
        sec.note(f"Best {DECK_SPELLS}: the per-copy values (points of win rate) of the best {DECK_SPELLS} playable cards you have in these colors, added up, "
                 f"where 0 = as good as {DECK_SPELLS} average cards of the set; use it to compare the pairs.  A card counts for a pair when all its colors "
                 "are in the pair.  The whole-set result is the pair's win rate against an average deck.")
        # b. your cards in this report
        cs = rep["cards"]
        trap_ids = {r["card_id"] for r in cs.get("traps") or []}
        good = [(r, "Underrated") for r in cs.get("underrated") or [] if r["card_id"] in self._pool[1]]
        watch = [(r, "Trap rare") for r in cs.get("traps") or [] if r["card_id"] in self._pool[1]] + \
                [(r, "Overrated") for r in cs.get("overrated") or [] if r["card_id"] in self._pool[1] and r["card_id"] not in trap_ids]
        h = QLabel("<b>Your cards in this report</b>")
        h.setTextFormat(Qt.TextFormat.RichText)
        sec.lay.addWidget(h)
        if not good and not watch:
            sec.lay.addWidget(muted(QLabel("None of your cards is in the Underrated, Overrated or Trap-rare lists of this report.")))
        for title, items in (("Good news: underrated cards you own", good), ("Watch out: cards players love that lose", watch)):
            if not items:
                continue
            t = muted(QLabel(title))
            t.setObjectName("poolGroupTitle")
            sec.lay.addWidget(t)
            for r, tag in items:
                sec.lay.addWidget(self._pool_card_row(r["name"], self._pool[1].get(r["card_id"], 1), tag, r["value_pp"], r["badge"], r["range"], r["reason"]))
        # c. best cards
        valued = [r for r in rows if r["pts"] is not None and not r["low"]]
        valued.sort(key=lambda r: (-r["pts"], r["name"]))
        h = QLabel("<b>Your best cards</b>")
        h.setTextFormat(Qt.TextFormat.RichText)
        sec.lay.addWidget(h)
        if not valued:
            sec.lay.addWidget(muted(QLabel("No card values are available for your pool yet.")))
        for r in valued[:POOL_BEST]:
            row = QWidget()
            row.setMaximumWidth(POOL_ROW_WIDTH)
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 1, 0, 1)
            hl.setSpacing(6)
            pips = ColorPips()
            pips.set_colors(r["colors"], "")
            hl.addWidget(pips)
            link = CardLink(r["name"], f"<b>{r['name']}</b><br>{pts(r['pts'])} points per copy (model estimate).")
            link.shown.connect(self._show_card)
            link.left.connect(self.hover_popup.hide)
            hl.addWidget(link, 1)
            if r["copies"] > 1:
                hl.addWidget(muted(QLabel(f"×{r['copies']}")))
            hl.addWidget(muted(QLabel(r["rarity"])))
            val = self._num_label(pts(r["pts"]), True, "good" if r["pts"] >= 0 else "bad")
            val.setToolTip(f"{pts(r['pts'])} points of win rate per copy, from the model.  " + POINTS_TIP)
            hl.addWidget(val)
            sec.lay.addWidget(row)
        n_low = sum(1 for r in rows if r["low"])
        if n_low:
            sec.note(f"{n_low} card{'s' if n_low != 1 else ''} of your pool with little data in this format {'are' if n_low != 1 else 'is'} left out: their value would mostly be a guess.")
        self.pool_summary = dict(pairs=support, good=[r["name"] for r, _t in good], watch=[r["name"] for r, _t in watch],
                                 best=[r["name"] for r in valued[:POOL_BEST]], best_pair=best)
        return sec

    def _on_card_hovered(self, name: str) -> None:
        if name:
            self._show_card(name, f"<b>{name}</b>")
        else:
            self.hover_popup.hide()

    def _intro(self, rep: dict) -> QWidget:
        gl = rep["glance"]
        lab = muted(QLabel(
            f"Based on {gl['n_decks']:,} real {'Best of 1' if rep['fmt'] == 'bo1' else 'Best of 3'} Sealed decks "
            f"({gl['n_games']:,} games) on 17lands.  Win rates are always compared between players of the same skill, and given in "
            f"<b>points</b> (one point = one percentage point of the chance to win a game).  Each estimate carries a badge: Strong, Likely or Weak."))
        lab.setWordWrap(True)
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setToolTip(POINTS_TIP)
        return lab

    # --- section 1
    def _glance(self, rep: dict) -> QWidget:
        gl = rep["glance"]
        sec = Section("The set at a glance", key="glance", short="Set at a glance")
        row = FlowLayout(spacing=8)
        host = QWidget()
        host.setLayout(row)
        if gl.get("median_turns") is not None:
            c = MetricCard("Typical game length")
            c.set(f"{gl['median_turns']:.0f} turns", "median, as counted by 17lands")
            row.addWidget(c)
        op = gl.get("on_play")
        if op:
            c = MetricCard("Playing first")
            c.set(pts(op["diff_pp"]) + " pts", f"{op['wr_play'] * 100:.1f}% on the play, {op['wr_draw'] * 100:.1f}% on the draw",
                  tip=H.tip("Playing first", "Win rate on the play against on the draw, among first games of players of the same skill. "
                            + op["range"] + "."))
            row.addWidget(c)
        if gl.get("mulligan_rate") is not None:
            c = MetricCard("Games with a mulligan")
            c.set(f"{gl['mulligan_rate'] * 100:.0f}%", "of games start with a mulligan")
            row.addWidget(c)
        c = MetricCard("Decks studied")
        c.set(f"{gl['n_decks']:,}", f"{gl['n_games']:,} games")
        row.addWidget(c)
        sec.lay.addWidget(host)
        for t in gl.get("text", []):
            sec.sentence(t)
        sec.note(rep["notes"]["glance"])
        return sec

    # --- section 2
    def _dim(self, w: QWidget) -> None:
        eff = QGraphicsOpacityEffect(w)
        eff.setOpacity(0.5)
        w.setGraphicsEffect(eff)

    def _fold_weak(self, key: str, rows: list[list[QWidget]], badges: list[str], effects: list[float]) -> QToolButton | None:
        """Dim the Weak rows of a table and fold them under a "Show N more with no clear difference" button (Strong and Likely rows always show).
        When every row is Weak the FOLD_TOP biggest effects stay visible.  `rows[i]` = the widgets of row i.  None when there is nothing to fold."""
        weak = [i for i, b in enumerate(badges) if b == "weak"]
        if not weak:
            return None
        for i in weak:
            for w in rows[i]:
                self._dim(w)
        if len(weak) == len(badges):
            keep = set(sorted(range(len(rows)), key=lambda i: -effects[i])[:FOLD_TOP])
            fold = [i for i in weak if i not in keep]
        else:
            fold = weak
        if not fold:
            return None
        btn = QToolButton()
        btn.setCheckable(True)
        btn.setAutoRaise(True)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setStyleSheet("QToolButton { border: none; }")
        btn.setObjectName("foldWeak")
        btn.setProperty("fold_rows", len(fold))
        btn.setToolTip("Results that could easily be chance (badge Weak): the bars are pale because there is no clear difference from an average deck")

        def apply(on: bool) -> None:
            btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
            btn.setText(f"Hide the {len(fold)} with no clear difference" if on else f"Show {len(fold)} more with no clear difference")
            for i in fold:
                for w in rows[i]:
                    w.setVisible(on)
            (self._fold_open.add if on else self._fold_open.discard)(key)

        btn.toggled.connect(apply)
        btn.setChecked(key in self._fold_open)
        apply(key in self._fold_open)
        return btn

    def _pairs(self, rep: dict) -> QWidget:
        pr = rep["pairs"]
        sec = Section("Color pairs: which decks win?", key="pairs", short="Color pairs")
        for t in pr.get("text", []):
            sec.sentence(t)
        rows = sorted([r for r in pr.get("rows", []) if not r["hidden"]], key=lambda r: -r["effect_pp"])
        if rows:
            scale = max(1.5, max(max(abs(r["effect_pp"]), abs(r["effect_pp"]) + 1.96 * r["se_pp"]) for r in rows if r["se_pp"] < 3) * 1.05)
            grid = QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(4)
            grid.setColumnStretch(2, 1)
            for c, (text, tip) in enumerate((("Colors", ""), ("Share of decks", "How many of the decks are built in these colors"),
                                              ("Win rate against an average deck (points)", POINTS_TIP + ""),
                                              ("", ""), ("How sure?", BADGE_TIP), ("Splash", "How often these decks also splash a third color"))):
                h = muted(QLabel(text))
                h.setFont(sized_font(self.font(), -1))
                h.setToolTip(tip)
                if c in (1, 5):
                    h.setAlignment(Qt.AlignmentFlag.AlignRight)
                grid.addWidget(h, 0, c)
            row_ws: list[list[QWidget]] = []
            for i, r in enumerate(rows, start=1):
                cell = QWidget()
                cl = QHBoxLayout(cell)
                cl.setContentsMargins(0, 0, 0, 0)
                cl.setSpacing(6)
                pips = ColorPips()                                  # (rows without colors keep the same indent, so the names line up)
                pips.set_colors(r["colors"] or "WU", "")
                if not r["colors"]:
                    pips.setVisible(False)
                    cl.addSpacing(pips.width())
                cl.addWidget(pips)
                name = ElidedLabel(r["label"], min_chars=8)
                name.setToolTip(f"{r['n_decks']:,} decks, {r['n_games']:,} games")
                cl.addWidget(name, 1)
                grid.addWidget(cell, i, 0)
                share = QLabel(f"{r['share'] * 100:.1f}%")
                share.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                grid.addWidget(share, i, 1)
                lo, hi = r["effect_pp"] - 1.96 * r["se_pp"], r["effect_pp"] + 1.96 * r["se_pp"]
                bar = EffectBar(r["effect_pp"], lo, hi, scale, r["badge"])
                bar.setToolTip(f"{pts(r['effect_pp'])} points; {r['range']}")
                grid.addWidget(bar, i, 2)
                val = QLabel(pts(r["effect_pp"]))
                val.setMinimumWidth(self.fontMetrics().horizontalAdvance("−8.88") + 4)
                val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                grid.addWidget(val, i, 3)
                grid.addWidget(Badge(r["badge"], f"{r['label']}: {r['range']}."), i, 4)
                sp = QLabel(f"{r['splash_share'] * 100:.0f}%")
                sp.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                grid.addWidget(sp, i, 5)
                row_ws.append([cell, share, bar, val, grid.itemAtPosition(i, 4).widget(), sp])
            wrap = QWidget()
            wrap.setLayout(grid)
            sec.lay.addWidget(wrap)
            fold = self._fold_weak("pairs", row_ws, [r["badge"] for r in rows], [r["effect_pp"] for r in rows])
            if fold is not None:
                sec.lay.addWidget(fold, 0, Qt.AlignmentFlag.AlignLeft)
        hidden = [r for r in pr.get("rows", []) if r["hidden"]]
        if hidden:
            sec.note(f"Not shown (fewer than {pr.get('min_decks')} decks): " + ", ".join(r["label"] for r in hidden) + ".")
        sec.note(rep["notes"]["pairs"])
        if (rep.get("pair_details") or {}).get("rows"):
            sec.lay.addWidget(self._pair_details(rep))
        return sec

    # --- section 3
    def _card_row(self, r: dict, long_reason: bool) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(0)
        top = QHBoxLayout()
        top.setSpacing(6)
        caption = (f"<b>{r['name']}</b><br>{r['reason']}" + ("<br><i>Limited data in this format.</i>" if r.get("thin") else ""))
        link = CardLink(r["name"], caption)
        self._wire(link)
        link.setToolTip(r["reason"])
        top.addWidget(link, 1)
        val = QLabel(pts(r["value_pp"]))
        val.setToolTip(f"{pts(r['value_pp'])} points per copy against its color's average played card; {r['range']}.")
        col = semantic()["good" if r["value_pp"] >= 0 else "bad"]
        pal = val.palette()
        pal.setColor(QPalette.ColorRole.WindowText, col)
        val.setPalette(pal)
        val.setFont(sized_font(self.font(), 0, bold=True))
        top.addWidget(val)
        top.addWidget(Badge(r["badge"], f"{r['name']}: {r['range']}."))
        v.addLayout(top)
        if long_reason:
            sub = muted(QLabel(r["reason"] + ("  (Limited data in this format.)" if r.get("thin") else "")))
            sub.setWordWrap(True)
            sub.setFont(sized_font(self.font(), -1))
        else:
            sp_txt = (f" · splashed {r['splash_rate'] * 100:.0f}%"
                      if (r.get("n_splash_elig") or 0) >= 50 and (r.get("splash_rate") or 0.0) >= 0.03 else "")
            sub = muted(ElidedLabel(f"Played in {r['play_rate'] * 100:.0f}% of decks" + sp_txt + (" · limited data" if r.get("thin") else "")))
            sub.setFont(sized_font(self.font(), -1))
            sub.setToolTip(r["reason"])
        v.addWidget(sub)
        return w

    def _show_card(self, name: str, caption: str) -> None:
        self.hover_popup.show_card(self.set_code(), name, caption, QCursor.pos())

    def _wire(self, link: CardLink) -> CardLink:
        link.shown.connect(self._show_card)
        link.left.connect(self.hover_popup.hide)
        self._card_links.append(link)
        link.set_owned(self._owned.get(link.name, 0))
        return link

    def hideEvent(self, event) -> None:
        self.hover_popup.hide()
        super().hideEvent(event)

    def _cards(self, rep: dict) -> QWidget:
        cs = rep["cards"]
        sec = Section("Cards: where players misjudge", key="cards", short="Cards")
        sec.sentence(f"Out of {cs.get('n_cards', 0)} cards, {cs.get('n_shown', 0)} have enough data to judge: at least "
                     f"{cs.get('min_eligible')} decks that could have played them and {cs.get('min_exposure')} games with them in the deck.", bullet=False)
        if cs.get("note"):
            sec.note(cs["note"])
        for title, key, blurb in (
                ("Underrated: players leave them out, but they win", "underrated",
                 "Played less often than most cards, yet clearly better than the average card you would play in that color."),
                ("Overrated: players love them, but they lose", "overrated",
                 "Played in most decks that can, yet clearly worse than the average card you would play in that color.")):
            h = QLabel(f"<b>{title}</b>")
            h.setTextFormat(Qt.TextFormat.RichText)
            sec.lay.addWidget(h)
            b = muted(QLabel(blurb))
            b.setWordWrap(True)
            sec.lay.addWidget(b)
            by = cs.get(key + "_by") or ({"all": cs[key]} if cs.get(key) else {})
            if not by:
                sec.lay.addWidget(muted(QLabel("Nothing clear enough to list.")))
            else:
                sec.lay.addWidget(self._group_grid(by, True))
            extra = cs.get("n_" + key, 0) - sum(len(v) for v in by.values())
            if extra > 0:
                sec.note(f"{extra} more not shown (each color lists its 3 clearest, up to 5 when the extra ones are Strong).")
        traps = cs.get("traps") or []
        if traps:
            h = QLabel("<b>Trap rares: auto-included, but they lose</b>")
            h.setTextFormat(Qt.TextFormat.RichText)
            sec.lay.addWidget(h)
            b = muted(QLabel("Rares and mythics that most players put in the deck without a second thought, yet clearly worse than the average card you would play in that color."))
            b.setWordWrap(True)
            sec.lay.addWidget(b)
            for r in traps:
                sec.lay.addWidget(self._card_row(r, True))
        sp_good, sp_bad = cs.get("splashed") or [], cs.get("splash_traps") or []
        if sp_good or sp_bad:
            h = QLabel("<b>Often splashed: cards players stretch their mana for</b>")
            h.setTextFormat(Qt.TextFormat.RichText)
            sec.lay.addWidget(h)
            b = muted(QLabel("Among decks missing one of the card's colors, these are splashed in at least 8% of them. The value is the card's value per copy in all the decks that play it (mostly in its own colors), not a measure of how it does when splashed."))
            b.setWordWrap(True)
            sec.lay.addWidget(b)
            sgrid = QGridLayout()
            sgrid.setHorizontalSpacing(14)
            for col_i, (stitle, sitems) in enumerate((("Often splashed, high value", sp_good), ("Often splashed, low value", sp_bad))):
                panel = QFrame()
                panel.setFrameShape(QFrame.Shape.NoFrame)
                pl = QVBoxLayout(panel)
                pl.setContentsMargins(0, 0, 0, 0)
                pl.setSpacing(2)
                pl.addWidget(QLabel(f"<b>{stitle}</b>"))
                if not sitems:
                    pl.addWidget(muted(QLabel("None clear enough.")))
                for r in sitems:
                    pl.addWidget(self._card_row(r, True))
                pl.addStretch(1)
                sgrid.addWidget(panel, 0, col_i, Qt.AlignmentFlag.AlignTop)
                sgrid.setColumnStretch(col_i, 1)
            sec.lay.addWidget(wrap_widget(sgrid))
        for rar, title in (("common", "Best commons by color"), ("uncommon", "Best uncommons by color")):
            h = QLabel(f"<b>{title}</b>")
            h.setTextFormat(Qt.TextFormat.RichText)
            sec.lay.addWidget(h)
            sec.lay.addWidget(self._group_grid((cs.get("top") or {}).get(rar, {}), False))
        sec.note(rep["notes"]["cards"] + "  Values are points of win rate per copy, compared with the average card of the same color that decks play.")
        if (cs.get("explorer") or {}).get("rows"):
            sec.lay.addWidget(self._explorer(rep))
        return sec

    def _group_grid(self, by: dict, reason_first: bool) -> QWidget:
        """Cards in a 3-column grid of color groups (pips + title, then the card rows)."""
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        for i, gk in enumerate([g for g in GROUP_ORDER + ["all"] if g in by]):
            panel = QFrame()
            panel.setFrameShape(QFrame.Shape.NoFrame)
            pl = QVBoxLayout(panel)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.setSpacing(2)
            hd = QHBoxLayout()
            hd.setSpacing(6)
            if gk in "WUBRG" and len(gk) == 1:
                pips = ColorPips()
                pips.set_colors(gk, "")
                hd.addWidget(pips)
            hd.addWidget(QLabel(f"<b>{GROUP_TITLE.get(gk, '')}</b>"), 1)
            pl.addLayout(hd)
            for r in by[gk]:
                pl.addWidget(self._card_row(r, reason_first))
            pl.addStretch(1)
            grid.addWidget(panel, i // 3, i % 3, Qt.AlignmentFlag.AlignTop)
        wrap = QWidget()
        wrap.setLayout(grid)
        return wrap

    # --- section 4: archetypes
    def _chips(self, shape: dict, words: list[str] | None = None) -> QWidget:
        host = QWidget()
        row = FlowLayout(spacing=6)
        host.setLayout(row)
        for w in words or []:
            row.addWidget(Chip(w, "How this style differs from the other decks of its color pair", accent=True))
        tip = "Typical (median) deck of this group"
        row.addWidget(Chip(f"{shape['lands']:.0f} lands", tip))
        row.addWidget(Chip(f"{shape['creatures']:.0f} creatures", tip))
        row.addWidget(Chip(f"{shape['two_drops']:.0f} two-drops", tip + "; two-drops are creatures that cost 2"))
        row.addWidget(Chip(f"average mana value {shape['avg_mv']:.1f}", tip + "; mana value of the non-land cards"))
        row.addWidget(Chip(f"{shape['splash_share'] * 100:.0f}% splash", "Share of these decks that splash a third color"))
        return host

    def _signature(self, cards: list[dict], label: str, ref: str = "the other decks", prefix: str = "Signature cards:") -> QWidget | None:
        if not cards:
            return None
        host = QWidget()
        row = FlowLayout(spacing=10, clamp=True)
        host.setLayout(row)
        if prefix:
            row.addWidget(muted(QLabel(prefix)))
        for c in cards:
            tip = f"In {c['rate'] * 100:.0f}% of {label} decks, against {c['ref_rate'] * 100:.0f}% of {ref}."
            link = CardLink(c["name"], f"<b>{c['name']}</b><br>{tip}")
            self._wire(link)
            link.setToolTip(tip)
            row.addWidget(link)
        return host

    def _best_cards(self, cards: list[dict]) -> QWidget | None:
        if not cards:
            return None
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(1)
        h = muted(QLabel("Best cards in these decks (points of win rate per copy, against the average card of their color):"))
        h.setWordWrap(True)
        v.addWidget(h)
        for r in cards:
            row = QHBoxLayout()
            row.setSpacing(6)
            link = CardLink(r["name"], f"<b>{r['name']}</b><br>{r['reason']}")
            self._wire(link)
            link.setToolTip(r["reason"])
            row.addWidget(link, 1)
            val = QLabel(pts(r["value_pp"]))
            pal = val.palette()
            pal.setColor(QPalette.ColorRole.WindowText, semantic()["good" if r["value_pp"] >= 0 else "bad"])
            val.setPalette(pal)
            val.setFont(sized_font(self.font(), 0, bold=True))
            row.addWidget(val)
            row.addWidget(Badge(r["badge"], f"{r['name']}: {r['range']}."))
            v.addLayout(row)
        return host

    def _spectrum(self, a: dict) -> QWidget | None:
        """The style spectrum of a color pair: the two end styles on a bar, signature cards under each end, and the win-rate sentence."""
        sp = a.get("spectrum") or {}
        if not sp.get("shown"):
            return None
        ea, eb, win = sp["a"], sp["b"], sp["win"]
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(4)
        intro = QLabel(sp["text"][0])
        intro.setWordWrap(True)
        intro.setToolTip(H.tip("Style spectrum",
                               f"The main way the {a['key']} decks differ from each other, found in the cards and the shape of the decks "
                               f"(agreement between two independent halves of the data: {sp['corr']:.2f}; 1.0 = identical).",
                               "A spectrum is a gradual range with many decks in between, not two separate archetypes."))
        v.addWidget(intro)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for col, e, align in ((0, ea, Qt.AlignmentFlag.AlignLeft), (1, eb, Qt.AlignmentFlag.AlignRight)):
            lab = QLabel(f"<b>{e['label']}</b>")
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setWordWrap(True)
            lab.setAlignment(align | Qt.AlignmentFlag.AlignBottom)
            lab.setToolTip(H.tip(e["label"], f"{e['description'].capitalize()}.  The {e['n_decks']:,} decks at this end (a third of the {a['key']} decks).",
                                 f"{pts(e['effect_pp'])} points of win rate against an average deck ({e['range']}); {BADGE_TEXT.get(e['badge'], 'Weak').lower()}."))
            grid.addWidget(lab, 0, col)
        bar = SpectrumBar(f"From {ea['label']} (left) to {eb['label']} (right).")
        grid.addWidget(bar, 1, 0, 1, 2)
        for col, e, o, align in ((0, ea, eb, Qt.AlignmentFlag.AlignLeft), (1, eb, ea, Qt.AlignmentFlag.AlignRight)):
            w = self._signature(e["signature"], e["label"], ref=f"the {o['label']} decks", prefix="")
            if w is not None:
                w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)      # a half-width column may be narrower than its widest card name
                grid.addWidget(w, 2, col, Qt.AlignmentFlag.AlignTop)
        v.addLayout(grid)
        row = QHBoxLayout()
        row.setSpacing(8)
        t = QLabel(sp["text"][1])
        t.setWordWrap(True)
        row.addWidget(t, 1)
        halves = " and ".join(pts(h) if h is not None else "n/a" for h in win["halves"])
        row.addWidget(Badge(win["badge"] if win["finding"] else "weak",
                            f"Top third minus bottom third of the spectrum: {pts(win['effect_pp'])} points of win rate ({win['range']}), skill-adjusted.  "
                            f"Fitted separately in the two halves of the data: {halves}.  Shown as a finding only if it is Likely or Strong "
                            "and has the same direction in both halves."))
        v.addLayout(row)
        return host

    def _arch_body(self, box: QVBoxLayout, a: dict, label: str, words: list[str] | None = None) -> None:
        box.addWidget(self._chips(a["shape"], words))
        for w in (self._signature(a.get("signature", []), label), self._best_cards(a.get("best_cards", []))):
            if w is not None:
                box.addWidget(w)

    def _value_label(self, eff: float, rng: str) -> QLabel:
        val = QLabel(pts(eff) + " pts")
        val.setToolTip(f"{pts(eff)} points of win rate against an average deck; {rng}.")
        pal = val.palette()
        pal.setColor(QPalette.ColorRole.WindowText, semantic()["good" if eff >= 0 else "bad"])
        val.setPalette(pal)
        val.setFont(sized_font(self.font(), 0, bold=True))
        return val

    def _pair_block(self, a: dict) -> QWidget:
        wrap = QFrame()
        wrap.setFrameShape(QFrame.Shape.NoFrame)
        outer = QVBoxLayout(wrap)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(8)
        body = QWidget()
        toggle = QToolButton()
        toggle.setCheckable(True)
        toggle.setAutoRaise(True)
        toggle.setToolTip("Show or hide the details of this color pair")
        pips = ColorPips()
        pips.set_colors(a["key"], "")
        name = ElidedLabel(f"{a['code']}  {a['label']}", min_chars=8)
        f = name.font()
        f.setBold(True)
        name.setFont(f)
        name.setToolTip(f"{a['n_decks']:,} decks.  {a['splash_share'] * 100:.0f}% splash a third color"
                        + (f", most often {a['splash_top_label'].lower()}" if a.get("splash_top_label") else "")
                        + (f".  Written {a['code']}: " + ", ".join(f"{a['splash_col_shares'][c] * 100:.0f}% splash {c.lower()}" for c in a["splash_cols"])
                           if a.get("splash_cols") else "") + ".")
        subs = a["subs"]
        head.addWidget(toggle)
        head.addWidget(pips)
        head.addWidget(name, 1)
        if subs:
            head.addWidget(muted(QLabel(f"{len(subs)} styles")))
        head.addWidget(QLabel(f"{a['share'] * 100:.1f}% of decks"))
        head.addWidget(self._value_label(a["effect_pp"], a["range"]))
        head.addWidget(Badge(a["badge"], f"{a['label']}: {a['range']}."))
        outer.addLayout(head)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(30, 0, 0, 6)
        bl.setSpacing(6)
        for t in a.get("text", []):
            lab = QLabel("•  " + t)
            lab.setWordWrap(True)
            bl.addWidget(lab)
        spw = self._spectrum(a)
        if spw is not None:
            bl.addWidget(spw)
        self._arch_body(bl, a, a["key"])
        for sub in subs:
            panel = QFrame()
            panel.setFrameShape(QFrame.Shape.StyledPanel)
            panel.setProperty("card", True)
            pl = QVBoxLayout(panel)
            pl.setContentsMargins(10, 6, 10, 8)
            pl.setSpacing(5)
            hd = QHBoxLayout()
            hd.setSpacing(8)
            nm = QLabel(f"<b>{sub['name']}</b>")
            nm.setTextFormat(Qt.TextFormat.RichText)
            nm.setToolTip(H.tip("Found in the data", f"Found again in two independent halves of the events (agreement {a['split']['ari']:.2f}, 1.0 = identical groups).",
                                "Only styles that replicate this way are shown."))
            hd.addWidget(nm, 1)
            hd.addWidget(QLabel(f"{sub['share_of_pair'] * 100:.0f}% of {a['key']} decks"))
            if sub.get("effect_pp") is not None:
                hd.addWidget(self._value_label(sub["effect_pp"], sub["range"]))
                hd.addWidget(Badge(sub["badge"], f"{sub['name']}: {sub['range']}."))
            pl.addLayout(hd)
            t = QLabel(sub.get("text", ""))
            t.setWordWrap(True)
            pl.addWidget(t)
            self._arch_body(pl, sub, sub["name"], sub.get("words"))
            bl.addWidget(panel)

        def toggled(on: bool, key: str = a["key"]) -> None:
            toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
            body.setVisible(on)
            (self._open.add if on else self._open.discard)(key)

        toggle.toggled.connect(toggled)
        on = a["key"] in self._open
        toggle.setChecked(on)
        toggled(on)
        outer.addWidget(body)
        self._pair_widgets[a["key"]] = (wrap, toggle)
        return wrap

    def _archetypes(self, rep: dict) -> QWidget:
        ar = rep["archetypes"]
        sec = Section("Archetypes: how each color pair is played", key="archetypes", short="Archetypes")
        sec.sentence("Each block is a main-color pair, with its share of decks and its win rate against an average deck (players of the same skill). "
                     "A plus sign after the colors (UB+r) names a color that at least a quarter of its decks splash. Inside a block, the style spectrum "
                     "shows how the decks of that pair range between two styles (only when the same range turns up in two independent halves of the data).",
                     bullet=False)
        if ar.get("n_with_subs"):
            sec.sentence(f"{ar['n_with_subs']} of the pairs split into clearly different styles (shown inside the block). A style is only shown if the same "
                         "groups turn up again in two independent halves of the data.", bullet=False)
        else:
            sec.sentence("No color pair splits into clearly different styles in this format: the decks of a pair vary smoothly (for example, more removal "
                         "and a lower curve against more creatures and fliers) rather than falling into separate groups.", bullet=False)
        for a in ar["pairs"]:
            sec.lay.addWidget(self._pair_block(a))
        sec.note(ar.get("note") or rep["notes"]["archetypes"])
        return sec

    # --- section 5: winning deck shape
    def _shape(self, rep: dict) -> QWidget:
        sh = rep["shape"]
        sec = Section("Winning deck shape: what decks look like", key="shape", short="Deck shape")
        for t in sh.get("text", []):
            sec.sentence(t)
        metrics = sh["metrics"]
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        for c, text in enumerate(["Colors", "Decks"] + [m["label"] for m in metrics]):
            h = muted(QLabel(text))
            h.setFont(sized_font(self.font(), -1))
            if c >= 2:
                h.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(h, 0, c)
        r_ = 1
        for row in sh["rows"]:
            nm = QLabel(f"<b>{row['label']}</b>")
            nm.setTextFormat(Qt.TextFormat.RichText)
            grid.addWidget(nm, r_, 0)
            nd = QLabel(f"{row['n_decks']:,}")
            nd.setToolTip(f"{row['n_decks']:,} decks, {row['n_strong']:,} of them with strong results ({sh['strong_def']})")
            grid.addWidget(nd, r_, 1)
            st = muted(QLabel(f"strong results: {row['n_strong']:,}"))
            st.setFont(sized_font(self.font(), -1))
            grid.addWidget(st, r_ + 1, 0, 1, 2)
            for c, m in enumerate(metrics, start=2):
                d = row["metrics"][m["key"]]
                dg = 1 if m["key"] == "avg_mv" else 0
                a = QLabel(rng_text(d["all"], dg))
                a.setAlignment(Qt.AlignmentFlag.AlignRight)
                if d["all"]:
                    a.setToolTip(f"{row['label']}: the middle half of decks play {rng_text(d['all'], dg)} (typical: {d['all']['med']:.{dg}f}).")
                grid.addWidget(a, r_, c)
                b = muted(QLabel(rng_text(d["strong"], dg) if d["strong"] else "too few"))
                b.setAlignment(Qt.AlignmentFlag.AlignRight)
                b.setFont(sized_font(self.font(), -1))
                if d["strong"]:
                    b.setToolTip(f"Decks with strong results ({sh['strong_def']}): the middle half play {rng_text(d['strong'], dg)} (typical: {d['strong']['med']:.{dg}f}).")
                grid.addWidget(b, r_ + 1, c)
            r_ += 2
        wrap = QWidget()
        wrap.setLayout(grid)
        sec.lay.addWidget(wrap)
        sec.note("First line of each group: the middle half of all its decks.  Second line, smaller: the middle half of the decks with strong results.")
        sec.note(sh.get("note") or rep["notes"]["shape"])
        if sh.get("details"):
            sec.lay.addWidget(self._shape_details(sh["details"]))
        return sec

    # --- "More detail" blocks
    def _detail_block(self, key: str, title: str, tip: str = "") -> Collapsible:
        """A collapsed block whose open state survives re-rendering (kept in `self._open`); the content goes into `.body_lay`."""
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 0, 0, 4)
        lay.setSpacing(8)
        k = "detail:" + key
        c = Collapsible(title, body, expanded=k in self._open)
        self._details[key] = c
        c.body_lay = lay
        if tip:
            c.toggle.setToolTip(tip)
        c.toggle.toggled.connect(lambda on: (self._open.add if on else self._open.discard)(k))
        return c

    def _wrapped(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        return lab

    def _bullet(self, text: str) -> QLabel:
        lab = self._wrapped("•  " + text)
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return lab

    def _explorer(self, rep: dict) -> QWidget:
        cs = rep["cards"]
        blk = self._detail_block("cards", "More detail: all cards", "Every card with enough data in one sortable, searchable table")
        n = len(cs["explorer"]["rows"])
        blk.body_lay.addWidget(muted(self._wrapped(
            f"The {n} cards with enough data ({cs.get('min_eligible')} or more decks that could play them and {cs.get('min_exposure')} card-games).  "
            "Sort by any column; the Bo1 vs Bo3 column names a difference only when the two formats clearly differ.")))
        ex = CardExplorer(cs["explorer"], rep["fmt"])
        ex.set_owned(self._owned)
        ex.shown.connect(self._show_card)
        ex.hidden.connect(self.hover_popup.hide)
        self.explorer = ex
        blk.body_lay.addWidget(ex)
        return blk

    def _grid_header(self, grid: QGridLayout, heads: list[tuple[str, str]], right: tuple[int, ...] = ()) -> None:
        for c, (text, tip) in enumerate(heads):
            h = muted(QLabel(text))
            h.setFont(sized_font(self.font(), -1))
            h.setToolTip(tip)
            if c in right:
                h.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(h, 0, c)

    def _num_label(self, text: str, bold: bool = False, color: str | None = None) -> QLabel:
        lab = QLabel(text)
        lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if bold:
            lab.setFont(sized_font(self.font(), 0, bold=True))
        if color:
            pal = lab.palette()
            pal.setColor(QPalette.ColorRole.WindowText, semantic()[color])
            lab.setPalette(pal)
        return lab

    def _pair_cell(self, r: dict) -> QWidget:
        cell = QWidget()
        cl = QHBoxLayout(cell)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        pips = ColorPips()
        pips.set_colors(r["colors"] or "WU", "")
        cl.addWidget(pips)
        cl.addWidget(ElidedLabel(r["label"], min_chars=8), 1)
        return cell

    def _pair_details(self, rep: dict) -> QWidget:
        pd = rep["pair_details"]
        blk = self._detail_block("pairs", "More detail: each color pair",
                                 "Going first or second, game length, mulligans and what a splash color does, pair by pair")
        blk.body_lay.addWidget(muted(self._wrapped(pd.get("note", ""))))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(5, 1)
        self._grid_header(grid, [("Colors", ""), ("Going first (points)", "Win rate on the play minus on the draw, first games, players of the same skill. " + POINTS_TIP),
                                 ("How sure?", BADGE_TIP), ("Typical game", "Median number of turns, as counted by 17lands"),
                                 ("Mulligans", "Share of games that start with a mulligan (a plain figure, not adjusted for skill)"),
                                 ("Splash pay-off (points)", "Win rate of the decks that splash this color against the same pair without a splash. Only Likely or Strong results are shown.")],
                           right=(1, 3, 4))
        for i, d in enumerate(pd["rows"], start=1):
            grid.addWidget(self._pair_cell(d), i, 0)
            op = d.get("on_play")
            if op:
                grid.addWidget(self._num_label(pts(op["diff_pp"]), True, "good" if op["diff_pp"] >= 0 else "bad"), i, 1)
                grid.addWidget(Badge(op["badge"], f"{d['label']}: {op['range']}."), i, 2)
            else:
                grid.addWidget(self._num_label("–"), i, 1)
            grid.addWidget(self._num_label(f"{d['median_turns']:.0f} turns"), i, 3)
            grid.addWidget(self._num_label(f"{d['mulligan_rate'] * 100:.0f}%"), i, 4)
            good = [s for s in d.get("splash", []) if s["badge"] != "weak"]
            txt = ", ".join(f"{s['label'].lower()} {pts(s['effect_pp'])} ({BADGE_TEXT[s['badge']]})" for s in good) if good else \
                ("no clear effect" if d.get("splash") else "too few splash decks")
            lab = ElidedLabel(txt, min_chars=10)
            lab.setToolTip("; ".join(f"{s['label']}: {pts(s['effect_pp'])} points, {s['range']}, {s['n_decks']} decks ({BADGE_TEXT[s['badge']]})"
                                     for s in d.get("splash", [])) or "No splash color with enough decks.")
            grid.addWidget(lab, i, 5)
        blk.body_lay.addWidget(wrap_widget(grid))
        return blk

    def _effect_grid(self, rows: list[dict], first: str, fold_key: str = "") -> QWidget:
        shown = [r for r in rows if not r.get("hidden")]
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 2)
        self._grid_header(grid, [(first, ""), ("Share of decks", ""), ("Win rate against an average deck (points)", POINTS_TIP), ("", ""), ("How sure?", BADGE_TIP)],
                          right=(1,))
        tight = [r for r in shown if r["se_pp"] < 3]
        scale = max(1.5, max(max(abs(r["effect_pp"]), abs(r["effect_pp"]) + 1.96 * r["se_pp"]) for r in tight) * 1.05) if tight else 3.0
        row_ws: list[list[QWidget]] = []
        for i, r in enumerate(shown, start=1):
            name = ElidedLabel(r["label"], min_chars=8)
            grid.addWidget(name, i, 0)
            share = self._num_label(f"{r['share'] * 100:.0f}%")
            grid.addWidget(share, i, 1)
            bar = EffectBar(r["effect_pp"], r["effect_pp"] - 1.96 * r["se_pp"], r["effect_pp"] + 1.96 * r["se_pp"], scale, r["badge"])
            bar.setToolTip(f"{pts(r['effect_pp'])} points; {r['range']}; {r['n_decks']:,} decks")
            grid.addWidget(bar, i, 2)
            val = self._num_label(pts(r["effect_pp"]))
            grid.addWidget(val, i, 3)
            badge = Badge(r["badge"], f"{r['label']}: {r['range']}.")
            grid.addWidget(badge, i, 4)
            row_ws.append([name, share, bar, val, badge])
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        v.addWidget(wrap_widget(grid))
        fold = self._fold_weak(fold_key or first, row_ws, [r["badge"] for r in shown], [r["effect_pp"] for r in shown])
        if fold is not None:
            v.addWidget(fold, 0, Qt.AlignmentFlag.AlignLeft)
        return host

    def _shape_details(self, det: dict) -> QWidget:
        blk = self._detail_block("shape", "More detail: win rate by land count and creatures",
                                 "Skill-adjusted win rate of decks with different land counts and creature counts")
        for t in det.get("text", []):
            blk.body_lay.addWidget(self._bullet(t))
        for dim, title in (("lands", "By land count"), ("creatures", "By number of creatures")):
            if det.get(dim):
                h = QLabel(f"<b>{title}</b>")
                h.setTextFormat(Qt.TextFormat.RichText)
                blk.body_lay.addWidget(h)
                blk.body_lay.addWidget(self._effect_grid(det[dim], "Deck type", f"shape:{dim}"))
        blk.body_lay.addWidget(muted(self._wrapped(det.get("note", ""))))
        return blk

    # --- format evolution
    def _evo_card_row(self, r: dict) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(0)
        top = QHBoxLayout()
        top.setSpacing(6)
        word = {True: "moved toward its measured value", False: "moved away from its measured value", None: ""}[r.get("learned")]
        caption = (f"<b>{r['name']}</b><br>Played in {r['pr_early'] * 100:.0f}% of the decks that could play it early in the format, "
                   f"{r['pr_late'] * 100:.0f}% late." + (f"<br>{word.capitalize()}." if word else ""))
        link = CardLink(r["name"], caption)
        self._wire(link)
        top.addWidget(link, 1)
        top.addWidget(self._num_label(f"{r['change_pp']:+.0f} pts".replace("-", "−"), True, "good" if r["change_pp"] > 0 else "bad"))
        top.addWidget(Badge(r["badge"], f"{r['name']}: {r['range']}."))
        v.addLayout(top)
        sub = muted(ElidedLabel(f"{r['pr_early'] * 100:.0f}% → {r['pr_late'] * 100:.0f}% of decks" + (f" · {word}" if word else "")))
        sub.setFont(sized_font(self.font(), -1))
        sub.setToolTip(caption.replace("<br>", " ").replace("<b>", "").replace("</b>", ""))
        v.addWidget(sub)
        return w

    def _evolution(self, rep: dict) -> QWidget:
        ev = rep["evolution"]
        sec = Section("How the format changed", key="evolution", short="Format changes")
        if ev.get("too_few"):
            sec.sentence(f"Not enough events ({ev.get('n_drafts', 0):,}; at least {ev.get('min_drafts', 0):,} are needed) to compare the start and the end of the format.",
                         bullet=False)
            return sec
        sec.sentence("The events are cut by date into an early, a middle and a late third.  A color pair \"got better\" or \"got worse\" only when the change "
                     "between the early and the late third is Likely or Strong and points the same way in two independent halves of the events; "
                     "otherwise it is called stable.", bullet=False)
        for t in ev.get("text", []):
            sec.sentence(t)
        pairs = ev.get("pairs", [])
        if pairs:
            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(4)
            grid.setColumnStretch(0, 1)
            th = ev["thirds"]
            self._grid_header(grid, [("Colors", ""), ("Early", f"{th[0]['from']} to {th[0]['to']}"), ("Middle", f"{th[1]['from']} to {th[1]['to']}"),
                                     ("Late", f"{th[2]['from']} to {th[2]['to']}"), ("Change (points)", "Late minus early. " + POINTS_TIP),
                                     ("How sure?", BADGE_TIP), ("Over the format", "Better or worse only when Likely or Strong and the same in both halves of the data")],
                               right=(1, 2, 3, 4))
            for i, p in enumerate(pairs, start=1):
                grid.addWidget(self._pair_cell(p), i, 0)
                for c in range(3):
                    grid.addWidget(self._num_label(pts(p["effect"][c])), i, 1 + c)
                grid.addWidget(self._num_label(pts(p["change"]), p["verdict"] != "stable",
                                               None if p["verdict"] == "stable" else ("good" if p["change"] > 0 else "bad")), i, 4)
                grid.addWidget(Badge(p["change_badge"], f"{p['label']}: change {p['range']}"), i, 5)
                lab = QLabel({"better": "got better", "worse": "got worse"}.get(p["verdict"], "stable"))
                if p["verdict"] == "stable":
                    muted(lab)
                lab.setToolTip(f"Change in the two halves of the data: {pts(p['halves'][0])} and {pts(p['halves'][1])} points.")
                grid.addWidget(lab, i, 6)
            sec.lay.addWidget(wrap_widget(grid))
            sec.note("Each figure is the pair's win rate against an average deck of that third (players of the same skill); the average deck itself shifts as the format changes.")
        cs = ev.get("cards") or {}
        if cs.get("up") or cs.get("down"):
            h = QLabel("<b>Cards whose play rate changed clearly</b>")
            h.setTextFormat(Qt.TextFormat.RichText)
            sec.lay.addWidget(h)
            grid = QGridLayout()
            grid.setHorizontalSpacing(14)
            for c, (key, title) in enumerate((("up", "Played more by the end"), ("down", "Played less by the end"))):
                panel = QFrame()
                panel.setFrameShape(QFrame.Shape.NoFrame)
                pl = QVBoxLayout(panel)
                pl.setContentsMargins(0, 0, 0, 0)
                pl.setSpacing(2)
                pl.addWidget(QLabel(f"<b>{title}</b>"))
                if not cs.get(key):
                    pl.addWidget(muted(QLabel("None clear enough.")))
                for r in cs.get(key, []):
                    pl.addWidget(self._evo_card_row(r))
                pl.addStretch(1)
                grid.addWidget(panel, 0, c, Qt.AlignmentFlag.AlignTop)
                grid.setColumnStretch(c, 1)
            sec.lay.addWidget(wrap_widget(grid))
            sec.note(f"Play rate = share of the decks that could play the card.  Listed: moved by at least {int(cs['min_change'] * 100)} points, Likely or Strong, "
                     f"the same direction in both halves of the data, at least {cs['min_elig']} eligible decks at each end.  \"Toward its measured value\": a card "
                     "players first misjudged (Likely or Strong) whose play rate moved toward its true value.")
        sec.note(ev.get("note", ""))
        return sec

    # --- Best of 1 against Best of 3
    def _compare(self, rep: dict) -> QWidget:
        cmp = rep["compare"]
        sec = Section("Best of 1 against Best of 3", key="compare", short="Bo1 vs Bo3")
        blk = self._detail_block("compare", "More detail: both formats side by side", "Color pairs and the biggest clear card differences")
        for t in cmp.get("text", []):
            blk.body_lay.addWidget(self._bullet(t))
        rows = sorted(cmp.get("pairs", []), key=lambda r: -r["bo1"])
        if rows:
            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(4)
            grid.setColumnStretch(0, 1)
            self._grid_header(grid, [("Colors", ""), ("Best of 1", "Win rate against an average deck, points"), ("Best of 3", "Win rate against an average deck, points"),
                                     ("Difference", "Best of 3 minus Best of 1. " + POINTS_TIP), ("How sure?", BADGE_TIP)], right=(1, 2, 3))
            for i, r in enumerate(rows, start=1):
                grid.addWidget(self._pair_cell(r), i, 0)
                grid.addWidget(self._num_label(pts(r["bo1"])), i, 1)
                grid.addWidget(self._num_label(pts(r["bo3"])), i, 2)
                grid.addWidget(self._num_label(pts(r["diff_effect_pp"]), r["diff_badge"] != "weak"), i, 3)
                grid.addWidget(Badge(r["diff_badge"], f"{r['label']}: difference {r['diff_range']}"), i, 4)
            blk.body_lay.addWidget(wrap_widget(grid))
        cards = cmp.get("cards") or []
        h = QLabel("<b>Cards that clearly differ between the formats</b>")
        h.setTextFormat(Qt.TextFormat.RichText)
        blk.body_lay.addWidget(h)
        if not cards:
            blk.body_lay.addWidget(muted(QLabel("No card clearly differs (cards with limited data in either format are left out).")))
        for r in cards:
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(8)
            cap = (f"<b>{r['name']}</b><br>Per copy: {pts(r['v_bo1'])} points in Best of 1, {pts(r['v_bo3'])} in Best of 3 "
                   f"(against the average {GROUP_TITLE.get(r['group'], r['group']).lower()} card decks play).")
            link = CardLink(r["name"], cap)
            self._wire(link)
            hl.addWidget(link, 1)
            hl.addWidget(muted(QLabel(f"Bo1 {pts(r['v_bo1'])}   Bo3 {pts(r['v_bo3'])}")))
            hl.addWidget(self._num_label(pts(r["diff_pp"]), True, "good" if r["diff_pp"] > 0 else "bad"))
            hl.addWidget(Badge(r["badge"], f"{r['name']}: Best of 3 minus Best of 1"))
            blk.body_lay.addWidget(row)
        blk.body_lay.addWidget(muted(self._wrapped("A difference is called clear only when it is Likely or Strong; Best of 3 card values lean on Best of 1 where games are few, "
                                                   "so a real gap can hide.")))
        sec.lay.addWidget(blk)
        return sec

    def _data_notes(self, rep: dict) -> QWidget:
        n = rep["data_notes"]
        sec = Section("About these numbers", key="notes", short="About the numbers")
        fmt = "Best of 1" if rep["fmt"] == "bo1" else "Best of 3"
        sec.sentence(
            f"This analysis rests on {n['n_games']:,} {fmt} games of {n['n_decks']:,} Sealed decks ({n['n_drafts']:,} events, {n['date_from']} to {n['date_to']}) on 17lands.  "
            f"A card is judged only with at least {n['min_eligible']} decks that could play it and {n['min_exposure']} games with it in the deck "
            f"(below {n['thin_exposure']:,} games it is marked limited data); a color pair needs {n['min_pair_decks']} decks, a splash color {n['min_splash_decks']} "
            f"and a deck-shape band {n['min_band_decks']}.  Strong means it would be very surprising if it were luck, Likely means probably real, Weak means no clear "
            "difference: only Strong and Likely results are stated as findings.", bullet=False)
        return sec
