"""The "Deck search" tab (Full Corpus Explorer): query real 17Lands runs, instant corpus analytics,
similar Sealed pools, mid-run rebuilds, Bo3 sideboarding diffs, and opening-hand / replay card profiles.

Read-only over the set's corpus through `api.deck_search_*` / `api.search_*` (background tasks, one at a time).
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .deckbuilder import ImageStore, PileView
from .widgets import (
    BusyStrip,
    ElidedCombo,
    ElidedLabel,
    FlowLayout,
    FloatingCardPreview,
    muted,
    new_settings,
    semantic,
    short_error,
    sized_font,
)
from .workers import TaskController

log = logging.getLogger(__name__)

SORTS = [("Best result", "result"), ("Win rate", "win_rate"), ("Newest", "newest"), ("Oldest", "oldest"), ("Model score", "model")]
MODEL_SORT_TIP = ("Model score: the set model's score of each real Sealed deck, the same number the deck scorer gives a pasted deck "
                  "(its percentile among the real builds of the format; the model's game win rate in brackets).  Draft runs have no "
                  "score and are listed last.  The first time for a set (and after a recalculation) every Sealed deck is scored "
                  "once, which takes a minute or two; after that it is instant.")
MODES = [
    ("decks", "Decks", "Search and filter real 17Lands decks, with corpus-wide analytics"),
    ("similar", "Similar pools", "Find real 17Lands Sealed pools closest to your current Deck builder pool"),
    ("rebuilds", "Rebuilds", "Inspect runs where players rebuilt their deck mid-event (including 0–2 rescues)"),
    ("sideboard", "Bo3 sideboarding", "Explore real Game 1 → Game 2/3 Bo3 sideboard swaps by card or matchup"),
]
DATE_PRESETS = [
    ("All", None, "Include all dates in this set's corpus"),
    ("Last 2 months", 60, "Last 60 days relative to the latest game recorded in this set"),
    ("Last month", 30, "Last 30 days relative to the latest game recorded in this set"),
    ("Last 2 weeks", 14, "Last 14 days relative to the latest game recorded in this set"),
    ("Last week", 7, "Last 7 days relative to the latest game recorded in this set"),
]
SKILL_FILTERS = [("All pilots", None), ("Good pilots (≥ 55%)", 0.55), ("Top pilots (≥ 60%)", 0.60)]
RANK_CHOICES = [(r, r) for r in ("Bronze", "Silver", "Gold", "Platinum", "Diamond", "Mythic")]
COLOR_PAIRS = ("WU", "UB", "BR", "RG", "WG", "WB", "UR", "BG", "WR", "UG")
COLOR_CHOICES = [("White", "W"), ("Blue", "U"), ("Black", "B"), ("Red", "R"), ("Green", "G")]
COLOR_MATCH_CHOICES = [("Exactly these colors", "exact"), ("Including these colors", "include")]
SPLASH_FILTERS = [
    ("Any splash", None), ("No splash", "none"), ("With splash", "any"),
    ("+W", "W"), ("+U", "U"), ("+B", "B"), ("+R", "R"), ("+G", "G"),
]
PLAY_DRAW_FILTERS = [("Play or draw", None), ("G2 on the play", True), ("G2 on the draw", False)]

PAGE = 200                          # decks fetched per request
DEBOUNCE_MS = 250                   # a filter change searches again after this long
TROPHY = "★"                        # marks a run that reached the event's maximum wins
CARD_W_DEFAULT, CARD_W_MIN, CARD_W_MAX = 110, 70, 170
SIDE_SCALE = 0.7
COLUMNS_DECKS = ["Record", "Format", "Date", "Colors", "Copies", "Card in games", "Rank", "Skill", "Model score"]
COLUMNS_SIMILAR = ["Similarity", "Record", "Format", "Colors", "Shared", "Date", "Skill", "Model score"]
COLUMNS_REBUILDS = ["Final record", "Before → After", "Model score", "Format", "Colors", "Changed", "Date", "Skill"]
COLUMNS_SB = ["Match", "Format", "My → Opp (G1)", "G2", "Swapped", "Date", "Skill"]
REBUILD_COL_SORTS = ("record", "before_after", "model_delta", "format", "colors", "changed", "date", "skill")
_RANK_ORDER = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4, "diamond": 5, "mythic": 6}
_ASC_FIRST_COLS = {
    "decks": {1, 3},
    "similar": {2, 3},
    "rebuilds": {3, 4},
    "sideboard": {1, 2, 3},
}
_HEADER_TIPS = {
    "Model score": ("Model score: the set model's 0–100 percentile rating for Sealed decks (50 = average real build, 90 = top 10%). "
                    "In Rebuilds, shows Before → After (change): positive (+N) means the rebuilt deck is stronger by the model. "
                    "Click any column header to sort."),
    "Before → After": "Game record with the first build → game record after rebuilding. ★ marks a 0–2 rescue. Click to sort.",
    "Changed": "Non-basic cards changed between builds. Click to sort.",
}

INTRO = ("Explore real 17Lands decks across the full set corpus.\n\n"
         "Type a card name (or leave it blank to browse all decks) and press Enter.  "
         "Use the date toggles above or open Advanced search for multi-card rules, colors, pilot skill, rank, trophy, and land filters — "
         "or switch modes for Similar pools, Rebuilds, and Bo3 sideboarding.")
INTRO_SIMILAR = ("Find real 17Lands Sealed pools closest to your own.\n\n"
                 "Paste or import a Sealed pool on the Deck builder tab, then click Search to see how other pilots built pools like yours.")
INTRO_REBUILDS = ("Inspect real runs where players changed their deck mid-event.\n\n"
                  "Click Search to see rebuilds and 0–2 rescue pivots with visual Cut / Added card diffs (optionally filter by a card).")
INTRO_SB = ("Explore real Best-of-Three sideboarding decisions (Game 1 → Game 2/3).\n\n"
            "Enter a card to see which opponent colors it comes in or out against, or leave the card blank and pick a matchup to see the top sideboard swaps.")

SHARE_TIP = ("When ticked, only runs where the card was in the deck for most of the run are listed.\n"
             "In Bo3 events and runs with rebuilds the deck can change between games, so a run may have played the card in only some "
             "of its games.  Unticked, any run that played it at least once is listed.")


def count_text(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100.0 * x:.{digits}f}%"


def _pp(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100.0 * x:+.{digits}f} pp"


def _wr_ci(wr: float | None, lo: float | None, hi: float | None, n: int | None) -> str:
    if wr is None or not n:
        return "—"
    if lo is not None and hi is not None:
        return f"{100.0 * wr:.1f}% [{100.0 * lo:.0f}–{100.0 * hi:.0f}%], n={count_text(int(n))}g"
    return f"{100.0 * wr:.1f}% (n={count_text(int(n))}g)"


def _hdivider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setFixedHeight(1)
    return line


def _pill_style(compact: bool = False) -> str:
    pad = "0.18em 0.65em" if compact else "0.24em 0.75em"
    return f"QPushButton {{ padding: {pad}; border-radius: 0.75em; }}"


class ReadOnlyPile(PileView):
    """A PileView that only shows cards (and reports the hovered one): no drag, drop, double-click or context menu."""

    def __init__(self, zone: str, store: ImageStore, parent: QWidget | None = None):
        super().__init__(zone, store, parent)
        self.accepts = set()
        self.setAcceptDrops(False)

    def mousePressEvent(self, event) -> None:
        event.ignore()

    def mouseDoubleClickEvent(self, event) -> None:
        event.ignore()

    def contextMenuEvent(self, event) -> None:
        event.ignore()


class CheckDropdown(QComboBox):
    """A drop-down of tick boxes: a click ticks or unticks one item and the list stays open, so several can be chosen.  Nothing
    ticked = no filter.  `modes` (optional) adds radio buttons under the boxes (e.g. exactly / including these colors)."""

    changed = Signal()

    def __init__(self, items: list[tuple[str, str]], empty_text: str, modes: list[tuple[str, str]] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._items = list(items)
        self._empty = empty_text
        self._quiet = False
        self.addItem("")                       # one item: its text is the summary of the ticks (looks like the other selectors)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(self.fontMetrics().horizontalAdvance("Including WUBRG") + 36)
        menu = QMenu(self)
        panel = QWidget(menu)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(4)
        self.boxes: dict[str, QCheckBox] = {}
        for label, val in self._items:
            cb = QCheckBox(label, panel)
            cb.toggled.connect(self._on_toggled)
            lay.addWidget(cb)
            self.boxes[val] = cb
        self.radios: dict[str, QRadioButton] = {}
        if modes:
            line = QFrame(panel)
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            lay.addWidget(line)
            grp = QButtonGroup(panel)
            for i, (label, val) in enumerate(modes):
                rb = QRadioButton(label, panel)
                rb.setChecked(i == 0)
                grp.addButton(rb)
                rb.toggled.connect(self._on_radio)
                lay.addWidget(rb)
                self.radios[val] = rb
        clear = QPushButton("Clear", panel)
        clear.clicked.connect(self.clear)
        lay.addWidget(clear)
        act = QWidgetAction(menu)
        act.setDefaultWidget(panel)
        menu.addAction(act)
        self._menu = menu
        self._refresh_text()

    def menu(self) -> QMenu:
        return self._menu

    def showPopup(self) -> None:                  # the tick list instead of the combo's own list
        self._menu.setMinimumWidth(self.width())
        self._menu.popup(self.mapToGlobal(self.rect().bottomLeft()))

    def hidePopup(self) -> None:
        self._menu.hide()

    def text(self) -> str:
        return self.itemText(0)

    def values(self) -> list[str]:
        """Ticked values, in the order of the items."""
        return [v for _l, v in self._items if self.boxes[v].isChecked()]

    def mode(self) -> str | None:
        return next((v for v, rb in self.radios.items() if rb.isChecked()), None)

    def set_values(self, values: list[str], mode: str | None = None) -> None:
        self._quiet = True
        try:
            for v, cb in self.boxes.items():
                cb.setChecked(v in values)
            if mode in self.radios:
                self.radios[mode].setChecked(True)
        finally:
            self._quiet = False
        self._on_toggled()

    def clear(self) -> None:
        self.set_values([], next(iter(self.radios), None))

    def summary(self) -> str:
        """Short text of the choice ("" when nothing is ticked)."""
        vals = self.values()
        if not vals:
            return ""
        labels = {v: lab for lab, v in self._items}
        m = self.mode()
        one_letter = all(len(v) == 1 for v in vals)
        if m == "exact" and one_letter and len(vals) == 1:
            return f"Mono {labels[vals[0]]}"
        txt = "".join(vals) if (one_letter and len(vals) > 1) else ", ".join(labels[v] for v in vals)
        return f"Including {txt}" if m == "include" else txt

    def _refresh_text(self) -> None:
        self.setItemText(0, self.summary() or self._empty)

    def _on_radio(self, on: bool) -> None:
        if on:
            self._on_toggled()

    def _on_toggled(self, *_a) -> None:
        if self._quiet:
            return
        self._refresh_text()
        self.changed.emit()


class DeckSearchTab(QWidget):
    logMessage = Signal(str)
    statusMessage = Signal(str)
    errorOccurred = Signal(str, str)
    openInBuilderRequested = Signal(str, str, str)        # (set_code, fmt, arena_text)

    def __init__(self, api: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self.statuses: dict[str, Any] = {}
        self.calculating: set[str] = set()
        self.cards: list[str] = []                           # autocomplete names of the loaded set
        self.events: list[dict] = []                         # the loaded set's events (the format checkboxes)
        self._meta_key: tuple[str, str] | None = None        # (set, built_at) the names and events belong to
        self._meta_failed: tuple[str, str] | None = None
        self._pending: str | None = None                     # "new" / "more"
        self.piles: list[PileView] = []                      # the pile views of the deck on screen
        self.results: list[dict] = []                        # the runs/matches on screen
        self.data: dict | None = None                        # the last answer
        self.searched: tuple[str, str] | None = None         # (set, card name) of the answer on screen
        self._format_boxes: dict[str, QCheckBox] = {}
        self._filling = False

        # Explorer mode, date window & multi-card constraints
        self.mode: str = "decks"
        self._days: int | None = None
        self._include_cards: list[tuple[str, int]] = []      # [(card_name, min_copies)]
        self._exclude_cards: list[str] = []
        self._pool_only: list[str] = []

        # Current Deck Builder pool (for Similar Pools mode)
        self._pool_set: str = ""
        self._pool_named: dict[str, int] = {}

        # Model card values overlay cache: {(set_code, fmt): {lower_name: (pp, low_data)}}
        self._values_cache: dict[tuple[str, str], dict[str, tuple[float, bool]]] = {}
        self._values_loading: set[tuple[str, str]] = set()
        self._selected_build_idx: int | None = None
        self._col_sort: tuple[int, bool] | None = None       # (column_index, descending) chosen by clicking a table header

        self.task = TaskController(self)
        self.task.started.connect(self._on_started)
        self.task.progress.connect(lambda f, m: self.strip.update_progress(f, m))
        self.task.result.connect(self._on_result)
        self.task.error.connect(self._on_error)
        self.task.cancelled.connect(lambda: self.strip.end("Cancelled"))
        self.task.ended.connect(self._on_ended)

        self.val_task = TaskController(self)
        self.val_task.result.connect(self._on_values_loaded)
        self.val_task.error.connect(lambda _tb: None)

        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(DEBOUNCE_MS)
        self.debounce.timeout.connect(lambda: self.request_search("new"))

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        # ---- Row 1: Header bar (Title + Set + Segmented Mode Switcher)
        head = QHBoxLayout()
        head.setSpacing(10)
        title = QLabel("Deck search")
        title.setFont(sized_font(self.font(), 3, bold=True))
        head.addWidget(title)
        head.addSpacing(4)
        head.addWidget(muted(QLabel("Set")))
        self.set_combo = ElidedCombo()
        self.set_combo.setToolTip("Sets with data to search")
        self.set_combo.activated.connect(lambda _i: self._apply_set())
        head.addWidget(self.set_combo)
        head.addStretch(1)

        self.mode_bar = QFrame()
        self.mode_bar.setProperty("card", True)
        mbl = QHBoxLayout(self.mode_bar)
        mbl.setContentsMargins(3, 3, 3, 3)
        mbl.setSpacing(3)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QPushButton] = {}
        for key, label, tip in MODES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key == "decks")
            btn.setToolTip(tip)
            btn.setStyleSheet("QPushButton { padding: 0.25em 0.85em; border-radius: 0.35em; }")
            btn.clicked.connect(lambda _chk=False, k=key: self.set_mode(k))
            self.mode_group.addButton(btn)
            self.mode_buttons[key] = btn
            mbl.addWidget(btn)
        head.addWidget(self.mode_bar)
        root.addLayout(head)

        # ---- Row 2: Primary search bar + Sort + Advanced search toggle
        srow = QHBoxLayout()
        srow.setSpacing(8)
        self.card_edit = QLineEdit()
        self.card_edit.setPlaceholderText("Card name (leave blank to browse all decks)")
        self.card_edit.setClearButtonEnabled(True)
        self.card_edit.setMinimumWidth(240)
        self.card_edit.returnPressed.connect(lambda: self.request_search("new"))
        self.completer = QCompleter(QStringListModel([]), self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.setMaxVisibleItems(12)
        self.completer.activated.connect(self._on_completed)
        self.card_edit.setCompleter(self.completer)
        srow.addWidget(self.card_edit, 1)

        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("primary")
        self.search_btn.clicked.connect(lambda: self.request_search("new"))
        srow.addWidget(self.search_btn)

        self._sort_wrap = self._labeled("Sort", self._make_sort_combo())
        srow.addWidget(self._sort_wrap)

        self.advanced_btn = QPushButton("Advanced search ▾")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setChecked(False)
        self.advanced_btn.setToolTip("Show or hide multi-card rules, colors, pilot skill, rank, trophy, and land filters")
        self.advanced_btn.toggled.connect(self._on_advanced_toggled)
        srow.addWidget(self.advanced_btn)
        root.addLayout(srow)

        # ---- Similar pools banner (only in "similar" mode)
        self.similar_bar = QFrame()
        self.similar_bar.setProperty("card", True)
        sbl = QHBoxLayout(self.similar_bar)
        sbl.setContentsMargins(10, 6, 10, 6)
        self.pool_banner = QLabel("Paste or import a Sealed pool on the Deck builder tab to find similar real pools.")
        self.pool_banner.setWordWrap(True)
        sbl.addWidget(self.pool_banner, 1)
        self.similar_bar.setVisible(False)
        root.addWidget(self.similar_bar)

        # ---- Row 3: Quick bar — Date window toggles + mode-specific quick controls
        quick_row = QHBoxLayout()
        quick_row.setSpacing(10)
        quick_row.addWidget(muted(QLabel("Date")))
        self.date_bar = QFrame()
        self.date_bar.setProperty("card", True)
        dbl = QHBoxLayout(self.date_bar)
        dbl.setContentsMargins(3, 2, 3, 2)
        dbl.setSpacing(2)
        self.date_group = QButtonGroup(self)
        self.date_group.setExclusive(True)
        self.date_buttons: dict[int | None, QPushButton] = {}
        for label, days_val, tip in DATE_PRESETS:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(days_val is None)
            b.setToolTip(tip)
            b.setStyleSheet("QPushButton { padding: 0.18em 0.65em; border-radius: 0.32em; }")
            b.clicked.connect(lambda _chk=False, dv=days_val: self.set_date_window(dv))
            self.date_group.addButton(b)
            self.date_buttons[days_val] = b
            dbl.addWidget(b)
        quick_row.addWidget(self.date_bar)

        # Mode-specific quick controls (inline in Row 3 when in Rebuilds or Bo3 Sideboarding)
        self.rescue_box = QCheckBox("0–2 rescues only")
        self.rescue_box.setToolTip("Only runs that started with ≥2 losses (≤1 win) before rebuilding into ≥3 wins or a trophy")
        self.rescue_box.toggled.connect(self._on_filter_changed)
        self.rescue_box.setVisible(False)
        quick_row.addWidget(self.rescue_box)

        self.pivot_box = QCheckBox("Color pivots only")
        self.pivot_box.setToolTip("Only runs whose rebuilt deck changed main colors")
        self.pivot_box.toggled.connect(self._on_filter_changed)
        self.pivot_box.setVisible(False)
        quick_row.addWidget(self.pivot_box)

        self._min_changes_wrap = self._labeled("Min. changed", self._make_min_changes())
        self._min_changes_wrap.setVisible(False)
        quick_row.addWidget(self._min_changes_wrap)

        self._sb_my_wrap = self._labeled("My colors", self._make_sb_my_combo())
        self._sb_my_wrap.setVisible(False)
        quick_row.addWidget(self._sb_my_wrap)

        self._sb_opp_wrap = self._labeled("Opp G1", self._make_sb_opp_combo())
        self._sb_opp_wrap.setVisible(False)
        quick_row.addWidget(self._sb_opp_wrap)

        self._sb_play_wrap = self._labeled("G2", self._make_sb_play_combo())
        self._sb_play_wrap.setVisible(False)
        quick_row.addWidget(self._sb_play_wrap)

        self.sb_pool_box = QCheckBox("Only cards in my pool")
        self.sb_pool_box.setToolTip("Restrict sideboard swap plans to cards present in your current Deck builder Sealed pool")
        self.sb_pool_box.toggled.connect(self._on_filter_changed)
        self.sb_pool_box.setVisible(False)
        quick_row.addWidget(self.sb_pool_box)

        quick_row.addStretch(1)
        root.addLayout(quick_row)

        # ---- Row 4: Format checkboxes (slim wrap row)
        self.format_row = QWidget()
        self.format_flow = FlowLayout(self.format_row, spacing=12)
        root.addWidget(self.format_row)

        # ---- Collapsible Advanced Search Panel (clean card, hidden by default)
        self.advanced_panel = QFrame()
        self.advanced_panel.setProperty("card", True)
        self.advanced_panel.setFrameShape(QFrame.Shape.StyledPanel)
        apl = QVBoxLayout(self.advanced_panel)
        apl.setContentsMargins(12, 10, 12, 10)
        apl.setSpacing(10)

        # Section A: Card constraints (Decks mode)
        self._card_rules_sec = QWidget()
        crl = QVBoxLayout(self._card_rules_sec)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.setSpacing(6)
        cr_head = QLabel("Card constraints")
        cr_head.setFont(sized_font(self.font(), 0, bold=True))
        crl.addWidget(cr_head)
        cr_row = QHBoxLayout()
        cr_row.setSpacing(8)
        cr_hint = muted(QLabel("Add card from search box:"))
        cr_row.addWidget(cr_hint)
        self.add_inc_btn = QPushButton("+ Must include")
        self.add_inc_btn.setToolTip("Require this card in the main deck (combine multiple cards)")
        self.add_inc_btn.clicked.connect(lambda: self._add_chip("include"))
        cr_row.addWidget(self.add_inc_btn)
        self.add_exc_btn = QPushButton("− Exclude")
        self.add_exc_btn.setToolTip("Require 0 copies of this card in the main deck")
        self.add_exc_btn.clicked.connect(lambda: self._add_chip("exclude"))
        cr_row.addWidget(self.add_exc_btn)
        self.add_pool_btn = QPushButton("Pool only (Sealed)")
        self.add_pool_btn.setToolTip("Sealed only: card must be in the pool (sideboard) and NOT in the main deck")
        self.add_pool_btn.clicked.connect(lambda: self._add_chip("pool"))
        cr_row.addWidget(self.add_pool_btn)
        cr_row.addSpacing(12)
        self.share_box = QCheckBox("Card in the deck for most of the run")
        self.share_box.setChecked(True)
        self.share_box.setToolTip(SHARE_TIP)
        self.share_box.toggled.connect(self._on_filter_changed)
        cr_row.addWidget(self.share_box)
        cr_row.addStretch(1)
        crl.addLayout(cr_row)
        apl.addWidget(self._card_rules_sec)

        self._adv_divider = _hdivider()
        apl.addWidget(self._adv_divider)

        # Section B: Structured Deck & Pilot filter grid
        dp_head = QLabel("Deck & pilot filters")
        dp_head.setFont(sized_font(self.font(), 0, bold=True))
        apl.addWidget(dp_head)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        self._color_lbl = muted(QLabel("Colors"))
        self.color_pick = CheckDropdown(COLOR_CHOICES, "Any colors", COLOR_MATCH_CHOICES)
        self.color_pick.setToolTip("Main deck colors (splashes are set apart under Splash).  Tick one or more colors.\n"
                                   "Exactly these colors: the deck's main colors are exactly the ticked ones (White + Blue = WU "
                                   "decks, White alone = mono-White).\nIncluding these colors: the ticked colors are among its main "
                                   "colors (White = every deck with white as a main color).")
        self.color_pick.changed.connect(self._on_filter_changed)
        self._splash_lbl = muted(QLabel("Splash"))
        self.splash_combo = self._make_splash_combo()
        self._lands_lbl = muted(QLabel("Lands"))
        self._lands_wrap = self._make_lands_widget()
        self.trophy_box = QCheckBox("Trophy runs only")
        self.trophy_box.setToolTip("Only runs that reached the event's maximum wins")
        self.trophy_box.toggled.connect(self._on_filter_changed)

        self._skill_lbl = muted(QLabel("Pilot skill"))
        self.skill_combo = self._make_skill_combo()
        self._rank_lbl = muted(QLabel("Arena rank"))
        self.rank_pick = CheckDropdown(RANK_CHOICES, "Any rank")
        self.rank_pick.setToolTip("Arena rank of the player when the run started; tick one or more.\n17Lands records a rank only for "
                                  "Premier Draft runs: Sealed and Traditional runs have none, so a rank filter leaves them out.")
        self.rank_pick.changed.connect(self._on_filter_changed)
        self._min_games_lbl = muted(QLabel("Min. games"))
        self.min_games = self._make_min_games()
        self.reset_filters_btn = QPushButton("Reset filters")
        self.reset_filters_btn.setToolTip("Reset all advanced filters and card chips back to defaults")
        self.reset_filters_btn.clicked.connect(self.reset_filters)

        # Row 0: Deck composition
        grid.addWidget(self._color_lbl, 0, 0)
        grid.addWidget(self.color_pick, 0, 1)
        grid.addWidget(self._splash_lbl, 0, 2)
        grid.addWidget(self.splash_combo, 0, 3)
        grid.addWidget(self._lands_lbl, 0, 4)
        grid.addWidget(self._lands_wrap, 0, 5)
        grid.addWidget(self.trophy_box, 0, 6)

        # Row 1: Pilot & run filters
        grid.addWidget(self._skill_lbl, 1, 0)
        grid.addWidget(self.skill_combo, 1, 1)
        grid.addWidget(self._rank_lbl, 1, 2)
        grid.addWidget(self.rank_pick, 1, 3)
        grid.addWidget(self._min_games_lbl, 1, 4)
        grid.addWidget(self.min_games, 1, 5)
        grid.addWidget(self.reset_filters_btn, 1, 6, Qt.AlignmentFlag.AlignRight)
        grid.setColumnStretch(7, 1)
        apl.addLayout(grid)

        self._adv_tip = muted(QLabel("Tip: click any card in Corpus insights → Top companions to add it as a filter; right-click a +Include chip to require ≥2 copies."))
        self._adv_tip.setWordWrap(True)
        apl.addWidget(self._adv_tip)

        self.advanced_panel.setVisible(False)
        root.addWidget(self.advanced_panel)

        # ---- Active card-constraint & filter summary chips (shown whenever any chip/advanced filter is active)
        self.chip_row = QWidget()
        self.chip_flow = FlowLayout(self.chip_row, spacing=6)
        self.chip_row.setVisible(False)
        root.addWidget(self.chip_row)

        self.inline = QLabel("")
        self.inline.setWordWrap(True)
        self.inline.setVisible(False)
        root.addWidget(self.inline)
        self.strip = BusyStrip()
        self.strip.cancelRequested.connect(self.task.cancel)
        root.addWidget(self.strip)

        self.message = QLabel(INTRO)
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        muted(self.message)
        root.addWidget(self.message, 1)

        # ---- Main body splitter
        self.body = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        # Collapsible Corpus Insights panel
        self.insights_box = QFrame()
        self.insights_box.setProperty("card", True)
        self.insights_box.setFrameShape(QFrame.Shape.StyledPanel)
        il = QVBoxLayout(self.insights_box)
        il.setContentsMargins(10, 8, 10, 8)
        il.setSpacing(5)
        ih = QHBoxLayout()
        self.insights_toggle = QToolButton()
        self.insights_toggle.setObjectName("foldWeak")
        self.insights_toggle.setAutoRaise(True)
        self.insights_toggle.setText("▾ Corpus insights")
        self.insights_toggle.setFont(sized_font(self.font(), 0, bold=True))
        self.insights_toggle.clicked.connect(self._toggle_insights)
        ih.addWidget(self.insights_toggle)
        self.insights_headline = muted(ElidedLabel(""))
        ih.addWidget(self.insights_headline, 1)
        il.addLayout(ih)
        self.insights_body = QWidget()
        self.insights_lay = QVBoxLayout(self.insights_body)
        self.insights_lay.setContentsMargins(2, 2, 2, 0)
        self.insights_lay.setSpacing(5)
        il.addWidget(self.insights_body)
        self._insights_open = str(new_settings().value("decksearch/insights_open", True)).lower() not in ("false", "0", "no")
        self.insights_body.setVisible(self._insights_open)
        self.insights_toggle.setText(("▾ " if self._insights_open else "▸ ") + "Corpus insights")
        self.insights_box.setVisible(False)
        ll.addWidget(self.insights_box)

        self.summary = ElidedLabel("")
        ll.addWidget(self.summary)
        self.tree = QTreeWidget()
        self.tree.header().setSectionsClickable(True)
        self.tree.header().sectionClicked.connect(self._on_header_clicked)
        self._configure_tree(COLUMNS_DECKS)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree.itemSelectionChanged.connect(self._on_selected)
        ll.addWidget(self.tree, 1)
        self.more_btn = QPushButton("Show more")
        self.more_btn.clicked.connect(lambda: self.request_search("more"))
        ll.addWidget(self.more_btn)
        self.body.addWidget(left)

        # Right-hand deck inspector
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        self.deck_scroll = QScrollArea()
        self.deck_scroll.setWidgetResizable(True)
        self.deck_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.deck_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.deck_box = QWidget()
        self.deck_lay = QVBoxLayout(self.deck_box)
        self.deck_lay.setContentsMargins(4, 0, 8, 0)
        self.deck_lay.setSpacing(2)
        self.deck_scroll.setWidget(self.deck_box)
        rl.addWidget(self.deck_scroll, 1)
        btns = QHBoxLayout()
        self.copy_btn = QPushButton("Copy deck")
        self.copy_btn.setToolTip("Copy the deck for Arena's Import button")
        self.copy_btn.clicked.connect(self.copy_deck)
        btns.addWidget(self.copy_btn)
        self.open_builder_btn = QPushButton("Open in Deck builder")
        self.open_builder_btn.setToolTip("Send this deck (and Sealed pool when stored) to the Deck builder tab and score it")
        self.open_builder_btn.clicked.connect(self.open_in_builder)
        btns.addWidget(self.open_builder_btn)
        self.values_box = QCheckBox("Model values")
        self.values_box.setToolTip("Overlay the set model's per-card value badges on the cards")
        self.values_box.setChecked(str(new_settings().value("decksearch/show_values", False)).lower() in ("true", "1", "yes"))
        self.values_box.toggled.connect(self._on_values_toggled)
        btns.addWidget(self.values_box)
        btns.addStretch(1)
        btns.addWidget(muted(QLabel("Card size")))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(CARD_W_MIN, CARD_W_MAX)
        self.size_slider.setSingleStep(10)
        self.size_slider.setFixedWidth(120)
        self.size_slider.setValue(self._saved_int("decksearch/card_w", CARD_W_DEFAULT, CARD_W_MIN, CARD_W_MAX))
        self.size_slider.setToolTip("Size of the card images")
        self.size_slider.valueChanged.connect(self._on_card_size)
        btns.addWidget(self.size_slider)
        rl.addLayout(btns)
        self.body.addWidget(right)
        self.body.setStretchFactor(0, 3)
        self.body.setStretchFactor(1, 2)
        self.body.setSizes([640, 440])
        self.body.setChildrenCollapsible(False)
        root.addWidget(self.body, 1)
        self.store = ImageStore.for_api(api)
        self.hover_popup = FloatingCardPreview(self.store, self)

        self._show_empty(INTRO)
        self._rebuild_chips()
        self._update_controls()

    # ------------------------------------------------------------------ widget factories
    @staticmethod
    def _labeled(label: str, widget: QWidget) -> QWidget:
        box = QWidget()
        hl = QHBoxLayout(box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(5)
        hl.addWidget(muted(QLabel(label)))
        hl.addWidget(widget)
        return box

    def _make_min_games(self) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(1, 20)
        sb.setValue(self._saved_int("decksearch/min_games", 3, 1, 20))
        sb.setToolTip("Leave out runs with fewer games than this")
        sb.valueChanged.connect(self._on_filter_changed)
        return sb

    def _make_min_changes(self) -> QSpinBox:
        self.min_changes = QSpinBox()
        self.min_changes.setRange(1, 20)
        self.min_changes.setSuffix(" cards")
        self.min_changes.setValue(self._saved_int("decksearch/min_rebuild_changes", 5, 1, 20))
        self.min_changes.setToolTip("Ignore minor tweaks: only count runs that changed at least this many non-basic cards (default 5)")
        self.min_changes.valueChanged.connect(self._on_filter_changed)
        return self.min_changes

    def _make_sort_combo(self) -> ElidedCombo:
        self.sort_combo = ElidedCombo()
        for label, key in SORTS:
            self.sort_combo.addItem(label, key)
            if key == "model":
                self.sort_combo.setItemData(self.sort_combo.count() - 1, MODEL_SORT_TIP, Qt.ItemDataRole.ToolTipRole)
        i = self.sort_combo.findData(str(new_settings().value("decksearch/sort", "result") or "result"))
        self.sort_combo.setCurrentIndex(max(i, 0))
        self.sort_combo.activated.connect(self._on_sort_chosen)
        return self.sort_combo

    def _make_skill_combo(self) -> ElidedCombo:
        cb = ElidedCombo()
        for label, val in SKILL_FILTERS:
            cb.addItem(label, val)
        cb.setToolTip("Filter by 17Lands player win-rate bracket")
        cb.activated.connect(self._on_filter_changed)
        return cb

    def _make_splash_combo(self) -> ElidedCombo:
        cb = ElidedCombo()
        for label, val in SPLASH_FILTERS:
            cb.addItem(label, val)
        cb.setToolTip("Filter by splashed color")
        cb.activated.connect(self._on_filter_changed)
        return cb

    def _make_lands_widget(self) -> QWidget:
        box = QWidget()
        hl = QHBoxLayout(box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        self.min_lands = QSpinBox()
        self.min_lands.setRange(0, 24)
        self.min_lands.setSpecialValueText("Any")
        self.min_lands.setValue(0)
        self.min_lands.setToolTip("Minimum lands in deck (0 = Any)")
        self.min_lands.valueChanged.connect(self._on_filter_changed)
        hl.addWidget(self.min_lands)
        hl.addWidget(muted(QLabel("–")))
        self.max_lands = QSpinBox()
        self.max_lands.setRange(0, 24)
        self.max_lands.setSpecialValueText("Any")
        self.max_lands.setValue(0)
        self.max_lands.setToolTip("Maximum lands in deck (0 = Any)")
        self.max_lands.valueChanged.connect(self._on_filter_changed)
        hl.addWidget(self.max_lands)
        return box

    def _make_sb_my_combo(self) -> ElidedCombo:
        self.sb_my_combo = ElidedCombo()
        self.sb_my_combo.addItem("Any", None)
        for p in COLOR_PAIRS:
            self.sb_my_combo.addItem(p, p)
        self.sb_my_combo.activated.connect(self._on_filter_changed)
        return self.sb_my_combo

    def _make_sb_opp_combo(self) -> ElidedCombo:
        self.sb_opp_combo = ElidedCombo()
        self.sb_opp_combo.addItem("Any", None)
        for p in COLOR_PAIRS:
            self.sb_opp_combo.addItem(p, p)
        self.sb_opp_combo.setToolTip("Opponent's colors observed in Game 1")
        self.sb_opp_combo.activated.connect(self._on_filter_changed)
        return self.sb_opp_combo

    def _make_sb_play_combo(self) -> ElidedCombo:
        self.sb_play_combo = ElidedCombo()
        for label, val in PLAY_DRAW_FILTERS:
            self.sb_play_combo.addItem(label, val)
        self.sb_play_combo.activated.connect(self._on_filter_changed)
        return self.sb_play_combo

    def _configure_tree(self, cols: list[str]) -> None:
        self.tree.setColumnCount(len(cols))
        self.tree.setHeaderLabels(cols)
        hi = self.tree.headerItem()
        for c, title in enumerate(cols):
            hi.setToolTip(c, _HEADER_TIPS.get(title, f"Click to sort by {title}"))
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSortIndicatorShown(False)
        for c in range(len(cols)):
            hdr.setSectionResizeMode(c, hdr.ResizeMode.ResizeToContents)
        if len(cols) > 1:
            stretch = 1 if cols is COLUMNS_DECKS else (3 if cols is COLUMNS_REBUILDS else 2)
            hdr.setSectionResizeMode(stretch, hdr.ResizeMode.Stretch)

    @staticmethod
    def _row_sort_key(mode: str, col: int, r: dict) -> tuple[bool, Any]:
        """Typed `(is_missing, value)` sort key for any column in any mode (missing values always stay last)."""
        def _rank_val(rk: str | None) -> tuple[bool, int]:
            head = str(rk or "").split("-", 1)[0].strip().lower()
            v = _RANK_ORDER.get(head, 0)
            return (v == 0, v)

        def _opt_float(x) -> tuple[bool, float]:
            return (x is None, 0.0 if x is None else float(x))

        def _rec_val() -> tuple[bool, tuple]:
            return (False, (float(r.get("score") or 0.0), -int(r.get("losses") or 0), float(r.get("win_rate") or 0.0), int(r.get("games") or 0)))

        if mode == "decks":
            if col == 0:
                return _rec_val()
            if col == 1:
                return (False, str(r.get("event_label") or "").lower())
            if col == 2:
                d = str(r.get("date") or "")
                return (not d, d)
            if col == 3:
                c = str(r.get("colors") or "").lower()
                return (not c, c)
            if col == 4:
                cp = int(r.get("copies") or 0)
                return (cp <= 0, cp)
            if col == 5:
                g = int(r.get("games") or 0)
                n = int(r.get("games_with_card") or 0)
                return (g <= 0, (n / max(g, 1), n, g))
            if col == 6:
                return _rank_val(r.get("rank"))
            if col == 7:
                return _opt_float(r.get("skill"))
            if col == 8:
                ms = r.get("model_score")
                return (ms is None, (0.0, 0.0) if ms is None else (float(ms), float(r.get("model_q") or 0.0)))
        elif mode == "similar":
            if col == 0:
                return _opt_float(r.get("similarity"))
            if col == 1:
                return _rec_val()
            if col == 2:
                return (False, str(r.get("event_label") or "").lower())
            if col == 3:
                c = str(r.get("colors") or "").lower()
                return (not c, c)
            if col == 4:
                return (False, int(r.get("shared_count") or 0))
            if col == 5:
                d = str(r.get("date") or "")
                return (not d, d)
            if col == 6:
                return _opt_float(r.get("skill"))
            if col == 7:
                ms = r.get("model_score")
                return (ms is None, (0.0, 0.0) if ms is None else (float(ms), float(r.get("model_q") or 0.0)))
        elif mode == "rebuilds":
            if col == 0:
                return _rec_val()
            if col == 1:
                return (False, (1 if r.get("is_rescue") else 0, float(r.get("wr_gain") or 0.0), float(r.get("score") or 0.0)))
            if col == 2:
                md = r.get("model_score_delta")
                return (md is None, (0.0, 0.0) if md is None else (float(md), float(r.get("b1_model_score") or 0.0)))
            if col == 3:
                return (False, str(r.get("event_label") or "").lower())
            if col == 4:
                c = str(r.get("colors") or "").lower()
                return (not c, c)
            if col == 5:
                return (False, int(r.get("cards_changed") or 0))
            if col == 6:
                d = str(r.get("date") or "")
                return (not d, d)
            if col == 7:
                return _opt_float(r.get("skill"))
        else:  # sideboard
            if col == 0:
                return (False, (1 if r.get("match_won") else 0, str(r.get("record") or "")))
            if col == 1:
                return (False, str(r.get("event_label") or "").lower())
            if col == 2:
                return (False, (str(r.get("colors") or "").lower(), str(r.get("opp_colors") or "").lower()))
            if col == 3:
                return (False, 1 if r.get("g2_on_play") else 0)
            if col == 4:
                return (False, int(r.get("n_swapped") or 0))
            if col == 5:
                d = str(r.get("date") or "")
                return (not d, d)
            if col == 6:
                return _opt_float(r.get("skill"))
        return (False, "")

    def _on_header_clicked(self, col: int) -> None:
        if not self.results and self.mode != "rebuilds":
            return
        if self._col_sort is not None and self._col_sort[0] == col:
            desc = not self._col_sort[1]
        else:
            desc = col not in _ASC_FIRST_COLS.get(self.mode, set())
        self._col_sort = (col, desc)
        hdr = self.tree.header()
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(col, Qt.SortOrder.DescendingOrder if desc else Qt.SortOrder.AscendingOrder)

        # Rebuilds: full-corpus backend sort across all matching rebuilt runs
        if self.mode == "rebuilds" and 0 <= col < len(REBUILD_COL_SORTS):
            self.request_search("new")
            return
        # Decks: delegate Date, Model score (desc), and Record (desc) to full-corpus backend sort when appropriate
        if self.mode == "decks":
            if col == 2:
                want = "newest" if desc else "oldest"
                self.sort_combo.setCurrentIndex(max(self.sort_combo.findData(want), 0))
                new_settings().setValue("decksearch/sort", want)
                self.request_search("new")
                return
            if col == 8 and desc:
                self.sort_combo.setCurrentIndex(max(self.sort_combo.findData("model"), 0))
                new_settings().setValue("decksearch/sort", "model")
                self.request_search("new")
                return
            if col == 0 and desc:
                self.sort_combo.setCurrentIndex(max(self.sort_combo.findData("result"), 0))
                new_settings().setValue("decksearch/sort", "result")
                self.request_search("new")
                return
        # All other columns (Decks non-dropdown / ascending, Similar pools, Bo3 sideboarding): instant typed sort on loaded rows
        self._apply_local_sort(col, desc)

    def _apply_local_sort(self, col: int, desc: bool) -> None:
        if not self.results:
            return
        cur_row = self._selected()
        cur_idx = self.results.index(cur_row) if cur_row in self.results else None
        pairs = []
        for pos in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(pos)
            idx = it.data(0, Qt.ItemDataRole.UserRole)
            r = self.results[idx] if isinstance(idx, int) and 0 <= idx < len(self.results) else {}
            pairs.append((self._row_sort_key(self.mode, col, r), idx))
        present = [p for p in pairs if not p[0][0]]
        missing = [p for p in pairs if p[0][0]]
        present.sort(key=lambda p: p[0][1], reverse=desc)
        ordered = [p[1] for p in present] + [p[1] for p in missing]
        # Re-insert items in sorted order without rebuilding text/colors
        by_idx: dict[int, QTreeWidgetItem] = {}
        while self.tree.topLevelItemCount():
            it = self.tree.takeTopLevelItem(0)
            by_idx[int(it.data(0, Qt.ItemDataRole.UserRole))] = it
        self.tree.blockSignals(True)
        try:
            for idx in ordered:
                it = by_idx.get(idx)
                if it is not None:
                    self.tree.addTopLevelItem(it)
            if cur_idx is not None and cur_idx in by_idx:
                self.tree.setCurrentItem(by_idx[cur_idx])
            elif self.tree.topLevelItemCount():
                self.tree.setCurrentItem(self.tree.topLevelItem(0))
        finally:
            self.tree.blockSignals(False)

    # ------------------------------------------------------------------ small helpers & advanced panel
    @staticmethod
    def _saved_int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(new_settings().value(key, default))))
        except (TypeError, ValueError):
            return default

    def set_code(self) -> str:
        d = self.set_combo.currentData()
        return str(d) if d else ""

    def date_days(self) -> int | None:
        return self._days

    def set_date_window(self, days: int | None) -> None:
        self._days = days
        btn = self.date_buttons.get(days)
        if btn is not None:
            btn.setChecked(True)
        self._on_filter_changed()

    def _on_advanced_toggled(self, on: bool) -> None:
        self.advanced_panel.setVisible(bool(on))
        self._refresh_advanced_label()

    def _active_advanced_count(self) -> int:
        n = 0
        if self.mode == "decks":
            n += len(self._include_cards) + len(self._exclude_cards) + len(self._pool_only)
            if self.color_pick.values():
                n += 1
            if self.splash_combo.currentData():
                n += 1
            if self.rank_pick.values():
                n += 1
            if self.min_lands.value() > 0 or self.max_lands.value() > 0:
                n += 1
            if not self.share_box.isChecked():
                n += 1
        if self.skill_combo.currentData() is not None:
            n += 1
        if self.mode in ("decks", "similar") and self.trophy_box.isChecked():
            n += 1
        if self.mode in ("decks", "similar") and self.min_games.value() != 3:
            n += 1
        return n

    def _refresh_advanced_label(self) -> None:
        cnt = self._active_advanced_count()
        arrow = "▴" if self.advanced_btn.isChecked() else "▾"
        self.advanced_btn.setText(f"Advanced search ({cnt}) {arrow}" if cnt else f"Advanced search {arrow}")

    def reset_filters(self) -> None:
        """Reset all advanced filters, date window, and card chips back to defaults."""
        self._filling = True
        try:
            self._include_cards.clear()
            self._exclude_cards.clear()
            self._pool_only.clear()
            self._days = None
            if None in self.date_buttons:
                self.date_buttons[None].setChecked(True)
            self.color_pick.clear()
            self.splash_combo.setCurrentIndex(0)
            self.skill_combo.setCurrentIndex(0)
            self.rank_pick.clear()
            self.trophy_box.setChecked(False)
            self.min_lands.setValue(0)
            self.max_lands.setValue(0)
            self.min_games.setValue(3)
            self.share_box.setChecked(True)
            self.rescue_box.setChecked(False)
            self.pivot_box.setChecked(False)
            self.min_changes.setValue(5)
            self.sb_my_combo.setCurrentIndex(0)
            self.sb_opp_combo.setCurrentIndex(0)
            self.sb_play_combo.setCurrentIndex(0)
            self.sb_pool_box.setChecked(False)
        finally:
            self._filling = False
        self._rebuild_chips()
        self._refresh_advanced_label()
        if self.searched is not None:
            self.request_search("new")

    def _meta_ok(self) -> bool:
        return self._meta_key is not None and self._meta_key[0] == self.set_code()

    def _usable(self) -> bool:
        sc = self.set_code()
        return bool(sc) and sc not in self.calculating

    def _show_empty(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(True)
        self.body.setVisible(False)

    def _show_body(self) -> None:
        self.message.setVisible(False)
        self.body.setVisible(True)

    def _say(self, text: str, warn: bool = True) -> None:
        self.inline.setText(text)
        self.inline.setVisible(bool(text))
        if text:
            self.inline.setStyleSheet(f"color: {semantic()['warn' if warn else 'accent'].name()};")

    def _toggle_insights(self) -> None:
        self._insights_open = not self._insights_open
        new_settings().setValue("decksearch/insights_open", self._insights_open)
        self.insights_body.setVisible(self._insights_open)
        self.insights_toggle.setText(("▾ " if self._insights_open else "▸ ") + "Corpus insights")

    def _update_controls(self) -> None:
        ok = self._usable()
        for w in (self.card_edit, self.search_btn, self.advanced_btn, self.min_games, self.share_box, self.sort_combo,
                  self.format_row, self.skill_combo, self.rank_pick, self.color_pick, self.splash_combo,
                  self.trophy_box, self.min_lands, self.max_lands, self.add_inc_btn, self.add_exc_btn,
                  self.add_pool_btn, self.reset_filters_btn, self.rescue_box, self.pivot_box, self.min_changes,
                  self.sb_my_combo, self.sb_opp_combo, self.sb_play_combo):
            w.setEnabled(ok)
        has_pool = self._has_matching_pool()
        self.sb_pool_box.setEnabled(ok and has_pool)
        self.sb_pool_box.setToolTip(
            "Restrict sideboard swap plans to cards present in your current Deck builder Sealed pool"
            if has_pool else "Load a Sealed pool for this set on the Deck builder tab to filter sideboard plans to your pool"
        )
        for btn in self.mode_buttons.values():
            btn.setEnabled(ok)
        for btn in self.date_buttons.values():
            btn.setEnabled(ok)
        has_more = (self.mode in ("decks", "rebuilds") and self.data is not None
                    and len(self.results) < int(self.data.get("total", 0)))
        self.more_btn.setVisible(self.mode in ("decks", "rebuilds"))
        self.more_btn.setEnabled(ok and not self.task.running and has_more)
        sel = self._selected()
        self.copy_btn.setEnabled(sel is not None)
        self.open_builder_btn.setEnabled(sel is not None)

    # ------------------------------------------------------------------ mode switching & current pool
    def set_mode(self, mode: str) -> None:
        if mode not in self.mode_buttons or mode == self.mode:
            return
        self.mode = mode
        self._col_sort = None
        self.mode_buttons[mode].setChecked(True)
        is_decks = mode == "decks"
        is_sim = mode == "similar"
        is_reb = mode == "rebuilds"
        is_sb = mode == "sideboard"

        # Primary bar & mode-specific quick controls
        self.card_edit.setVisible(not is_sim)
        self._sort_wrap.setVisible(is_decks)
        self.similar_bar.setVisible(is_sim)
        self.rescue_box.setVisible(is_reb)
        self.pivot_box.setVisible(is_reb)
        self._min_changes_wrap.setVisible(is_reb)
        self._sb_my_wrap.setVisible(is_sb)
        self._sb_opp_wrap.setVisible(is_sb)
        self._sb_play_wrap.setVisible(is_sb)
        self.sb_pool_box.setVisible(is_sb)

        # Advanced panel section visibility per mode
        self._card_rules_sec.setVisible(is_decks)
        self._adv_divider.setVisible(is_decks)
        for w in (self._color_lbl, self.color_pick, self._splash_lbl, self.splash_combo,
                  self._lands_lbl, self._lands_wrap, self._rank_lbl, self.rank_pick, self._adv_tip):
            w.setVisible(is_decks)
        for w in (self._min_games_lbl, self.min_games, self.trophy_box):
            w.setVisible(is_decks or is_sim)

        if is_decks:
            self.card_edit.setPlaceholderText("Card name (leave blank to browse all decks)")
            self._configure_tree(COLUMNS_DECKS)
            intro = INTRO
        elif is_sim:
            self._configure_tree(COLUMNS_SIMILAR)
            self._refresh_pool_banner()
            intro = INTRO_SIMILAR
        elif is_reb:
            self.card_edit.setPlaceholderText("Optional card filter (leave blank for all mid-run rebuilds)")
            self._configure_tree(COLUMNS_REBUILDS)
            intro = INTRO_REBUILDS
        else:
            self.card_edit.setPlaceholderText("Card name (or leave blank for Matchup Sideboard Plan)")
            self._configure_tree(COLUMNS_SB)
            intro = INTRO_SB

        self.debounce.stop()
        self._pending = None
        self._say("")
        self._reset_results()
        self._rebuild_chips()
        self._refresh_advanced_label()
        self._show_empty(intro)
        self._update_controls()
        if self._usable() and self._meta_ok() and (is_reb or is_sb or (is_sim and self._has_matching_pool())):
            self.request_search("new")

    def set_current_pool(self, set_code: str, pool: dict[int, int]) -> None:
        """Called when the Deck builder tab's pool changes: store named pool counts for Similar pools mode."""
        sc = (set_code or "").upper()
        named: dict[str, int] = {}
        if sc and pool:
            try:
                info = self.api.card_info(sc) or {}
            except Exception:  # noqa: BLE001
                info = {}
            for cid, n in pool.items():
                if int(n or 0) <= 0:
                    continue
                d = info.get(int(cid))
                nm = d.get("name") if isinstance(d, dict) else None
                if nm and not d.get("is_basic"):
                    named[str(nm)] = named.get(str(nm), 0) + int(n)
        self._pool_set = sc if named else ""
        self._pool_named = named
        self._refresh_pool_banner()
        self._update_controls()
        if self.mode == "similar" and self._has_matching_pool() and self.isVisible():
            self.request_search("new")

    def open_similar_for_pool(self, set_code: str, pool: dict[int, int]) -> None:
        """Switch to `set_code` (if tracked), load `pool`, switch to Similar pools mode, and search (even before meta loads)."""
        sc = (set_code or "").upper()
        if sc and sc != self.set_code():
            ix = self.set_combo.findData(sc)
            if ix >= 0:
                self.set_combo.setCurrentIndex(ix)
                self._apply_set()
        self.set_current_pool(sc, pool)
        if self.mode != "similar":
            self.set_mode("similar")
        if self._usable() and self._has_matching_pool():
            self.request_search("new")

    def _filter_color_pair(self, pair: str) -> None:
        """Click a color pair pill in Corpus insights -> filter to that exact color pair."""
        cols = [c for c in "WUBRG" if c in (pair or "").upper()]
        if not cols:
            return
        self.color_pick.set_values(cols, "exact")

    def _has_matching_pool(self) -> bool:
        return bool(self._pool_named) and self._pool_set == self.set_code()

    def _refresh_pool_banner(self) -> None:
        if not self._pool_named:
            self.pool_banner.setText("No pool loaded yet.  Paste or import a Sealed pool on the Deck builder tab, then click Search.")
        elif self._pool_set != self.set_code():
            self.pool_banner.setText(
                f"Deck builder currently holds a <b>{self._pool_set}</b> pool ({sum(self._pool_named.values())} cards).  "
                f"Switch the Set picker above to <b>{self._pool_set}</b> to search similar pools."
            )
        else:
            self.pool_banner.setText(
                f"Ready to match your current <b>{self._pool_set}</b> Sealed pool "
                f"({sum(self._pool_named.values())} non-basic cards, {len(self._pool_named)} unique)."
            )

    # ------------------------------------------------------------------ active chips & filter badges
    def _add_chip(self, kind: str, name: str | None = None) -> None:
        nm = (name if name is not None else self.card_edit.text()).strip()
        if not nm:
            self._say("Type a card name first.", warn=False)
            return
        low = nm.casefold()
        for c in self.cards:
            if c.casefold() == low:
                nm = c
                break
        if kind == "include":
            if not any(x.casefold() == nm.casefold() for x, _ in self._include_cards):
                self._include_cards.append((nm, 1))
        elif kind == "exclude":
            if not any(x.casefold() == nm.casefold() for x in self._exclude_cards):
                self._exclude_cards.append(nm)
        elif kind == "pool":
            if not any(x.casefold() == nm.casefold() for x in self._pool_only):
                self._pool_only.append(nm)
        if name is None:
            self.card_edit.clear()
        self._say("")
        self._rebuild_chips()
        self._refresh_advanced_label()
        self.request_search("new")

    def _add_pill(self, text: str, tip: str, on_click) -> QPushButton:
        btn = QPushButton(text)
        btn.setToolTip(tip)
        btn.setStyleSheet(_pill_style(compact=True))
        btn.clicked.connect(on_click)
        self.chip_flow.addWidget(btn)
        return btn

    def _rebuild_chips(self) -> None:
        while self.chip_flow.count():
            it = self.chip_flow.takeAt(0)
            if it is not None and it.widget() is not None:
                it.widget().deleteLater()
        count = 0
        if self.mode == "decks":
            for i, (nm, mn) in enumerate(self._include_cards):
                btn = self._add_pill(
                    f"+ {nm}" + (f" (≥{mn})" if mn > 1 else "") + "  ×",
                    "Required in main deck — click to remove (right-click to toggle ≥1 / ≥2 copies)",
                    lambda _c=False, idx=i: self._remove_chip("include", idx),
                )
                btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                btn.customContextMenuRequested.connect(lambda _p, idx=i: self._toggle_inc_copies(idx))
                count += 1
            for i, nm in enumerate(self._exclude_cards):
                self._add_pill(f"− {nm}  ×", "Excluded from main deck — click to remove",
                               lambda _c=False, idx=i: self._remove_chip("exclude", idx))
                count += 1
            for i, nm in enumerate(self._pool_only):
                self._add_pill(f"Pool only: {nm}  ×", "In Sealed pool but cut from main deck — click to remove",
                               lambda _c=False, idx=i: self._remove_chip("pool", idx))
                count += 1
            if self.color_pick.values():
                self._add_pill(f"Colors: {self.color_pick.summary()}  ×", "Click to clear color filter",
                               lambda _c=False: self._clear_one("color"))
                count += 1
            if self.splash_combo.currentData():
                self._add_pill(f"Splash: {self.splash_combo.currentText()}  ×", "Click to clear splash filter",
                               lambda _c=False: self._clear_one("splash"))
                count += 1
            if self.rank_pick.values():
                self._add_pill(f"Rank: {self.rank_pick.summary()}  ×", "Click to clear rank filter",
                               lambda _c=False: self._clear_one("rank"))
                count += 1
            if self.min_lands.value() > 0 or self.max_lands.value() > 0:
                lo = str(self.min_lands.value()) if self.min_lands.value() > 0 else "Any"
                hi = str(self.max_lands.value()) if self.max_lands.value() > 0 else "Any"
                self._add_pill(f"Lands: {lo}–{hi}  ×", "Click to clear land count filter",
                               lambda _c=False: self._clear_one("lands"))
                count += 1
        if self.skill_combo.currentData() is not None:
            self._add_pill(f"{self.skill_combo.currentText()}  ×", "Click to clear pilot skill filter",
                           lambda _c=False: self._clear_one("skill"))
            count += 1
        if self.mode in ("decks", "similar") and self.trophy_box.isChecked():
            self._add_pill("★ Trophy only  ×", "Click to show non-trophy runs too",
                           lambda _c=False: self._clear_one("trophy"))
            count += 1
        if count > 1:
            clr = QPushButton("Clear all")
            clr.setStyleSheet(_pill_style(compact=True))
            clr.clicked.connect(self.reset_filters)
            self.chip_flow.addWidget(clr)
        self.chip_row.setVisible(count > 0)

    def _clear_one(self, which: str) -> None:
        self._filling = True
        try:
            if which == "color":
                self.color_pick.clear()
            elif which == "splash":
                self.splash_combo.setCurrentIndex(0)
            elif which == "rank":
                self.rank_pick.clear()
            elif which == "skill":
                self.skill_combo.setCurrentIndex(0)
            elif which == "trophy":
                self.trophy_box.setChecked(False)
            elif which == "lands":
                self.min_lands.setValue(0)
                self.max_lands.setValue(0)
        finally:
            self._filling = False
        self._rebuild_chips()
        self._refresh_advanced_label()
        if self.searched is not None:
            self.request_search("new")

    def _toggle_inc_copies(self, idx: int) -> None:
        if 0 <= idx < len(self._include_cards):
            nm, mn = self._include_cards[idx]
            self._include_cards[idx] = (nm, 2 if mn == 1 else 1)
            self._rebuild_chips()
            self.request_search("new")

    def _remove_chip(self, kind: str, idx: int) -> None:
        if kind == "include" and 0 <= idx < len(self._include_cards):
            self._include_cards.pop(idx)
        elif kind == "exclude" and 0 <= idx < len(self._exclude_cards):
            self._exclude_cards.pop(idx)
        elif kind == "pool" and 0 <= idx < len(self._pool_only):
            self._pool_only.pop(idx)
        self._rebuild_chips()
        self._refresh_advanced_label()
        if self.searched is not None:
            self.request_search("new")

    # ------------------------------------------------------------------ sets
    def refresh_sets(self, statuses: list) -> None:
        """The Sets tab's list changed: keep the sets that have a corpus; a rebuilt set drops its names and results."""
        self.statuses = {st.set_code: st for st in statuses if st.has_corpus}
        prev = self.set_code() or str(new_settings().value("decksearch/set", "") or "")
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for code in self.statuses:
            self.set_combo.addItem(code, code)
        self.set_combo.setCurrentIndex(self.set_combo.findData(prev) if self.set_combo.findData(prev) >= 0 else (0 if self.statuses else -1))
        self.set_combo.blockSignals(False)
        if not self.statuses:
            self._reset_results()
            self._show_empty("No set has data to search yet.  Calculate a set on the Sets tab first.")
        elif self._meta_key is not None and self._meta_key != self._key():
            self._meta_key = None
            self._values_cache = {k: v for k, v in self._values_cache.items() if k[0] != self.set_code()}
            self._reset_results()
            self._show_empty(INTRO)
        self._refresh_pool_banner()
        self._update_controls()
        if self.isVisible():
            self._pump()

    def _key(self) -> tuple[str, str] | None:
        sc = self.set_code()
        st = self.statuses.get(sc)
        return (sc, str(st.built_at or "")) if st is not None else None

    def set_calculating(self, sets: set[str]) -> None:
        self.calculating = {s.upper() for s in sets}
        self._update_controls()
        if self.set_code() in self.calculating:
            self._pending = None
            self._reset_results()
            self._show_empty(f"{self.set_code()} is being recalculated: its decks can be searched again when the calculation finishes.")
        elif self.statuses and self.message.isVisible() and "recalculated" in self.message.text():
            self._show_empty(INTRO)
        if self.isVisible():
            self._pump()

    def _apply_set(self) -> None:
        new_settings().setValue("decksearch/set", self.set_code())
        self._pending = None
        self.debounce.stop()
        self._include_cards.clear()
        self._exclude_cards.clear()
        self._pool_only.clear()
        self._rebuild_chips()
        self._refresh_advanced_label()
        self._reset_results()
        self._say("")
        self._refresh_pool_banner()
        self._show_empty(INTRO if self.mode == "decks" else INTRO_SIMILAR if self.mode == "similar" else INTRO_REBUILDS if self.mode == "rebuilds" else INTRO_SB)
        self._update_controls()
        if self.isVisible():
            self._pump()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._pump)

    # ------------------------------------------------------------------ names and formats
    def _fill_meta(self, key: tuple[str, str], cards: list[str], events: list[dict]) -> None:
        self._meta_key = key
        self.cards = list(cards)
        self.completer.setModel(QStringListModel(self.cards, self.completer))
        self.events = list(events)
        old = {k: b.isChecked() for k, b in self._format_boxes.items()}
        while self.format_flow.count():
            it = self.format_flow.takeAt(0)
            if it is not None and it.widget() is not None:
                it.widget().deleteLater()
        self._format_boxes = {}
        self._filling = True
        for ev in self.events:
            box = QCheckBox(f"{ev['label']} ({count_text(int(ev.get('n_runs', 0)))})")
            box.setToolTip(f"{count_text(int(ev.get('n_runs', 0)))} runs, {count_text(int(ev.get('n_games', 0)))} games in the data")
            box.setChecked(old.get(ev["event"], True))
            box.toggled.connect(self._on_filter_changed)
            self.format_flow.addWidget(box)
            self._format_boxes[ev["event"]] = box
        self._filling = False

    def selected_events(self) -> list[str] | None:
        """None while every format is ticked (no filter), else the ticked event names."""
        if not self._format_boxes:
            return None
        on = [e for e, b in self._format_boxes.items() if b.isChecked()]
        return None if len(on) == len(self._format_boxes) else on

    # ------------------------------------------------------------------ searching
    def _on_completed(self, text: str) -> None:
        self.card_edit.setText(text)
        QTimer.singleShot(0, lambda: self.request_search("new"))

    def _on_sort_chosen(self, _i: int) -> None:
        self._col_sort = None
        self.tree.header().setSortIndicatorShown(False)
        new_settings().setValue("decksearch/sort", self.sort_combo.currentData())
        self._on_filter_changed()

    def _on_filter_changed(self, *_a) -> None:
        if self._filling:
            return
        new_settings().setValue("decksearch/min_games", self.min_games.value())
        new_settings().setValue("decksearch/min_rebuild_changes", self.min_changes.value())
        self._rebuild_chips()
        self._refresh_advanced_label()
        if self.searched is not None:
            self.debounce.start()

    def request_search(self, kind: str = "new") -> None:
        """Ask for a search ("new" = from the first run, "more" = the next page); it starts as soon as the task is free."""
        if kind == "more" and self.debounce.isActive():
            kind = "new"
        self.debounce.stop()
        if not self._usable():
            return
        if kind == "more" and self.searched is None:
            return
        if self.mode == "similar" and not self._has_matching_pool():
            self._say("Load a Sealed pool for this set on the Deck builder tab first.", warn=False)
            return
        self._pending = "more" if kind == "more" and self._pending != "new" else "new"
        self._pump()

    def _extra_deck_filters(self) -> dict:
        out: dict[str, Any] = {}
        if self._include_cards:
            out["include_cards"] = list(self._include_cards)
        if self._exclude_cards:
            out["exclude_cards"] = list(self._exclude_cards)
        if self._pool_only:
            out["in_pool_only"] = list(self._pool_only)
        if self.color_pick.values():
            out["colors"] = "".join(self.color_pick.values())
            out["color_match"] = self.color_pick.mode() or "exact"
        if self.splash_combo.currentData():
            out["splash"] = str(self.splash_combo.currentData())
        if self.skill_combo.currentData() is not None:
            out["min_skill"] = float(self.skill_combo.currentData())
        if self.rank_pick.values():
            out["ranks"] = self.rank_pick.values()
        if self.trophy_box.isChecked():
            out["trophy_only"] = True
        if self.min_lands.value() > 0:
            out["min_lands"] = int(self.min_lands.value())
        if self.max_lands.value() > 0:
            out["max_lands"] = int(self.max_lands.value())
        if self._days is not None:
            out["days"] = int(self._days)
        return out

    def _pump(self) -> None:
        """Start what is wanted, one task at a time: first the names of the chosen set, then the search."""
        if self.task.running or not self._usable():
            return
        key = self._key()
        if key is None:
            return
        sc = key[0]
        if not self._meta_ok() or self._meta_key != key:
            if self._meta_failed == key:
                return
            api = self.api
            self.task.run(lambda progress, cancel: ("meta", key, list(api.deck_search_cards(sc)), list(api.deck_search_events(sc))),
                          label=f"deck search names {sc}")
            return
        if self._pending is None:
            return
        kind, self._pending = self._pending, None
        evs = self.selected_events()
        if evs is not None and not evs:
            self._say("Tick at least one format.", warn=False)
            return
        api = self.api
        mode = self.mode
        days = self._days
        self._say("")
        if mode == "decks":
            name = self.card_edit.text().strip()
            if kind == "more" and self.searched:
                name = self.searched[1]
            args = dict(events=evs, sort=str(self.sort_combo.currentData()), min_games=self.min_games.value(),
                        min_card_share=0.5 if self.share_box.isChecked() else 0.0)
            args.update(self._extra_deck_filters())
            offset = len(self.results) if kind == "more" else 0
            if args["sort"] == "model":
                st = self.statuses.get(sc)
                if st is not None and not getattr(st, "has_model", False):
                    self._say(f"Model score needs {sc}'s model: calculate the set on the Sets tab first.", warn=True)
                    return
            scorer = getattr(api, "deck_model_scores", None) if args["sort"] == "model" else None

            def job(progress, cancel):
                if scorer is not None and not api.has_deck_model_scores(sc):
                    scorer(sc, progress=progress, cancel=cancel)
                return ("search", mode, kind, sc, api.search_decks(sc, name or None, limit=PAGE, offset=offset, **args))
            self.task.run(job, label=f"deck search {sc} {name or 'all'}")
        elif mode == "similar":
            pool_copy = dict(self._pool_named)
            ms = self.skill_combo.currentData()
            tr = self.trophy_box.isChecked()
            mg = self.min_games.value()
            self.task.run(lambda progress, cancel: ("search", mode, "new", sc,
                                                    api.search_similar_pools(sc, pool_copy, events=evs,
                                                                             min_skill=ms, trophy_only=tr,
                                                                             limit=80, min_games=mg, days=days)),
                          label=f"similar pools {sc}")
        elif mode == "rebuilds":
            name = self.card_edit.text().strip()
            if kind == "more" and self.searched:
                name = self.searched[1]
            ms = self.skill_combo.currentData()
            ro = self.rescue_box.isChecked()
            po = self.pivot_box.isChecked()
            mc = int(self.min_changes.value())
            s_by, s_desc = ("default", True)
            if self._col_sort is not None and 0 <= self._col_sort[0] < len(REBUILD_COL_SORTS):
                s_by, s_desc = REBUILD_COL_SORTS[self._col_sort[0]], self._col_sort[1]
            offset = len(self.results) if kind == "more" else 0
            self.task.run(lambda progress, cancel: ("search", mode, kind, sc,
                                                    api.search_rebuilds(sc, card=name or None, events=evs,
                                                                        min_skill=ms, rescue_only=ro,
                                                                        color_pivot_only=po, limit=PAGE, offset=offset,
                                                                        days=days, min_changes=mc,
                                                                        sort_by=s_by, sort_desc=s_desc)),
                          label=f"rebuilds {sc}")
        elif mode == "sideboard":
            name = self.card_edit.text().strip()
            my_c = self.sb_my_combo.currentData()
            opp_c = self.sb_opp_combo.currentData()
            op = self.sb_play_combo.currentData()
            ms = self.skill_combo.currentData()
            pcards = list(self._pool_named.keys()) if (self.sb_pool_box.isChecked() and self._has_matching_pool()) else None
            self.task.run(lambda progress, cancel: ("search", mode, "new", sc,
                                                    api.search_sideboarding(sc, card=name or None, events=evs,
                                                                            my_colors=my_c, opp_colors=opp_c,
                                                                            on_play=op, min_skill=ms, limit=120,
                                                                            days=days, pool_cards=pcards)),
                          label=f"bo3 sideboarding {sc}")

    def _score_sealed_decks_and_refresh(self) -> None:
        """Compute cached model scores for the current set (same scorer as Sort -> Model score) and re-run the active search."""
        sc = self.set_code()
        scorer = getattr(self.api, "deck_model_scores", None)
        if not sc or scorer is None or self.task.running:
            return
        st = self.statuses.get(sc)
        if st is not None and not getattr(st, "has_model", False):
            self._say(f"Model score needs {sc}'s model: calculate the set on the Sets tab first.", warn=True)
            return

        def job(progress, cancel):
            scorer(sc, progress=progress, cancel=cancel)
            return ("scored", sc)

        self.task.run(job, label=f"model scores {sc}")

    def _on_started(self) -> None:
        self.strip.begin("Searching…")
        self._update_controls()

    def _on_result(self, res: object) -> None:
        if not isinstance(res, tuple):
            return
        if res[0] == "meta":
            _tag, key, cards, events = res
            if key[0] == self.set_code():
                self._fill_meta(key, cards, events)
            self.strip.end("Ready")
            return
        if res[0] == "scored":
            _tag, sc_done = res
            self.strip.end("Ready")
            if sc_done == self.set_code():
                self._pending = "new"
            return
        _tag, mode, kind, sc, data = res
        if sc != self.set_code() or mode != self.mode or self._pending is not None:
            self.strip.end("Ready")
            return
        self.strip.end("Ready")
        self._show_data(mode, kind, data)

    def _on_error(self, tb: str) -> None:
        if "SetBusy" in tb:
            self.strip.end("Waiting for the calculation")
            self._pending = None
            return
        self.strip.end("Failed")
        last_line = tb.strip().splitlines()[-1] if tb.strip() else ""
        if "ValueError" in last_line:
            name = self.card_edit.text().strip()
            self._reset_results()
            self._say(f"No card called \"{name}\" in this set.  Pick a name from the list that opens as you type.")
            self._show_empty(INTRO)
            return
        if "FileNotFoundError" in last_line:
            self._meta_failed = self._key()
            self._show_empty("This set has no data to search.  Calculate it on the Sets tab first.")
            return
        if self._meta_key is None or not self._meta_ok():
            self._meta_failed = self._key()
        self._show_empty("The search failed (see the error message).")
        self.errorOccurred.emit(short_error(tb), tb)

    def _on_ended(self) -> None:
        self._update_controls()
        self._pump()

    # ------------------------------------------------------------------ results & insights rendering
    def _reset_results(self) -> None:
        self.results, self.data, self.searched = [], None, None
        self._selected_build_idx = None
        self.tree.clear()
        self._clear_deck()
        self._clear_insights()
        self.insights_box.setVisible(False)
        self.summary.setText("")

    def _clear_insights(self) -> None:
        while self.insights_lay.count():
            it = self.insights_lay.takeAt(0)
            if it is not None and it.widget() is not None:
                it.widget().deleteLater()

    def _show_data(self, mode: str, kind: str, data: dict) -> None:
        card = str(data.get("card") or (self.card_edit.text().strip() if mode != "similar" else ""))
        rows_key = "matches" if mode == "sideboard" else "results"
        new_rows = list(data.get(rows_key) or [])
        if kind == "more" and self.searched == (data.get("set_code", self.set_code()), card):
            self.results += new_rows
            start = self.tree.topLevelItemCount()
            self._add_rows(mode, start)
            self.data = data
        else:
            self.results = new_rows
            self.data = data
            self.searched = (self.set_code(), card)
            self._selected_build_idx = None
            self.tree.clear()
            self._add_rows(mode, 0)
            self._clear_deck()

        if not self.results:
            if mode == "decks" and card and not (self._include_cards or self._exclude_cards or self._pool_only):
                self._show_empty(f"No deck with {card} matches these filters.")
            elif mode == "similar":
                self._show_empty("No matching Sealed pools found in this set's corpus.")
            elif mode == "rebuilds":
                self._show_empty("No mid-run rebuilds match these filters.")
            elif mode == "sideboard":
                self._show_empty("No Bo3 sideboarded matches found for these filters.")
            else:
                self._show_empty("No decks match these filters.")
            self._update_controls()
            return

        self._show_body()
        self._render_insights(mode, data)
        since = f" · since {data['date_cutoff']}" if data.get("date_cutoff") else ""
        if mode == "decks":
            total = int(data.get("total", len(self.results)))
            subject = f"decks with {card}" if card else "decks"
            self.summary.setText(f"{count_text(total)} {subject} (showing {count_text(len(self.results))}){since} · sideboards stored for Sealed only")
        elif mode == "similar":
            total = int(data.get("total", len(self.results)))
            self.summary.setText(f"{count_text(total)} Sealed pools compared · showing top {count_text(len(self.results))} closest to your {data.get('n_pool_cards', 0)}-card pool{since}")
        elif mode == "rebuilds":
            total = int(data.get("total", len(self.results)))
            if self._col_sort is not None and 0 <= self._col_sort[0] < len(COLUMNS_REBUILDS):
                c_name = COLUMNS_REBUILDS[self._col_sort[0]]
                arrow = "↓" if self._col_sort[1] else "↑"
                tail = f"sorted by {c_name} {arrow}"
            else:
                tail = "rescues & biggest model upgrades first"
            self.summary.setText(f"{count_text(total)} rebuilt runs (showing {count_text(len(self.results))}){since} · {tail}")
        else:
            nb = int(data.get("boarded_matches", 0))
            nt = int(data.get("total_postboard", 0))
            self.summary.setText(f"{count_text(nb)} sideboarded Bo3 matches of {count_text(nt)} ({_pct(data.get('boarded_rate'))}) · showing {count_text(len(self.results))}{since}")

        # Re-apply active local column sort when rows arrive (Decks non-dropdown / Similar / Sideboarding)
        if self._col_sort is not None and mode != "rebuilds":
            scol, sdesc = self._col_sort
            if not (mode == "decks" and (scol == 2 or (scol in (0, 8) and sdesc))):
                self._apply_local_sort(scol, sdesc)

        if kind != "more" or self._selected() is None:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._update_controls()

    def _render_insights(self, mode: str, data: dict) -> None:
        self._clear_insights()
        if mode == "decks":
            an = data.get("analytics")
            if not isinstance(an, dict) or not an.get("n_runs"):
                self.insights_box.setVisible(False)
                return
            self.insights_headline.setText(
                f"{count_text(an['n_runs'])} runs · {count_text(an['n_games'])} games · "
                f"WR {_pct(an.get('avg_wr'))} · Trophy {_pct(an.get('trophy_rate'))}"
            )
            bc = an.get("by_color") or []
            if bc:
                cw_col = QWidget()
                cf_col = FlowLayout(cw_col, spacing=5)
                chdr = QLabel("<b>Colors:</b>")
                chdr.setToolTip("Raw game win rate of the matching runs per color pair, and the average 17Lands win-rate bucket of their "
                                "pilots (who played it). Not adjusted for skill: stronger pilots raise a pair's win rate.\n"
                                "Model (once Sealed decks are scored, see Sort → Model score): the median and the best model score of "
                                "the pair's matching Sealed decks.  Click any color pill to filter to that exact color pair.")
                cf_col.addWidget(chdr)
                for c in bc[:6]:
                    lbl = (f"{c['colors']} {_pct(c['raw_wr'])} ({count_text(c['runs'])}"
                           + (f", pilots {_pct(c['avg_skill'], 0)}" if c.get("avg_skill") is not None else "")
                           + (f", model {c['model_median']:.0f}/{c['model_best']:.0f}" if c.get("model_median") is not None else "")
                           + ")")
                    btn = QPushButton(lbl)
                    btn.setStyleSheet(_pill_style(compact=True))
                    btn.setToolTip(
                        f"{c['colors']}: {_pct(c['raw_wr'])} raw WR across {count_text(c['runs'])} runs"
                        + (f", pilots avg {_pct(c['avg_skill'], 0)}" if c.get("avg_skill") is not None else "")
                        + (f", Sealed model score median {c['model_median']:.0f} / best {c['model_best']:.0f}" if c.get("model_median") is not None else "")
                        + ".  Click to filter to this exact color pair."
                    )
                    btn.clicked.connect(lambda _chk=False, pair=c["colors"]: self._filter_color_pair(pair))
                    cf_col.addWidget(btn)
                self.insights_lay.addWidget(cw_col)
            extra_bits = []
            ms = an.get("main_vs_splash")
            if ms:
                m, s = ms.get("main") or {}, ms.get("splash") or {}
                if m.get("runs") and s.get("runs"):
                    extra_bits.append(f"<b>Main vs Splash:</b> Main {_pct(m.get('wr'))} ({count_text(m['runs'])}) vs Splash {_pct(s.get('wr'))} ({count_text(s['runs'])})")
            cp = an.get("by_copies") or []
            if len(cp) > 1:
                extra_bits.append("<b>Copies:</b> " + ", ".join(f"{x['copies']}× {_pct(x.get('wr'))} ({count_text(x['runs'])})" for x in cp))
            ig = an.get("inclusion_gap")
            if ig and ig.get("all_n"):
                s_ig = f"<b>Sealed play rate (on-color):</b> All {_pct(ig.get('all_rate'), 0)} (n={count_text(ig['all_n'])})"
                if ig.get("top_rate") is not None:
                    s_ig += f" vs Top ≥60% pilots {_pct(ig.get('top_rate'), 0)} (n={count_text(ig['top_n'])})"
                extra_bits.append(s_ig)
            if extra_bits:
                lab = QLabel("  ·  ".join(extra_bits))
                lab.setWordWrap(True)
                self.insights_lay.addWidget(lab)
            pp = an.get("play_profile")
            if isinstance(pp, dict) and pp.get("n_games"):
                p_bits = []
                if pp.get("oh_delta") is not None:
                    p_bits.append(f"<b>Opening hand:</b> {_pct(pp.get('oh_wr'))} ({_pp(pp.get('oh_delta'))} vs not in opener, {count_text(pp['oh_games'])}g)")
                if pp.get("drawn_wr") is not None and pp.get("drawn_games"):
                    # no "vs unseen" difference: longer games see more cards, so drawn-vs-unseen mostly measures game length
                    p_bits.append(f"<b>Drawn later:</b> {_pct(pp.get('drawn_wr'))} ({count_text(pp['drawn_games'])}g)")
                rep = pp.get("replay")
                if isinstance(rep, dict):
                    rb = []
                    if rep.get("cand1_keep_rate") is not None:
                        kd = rep.get("keep_delta")
                        rb.append(f"1st-hand keep {_pct(rep['cand1_keep_rate'], 0)}"
                                  + (f" ({_pp(kd)} vs avg)" if kd is not None else ""))
                    if rep.get("mean_cast_turn") is not None:
                        rb.append(f"avg cast T{rep['mean_cast_turn']:.1f}")
                    if rep.get("cast_rate") is not None:
                        rb.append(f"cast {_pct(rep['cast_rate'], 0)} when seen")
                    hist = rep.get("turn_hist") or []
                    if hist:
                        top_t = sorted(hist, key=lambda h: -h["count"])[:3]
                        rb.append("peak " + "/".join(f"T{h['turn']} ({_pct(h['share'], 0)})" for h in top_t))
                    if rb:
                        p_bits.append("<b>Replay:</b> " + ", ".join(rb))
                if p_bits:
                    lab = QLabel("  ·  ".join(p_bits))
                    lab.setWordWrap(True)
                    lab.setToolTip("Within-deck game outcomes (hands.parquet) and replay stats. Opening hand: games where the card was in "
                                   "the kept opening hand vs games of the same decks where it was not. 1st-hand keep: how often players kept "
                                   "their initial 7-card hand (candidate_hand_1, before any mulligan) when it held this card vs all replay "
                                   "games of the set. Drawn later: win rate only (comparing with unseen games would mostly measure game length). "
                                   "Descriptive, not a causal effect.")
                    self.insights_lay.addWidget(lab)
            comps = an.get("companions") or []
            if comps:
                cw = QWidget()
                cf = FlowLayout(cw, spacing=5)
                hdr = QLabel("<b>Strong pilots also play:</b>")
                hdr.setToolTip("Cards in the matching decks of pilots with a 17Lands win rate of 60% or more, compared with decks of the same "
                               "color pairs in the same events. What strong pilots choose, not proof that the card wins.")
                cf.addWidget(hdr)
                for c in comps:
                    b = QPushButton(f"+ {c['name']} ({_pct(c['pct_in_matches'], 0)}, {c['lift']:.1f}×)")
                    b.setStyleSheet(_pill_style(compact=True))
                    b.setToolTip(f"In {_pct(c['pct_in_matches'])} of strong pilots' matching decks vs {_pct(c.get('base_pct'))} of decks of the "
                                 f"same color pairs ({c['lift']:.2f}×). Click to add as +Include filter.")
                    b.clicked.connect(lambda _chk=False, nm=c["name"]: self._add_chip("include", nm))
                    cf.addWidget(b)
                self.insights_lay.addWidget(cw)
            self.insights_box.setVisible(True)

        elif mode == "similar":
            cb = data.get("color_breakdown") or []
            if not cb:
                self.insights_box.setVisible(False)
                return
            self.insights_headline.setText("How 17Lands pilots built the closest real pools")
            bits = [f"<b>{c['colors']}</b> {_pct(c['share'], 0)} ({count_text(c['runs'])} pools, WR {_pct(c.get('avg_wr'))}"
                    + (f", model {c['model_median']:.0f} / best {c['model_best']:.0f}" if c.get("model_median") is not None else "")
                    + ")" for c in cb[:6]]
            lab = QLabel("<b>Most common builds:</b> " + "  ·  ".join(bits))
            lab.setWordWrap(True)
            self.insights_lay.addWidget(lab)
            self.insights_box.setVisible(True)

        elif mode == "rebuilds":
            an = data.get("analytics") or {}
            if not an.get("n_multi"):
                self.insights_box.setVisible(False)
                return
            mc_n = int(an.get("min_changes") or self.min_changes.value())
            self.insights_headline.setText(
                f"{count_text(an['n_multi'])} rebuilt runs (≥{mc_n} cards) · {_pct(an.get('pct_color_pivot'), 0)} color pivots · "
                f"{count_text(an.get('n_rescues', 0))} 0–2 rescues"
            )
            if an.get("n_scored"):
                b0m, b1m, dm = an["median_before_score"], an["median_after_score"], an["median_score_delta"]
                imp = _pct(an.get("pct_improved"), 0)
                lab = QLabel(
                    f"<b>Did rebuilding improve the deck? (Sealed only, {count_text(an['n_scored'])} runs):</b> "
                    f"median model score <b>{b0m:.0f} → {b1m:.0f}</b> (median change {dm:+.0f} · {imp} built a stronger deck)"
                )
                lab.setWordWrap(True)
                self.insights_lay.addWidget(lab)
                sub = muted(QLabel(
                    "Grades each Sealed run's first build vs its final rebuilt deck on the model's 0–100 deck-strength scale "
                    "(50 = average real build) to measure deck quality without post-0–2 win-rate luck. Draft runs are not scored "
                    "because the model is Sealed-only."
                ))
                sub.setWordWrap(True)
                self.insights_lay.addWidget(sub)
            else:
                st = self.statuses.get(self.set_code())
                has_model = bool(st is not None and getattr(st, "has_model", False))
                row_w = QWidget()
                rl = QHBoxLayout(row_w)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.setSpacing(8)
                if has_model and not an.get("has_model_scores"):
                    msg = muted(QLabel("Score this set's Sealed decks once to see unbiased before → after model score changes."))
                    msg.setWordWrap(True)
                    rl.addWidget(msg, 1)
                    sbtn = QPushButton("Score Sealed decks")
                    sbtn.setStyleSheet(_pill_style(compact=True))
                    sbtn.setToolTip(MODEL_SORT_TIP)
                    sbtn.clicked.connect(self._score_sealed_decks_and_refresh)
                    rl.addWidget(sbtn)
                else:
                    msg = muted(QLabel("Before → after model score is available for Sealed runs once the set has a calculated model."))
                    msg.setWordWrap(True)
                    rl.addWidget(msg, 1)
                self.insights_lay.addWidget(row_w)
            self.insights_box.setVisible(True)

        elif mode == "sideboard":
            cs = data.get("card_summary")
            if cs is not None:
                self.insights_headline.setText(
                    f"Boarded IN {count_text(cs['n_in'])}× ({_pct(cs['in_rate'])}) · "
                    f"OUT {count_text(cs['n_out'])}× ({_pct(cs['out_rate'])})"
                )
                in_ci_s = _wr_ci(cs.get("in_post_wr"), cs.get("in_post_lo"), cs.get("in_post_hi"), cs.get("in_post_games"))
                out_ci_s = _wr_ci(cs.get("out_post_wr"), cs.get("out_post_lo"), cs.get("out_post_hi"), cs.get("out_post_games"))
                bits = [
                    f"<b>Post-board WR (descriptive):</b> IN {in_ci_s} · OUT {out_ci_s}",
                    f"<b>Play/Draw (IN):</b> {cs['in_on_play']} play / {cs['in_on_draw']} draw",
                ]
                if cs.get("top_in_rate") is not None:
                    bits.append(f"<b>Strong ≥60% pilots:</b> IN {_pct(cs['top_in_rate'])} / OUT {_pct(cs['top_out_rate'])}")
                lab = QLabel("  ·  ".join(bits))
                lab.setWordWrap(True)
                lab.setToolTip("Post-board win rate with 95% Wilson interval and game count. Descriptive only: Bo3 boarding depends on "
                               "Game 1's result and Game 3 only happens after a 1–1 split.")
                self.insights_lay.addWidget(lab)
                bo = data.get("by_opp_colors") or []
                if bo:
                    ob = [f"<b>vs {x['opp_colors']}</b> (+{x['n_in']}/−{x['n_out']})" for x in bo[:6]]
                    olab = QLabel("<b>By G1 opponent colors:</b> " + "  ·  ".join(ob))
                    olab.setWordWrap(True)
                    self.insights_lay.addWidget(olab)
                cc = data.get("common_cut_for") or []
                ci = data.get("common_in_for") or []
                if cc or ci:
                    sb_bits = []
                    if cc:
                        sb_bits.append("<b>Often cut when brought IN:</b> " + ", ".join(f"{x['name']} ({x['count']})" for x in cc[:4]))
                    if ci:
                        sb_bits.append("<b>Often brought in when cut:</b> " + ", ".join(f"{x['name']} ({x['count']})" for x in ci[:4]))
                    slab = QLabel("  ·  ".join(sb_bits))
                    slab.setWordWrap(True)
                    self.insights_lay.addWidget(slab)
                self.insights_box.setVisible(True)
            else:
                tin, tout = data.get("top_in") or [], data.get("top_out") or []
                if not tin and not tout:
                    self.insights_box.setVisible(False)
                    return
                by_strong = data.get("sorted_by") == "strong"
                p_note = " · filtered to your pool" if data.get("pool_filtered") else ""
                sort_note = "strong-pilot (≥60%) swap frequency" if by_strong else "pilot swap frequency"
                self.insights_headline.setText(f"Matchup Sideboard Plan — sorted by {sort_note}{p_note}")
                for heading, items in (("Most boarded IN:", tin[:8]), ("Most boarded OUT:", tout[:8])):
                    if not items:
                        continue
                    w = QWidget()
                    fl = FlowLayout(w, spacing=5)
                    fl.addWidget(QLabel(f"<b>{heading}</b>"))
                    for x in items:
                        if by_strong and x.get("strong_rate") is not None:
                            btn_txt = f"{x['name']} ({x['strong_count']}×, {_pct(x['strong_rate'], 0)} strong)"
                        else:
                            btn_txt = f"{x['name']} ({x['count']}×, {_pct(x['rate'], 0)})"
                        b = QPushButton(btn_txt)
                        b.setStyleSheet(_pill_style(compact=True))
                        wr_s = _wr_ci(x.get("post_wr"), x.get("post_wr_lo"), x.get("post_wr_hi"), x.get("post_games"))
                        strong_s = (f"Strong ≥60% pilots: {x['strong_count']}× ({_pct(x.get('strong_rate'))}). "
                                    if x.get("strong_rate") is not None else "")
                        b.setToolTip(
                            f"{strong_s}All pilots: {x['count']}× ({_pct(x['rate'])}).\n"
                            f"Post-board WR: {wr_s} (descriptive only — confounded by Game 1 result and Game 3 selection).\n"
                            f"Click to drill into this card's sideboarding."
                        )
                        b.clicked.connect(lambda _chk=False, nm=x["name"]: self._drill_sb_card(nm))
                        fl.addWidget(b)
                    self.insights_lay.addWidget(w)
                self.insights_box.setVisible(True)

    def _drill_sb_card(self, name: str) -> None:
        self.card_edit.setText(name)
        self.request_search("new")

    def _add_rows(self, mode: str, start: int) -> None:
        for i in range(start, len(self.results)):
            r = self.results[i]
            sk = _pct(r.get("skill"), 0) if r.get("skill") is not None else ""
            if mode == "decks":
                rec = (f"{TROPHY} " if r.get("trophy") else "") + str(r.get("record", ""))
                n, g = int(r.get("games_with_card", 0)), int(r.get("games", 0))
                cp = str(r.get("copies", "")) if r.get("copies") else "—"
                ms, mq = r.get("model_score"), r.get("model_q")
                ms_s = "" if ms is None else (f"{ms:.0f}" + ("" if mq is None else f" ({mq * 100:.1f}%)"))
                it = QTreeWidgetItem([rec, str(r.get("event_label", "")), str(r.get("date", "")), str(r.get("colors", "")),
                                      cp, f"{n} / {g}", str(r.get("rank") or ""), sk, ms_s])
                if ms is not None:
                    it.setToolTip(8, f"Model score {ms:.0f}: better than {ms:.0f}% of the real builds of this format by the set model"
                                     + ("" if mq is None else f"; the model's game win rate for this deck is {mq * 100:.1f}%") + ".")
                elif r.get("fmt") not in ("bo1", "bo3"):
                    it.setToolTip(8, "Draft run: not scored because the set model is Sealed-only.")
                if r.get("trophy"):
                    it.setToolTip(0, "Reached the event's maximum wins")
                    it.setForeground(0, QColor(semantic()["good"]))
            elif mode == "similar":
                rec = (f"{TROPHY} " if r.get("trophy") else "") + str(r.get("record", ""))
                sim_s = _pct(r.get("similarity"), 0)
                sh = f"{r.get('shared_count', 0)} cards"
                ms, mq = r.get("model_score"), r.get("model_q")
                ms_s = "" if ms is None else (f"{ms:.0f}" + ("" if mq is None else f" ({mq * 100:.1f}%)"))
                it = QTreeWidgetItem([sim_s, rec, str(r.get("event_label", "")), str(r.get("colors", "")),
                                      sh, str(r.get("date", "")), sk, ms_s])
                if ms is not None:
                    it.setToolTip(7, f"Model score {ms:.0f}" + ("" if mq is None else f" ({mq * 100:.1f}% model game WR)"))
                if r.get("trophy"):
                    it.setForeground(1, QColor(semantic()["good"]))
            elif mode == "rebuilds":
                rec = (f"{TROPHY} " if r.get("trophy") else "") + str(r.get("record", ""))
                ba = f"{r.get('b0_record', '')} → {r.get('after_record', '')}"
                if r.get("is_rescue"):
                    ba = "★ " + ba
                b0m, b1m, md = r.get("b0_model_score"), r.get("b1_model_score"), r.get("model_score_delta")
                is_draft = r.get("fmt") not in ("bo1", "bo3")
                if b0m is not None and b1m is not None and md is not None:
                    md_s = f"{b0m:.0f} → {b1m:.0f} ({md:+.0f})"
                else:
                    md_s = "— (draft)" if is_draft else "—"
                it = QTreeWidgetItem([rec, ba, md_s, str(r.get("event_label", "")), str(r.get("colors", "")),
                                      f"{r.get('cards_changed', 0)} cards", str(r.get("date", "")), sk])
                if r.get("is_rescue"):
                    it.setForeground(1, QColor(semantic()["good"]))
                if md is not None:
                    it.setForeground(2, QColor(semantic()["good" if md >= 0 else "bad"]))
                    it.setToolTip(2, f"Model score changed from {b0m:.0f} to {b1m:.0f} ({md:+.1f} percentile points)")
                elif is_draft:
                    it.setForeground(2, self.tree.palette().placeholderText())
                    it.setToolTip(2, "Draft run: not scored because the set model is Sealed-only.")
            else:
                g2s = "Play" if r.get("g2_on_play") else "Draw"
                mu = f"{r.get('colors', '')} vs {r.get('opp_colors', '?')}"
                it = QTreeWidgetItem([str(r.get("record", "")), str(r.get("event_label", "")), mu, g2s,
                                      f"±{r.get('n_swapped', 0)}", str(r.get("date", "")), sk])
                if r.get("match_won"):
                    it.setForeground(0, QColor(semantic()["good"]))
            it.setData(0, Qt.ItemDataRole.UserRole, i)
            self.tree.addTopLevelItem(it)

    def _selected(self) -> dict | None:
        its = self.tree.selectedItems()
        if not its:
            return None
        i = its[0].data(0, Qt.ItemDataRole.UserRole)
        return self.results[i] if isinstance(i, int) and 0 <= i < len(self.results) else None

    # ------------------------------------------------------------------ model values overlay
    def _values_fmt_for(self, sc: str, r: dict) -> str | None:
        st = self.statuses.get(sc)
        if st is None or not getattr(st, "has_model", False):
            return None
        fmts = list(getattr(st, "formats", []) or [])
        if not fmts:
            return None
        rf = str(r.get("fmt") or "")
        if rf in fmts:
            return rf
        return "bo3" if "bo3" in fmts else fmts[0]

    def _ensure_values(self, sc: str, fmt: str) -> dict[str, tuple[float, bool]]:
        key = (sc, fmt)
        if key in self._values_cache:
            return self._values_cache[key]
        if key not in self._values_loading and not self.val_task.running and hasattr(self.api, "card_values"):
            self._values_loading.add(key)
            api = self.api

            def job(progress, cancel):
                info = api.card_info(sc) or {}
                vals = api.card_values(sc, fmt) or {}
                by_name: dict[str, tuple[float, bool]] = {}
                for cid, v in vals.items():
                    d = info.get(int(cid))
                    if isinstance(d, dict) and d.get("name"):
                        val_t = (25.0 * float(v.get("beta", 0.0)), bool(v.get("low_data", False)))
                        full_nm = str(d["name"]).strip()
                        by_name[full_nm.casefold()] = val_t
                        if " // " in full_nm:
                            by_name.setdefault(full_nm.split(" // ", 1)[0].strip().casefold(), val_t)
                return key, by_name

            self.val_task.run(job, label=f"card values {sc} {fmt}")
        return {}

    def _on_values_loaded(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        key, by_name = payload
        self._values_loading.discard(key)
        self._values_cache[key] = by_name
        if self.values_box.isChecked() and key[0] == self.set_code():
            for pile in self.piles:
                self._fill_pile(pile)

    def _on_values_toggled(self, on: bool) -> None:
        new_settings().setValue("decksearch/show_values", bool(on))
        r = self._selected()
        if r is not None:
            sc = self.set_code()
            fmt = self._values_fmt_for(sc, r)
            if on and fmt:
                self._ensure_values(sc, fmt)
            for pile in self.piles:
                self._fill_pile(pile)

    def _values_for_pile(self, sc: str, info: dict[int, dict]) -> dict[int, tuple[float, bool]]:
        if not self.values_box.isChecked():
            return {}
        r = self._selected()
        if r is None:
            return {}
        fmt = self._values_fmt_for(sc, r)
        if not fmt:
            return {}
        by_name = self._ensure_values(sc, fmt)
        if not by_name:
            return {}
        out: dict[int, tuple[float, bool]] = {}
        for cid, d in info.items():
            nm = str(d.get("name", "")).casefold()
            if nm in by_name:
                out[cid] = by_name[nm]
        return out

    # ------------------------------------------------------------------ the deck list pane
    def _clear_deck(self) -> None:
        self.hover_popup.hide()
        self.piles = []
        while self.deck_lay.count():
            it = self.deck_lay.takeAt(0)
            if it is not None and it.widget() is not None:
                it.widget().deleteLater()

    def _on_selected(self) -> None:
        self._selected_build_idx = None
        self._render_selected()

    def _render_selected(self) -> None:
        self._update_controls()
        r = self._selected()
        if r is None:
            return
        self._clear_deck()
        lay, sc = self.deck_lay, self.set_code()

        rec = QLabel((f"{TROPHY} " if r.get("trophy") else "") + str(r.get("record", "")))
        rec.setFont(sized_font(self.font(), 4, bold=True))
        lay.addWidget(rec)
        sk_s = f"Pilot {_pct(r.get('skill'), 0)}" if r.get("skill") is not None else ""
        bits = [r.get("event_label"), r.get("date"), r.get("colors"), r.get("rank"), sk_s]
        lay.addWidget(self._note(" · ".join(str(b) for b in bits if b)))

        if self.mode == "decks":
            card_nm = self.searched[1] if self.searched else ""
            if card_nm:
                lay.addWidget(self._note(
                    f"{r.get('n_cards', 0)} cards, {r.get('n_lands', 0)} lands · {r.get('copies', 0)} × {card_nm}, "
                    f"in the deck for {r.get('games_with_card', 0)} of {r.get('games', 0)} games"
                ))
            else:
                lay.addWidget(self._note(f"{r.get('n_cards', 0)} cards, {r.get('n_lands', 0)} lands"))
            if int(r.get("n_builds", 1) or 1) > 1:
                bw, bg = int(r.get("build_wins", 0)), int(r.get("build_games", 0))
                which = "the build with the card" if card_nm else "the main build"
                lay.addWidget(self._note(f"The deck changed during the run: showing {which}, {bw}–{bg - bw} in its {bg} games.", wrap=True))
            self._render_deck_piles(lay, sc, list(r.get("main") or []), r.get("sideboard"), r.get("sideboard_note"))

        elif self.mode == "similar":
            ms_s = f" · Model score {r['model_score']:.0f}" if r.get("model_score") is not None else ""
            lay.addWidget(self._note(
                f"Similarity {_pct(r.get('similarity'), 0)} · {r.get('shared_count', 0)} shared non-basic cards · "
                f"{r.get('n_cards', 0)} cards, {r.get('n_lands', 0)} lands{ms_s}"
            ))
            sh = r.get("shared_cards") or []
            if sh:
                lay.addWidget(self._note("Top shared cards: " + ", ".join(sh), wrap=True))
            self._render_deck_piles(lay, sc, list(r.get("main") or []), r.get("sideboard"), r.get("sideboard_note"))

        elif self.mode == "rebuilds":
            b0m, b1m, md = r.get("b0_model_score"), r.get("b1_model_score"), r.get("model_score_delta")
            if b0m is not None and b1m is not None and md is not None:
                lay.addWidget(self._note(f"Model score change: {b0m:.0f} → {b1m:.0f} ({md:+.0f}) · {r.get('cards_changed', 0)} non-basic cards changed"))
            builds = list(r.get("builds") or [])
            if builds:
                idx = len(builds) - 1 if self._selected_build_idx is None else max(0, min(self._selected_build_idx, len(builds) - 1))
                brow = QHBoxLayout()
                brow.addWidget(QLabel("Inspect build:"))
                bcombo = ElidedCombo()
                for j, b in enumerate(builds):
                    bm = f", model {b['model_score']:.0f}" if b.get("model_score") is not None else ""
                    lbl = f"Build {j + 1} ({b['wins']}–{b['losses']}, {b['colors'] or '?'}{bm}" + (f", ±{b['cards_changed']} cards" if j > 0 else ", initial") + ")"
                    bcombo.addItem(lbl, j)
                bcombo.setCurrentIndex(idx)
                bcombo.activated.connect(self._on_build_chosen)
                brow.addWidget(bcombo, 1)
                bw = QWidget()
                bw.setLayout(brow)
                lay.addWidget(bw)
                cur = builds[idx]
                added, cut = list(cur.get("added") or []), list(cur.get("cut") or [])
                if added:
                    self._pile(lay, f"Added vs previous build ({sum(c['n'] for c in added)})", added, sc, SIDE_SCALE, zone="diff_in", mark="+")
                if cut:
                    self._pile(lay, f"Cut vs previous build ({sum(c['n'] for c in cut)})", cut, sc, SIDE_SCALE, zone="diff_out", mark="-")
                self._render_deck_piles(lay, sc, list(cur.get("main") or []), r.get("sideboard"), r.get("sideboard_note"))
            else:
                self._render_deck_piles(lay, sc, list(r.get("main") or []), r.get("sideboard"), r.get("sideboard_note"))

        elif self.mode == "sideboard":
            g1s = "Won G1" if r.get("g1_won") else "Lost G1"
            g2s = "G2 on the play" if r.get("g2_on_play") else "G2 on the draw"
            lay.addWidget(self._note(f"vs Opp G1 {r.get('opp_colors', '?')} · {g1s} · {g2s} · ±{r.get('n_swapped', 0)} non-basic swaps"))
            inc, outc = list(r.get("in_cards") or []), list(r.get("out_cards") or [])
            if inc:
                self._pile(lay, f"Boarded IN ({sum(c['n'] for c in inc)})", inc, sc, SIDE_SCALE, zone="diff_in", mark="+")
            if outc:
                self._pile(lay, f"Boarded OUT ({sum(c['n'] for c in outc)})", outc, sc, SIDE_SCALE, zone="diff_out", mark="-")
            self._render_deck_piles(lay, sc, list(r.get("main") or []), r.get("sideboard"), r.get("sideboard_note"))

        lay.addStretch(1)

    def _on_build_chosen(self, index: int) -> None:
        self._selected_build_idx = index
        self._render_selected()

    def _render_deck_piles(self, lay: QVBoxLayout, sc: str, main: list[dict], side: list[dict] | None, sb_note: str | None) -> None:
        lay.addSpacing(6)
        self.store.prefetch(sc, [str(c.get("name", "")) for c in main + (side if isinstance(side, list) else [])])
        basics = [c for c in main if c.get("is_basic")]
        if basics:
            lay.addWidget(self._note("Basic lands: " + ", ".join(f"{c.get('n', 0)} {c.get('name', '')}" for c in basics), wrap=True))
        self._pile(lay, f"Main deck ({sum(int(c.get('n', 0)) for c in main)})", [c for c in main if not c.get("is_basic")], sc, 1.0)
        if isinstance(side, list):
            if side:
                self._pile(lay, f"Sideboard ({sum(int(c.get('n', 0)) for c in side)})", side, sc, SIDE_SCALE, zone="side")
        elif sb_note:
            lay.addSpacing(6)
            lay.addWidget(self._note(str(sb_note), wrap=True))

    def _note(self, text: str, wrap: bool = False) -> QLabel:
        lab = muted(QLabel(text))
        lab.setWordWrap(wrap)
        return lab

    def _card_w(self) -> int:
        return int(self.size_slider.value())

    def _pile(self, lay: QVBoxLayout, title: str, cards: list[dict], sc: str, scale: float,
              zone: str = "deck", mark: str | None = None) -> None:
        """A read-only pile of card images (columns by mana value, creatures apart) under a heading."""
        head = QLabel(title)
        head.setFont(sized_font(self.font(), 1, bold=True))
        lay.addWidget(head)
        targets = set()
        if self.searched and self.searched[1]:
            targets.add(self.searched[1].casefold())
        for nm, _ in self._include_cards:
            targets.add(nm.casefold())
        ids: list[int] = []
        info: dict[int, dict] = {}
        chips: dict[int, str] = {}
        marks: dict[int, str] = {}
        for i, c in enumerate(cards):
            cid = c.get("card_id")
            cid = int(cid) if cid is not None else -1 - i
            info[cid] = dict(c, cost="", produces="")
            ids += [cid] * int(c.get("n", 0))
            if str(c.get("name", "")).casefold() in targets and mark is None:
                chips[cid] = "searched"
            if mark:
                marks[cid] = mark
        pile = ReadOnlyPile(zone, self.store, self.deck_box)
        pile.chips = chips
        pile.marks = marks
        pile.tips = {cid: "The card searched for" for cid in chips}
        pile.hovered.connect(lambda cid, p=pile, sc=sc: self._on_pile_hover(p, sc, cid))
        pile.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        pile._scale, pile._content = scale, (sc, ids, info)
        self._fill_pile(pile)
        self.piles.append(pile)
        lay.addWidget(pile)

    def _fill_pile(self, pile: PileView) -> None:
        sc, ids, info = pile._content
        vals = self._values_for_pile(sc, info)
        pile.set_content(sc, ids, info, vals, dict(
            group="cmc", split=True, card_w=max(40, int(self._card_w() * pile._scale)),
            show_values=bool(vals), sort="mv", badge="avg",
        ))
        pile.setFixedHeight(max(pile.minimumHeight(), 40))

    def _on_card_size(self, value: int) -> None:
        new_settings().setValue("decksearch/card_w", int(value))
        for pile in self.piles:
            self._fill_pile(pile)

    def _on_pile_hover(self, pile: PileView, sc: str, cid: int) -> None:
        if cid < 0:
            self.hover_popup.hide()
            return
        name = str((pile.info.get(cid) or {}).get("name", ""))
        if name:
            self.hover_popup.show_card(sc, name, "", QCursor.pos())

    # ------------------------------------------------------------------ copy & open in builder
    def _export_result_for_selected(self) -> dict | None:
        r = self._selected()
        if r is None:
            return None
        if self.mode == "rebuilds" and r.get("builds") and self._selected_build_idx is not None:
            builds = r["builds"]
            if 0 <= self._selected_build_idx < len(builds):
                return dict(r, main=builds[self._selected_build_idx]["main"])
        return r

    def copy_deck(self) -> None:
        r = self._export_result_for_selected()
        if r is None:
            return
        QApplication.clipboard().setText(self.api.deck_search_text(self.set_code(), r))
        self.statusMessage.emit("Deck copied: paste it in Arena's Import")

    def open_in_builder(self) -> None:
        r = self._export_result_for_selected()
        if r is None:
            return
        sc = self.set_code()
        fmt = str(r.get("fmt") or "bo3")
        text = self.api.deck_search_text(sc, r)
        self.openInBuilderRequested.emit(sc, fmt, text)
        self.statusMessage.emit(f"Opened {sc} deck in Deck builder")

    def wait(self, ms: int = 8000) -> None:
        self.task.wait(ms)
        self.val_task.wait(ms)
