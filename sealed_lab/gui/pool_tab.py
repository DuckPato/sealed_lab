"""Pool -> Best deck tab (also scores a pasted deck).

Paste a sealed pool as an Arena export.  The set is detected from the cards ("Set: HOB — detected (240/240 cards matched)"; an inline
chooser appears only when the cards do not tell), the format is a pair of radio buttons (default: the format with more data for the
set), card names that are not exact are guessed and flagged for confirmation, and the result is a set of deck families in the visual
deck builder.  If the pasted text is an Arena deck export with a Deck and a Sideboard section, the Deck is the user's own build: it
opens first ("Your deck", fully editable) and is compared with the best family; "Score only" scores it without searching.

Locking is per set: a set being recalculated blocks only the actions on that set (`set_calculating`).
"""
from __future__ import annotations

from html import escape
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sealed_lab.types import BuildResult, ScoreResult

from . import helptext as H
from .archetype_line import ArchetypeLine
from .deck_view import DeckView, fetch_context
from .deckbuilder import DeckBuilder, ImageStore
from .set_detector import SetDetector
from .widgets import (
    BusyStrip,
    CardPreview,
    ElidedCombo,
    ElidedLabel,
    FloatingCardPreview,
    PasteEdit,
    WarningBox,
    color_tag,
    muted,
    new_settings,
    pct,
    primary_button,
    remember_splitter,
    semantic,
    short_error,
    sized_font,
)
from .workers import TaskController, fetch_card_info


MAX_FAMILIES = 5       # most deck families shown (sealed_lab.deck.families.MAX_FAMILIES)
FORMATS = ("bo1", "bo3")


def fetch_card_values(api: Any, set_code: str, fmt: str) -> dict[int, dict] | None:
    """Optional ``api.card_values`` -> {card_id: {beta, low_data}} (sideboard value badges); None if unavailable."""
    fn = getattr(api, "card_values", None)
    if fn is None:
        return None
    try:
        return dict(fn(set_code, fmt) or {})
    except Exception:  # noqa: BLE001 - cosmetic
        return None


def default_format(status: Any) -> str | None:
    """The format with more data for the set (among the formats its model has); Bo3 on a tie."""
    opts = [f for f in ("bo3", "bo1") if status is not None and f in status.formats]
    if not opts:
        return None
    return max(opts, key=lambda f: status.n_games.get(f, 0))


class _ClickLabel(QLabel):
    """Wrapped text beside a radio button: clicking it picks the button."""

    def __init__(self, text: str, button: QRadioButton):
        super().__init__(text)
        self._button = button

    def mouseReleaseEvent(self, event) -> None:
        if self._button.isEnabled():
            self._button.click()
        super().mouseReleaseEvent(event)


WATCH_INTERVAL_MS = 2000


class PoolTab(QWidget):
    logMessage = Signal(str)
    errorOccurred = Signal(str, str)
    busyChanged = Signal(bool)
    busySetChanged = Signal(str)          # the set being built / scored right now ("" = none)
    setChanged = Signal(str)              # the set of the pasted pool changed ("" = none)
    archetypeRequested = Signal(str, str, str)   # set, format, colour pair: show this archetype on the Report tab (the main window wires it)
    poolChanged = Signal(str, object)     # (object: a dict with int keys does not survive Qt's own dict conversion)  # set code and {card_id: copies} of the recognised non-basic cards of the pasted pool; ("", {}) when there is none
    bestDeckChanged = Signal(str, str, str)   # set, format, main colours ("WU") of the builder's best deck from this pool; ("", "", "") when there is none

    def __init__(self, api: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self.statuses: dict[str, Any] = {}           # sets that have a model
        self.set_code: str | None = None
        self.set_source = ""                         # "detected" | "chosen" | ""
        self.manual_set: str | None = None           # the user's choice when the cards do not tell the set
        self.calculating: set[str] = set()
        self.result: BuildResult | ScoreResult | None = None
        self.result_key: tuple | None = None
        self.views: list[DeckView] = []
        self.builders: list[DeckBuilder] = []
        self.user_builder: DeckBuilder | None = None
        self.user_view: DeckView | None = None
        self.info: dict[int, dict] = {}              # card_info of the current set
        self._index: Any = None                      # NameIndex over `info` (the pool's card names -> ids); None without a set
        self._pool_sent: tuple[str, tuple] = ("", ())    # what poolChanged last said (it is emitted only when this changes)
        self._best_sent: tuple[str, str, str] = ("", "", "")
        self._names: dict[str, str] = {}             # normalised card name -> name (hover in the pasted list)
        self.guesses: list[dict] = []                # api.check_pool "guesses" + "unmatched" rows
        self.choices: dict[str, str] = {}            # pasted name -> confirmed card name ("" = remove the card)
        self.pool_check: dict | None = None
        self._busy_set = ""
        self._fmt_user: tuple[str, str] | None = None    # (set, format) the user picked by hand
        self.detect_res: dict | None = None
        self.import_task = TaskController(self)
        self.import_task.result.connect(self._on_import_result)
        self.import_task.error.connect(self._on_import_error)
        self.import_task.ended.connect(self._update_buttons)
        self.watch_task = TaskController(self)
        self.watch_task.result.connect(self._on_watch_result)
        self.watch_task.error.connect(lambda tb: None)       # the watcher is quiet: errors.log already has the traceback
        self.watch_task.ended.connect(self._update_buttons)
        self._watcher = None
        self._watch_offer: dict | None = None                # the pool the banner offers
        self._watch_dismissed: set = set()
        self.check_task = TaskController(self)
        self.check_task.result.connect(self._on_check_result)
        self.check_task.ended.connect(self._on_check_ended)
        self._recheck = False
        self._check_snapshot: tuple = (None, "")
        self.task = TaskController(self)
        self.task.started.connect(lambda: self._set_busy(True))
        self.task.ended.connect(lambda: self._set_busy(False))
        self.task.progress.connect(lambda f, m: self.strip.update_progress(f, m))
        self.task.log.connect(self.logMessage)
        self.task.result.connect(self._on_result)
        self.task.error.connect(self._on_error)
        self.task.cancelled.connect(self._on_cancelled)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        # the "new sealed pool in Arena" banner: the very top of the tab, so it shows whether or not the pool input is
        self.watch_banner = QFrame()
        self.watch_banner.setObjectName("watchBanner")
        self.watch_banner.setFrameShape(QFrame.Shape.StyledPanel)
        self.watch_banner.setProperty("card", True)
        bl = QHBoxLayout(self.watch_banner)
        bl.setContentsMargins(10, 6, 10, 6)
        self.watch_label = QLabel("")
        self.watch_label.setWordWrap(True)
        bl.addWidget(self.watch_label, 1)
        self.watch_import_btn = primary_button("Import")
        self.watch_dismiss_btn = QPushButton("Dismiss")
        bl.addWidget(self.watch_import_btn)
        bl.addWidget(self.watch_dismiss_btn)
        self.watch_banner.setVisible(False)
        root.addWidget(self.watch_banner)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(True)
        self.split = split
        root.addWidget(split, 1)

        # left: inputs (top) and the docked card preview (bottom) ----------
        left = QSplitter(Qt.Orientation.Vertical)
        left.setChildrenCollapsible(False)
        self.left = left
        top_scroll = QScrollArea()
        top_scroll.setWidgetResizable(True)
        top_scroll.setFrameShape(QFrame.Shape.NoFrame)
        top = QWidget()
        ll = QVBoxLayout(top)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(8)
        head = QLabel("Deck builder")
        head.setFont(sized_font(self.font(), 3, bold=True))
        ll.addWidget(head)
        srow = QHBoxLayout()
        self.set_label = QLabel("Set: paste a pool to detect it")
        self.set_label.setTextFormat(Qt.TextFormat.RichText)
        self.set_label.setWordWrap(True)
        srow.addWidget(self.set_label, 1)
        self.set_combo = ElidedCombo()
        self.set_combo.setToolTip("The cards do not tell which set this is: choose it (only sets with a model are listed)")
        self.set_combo.setVisible(False)
        self.set_combo.activated.connect(self._on_set_chosen)
        srow.addWidget(self.set_combo)
        ll.addLayout(srow)

        fbox = QFrame()
        fl = QVBoxLayout(fbox)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(2)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("<b>Format</b>"))
        self.format_help = QToolButton()
        self.format_help.setText("?")
        self.format_help.setAutoRaise(True)
        self.format_help.setToolTip(H.FORMAT_HELP)
        frow.addWidget(self.format_help)
        frow.addStretch(1)
        fl.addLayout(frow)
        self.fmt_bo1, self.fmt_bo3 = QRadioButton(), QRadioButton()
        self.fmt_labels: dict[str, QLabel] = {}
        for f, r, text in (("bo1", self.fmt_bo1, H.FORMAT_BO1_LABEL), ("bo3", self.fmt_bo3, H.FORMAT_BO3_LABEL)):
            r.setToolTip(H.FORMAT_HELP)
            lab = _ClickLabel(text, r)
            lab.setWordWrap(True)
            lab.setToolTip(H.FORMAT_HELP)
            self.fmt_labels[f] = lab
            rr = QHBoxLayout()
            rr.setSpacing(2)
            rr.addWidget(r, 0, Qt.AlignmentFlag.AlignTop)
            rr.addWidget(lab, 1)
            fl.addLayout(rr)
        self.fmt_group = QButtonGroup(self)
        self.fmt_group.addButton(self.fmt_bo1)
        self.fmt_group.addButton(self.fmt_bo3)
        self.fmt_bo3.setChecked(True)
        self.fmt_bo1.clicked.connect(lambda: self._on_format_clicked("bo1"))
        self.fmt_bo3.clicked.connect(lambda: self._on_format_clicked("bo3"))
        self.fmt_note = muted(QLabel(""))
        self.fmt_note.setWordWrap(True)
        self.fmt_note.setVisible(False)
        fl.addWidget(self.fmt_note)
        ll.addWidget(fbox)

        self.text = PasteEdit()
        self.text.setMinimumHeight(110)
        ll.addWidget(self.text, 1)
        self.deck_note = muted(QLabel(""))
        self.deck_note.setWordWrap(True)
        self.deck_note.setVisible(False)
        ll.addWidget(self.deck_note)
        self.detector = SetDetector(api, self.text, self)
        self.detector.detected.connect(self._on_detected)
        self.detector.newPaste.connect(self._on_new_paste)
        self.detector.cleared.connect(self._on_cleared)
        self.text.textChanged.connect(self._on_text_changed)

        row = QHBoxLayout()
        self.build_btn = primary_button("Build best decks")
        self.score_btn = QPushButton("Score only")
        self.score_btn.setToolTip("Score the pasted deck (the Deck section) without searching for better builds: fast.  The result opens "
                                  "in the deck builder, where you can edit it.")
        self.import_btn = QPushButton("Import from Arena log")
        self.import_btn.setToolTip("Read the card pool of your most recent Sealed event from Arena's Player.log "
                                   "(needs Options > Account > 'Detailed Logs (Plugin Support)' enabled in Arena)")
        row.addWidget(self.build_btn)
        row.addWidget(self.score_btn)
        row.addWidget(self.import_btn)
        row.addStretch(1)
        ll.addLayout(row)
        self.lock_note = QLabel("")
        self.lock_note.setWordWrap(True)
        self.lock_note.setVisible(False)
        ll.addWidget(self.lock_note)
        self.import_msg = muted(QLabel(""))
        self.import_msg.setWordWrap(True)
        self.import_msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ll.addWidget(self.import_msg)
        self.watch_check = QCheckBox("Watch Arena for new sealed pools")
        self.watch_check.setToolTip(H.ARENA_WATCH_HELP)
        self.watch_check.setChecked(self._watch_setting())
        ll.addWidget(self.watch_check)

        self.guess_box = QFrame()
        self.guess_box.setFrameShape(QFrame.Shape.StyledPanel)
        gl = QVBoxLayout(self.guess_box)
        gl.setContentsMargins(8, 4, 8, 4)
        gl.setSpacing(2)
        self.guess_title = QLabel("")
        self.guess_title.setTextFormat(Qt.TextFormat.RichText)
        self.guess_title.setWordWrap(True)
        self.guess_title.setToolTip(H.GUESS_HELP)
        gl.addWidget(self.guess_title)
        self.guess_grid_host = QWidget()
        self.guess_grid = QGridLayout(self.guess_grid_host)
        self.guess_grid.setContentsMargins(0, 0, 0, 0)
        self.guess_grid.setColumnStretch(2, 1)
        gscroll = QScrollArea()
        gscroll.setWidgetResizable(True)
        gscroll.setFrameShape(QFrame.Shape.NoFrame)
        gscroll.setStyleSheet("QScrollArea { background: transparent; } QScrollArea > QWidget > QWidget { background: transparent; }")
        gscroll.setWidget(self.guess_grid_host)
        self.guess_scroll = gscroll
        gl.addWidget(gscroll)
        self.guess_box.setVisible(False)
        self.guess_rows: list[tuple[str, QLabel, QComboBox]] = []

        self.strip = BusyStrip()
        ll.addWidget(self.strip)
        top_scroll.setWidget(top)
        left.addWidget(top_scroll)

        self.preview = CardPreview(ImageStore.for_api(api))
        # the card under the pointer in a floating window while the input (and so the docked preview) is hidden
        self.hover_popup = FloatingCardPreview(ImageStore.for_api(api), self)
        left.addWidget(self.preview)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)
        left.setSizes([450, 340])                       # the inputs (Build best decks included) are on screen; the card keeps a readable size
        remember_splitter(left, "pool/left_v3")         # new key: the old saved sizes cut the inputs off
        # the column never gets narrower than its inputs need (buttons, radio texts): scroll area + frame included
        left.setMinimumWidth(max(330, top.minimumSizeHint().width() + top_scroll.verticalScrollBar().sizeHint().width()
                                 + 2 * top_scroll.frameWidth() + 8))
        self._left_min = left.minimumWidth()
        split.addWidget(left)

        # right: results -------------------------------------------------
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        tr = QHBoxLayout()
        self.toggle_input = QToolButton()
        self.toggle_input.setToolTip("Hide / show the pool input to give the deck builder more room "
                                     "(you can also drag the divider)")
        self.toggle_input.clicked.connect(self._toggle_input)
        tr.addWidget(self.toggle_input)
        # the compact strip that stands in for the pool input once a result is shown
        self.pool_strip = QFrame()
        self.pool_strip.setObjectName("poolStrip")
        self.pool_strip.setFrameShape(QFrame.Shape.StyledPanel)
        self.pool_strip.setProperty("card", True)
        sl = QHBoxLayout(self.pool_strip)
        sl.setContentsMargins(10, 3, 6, 3)
        sl.setSpacing(8)
        self.pool_strip_label = ElidedLabel("", min_chars=16)
        self.pool_strip_label.setFont(sized_font(self.font(), 0, bold=True))
        sl.addWidget(self.pool_strip_label, 1)
        self.edit_pool_btn = QPushButton("Edit pool")
        self.edit_pool_btn.setToolTip("Show the pool input again (the pasted list, the format and the card-name choices)")
        self.edit_pool_btn.clicked.connect(self.show_input)
        self.strip_import_btn = QPushButton("Import from Arena")
        self.strip_import_btn.setToolTip(self.import_btn.toolTip())
        self.strip_import_btn.clicked.connect(self.import_from_log)
        sl.addWidget(self.edit_pool_btn)
        sl.addWidget(self.strip_import_btn)
        self.pool_strip.setVisible(False)
        tr.addWidget(self.pool_strip, 1)
        tr.addStretch(0)
        rl.addLayout(tr)
        rl.addWidget(self.guess_box)
        self.stale_banner = QFrame()
        self.stale_banner.setFrameShape(QFrame.Shape.StyledPanel)
        sb = QHBoxLayout(self.stale_banner)
        sb.setContentsMargins(10, 4, 10, 4)
        self.stale_label = QLabel("")
        self.stale_label.setWordWrap(True)
        sb.addWidget(self.stale_label, 1)
        self.stale_btn = QPushButton("Build again")
        self.stale_btn.clicked.connect(self._rebuild_clicked)
        sb.addWidget(self.stale_btn)
        self.stale_banner.setVisible(False)
        rl.addWidget(self.stale_banner)
        self.top_warn = WarningBox("Pool notes", compact=True)
        rl.addWidget(self.top_warn)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideRight)         # long titles shrink with "…" (the tab's tooltip has the whole title)
        self.tabs.setUsesScrollButtons(False)
        rl.addWidget(self.tabs, 1)
        self.placeholder = muted(QLabel("Results appear here: the deck families the model cannot tell apart, best first."))
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(self.placeholder, 1)
        self.tabs.setVisible(False)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 1000])
        split.setCollapsible(1, False)
        remember_splitter(split, "pool/main")
        split.splitterMoved.connect(lambda *_: self._update_toggle())
        self._update_toggle()
        self._style_boxes()
        self.build_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.build_shortcut2 = QShortcut(QKeySequence("Ctrl+Enter"), self)
        for sc in (self.build_shortcut, self.build_shortcut2):
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(self._build_shortcut)
        self.build_btn.setToolTip("Search for the best decks of this pool (Ctrl+Enter)")

        self.build_btn.clicked.connect(self.build)
        self.score_btn.clicked.connect(self.score_only)
        self.import_btn.clicked.connect(self.import_from_log)
        self.watch_import_btn.clicked.connect(self._watch_import)
        self.watch_dismiss_btn.clicked.connect(self._watch_dismiss)
        self.watch_check.toggled.connect(self._on_watch_toggled)
        self.watch_timer = QTimer(self)
        self.watch_timer.setInterval(WATCH_INTERVAL_MS)
        self.watch_timer.timeout.connect(self._watch_tick)
        if getattr(api, "import_arena_log", None) is not None:
            self.watch_timer.start()
        self.strip.cancelRequested.connect(self.task.cancel)
        self.text.nameHovered.connect(self._on_text_hover)
        self._update_formats()
        self._update_buttons()

    # ------------------------------------------------------------------ small helpers
    def _style_boxes(self) -> None:
        warn = semantic()["warn"]
        r, g, b = warn.red(), warn.green(), warn.blue()
        self.guess_box.setObjectName("guessBox")
        self.guess_box.setStyleSheet(f"QFrame#guessBox {{ background: rgba({r},{g},{b},40); border: 1px solid rgb({r},{g},{b}); "
                                     "border-radius: 4px; }")
        self.stale_banner.setObjectName("staleBanner")
        self.stale_banner.setStyleSheet(f"QFrame#staleBanner {{ background: rgba({r},{g},{b},40); border: 1px solid rgb({r},{g},{b}); "
                                        "border-radius: 4px; }")

    def fmt(self) -> str:
        return "bo1" if self.fmt_bo1.isChecked() else "bo3"

    def input_visible(self) -> bool:
        return self.split.sizes()[0] > 0

    def show_input(self) -> None:
        """The Edit pool button: the pool input is shown again (and stays until the next result arrives)."""
        if not self.input_visible():
            self._toggle_input()
        self.text.setFocus()

    def _build_shortcut(self) -> None:
        if self.build_btn.isEnabled():
            self.build()

    def _toggle_input(self) -> None:
        sizes = self.split.sizes()
        total = sum(sizes)
        if sizes[0] > 0:
            self._saved_left = sizes[0]
            self.split.setSizes([0, total])
        else:
            left = max(getattr(self, "_saved_left", 400), 330)
            self.split.setSizes([left, max(total - left, 1)])
        self._update_strip()
        self._update_toggle()

    def _update_toggle(self) -> None:
        vis = self.input_visible()
        left = self.split.widget(0)                             # a collapsed pool column must not keep the window as wide as it was
        want = self._left_min if vis else 0
        if left.minimumWidth() != want:
            left.setMinimumWidth(want)
        self.toggle_input.setText("▶ Show pool input" if not vis else "◀ Hide pool input")
        strip = self.result is not None and not vis
        self.pool_strip.setVisible(strip)
        self.toggle_input.setVisible(not strip)                 # the strip's "Edit pool" does its job
        if vis:
            self.hover_popup.hide()                             # the docked preview shows the card again

    def _update_strip(self) -> None:
        res = self.result
        if res is None:
            return
        pool = getattr(res, "pool", None)
        n = sum(pool.values()) if pool else 0
        bits = [str(res.set_code), str(res.fmt).capitalize()]
        if n:
            bits.append(f"{n} cards")
        self.pool_strip_label.setText("  ·  ".join(bits))
        self.pool_strip_label.setToolTip("The pool of the result below")

    def _auto_hide_input(self) -> None:
        """A result arrived: the pool input gives way to the result (use "Edit pool" to get it back)."""
        if self.input_visible():
            self._toggle_input()
        self._update_strip()
        self._update_toggle()

    # ------------------------------------------------------------------ sets and locking
    def refresh_sets(self, statuses: list) -> None:
        """The Sets tab's list changed: keep the sets that have a model."""
        self.statuses = {st.set_code: st for st in statuses if st.has_model}
        self._fill_set_combo(self.manual_set)
        if self.set_code and self.set_code not in self.statuses:
            self._set_current(None, "")
        elif self.detect_res is not None and self.set_code is None:
            self._apply_detection(self.detect_res)
        self._update_formats()
        self._update_buttons()

    def _fill_set_combo(self, select: str | None) -> None:
        """The inline chooser: a "choose" placeholder, then the sets that have a model."""
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        self.set_combo.addItem("— choose the set —", "")
        for code in sorted(self.statuses):
            self.set_combo.addItem(code, code)
        self.set_combo.setCurrentIndex(max(self.set_combo.findData(select or ""), 0))
        self.set_combo.blockSignals(False)

    def set_calculating(self, sets: set[str]) -> None:
        """Sets being recalculated right now: actions on them wait, everything else stays usable."""
        self.calculating = {s.upper() for s in sets}
        for b in self.builders:
            b.set_frozen(f"{b.set_code} is being recalculated — editing resumes when it finishes"
                         if b.set_code in self.calculating else "")
        self._update_buttons()

    def busy_set(self) -> str:
        return self._busy_set if self.task.running else ""

    def _set_busy(self, busy: bool) -> None:
        self._busy_set = (self.set_code or "") if busy else ""
        self._update_buttons()
        self.busySetChanged.emit(self._busy_set)
        self.busyChanged.emit(busy)

    def _locked_set(self) -> bool:
        return bool(self.set_code and self.set_code in self.calculating)

    def _update_buttons(self) -> None:
        free = not self.task.running
        ok = free and bool(self.set_code) and not self._locked_set() and self._fmt_ok()
        has_text = bool(self.text.toPlainText().strip())
        self.build_btn.setEnabled(ok and has_text)
        self.score_btn.setEnabled(ok and has_text)
        self.import_btn.setEnabled(free and not self.import_task.running and getattr(self.api, "import_arena_log", None) is not None)
        self.strip_import_btn.setEnabled(self.import_btn.isEnabled())
        self.watch_import_btn.setEnabled(free and not self.import_task.running)
        self.text.setReadOnly(self.task.running)
        self.set_combo.setEnabled(free)
        for r, f in ((self.fmt_bo1, "bo1"), (self.fmt_bo3, "bo3")):
            r.setEnabled(free and self._fmt_available(f))
        if self._locked_set():
            self.lock_note.setText(f"<b>{self.set_code}</b> is being recalculated — available when it finishes")
            self.lock_note.setVisible(True)
        else:
            self.lock_note.setVisible(False)
        self._check_stale()

    def _fmt_available(self, f: str) -> bool:
        st = self.statuses.get(self.set_code or "")
        return st is not None and f in st.formats

    def _fmt_ok(self) -> bool:
        return self._fmt_available(self.fmt())

    # ------------------------------------------------------------------ detection of the set
    def _on_text_changed(self) -> None:
        self._update_buttons()

    def _on_new_paste(self) -> None:
        self.manual_set = None
        self.choices = {}

    def _on_cleared(self) -> None:
        self.manual_set = None
        self.detect_res = None
        self.pool_check = None
        self._check_snapshot = (None, "")
        self.guesses, self.choices = [], {}
        self._fill_guess_box()
        self.deck_note.setVisible(False)
        self.set_combo.setVisible(False)
        self._set_current(None, "")
        self.set_label.setText("Set: paste a pool to detect it")
        self._emit_pool()
        self._emit_best("", "", "")

    def _on_detected(self, res: dict) -> None:
        self.detect_res = res
        self._apply_detection(res)
        self._emit_pool()
        self._run_check()

    def _apply_detection(self, res: dict) -> None:
        code = res.get("set_code")
        total = res.get("total") or 0
        if code and code in self.statuses:
            self.set_combo.setVisible(False)
            self._set_current(code, "detected", f"{res.get('matched')}/{total} cards matched")
            return
        # the cards do not tell (or the set has no model): inline chooser limited to sets with a model
        scores = sorted((res.get("scores") or {}).items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        if scores and scores[0][1] > 0:
            msg = "the cards fit several sets about equally (" + ", ".join(f"{k} {v}/{total}" for k, v in scores) + ")"
        else:
            msg = "no card was recognised in any set that has a model"
        detected_code = res.get("detected_code") or code
        if detected_code and detected_code not in self.statuses:
            msg = (f"Detected <b>{detected_code}</b> ({res.get('matched') or 0}/{total} cards matched), but it has no model yet: "
                   "calculate it on the Sets tab first.")
            self.set_combo.setVisible(False)
            self._set_current(None, "", msg)
            return
        if not self.statuses:
            self.set_combo.setVisible(False)
            self._set_current(None, "", "No set has a model yet: calculate one on the Sets tab.")
            return
        self.set_combo.setVisible(True)
        choice = self.manual_set if self.manual_set in self.statuses else None
        if choice is None:
            self.set_combo.setCurrentIndex(0)
            self._set_current(None, "", f"{msg} — choose it:")
        else:
            self.set_combo.setCurrentIndex(self.set_combo.findData(choice))
            sc = (res.get("scores") or {}).get(choice, 0)
            self._set_current(choice, "chosen", f"{sc}/{total} cards matched")

    def _on_set_chosen(self, _i: int = 0) -> None:
        code = self.set_combo.currentData()
        if not code:
            self.manual_set = None
            self._set_current(None, "", "choose it:")
            return
        self.manual_set = code
        sc = ((self.detect_res or {}).get("scores") or {}).get(code, 0)
        total = (self.detect_res or {}).get("total") or 0
        self._set_current(code, "chosen", f"{sc}/{total} cards matched" if total else "")
        self._run_check()

    def _set_current(self, code: str | None, source: str, detail: str = "") -> None:
        changed = code != self.set_code
        self.set_code, self.set_source = code, source
        if code:
            tag = "detected" if source == "detected" else "chosen by you"
            self.set_label.setText(f"Set: <b>{code}</b> — {tag}" + (f" ({detail})" if detail else ""))
        else:
            self.set_label.setText(f"Set: {detail}" if detail else "Set: not known yet")
        if changed:
            self._load_set_info()
            self.setChanged.emit(code or "")
            self._emit_pool()
            if self._best_sent[0] != (code or ""):
                self._emit_best("", "", "")                  # (the best deck was built for another set)
            if code:
                self.fmt_note.setVisible(False)
        self._update_formats()
        self._update_buttons()

    def _load_set_info(self) -> None:
        self.info = {}
        self._names = {}
        if not self.set_code:
            return
        from sealed_lab.deck.names import normalize
        from .workers import fetch_card_info as _fci

        try:
            info = _fci(self.api, self.set_code) or {}
        except Exception:  # noqa: BLE001 - cosmetic
            info = {}
        self.info = info
        self._index = self._make_index(info)
        self._names = {normalize(d["name"]): d["name"] for d in info.values()}
        for d in info.values():            # front faces of "A // B"
            self._names.setdefault(normalize(d["name"].split(" // ")[0]), d["name"])

    @staticmethod
    def _make_index(info: dict) -> Any:
        if not info:
            return None
        try:
            import polars as pl

            from sealed_lab.deck.names import NameIndex

            ids = sorted(info)
            return NameIndex(pl.DataFrame({"card_id": ids, "name": [info[i]["name"] for i in ids], "is_basic": [bool(info[i].get("is_basic")) for i in ids]}))
        except Exception:  # noqa: BLE001 - only the Report tab's "your pool" part depends on it
            return None

    # ------------------------------------------------------------------ what the Report tab is told
    def pool_cards(self) -> dict[int, int]:
        """{card_id: copies} of the recognised non-basic cards of the pasted text (main deck and sideboard together), with the name
        choices the user confirmed applied; {} without a set or an index.  Names are matched without fuzzy guessing (a guessed name counts
        once it is confirmed in the list of guessed names)."""
        text = self.text.toPlainText()
        if not self.set_code or self._index is None or not text.strip():
            return {}
        from sealed_lab.deck.arena import parse_arena_full
        from sealed_lab.deck.names import apply_overrides

        parsed = parse_arena_full(text)
        named: dict[str, int] = {}
        for d in (parsed.main, parsed.side):
            for k, n in d.items():
                named[k] = named.get(k, 0) + n
        pool: dict[int, int] = {}
        for name, n in apply_overrides(named, self.overrides()).items():
            m = self._index.match(name, fuzzy=False)
            if m.ok and not m.is_basic:
                pool[int(m.card_id)] = pool.get(int(m.card_id), 0) + int(n)
        return pool

    def _emit_pool(self) -> None:
        """Tell the Report tab about the pool when the set or the cards changed (nothing is sent twice)."""
        pool = self.pool_cards() if self.set_code else {}
        code = (self.set_code or "") if pool else ""
        sent = (code, tuple(sorted(pool.items())))
        if sent == self._pool_sent:
            return
        self._pool_sent = sent
        self.poolChanged.emit(code, dict(pool) if code else {})

    def _emit_best(self, code: str, fmt: str, colors: str) -> None:
        if (code, fmt, colors) == self._best_sent:
            return
        self._best_sent = (code, fmt, colors)
        self.bestDeckChanged.emit(code, fmt, colors)

    def _announce_best(self, res: Any) -> None:
        """The builder's best deck of a finished build / score: its main colours go to the Report tab (nothing when the run found no best deck)."""
        best = None
        if isinstance(res, BuildResult):
            best = res.decks[0] if res.decks else None
        elif isinstance(res, ScoreResult):
            best = res.best_alternative
        colors = "".join(c for c in "WUBRG" if c in (getattr(best, "colors", "") or "")) if best is not None else ""
        self._emit_best(res.set_code if colors else "", res.fmt if colors else "", colors)

    # ------------------------------------------------------------------ format
    def _update_formats(self) -> None:
        st = self.statuses.get(self.set_code or "")
        for r, f in ((self.fmt_bo1, "bo1"), (self.fmt_bo3, "bo3")):
            r.setEnabled(not self.task.running and st is not None and f in st.formats)
        if st is None:
            return
        if self._fmt_user and self._fmt_user[0] == self.set_code and self._fmt_available(self._fmt_user[1]):
            want = self._fmt_user[1]
        else:
            want = default_format(st) or self.fmt()
        if want != self.fmt():
            (self.fmt_bo1 if want == "bo1" else self.fmt_bo3).setChecked(True)
        n = {f: st.n_games.get(f, 0) for f in FORMATS if f in st.formats}
        if n:
            self.fmt_note.setText("Games in the data: " + " · ".join(f"{f.upper()} {v:,}" for f, v in n.items())
                                  + ("" if self._fmt_user and self._fmt_user[0] == self.set_code else " — preselected: more data"))
            self.fmt_note.setVisible(True)

    def _on_format_clicked(self, f: str) -> None:
        self._fmt_user = (self.set_code or "", f)
        self._update_buttons()

    # ------------------------------------------------------------------ guessed card names
    def _run_check(self) -> None:
        fn = getattr(self.api, "check_pool", None)
        text = self.text.toPlainText()
        if fn is None or not self.set_code or not text.strip():
            return
        if self._check_snapshot == (self.set_code, text):
            return                                     # already checked (or being checked)
        if self.check_task.running:
            self._recheck = True
            return
        sc = self.set_code
        self._check_snapshot = (sc, text)

        def job(progress, cancel):
            return fn(sc, text)
        self.check_task.run(job, label=f"check pool {sc}")

    def _on_check_ended(self) -> None:
        if self._recheck:
            self._recheck = False
            self._run_check()

    def _on_check_result(self, res: object) -> None:
        if not isinstance(res, dict) or self._check_snapshot != (self.set_code, self.text.toPlainText()):
            return                                     # a newer paste is being checked
        self.pool_check = res
        rows = [dict(r, kind="guess") for r in res.get("guesses", [])] + [dict(r, kind="unmatched") for r in res.get("unmatched", [])]
        self.guesses = rows
        keep = {r["raw"] for r in rows}
        self.choices = {k: v for k, v in self.choices.items() if k in keep}
        for r in rows:
            if r["raw"] not in self.choices:
                self.choices[r["raw"]] = r["name"] if r["kind"] == "guess" else ""
        self._fill_guess_box()
        self._emit_pool()
        if res.get("has_deck"):
            self.deck_note.setText(f"Deck + Sideboard export detected: the Deck ({res.get('n_main')} cards) is taken as your own "
                                   f"build and the whole list ({res.get('n_main', 0) + res.get('n_side', 0)} cards) as the pool.")
            self.deck_note.setVisible(True)
        else:
            self.deck_note.setVisible(False)
        self._check_stale()

    def _fill_guess_box(self) -> None:
        while self.guess_grid.count():
            it = self.guess_grid.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        self.guess_rows = []
        n_guess = sum(1 for r in self.guesses if r["kind"] == "guess")
        n_un = len(self.guesses) - n_guess
        if not self.guesses:
            self.guess_box.setVisible(False)
            return
        col = semantic()["warn"].name()
        parts = []
        if n_guess:
            parts.append(f"<b style='color:{col}'>⚠ {n_guess} card name{'s were' if n_guess != 1 else ' was'} guessed — please confirm</b>")
        if n_un:
            parts.append(f"<b>{n_un} name{'s are' if n_un != 1 else ' is'} not a card of this set</b> (ignored unless you pick one)")
        self.guess_title.setText("  ·  ".join(parts))
        for k, r in enumerate(self.guesses):
            lab = ElidedLabel(f"‘{r['raw']}’" + (f" ×{r['count']}" if r["count"] > 1 else "") + " →", min_chars=12)
            lab.setToolTip(f"As pasted in the {'Deck' if r.get('section') == 'deck' else 'Sideboard'} section")
            cb = ElidedCombo()
            for c in r["candidates"]:
                cb.addItem(f"{c['name']}  ({c['score'] * 100:.0f}%)", c["name"])
            cb.addItem("Remove this card", "")
            cb.setToolTip(H.GUESS_HELP)
            cb.setMinimumContentsLength(20)                 # shrinks to ~20 characters; the text is then elided (full name in the tooltip)
            cb.setMaximumWidth(520)
            want = self.choices.get(r["raw"], "")
            i = cb.findData(want)
            cb.setCurrentIndex(i if i >= 0 else cb.count() - 1)
            cb.activated.connect(lambda _i, raw=r["raw"], box=cb: self._on_guess_changed(raw, box))
            self.guess_grid.addWidget(lab, k, 0)
            self.guess_grid.addWidget(cb, k, 1)
            self.guess_rows.append((r["raw"], lab, cb))
        row_h = max(w.sizeHint().height() for _raw, _lab, w in self.guess_rows) + max(self.guess_grid.verticalSpacing(), 6)
        self.guess_scroll.setFixedHeight(min(self.guess_grid_host.sizeHint().height() + 4, 3 * row_h + 4))   # grows with the rows, scrolls beyond 3
        self.guess_box.setVisible(True)

    def _on_guess_changed(self, raw: str, box: QComboBox) -> None:
        self.choices[raw] = str(box.currentData() or "")
        self.logMessage.emit(f"Card name ‘{raw}’ → {self.choices[raw] or 'removed'}")
        self._emit_pool()
        self._check_stale()

    def overrides(self) -> dict[str, str]:
        """{pasted name: confirmed card name ('' = remove)} for the guessed names (and the unrecognised ones the user matched)."""
        out: dict[str, str] = {}
        for r in self.guesses:
            c = self.choices.get(r["raw"], "")
            if r["kind"] == "guess" or c:
                out[r["raw"]] = c
        return out

    # ------------------------------------------------------------------ hover in the pasted list
    def _on_text_hover(self, name: str) -> None:
        from sealed_lab.deck.names import normalize

        if not self.set_code:
            return
        canon = self._names.get(normalize(name))
        if canon is None:
            return
        d = next((x for x in self.info.values() if x["name"] == canon), {})
        bits = [f"<b>{canon}</b>"]
        if d.get("types"):
            bits.append(d["types"])
        self.preview.show_card(self.set_code, canon, "<br>".join(bits))

    def _on_builder_preview(self, set_code: str, name: str, caption: str, tip: str) -> None:
        self.preview.show_card(set_code, name, caption)
        self.preview.caption.setToolTip(tip)
        if not self.input_visible() and self.isVisible():
            self.hover_popup.show_card(set_code, name, caption, QCursor.pos())

    def _on_builder_preview_cleared(self) -> None:
        self.hover_popup.hide()

    def hideEvent(self, event) -> None:
        self.hover_popup.hide()
        super().hideEvent(event)

    # ------------------------------------------------------------------ Arena log import
    def import_from_log(self) -> None:
        fn = getattr(self.api, "import_arena_log", None)
        if fn is None or self.import_task.running:
            return
        sc = self.set_code
        self.import_msg.setText("Reading Arena's Player.log…")
        self.logMessage.emit("=== Import pool from the Arena log ===")

        def job(progress, cancel):
            from sealed_lab.deck.arenalog import ArenaLogError

            try:
                return "ok", fn(sc)
            except ArenaLogError as e:
                return "msg", str(e)
        self.import_task.run(job, label="import Arena log")
        self._update_buttons()

    def _on_import_result(self, payload: object) -> None:
        kind, val = payload  # type: ignore[misc]
        if kind != "ok":
            self.import_msg.setText(str(val))
            self.logMessage.emit(f"Arena log import: {val}")
            return
        res = val
        self.text.setPlainText(res.text)         # the set is detected from the pasted cards, like any paste
        when = f" ({res.timestamp.replace('T', ' ')})" if res.timestamp else ""
        msg = f"Imported {res.n_cards} cards from event {res.event}{when}."
        if res.set_code and res.set_code not in self.statuses:
            msg += f" No model for set {res.set_code} yet."
        self.import_msg.setText(msg + ("  " + " ".join(res.warnings) if res.warnings else ""))
        self.logMessage.emit(msg)
        for w in res.warnings:
            self.logMessage.emit(f"Arena log import: {w}")

    # ------------------------------------------------------------------ watching Arena's log for a new sealed pool
    @staticmethod
    def _watch_setting() -> bool:
        return str(new_settings().value("pool/watch_arena", True)).lower() not in ("false", "0", "no")

    def _on_watch_toggled(self, on: bool) -> None:
        new_settings().setValue("pool/watch_arena", bool(on))
        if not on:
            self._watch_offer = None
            self.watch_banner.setVisible(False)

    def _watch_tick(self) -> None:
        """Every ~2 s: one os.stat on the GUI thread; the (possibly large) read and parse run in a worker, only when the file changed."""
        if not self.watch_check.isChecked() or self.watch_task.running:
            return
        if self._watcher is None:
            from sealed_lab.deck.arenawatch import ArenaWatcher

            self._watcher = ArenaWatcher()                   # SEALED_LAB_ARENA_LOG_DIR or Arena's default folder, looked up per poll
        w = self._watcher
        if not w.changed():
            return
        self.watch_task.run(lambda progress, cancel: w.poll(), label="watch Arena log")

    def _on_watch_result(self, found: object) -> None:
        if not found or not self.watch_check.isChecked():
            return
        c = found[-1]                                         # type: ignore[index]
        from sealed_lab.deck.arenawatch import course_key

        key = course_key(c)
        if key in self._watch_dismissed:
            return
        self._watch_offer = dict(key=key, event=c.event, n=len(c.pool))
        if not self.text.toPlainText().strip() and not self.task.running and not self.import_task.running:
            self._watch_offer = None
            self.watch_banner.setVisible(False)
            self.logMessage.emit(f"Arena: new sealed pool {c.event} ({len(c.pool)} cards): importing it")
            self.import_from_log()
            return
        self.watch_label.setText(f"New sealed pool in Arena: <b>{escape(c.event)}</b> ({len(c.pool)} cards)")
        self.watch_banner.setVisible(True)
        self._update_buttons()

    def _watch_import(self) -> None:
        if self.task.running or self.import_task.running:
            return
        self._watch_offer = None
        self.watch_banner.setVisible(False)
        self.import_from_log()

    def _watch_dismiss(self) -> None:
        if self._watch_offer:
            self._watch_dismissed.add(self._watch_offer["key"])
        self._watch_offer = None
        self.watch_banner.setVisible(False)

    def _on_import_error(self, tb: str) -> None:
        self.import_msg.setText("Import failed.")
        self.errorOccurred.emit(short_error(tb), tb)

    # ------------------------------------------------------------------ build / score
    def _key(self, mode: str) -> tuple:
        return (self.set_code, self.fmt(), self.text.toPlainText(), tuple(sorted(self.overrides().items())), mode)

    def build(self) -> None:
        self._start("build")

    def score_only(self) -> None:
        self._start("score")

    def _rebuild_clicked(self) -> None:
        self._start("score" if self.result_key and self.result_key[-1] == "score" else "build")

    def _start(self, mode: str) -> None:
        sc, fmt = self.set_code, self.fmt()
        text = self.text.toPlainText()
        if not sc or self.task.running:
            return
        if not text.strip():
            self.strip.end("Paste a pool first")
            self.errorOccurred.emit("Paste your pool (Arena export) first.", "")
            return
        if self._locked_set():
            return
        api = self.api
        ov = self.overrides()
        key = self._key(mode)

        if mode == "build":
            def job(progress, cancel):
                res = api.build_pool(sc, text, fmt=fmt, n_decks=MAX_FAMILIES, min_diff=10, progress=progress, cancel=cancel,
                                     name_overrides=ov or None)
                ctx = fetch_context(api, sc, fmt, res.decks[0] if res.decks else None)
                return "build", key, res, fetch_card_info(api, sc), fetch_card_values(api, sc, fmt), ctx
            self.logMessage.emit(f"=== Build {sc} {fmt} ===")
            self.strip.begin("Building decks…")
            self.task.run(job, label=f"build pool {sc} {fmt}")
        else:
            def job(progress, cancel):
                res = api.score_deck(sc, text, fmt=fmt, name_overrides=ov or None, search=False)
                return "score", key, res, fetch_card_info(api, sc), fetch_card_values(api, sc, fmt), fetch_context(api, sc, fmt, res.deck)
            self.logMessage.emit(f"=== Score {sc} {fmt} ===")
            self.strip.begin("Scoring deck…")
            self.task.run(job, label=f"score deck {sc} {fmt}")

    def _on_result(self, payload: object) -> None:
        mode, key, res, info, betas, ctx = payload  # type: ignore[misc]
        self.result, self.result_key = res, key
        if mode == "build":
            self._show_build(res, info, betas, ctx)
            what = "deck famil" + ("y" if len(res.decks) == 1 else "ies") if res.families else "deck(s)"
            extra = ""
            if res.family_meta:
                m = res.family_meta
                extra = f" ({m['n_candidates']} candidate decks, {m['n_families']} families found, {m['seconds']:.1f}s)"
                if m.get("n_capped"):
                    extra += f"; {m['n_capped']} more qualified but only {m['max_families']} are shown"
            mine = " + your deck" if res.user_deck is not None else ""
            self.strip.end(f"Done: {len(res.decks)} {what.replace('deck famil', 'famil')}{mine}")
            self.logMessage.emit(f"Build finished: {len(res.decks)} {what}{extra}")
        else:
            self._show_score(res, info, betas, ctx)
            self.strip.end("Done")
            self.logMessage.emit(f"Score finished: {res.deck.score:.0f}")
        self._announce_best(res)
        self._auto_hide_input()
        self._check_stale()

    def _on_error(self, tb: str) -> None:
        self.strip.end("Failed")
        self.errorOccurred.emit(short_error(tb), tb)

    def _on_cancelled(self) -> None:
        self.strip.end("Cancelled")
        self.logMessage.emit("Build cancelled.")

    def _check_stale(self) -> None:
        """A result stays on screen when the inputs change; the banner says it is out of date and offers to build again."""
        if self.result is None or self.result_key is None:
            self.stale_banner.setVisible(False)
            return
        now = self._key(self.result_key[-1])
        stale = now != self.result_key
        if stale:
            why = []
            if now[0] != self.result_key[0]:
                why.append("the set")
            if now[1] != self.result_key[1]:
                why.append("the format")
            if now[2] != self.result_key[2]:
                why.append("the pasted list")
            if now[3] != self.result_key[3]:
                why.append("a card-name choice")
            self.stale_label.setText(f"<b>This result is out of date</b> — {', '.join(why) or 'the input'} changed since it was built.")
            self.stale_btn.setEnabled(not self.task.running and bool(self.set_code) and not self._locked_set() and self._fmt_ok())
        self.stale_banner.setVisible(stale)

    # ------------------------------------------------------------------ showing results
    def _make_page(self, deck, res_fmt: str, set_code: str, pool, info, betas, ctx, exporter, **view_kw):
        v = DeckView(exporter=exporter)
        v.level_note = ctx.get("level")
        v.errorOccurred.connect(self.errorOccurred)
        v.set_deck(deck, info, res_fmt, reference_wr=ctx.get("reference_wr"), smoothing=bool(ctx.get("smoothing")), **view_kw)
        b = DeckBuilder(self.api)
        b.errorOccurred.connect(self.errorOccurred)
        b.previewChanged.connect(self._on_builder_preview)
        b.previewCleared.connect(self._on_builder_preview_cleared)
        b.load(set_code, res_fmt, deck, pool or {}, info, betas)
        line = ArchetypeLine(self.api, set_code, res_fmt)      # which archetype of the Report tab the deck is (background; edits refresh it)
        line.openRequested.connect(self.archetypeRequested)
        b.layout().addWidget(line)
        b.evaluated.connect(lambda _ev, b=b, line=line: line.request(b.deck))
        line.request(b.deck)
        b.archetype_line = line
        page = QTabWidget()
        page.setDocumentMode(True)
        page.addTab(b, "Deck builder")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(v)
        page.addTab(scroll, "Details (as generated)")
        return page, v, b

    def _set_tab_text(self, i: int, text: str) -> None:
        self.tabs.setTabText(i, text)
        self.tabs.setTabToolTip(i, text)

    def _reset_results(self) -> None:
        self.tabs.clear()
        self.views = []
        self.builders = []
        self.user_builder = None
        self.user_view = None

    def _show_build(self, res: BuildResult, info: dict | None, betas: dict | None = None, ctx: dict | None = None) -> None:
        ctx = ctx or {}
        self._reset_results()
        notes = []
        if res.pool_unmatched:
            notes.append("Not recognised (ignored): " + ", ".join(res.pool_unmatched))
        if res.user_unmatched:
            notes.append("Not recognised in your deck: " + ", ".join(res.user_unmatched))
        notes += list(res.warnings)
        self.top_warn.set_lines(notes)
        exporter = lambda deck: self.api.export_arena(deck, None, set_code=res.set_code)  # noqa: E731
        offset = 0
        best = res.decks[0] if res.decks else None
        labels = [f"#{i + 1} {color_tag(d.colors, d.splash)} · {pct(d.game_wr)}"
                  + (f" · P(best) {pct(res.p_best[i], 0)}" if (res.families and i < len(res.p_best)) else "")
                  for i, d in enumerate(res.decks)]
        if res.user_deck is not None:
            ud = res.user_deck
            gap = 100.0 * (ud.game_wr - best.game_wr) if best is not None else 0.0
            label = f"Your deck · {pct(ud.game_wr)}" + (f" · {gap:+.1f} pp vs #1" if best is not None else "")
            page, v, b = self._make_page(ud, res.fmt, res.set_code, res.pool, info, betas, ctx, exporter)
            b.set_origin_label("your pasted deck")
            b.set_best_reference(best)
            b.edited.connect(lambda on, base=label: self._set_tab_text(0, base + ("  • edited" if on else "")))
            self._set_tab_text(self.tabs.addTab(page, label), label)
            self.views.append(v)
            self.builders.append(b)
            self.user_builder, self.user_view = b, v
            offset = 1
        for i, deck in enumerate(res.decks):
            diff = res.diffs[i] if i < len(res.diffs) else None
            pb = res.p_best[i] if i < len(res.p_best) else None
            fam = res.families[i] if i < len(res.families) else None
            page, v, b = self._make_page(deck, res.fmt, res.set_code, getattr(res, "pool", None), info, betas, ctx, exporter)
            v.set_deck(deck, info, res.fmt, p_best=pb if (fam or len(res.decks) > 1) else None, diff=diff, rank=i, family=fam,
                       reference_wr=ctx.get("reference_wr"), smoothing=bool(ctx.get("smoothing")))
            b.set_family(fam, res.family_meta)
            b.set_best_reference(best)
            k = i + offset
            b.edited.connect(lambda on, k=k, base=labels[i]: self._set_tab_text(k, base + ("  • edited" if on else "")))
            self._set_tab_text(self.tabs.addTab(page, labels[i]), labels[i])
            self.views.append(v)
            self.builders.append(b)
        fam_builders = self.builders[offset:]
        for i, b in enumerate(fam_builders):
            b.set_alternatives([(labels[j], res.decks[j]) for j in range(len(res.decks)) if j != i])
        if self.user_builder is not None:
            self.user_builder.set_alternatives([(labels[j], res.decks[j]) for j in range(len(res.decks))], compare_first=True)
        has = bool(res.decks) or res.user_deck is not None
        self.tabs.setVisible(has)
        self.placeholder.setVisible(not has)
        if not has:
            self.placeholder.setText("No legal deck could be built from this pool.")
        self.set_calculating(self.calculating)

    def _show_score(self, res: ScoreResult, info: dict | None, betas: dict | None = None, ctx: dict | None = None) -> None:
        ctx = ctx or {}
        self._reset_results()
        notes = []
        if res.unmatched:
            notes.append("Not recognised (ignored): " + ", ".join(res.unmatched))
        notes += list(res.warnings)
        self.top_warn.set_lines(notes)
        exporter = lambda deck: self.api.export_arena(deck, None, set_code=res.set_code)  # noqa: E731
        label = f"Your deck · {pct(res.deck.game_wr)} · score {res.deck.score:.0f}"
        page, v, b = self._make_page(res.deck, res.fmt, res.set_code, res.pool, info, betas, ctx, exporter)
        b.set_origin_label("your pasted deck")
        if res.best_alternative is not None:
            b.set_best_reference(res.best_alternative, "the best build from your pool")
            b.set_alternatives([("Best build from your pool", res.best_alternative)], compare_first=True)
        b.edited.connect(lambda on, base=label: self._set_tab_text(0, base + ("  • edited" if on else "")))
        self._set_tab_text(self.tabs.addTab(page, label), label)
        self.views.append(v)
        self.builders.append(b)
        self.user_builder, self.user_view = b, v
        self.tabs.setVisible(True)
        self.placeholder.setVisible(False)
        self.set_calculating(self.calculating)

    def wait(self, ms: int = 8000) -> None:
        self.detector.wait(ms)
        self.check_task.wait(ms)
        self.import_task.wait(ms)
        self.watch_task.wait(ms)
        self.task.wait(ms)
