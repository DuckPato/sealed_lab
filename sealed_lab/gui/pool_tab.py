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

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
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
    FlowLayout,
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


def fetch_play_rates(api: Any, set_code: str) -> dict[int, dict]:
    """Optional ``api.card_play_rates`` -> {card_id: {eligible, played, rate, marker, tip}}; empty dict if unavailable."""
    fn = getattr(api, "card_play_rates", None)
    if fn is None:
        return {}
    try:
        return dict(fn(set_code) or {})
    except Exception:  # noqa: BLE001 - cosmetic
        return {}


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


def _accepts_kwarg(fn: Any, name: str) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class PoolTab(QWidget):
    logMessage = Signal(str)
    errorOccurred = Signal(str, str)
    busyChanged = Signal(bool)
    busySetChanged = Signal(str)          # the set being built / scored right now ("" = none)
    setChanged = Signal(str)              # the set of the pasted pool changed ("" = none)
    archetypeRequested = Signal(str, str, str)   # set, format, color pair: show this archetype on the Report tab (the main window wires it)
    similarPoolsRequested = Signal(str, object)  # set, {card_id: copies}: open Deck search in Similar pools mode for this pool
    poolChanged = Signal(str, object)     # (object: a dict with int keys does not survive Qt's own dict conversion)  # set code and {card_id: copies} of the recognised non-basic cards of the pasted pool; ("", {}) when there is none
    bestDeckChanged = Signal(str, str, str)   # set, format, main colors ("WU") of the builder's best deck from this pool; ("", "", "") when there is none

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
        self.result_built: str | None = None          # built_at of the result's model when it was made
        self.views: list[DeckView] = []
        self.builders: list[DeckBuilder] = []
        self.user_builder: DeckBuilder | None = None
        self.user_view: DeckView | None = None
        self.trainer_state: dict | None = None       # active Sealed Pool Trainer session ({set_code, fmt, pool, pilot, revealed})
        self.trainer_builder: DeckBuilder | None = None
        self._trainer_seen: set[str] = set()         # draft_ids already dealt (synced with persistent settings)
        self._trainer_session: dict[str, int] = dict(pools=0, close=0, beat_pilot=0)
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

        row = FlowLayout(spacing=6)                     # wraps in a narrow column (wider system fonts) instead of widening it
        self.build_btn = primary_button("Build best decks")
        self.score_btn = QPushButton("Score only")
        self.score_btn.setToolTip("Score the pasted deck (the Deck section) without searching for better builds: fast.  The result opens "
                                  "in the deck builder, where you can edit it.")
        self.import_btn = QPushButton("Import from Arena log")
        self.import_btn.setToolTip("Read the card pool of your most recent Sealed event from Arena's Player.log "
                                   "(needs Options > Account > 'Detailed Logs (Plugin Support)' enabled in Arena; on Linux it is searched for "
                                   "under Steam/Proton, Lutris, Bottles and Wine)")
        self.trainer_btn = QPushButton("Pool trainer")
        self.trainer_btn.setToolTip("Practice building a real top-pilot 17Lands Sealed pool blind (model values hidden), then Reveal & grade "
                                    "your 40-card build against the model's #1 family and the 17Lands pilot's actual deck & record.")
        row.addWidget(self.build_btn)
        row.addWidget(self.score_btn)
        row.addWidget(self.import_btn)
        row.addWidget(self.trainer_btn)
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
        # the column never gets narrower than its inputs need (buttons, radio texts): scroll area + frame included.  Measured again
        # whenever those inputs change (texts set later, a wider system font), see eventFilter
        self._left_top, self._left_scroll = top, top_scroll
        self._left_min = self._inputs_min_width()
        left.setMinimumWidth(self._left_min)
        top.installEventFilter(self)
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
        self.strip_trainer_btn = QPushButton("Pool trainer")
        self.strip_trainer_btn.setToolTip(self.trainer_btn.toolTip())
        self.strip_trainer_btn.clicked.connect(lambda: self.start_trainer())
        sl.addWidget(self.edit_pool_btn)
        sl.addWidget(self.strip_import_btn)
        sl.addWidget(self.strip_trainer_btn)
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
        # a provisional model (17Lands pick data only, no game results yet): said above every result
        self.prov_banner = QFrame()
        self.prov_banner.setFrameShape(QFrame.Shape.StyledPanel)
        pb = QHBoxLayout(self.prov_banner)
        pb.setContentsMargins(10, 4, 10, 4)
        self.prov_label = QLabel("")
        self.prov_label.setWordWrap(True)
        pb.addWidget(self.prov_label, 1)
        self.prov_banner.setVisible(False)
        rl.addWidget(self.prov_banner)
        self.top_warn = WarningBox("Pool notes", compact=True)
        rl.addWidget(self.top_warn)
        self.trainer_banner = QFrame()
        self.trainer_banner.setObjectName("trainerBanner")
        self.trainer_banner.setFrameShape(QFrame.Shape.StyledPanel)
        self.trainer_banner.setProperty("card", True)
        tbl = QVBoxLayout(self.trainer_banner)
        tbl.setContentsMargins(10, 8, 10, 8)
        tbl.setSpacing(4)
        t_top = QHBoxLayout()
        t_top.setSpacing(8)
        self.trainer_title = QLabel("")
        self.trainer_title.setTextFormat(Qt.TextFormat.RichText)
        self.trainer_title.setWordWrap(True)
        t_top.addWidget(self.trainer_title, 1)
        self.trainer_set_combo = ElidedCombo()
        self.trainer_set_combo.setToolTip("Set to deal practice pools from")
        self.trainer_set_combo.activated.connect(lambda _i: self.start_trainer(str(self.trainer_set_combo.currentData() or "")))
        t_top.addWidget(self.trainer_set_combo)
        self.trainer_mode_combo = ElidedCombo()
        self.trainer_mode_combo.addItem("Strong pilot", "standard")
        self.trainer_mode_combo.addItem("Tricky (7+ of 8 held-out draws)", "tricky")
        self.trainer_mode_combo.setToolTip(
            "Strong pilot: random real Sealed pool piloted by a ≥62% 17Lands player.\n"
            "Tricky: a strong-pilot pool where the model picks other colors and the model's deck beats the pilot's in 7+ of 8 draws it wasn't chosen on.\n"
            "A disagreement to study, not proof the pilot was wrong."
        )
        self.trainer_mode_combo.activated.connect(lambda _i: self.start_trainer(str(self.trainer_set_combo.currentData() or "")))
        t_top.addWidget(self.trainer_mode_combo)
        self.trainer_reveal_btn = primary_button("Reveal & grade")
        self.trainer_reveal_btn.setToolTip("Unlock model values and grade your deck across the model's 16 bootstrap draws")
        self.trainer_reveal_btn.clicked.connect(self.reveal_trainer)
        self.trainer_similar_btn = QPushButton("Find similar pools")
        self.trainer_similar_btn.setToolTip("Open Deck search → Similar pools to see how other 17Lands pilots built pools like this one")
        self.trainer_similar_btn.clicked.connect(self._open_trainer_similar)
        self.trainer_similar_btn.setVisible(False)
        self.trainer_next_btn = QPushButton("New pool")
        self.trainer_next_btn.setToolTip("Deal another real Sealed pool")
        self.trainer_next_btn.clicked.connect(lambda: self.start_trainer(str(self.trainer_set_combo.currentData() or "")))
        self.trainer_exit_btn = QPushButton("Exit trainer")
        self.trainer_exit_btn.setToolTip("Leave Pool trainer mode")
        self.trainer_exit_btn.clicked.connect(self.exit_trainer)
        t_top.addWidget(self.trainer_reveal_btn)
        t_top.addWidget(self.trainer_similar_btn)
        t_top.addWidget(self.trainer_next_btn)
        t_top.addWidget(self.trainer_exit_btn)
        tbl.addLayout(t_top)
        self.trainer_detail = muted(QLabel(""))
        self.trainer_detail.setTextFormat(Qt.TextFormat.RichText)
        self.trainer_detail.setWordWrap(True)
        self.trainer_detail.setVisible(False)
        tbl.addWidget(self.trainer_detail)
        self.trainer_banner.setVisible(False)
        rl.addWidget(self.trainer_banner)
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
        self.trainer_btn.clicked.connect(lambda: self.start_trainer())
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
        self.prov_banner.setObjectName("provBanner")
        self.prov_banner.setStyleSheet(f"QFrame#provBanner {{ background: rgba({r},{g},{b},40); border: 1px solid rgb({r},{g},{b}); "
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

    def _inputs_min_width(self) -> int:
        top, sc = self._left_top, self._left_scroll
        return max(330, top.minimumSizeHint().width() + sc.verticalScrollBar().sizeHint().width() + 2 * sc.frameWidth() + 8)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is getattr(self, "_left_top", None) and event.type() == QEvent.Type.LayoutRequest:
            want = self._inputs_min_width()
            if want != self._left_min:
                self._left_min = want
                if hasattr(self, "pool_strip"):                 # (not during __init__)
                    self._update_toggle()                       # applies it while the input column is shown
        return super().eventFilter(obj, event)

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
        old_built = {c: str(st.built_at or "") for c, st in self.statuses.items()}
        self.statuses = {st.set_code: st for st in statuses if st.has_model}
        self._fill_set_combo(self.manual_set)
        if self.set_code and self.set_code not in self.statuses:
            self._set_current(None, "")
        elif self.detect_res is not None and self.set_code is None:
            self._apply_detection(self.detect_res)
        self._update_formats()
        self._update_buttons()
        sc = self.set_code
        if sc and sc in self.statuses and sc in old_built and old_built[sc] != str(self.statuses[sc].built_at or ""):
            self._load_set_info()                            # the model was recalculated: names and card table may have changed
            self._pool_sent = None
            self._emit_pool()                                # (the Report tab's "For your pool" reads fresh card info)
        self._check_stale()

    def _fill_set_combo(self, select: str | None) -> None:
        """The inline chooser: a "choose" placeholder, then the sets that have a model."""
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        self.set_combo.addItem("— choose the set —", "")
        for code in self.statuses:                             # (the Sets tab's order: newest release first)
            self.set_combo.addItem(code, code)
        self.set_combo.setCurrentIndex(max(self.set_combo.findData(select or ""), 0))
        self.set_combo.blockSignals(False)
        self.trainer_set_combo.blockSignals(True)
        cur_tr = str(self.trainer_set_combo.currentData() or "")
        self.trainer_set_combo.clear()
        for code in self.statuses:
            if not getattr(self.statuses[code], "provisional", False):          # (the trainer deals real pools: it needs game data)
                self.trainer_set_combo.addItem(code, code)
        if cur_tr in self.statuses:
            self.trainer_set_combo.setCurrentIndex(max(self.trainer_set_combo.findData(cur_tr), 0))
        self.trainer_set_combo.blockSignals(False)

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
        trainer_ok = (free and any(not getattr(st_, "provisional", False) for st_ in self.statuses.values())
                      and getattr(self.api, "sample_trainer_pool", None) is not None)
        self.trainer_btn.setEnabled(trainer_ok)
        self.strip_trainer_btn.setEnabled(trainer_ok)
        if self.trainer_state is not None:
            tb = self.trainer_builder
            can_rev = free and not self.trainer_state.get("revealed") and tb is not None and bool(tb.deck) and (self.trainer_state["set_code"] not in self.calculating)
            self.trainer_reveal_btn.setEnabled(can_rev)
            self.trainer_next_btn.setEnabled(trainer_ok)
            self.trainer_set_combo.setEnabled(trainer_ok)
            self.trainer_mode_combo.setEnabled(trainer_ok)
            self.trainer_similar_btn.setEnabled(free and bool(self.trainer_state.get("revealed")))
        self.watch_import_btn.setEnabled(free and not self.import_task.running)
        self.text.setReadOnly(self.task.running)
        self.set_combo.setEnabled(free)
        for r, f in ((self.fmt_bo1, "bo1"), (self.fmt_bo3, "bo3")):
            r.setEnabled(free and self._fmt_available(f))
        pst = self.statuses.get(self.set_code or "")
        if pst is not None and getattr(pst, "provisional", False):
            from sealed_lab import provisional as _pv

            self.prov_label.setText(f"<b>Provisional model.</b> {_pv.BANNER}")
            self.prov_banner.setVisible(True)
        else:
            self.prov_banner.setVisible(False)
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
        """The builder's best deck of a finished build / score: its main colors go to the Report tab (nothing when the run found no best deck)."""
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
    def import_from_log(self, course_key: tuple | None = None) -> None:
        """Import the pool of Arena's log; `course_key` (from the watcher's offer) imports exactly that course."""
        fn = getattr(self.api, "import_arena_log", None)
        if fn is None or self.import_task.running:
            return
        sc = self.set_code
        self.import_msg.setText("Reading Arena's Player.log…")
        self.logMessage.emit("=== Import pool from the Arena log ===")

        def job(progress, cancel):
            from sealed_lab.deck.arenalog import ArenaLogError

            try:
                if course_key is not None and _accepts_kwarg(fn, "course_key"):
                    return "ok", fn(sc, course_key=course_key)
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
            self.import_from_log(key)
            return
        self.watch_label.setText(f"New sealed pool in Arena: <b>{escape(c.event)}</b> ({len(c.pool)} cards)")
        self.watch_banner.setVisible(True)
        self._update_buttons()

    def _watch_import(self) -> None:
        if self.task.running or self.import_task.running:
            return
        key = (self._watch_offer or {}).get("key")
        self._watch_offer = None
        self.watch_banner.setVisible(False)
        self.import_from_log(key)

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

    def load_exported_deck(self, set_code: str, fmt: str, text: str, auto_score: bool = True) -> bool:
        """Load an Arena Deck/Sideboard export sent from Deck search, select its set/format, and optionally score it."""
        sc = (set_code or "").upper()
        if not text.strip():
            return False
        if not self.input_visible():
            self._toggle_input()
        self.manual_set = sc if sc in self.statuses else None
        self.choices = {}
        self.text.setPlainText(text)
        if sc in self.statuses:
            self.set_combo.setVisible(False)
            self._set_current(sc, "chosen", "opened from Deck search")
            want = fmt if fmt in ("bo1", "bo3") and self._fmt_available(fmt) else None
            if want:
                self._fmt_user = (sc, want)
                (self.fmt_bo1 if want == "bo1" else self.fmt_bo3).setChecked(True)
                self._update_buttons()
            if auto_score and not self.task.running and not self._locked_set() and self._fmt_ok():
                self.score_only()
            return True
        return False

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
        if self.trainer_state is not None:
            self.trainer_state = None
            self.trainer_builder = None
            self.trainer_banner.setVisible(False)
        api = self.api
        ov = self.overrides()
        key = self._key(mode)

        if mode == "build":
            def job(progress, cancel):
                res = api.build_pool(sc, text, fmt=fmt, n_decks=MAX_FAMILIES, min_diff=10, progress=progress, cancel=cancel,
                                     name_overrides=ov or None)
                ctx = fetch_context(api, sc, fmt, res.decks[0] if res.decks else None)
                return "build", key, res, fetch_card_info(api, sc), fetch_card_values(api, sc, fmt), ctx, fetch_play_rates(api, sc)
            self.logMessage.emit(f"=== Build {sc} {fmt} ===")
            self.strip.begin("Building decks…")
            self.task.run(job, label=f"build pool {sc} {fmt}")
        else:
            def job(progress, cancel):
                res = api.score_deck(sc, text, fmt=fmt, name_overrides=ov or None, search=False)
                return ("score", key, res, fetch_card_info(api, sc), fetch_card_values(api, sc, fmt),
                        fetch_context(api, sc, fmt, res.deck), fetch_play_rates(api, sc))
            self.logMessage.emit(f"=== Score {sc} {fmt} ===")
            self.strip.begin("Scoring deck…")
            self.task.run(job, label=f"score deck {sc} {fmt}")

    def _on_result(self, payload: object) -> None:
        mode = payload[0]  # type: ignore[index]
        if mode == "trainer_sample":
            _, sample, info = payload  # type: ignore[misc]
            self._on_trainer_sampled(sample, info)
            return
        if mode == "trainer_grade":
            _, graded, info, betas, ctx, play_rates = payload  # type: ignore[misc]
            self._on_trainer_graded(graded, info, betas, ctx, play_rates)
            return
        mode, key, res, info, betas, ctx, play_rates = payload  # type: ignore[misc]
        self.result, self.result_key = res, key
        st = self.statuses.get(key[0])
        self.result_built = str(st.built_at or "") if st is not None else None     # the model the result was made from
        if mode == "build":
            self._show_build(res, info, betas, ctx, play_rates)
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
            self._show_score(res, info, betas, ctx, play_rates)
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
        st = self.statuses.get(self.result_key[0])
        model_changed = st is not None and self.result_built is not None and str(st.built_at or "") != self.result_built
        stale = stale or model_changed
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
            if model_changed:
                why.append("the set's model (it was recalculated)")
            self.stale_label.setText(f"<b>This result is out of date</b> — {', '.join(why) or 'the input'} changed since it was built.")
            self.stale_btn.setEnabled(not self.task.running and bool(self.set_code) and not self._locked_set() and self._fmt_ok())
        self.stale_banner.setVisible(stale)

    # ------------------------------------------------------------------ showing results
    def _make_page(self, deck, res_fmt: str, set_code: str, pool, info, betas, ctx, exporter,
                   play_rates: dict[int, dict] | None = None, **view_kw):
        v = DeckView(exporter=exporter)
        v.level_note = ctx.get("level")
        v.errorOccurred.connect(self.errorOccurred)
        v.set_deck(deck, info, res_fmt, reference_wr=ctx.get("reference_wr"), smoothing=bool(ctx.get("smoothing")), **view_kw)
        b = DeckBuilder(self.api)
        b.errorOccurred.connect(self.errorOccurred)
        b.previewChanged.connect(self._on_builder_preview)
        b.previewCleared.connect(self._on_builder_preview_cleared)
        b.load(set_code, res_fmt, deck, pool or {}, info, betas, rarely_played=play_rates)
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
        pages = [self.tabs.widget(i) for i in range(self.tabs.count())]
        for b in self.builders:
            try:
                b.shutdown()                                  # cancel the page's background work
            except RuntimeError:
                pass
        self.tabs.clear()
        for w in pages:
            if w is not None:
                w.deleteLater()
        self.views = []
        self.builders = []
        self.user_builder = None
        self.user_view = None

    def _show_build(self, res: BuildResult, info: dict | None, betas: dict | None = None, ctx: dict | None = None,
                    play_rates: dict[int, dict] | None = None) -> None:
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
            page, v, b = self._make_page(ud, res.fmt, res.set_code, res.pool, info, betas, ctx, exporter, play_rates=play_rates)
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
            page, v, b = self._make_page(deck, res.fmt, res.set_code, getattr(res, "pool", None), info, betas, ctx, exporter,
                                         play_rates=play_rates)
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

    def _show_score(self, res: ScoreResult, info: dict | None, betas: dict | None = None, ctx: dict | None = None,
                    play_rates: dict[int, dict] | None = None) -> None:
        ctx = ctx or {}
        self._reset_results()
        notes = []
        if res.unmatched:
            notes.append("Not recognised (ignored): " + ", ".join(res.unmatched))
        notes += list(res.warnings)
        self.top_warn.set_lines(notes)
        exporter = lambda deck: self.api.export_arena(deck, None, set_code=res.set_code)  # noqa: E731
        label = f"Your deck · {pct(res.deck.game_wr)} · score {res.deck.score:.0f}"
        page, v, b = self._make_page(res.deck, res.fmt, res.set_code, res.pool, info, betas, ctx, exporter, play_rates=play_rates)
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

    # ------------------------------------------------------------------ Sealed Pool Trainer
    @staticmethod
    def _load_trainer_seen(sc: str) -> list[str]:
        import json

        raw = new_settings().value(f"trainer/seen/{sc.upper()}", "")
        if isinstance(raw, list):
            return [str(x) for x in raw if x]
        if isinstance(raw, str) and raw.strip():
            try:
                vals = json.loads(raw)
                if isinstance(vals, list):
                    return [str(x) for x in vals if x]
            except ValueError:
                pass
        return []

    def _remember_trainer_seen(self, sc: str, draft_id: str) -> None:
        import json

        self._trainer_seen.add(draft_id)
        seen = [x for x in self._load_trainer_seen(sc) if x != draft_id]
        seen.append(draft_id)
        if len(seen) > 500:
            seen = seen[-500:]
        new_settings().setValue(f"trainer/seen/{sc.upper()}", json.dumps(seen))

    def _open_trainer_similar(self) -> None:
        st = self.trainer_state
        if not st or not st.get("revealed"):
            return
        pool = dict((st.get("sample") or {}).get("pool") or {})
        if pool:
            self.similarPoolsRequested.emit(st["set_code"], pool)

    def start_trainer(self, set_code: str | None = None, mode: str | None = None) -> bool:
        """Deal a real top-pilot Sealed pool for blind building (values/families hidden until Reveal & grade)."""
        fn = getattr(self.api, "sample_trainer_pool", None)
        if fn is None or self.task.running or not self.statuses:
            return False
        real = [c for c, st_ in self.statuses.items() if not getattr(st_, "provisional", False)]
        if not real:
            return False
        sc = (set_code or self.set_code or real[0]).upper()
        if sc not in self.statuses or sc in self.calculating or getattr(self.statuses[sc], "provisional", False):
            return False
        st = self.statuses[sc]
        fmt = self.fmt() if self.fmt() in st.formats else (default_format(st) or "bo3")
        use_mode = mode or str(self.trainer_mode_combo.currentData() or "standard")
        if mode is not None:
            ix = self.trainer_mode_combo.findData(use_mode)
            if ix >= 0:
                self.trainer_mode_combo.blockSignals(True)
                self.trainer_mode_combo.setCurrentIndex(ix)
                self.trainer_mode_combo.blockSignals(False)
        seen = sorted(self._trainer_seen | set(self._load_trainer_seen(sc)))
        api = self.api

        def job(progress, cancel):
            kw: dict[str, Any] = dict(fmt=fmt, exclude_ids=seen)
            if _accepts_kwarg(fn, "mode"):
                kw["mode"] = use_mode
            if _accepts_kwarg(fn, "progress"):
                kw["progress"] = progress
            if _accepts_kwarg(fn, "cancel"):
                kw["cancel"] = cancel
            sample = fn(sc, **kw)
            info = fetch_card_info(api, sc)
            return "trainer_sample", sample, info

        self._busy_set = sc
        msg = (f"Finding a tricky {sc} pool (the model's deck beats the pilot's in 7+ of 8 draws it wasn't chosen on)…" if use_mode == "tricky"
               else f"Dealing {sc} practice pool…")
        self.strip.begin(msg)
        self.logMessage.emit(f"=== Pool trainer: dealing {sc} {fmt} pool ({use_mode}) ===")
        return self.task.run(job, label=f"trainer sample {sc}")

    def _on_trainer_sampled(self, sample: dict, info: dict) -> None:
        sc, fmt = sample["set_code"], sample["fmt"]
        self._remember_trainer_seen(sc, sample["draft_id"])
        self._reset_results()
        self.result, self.result_key = None, None
        self.stale_banner.setVisible(False)
        self.top_warn.set_lines([])
        b = DeckBuilder(self.api)
        b.errorOccurred.connect(self.errorOccurred)
        b.previewChanged.connect(self._on_builder_preview)
        b.previewCleared.connect(self._on_builder_preview_cleared)
        b.load_blind(sc, fmt, sample["pool"], info)
        b.deckChanged.connect(self._refresh_trainer_counts)
        self.builders = [b]
        self.trainer_builder = b
        self.trainer_state = dict(set_code=sc, fmt=fmt, sample=sample, info=info, revealed=False)
        self.trainer_set_combo.blockSignals(True)
        idx = self.trainer_set_combo.findData(sc)
        if idx >= 0:
            self.trainer_set_combo.setCurrentIndex(idx)
        self.trainer_set_combo.blockSignals(False)
        self.trainer_reveal_btn.setVisible(True)
        self.trainer_similar_btn.setVisible(False)
        self.trainer_detail.setVisible(False)
        self._set_tab_text(self.tabs.addTab(b, "Your blind build"), "Your blind build")
        self.tabs.setVisible(True)
        self.placeholder.setVisible(False)
        self.trainer_banner.setVisible(True)
        self._refresh_trainer_counts()
        self.strip.end("Practice pool ready — build blind, then Reveal & grade")
        if self.input_visible():
            self._toggle_input()
        self.pool_strip.setVisible(False)
        self.toggle_input.setVisible(True)
        self._update_buttons()

    def _refresh_trainer_counts(self) -> None:
        st = self.trainer_state
        tb = self.trainer_builder
        if not st or st.get("revealed") or tb is None:
            return
        n_tot = sum(tb.deck.values())
        n_lands = sum(n for cid, n in tb.deck.items() if (tb.info.get(cid) or {}).get("is_land"))
        n_spells = n_tot - n_lands
        fmt_lbl = "Best of 3" if st["fmt"] == "bo3" else "Best of 1"
        mode_lbl = " · Tricky pool" if (st.get("sample") or {}).get("mode") == "tricky" else ""
        col = semantic()["good"].name() if n_tot == 40 else semantic()["warn"].name()
        self.trainer_title.setText(
            f"<b>Pool trainer · {st['set_code']} {fmt_lbl}{mode_lbl}</b> — Build a 40-card deck blind from a real strong-pilot 17Lands pool  ·  "
            f"<span style='color:{col}'><b>{n_tot} / 40 cards</b></span> ({n_spells} spells, {n_lands} lands)"
        )
        self._update_buttons()

    def reveal_trainer(self) -> bool:
        """Evaluate the user's blind build and compare against Model #1 and the real 17Lands pilot across bootstrap draws."""
        st = self.trainer_state
        tb = self.trainer_builder
        fn = getattr(self.api, "grade_trainer_build", None)
        if not st or st.get("revealed") or tb is None or fn is None or self.task.running:
            return False
        if not tb.deck:
            self.errorOccurred.emit("Add cards to your deck before clicking Reveal & grade.", "")
            return False
        sc, fmt = st["set_code"], st["fmt"]
        if sc in self.calculating:
            return False
        pool = dict(st["sample"]["pool"])
        pilot = dict(st["sample"]["pilot"])
        user_deck = dict(tb.deck)
        api = self.api

        def job(progress, cancel):
            graded = fn(sc, fmt, pool, user_deck, pilot, max_families=MAX_FAMILIES, progress=progress, cancel=cancel)
            best = graded["build_result"].decks[0] if graded["build_result"].decks else graded["user_eval"]
            ctx = fetch_context(api, sc, fmt, best)
            return "trainer_grade", graded, fetch_card_info(api, sc), fetch_card_values(api, sc, fmt), ctx, fetch_play_rates(api, sc)

        self._busy_set = sc
        self.strip.begin("Grading your build against the model and 17Lands pilot…")
        self.logMessage.emit(f"=== Pool trainer: grading {sc} {fmt} build ===")
        return self.task.run(job, label=f"trainer grade {sc}")

    def _on_trainer_graded(self, graded: dict, info: dict, betas: dict | None, ctx: dict | None,
                           play_rates: dict[int, dict] | None = None) -> None:
        ctx = ctx or {}
        sc, fmt = graded["set_code"], graded["fmt"]
        pool = graded["pool"]
        ud = graded["user_eval"]
        pd = graded["pilot_eval"]
        pilot = graded["pilot"]
        res: BuildResult = graded["build_result"]
        sm = graded["summary"]
        tricky_info = ((self.trainer_state or {}).get("sample") or {}).get("tricky_info")

        if self.trainer_state is not None:
            self.trainer_state["revealed"] = True
            self.trainer_state["graded"] = graded
        self._trainer_session["pools"] += 1
        if sm.get("within_noise"):
            self._trainer_session["close"] += 1
        if sm.get("n_draws", 0) > 0 and sm.get("user_beats_pilot", 0) > sm.get("pilot_beats_user", 0):
            self._trainer_session["beat_pilot"] += 1

        self.trainer_builder = None
        self._reset_results()
        self.result = res
        self.result_key = None
        self.stale_banner.setVisible(False)
        self.top_warn.set_lines(list(res.warnings))

        exporter = lambda deck: self.api.export_arena(deck, None, set_code=sc)  # noqa: E731
        best = res.decks[0] if res.decks else ud

        # Tab 0: User's build (unlocked with full model insights)
        n_dr = int(sm.get("n_draws") or 0)
        m_u = int(sm.get("model_beats_user") or 0)
        u_p = int(sm.get("user_beats_pilot") or 0)
        u_tail = f" · #1 ahead {m_u}/{n_dr} draws" if n_dr > 0 else ""
        u_label = f"Your deck · {color_tag(ud.colors, ud.splash)} · {pct(ud.game_wr)}{u_tail}"
        u_page, u_v, u_b = self._make_page(ud, fmt, sc, pool, info, betas, ctx, exporter, play_rates=play_rates)
        u_b.set_origin_label("your blind build")
        u_b.set_best_reference(best)
        self._set_tab_text(self.tabs.addTab(u_page, u_label), u_label)
        self.views.append(u_v)
        self.builders.append(u_b)
        self.user_builder, self.user_view = u_b, u_v

        # Tab 1: 17Lands Pilot's actual build & record
        trophy_txt = " 🏆" if pilot.get("trophy") else ""
        sk_txt = f", {pilot['skill'] * 100:.0f}% pilot" if pilot.get("skill") is not None else ""
        p_label = f"17Lands pilot ({pilot['record']}{trophy_txt}{sk_txt}) · {color_tag(pd.colors, pd.splash)} · {pct(pd.game_wr)}"
        p_page, p_v, p_b = self._make_page(pd, fmt, sc, pool, info, betas, ctx, exporter, play_rates=play_rates)
        p_b.set_origin_label("17Lands pilot's build")
        p_b.set_best_reference(best)
        self._set_tab_text(self.tabs.addTab(p_page, p_label), p_label)
        self.views.append(p_v)
        self.builders.append(p_b)

        # Tabs 2..K: Model's deck families
        fam_labels = [f"#{i + 1} {color_tag(d.colors, d.splash)} · {pct(d.game_wr)}"
                      + (f" · P(best) {pct(res.p_best[i], 0)}" if (res.families and i < len(res.p_best)) else "")
                      for i, d in enumerate(res.decks)]
        for i, deck in enumerate(res.decks):
            diff = res.diffs[i] if i < len(res.diffs) else None
            pb = res.p_best[i] if i < len(res.p_best) else None
            fam = res.families[i] if i < len(res.families) else None
            page, v, b = self._make_page(deck, fmt, sc, pool, info, betas, ctx, exporter, play_rates=play_rates)
            v.set_deck(deck, info, fmt, p_best=pb if (fam or len(res.decks) > 1) else None, diff=diff, rank=i, family=fam,
                       reference_wr=ctx.get("reference_wr"), smoothing=bool(ctx.get("smoothing")))
            b.set_family(fam, res.family_meta)
            b.set_best_reference(best)
            self._set_tab_text(self.tabs.addTab(page, fam_labels[i]), fam_labels[i])
            self.views.append(v)
            self.builders.append(b)

        # Cross-link Compare-with dropdowns across Your deck, Pilot, and Model families
        all_alts = [(fam_labels[j], res.decks[j]) for j in range(len(res.decks))]
        u_b.set_alternatives(all_alts + [(p_label, pd)], compare_first=True)
        p_b.set_alternatives(all_alts + [(u_label, ud)], compare_first=True)
        for j, fb in enumerate(self.builders[2:]):
            fb.set_alternatives([(u_label, ud), (p_label, pd)] + [(fam_labels[m], res.decks[m]) for m in range(len(res.decks)) if m != j])

        # Populate Trainer Grade banner (graded across bootstrap draws, avoiding noisy point-estimate cut-offs)
        good_c, bad_c = semantic()["good"].name(), semantic()["bad"].name()
        verdict_c = good_c if sm.get("within_noise") else bad_c
        vm, vp = sm["vs_model"], sm["vs_pilot"]
        draw_pilot_s = (f" · Your deck beats pilot in <b>{u_p} of {n_dr} draws</b>" if n_dr > 0 else "")
        sess = self._trainer_session
        sess_s = (f" · Session: {sess['pools']} pool" + ("s" if sess["pools"] != 1 else "")
                  + f" (close to #1: {sess['close']}/{sess['pools']}, ahead of pilot: {sess['beat_pilot']}/{sess['pools']})")
        self.trainer_title.setText(
            f"<span style='color:{verdict_c}'><b>{escape(sm['verdict'])}</b></span>{draw_pilot_s}  ·  "
            f"17Lands pilot ({escape(pilot['record'])}{trophy_txt}{sk_txt}) built <b>{escape(pilot['colors'])}</b>"
            f"{sess_s}"
        )
        det_lines = [
            f"<b>Overlap:</b> {vm['shared_spells']} / {max(vm['ref_spells'], 1)} spells shared with Model #1 · "
            f"{vp['shared_spells']} / {max(vp['ref_spells'], 1)} spells shared with the 17Lands pilot."
        ]
        if tricky_info:
            det_lines.append(
                f"<b>Tricky pool:</b> Model's <b>{escape(str(tricky_info.get('model_colors', '')))}</b> beat the pilot's "
                f"<b>{escape(str(tricky_info.get('pilot_colors', '')))}</b> in "
                f"<b>{int(tricky_info.get('model_beats_pilot', 0))} of {int(tricky_info.get('n_draws', 8))}</b> draws it wasn't chosen on. "
                f"A disagreement to study, not proof the pilot was wrong."
            )
        if n_dr > 0:
            det_lines.append("<b>About the draws:</b> Model #1 is found by searching under these same model draws, so it is "
                             "usually ahead in most of them. Being ahead of #1 in a few draws already means a close build.")
        if vm["only_ref"]:
            det_lines.append("<b>In Model #1, not yours:</b> " + escape(", ".join(vm["only_ref"][:6]))
                             + (f" (+{len(vm['only_ref']) - 6} more)" if len(vm["only_ref"]) > 6 else ""))
        if vp["only_ref"]:
            det_lines.append("<b>In Pilot's deck, not yours:</b> " + escape(", ".join(vp["only_ref"][:6]))
                             + (f" (+{len(vp['only_ref']) - 6} more)" if len(vp["only_ref"]) > 6 else ""))
        self.trainer_detail.setText("<br>".join(det_lines))
        self.trainer_detail.setVisible(True)
        self.trainer_reveal_btn.setVisible(False)
        self.trainer_similar_btn.setVisible(True)
        self.trainer_banner.setVisible(True)
        self.tabs.setVisible(True)
        self.placeholder.setVisible(False)
        self.strip.end("Graded")
        self._update_buttons()

    def exit_trainer(self) -> None:
        """Leave Pool trainer mode and clear the trainer banner."""
        self.trainer_state = None
        self.trainer_builder = None
        self.trainer_similar_btn.setVisible(False)
        self.trainer_banner.setVisible(False)
        self._reset_results()
        self.result, self.result_key = None, None
        self.tabs.setVisible(False)
        self.placeholder.setVisible(True)
        if not self.input_visible():
            self._toggle_input()
        self._update_buttons()

    def wait(self, ms: int = 8000) -> None:
        self.detector.wait(ms)
        self.check_task.wait(ms)
        self.import_task.wait(ms)
        self.watch_task.wait(ms)
        self.task.wait(ms)
