"""Sets tab: the tracked 17lands data folder, one row per set with a clear status, statistics of the selected set, and a calculation
queue.

* The app follows one data folder (default: where the 17lands downloader saves, one sub-folder per set).  New or changed files of the
  sets it knows are registered automatically: at start, when the folder changes on disk (QFileSystemWatcher, debounced), when the window
  is activated again and on "Rescan".  A set can point at its own folder.
* Each set shows one status: Ready / Recalculation pending / Queued / Calculating NN% / Failed; sets that exist on disk or on 17lands'
  list but are not tracked are listed greyed out.
* "Calculate" adds the set to a queue; sets run one after another.  A calculation blocks only its own set elsewhere in the app.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QFileSystemWatcher, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QBrush, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sealed_lab.ingest.setinfo import order_sets
from sealed_lab.types import SetStatus

from . import helptext as H
from .set_panel import SetPanel, fmt_time
from .widgets import (
    Collapsible,
    ElidedLabel,
    FlowLayout,
    LogPane,
    make_table,
    row_height,
    muted,
    new_settings,
    primary_button,
    remember_splitter,
    semantic,
    short_error,
    sized_font,
)
from .workers import TaskController

SET_COLS = ["Set", "Status", "Games per format", "Built at"]
RECENCY_ITEMS = [("Auto", "auto"), ("On", "on"), ("Off", "off")]
MATCHUPS_ITEMS = [("Pooled across sets", "pooled"), ("Auto (this set only)", "auto"), ("On", "on"), ("Off", "off")]
RESCAN_DEBOUNCE_MS = 2000       # a download writes many files: scan once after the folder has been quiet this long
ACTIVATE_RESCAN_S = 60.0        # rescan when the window is activated again, at most this often


def fmt_games(n_games: dict[str, int]) -> str:
    if not n_games:
        return "—"
    return " · ".join(f"{k} {v:,}" for k, v in n_games.items())


class QueueStrip(QWidget):
    """The calculation queue: the running set, then the waiting sets in order, each with a remove button."""

    removeRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.lay = FlowLayout(self)                  # a long queue wraps to a second row instead of widening the window
        self.label = muted(QLabel("Queue:"))
        self.lay.addWidget(self.label)
        self.chips: dict[str, QFrame] = {}
        self.remove_buttons: dict[str, QToolButton] = {}
        self.empty = muted(QLabel("empty"))
        self.lay.addWidget(self.empty)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._sig: tuple | None = None
        self._run_label: QLabel | None = None

    def set_items(self, running: tuple[str, int] | None, queued: list[str], waiting_for: str = "") -> None:
        sig = (running[0] if running else None, tuple(queued), waiting_for)
        if sig == self._sig and running is not None and self._run_label is not None:
            self._run_label.setText(f"▶ {running[0]} {running[1]}%")           # a progress tick: no rebuild
            return
        self._sig = sig
        self._run_label = None
        for w in list(self.chips.values()):
            self.lay.removeWidget(w)
            w.hide()
            w.setParent(None)
            w.deleteLater()
        self.chips, self.remove_buttons = {}, {}
        if running is not None:
            self.chips[running[0]] = self._chip(f"▶ {running[0]} {running[1]}%", None, "Running now")
            self._run_label = self.chips[running[0]].findChild(QLabel)
        for k, code in enumerate(queued):
            tip = f"Waiting for {waiting_for} (the deck builder is using it)" if code == waiting_for else f"Remove {code} from the queue"
            self.chips[code] = self._chip(f"{k + 1}. {code}", code, tip)
        for code, chip in self.chips.items():
            self.lay.addWidget(chip)
        self.empty.setVisible(not self.chips)
        self.lay.invalidate()
        self.updateGeometry()

    def _chip(self, text: str, removable: str | None, tip: str) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.StyledPanel)
        f.setProperty("card", True)
        h = QHBoxLayout(f)
        h.setContentsMargins(6, 0, 2, 0)
        h.setSpacing(2)
        lab = QLabel(text)
        lab.setToolTip(tip)
        h.addWidget(lab)
        if removable:
            b = QToolButton()
            b.setText("✕")
            b.setAutoRaise(True)
            b.setToolTip(f"Remove {removable} from the queue")
            b.clicked.connect(lambda _=False, c=removable: self.removeRequested.emit(c))
            h.addWidget(b)
            self.remove_buttons[removable] = b
        return f


class SetsTab(QWidget):
    setsChanged = Signal(list)          # list[SetStatus]
    logMessage = Signal(str)
    errorOccurred = Signal(str, str)    # short message, full traceback
    busyChanged = Signal(bool)          # something is calculating, queued or being scanned
    calcSetChanged = Signal(str)        # the set being calculated right now ("" = none)
    reportMissing = Signal(str)
    dataDirChanged = Signal(str)
    downloadRequested = Signal(list)    # DatasetItems of a set to download (the app shows them in its datasets panel)
    statusMessage = Signal(str)

    def __init__(self, api: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self.statuses: list[SetStatus] = []
        self.untracked: list[dict] = []
        self.release: dict[str, str] = {}          # set code -> release date (newest set first in the table)
        self.overviews: dict[str, dict] = {}
        self._ov_sig: dict[str, tuple] = {}
        self.failed: dict[str, tuple[str, str]] = {}      # set -> (short message, traceback); lives for the session
        self.queue: list[tuple[str, dict]] = []
        self.running: str | None = None
        self.run_frac = 0.0
        self.pool_busy_set = ""                          # set the Pool tab is building right now: its calculation waits
        self._saved_set = ""
        self._io_jobs: list[tuple[str, Any, Any, str]] = []
        self._io_handler = None
        self._ov_wanted = ""
        self._ov_code = ""
        self._rescan_after_calc = False
        self._last_pct = -1
        self._last_scan = 0.0
        self.scan_count = 0

        self.task = TaskController(self)                  # the calculation
        self.task.progress.connect(self._on_progress)
        self.task.log.connect(self.logMessage)
        self.task.result.connect(self._on_calc_result)
        self.task.error.connect(self._on_calc_error)
        self.task.cancelled.connect(self._on_calc_cancelled)
        self.task.ended.connect(self._on_calc_ended)
        self.io_task = TaskController(self)               # listing, scanning, registering files
        self.io_task.result.connect(self._on_io_result)
        self.io_task.error.connect(self._on_io_error)
        self.io_task.ended.connect(self._on_io_ended)
        self.ov_task = TaskController(self)               # the statistics of the selected set
        self.ov_task.result.connect(self._on_overview)
        self.ov_task.ended.connect(self._on_ov_ended)

        self.watcher = QFileSystemWatcher(self)
        self.watcher.directoryChanged.connect(self._on_dir_changed)
        self.rescan_timer = QTimer(self)
        self.rescan_timer.setSingleShot(True)
        self.rescan_timer.setInterval(RESCAN_DEBOUNCE_MS)
        self.rescan_timer.timeout.connect(self._rescan_now)

        self._build_ui()
        self._load_settings()
        self._update_folder_label()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # the tracked data folder ------------------------------------------------
        frow = QHBoxLayout()
        frow.setSpacing(8)
        frow.addWidget(QLabel("<b>Data folder:</b>"))
        self.folder_label = ElidedLabel("", Qt.TextElideMode.ElideMiddle, min_chars=12)     # a long path: "…" in the middle, whole path in the tooltip
        frow.addWidget(self.folder_label)           # no stretch: it gets its full width when there is room
        self.change_btn = QPushButton("Change…")
        self.open_btn = QPushButton("Open")
        self.rescan_btn = QPushButton("Rescan")
        self.change_btn.setToolTip("Pick the folder the app follows (one sub-folder per set code, as the 17lands downloader writes it)")
        self.open_btn.setToolTip("Show the folder in the file explorer")
        self.rescan_btn.setToolTip("Look for new or changed 17lands files now (this also happens automatically)")
        for b in (self.change_btn, self.open_btn, self.rescan_btn):
            frow.addWidget(b)
        frow.addSpacing(10)
        self.folder_note = muted(ElidedLabel("17lands files placed here are picked up automatically"))
        self.scan_label = muted(ElidedLabel(""))
        self.scan_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        for lab in (self.folder_note, self.scan_label):         # the path and the buttons come first: these two take what is left (and elide)
            lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        frow.addWidget(self.folder_note, 2)
        frow.addWidget(self.scan_label, 1)
        root.addLayout(frow)
        self.change_btn.clicked.connect(self.choose_folder)
        self.open_btn.clicked.connect(self.open_folder)
        self.rescan_btn.clicked.connect(lambda: self.rescan(manual=True))

        # sets | statistics ------------------------------------------------------
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        hsplit.setChildrenCollapsible(False)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        ll.setSpacing(4)
        self.set_table = make_table(SET_COLS)
        self.set_table.verticalHeader().setDefaultSectionSize(row_height(self.set_table.font(), 32))      # comfortable rows in the main list
        self._interactive_columns(self.set_table, {})
        self.set_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)     # over left-aligned cells
        self._user_cols: set[int] = set()            # columns the user dragged: they keep their width
        self._last_min = 0
        self.set_table.viewport().installEventFilter(self)
        self._fitting = False
        self.set_table.horizontalHeader().sectionResized.connect(self._on_col_resized)
        self.set_table.itemSelectionChanged.connect(self._on_set_selected)
        ll.addWidget(self.set_table, 1)
        self.show_untracked_cb = QCheckBox("Show sets that are not tracked")
        self.show_untracked_cb.setChecked(True)
        self.show_untracked_cb.setToolTip("Greyed rows: sets with a folder in the data folder or on 17lands' list that the app does not track")
        self.show_untracked_cb.toggled.connect(lambda _on: (self._fill_set_table(self.selected_set()), self._on_set_selected()))
        ll.addWidget(self.show_untracked_cb)
        left.setMinimumWidth(max(300, 34 * self.fontMetrics().horizontalAdvance("0")))          # the set list stays readable
        hsplit.addWidget(left)
        self.panel = SetPanel()
        self.panel.fit_width()                                                                      # so do the statistics (a floor, then what the tables need)
        self.panel.trackRequested.connect(self._track_default_folder)
        self.panel.chooseFolderRequested.connect(self.choose_set_folder)
        self.panel.downloadRequested.connect(self.download_set)
        hsplit.addWidget(self.panel)
        hsplit.setStretchFactor(0, 5)
        hsplit.setStretchFactor(1, 5)
        hsplit.setSizes([700, 700])
        remember_splitter(hsplit, "sets/top_h3")
        root.addWidget(hsplit, 1)

        # calculation: compact ------------------------------------------------------
        calc = QFrame()
        calc.setFrameShape(QFrame.Shape.StyledPanel)
        calc.setProperty("card", True)
        cl = QVBoxLayout(calc)
        cl.setContentsMargins(10, 6, 10, 6)
        cl.setSpacing(4)
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self.selected_label = QLabel("")
        self.selected_label.setFont(sized_font(self.font(), 1, bold=True))
        row1.addWidget(self.selected_label)
        self.calc_btn = primary_button("Calculate")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_all_btn = QPushButton("Cancel all")
        self.report_btn = QPushButton("Open report")
        self.untrack_btn = QPushButton("Stop tracking…")
        self.cancel_btn.setToolTip("Cancel the calculation that is running (the queue goes on with the next set)")
        self.cancel_all_btn.setToolTip("Cancel the running calculation and empty the queue")
        self.untrack_btn.setToolTip("Remove this set from the app (its model and ingested data); your 17lands files are kept")
        for b in (self.calc_btn, self.cancel_btn, self.cancel_all_btn, self.report_btn, self.untrack_btn):
            row1.addWidget(b)
        row1.addSpacing(10)
        self.queue_strip = QueueStrip()
        self.queue_strip.removeRequested.connect(self.remove_from_queue)
        row1.addWidget(self.queue_strip, 1, Qt.AlignmentFlag.AlignVCenter)        # (its text on the buttons' centre line, not at their top)
        cl.addLayout(row1)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.progress = QProgressBar()
        self.progress.setObjectName("busyBar")                                       # (the style sheet gives the track an outline in both looks)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.progress.setFixedHeight(max(16, self.fontMetrics().height() + 4))        # holds the "%p%" text
        self.progress.setMinimumWidth(40)
        self.progress_label = muted(ElidedLabel("Idle"))                            # a long message is elided, full text in the tooltip
        row2.addWidget(self.progress, 3)
        row2.addWidget(self.progress_label, 5)
        self.advanced_btn = QToolButton()
        self.advanced_btn.setText("Advanced")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.log_btn = QToolButton()
        self.log_btn.setText("Log")
        self.log_btn.setCheckable(True)
        self.log_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.log_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        row2.addWidget(self.advanced_btn)
        row2.addWidget(self.log_btn)
        cl.addLayout(row2)
        self.advanced = QWidget()
        self._build_advanced(self.advanced)
        self.advanced.setVisible(False)
        cl.addWidget(self.advanced)
        self.log = LogPane()
        ls = self.log.fontMetrics().lineSpacing()
        self.log.setMinimumHeight(6 * ls + 10)
        self.log.setMaximumHeight(14 * ls + 10)
        self.log.setVisible(False)
        cl.addWidget(self.log)
        root.addWidget(calc)
        self.logMessage.connect(self.log.append_line)
        self.advanced_btn.toggled.connect(lambda on: (self.advanced.setVisible(on), self.advanced_btn.setArrowType(
            Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow), self._save("sets/advanced_open", on)))
        self.log_btn.toggled.connect(lambda on: (self.log.setVisible(on), self.log_btn.setArrowType(
            Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow), self._save("sets/log_open", on)))
        self.calc_btn.clicked.connect(self.calculate)
        self.cancel_btn.clicked.connect(self.cancel_running)
        self.cancel_all_btn.clicked.connect(self.cancel_all)
        self.report_btn.clicked.connect(self.open_report)
        self.untrack_btn.clicked.connect(self.stop_tracking)
        self.queue_strip.set_items(None, [])
        self._update_buttons()

    def _build_advanced(self, host: QWidget) -> None:
        """Settings of a calculation, kept out of the way (they apply to sets added to the queue from now on)."""
        outer = QHBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)
        ncpu = os.cpu_count() or 1
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, ncpu)
        self.workers_spin.setValue(ncpu)
        self.workers_spin.setToolTip("Parallel worker processes used for ingest and model fitting")
        self.boot_spin = QSpinBox()
        self.boot_spin.setRange(4, 100)
        self.boot_spin.setValue(16)
        self.boot_spin.setToolTip("Bootstrap replicates: more = smoother intervals, slower")
        self.recency_combo = QComboBox()
        for label, key in RECENCY_ITEMS:
            self.recency_combo.addItem(label, key)
        self.recency_combo.setToolTip(
            "Weight recent events more heavily than old ones (the format's metagame and players change).\n"
            "Auto: a half-life is chosen on older data and used only if it improves held-out predictions\n"
            "with a 95% interval above zero. The report shows the test either way.")
        self.attributes_combo = QComboBox()
        for label, key in RECENCY_ITEMS:
            self.attributes_combo.addItem(label, key)
        self.attributes_combo.setToolTip(
            "Card-attribute prior: shrink each card's effect towards what its mana value, rarity, types, keywords\n"
            "and rules-text tags predict, instead of towards the previous stage's estimate alone.\n"
            "Auto: used only if it improves held-out predictions with a 95% interval above zero.\n"
            "On: always used (the hold-out test still runs and is reported). The report shows the test either way.")
        self.manasim_combo = QComboBox()
        for label, key in (("Auto", "auto"), ("v2", "v2"), ("v1", "v1")):
            self.manasim_combo.addItem(label, key)
        self.manasim_combo.setToolTip(
            "Mana simulator behind the screw / flood / colour / mana-spent features.\n"
            "v2 is calibrated to real play: the hand and draws of each queue (Arena's Bo1 hand smoother, land-balanced Sealed\n"
            "draws), the real keep rule, extra cards from draw and land-search spells.\n"
            "Auto: v2 is used only if refitting the whole model on it improves held-out predictions with a 95% interval\n"
            "above zero. The report shows the test either way.")
        self.cast_combo = QComboBox()
        for label, key in RECENCY_ITEMS:
            self.cast_combo.addItem(label, key)
        self.cast_combo.setToolTip(
            "Castability: card values depend on whether the deck's mana can cast the card on time (Karsten-style\n"
            "simulation), and uncastable cards are priced as dead cards. A safety constraint against decks of strong\n"
            "off-colour cards: Auto ships it unless the hold-out test shows a material loss. On: always. Off: never.")
        self.extra_combo = QComboBox()
        for label, key in RECENCY_ITEMS:
            self.extra_combo.addItem(label, key)
        self.extra_combo.setToolTip(
            "Extra draft events: the draft games of PickTwo / QuickDraft events (the standard draft data are PremierDraft and\n"
            "TradDraft). Only sets that have such game files are affected.\n"
            "Auto: they are kept only if the model with them beats the model without them on the hold-out (decided first,\n"
            "with a 95% interval above zero, confirmed on the later half). On: always kept. Off: never.")
        self.matchups_combo = QComboBox()
        for label, key in MATCHUPS_ITEMS:
            self.matchups_combo.addItem(label, key)
        self.matchups_combo.setToolTip(
            "Opponent matchups: card values against the opponent's colours (sideboard advice only; the deck score and the builder\n"
            "never use them).\n"
            "Pooled across sets: used where this set's own hold-out test passes (Auto), and in each other format where the hold-out\n"
            "evidence pooled over the OTHER sets' latest models passes (at least 5 other sets) and this set's own numbers are not\n"
            "materially worse. Auto: this set's own test only. On: always. Off: never. The report shows the evidence either way.")
        self.fetch_cb = QCheckBox("Fetch card data from Scryfall")
        self.fetch_cb.setChecked(True)
        self.fetch_cb.setToolTip("Needs internet; without it card types and colours are inferred from the data")
        for title, widgets in ((None, [("CPU workers", self.workers_spin), ("Bootstrap replicates", self.boot_spin)]),
                               (None, [("Recency weighting", self.recency_combo), ("Card-attribute prior", self.attributes_combo)]),
                               (None, [("Mana simulator", self.manasim_combo), ("Castability", self.cast_combo)]),
                               (None, [("Extra draft events", self.extra_combo), ("Opponent matchups", self.matchups_combo)])):
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            for lab, w in widgets:
                form.addRow(lab, w)
            outer.addLayout(form)
            outer.addSpacing(18)
        outer.addWidget(self.fetch_cb, 0, Qt.AlignmentFlag.AlignTop)
        outer.addStretch(1)

    @staticmethod
    def _interactive_columns(t, widths: dict[int, int]) -> None:
        """Every column resizable by the user (double-click a divider to fit); the last one fills the rest."""
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(True)
        hh.setMinimumSectionSize(36)
        t.setHorizontalScrollMode(t.ScrollMode.ScrollPerPixel)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        t.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        for c, w in widths.items():
            t.setColumnWidth(c, w)

    def _on_col_resized(self, c: int, _old: int, _new: int) -> None:
        if not self._fitting and c < self.set_table.columnCount() - 1:       # (the last column follows the room that is left)
            self._user_cols.add(c)

    def _fit_set_columns(self) -> None:
        """Column widths from the content (and the header text), so a status or a games count is not cut: only columns the user has
        not dragged are fitted; the last one takes the rest."""
        t = self.set_table
        hfm = t.horizontalHeader().fontMetrics()
        self._fitting = True
        try:
            for c in range(t.columnCount() - 1):
                if c in self._user_cols:
                    continue
                head = hfm.horizontalAdvance(t.horizontalHeaderItem(c).text()) + 24
                t.setColumnWidth(c, max(t.sizeHintForColumn(c) + 8, head))
            last = t.columnCount() - 1
            self._last_min = max(t.sizeHintForColumn(last) + 8, hfm.horizontalAdvance(t.horizontalHeaderItem(last).text()) + 24)
        finally:
            self._fitting = False
        self._stretch_last()

    def _stretch_last(self) -> None:
        """The last column ("Built at") fills the room that is left but never gets narrower than its content: below that the table
        scrolls sideways (a stretched last section would squeeze it and cut the date)."""
        t = self.set_table
        last = t.columnCount() - 1
        if last < 0:
            return
        t.horizontalHeader().setStretchLastSection(False)
        others = sum(t.columnWidth(c) for c in range(last))
        self._fitting = True
        try:
            t.setColumnWidth(last, max(self._last_min, t.viewport().width() - others))
        finally:
            self._fitting = False

    def eventFilter(self, obj, event) -> bool:
        if obj is self.set_table.viewport() and event.type() == QEvent.Type.Resize:
            self._stretch_last()
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ settings
    def _load_settings(self) -> None:
        s = new_settings()
        try:
            w = int(s.value("workers", self.workers_spin.value()))
            self.workers_spin.setValue(max(self.workers_spin.minimum(), min(self.workers_spin.maximum(), w)))
            self.boot_spin.setValue(int(s.value("n_boot", self.boot_spin.value())))
            self.fetch_cb.setChecked(str(s.value("fetch_scryfall", "true")).lower() in ("true", "1"))
            for combo, key in ((self.recency_combo, "recency"), (self.attributes_combo, "attributes"), (self.manasim_combo, "manasim"),
                               (self.cast_combo, "castability"), (self.extra_combo, "extra_draft")):
                combo.setCurrentIndex(max(combo.findData(str(s.value(key, "auto"))), 0))
            self.matchups_combo.setCurrentIndex(max(self.matchups_combo.findData(str(s.value("matchups", "pooled"))), 0))
        except (TypeError, ValueError):
            pass
        self._saved_set = str(s.value("last_set", "") or "")
        self.workers_spin.valueChanged.connect(lambda v: self._save("workers", v))
        self.boot_spin.valueChanged.connect(lambda v: self._save("n_boot", v))
        self.fetch_cb.toggled.connect(lambda v: self._save("fetch_scryfall", v))
        self.recency_combo.currentIndexChanged.connect(lambda _i: self._save("recency", self.recency_combo.currentData()))
        self.attributes_combo.currentIndexChanged.connect(lambda _i: self._save("attributes", self.attributes_combo.currentData()))
        self.manasim_combo.currentIndexChanged.connect(lambda _i: self._save("manasim", self.manasim_combo.currentData()))
        self.cast_combo.currentIndexChanged.connect(lambda _i: self._save("castability", self.cast_combo.currentData()))
        self.extra_combo.currentIndexChanged.connect(lambda _i: self._save("extra_draft", self.extra_combo.currentData()))
        self.matchups_combo.currentIndexChanged.connect(lambda _i: self._save("matchups", self.matchups_combo.currentData()))
        if str(s.value("sets/advanced_open", "false")).lower() in ("true", "1"):
            self.advanced_btn.setChecked(True)
        if str(s.value("sets/log_open", "false")).lower() in ("true", "1"):
            self.log_btn.setChecked(True)

    @staticmethod
    def _save(key: str, value) -> None:
        new_settings().setValue(key, value)

    # ------------------------------------------------------------------ the data folder
    def data_dir(self) -> str:
        v = new_settings().value("datasets/data_dir", "")
        if v:
            return str(v)
        fn = getattr(self.api, "datasets_data_dir", None)
        return str(fn()) if fn is not None else ""

    def _update_folder_label(self) -> None:
        d = self.data_dir()
        ok = bool(d) and Path(d).is_dir()
        pal = self.folder_label.palette()
        pal.setColor(self.folder_label.foregroundRole(), semantic()["bad"] if not ok else QApplication.palette().color(self.folder_label.foregroundRole()))
        self.folder_label.setPalette(pal)
        self.folder_label.setText((d or "(not set)") + ("" if ok else " — folder not found"))
        self.folder_label.setToolTip(d)
        self.open_btn.setEnabled(ok)
        self._sync_watcher()

    def choose_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "17lands data folder", self.data_dir())
        if d:
            self.set_data_dir(d)

    def set_data_dir(self, d: str, notify: bool = True) -> None:
        """Follow another folder: remembered, watched, scanned at once."""
        new_settings().setValue("datasets/data_dir", d)
        self._update_folder_label()
        if notify:
            self.dataDirChanged.emit(d)
        self.logMessage.emit(f"Data folder: {d}")
        self.rescan(manual=True)

    def open_folder(self) -> None:
        d = self.data_dir()
        if d and Path(d).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _watch_paths(self) -> list[str]:
        d = self.data_dir()
        out = []
        if d and Path(d).is_dir():
            out.append(d)
            for st in self.statuses:
                f = st.folder or str(Path(d) / st.set_code)
                if Path(f).is_dir():
                    out.append(f)
        return out

    def _sync_watcher(self) -> None:
        want = {os.path.normpath(p) for p in self._watch_paths()}
        have = {os.path.normpath(p) for p in self.watcher.directories()}
        if have - want:
            self.watcher.removePaths(sorted(have - want))
        if want - have:
            self.watcher.addPaths(sorted(want - have))

    def _on_dir_changed(self, _path: str) -> None:
        self.rescan_timer.start()            # debounce: a download touches the folder many times

    def on_app_activated(self) -> None:
        """The window became active again: look for files that changed while it was in the background."""
        if time.monotonic() - self._last_scan > ACTIVATE_RESCAN_S:
            self.rescan()

    # ------------------------------------------------------------------ state
    def selected_set(self) -> str | None:
        rows = self.set_table.selectionModel().selectedRows() if self.set_table.selectionModel() else []
        if rows:
            it = self.set_table.item(rows[0].row(), 0)
            return it.text() if it else None
        return None

    def status_for(self, set_code: str | None) -> SetStatus | None:
        return next((s for s in self.statuses if s.set_code == set_code), None)

    def untracked_for(self, set_code: str | None) -> dict | None:
        return next((u for u in self.untracked if u["set_code"] == set_code), None)

    @property
    def busy(self) -> bool:
        return bool(self.running or self.queue or self.io_task.running or self._io_jobs or self.ov_task.running)

    def queued(self) -> list[str]:
        return [c for c, _ in self.queue]

    def active_sets(self) -> set[str]:
        """Sets that a calculation is running on or waiting for."""
        return set(self.queued()) | ({self.running} if self.running else set())

    def state_of(self, code: str) -> dict:
        """{key, text, role, tip, failed} of one set's status in the list."""
        st = self.status_for(code)
        if code == self.running:
            return dict(key="running", text=f"Calculating {int(self.run_frac * 100)}%", role="accent", tip=H.set_status_tip("running"))
        if code in self.queued():
            n = self.queued().index(code) + 1
            return dict(key="queued", text=f"Queued (#{n})", role="accent", tip=H.set_status_tip("queued"))
        if code in self.failed:
            short = self.failed[code][0]
            return dict(key="failed", text="Failed — see details", role="bad", tip=H.set_status_tip("failed", short),
                        failed=self.failed[code])
        if st is None:
            return dict(key="untracked", text="Not tracked", role=None, tip=H.set_status_tip("untracked"))
        if not any(f.kind == "game" for f in st.files):
            return dict(key="empty", text="No game data", role="warn", tip="This set has no game_data file registered.")
        if st.stale or not st.has_model:
            detail = "<br>".join(st.pending) if st.pending else ""
            return dict(key="pending", text="Recalculation pending", role="warn", tip=H.set_status_tip("pending", detail))
        return dict(key="ready", text="Ready", role="good", tip=H.set_status_tip("ready"))

    # ------------------------------------------------------------------ jobs on the worker thread (listing, scanning, registering)
    def _io(self, name: str, fn, handler, label: str) -> None:
        """Queue a job; a job with the same name that has not started yet is not queued twice."""
        if any(j[0] == name for j in self._io_jobs):
            return
        self._io_jobs.append((name, fn, handler, label))
        self._pump_io()

    def _pump_io(self) -> None:
        if self.io_task.running or not self._io_jobs:
            return
        _name, fn, handler, label = self._io_jobs.pop(0)
        self._io_handler = handler
        self.io_task.run(fn, label=label)

    def _on_io_result(self, res: object) -> None:
        h, self._io_handler = self._io_handler, None
        if h is not None:
            h(res)

    def _on_io_error(self, tb: str) -> None:
        self._io_handler = None
        self.progress_label.setText("Failed")
        self.errorOccurred.emit(short_error(tb), tb)

    def _on_io_ended(self) -> None:
        self._pump_io()
        if not self.io_task.running and not self._io_jobs:
            self.busyChanged.emit(self.busy)

    def _lists(self, api, dd: str) -> dict:
        out = {"sets": api.list_sets()}
        fn = getattr(api, "untracked_sets", None)
        out["untracked"] = fn(dd) if fn is not None else []
        fn = getattr(api, "set_release_dates", None)
        codes = [s.set_code for s in out["sets"]] + [u["set_code"] for u in out["untracked"]]
        out["release"] = fn(codes) if fn is not None else {}
        return out

    def refresh(self) -> None:
        api, dd = self.api, self.data_dir()

        def job(progress, cancel):
            return self._lists(api, dd)
        self._io("refresh", job, self._apply_lists, "list sets")

    def start(self) -> None:
        """First look at start-up: list the sets, then register what is new in the data folder."""
        self.refresh()
        self.rescan()

    def rescan(self, manual: bool = False) -> None:
        """Register new / changed files of the tracked folders now (after a short pause when the folder is still changing)."""
        if manual:
            self.rescan_timer.stop()
            self._rescan_now(manual=True)
        else:
            self.rescan_timer.start()

    def _rescan_now(self, manual: bool = False, track: list[str] | None = None) -> None:
        fn = getattr(self.api, "scan_data_folders", None)
        if fn is None:
            self.refresh()
            return
        api, dd = self.api, self.data_dir()
        skip = [self.running] if self.running else []          # a set being calculated is left alone until it is done
        track = list(track or [])

        def job(progress, cancel):
            rep = fn(dd, track=track or None, skip_sets=skip or None, progress=progress, cancel=cancel)
            return dict(report=rep, **self._lists(api, dd))
        self._last_scan = time.monotonic()
        self.scan_count += 1
        if manual:
            self.progress_label.setText("Scanning the data folder…")
        self._io("scan", job, lambda res: self._on_scan(res, manual), "scan data folder")

    def _on_scan(self, res: dict, manual: bool) -> None:
        rep = res["report"]
        if rep.added:
            names = ", ".join(sorted({Path(f.path).name for f in rep.added})[:3]) + (" …" if len(rep.added) > 3 else "")
            msg = f"Data folder: {len(rep.added)} new or changed file(s) registered for {', '.join(rep.changed_sets) or 'shared data'}: {names}"
            self.logMessage.emit(msg)
            self.statusMessage.emit(msg)
            self.scan_label.setText(f"Last scan {time.strftime('%H:%M')}: {len(rep.added)} new file(s) in {', '.join(rep.changed_sets)}")
        else:
            self.scan_label.setText(f"Last scan {time.strftime('%H:%M')}: nothing new")
            if manual:
                self.statusMessage.emit("Data folder scanned: nothing new")
        for n in rep.notes[:5]:
            self.logMessage.emit(f"Data folder: {n}")
        if rep.deferred and self.running:
            self._rescan_after_calc = True
        if manual and not self.running:
            self.progress_label.setText("Idle")
        self._apply_lists(res)

    def _apply_lists(self, res: dict) -> None:
        self.release = res.get("release") or self.release
        self.set_statuses(res["sets"], res.get("untracked"))

    def set_statuses(self, statuses: list[SetStatus], untracked: list[dict] | None = None) -> None:
        keep = self.selected_set() or self._saved_set
        self.statuses = list(statuses)
        if untracked is not None:
            self.untracked = list(untracked)
        for st in self.statuses:                  # the cached statistics of a set whose data / model changed are stale
            sig = (st.built_at, st.stale, tuple(sorted(f.fingerprint for f in st.files)), tuple(st.missing))
            if self._ov_sig.get(st.set_code) != sig:
                self.overviews.pop(st.set_code, None)
                self._ov_sig[st.set_code] = sig
        self._fill_set_table(keep)
        self._sync_watcher()
        self._on_set_selected()
        self.setsChanged.emit(self.statuses)

    # ------------------------------------------------------------------ set table
    def _rows(self) -> list[tuple[str, bool]]:
        have = {s.set_code for s in self.statuses}
        rows = [(c, True) for c in have]
        if self.show_untracked_cb.isChecked():
            rows += [(u["set_code"], False) for u in self.untracked if u["set_code"] not in have]
        rank = {c: i for i, c in enumerate(order_sets([c for c, _ in rows], self.release))}
        return sorted(rows, key=lambda r: (not r[1], rank[r[0]]))       # tracked sets first, each group newest release first

    def _fill_set_table(self, keep: str | None) -> None:
        sem = semantic()
        t = self.set_table
        rows = self._rows()
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for r, (code, tracked) in enumerate(rows):
            st = self.status_for(code)
            if tracked:
                state = self.state_of(code)
                games = fmt_games(st.n_games)
                built = st.built_at or "—"
                if st.built_at:
                    built = fmt_time(st.built_at)
                tip_extra = "\n".join(st.warnings)
            else:
                u = self.untracked_for(code) or {}
                state = self.state_of(code)
                bits = []
                if u.get("n_local"):
                    bits.append(f"{u['n_local']} file(s) on disk")
                if u.get("n_remote"):
                    bits.append(f"{u['n_remote']} on 17lands")
                games = " · ".join(bits) or "—"
                built = "—"
                tip_extra = ""
            cells = [code, self._badge(state), games, built]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 0:
                    it.setFont(sized_font(t.font(), 1, bold=True))
                if c == 1:
                    it.setToolTip(state["tip"])
                    if state["role"]:
                        it.setForeground(QBrush(sem[state["role"]]))
                elif tip_extra:
                    it.setToolTip(tip_extra)
                if not tracked:
                    f = it.font()
                    f.setItalic(True)
                    it.setFont(f)
                    it.setForeground(QBrush(t.palette().placeholderText().color()))
                t.setItem(r, c, it)
        t.blockSignals(False)
        self._fit_set_columns()
        codes = [c for c, _ in rows]
        target = keep if keep in codes else (codes[0] if codes else None)
        if target:
            t.selectRow(codes.index(target))
        else:
            t.clearSelection()

    @staticmethod
    def _badge(state: dict) -> str:
        mark = {"ready": "●", "pending": "●", "failed": "●", "running": "◔", "queued": "◔", "untracked": "○", "empty": "○"}.get(state["key"], "●")
        return f"{mark} {state['text']}"

    def _update_row(self, code: str) -> None:
        """Refresh the status cell of one row in place (progress ticks)."""
        t = self.set_table
        for r in range(t.rowCount()):
            if t.item(r, 0).text() == code:
                state = self.state_of(code)
                it = t.item(r, 1)
                it.setText(self._badge(state))
                it.setToolTip(state["tip"])
                if state["role"]:
                    it.setForeground(QBrush(semantic()[state["role"]]))
                self._fit_set_columns()
                return

    def _on_set_selected(self) -> None:
        sc = self.selected_set()
        if sc:
            self._save("last_set", sc)
        self._show_selected()
        self._update_buttons()

    def _show_selected(self) -> None:
        sc = self.selected_set()
        st = self.status_for(sc)
        if sc is None:
            self.panel.clear()
        elif st is not None:
            self.panel.show_set(st, self.overviews.get(sc), self.state_of(sc))
            if sc not in self.overviews:
                self._load_overview(sc)
        else:
            u = self.untracked_for(sc) or dict(set_code=sc, folder="", n_local=0, n_remote=0, events=[])
            self.panel.show_untracked(u, self.data_dir(), bool(u.get("n_remote")) and getattr(self.api, "dataset_files_for_set", None) is not None)

    def _load_overview(self, code: str) -> None:
        fn = getattr(self.api, "set_overview", None)
        if fn is None:
            return
        if self.ov_task.running:
            self._ov_wanted = code
            return
        self._ov_code = code

        def job(progress, cancel):
            return fn(code)
        self.ov_task.run(job, label=f"read statistics of {code}")

    def _on_overview(self, res: object) -> None:
        if not isinstance(res, dict):
            return
        code = res.get("set_code") or self._ov_code
        self.overviews[code] = res
        if code == self.selected_set() and self.status_for(code) is not None:
            self.panel.show_set(self.status_for(code), res, self.state_of(code))

    def _on_ov_ended(self) -> None:
        want, self._ov_wanted = self._ov_wanted, ""
        if want and want not in self.overviews and want == self.selected_set():
            self._load_overview(want)

    # ------------------------------------------------------------------ buttons
    def _update_buttons(self) -> None:
        sc = self.selected_set()
        st = self.status_for(sc)
        active = bool(self.running or self.queue)
        self.calc_btn.setText("Add to queue" if active else "Calculate")
        can = st is not None and any(f.kind == "game" for f in st.files) and sc != self.running and sc not in self.queued()
        self.calc_btn.setEnabled(can)
        if st is not None and (sc == self.running or sc in self.queued()):
            self.calc_btn.setToolTip(f"{sc} is already {'being calculated' if sc == self.running else 'in the queue'}")
        else:
            self.calc_btn.setToolTip("Fit the model for this set" + (" (it is added to the queue behind the running calculation)" if active else ""))
        self.cancel_btn.setEnabled(self.running is not None)
        self.cancel_all_btn.setVisible(bool(self.queue))
        self.cancel_all_btn.setEnabled(active)
        self.report_btn.setEnabled(st is not None and st.has_model)
        self.untrack_btn.setEnabled(st is not None and sc != self.running and sc not in self.queued())
        self.selected_label.setText(f"Set: {sc}" if sc else "Select a set")
        self.queue_strip.set_items((self.running, int(self.run_frac * 100)) if self.running else None, self.queued(),
                                   waiting_for=self.pool_busy_set if self.pool_busy_set in self.queued() and not self.running else "")

    # ------------------------------------------------------------------ the calculation queue
    def calculate(self) -> None:
        sc = self.selected_set()
        if sc is not None:
            self.enqueue(sc)

    def _options(self) -> dict:
        return dict(n_jobs=self.workers_spin.value(), n_boot=self.boot_spin.value(), fetch_scryfall=self.fetch_cb.isChecked(),
                    recency=self.recency_combo.currentData() or "auto", attributes=self.attributes_combo.currentData() or "auto",
                    manasim=self.manasim_combo.currentData() or "auto", castability=self.cast_combo.currentData() or "auto",
                    extra_draft=self.extra_combo.currentData() or "auto", matchups=self.matchups_combo.currentData() or "pooled")

    def enqueue(self, code: str) -> bool:
        """Add a set to the calculation queue (the options are those of the Advanced group now)."""
        st = self.status_for(code)
        if st is None or code == self.running or code in self.queued():
            return False
        self.queue.append((code, self._options()))
        self.failed.pop(code, None)
        self.logMessage.emit(f"Queued {code}" + (f" (position {len(self.queue)})" if self.running or len(self.queue) > 1 else ""))
        self._update_row(code)
        self._update_buttons()
        self._show_selected()
        self.busyChanged.emit(True)
        self._start_next()
        return True

    def remove_from_queue(self, code: str) -> None:
        before = len(self.queue)
        self.queue = [(c, o) for c, o in self.queue if c != code]
        if len(self.queue) != before:
            self.logMessage.emit(f"Removed {code} from the queue")
            self._update_row(code)
            self._update_buttons()
            self._show_selected()
            if not self.busy:
                self.busyChanged.emit(False)

    def cancel_running(self) -> None:
        self.task.cancel()

    def cancel_all(self) -> None:
        gone = self.queued()
        self.queue.clear()
        for c in gone:
            self._update_row(c)
        self.task.cancel()
        self._update_buttons()
        if not self.running:
            self.busyChanged.emit(self.busy)

    def set_pool_busy(self, code: str) -> None:
        """The Pool tab is building on `code`: that set's calculation waits for it."""
        self.pool_busy_set = code or ""
        self._update_buttons()
        self._start_next()

    def _start_next(self) -> None:
        if self.running is not None or not self.queue:
            return
        pick = next((i for i, (c, _o) in enumerate(self.queue) if c != self.pool_busy_set), None)
        if pick is None:
            self.progress_label.setText(f"Waiting for the deck builder to finish with {self.pool_busy_set}…")
            return
        sc, o = self.queue.pop(pick)
        self.running, self.run_frac = sc, 0.0
        self.progress.setValue(0)
        self.progress_label.setText(f"{sc}: starting…")
        self.logMessage.emit(f"=== Calculate {sc}: {o['n_jobs']} worker(s), {o['n_boot']} bootstrap replicates, "
                             f"Scryfall {'on' if o['fetch_scryfall'] else 'off'}, recency weighting {o['recency']}, "
                             f"card-attribute prior {o['attributes']}, mana simulator {o['manasim']}, castability {o['castability']}, "
                             f"extra draft events {o['extra_draft']}, opponent matchups {o['matchups']} ===")
        api, dd = self.api, self.data_dir()

        calc = getattr(api, "calculate_set_isolated", None) or api.calculate_set      # own process: a native crash spares the app

        def job(progress, cancel):
            st = calc(sc, progress=progress, cancel=cancel, **o)
            return {"status": st, **self._lists(api, dd)}
        self.task.run(job, label=f"calculate {sc}")
        self._update_row(sc)
        self._update_buttons()
        self._show_selected()
        self.calcSetChanged.emit(sc)
        self.busyChanged.emit(True)

    def _on_progress(self, frac: float, msg: str) -> None:
        self.run_frac = max(0.0, min(1.0, frac))
        self.progress.setValue(int(self.run_frac * 1000))
        if msg and self.running:
            self.progress_label.setText(f"{self.running}: {msg}")
        if self.running:
            self._update_row(self.running)
            if int(self.run_frac * 100) != self._last_pct:
                self._last_pct = int(self.run_frac * 100)
                self.queue_strip.set_items((self.running, self._last_pct), self.queued())
                if self.selected_set() == self.running:
                    self.panel.set_status_pill(self.state_of(self.running)["text"], "accent")

    def _on_calc_result(self, res: object) -> None:
        st = res["status"]  # type: ignore[index]
        self.progress.setValue(1000)
        self.progress_label.setText(f"Done: {st.set_code} model built")
        for w in st.warnings:
            self.logMessage.emit(f"WARNING {w}")
        self.failed.pop(st.set_code, None)
        self.overviews.pop(st.set_code, None)
        self._apply_lists(res)  # type: ignore[arg-type]

    def _on_calc_error(self, tb: str) -> None:
        sc = self.running or "?"
        short = short_error(tb)
        self.failed[sc] = (short, tb)
        self.progress_label.setText(f"{sc}: failed — {short}")
        self.errorOccurred.emit(f"Calculating {sc} failed: {short}", tb)

    def _on_calc_cancelled(self) -> None:
        self.progress.setValue(0)
        self.progress_label.setText(f"{self.running or ''}: cancelled".strip(": "))
        self.logMessage.emit("Cancelled by user.")

    def _on_calc_ended(self) -> None:
        done, self.running = self.running, None
        self.run_frac = 0.0
        if done:
            self._update_row(done)
        self.refresh()                                   # what is on disk may have changed (a failed or cancelled run too)
        self._update_buttons()
        self._show_selected()
        self.calcSetChanged.emit("")
        if self.queue:
            self._start_next()
            return
        if self._rescan_after_calc:
            self._rescan_after_calc = False
            self.rescan()
        self.busyChanged.emit(self.busy)

    # ------------------------------------------------------------------ tracking sets, dropped paths
    def track_set(self, code: str, folder: str | None = None) -> None:
        """Start tracking a set: optionally from its own folder; its files are registered at once."""
        api, dd, code = self.api, self.data_dir(), code.upper()
        set_folder = getattr(api, "set_set_folder", None)
        scan = getattr(api, "scan_data_folders", None)

        def job(progress, cancel):
            if folder and set_folder is not None:
                set_folder(code, folder)
            rep = scan(dd, track=[code], progress=progress, cancel=cancel) if scan is not None else None
            return dict(report=rep, **self._lists(api, dd))
        self.logMessage.emit(f"Tracking {code}" + (f" from {folder}" if folder else ""))
        self._io(f"track {code}", job, lambda res, c=code: self._on_tracked(c, res), f"track set {code}")

    def _on_tracked(self, code: str, res: dict) -> None:
        rep = res.get("report")
        added = [f for f in (rep.added if rep else []) if (f.set_code or "").upper() == code]
        if added:
            self.logMessage.emit(f"{code}: {len(added)} file(s) registered; press Calculate to build its model")
            self.statusMessage.emit(f"{code} is tracked now ({len(added)} files)")
        else:
            self.logMessage.emit(f"{code}: no 17lands files found")
            self.errorOccurred.emit(f"No 17lands files for {code} were found in that folder. Dataset files are called "
                                    f"game_data_public.{code}.<event>.csv.gz.", "")
        self._saved_set = code
        self._apply_lists(res)
        self.rescan_timer.stop()

    def _confirm_untrack(self, code: str, st: SetStatus) -> bool:
        model = " its calculated model," if st.has_model else ""
        text = (f"Stop tracking {code}?\n\nThe app deletes{model} its ingested data and its list of files. Your 17lands files"
                + (f" in {st.folder}" if st.folder else "") + " are not touched, and you can track the set again later"
                + (" (its model then needs Calculate again)." if st.has_model else "."))
        return QMessageBox.question(self, "Stop tracking", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                    QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes

    def stop_tracking(self) -> None:
        sc = self.selected_set()
        st = self.status_for(sc)
        fn = getattr(self.api, "untrack_set", None)
        if st is None or fn is None or sc == self.running or sc in self.queued() or not self._confirm_untrack(sc, st):
            return
        api, dd = self.api, self.data_dir()

        def job(progress, cancel):
            fn(sc)
            return self._lists(api, dd)
        self.logMessage.emit(f"Stopped tracking {sc}")
        self._io(f"untrack {sc}", job, lambda res, c=sc: self._on_untracked(c, res), f"stop tracking {sc}")

    def _on_untracked(self, code: str, res: dict) -> None:
        self.overviews.pop(code, None)
        self.statusMessage.emit(f"{code} is no longer tracked")
        self._apply_lists(res)

    def _track_default_folder(self, code: str) -> None:
        u = self.untracked_for(code) or {}
        self.track_set(code, u.get("folder") or None)

    def choose_set_folder(self, code: str) -> None:
        d = QFileDialog.getExistingDirectory(self, f"Folder with the 17lands files of {code}", self.data_dir())
        if d:
            self.track_set(code, d)

    def download_set(self, code: str) -> None:
        fn = getattr(self.api, "dataset_files_for_set", None)
        items = fn(code, self.data_dir()) if fn is not None else []
        if not items:
            self.statusMessage.emit(f"17lands lists no files for {code} that are missing from the data folder")
            return
        self.downloadRequested.emit(list(items))

    def add_paths(self, paths: list[str]) -> None:
        """Dropped on the window.  A folder becomes the tracked folder of the set it holds (nothing is copied); files are registered."""
        paths = [os.path.normpath(str(p)) for p in paths]
        self.logMessage.emit(f"Scanning {len(paths)} dropped path(s)…")
        self.progress_label.setText("Scanning files…")
        dirs = [p for p in paths if Path(p).is_dir()]
        files = [p for p in paths if p not in dirs]
        api, dd = self.api, self.data_dir()

        def job(progress, cancel):
            added, tracked = [], []
            codes_fn = getattr(api, "folder_set_codes", None)
            for d in dirs:
                codes = codes_fn(d) if codes_fn is not None else {}
                if len(codes) == 1:
                    code = next(iter(codes))
                    default = Path(dd) / code
                    if os.path.normcase(os.path.abspath(d)) != os.path.normcase(os.path.abspath(default)):
                        api.set_set_folder(code, d)
                    rep = api.scan_data_folders(dd, track=[code], progress=progress, cancel=cancel)
                    added += list(rep.added)
                    tracked.append((code, d))
                else:
                    files.append(d)               # several sets (or none we can tell): register everything found, as for files
            if files:
                added += list(api.add_files(files))
            return dict(added=added, tracked=tracked, **self._lists(api, dd))
        self._io("drop", job, self._on_dropped, "register dropped files")

    def _on_dropped(self, res: dict) -> None:
        added = res.get("added") or []
        for code, d in res.get("tracked") or []:
            self.logMessage.emit(f"{code} is now read from {d}")
            self._saved_set = code
        if added:
            self.logMessage.emit(f"Added {len(added)} file(s): " + ", ".join(sorted({f"{f.set_code or '?'} {f.kind}" for f in added})))
        self.progress_label.setText("Idle" if not self.running else self.progress_label.text())
        self._apply_lists(res)

    # ------------------------------------------------------------------ report
    def open_report(self) -> None:
        sc = self.selected_set()
        if not sc:
            return
        try:
            p = self.api.report_path(sc)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.errorOccurred.emit(f"Could not locate the report: {exc}", traceback.format_exc())
            return
        if not p or not Path(p).exists():
            self.reportMissing.emit(sc)
            self.logMessage.emit(f"No report for {sc} yet – run Calculate first.")
            self.errorOccurred.emit(f"No report for {sc} yet. Run Calculate first.", "")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def wait(self, ms: int = 8000) -> None:
        """Stop everything (empties the queue, cancels the running calculation) and wait for the threads."""
        self.queue.clear()
        self.rescan_timer.stop()
        self.task.wait(ms)
        self.io_task.wait(ms)
        self.ov_task.wait(ms)
