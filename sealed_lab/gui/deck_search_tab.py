"""The "Deck search" tab: pick a set and a card, see the real decks (17lands runs) that played it, best first.

For a card whose place in a deck is hard to imagine.  Everything is read from the set's corpus through `api.deck_search_*`
(background tasks, one at a time); the tab holds the list of runs on the left and the selected deck's list on the right.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
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

SORTS = [("Best result", "result"), ("Win rate", "win_rate"), ("Newest", "newest"), ("Oldest", "oldest")]
PAGE = 200                          # decks fetched per request
DEBOUNCE_MS = 250                   # a filter change searches again after this long
TROPHY = "★"                   # a black star marks a run that reached the event's maximum wins
CARD_W_DEFAULT, CARD_W_MIN, CARD_W_MAX = 110, 70, 170   # width in px of a card image in the main deck's pile
SIDE_SCALE = 0.7                 # the sideboard pile's cards are this much smaller
COLUMNS = ["Record", "Format", "Date", "Colours", "Copies", "Card in games", "Rank"]
INTRO = ("Pick a card, see the decks that played it.\n\n"
         "Type a card name (the list completes it) and press Enter.  The decks are real 17Lands runs of the set, best result first.  "
         "Use the filters to choose the formats, and sort by result or by date.")
SHARE_TIP = ("When ticked, only runs where the card was in the deck for most of the run are listed.\n"
             "In Bo3 events and runs with rebuilds the deck can change between games, so a run may have played the card in only some "
             "of its games.  Unticked, any run that played it at least once is listed.")


def count_text(n: int) -> str:
    return f"{n:,}".replace(",", " ")


class ReadOnlyPile(PileView):
    """A PileView that only shows cards (and reports the hovered one): no drag, drop, double-click or context menu, no action signal used."""

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


class DeckSearchTab(QWidget):
    logMessage = Signal(str)
    statusMessage = Signal(str)
    errorOccurred = Signal(str, str)

    def __init__(self, api: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self.statuses: dict[str, Any] = {}
        self.calculating: set[str] = set()
        self.cards: list[str] = []                           # autocomplete names of the loaded set
        self.events: list[dict] = []                         # the loaded set's events (the format checkboxes)
        self._meta_key: tuple[str, str] | None = None        # (set, built_at) the names and events belong to
        self._meta_failed: tuple[str, str] | None = None
        self._pending: str | None = None                     # "new" / "more": a search wanted as soon as the task is free
        self.piles: list[PileView] = []                      # the pile views of the deck on screen
        self.results: list[dict] = []                        # the runs on screen
        self.data: dict | None = None                        # the last answer (totals)
        self.searched: tuple[str, str] | None = None         # (set, card name) of the answer on screen
        self._format_boxes: dict[str, QCheckBox] = {}
        self._filling = False

        self.task = TaskController(self)
        self.task.started.connect(self._on_started)
        self.task.result.connect(self._on_result)
        self.task.error.connect(self._on_error)
        self.task.cancelled.connect(lambda: self.strip.end("Cancelled"))
        self.task.ended.connect(self._on_ended)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(DEBOUNCE_MS)
        self.debounce.timeout.connect(lambda: self.request_search("new"))

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("Deck search")
        title.setFont(sized_font(self.font(), 3, bold=True))
        head.addWidget(title)
        head.addSpacing(10)
        head.addWidget(QLabel("Set"))
        self.set_combo = ElidedCombo()
        self.set_combo.setToolTip("Sets with data to search")
        self.set_combo.activated.connect(lambda _i: self._apply_set())
        head.addWidget(self.set_combo)
        head.addSpacing(10)
        self.card_edit = QLineEdit()
        self.card_edit.setPlaceholderText("Card name")
        self.card_edit.setClearButtonEnabled(True)
        self.card_edit.setMinimumWidth(260)
        self.card_edit.returnPressed.connect(lambda: self.request_search("new"))
        self.completer = QCompleter(QStringListModel([]), self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.setMaxVisibleItems(12)
        self.completer.activated.connect(self._on_completed)
        self.card_edit.setCompleter(self.completer)
        head.addWidget(self.card_edit, 1)
        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("primary")
        self.search_btn.clicked.connect(lambda: self.request_search("new"))
        head.addWidget(self.search_btn)
        root.addLayout(head)

        self.format_row = QWidget()                          # the formats get a full-width row of their own (they wrap on a narrow window)
        self.format_flow = FlowLayout(self.format_row, spacing=10)
        root.addWidget(self.format_row)
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Min. games"))
        self.min_games = QSpinBox()
        self.min_games.setRange(1, 20)
        self.min_games.setValue(self._saved_int("decksearch/min_games", 3, 1, 20))
        self.min_games.setToolTip("Leave out runs with fewer games than this")
        self.min_games.valueChanged.connect(self._on_filter_changed)
        opts.addWidget(self.min_games)
        self.share_box = QCheckBox("Card in the deck for most of the run")
        self.share_box.setChecked(True)
        self.share_box.setToolTip(SHARE_TIP)
        self.share_box.toggled.connect(self._on_filter_changed)
        opts.addWidget(self.share_box)
        opts.addWidget(QLabel("Sort"))
        self.sort_combo = ElidedCombo()
        for label, key in SORTS:
            self.sort_combo.addItem(label, key)
        i = self.sort_combo.findData(str(new_settings().value("decksearch/sort", "result") or "result"))
        self.sort_combo.setCurrentIndex(max(i, 0))
        self.sort_combo.activated.connect(self._on_sort_chosen)
        opts.addWidget(self.sort_combo)
        opts.addStretch(1)
        root.addLayout(opts)

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

        self.body = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        self.summary = ElidedLabel("")
        ll.addWidget(self.summary)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree.itemSelectionChanged.connect(self._on_selected)
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        for c in range(len(COLUMNS)):
            hdr.setSectionResizeMode(c, hdr.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, hdr.ResizeMode.Stretch)
        ll.addWidget(self.tree, 1)
        self.more_btn = QPushButton("Show more")
        self.more_btn.clicked.connect(lambda: self.request_search("more"))
        ll.addWidget(self.more_btn)
        self.body.addWidget(left)

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
        self.deck_lay.setContentsMargins(0, 0, 8, 0)
        self.deck_lay.setSpacing(2)
        self.deck_scroll.setWidget(self.deck_box)
        rl.addWidget(self.deck_scroll, 1)
        btns = QHBoxLayout()
        self.copy_btn = QPushButton("Copy deck")
        self.copy_btn.setToolTip("Copy the deck for Arena's Import button")
        self.copy_btn.clicked.connect(self.copy_deck)
        btns.addWidget(self.copy_btn)
        btns.addStretch(1)
        btns.addWidget(QLabel("Card size"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(CARD_W_MIN, CARD_W_MAX)
        self.size_slider.setSingleStep(10)
        self.size_slider.setFixedWidth(130)
        self.size_slider.setValue(self._saved_int("decksearch/card_w", CARD_W_DEFAULT, CARD_W_MIN, CARD_W_MAX))
        self.size_slider.setToolTip("Size of the card images")
        self.size_slider.valueChanged.connect(self._on_card_size)
        btns.addWidget(self.size_slider)
        rl.addLayout(btns)
        self.body.addWidget(right)
        self.body.setStretchFactor(0, 3)
        self.body.setStretchFactor(1, 2)
        self.body.setSizes([620, 420])
        self.body.setChildrenCollapsible(False)
        root.addWidget(self.body, 1)
        self.store = ImageStore.for_api(api)
        self.hover_popup = FloatingCardPreview(self.store, self)

        self._show_empty(INTRO)
        self._update_controls()

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _saved_int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(new_settings().value(key, default))))
        except (TypeError, ValueError):
            return default

    def set_code(self) -> str:
        d = self.set_combo.currentData()
        return str(d) if d else ""

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

    def _update_controls(self) -> None:
        ok = self._usable()
        for w in (self.card_edit, self.search_btn, self.min_games, self.share_box, self.sort_combo, self.format_row):
            w.setEnabled(ok)
        self.more_btn.setEnabled(ok and not self.task.running and self.data is not None and len(self.results) < int(self.data.get("total", 0)))
        self.copy_btn.setEnabled(self._selected() is not None)

    # ------------------------------------------------------------------ sets
    def refresh_sets(self, statuses: list) -> None:
        """The Sets tab's list changed: keep the sets that have a corpus; a rebuilt set drops its names and results."""
        self.statuses = {st.set_code: st for st in statuses if st.has_corpus}
        prev = self.set_code() or str(new_settings().value("decksearch/set", "") or "")
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for code in sorted(self.statuses):
            self.set_combo.addItem(code, code)
        self.set_combo.setCurrentIndex(self.set_combo.findData(prev) if self.set_combo.findData(prev) >= 0 else (0 if self.statuses else -1))
        self.set_combo.blockSignals(False)
        if not self.statuses:
            self._reset_results()
            self._show_empty("No set has data to search yet.  Calculate a set on the Sets tab first.")
        elif self._meta_key is not None and self._meta_key != self._key():
            self._meta_key = None                            # the corpus was rebuilt: names, events and runs are out of date
            self._reset_results()
            self._show_empty(INTRO)
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
        self._reset_results()
        self._say("")
        self._show_empty(INTRO)
        self._update_controls()
        if self.isVisible():
            self._pump()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._pump)

    # ------------------------------------------------------------------ names and formats (loaded with the set)
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
        new_settings().setValue("decksearch/sort", self.sort_combo.currentData())
        self._on_filter_changed()

    def _on_filter_changed(self, *_a) -> None:
        if self._filling:
            return
        new_settings().setValue("decksearch/min_games", self.min_games.value())
        if self.searched is not None:                         # a search is on screen: run it again with the new filters
            self.debounce.start()

    def request_search(self, kind: str = "new") -> None:
        """Ask for a search ("new" = from the first run, "more" = the next page); it starts as soon as the task is free."""
        if kind == "more" and self.debounce.isActive():
            kind = "new"                                      # a filter change is waiting: run that new search, do not append to the old one
        self.debounce.stop()
        if not self._usable():
            return
        if kind == "new" and not self.card_edit.text().strip():
            self._say("Type a card name first.", warn=False)
            return
        if kind == "more" and self.searched is None:
            return
        self._pending = "more" if kind == "more" and self._pending != "new" else "new"
        self._pump()

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
        name = self.card_edit.text().strip()
        if kind == "more":
            name = self.searched[1] if self.searched else name
        args = dict(events=self.selected_events(), sort=str(self.sort_combo.currentData()), min_games=self.min_games.value(),
                    min_card_share=0.5 if self.share_box.isChecked() else 0.0)
        if args["events"] is not None and not args["events"]:
            self._say("Tick at least one format.", warn=False)
            return
        offset = len(self.results) if kind == "more" else 0
        api = self.api
        self._say("")
        self.task.run(lambda progress, cancel: ("search", kind, sc, api.search_decks(sc, name, limit=PAGE, offset=offset, **args)),
                      label=f"deck search {sc} {name}")

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
        _tag, kind, sc, data = res
        if sc != self.set_code() or self._pending is not None:
            self.strip.end("Ready")
            return                                            # a newer request is waiting: this answer is out of date
        self.strip.end("Ready")
        self._show_data(kind, data)

    def _on_error(self, tb: str) -> None:
        if "SetBusy" in tb:
            self.strip.end("Waiting for the calculation")
            self._pending = None
            return
        self.strip.end("Failed")
        if "ValueError" in tb.strip().splitlines()[-1]:
            name = self.card_edit.text().strip()
            self._reset_results()
            self._say(f"No card called \"{name}\" in this set.  Pick a name from the list that opens as you type.")
            self._show_empty(INTRO)
            return
        if "FileNotFoundError" in tb.strip().splitlines()[-1]:
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

    # ------------------------------------------------------------------ results
    def _reset_results(self) -> None:
        self.results, self.data, self.searched = [], None, None
        self.tree.clear()
        self._clear_deck()
        self.summary.setText("")

    def _show_data(self, kind: str, data: dict) -> None:
        card = str(data.get("card") or self.card_edit.text().strip())
        if kind == "more" and self.searched == (data.get("set_code", self.set_code()), card):
            self.results += list(data.get("results") or [])
            start = self.tree.topLevelItemCount()
            self._add_rows(start)
            self.data = data
        else:
            self.results = list(data.get("results") or [])
            self.data = data
            self.searched = (self.set_code(), card)
            self.tree.clear()
            self._add_rows(0)
            self._clear_deck()
        total = int(data.get("total", len(self.results)))
        if not self.results:
            self._show_empty(f"No deck with {card} matches these filters.")
            self._update_controls()
            return
        self._show_body()
        self.summary.setText(f"{count_text(total)} decks with {card} (showing {count_text(len(self.results))}) · sideboards stored for Sealed only")
        if kind != "more" or self._selected() is None:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._update_controls()

    def _add_rows(self, start: int) -> None:
        for i in range(start, len(self.results)):
            r = self.results[i]
            rec = (f"{TROPHY} " if r.get("trophy") else "") + str(r.get("record", ""))
            n, g = int(r.get("games_with_card", 0)), int(r.get("games", 0))
            it = QTreeWidgetItem([rec, str(r.get("event_label", "")), str(r.get("date", "")), str(r.get("colors", "")),
                                  str(r.get("copies", "")), f"{n} / {g}", str(r.get("rank") or "")])
            it.setData(0, Qt.ItemDataRole.UserRole, i)
            if r.get("trophy"):
                it.setToolTip(0, "Reached the event's maximum wins")
                it.setForeground(0, QColor(semantic()["good"]))
            self.tree.addTopLevelItem(it)

    def _selected(self) -> dict | None:
        its = self.tree.selectedItems()
        if not its:
            return None
        i = its[0].data(0, Qt.ItemDataRole.UserRole)
        return self.results[i] if isinstance(i, int) and 0 <= i < len(self.results) else None

    # ------------------------------------------------------------------ the deck list
    def _clear_deck(self) -> None:
        self.hover_popup.hide()
        self.piles = []
        while self.deck_lay.count():
            it = self.deck_lay.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()

    def _on_selected(self) -> None:
        self._update_controls()
        r = self._selected()
        if r is None:
            return
        self._clear_deck()
        lay, sc = self.deck_lay, self.set_code()
        rec = QLabel(("%s " % TROPHY if r.get("trophy") else "") + str(r.get("record", "")))
        rec.setFont(sized_font(self.font(), 4, bold=True))
        lay.addWidget(rec)
        bits = [r.get("event_label"), r.get("date"), r.get("colors"), r.get("rank")]
        lay.addWidget(self._note(" · ".join(str(b) for b in bits if b)))
        lay.addWidget(self._note(f"{r.get('n_cards', 0)} cards, {r.get('n_lands', 0)} lands · {r.get('copies', 0)} × {self.searched[1] if self.searched else ''}"
                                 f", in the deck for {r.get('games_with_card', 0)} of {r.get('games', 0)} games"))
        if int(r.get("n_builds", 1) or 1) > 1:
            bw, bg = int(r.get("build_wins", 0)), int(r.get("build_games", 0))
            lay.addWidget(self._note(f"The deck changed during the run: showing the build with the card, {bw}–{bg - bw} in its {bg} games.", wrap=True))
        lay.addSpacing(6)
        main = list(r.get("main") or [])
        side = r.get("sideboard")
        self.store.prefetch(sc, [str(c.get("name", "")) for c in main + (side if isinstance(side, list) else [])])
        basics = [c for c in main if c.get("is_basic")]
        if basics:
            lay.addWidget(self._note("Basic lands: " + ", ".join(f"{c.get('n', 0)} {c.get('name', '')}" for c in basics), wrap=True))
        self._pile(lay, f"Main deck ({sum(int(c.get('n', 0)) for c in main)})", [c for c in main if not c.get("is_basic")], sc, 1.0)
        if isinstance(side, list):
            if side:
                self._pile(lay, f"Sideboard ({sum(int(c.get('n', 0)) for c in side)})", side, sc, SIDE_SCALE, zone="side")
        elif r.get("sideboard_note"):
            lay.addSpacing(6)
            lay.addWidget(self._note(str(r["sideboard_note"]), wrap=True))
        lay.addStretch(1)

    def _note(self, text: str, wrap: bool = False) -> QLabel:
        lab = muted(QLabel(text))
        lab.setWordWrap(wrap)
        return lab

    def _card_w(self) -> int:
        return int(self.size_slider.value())

    def _pile(self, lay: QVBoxLayout, title: str, cards: list[dict], sc: str, scale: float, zone: str = "deck") -> None:
        """A read-only pile of card images (columns by mana value, creatures apart) under a heading; the searched card wears a chip."""
        head = QLabel(title)
        head.setFont(sized_font(self.font(), 1, bold=True))
        lay.addWidget(head)
        target = (self.searched[1] if self.searched else "").casefold()
        ids: list[int] = []
        info: dict[int, dict] = {}
        chips: dict[int, str] = {}
        for i, c in enumerate(cards):
            cid = c.get("card_id")
            cid = int(cid) if cid is not None else -1 - i
            info[cid] = dict(c, cost="", produces="")
            ids += [cid] * int(c.get("n", 0))
            if str(c.get("name", "")).casefold() == target:
                chips[cid] = "searched"
        pile = ReadOnlyPile(zone, self.store, self.deck_box)
        pile.chips = chips
        pile.tips = {cid: "The card searched for" for cid in chips}
        pile.hovered.connect(lambda cid, p=pile, sc=sc: self._on_pile_hover(p, sc, cid))
        pile.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        pile._scale, pile._content = scale, (sc, ids, info)
        self._fill_pile(pile)
        self.piles.append(pile)
        lay.addWidget(pile)

    def _fill_pile(self, pile: PileView) -> None:
        sc, ids, info = pile._content
        pile.set_content(sc, ids, info, {}, dict(group="cmc", split=True, card_w=max(40, int(self._card_w() * pile._scale)),
                                                 show_values=False, sort="mv", badge="avg"))
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

    # ------------------------------------------------------------------ copy
    def copy_deck(self) -> None:
        r = self._selected()
        if r is None:
            return
        QApplication.clipboard().setText(self.api.deck_search_text(self.set_code(), r))
        self.statusMessage.emit("Deck copied: paste it in Arena's Import")

    def wait(self, ms: int = 8000) -> None:
        self.task.wait(ms)
