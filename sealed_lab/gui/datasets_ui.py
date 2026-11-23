"""'New 17lands data?' panel: a non-modal strip at the top of the main window.

The check runs on a worker thread after the window shows (and on demand from the Data menu); when files are new or changed the panel
lists them per set (checkboxes, sizes, server dates).  "Download selected" streams them into the data folder with progress / cancel,
registers them with the app (their sets become stale: press Calculate) and "Don't ask again for these" remembers the listed versions.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from sealed_lab.ingest.setinfo import order_sets, release_dates

from .widgets import BusyStrip, ElidedLabel, FullTextTipDelegate, muted, new_settings, primary_button, row_height, sized_font
from .workers import TaskController

STATUS_TEXT = {"new": "New", "updated": "Updated", "unregistered": "On disk, not loaded"}


def human_size(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    return f"{max(n, 0) / 1024:.0f} KB"


SEALED_EVENTS = ("Sealed", "TradSealed")
ANNOUNCED_KEY = "datasets/announced"


def _real_set(code: str) -> bool:
    c = (code or "").upper()
    return bool(c) and "CUBE" not in c and "_" not in c


def announcements(items: list, announced: set[str]) -> list[dict]:
    """What deserves a popup right away: a new set (not in the app yet) or the first Sealed data of a set.
    Returns dicts(key, keys, set_code, reason 'new_set'|'sealed', items); `keys` are all the keys this popup covers
    (a new set that already has Sealed data announces both at once).  Keys already in `announced` are skipped."""
    by_set: dict[str, list] = {}
    for it in items:
        if getattr(it, "status", "") == "new" and _real_set(it.set_code) and not it.optional:
            by_set.setdefault(it.set_code, []).append(it)
    out = []
    for code, its in sorted(by_set.items()):
        sealed = [i for i in its if i.kind == "game" and i.event in SEALED_EVENTS]
        k_new, k_sealed = f"{code}:new_set", f"{code}:sealed"
        if not any(i.set_in_app for i in its) and k_new not in announced:
            out.append(dict(key=k_new, keys=[k_new] + ([k_sealed] if sealed else []), set_code=code, reason="new_set", items=its))
        elif sealed and k_sealed not in announced:
            out.append(dict(key=k_sealed, keys=[k_sealed], set_code=code, reason="sealed", items=sealed))
    return out


def announcement_text(a: dict) -> str:
    events = ", ".join(dict.fromkeys(i.event for i in a["items"]))
    total = human_size(sum(i.size or 0 for i in a["items"] if i.needs_download))
    what = (f"17lands now has data for a new set: {a['set_code']}" if a["reason"] == "new_set"
            else f"Sealed data for {a['set_code']} is now on 17lands")
    return f"{what} ({events}, {total} in total). Download it now?"


def ask(parent, title: str, text: str) -> bool:
    """Download / Later box (tests replace this)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    yes = box.addButton("Download", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(yes)
    box.exec()
    return box.clickedButton() is yes


class DatasetPanel(QFrame):
    logMessage = Signal(str)
    errorOccurred = Signal(str, str)
    statusMessage = Signal(str)
    busyChanged = Signal(bool)
    filesRegistered = Signal()          # downloaded / loaded files were registered: refresh the sets
    dataDirChanged = Signal(str)        # the user picked another data folder here (the Sets tab follows it)

    def __init__(self, api: Any, parent=None):
        super().__init__(parent)
        self.api = api
        self.items: list = []
        self.locked_sets: set[str] = set()   # sets being calculated: their files are not downloaded / registered meanwhile
        self.check = None               # last DatasetCheck
        self._manual = False
        self.check_task = TaskController(self)
        self.check_task.result.connect(self._on_check_result)
        self.check_task.error.connect(self._on_check_error)
        self.dl_task = TaskController(self)
        self.dl_task.started.connect(lambda: self.busyChanged.emit(True))
        self.dl_task.ended.connect(lambda: self.busyChanged.emit(False))
        self.dl_task.ended.connect(self._update_buttons)
        self.dl_task.progress.connect(lambda f, m: self.strip.update_progress(f, m))
        self.dl_task.log.connect(self.logMessage)
        self.dl_task.result.connect(self._on_download_result)
        self.dl_task.error.connect(self._on_download_error)
        self.dl_task.cancelled.connect(lambda: self.strip.end("Cancelled"))

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setProperty("card", True)
        self.setObjectName("datasetPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        self.title = QLabel("")
        self.title.setFont(sized_font(self.font(), 1, bold=True))
        lay.addWidget(self.title)
        self.subtitle = muted(QLabel(""))
        self.subtitle.setWordWrap(True)
        lay.addWidget(self.subtitle)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["File", "Status", "Size", "Server date", "Note"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        rh = row_height(self.tree.font(), 20)                       # rows are as high as the font needs; the list scrolls beyond ~9
        self._tree_max = max(210, 9 * rh)
        self.tree.setMaximumHeight(self._tree_max)
        self.tree.setMinimumHeight(max(90, 4 * rh))
        self.tree.setMinimumWidth(300)
        self.tree.setItemDelegate(FullTextTipDelegate(self.tree))
        self.tree.header().setStretchLastSection(True)
        self.tree.itemChanged.connect(lambda *_: self._update_buttons())
        lay.addWidget(self.tree)
        row = QHBoxLayout()
        row.addWidget(QLabel("Data folder:"))
        self.folder_label = ElidedLabel("", Qt.TextElideMode.ElideMiddle)         # a long path is elided in the middle, full path in the tooltip
        row.addWidget(self.folder_label, 1)
        self.folder_btn = QPushButton("Data folder…")
        self.folder_btn.setToolTip("Where 17lands files are downloaded to (one sub-folder per set)")
        self.folder_btn.clicked.connect(self.choose_folder)
        row.addWidget(self.folder_btn)
        lay.addLayout(row)
        btns = QHBoxLayout()
        self.download_btn = primary_button("Download selected")
        self.download_btn.clicked.connect(self.download)
        self.later_btn = QPushButton("Not now")
        self.later_btn.clicked.connect(self.hide_panel)
        self.never_btn = QPushButton("Don't ask again for these")
        self.never_btn.setToolTip("Hide these file versions in future checks (a newer upload is announced again)")
        self.never_btn.clicked.connect(self.dismiss)
        btns.addWidget(self.download_btn)
        btns.addWidget(self.later_btn)
        btns.addWidget(self.never_btn)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.result_label)
        self.strip = BusyStrip()
        self.strip.cancelRequested.connect(self.dl_task.cancel)
        self.strip.setVisible(False)
        lay.addWidget(self.strip)
        self.setVisible(False)
        self._refresh_folder()

    # ------------------------------------------------------------------ state
    @property
    def busy(self) -> bool:
        return self.dl_task.running

    def set_locked_sets(self, sets: set[str]) -> None:
        """Sets being calculated right now: downloading files of those sets waits (other sets are not affected)."""
        self.locked_sets = {s.upper() for s in sets}
        self._update_buttons()

    def show_items(self, items: list) -> None:
        """List files to download (e.g. all the files 17lands has for one set) in the panel."""
        self.populate(items)

    def data_dir(self) -> str:
        v = new_settings().value("datasets/data_dir", "")
        if v:
            return str(v)
        fn = getattr(self.api, "datasets_data_dir", None)
        return str(fn()) if fn is not None else ""

    def _refresh_folder(self) -> None:
        self.folder_label.setText(self.data_dir())

    def choose_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "17lands data folder", self.data_dir())
        if d:
            self.set_data_dir(d)

    def set_data_dir(self, d: str, notify: bool = True) -> None:
        new_settings().setValue("datasets/data_dir", d)
        self._refresh_folder()
        if notify:
            self.dataDirChanged.emit(d)
        self.run_check(force=False, manual=True)

    # ------------------------------------------------------------------ check
    def run_check(self, force: bool = False, manual: bool = False) -> bool:
        fn = getattr(self.api, "check_new_datasets", None)
        if fn is None or self.check_task.running or self.dl_task.running:
            return False
        data_dir = self.data_dir() or None
        self._manual = manual
        if manual:
            self.statusMessage.emit("Checking 17lands for new data…")

        def job(progress, cancel):
            return fn(data_dir=data_dir, force=force)
        self.check_task.run(job, label="check for new 17lands data")
        return True

    def _on_check_error(self, tb: str) -> None:
        self.logMessage.emit("17lands data check failed: " + (tb.strip().splitlines() or ["?"])[-1])
        if getattr(self, "_manual", False):
            self.statusMessage.emit("Could not check for new 17lands data")

    def _on_check_result(self, res: object) -> None:
        self.check = res
        err = getattr(res, "error", "")
        items = list(getattr(res, "items", []) or [])
        manual = getattr(self, "_manual", False)
        if err:
            self.logMessage.emit(f"17lands data check: {err}")
            if manual:
                self.statusMessage.emit(f"Could not check for new 17lands data: {err}")
            return
        self.logMessage.emit(f"17lands data check: {len(items)} new / updated file(s)"
                             + (f", {res.n_dismissed} hidden by you" if getattr(res, "n_dismissed", 0) else "")
                             + f" ({getattr(res, 'elapsed', 0.0):.1f} s)")
        if not items:
            if manual:
                extra = f" ({res.n_dismissed} hidden by \"don't ask again\")" if getattr(res, "n_dismissed", 0) else ""
                self.statusMessage.emit(f"17lands data is up to date{extra}")
            if not self.dl_task.running and not self.result_label.text():
                self.setVisible(False)
            return
        self.populate(items)
        self.announce(items)

    def announce(self, items: list) -> None:
        """Popup for a new set / first Sealed data (once per set and reason, ever); 'Download' ticks only those files and starts."""
        st = new_settings()
        seen = st.value(ANNOUNCED_KEY, [])
        seen = {seen} if isinstance(seen, str) else set(seen or [])
        todo = announcements(items, seen)
        if not todo:
            return
        for a in todo:
            seen.update(a["keys"])
        st.setValue(ANNOUNCED_KEY, sorted(seen))
        st.sync()
        chosen: list = []
        for a in todo:
            if ask(self.window(), "New 17lands data", announcement_text(a)):
                chosen.extend(a["items"])
        if chosen:
            urls = {i.url for i in chosen}
            self.tree.blockSignals(True)
            for i in range(self.tree.topLevelItemCount()):
                top = self.tree.topLevelItem(i)
                for j in range(top.childCount()):
                    ch = top.child(j)
                    ch.setCheckState(0, Qt.CheckState.Checked if ch.data(0, Qt.ItemDataRole.UserRole).url in urls
                                     else Qt.CheckState.Unchecked)
            self.tree.blockSignals(False)
            self._update_buttons()
            self.download()

    # ------------------------------------------------------------------ list
    def populate(self, items: list) -> None:
        self.items = list(items)
        self.tree.blockSignals(True)
        self.tree.clear()
        by_set: dict[str, list] = {}
        for it in items:
            by_set.setdefault(it.set_code, []).append(it)
        for code in order_sets(by_set, release_dates(by_set, fetch=False)):
            its = by_set[code]
            model = any(i.set_has_model for i in its)
            tag = " - has a model in the app (will need Calculate)" if model else (" - in the app" if any(i.set_in_app for i in its) else "")
            top = QTreeWidgetItem([f"{code}{tag}", "", "", "", ""])
            top.setFlags(top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
            top.setFont(0, sized_font(self.font(), 0, bold=True))
            self.tree.addTopLevelItem(top)
            for it in its:
                size = human_size(it.size) if it.needs_download else f"{human_size(it.local_size)} (local)"
                child = QTreeWidgetItem([it.name, STATUS_TEXT.get(it.status, it.status), size, (it.last_modified or "")[:10],
                                         it.reason])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked if (it.optional and it.status == "new") else Qt.CheckState.Checked)
                child.setData(0, Qt.ItemDataRole.UserRole, it)
                child.setToolTip(4, it.reason)
                top.addChild(child)
            top.setExpanded(True)
        for c in range(4):
            self.tree.resizeColumnToContents(c)
        self.tree.blockSignals(False)
        self._fit_tree()
        n_new = sum(1 for i in items if i.status == "new")
        n_upd = sum(1 for i in items if i.status == "updated")
        n_reg = sum(1 for i in items if i.status == "unregistered")
        bits = [f"{n} {w}" for n, w in ((n_new, "new"), (n_upd, "updated"), (n_reg, "on disk but not loaded")) if n]
        self.title.setText("New 17lands data: " + ", ".join(bits))
        self.subtitle.setText("Tick what you want. Downloaded files are registered with the app; the sets they belong to then show "
                              "as stale - press Calculate on the Sets tab to refit them.")
        self.result_label.setText("")
        self.tree.setVisible(True)
        self.never_btn.setVisible(True)
        self.later_btn.setText("Not now")
        self.setVisible(True)
        self._update_buttons()

    def _fit_tree(self) -> None:
        """The list is as tall as its rows (up to its maximum, then it scrolls): no empty band under a short list."""
        rows = sum(1 + self.tree.topLevelItem(i).childCount() for i in range(self.tree.topLevelItemCount()))
        rh = max(self.tree.sizeHintForRow(0), 1)
        want = self.tree.header().sizeHint().height() + rows * rh + 2 * self.tree.frameWidth() + 4
        self.tree.setFixedHeight(max(2 * rh + self.tree.header().sizeHint().height(), min(want, self._tree_max)))

    def selected(self) -> list:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                ch = top.child(j)
                if ch.checkState(0) == Qt.CheckState.Checked:
                    out.append(ch.data(0, Qt.ItemDataRole.UserRole))
        return out

    def _update_buttons(self) -> None:
        sel = self.selected()
        n_dl = sum(1 for i in sel if i.needs_download)
        mb = sum(i.size or 0 for i in sel if i.needs_download)
        if n_dl:
            txt = f"Download selected ({n_dl} file{'s' if n_dl != 1 else ''}, {human_size(mb)})"
        elif sel:
            txt = f"Load selected ({len(sel)} file{'s' if len(sel) != 1 else ''})"
        else:
            txt = "Download selected"
        self.download_btn.setText(txt)
        blocked = sorted({i.set_code.upper() for i in sel} & self.locked_sets)
        free = not self.dl_task.running and not blocked
        self.download_btn.setEnabled(free and bool(sel))
        self.download_btn.setToolTip(f"{', '.join(blocked)} is being recalculated: its files can be downloaded when it finishes" if blocked else "")
        self.never_btn.setEnabled(not self.dl_task.running and bool(self.items))
        self.later_btn.setEnabled(not self.dl_task.running)
        self.folder_btn.setEnabled(not self.dl_task.running)
        self.tree.setEnabled(not self.dl_task.running)

    def hide_panel(self) -> None:
        self.setVisible(False)
        self.result_label.setText("")

    def dismiss(self) -> None:
        fn = getattr(self.api, "dismiss_datasets", None)
        if fn is not None and self.items:
            try:
                fn(self.items)
                self.logMessage.emit(f"17lands data: {len(self.items)} file version(s) will not be announced again")
            except Exception as e:  # noqa: BLE001 - cosmetic
                self.logMessage.emit(f"could not remember the choice: {e}")
        self.items = []
        self.hide_panel()

    # ------------------------------------------------------------------ download
    def download(self) -> None:
        fn = getattr(self.api, "download_datasets", None)
        sel = self.selected()
        if fn is None or not sel or self.dl_task.running:
            return
        data_dir = self.data_dir()
        self.logMessage.emit(f"=== Download 17lands data into {data_dir} ===")
        self.strip.setVisible(True)
        self.strip.begin("Starting download…")
        self.result_label.setText("")

        def job(progress, cancel):
            return fn(sel, data_dir, progress=progress, cancel=cancel)
        self.dl_task.run(job, label="download 17lands data")
        self._update_buttons()

    def _on_download_error(self, tb: str) -> None:
        self.strip.end("Failed")
        self.strip.setVisible(False)
        self.errorOccurred.emit("Download failed: " + (tb.strip().splitlines() or ["?"])[-1], tb)

    def _on_download_result(self, rep: object) -> None:
        self.strip.end("Done" if not getattr(rep, "cancelled", False) else "Cancelled")
        self.strip.setVisible(False)
        msg = getattr(rep, "message", "Done")
        for name, err in getattr(rep, "failed", []):
            msg += f"\nFailed: {name}: {err}"
            self.logMessage.emit(f"Download failed: {name}: {err}")
        self.logMessage.emit(msg.splitlines()[0])
        # drop what was done from the list, keep failures / the rest
        done_names = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in getattr(rep, "downloaded", [])}
        reg_only = {i.url for i in self.items if i.status == "unregistered" and i.local_path
                    and any(f.path == i.local_path for f in getattr(rep, "registered", []))}
        self.items = [i for i in self.items if i.name not in done_names and i.url not in reg_only]
        if self.items:
            self.populate(self.items)
        else:
            self.tree.clear()
            self.tree.setVisible(False)
            self.never_btn.setVisible(False)
            self.later_btn.setText("Close")
            self.title.setText("17lands data")
            self.subtitle.setText("")
        self.result_label.setText(msg)
        self.statusMessage.emit(msg.splitlines()[0])
        if getattr(rep, "registered", None):
            self.filesRegistered.emit()
        self._update_buttons()

    def wait(self, ms: int = 8000) -> None:
        self.check_task.wait(ms)
        self.dl_task.wait(ms)
