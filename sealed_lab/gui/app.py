"""sealed_lab GUI entry point."""
from __future__ import annotations

import os
import sys
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from sealed_lab.attribution import CREDIT_TEXT, credit_html

from .datasets_ui import DatasetPanel
from .deck_search_tab import DeckSearchTab
from .pool_tab import PoolTab
from .report_tab import ReportTab
from .sets_tab import SetsTab
from .update_ui import AUTO_CHECK_DELAY_MS, UpdateController
from .widgets import ElidedStatusBar, apply_theme, new_settings, show_message


def load_backend() -> Any:
    """Fake backend when SEALED_LAB_FAKE_API=1, else the real sealed_lab.api (imported lazily)."""
    if os.environ.get("SEALED_LAB_FAKE_API", "").strip().lower() in ("1", "true", "yes"):
        from . import fake_api as api
    else:
        from sealed_lab import api  # type: ignore[no-redef]
    return api


RECHECK_MS = 6 * 60 * 60 * 1000      # periodic 17lands re-check while the app is open


class _CreditLabel(QLabel):
    """The 17Lands credit beside the tabs: as tall as the tab bar (whatever the style sheet makes it), so its text is centred on the tab
    labels instead of sitting at the bottom edge of the corner."""

    def __init__(self, text: str, bar: QWidget):
        super().__init__(text)
        self._bar = bar

    def sizeHint(self) -> QSize:
        s = super().sizeHint()
        return QSize(s.width(), max(s.height(), self._bar.sizeHint().height()))

    def minimumSizeHint(self) -> QSize:
        s = super().minimumSizeHint()
        return QSize(s.width(), max(s.height(), self._bar.sizeHint().height()))


class MainWindow(QMainWindow):
    def __init__(self, api: Any | None = None, auto_refresh: bool = True, auto_check: bool | None = None,
                 check_delay_ms: int = 800, updater: Any | None = None):
        """`auto_check`: look for new 17lands data in the background after the window shows (default: with `auto_refresh`,
        unless env SEALED_LAB_NO_DATASET_CHECK is set).  `updater`: the module behind Help > Check for updates (tests pass a fake).
        The once-a-day update check runs only with `auto_refresh` (and not with env SEALED_LAB_NO_UPDATE_CHECK)."""
        super().__init__()
        self.api = api if api is not None else load_backend()
        self.last_error: tuple[str, str] | None = None
        self._dialogs: list = []
        self.setWindowTitle("sealed_lab")
        self.resize(1400, 900)
        self._restore_geometry()
        self.setAcceptDrops(True)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setObjectName("mainTabs")
        self.tabs.tabBar().setDrawBase(False)                # its base line stopped where the 17Lands credit (corner widget) begins                  # the style sheet makes these tabs larger than the nested ones
        self.sets_tab = SetsTab(self.api)
        self.pool_tab = PoolTab(self.api)
        self.tabs.addTab(self.sets_tab, "Sets")
        self.report_tab = ReportTab(self.api)
        self.tabs.addTab(self.pool_tab, "Deck builder")
        self.tabs.addTab(self.report_tab, "Report")          # the written summary first, then the analysis
        self.deck_search_tab = DeckSearchTab(self.api)
        self.tabs.addTab(self.deck_search_tab, "Deck search")      # real decks that played a card
        self.credit = _CreditLabel(credit_html(short=True), self.tabs.tabBar())        # 17Lands credit (CC BY 4.0 + their usage guidelines): always visible
        self.credit.setObjectName("dataCredit")
        self.credit.setTextFormat(Qt.TextFormat.RichText)
        self.credit.setOpenExternalLinks(True)
        self.credit.setToolTip(CREDIT_TEXT)
        self.credit.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.credit.setContentsMargins(8, 0, 10, 3)           # (the selected tab's 3 px underline is not part of its text's box)
        self.tabs.setCornerWidget(self.credit, Qt.Corner.TopRightCorner)
        self.datasets = DatasetPanel(self.api)
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.datasets)
        cl.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        menu = self.menuBar().addMenu("&Data")
        self.check_action = menu.addAction("Check for new 17lands data")
        self.check_action.triggered.connect(lambda: self.check_datasets(manual=True))
        self.folder_action = menu.addAction("17lands data folder…")
        self.folder_action.triggered.connect(self.sets_tab.choose_folder)

        self.updates = UpdateController(self, updater)
        self.updates.install_menu(self.menuBar())

        self.setStatusBar(ElidedStatusBar())                 # a long message is elided, not cut off
        self._status = QLabel("")
        self.statusBar().addPermanentWidget(self._status)
        backend = "fake backend" if getattr(self.api, "IS_FAKE", False) else ""
        self._status.setText(backend)

        for tab in (self.sets_tab, self.pool_tab, self.report_tab, self.deck_search_tab):
            tab.errorOccurred.connect(self.show_error)
        self.pool_tab.logMessage.connect(self.sets_tab.logMessage)
        self.report_tab.logMessage.connect(self.sets_tab.logMessage)
        self.deck_search_tab.logMessage.connect(self.sets_tab.logMessage)
        self.deck_search_tab.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 12000))
        self.report_tab.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 12000))
        self.datasets.logMessage.connect(self.sets_tab.logMessage)
        self.updates.logMessage.connect(self.sets_tab.logMessage)
        self.datasets.errorOccurred.connect(self.show_error)
        self.datasets.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 12000))
        self.sets_tab.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 12000))
        self.datasets.filesRegistered.connect(self.sets_tab.refresh)
        self.sets_tab.setsChanged.connect(self.pool_tab.refresh_sets)
        self.sets_tab.setsChanged.connect(self.report_tab.refresh_sets)
        self.sets_tab.setsChanged.connect(self.deck_search_tab.refresh_sets)
        self.pool_tab.archetypeRequested.connect(self.show_archetype)
        self.pool_tab.poolChanged.connect(self.report_tab.set_pool)            # the Report tab shows what applies to the pool (a section and marks)
        self.pool_tab.bestDeckChanged.connect(self.report_tab.set_best_deck)
        # per-set locking: a calculation blocks only its own set elsewhere (Pool tab builds, dataset downloads); a build of a
        # set holds back that set's calculation in the queue
        self.sets_tab.calcSetChanged.connect(self._on_calc_set)
        self.pool_tab.busySetChanged.connect(self.sets_tab.set_pool_busy)
        # one data folder, shared by the Sets tab and the "new data" panel
        self.sets_tab.dataDirChanged.connect(lambda d: self.datasets.set_data_dir(d, notify=False))
        self.datasets.dataDirChanged.connect(lambda d: self.sets_tab.set_data_dir(d, notify=False))
        self.sets_tab.downloadRequested.connect(self._show_downloads)

        s = new_settings()
        try:
            self.tabs.setCurrentIndex(min(int(s.value("tab", 0)), self.tabs.count() - 1))      # (an older version had one more tab)
        except (TypeError, ValueError):
            pass
        self.tabs.currentChanged.connect(lambda i: new_settings().setValue("tab", i))

        if auto_refresh:
            QTimer.singleShot(0, self.sets_tab.start)
            app = QApplication.instance()
            if app is not None:
                app.applicationStateChanged.connect(self._on_app_state)
        if auto_refresh and not os.environ.get("SEALED_LAB_NO_UPDATE_CHECK"):
            QTimer.singleShot(AUTO_CHECK_DELAY_MS, self.updates.maybe_auto_check)          # once a day, in the background, never a popup
        if auto_check is None:
            auto_check = auto_refresh and not os.environ.get("SEALED_LAB_NO_DATASET_CHECK")
        if auto_check:
            QTimer.singleShot(check_delay_ms, lambda: self.check_datasets(manual=False))
            self._recheck_timer = QTimer(self)                  # keep looking while the app stays open
            self._recheck_timer.setInterval(RECHECK_MS)
            self._recheck_timer.timeout.connect(lambda: self.check_datasets(manual=False))
            self._recheck_timer.start()

    # ------------------------------------------------------------------ new 17lands data
    def check_datasets(self, manual: bool = True) -> None:
        """Look for new / updated 17lands files (background; the panel at the top appears only if there is something)."""
        self.datasets.run_check(force=manual, manual=manual)

    # ------------------------------------------------------------------ locking, data folder
    def _on_calc_set(self, code: str) -> None:
        """The set being calculated changed: only that set is locked in the other places."""
        sets = {code} if code else set()
        self.pool_tab.set_calculating(sets)
        self.report_tab.set_calculating(sets)
        self.deck_search_tab.set_calculating(sets)
        self.datasets.set_locked_sets(sets)

    def show_archetype(self, code: str, fmt: str, pair: str) -> None:
        """The Deck builder's "See in Report" link: Report tab, that set and format, that colour pair's block."""
        self.tabs.setCurrentWidget(self.report_tab)
        self.report_tab.show_pair(code, fmt, pair)

    def _show_downloads(self, items: list) -> None:
        self.datasets.show_items(items)

    def _on_app_state(self, state) -> None:
        if state == Qt.ApplicationState.ApplicationActive:
            self.sets_tab.on_app_activated()

    # ------------------------------------------------------------------ errors
    def show_error(self, short: str, details: str = "") -> None:
        self.last_error = (short, details)
        if details:
            self.sets_tab.logMessage.emit(details)
        else:
            self.sets_tab.logMessage.emit(f"ERROR {short}")
        box = show_message(self, "sealed_lab", short, details="")
        self._dialogs = [d for d in self._dialogs if d is not None][-4:] + [box]

    # ------------------------------------------------------------------ drag & drop anywhere
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.tabs.setCurrentWidget(self.sets_tab)
            self.sets_tab.add_paths(paths)

    # ------------------------------------------------------------------ window size and position
    def _restore_geometry(self) -> None:
        """The size, position (and maximised state) of the last session; the default size when there is none or it is unusable."""
        geo = new_settings().value("window/geometry")
        if geo is None:
            return
        try:
            ok = self.restoreGeometry(geo)
        except TypeError:
            ok = False
        if not ok:
            self.resize(1400, 900)

    def save_geometry(self) -> None:
        new_settings().setValue("window/geometry", self.saveGeometry())

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_geometry()
        self.sets_tab.wait(8000)
        self.pool_tab.wait(8000)
        self.report_tab.wait(8000)
        self.deck_search_tab.wait(8000)
        self.datasets.wait(8000)
        self.updates.wait(8000)
        event.accept()


def _set_windows_app_id() -> None:
    """Own taskbar identity on Windows, so the taskbar shows the sealed_lab icon instead of Python's."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("sealed_lab.app")
    except Exception:  # noqa: BLE001 - cosmetic
        pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    _set_windows_app_id()
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("sealed_lab")
    app.setOrganizationName("sealed_lab")
    if sys.platform.startswith("linux"):
        app.setDesktopFileName("sealed_lab")             # matches sealed_lab.desktop, so the dock shows the icon
    from .icon import app_icon

    app.setWindowIcon(app_icon())
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":  # python -m sealed_lab.gui.app
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
