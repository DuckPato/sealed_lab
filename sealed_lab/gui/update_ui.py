"""Help menu and the update flow: check (manual or once a day), prompt, download, stage, hand over to apply_update.ps1 (Windows) / apply_update.sh (Linux).

`UpdateController` takes the updater module as an argument (default `sealed_lab.updater`) so GUI tests can drive the whole flow with a
fake and no network.  All network and disk work runs in a TaskController worker; the dialogs are non-modal (`open()`), like the rest of
the app, so nothing here blocks the UI thread or a test.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressDialog, QPushButton, QVBoxLayout, QWidget)

from sealed_lab import __version__
from sealed_lab import updater as default_updater
from sealed_lab.attribution import CREDIT_TEXT, WOTC_DISCLAIMER
from sealed_lab.config import PROJECT_URL

from .widgets import new_settings, show_message
from .workers import TaskController

AUTO_CHECK_DELAY_MS = 5000
KEY_AUTO, KEY_TOKEN, KEY_LAST, KEY_SKIPPED = "updates/auto_check", "updates/token", "updates/last_check", "updates/skipped"


def _as_bool(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() not in ("false", "0", "no", "")


class UpdatePrompt(QDialog):
    """"Version X is available": the release notes (plain text, scrollable) and Update now / Later / Skip this version."""

    def __init__(self, info: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("sealed_lab update")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(560, 420)
        lay = QVBoxLayout(self)
        self.heading = QLabel(f"<b>sealed_lab {info.latest}</b> is available (you have {info.current}).")
        lay.addWidget(self.heading)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setPlainText(info.notes.strip() or "(no release notes)")
        lay.addWidget(self.notes, 1)
        row = QDialogButtonBox()
        self.update_btn = row.addButton("Update now", QDialogButtonBox.ButtonRole.AcceptRole)
        self.later_btn = row.addButton("Later", QDialogButtonBox.ButtonRole.RejectRole)
        self.skip_btn = row.addButton("Skip this version", QDialogButtonBox.ButtonRole.DestructiveRole)
        lay.addWidget(row)
        self.update_btn.setDefault(True)
        self.later_btn.clicked.connect(self.close)


class UpdateController(QObject):
    """Owns the Help menu actions and the check / download / apply flow of one main window."""

    statusMessage = Signal(str)
    logMessage = Signal(str)

    def __init__(self, window: QWidget, updater: Any = None, quit_fn: Callable[[], None] | None = None,
                 opener: Callable[[str], Any] | None = None):
        super().__init__(window)
        self.window = window
        self.updater = updater or default_updater
        self.quit_fn = quit_fn or self._quit
        self.opener = opener or (lambda url: QDesktopServices.openUrl(QUrl(url)))
        self.info: Any = None
        self.prompt: UpdatePrompt | None = None
        self.progress_dialog: QProgressDialog | None = None
        self.confirm_box: QMessageBox | None = None
        self.last_box: QMessageBox | None = None
        self.launched: list = []                          # (staged, app_dir, pid) of each handover (tests read it)
        self._manual = False
        self.check_task = TaskController(self)
        self.check_task.result.connect(self._on_check_result)
        self.check_task.error.connect(self._on_check_error)
        self.download_task = TaskController(self)
        self.download_task.progress.connect(self._on_progress)
        self.download_task.result.connect(self._on_downloaded)
        self.download_task.error.connect(self._on_download_error)
        self.download_task.cancelled.connect(self._close_progress)
        self.statusMessage.connect(lambda m: window.statusBar().showMessage(m, 12000))

    # ------------------------------------------------------------------ menu
    def install_menu(self, menubar) -> None:
        menu = menubar.addMenu("&Help")
        self.check_action = menu.addAction("Check for updates…")
        self.check_action.triggered.connect(lambda: self.check(manual=True))
        self.auto_action = QAction("Check for updates automatically", menu)
        self.auto_action.setCheckable(True)
        self.auto_action.setChecked(self.auto_enabled())
        self.auto_action.toggled.connect(lambda on: new_settings().setValue(KEY_AUTO, bool(on)))
        menu.addAction(self.auto_action)
        self.token_action = menu.addAction("GitHub sign-in token…")
        self.token_action.triggered.connect(self.edit_token)
        menu.addSeparator()
        self.about_action = menu.addAction("About sealed_lab")
        self.about_action.triggered.connect(self.show_about)

    def auto_enabled(self) -> bool:
        return _as_bool(new_settings().value(KEY_AUTO), True)

    def show_about(self) -> None:
        self.last_box = show_message(self.window, "About sealed_lab", f"sealed_lab {__version__}\n{PROJECT_URL}\n\n{CREDIT_TEXT}\n\n{WOTC_DISCLAIMER}",
                                     icon=QMessageBox.Icon.Information)

    def edit_token(self) -> None:
        """A GitHub token, only needed if the repo is made private (or past GitHub's anonymous rate limit); kept in the app's settings
        (plain text: `gh auth login` is the safer way)."""
        s = new_settings()
        text, ok = QInputDialog.getText(
            self.window, "GitHub sign-in token",
            "Normally nothing is needed here: sealed_lab's releases are public.  A token only helps if the repository is made private.\n"
            "The easiest way is the GitHub CLI: run `gh auth login` once and sealed_lab uses that sign-in.\n\n"
            "Otherwise paste a personal access token that can read the repository's contents.  It is stored in this app's settings on this\n"
            "computer.  Leave empty to remove a stored token.", QLineEdit.EchoMode.Password, str(s.value(KEY_TOKEN, "") or ""))
        if not ok:
            return
        text = text.strip()
        if text:
            s.setValue(KEY_TOKEN, text)
        else:
            s.remove(KEY_TOKEN)

    # ------------------------------------------------------------------ check
    def maybe_auto_check(self) -> bool:
        """The once-a-day automatic check (the setting, the day, and an env switch for tests / offline machines)."""
        if os.environ.get("SEALED_LAB_NO_UPDATE_CHECK") or not self.auto_enabled():
            return False
        if not self.updater.due(new_settings().value(KEY_LAST), datetime.now()):
            return False
        return self.check(manual=False)

    def check(self, manual: bool = True) -> bool:
        if self.check_task.running or self.download_task.running:
            return False
        self._manual = manual
        if manual:
            self.statusMessage.emit("Checking for a new version…")
        stored = str(new_settings().value(KEY_TOKEN, "") or "")
        up = self.updater

        def job(progress, cancel):
            return up.check(token=up.find_token(stored))
        return self.check_task.run(job, label="check for updates")

    def _on_check_error(self, tb: str) -> None:
        msg = (tb.strip().splitlines() or ["?"])[-1]
        self.logMessage.emit("Update check failed: " + msg)
        if self._manual:
            self.last_box = show_message(self.window, "sealed_lab update", f"The update check failed: {msg}")

    def _on_check_result(self, info: Any) -> None:
        self.info = info
        if info.error:
            self.logMessage.emit("Update check: " + info.error)
            if self._manual:
                self.statusMessage.emit("Could not check for updates")
                self.last_box = show_message(self.window, "sealed_lab update", info.error, icon=QMessageBox.Icon.Warning)
            return
        new_settings().setValue(KEY_LAST, datetime.now().isoformat(timespec="seconds"))
        if not info.available:
            self.logMessage.emit(f"Update check: {info.current} is the latest version")
            if self._manual:
                self.statusMessage.emit(f"You have the latest version ({info.current})")
                self.last_box = show_message(self.window, "sealed_lab update", f"You have the latest version ({info.current}).",
                                             icon=QMessageBox.Icon.Information)
            return
        self.logMessage.emit(f"Update check: version {info.latest} is available")
        if not self._manual and str(new_settings().value(KEY_SKIPPED, "") or "") == info.latest:
            return
        self.show_prompt(info)

    # ------------------------------------------------------------------ prompt
    def show_prompt(self, info: Any) -> None:
        if self.prompt is not None:
            self.prompt.close()
        dlg = UpdatePrompt(info, self.window)
        dlg.update_btn.clicked.connect(lambda: (dlg.close(), self.update_now()))
        dlg.skip_btn.clicked.connect(lambda: (self.skip(info.latest), dlg.close()))
        self.prompt = dlg
        dlg.show()

    def skip(self, version: str) -> None:
        new_settings().setValue(KEY_SKIPPED, version)
        self.statusMessage.emit(f"Version {version} will not be offered again")

    # ------------------------------------------------------------------ update
    def update_now(self) -> None:
        info = self.info
        if info is None or self.download_task.running:
            return
        up = self.updater
        ok, why = up.can_self_update(up.app_dir())
        if not ok:
            self._show_manual_update(info, why)
            return
        dlg = QProgressDialog(f"Downloading sealed_lab {info.latest}…", "Cancel", 0, 100, self.window)
        dlg.setWindowTitle("sealed_lab update")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        dlg.canceled.connect(self.download_task.cancel)
        self.progress_dialog = dlg
        stored = str(new_settings().value(KEY_TOKEN, "") or "")

        def job(progress, cancel):
            token = up.find_token(stored)
            zip_path = up.download(info, up.staging_root() / "download", token=token, progress=progress, cancel=cancel)
            return up.stage(zip_path, info.latest)
        self.download_task.run(job, label="download update")
        dlg.show()

    def _show_manual_update(self, info: Any, why: str) -> None:
        box = QMessageBox(QMessageBox.Icon.Information, "sealed_lab update",
                          f"This copy cannot update itself.\n\n{why}\n\nVersion {info.latest} is on the release page.",
                          QMessageBox.StandardButton.Close, self.window)
        btn = box.addButton("Open release page", QMessageBox.ButtonRole.ActionRole)
        btn.clicked.connect(lambda: self.opener(info.html_url))
        box.setWindowModality(Qt.WindowModality.WindowModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.last_box = box
        self.open_button = btn
        box.open()

    def _on_progress(self, frac: float, msg: str) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.setValue(int(frac * 100))
            if msg:
                self.progress_dialog.setLabelText(msg)

    def _close_progress(self) -> None:
        dlg, self.progress_dialog = self.progress_dialog, None
        if dlg is not None:
            try:
                dlg.canceled.disconnect()
            except (RuntimeError, TypeError):
                pass
            dlg.close()
            dlg.deleteLater()

    def _on_download_error(self, tb: str) -> None:
        self._close_progress()
        msg = (tb.strip().splitlines() or ["?"])[-1]
        self.logMessage.emit("Update download failed: " + msg)
        self.last_box = show_message(self.window, "sealed_lab update", f"The update could not be downloaded:\n{msg}")

    def _on_downloaded(self, staged: Any) -> None:
        self._close_progress()
        box = QMessageBox(QMessageBox.Icon.Question, "sealed_lab update",
                          f"sealed_lab {self.info.latest} is ready.\n\nsealed_lab will close, update and reopen.",
                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self.window)
        box.button(QMessageBox.StandardButton.Yes).setText("Update and restart")
        box.button(QMessageBox.StandardButton.No).setText("Not now")
        box.setWindowModality(Qt.WindowModality.WindowModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.button(QMessageBox.StandardButton.Yes).clicked.connect(lambda: self.apply(staged))
        self.confirm_box = box
        box.open()

    def apply(self, staged: Any) -> None:
        up = self.updater
        try:
            up.launch_apply(staged, up.app_dir(), os.getpid())
        except Exception as exc:  # noqa: BLE001
            self.logMessage.emit(f"Update could not be started: {exc}")
            self.last_box = show_message(self.window, "sealed_lab update", f"The update could not be started: {exc}")
            return
        self.launched.append((staged, up.app_dir(), os.getpid()))
        self.quit_fn()

    def _quit(self) -> None:
        self.window.close()
        QTimer.singleShot(0, QApplication.quit)

    def wait(self, ms: int = 8000) -> None:
        self.check_task.wait(ms)
        self.download_task.wait(ms)
