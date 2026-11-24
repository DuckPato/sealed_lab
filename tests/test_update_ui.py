"""Help menu and update flow of the main window, driven by a fake updater: no network, no gh, no real launch, no real quit."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["SEALED_LAB_FAKE_API"] = "1"
os.environ.setdefault("SEALED_LAB_SETTINGS", os.path.join(tempfile.mkdtemp(prefix="sealed_lab_upd_cfg_"), "settings.ini"))

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from sealed_lab import __version__, updater as real_updater  # noqa: E402
from sealed_lab.gui import fake_api  # noqa: E402
from sealed_lab.gui.app import MainWindow  # noqa: E402
from sealed_lab.gui.update_ui import KEY_AUTO, KEY_LAST, KEY_SKIPPED, UpdatePrompt  # noqa: E402
from sealed_lab.gui.widgets import apply_theme, new_settings  # noqa: E402


class FakeUpdater:
    """The part of sealed_lab.updater the controller uses, with canned answers and a call log."""

    UpdateInfo = real_updater.UpdateInfo

    def __init__(self, tmp: Path, latest="v9.9.9", error="", self_update=(True, ""), signed=True):
        self.tmp = tmp
        self.info = real_updater.UpdateInfo(available=bool(latest) and not error, current=__version__, latest=latest, notes="Notes line 1\nline 2",
                                            html_url="https://example.test/release", asset_name="sealed_lab-9.9.9.zip",
                                            asset_url="https://example.test/asset", error=error,
                                            manifest_url="https://example.test/manifest" if signed else "")
        self.self_update = self_update
        self.calls: list[str] = []
        self.launched: list[tuple] = []
        self.is_due = True

    def due(self, last, now, hours=24):
        return self.is_due

    def check(self, **kw):
        self.calls.append("check")
        return self.info

    def app_dir(self):
        return self.tmp / "app"

    def can_self_update(self, folder=None):
        return self.self_update

    def staging_root(self):
        return self.tmp / "update"

    def release_page(self, info, repo=None):
        return info.html_url

    def download(self, info, dest, progress=None, cancel=None):
        self.calls.append("download")
        if progress:
            progress(0.5, "half")
            progress(1.0, "done")
        return self.tmp / "x.zip", {"sha256": "ab" * 32}

    def stage(self, zip_path, version, sha256=None):
        self.calls.append(f"stage:{sha256}")
        return self.tmp / "staged" / "sealed_lab-9.9.9"

    def launch_apply(self, staged, folder, pid):
        self.launched.append((staged, folder, pid))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_theme(app, "light")
    return app


@pytest.fixture()
def make(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    new_settings().clear()
    wins = []

    def build(fake=None, **kw):
        fake = fake or FakeUpdater(tmp_path)
        w = MainWindow(fake_api, auto_refresh=False, updater=fake, **kw)
        quits = []
        w.updates.quit_fn = lambda: quits.append(1)
        opened = []
        w.updates.opener = opened.append
        w.show()
        wins.append(w)
        return w, fake, quits, opened
    yield build
    for w in wins:
        for d in (w.updates.prompt, w.updates.last_box, w.updates.confirm_box):
            if d is not None:
                try:
                    d.close()
                except RuntimeError:
                    pass
        w.close()
    QApplication.processEvents()
    new_settings().clear()


def wait_until(pred, timeout=10.0, what="condition"):
    import time

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        QApplication.processEvents()
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError(f"timeout waiting for {what}")


def box_text(box) -> str:
    return box.text() if box is not None else ""


def test_help_menu_entries(make):
    w, *_ = make()
    menus = {a.text(): a.menu() for a in w.menuBar().actions() if a.menu()}
    assert "&Help" in menus
    texts = [a.text() for a in menus["&Help"].actions() if a.text()]
    assert texts == ["Check for updates…", "Check for updates automatically", "About sealed_lab"]
    assert w.updates.auto_action.isCheckable() and w.updates.auto_action.isChecked()          # default on
    w.updates.auto_action.setChecked(False)
    assert new_settings().value(KEY_AUTO) in (False, "false")
    w.updates.show_about()
    assert __version__ in box_text(w.updates.last_box) and "17Lands" in box_text(w.updates.last_box)


def test_manual_check_up_to_date(make):
    w, fake, *_ = make(FakeUpdater(Path(tempfile.gettempdir()), latest="v0.0.1"))
    fake.info.available = False
    w.updates.check_action.trigger()
    wait_until(lambda: w.updates.last_box is not None, what="up-to-date box")
    assert "latest version" in box_text(w.updates.last_box) and __version__ in box_text(w.updates.last_box)
    assert w.updates.prompt is None and new_settings().value(KEY_LAST)


def test_manual_check_error_is_shown(make):
    w, fake, *_ = make(FakeUpdater(Path(tempfile.gettempdir()), error="GitHub says the repository was not found."))
    w.updates.check(manual=True)
    wait_until(lambda: w.updates.last_box is not None, what="error box")
    assert "not found" in box_text(w.updates.last_box)


def test_newer_version_opens_prompt_with_notes_and_buttons(make):
    w, fake, *_ = make()
    w.updates.check(manual=True)
    wait_until(lambda: w.updates.prompt is not None, what="prompt")
    p = w.updates.prompt
    assert isinstance(p, UpdatePrompt) and p.isVisible() and not p.isModal()
    assert "v9.9.9" in p.heading.text() and "Notes line 1" in p.notes.toPlainText()
    assert [p.update_btn.text(), p.later_btn.text(), p.skip_btn.text()] == ["Update now", "Later", "Skip this version"]
    p.later_btn.click()
    QApplication.processEvents()
    assert not new_settings().value(KEY_SKIPPED)


def test_skip_this_version_suppresses_the_automatic_prompt_only(make, monkeypatch):
    w, fake, *_ = make()
    w.updates.check(manual=True)
    wait_until(lambda: w.updates.prompt is not None)
    w.updates.prompt.skip_btn.click()
    assert new_settings().value(KEY_SKIPPED) == "v9.9.9"
    w.updates.prompt = None
    monkeypatch.delenv("SEALED_LAB_NO_UPDATE_CHECK", raising=False)
    assert w.updates.maybe_auto_check()
    wait_until(lambda: not w.updates.check_task.running)
    QApplication.processEvents()
    assert w.updates.prompt is None                                  # skipped: silent
    w.updates.check(manual=True)                                     # asking by hand still offers it
    wait_until(lambda: w.updates.prompt is not None)
    fake.info.latest = "v10.0.0"                                     # a newer one than the skipped is offered again
    w.updates.prompt.close()
    w.updates.prompt = None
    w.updates.maybe_auto_check()
    wait_until(lambda: w.updates.prompt is not None, what="prompt for a newer version")


def test_auto_check_off_or_env_or_not_due_runs_nothing(make, monkeypatch):
    w, fake, *_ = make()
    monkeypatch.delenv("SEALED_LAB_NO_UPDATE_CHECK", raising=False)
    w.updates.auto_action.setChecked(False)
    assert not w.updates.maybe_auto_check() and "check" not in fake.calls
    w.updates.auto_action.setChecked(True)
    fake.is_due = False
    assert not w.updates.maybe_auto_check() and "check" not in fake.calls
    fake.is_due = True
    monkeypatch.setenv("SEALED_LAB_NO_UPDATE_CHECK", "1")
    assert not w.updates.maybe_auto_check() and "check" not in fake.calls


def test_auto_check_error_is_silent(make, monkeypatch):
    w, fake, *_ = make(FakeUpdater(Path(tempfile.gettempdir()), error="could not reach GitHub"))
    monkeypatch.delenv("SEALED_LAB_NO_UPDATE_CHECK", raising=False)
    logs = []
    w.updates.logMessage.connect(logs.append)
    assert w.updates.maybe_auto_check()
    wait_until(lambda: any("could not reach" in m for m in logs), what="logged error")
    assert w.updates.last_box is None and w.updates.prompt is None


def test_window_built_without_auto_refresh_schedules_no_update_check(make, monkeypatch):
    monkeypatch.delenv("SEALED_LAB_NO_UPDATE_CHECK", raising=False)
    w, fake, *_ = make()
    QApplication.processEvents()
    assert fake.calls == []


def test_dev_checkout_shows_reason_and_launches_nothing(make):
    w, fake, quits, opened = make(FakeUpdater(Path(tempfile.gettempdir()), self_update=(False, "This copy is a git checkout; update it with git.")))
    w.updates.check(manual=True)
    wait_until(lambda: w.updates.prompt is not None)
    w.updates.prompt.update_btn.click()
    QApplication.processEvents()
    assert "git checkout" in box_text(w.updates.last_box)
    assert "download" not in fake.calls and not fake.launched and not quits
    w.updates.open_button.click()
    assert opened == ["https://example.test/release"]


def test_happy_path_downloads_stages_and_hands_over(make):
    w, fake, quits, opened = make()
    w.updates.check(manual=True)
    wait_until(lambda: w.updates.prompt is not None)
    w.updates.prompt.update_btn.click()
    wait_until(lambda: w.updates.confirm_box is not None, what="confirmation")
    assert "close, update and reopen" in box_text(w.updates.confirm_box)
    assert fake.calls.count("download") == 1 and fake.calls.count("stage:" + "ab" * 32) == 1 and not fake.launched
    w.updates.confirm_box.button(QMessageBox.StandardButton.Yes).click()
    assert fake.launched == [(fake.tmp / "staged" / "sealed_lab-9.9.9", fake.tmp / "app", os.getpid())]
    assert quits == [1]


def test_declining_the_confirmation_launches_nothing(make):
    w, fake, quits, _ = make()
    w.updates.check(manual=True)
    wait_until(lambda: w.updates.prompt is not None)
    w.updates.prompt.update_btn.click()
    wait_until(lambda: w.updates.confirm_box is not None)
    w.updates.confirm_box.button(QMessageBox.StandardButton.No).click()
    QApplication.processEvents()
    assert not fake.launched and not quits


def test_download_failure_is_reported(make):
    w, fake, quits, _ = make()

    def boom(*a, **k):
        raise RuntimeError("GitHub refused the request")
    fake.download = boom
    w.updates.check(manual=True)
    wait_until(lambda: w.updates.prompt is not None)
    w.updates.prompt.update_btn.click()
    wait_until(lambda: w.updates.last_box is not None, what="download error box")
    assert "refused" in box_text(w.updates.last_box) and not fake.launched
    assert w.updates.progress_dialog is None


def test_unsigned_release_is_never_installed_from_the_app(make):
    w, fake, quits, opened = make(FakeUpdater(Path(tempfile.gettempdir()), signed=False))
    w.updates.check(manual=True)
    wait_until(lambda: w.updates.prompt is not None, what="prompt")
    w.updates.prompt.update_btn.click()
    wait_until(lambda: w.updates.last_box is not None, what="manual-update box")
    assert "not signed" in w.updates.last_box.text()
    assert "download" not in fake.calls and not fake.launched and not quits
    w.updates.open_button.click()
    assert opened == ["https://example.test/release"]


def test_a_token_saved_by_an_older_version_is_deleted(make):
    from sealed_lab.gui.update_ui import LEGACY_KEY_TOKEN

    new_settings().setValue(LEGACY_KEY_TOKEN, "ghp_old")
    new_settings().sync()
    make()
    assert not new_settings().contains(LEGACY_KEY_TOKEN)
