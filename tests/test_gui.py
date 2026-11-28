"""GUI tests: offscreen Qt + the fake backend; no pytest-qt needed."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:\Windows\Fonts"):                      # the offscreen platform has no fonts of its own: real glyphs in the screenshots
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
os.environ["SEALED_LAB_FAKE_API"] = "1"
os.environ["SEALED_LAB_FAKE_DELAY"] = "0.002"
# keep QSettings out of the real user profile
_CFG = tempfile.mkdtemp(prefix="sealed_lab_gui_cfg_")
os.environ["SEALED_LAB_SETTINGS"] = os.path.join(_CFG, "settings.ini")   # widgets.new_settings(): never the registry
os.environ["XDG_CONFIG_HOME"] = _CFG
os.environ.setdefault("HOME", _CFG)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QToolButton  # noqa: E402

QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, _CFG)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _CFG)

from sealed_lab.gui import fake_api  # noqa: E402
from sealed_lab.gui.app import MainWindow, load_backend  # noqa: E402
from sealed_lab.gui.deck_view import CREATURES, LANDS, SPELLS, group_cards  # noqa: E402
from sealed_lab.gui import datasets_ui  # noqa: E402
from sealed_lab.gui.widgets import apply_theme, new_settings  # noqa: E402
from sealed_lab.gui.workers import fetch_card_info  # noqa: E402
from sealed_lab.types import CardLine  # noqa: E402

SHOTS = ROOT / "docs" / "screenshots"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_theme(app, os.environ.get("SEALED_LAB_THEME", "light"))
    return app


def wait_until(pred, timeout: float = 15.0, what: str = "condition"):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        QApplication.processEvents()
        if pred():
            QApplication.processEvents()
            return
        time.sleep(0.005)
    raise AssertionError(f"timeout waiting for {what}")


def idle(win: MainWindow) -> bool:
    pt = win.pool_tab
    return not (win.sets_tab.busy or pt.task.running or pt.import_task.running or pt.check_task.running or win.datasets.dl_task.running)


@pytest.fixture()
def data_dir(tmp_path):
    """The tracked data folder of the window (the fake backend's default), a scratch folder."""
    old = fake_api.fake_data_dir
    d = tmp_path / "17lands"
    d.mkdir()
    fake_api.fake_data_dir = str(d)
    yield d
    fake_api.fake_data_dir = old


@pytest.fixture()
def win(qapp, data_dir):
    fake_api.reset()
    new_settings().clear()
    w = MainWindow(fake_api, auto_refresh=True)
    w.pool_tab.detector.timer.setInterval(5)
    w.sets_tab.rescan_timer.setInterval(30)
    w.show()
    wait_until(lambda: w.sets_tab.set_table.rowCount() == 2 and idle(w), what="initial set list")
    yield w
    w.sets_tab.wait(3000)
    w.pool_tab.wait(3000)
    w.close()
    QApplication.processEvents()


def select_set(win: MainWindow, code: str) -> None:
    t = win.sets_tab.set_table
    for r in range(t.rowCount()):
        if t.item(r, 0).text() == code:
            t.selectRow(r)
            return
    raise AssertionError(code)


def row_state(win: MainWindow, code: str) -> str:
    t = win.sets_tab.set_table
    for r in range(t.rowCount()):
        if t.item(r, 0).text() == code:
            return t.item(r, 1).text()
    raise AssertionError(code)


def calculate(win: MainWindow, code: str = "NEO") -> None:
    select_set(win, code)
    win.sets_tab.calc_btn.click()
    wait_until(lambda: idle(win), what="calculate")


def grab(win: MainWindow, name: str, size: tuple[int, int] = (1400, 900)) -> Path:
    SHOTS.mkdir(parents=True, exist_ok=True)
    win.resize(*size)
    QApplication.processEvents()
    p = SHOTS / name
    assert win.grab().save(str(p))
    return p


def write_dataset(folder: Path, name: str, size: int = 2048) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"x" * size)
    return p


# --------------------------------------------------------------------------- basics

def test_backend_selection():
    assert load_backend() is fake_api


def test_window_structure(win):
    assert win.windowTitle() == "sealed_lab"
    assert (win.width(), win.height()) == (1400, 900)
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Sets", "Deck builder", "Report", "Deck search"]     # the Deck score tab is merged; the written report and the analysis are one tab
    from sealed_lab.attribution import DATASETS_URL, LICENSE_URL
    credit = win.tabs.cornerWidget(Qt.Corner.TopRightCorner)                # 17Lands credit, visible at the top level
    assert credit is win.credit and credit.isVisible() and credit.openExternalLinks()
    assert DATASETS_URL in credit.text() and LICENSE_URL in credit.text() and "not affiliated" in credit.text()
    st = win.sets_tab
    assert st.workers_spin.minimum() == 1 and st.workers_spin.maximum() == (os.cpu_count() or 1)
    assert st.workers_spin.value() == (os.cpu_count() or 1)
    assert (st.boot_spin.minimum(), st.boot_spin.maximum(), st.boot_spin.value()) == (4, 100, 16)
    assert st.fetch_cb.isChecked()
    assert not hasattr(st, "file_table") and not hasattr(st, "add_files_btn")                      # no file drop UI any more
    assert win.acceptDrops()
    assert st.advanced.isHidden() and st.log.isHidden()                                              # compact: Advanced and the log are folded away
    st.advanced_btn.setChecked(True)
    assert not st.advanced.isHidden() and st.recency_combo.isVisibleTo(st)


def test_sets_listing_and_status_texts(win):
    st = win.sets_tab
    t = st.set_table
    assert t.rowCount() == 2 and [t.horizontalHeaderItem(c).text() for c in range(4)] == ["Set", "Status", "Games per format", "Built at"]
    assert "Ready" in row_state(win, "TST") and "Recalculation pending" in row_state(win, "NEO")
    r_tst = [r for r in range(2) if t.item(r, 0).text() == "TST"][0]
    assert "bo3 20,310" in t.item(r_tst, 2).text()
    r_neo = 1 - r_tst
    assert "data never used yet" in t.item(r_neo, 1).toolTip()
    assert list(win.pool_tab.statuses) == ["TST"]                                 # only sets with a model feed the Pool tab
    assert st.report_btn.isEnabled() is False                                      # NEO is selected: no model, no report
    select_set(win, "TST")
    assert st.report_btn.isEnabled()


def test_pending_reason_names_the_new_files(win, data_dir):
    write_dataset(data_dir / "TST", "game_data_public.TST.TradSealed.csv.gz")
    write_dataset(data_dir / "TST", "game_data_public.TST.Sealed.csv.gz")
    win.sets_tab.rescan(manual=True)
    wait_until(lambda: idle(win) and "Recalculation pending" in row_state(win, "TST"), what="scan")
    st = fake_api.set_status("TST")
    assert st.stale and st.pending and st.pending[0].startswith("2 new or changed files: game_data_public.TST.")
    t = win.sets_tab.set_table
    r = [r for r in range(t.rowCount()) if t.item(r, 0).text() == "TST"][0]
    assert "2 new or changed files" in t.item(r, 1).toolTip()
    select_set(win, "TST")
    wait_until(lambda: win.sets_tab.panel.pending_box.isVisibleTo(win.sets_tab.panel), what="pending box")
    assert "2 new or changed files" in win.sets_tab.panel.pending_label.text()


# --------------------------------------------------------------------------- tracked data folder

def test_folder_row_and_scan_registers_new_files(win, data_dir):
    st = win.sets_tab
    assert str(data_dir) in st.folder_label.text() and "picked up automatically" in st.folder_note.text()
    assert {st.change_btn.text(), st.open_btn.text(), st.rescan_btn.text()} == {"Change…", "Open", "Rescan"}
    assert st.open_btn.isEnabled()
    # files for a tracked set (a model exists) and for an unknown set (not registered: the set stays untracked)
    write_dataset(data_dir / "TST", "game_data_public.TST.TradSealed.csv.gz")
    write_dataset(data_dir / "TST", "replay_data_public.TST.TradSealed.csv.gz")
    write_dataset(data_dir / "ABC", "game_data_public.ABC.Sealed.csv.gz")
    calls = len(fake_api.scan_calls)
    st.rescan(manual=True)
    wait_until(lambda: len(fake_api.scan_calls) > calls and idle(win), what="scan")
    kinds = {(f.kind, f.event) for f in fake_api.set_status("TST").files}
    assert ("game", "TradSealed") in kinds and ("replay", "TradSealed") in kinds                 # replay stays replay
    assert fake_api.set_status("ABC").files == []
    assert st.scan_label.text().startswith("Last scan")                       # (the folder watcher may have registered them first)
    # a second scan finds nothing new (size and mtime are known)
    st.rescan(manual=True)
    wait_until(lambda: idle(win) and "nothing new" in st.scan_label.text(), what="second scan")
    # ABC shows up greyed (a folder with 17lands files, not tracked)
    wait_until(lambda: "ABC" in [st.set_table.item(r, 0).text() for r in range(st.set_table.rowCount())], what="untracked row")


def test_watcher_picks_up_files_placed_in_the_folder(win, data_dir):
    st = win.sets_tab
    wait_until(lambda: str(data_dir) in {os.path.normpath(p) for p in st.watcher.directories()} or
               os.path.normpath(str(data_dir)) in {os.path.normpath(p) for p in st.watcher.directories()}, what="watcher on the folder")
    n = len(fake_api.scan_calls)
    write_dataset(data_dir / "TST", "game_data_public.TST.TradDraft.csv.gz")    # a new file in a set folder
    wait_until(lambda: len(fake_api.scan_calls) > n and idle(win), 20, "automatic scan after the folder changed")
    assert any(f.event == "TradDraft" for f in fake_api.set_status("TST").files)


def test_scan_leaves_the_calculating_set_alone_and_looks_again_afterwards(win, data_dir, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.06")
    st = win.sets_tab
    fake_api._models["TST"]["stale"] = True
    st.refresh()
    wait_until(lambda: idle(win), what="refresh")
    select_set(win, "TST")
    st.calc_btn.click()
    wait_until(lambda: st.running == "TST", what="running")
    write_dataset(data_dir / "TST", "game_data_public.TST.TradDraft.csv.gz")
    n = len(fake_api.scan_calls)
    st.rescan(manual=True)
    wait_until(lambda: len(fake_api.scan_calls) > n, what="scan during the calculation")
    assert fake_api.scan_calls[-1]["skip_sets"] == ["TST"]                              # the set being calculated is not touched
    assert not any(f.event == "TradDraft" for f in fake_api.set_status("TST").files)
    n = len(fake_api.scan_calls)
    wait_until(lambda: not st.running, 30, "calculation done")
    wait_until(lambda: len(fake_api.scan_calls) > n and idle(win), what="the deferred scan after the calculation")
    assert any(f.event == "TradDraft" for f in fake_api.set_status("TST").files)
    assert "Recalculation pending" in row_state(win, "TST")                           # the model is now older than the data


def test_window_activation_rescans_at_most_once_a_minute(win):
    st = win.sets_tab
    wait_until(lambda: idle(win), what="start-up scan")
    n = len(fake_api.scan_calls)
    st.on_app_activated()                                                              # just scanned: nothing to do yet
    wait_until(lambda: idle(win) and not st.rescan_timer.isActive(), what="quiet")
    assert len(fake_api.scan_calls) == n
    st._last_scan -= 120
    st.on_app_activated()
    wait_until(lambda: len(fake_api.scan_calls) > n and idle(win), what="rescan on activation")


def test_missing_files_are_flagged_not_dropped(win, data_dir):
    f = write_dataset(data_dir / "TST", "game_data_public.TST.TradSealed.csv.gz")
    win.sets_tab.rescan(manual=True)
    wait_until(lambda: idle(win) and "pending" in row_state(win, "TST"), what="registered")
    n_files = len(fake_api.set_status("TST").files)
    f.unlink()
    win.sets_tab.rescan(manual=True)
    wait_until(lambda: idle(win) and fake_api.set_status("TST").missing, what="missing flagged")
    assert len(fake_api.set_status("TST").files) == n_files                                       # still registered
    select_set(win, "TST")
    panel = win.sets_tab.panel
    wait_until(lambda: "TST" in win.sets_tab.overviews and panel.files_table.rowCount() == n_files, what="files in the expander")
    assert any(panel.files_table.item(r, 5).text() == "MISSING" for r in range(panel.files_table.rowCount()))
    assert "no longer on disk" in panel.warn_label.text() and "MISSING" in panel.files.toggle.text() or "missing" in panel.files.toggle.text()


def test_untracked_sets_are_greyed_and_can_be_tracked(win, data_dir):
    write_dataset(data_dir / "XYZ", "game_data_public.XYZ.TradSealed.csv.gz")
    write_dataset(data_dir / "XYZ", "game_data_public.XYZ.Sealed.csv.gz")
    fake_api.untracked_remote.append(dict(set_code="QQQ", events=["PremierDraft", "Sealed"]))
    st = win.sets_tab
    st.refresh()
    wait_until(lambda: st.set_table.rowCount() == 4 and idle(win), what="untracked rows")
    codes = [st.set_table.item(r, 0).text() for r in range(st.set_table.rowCount())]
    assert codes == ["NEO", "TST", "QQQ", "XYZ"]
    assert row_state(win, "XYZ").endswith("Not tracked")
    it = st.set_table.item(3, 0)
    assert it.font().italic() and "2 file(s) on disk" in st.set_table.item(3, 2).text()
    assert "1 on 17lands" not in st.set_table.item(2, 2).text() and "on 17lands" in st.set_table.item(2, 2).text()
    # greyed rows can be hidden
    st.show_untracked_cb.setChecked(False)
    assert st.set_table.rowCount() == 2
    st.show_untracked_cb.setChecked(True)
    # click on XYZ: the panel offers the folder, and a download is offered only for sets 17lands lists
    select_set(win, "XYZ")
    p = st.panel
    assert p.untracked_box.isVisibleTo(p) and p.use_btn.isVisibleTo(p) and not p.download_btn.isVisibleTo(p)
    assert "XYZ" in p.untracked_label.text() and "2 17lands file" in p.untracked_label.text()
    assert not st.calc_btn.isEnabled()
    select_set(win, "QQQ")
    assert p.download_btn.isVisibleTo(p) and not p.use_btn.isVisibleTo(p) and p.choose_btn.isVisibleTo(p)
    p.download_btn.click()                                                                       # the existing download flow lists the files
    wait_until(lambda: win.datasets.isVisible(), what="datasets panel")
    assert [i.name for i in win.datasets.items] == ["game_data_public.QQQ.PremierDraft.csv.gz", "game_data_public.QQQ.Sealed.csv.gz"]
    win.datasets.hide_panel()
    # track XYZ from its folder: its files are registered at once and the row turns into a normal one
    select_set(win, "XYZ")
    p.use_btn.click()
    wait_until(lambda: idle(win) and "XYZ" in [s.set_code for s in win.sets_tab.statuses], what="tracked")
    assert "Recalculation pending" in row_state(win, "XYZ") and len(fake_api.set_status("XYZ").files) == 2
    assert win.sets_tab.selected_set() == "XYZ" and win.sets_tab.calc_btn.isEnabled()


def test_set_can_have_its_own_folder(win, tmp_path):
    other = tmp_path / "elsewhere"
    write_dataset(other, "game_data_public.ZZZ.TradSealed.csv.gz")
    win.sets_tab.track_set("ZZZ", str(other))
    wait_until(lambda: idle(win) and "ZZZ" in [s.set_code for s in win.sets_tab.statuses], what="tracked from its own folder")
    assert fake_api.set_folders()["ZZZ"] == str(other) and fake_api.set_status("ZZZ").folder == str(other)
    # nothing there: the user is told
    errs = []
    win.errorOccurred = None
    win.sets_tab.errorOccurred.connect(lambda m, t: errs.append(m))
    win.sets_tab.track_set("EMPTY", str(tmp_path))
    wait_until(lambda: errs, what="no files message")
    assert "No 17lands files for EMPTY" in errs[0]


def test_drop_a_folder_tracks_it_a_file_registers_it(win, tmp_path, data_dir):
    folder = tmp_path / "dl" / "AAA"
    write_dataset(folder, "game_data_public.AAA.Sealed.csv.gz")
    write_dataset(folder, "game_data_public.AAA.TradSealed.csv.gz")
    win.tabs.setCurrentWidget(win.pool_tab)
    _drop(win, [folder])                                                    # anywhere on the window -> switches to Sets and tracks the folder
    wait_until(lambda: idle(win) and "AAA" in [s.set_code for s in win.sets_tab.statuses], what="folder drop")
    assert win.tabs.currentWidget() is win.sets_tab
    assert fake_api.set_folders()["AAA"] == str(folder) and len(fake_api.set_status("AAA").files) == 2
    f2 = write_dataset(tmp_path / "loose", "game_data_public.BBB.TradSealed.csv.gz")
    _drop(win, [f2])
    wait_until(lambda: idle(win) and "BBB" in [s.set_code for s in win.sets_tab.statuses], what="file drop")
    assert fake_api.set_status("BBB").files[0].path == str(f2)
    assert not (data_dir / "AAA").exists() and not (data_dir / "BBB").exists()        # nothing is copied


def test_set_statistics_panel(win):
    from PySide6.QtCore import QCoreApplication  # noqa: F401

    st = win.sets_tab
    select_set(win, "TST")
    p = st.panel
    wait_until(lambda: "TST" in st.overviews and "601,204" in p.data_label.text(), what="statistics")
    d = p.data_label.text()
    assert "Draft" in d and "Bo1 Sealed" in d and "Bo3 Traditional Sealed" in d and "49,977" in d and "20,310" in d
    assert "83,410" in d and "2026-09-03" in d and "Replay games" in d and "57,000" in d
    m = p.model_label.text()
    assert "Built" in m and "7 min 49 s" in m and "matches the tracked files" in m
    assert "✔&nbsp;Opponent&nbsp;matchups" in m and "✘&nbsp;Recency&nbsp;weighting" in m and "✔&nbsp;Castability" in m             # components on / off
    assert "beats the no-deck-information baseline" in m and "calibration slope 1.08" in m
    assert "Deployed combination" in m and "matchups, castability" in m
    assert "λ = 0.80 ± 0.05" in m and "Draft → Bo1" in m
    top = p.top_label.text()
    assert top.count("Iron Knight") == 2 or top.count("pp</b>") == 10                          # 5 cards per format, two formats
    assert "Bo3 Traditional Sealed" in top and "Bo1 Sealed" in top
    # the file list is a collapsed expander, still reachable
    assert not p.files.toggle.isChecked() and p.files_table.isHidden()
    assert p.files.toggle.text() == "Files (4)"
    p.files.set_expanded(True)
    assert p.files_table.isVisibleTo(p) and p.files_table.rowCount() == 4
    assert p.files_table.item(0, 0).text().startswith("game_data_public.TST.")
    # a set without a model: the panel says so
    select_set(win, "NEO")
    wait_until(lambda: "NEO" in st.overviews, what="NEO statistics")
    assert "No model yet" in p.model_label.text() and p.pending_box.isVisibleTo(p)
    grab(win, "sets_statistics.png")


def test_verdict_texts():
    from sealed_lab.gui.set_panel import fmt_duration, verdict

    assert verdict(dict(status="ok", gain_vs_B0=0.004, lo_vs_B0=0.001, hi_vs_B0=0.007))[1] == "good"
    assert verdict(dict(status="ok", gain_vs_B0=0.004, lo_vs_B0=-0.001, hi_vs_B0=0.007))[1] == "warn"
    text, role = verdict(dict(status="ok", gain_vs_B0=-0.004, lo_vs_B0=-0.007, hi_vs_B0=-0.001))
    assert role == "bad" and "WORSE" in text
    assert verdict(dict(status="skipped", reason="too little data"))[1] == "warn"
    assert fmt_duration(468.6) == "7 min 49 s"
    assert fmt_duration(30) == "30 s" and fmt_duration(3700) == "1 h 01 min"


# --------------------------------------------------------------------------- calculate: queue, progress, failure

def test_calculate_progress_and_refresh(win):
    seen = []
    win.sets_tab.task.progress.connect(lambda f, m: seen.append((f, m)))
    select_set(win, "NEO")
    assert win.sets_tab.calc_btn.text() == "Calculate"
    win.sets_tab.calc_btn.click()
    wait_until(lambda: win.sets_tab.running == "NEO", what="start")
    assert win.sets_tab.cancel_btn.isEnabled()
    assert win.sets_tab.calc_btn.text() == "Add to queue" and not win.sets_tab.calc_btn.isEnabled()      # NEO is already running
    assert "Calculating" in row_state(win, "NEO")
    wait_until(lambda: idle(win), what="calculate")
    assert seen and seen[-1][0] == pytest.approx(1.0)
    assert win.sets_tab.progress.value() == 1000 and "Done: NEO model built" in win.sets_tab.progress_label.text()
    assert "Ready" in row_state(win, "NEO") and win.sets_tab.calc_btn.text() == "Calculate"
    assert "NEO" in win.pool_tab.statuses
    assert "Ingesting" in win.sets_tab.log.toPlainText()


def test_calculate_cancel(win, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.03")
    select_set(win, "NEO")
    win.sets_tab.calc_btn.click()
    wait_until(lambda: win.sets_tab.progress.value() > 0, what="some progress")
    win.sets_tab.cancel_btn.click()
    wait_until(lambda: idle(win), what="cancel")
    assert "cancelled" in win.sets_tab.progress_label.text()
    assert fake_api.set_status("NEO").has_model is False and "NEO" not in win.sets_tab.failed


def test_queue_runs_sets_one_after_another(win, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.01")
    fake_api.add_files([str(_mk(win, "game_data_public.AAA.TradSealed.csv.gz")), str(_mk(win, "game_data_public.BBB.Sealed.csv.gz"))])
    st = win.sets_tab
    st.refresh()
    wait_until(lambda: st.set_table.rowCount() == 4 and idle(win), what="four sets")
    order = []
    st.calcSetChanged.connect(lambda c: order.append(c))
    for code in ("NEO", "AAA", "BBB"):
        select_set(win, code)
        st.calc_btn.click()
    assert st.running == "NEO" and st.queued() == ["AAA", "BBB"]
    assert st.calc_btn.text() == "Add to queue"
    assert "Queued (#1)" in row_state(win, "AAA") and "Queued (#2)" in row_state(win, "BBB")
    assert list(st.queue_strip.chips) == ["NEO", "AAA", "BBB"] and set(st.queue_strip.remove_buttons) == {"AAA", "BBB"}
    assert st.cancel_all_btn.isVisibleTo(st)
    # removing a waiting set takes it out of the queue
    st.queue_strip.remove_buttons["BBB"].click()
    assert st.queued() == ["AAA"] and "BBB" not in st.queue_strip.chips and "pending" in row_state(win, "BBB")
    # the same set cannot be queued twice
    select_set(win, "AAA")
    assert not st.calc_btn.isEnabled()
    wait_until(lambda: idle(win), 30, "queue")
    assert [c for c in order if c] == ["NEO", "AAA"] and order[-1] == ""                  # one after the other, none at the end
    assert "Ready" in row_state(win, "NEO") and "Ready" in row_state(win, "AAA") and "pending" in row_state(win, "BBB")
    assert st.queue_strip.empty.isVisibleTo(st)


def _mk(win, name: str) -> Path:
    d = Path(fake_api.fake_data_dir).parent / "loose"
    return write_dataset(d, name)


def test_cancel_cancels_the_running_set_and_the_queue_continues(win, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.02")
    fake_api.add_files([str(_mk(win, "game_data_public.AAA.TradSealed.csv.gz"))])
    st = win.sets_tab
    st.refresh()
    wait_until(lambda: st.set_table.rowCount() == 3 and idle(win), what="three sets")
    for code in ("NEO", "AAA"):
        select_set(win, code)
        st.calc_btn.click()
    wait_until(lambda: st.progress.value() > 50, what="NEO running")
    st.cancel_btn.click()                                                              # "Cancel": the running set only
    wait_until(lambda: st.running == "AAA", what="the queue goes on")
    assert not fake_api.set_status("NEO").has_model
    # "Cancel all" empties the queue and cancels the running set
    select_set(win, "NEO")
    st.calc_btn.click()
    assert st.queued() == ["NEO"]
    st.cancel_all_btn.click()
    assert st.queued() == [] and not st.cancel_all_btn.isVisibleTo(st)
    wait_until(lambda: idle(win), what="all cancelled")
    assert not fake_api.set_status("AAA").has_model and not fake_api.set_status("NEO").has_model


def test_failed_calculation_shows_the_failed_state_and_logs(win):
    from sealed_lab.gui import workers

    fake_api.calc_fail["NEO"] = "scryfall exploded"
    st = win.sets_tab
    select_set(win, "NEO")
    st.calc_btn.click()
    wait_until(lambda: idle(win) and win.last_error is not None, what="failure")
    assert "scryfall exploded" in win.last_error[0] and "Traceback" in win.last_error[1]
    assert "Failed — see details" in row_state(win, "NEO")
    t = st.set_table
    assert "scryfall exploded" in t.item([r for r in range(2) if t.item(r, 0).text() == "NEO"][0], 1).toolTip()
    assert st.failed["NEO"][0].endswith("scryfall exploded")
    p = st.panel
    assert p.failed_box.isVisibleTo(p) and "scryfall exploded" in p.failed_label.text() and "Traceback" in p.trace.toPlainText()
    # one line per failed operation in errors.log: timestamp, operation (with the set), traceback
    log = workers.error_log_path().read_text(encoding="utf-8")
    assert "calculate NEO" in log and "scryfall exploded" in log and "Traceback" in log
    # the failed state lasts for the session; a new calculation clears it
    fake_api.calc_fail.clear()
    st.calc_btn.click()
    wait_until(lambda: idle(win), what="second try")
    assert "NEO" not in st.failed and "Ready" in row_state(win, "NEO")


# --------------------------------------------------------------------------- pool: detected set, format, guessed names

def paste(win: MainWindow, text: str, code: str = "TST") -> None:
    pt = win.pool_tab
    pt.text.setPlainText(text)
    wait_until(lambda: pt.set_code == code and pt.detector.idle() and not pt.check_task.running
               and pt._check_snapshot == (code, text), what=f"set {code} detected and the names checked")


def build(win: MainWindow, text: str | None = None, fmt: str = "bo3"):
    pt = win.pool_tab
    wait_until(lambda: "TST" in pt.statuses, what="set known")
    paste(win, text if text is not None else fake_api.sample_pool_text())
    pt.fmt_radios[fmt].click()
    pt.build_btn.click()
    wait_until(lambda: pt.task.running or pt.result is not None, what="build start")
    wait_until(lambda: idle(win), what="build")


def test_set_is_detected_and_there_is_no_set_dropdown(win):
    pt = win.pool_tab
    assert not hasattr(pt, "bar")
    assert "paste a pool" in pt.set_label.text() and pt.set_combo.isHidden()
    paste(win, fake_api.sample_pool_text())
    assert "TST" in pt.set_label.text() and "detected (58/59 cards matched)" in pt.set_label.text()
    assert pt.set_combo.isHidden() and pt.set_source == "detected"
    pt.text.clear()
    wait_until(lambda: pt.set_code is None, what="cleared")
    assert "paste a pool" in pt.set_label.text() and not pt.build_btn.isEnabled()


def test_inline_set_chooser_only_when_the_cards_do_not_tell(win):
    calculate(win, "NEO")                                              # now two modelled sets (identical card names)
    wait_until(lambda: set(win.pool_tab.statuses) == {"NEO", "TST"}, what="two sets")
    pt = win.pool_tab
    pt.text.setPlainText(fake_api.sample_pool_text())
    wait_until(lambda: pt.detector.idle() and not pt.set_combo.isHidden(), what="chooser")
    assert pt.set_code is None and not pt.build_btn.isEnabled()
    assert "choose" in pt.set_label.text() and [pt.set_combo.itemText(i) for i in range(pt.set_combo.count())] == ["— choose the set —", "NEO", "TST"]
    i = pt.set_combo.findData("NEO")
    pt.set_combo.setCurrentIndex(i)
    pt.set_combo.activated.emit(i)
    wait_until(lambda: pt.set_code == "NEO", what="chosen")
    assert "chosen by you" in pt.set_label.text() and pt.build_btn.isEnabled()
    # the set code suffixes decide: no chooser
    rows = [f"1 {d['name']} (TST) {i}" for i, d in fake_api._CARDS.items() if not d["is_land"]][:50]
    pt.text.setPlainText("Deck\n" + "\n".join(rows) + "\n")
    wait_until(lambda: pt.set_code == "TST" and pt.set_combo.isHidden() and pt.detector.idle(), what="suffix detection")
    assert pt.set_source == "detected"


def test_detected_set_without_a_model_is_said_plainly(win):
    pt = win.pool_tab
    rows = [f"1 {d['name']} (FDN) {i}" for i, d in fake_api._CARDS.items() if not d["is_land"]][:50]
    pt.text.setPlainText("Deck\n" + "\n".join(rows) + "\n")
    wait_until(lambda: "FDN" in pt.set_label.text() and pt.detector.idle(), what="note")
    assert "no model yet" in pt.set_label.text() and pt.set_code is None and not pt.build_btn.isEnabled()


def test_format_is_two_radio_buttons_with_the_more_data_default(win):
    from sealed_lab.gui import helptext as H

    pt = win.pool_tab
    assert not hasattr(pt, "fmt_combo")
    assert pt.fmt_labels["bo1"].text() == "Bo1 — Arena Sealed: single games, no sideboarding"
    assert pt.fmt_labels["bo3"].text() == "Bo3 — Traditional Sealed: best-of-three, sideboard between games"
    for tip in (pt.format_help.toolTip(), pt.fmt_bo1.toolTip(), pt.fmt_bo3.toolTip(), pt.fmt_labels["bo1"].toolTip()):
        assert tip == H.FORMAT_HELP
    assert "separately for each format" in H.FORMAT_HELP and "faster, more aggressive" in H.FORMAT_HELP
    assert "more data" in H.FORMAT_HELP and "sideboard plans" in H.FORMAT_HELP
    paste(win, fake_api.sample_pool_text())
    assert pt.fmt() == "bo1" and pt.fmt_bo1.isChecked() and "preselected" in pt.fmt_note.text()       # TST: Bo1 49,977 games vs Bo3 20,310
    assert "BO1 49,977" in pt.fmt_note.text() and "BO3 20,310" in pt.fmt_note.text()
    pt.fmt_bo3.click()                                                                                # the user's own choice is kept
    assert pt.fmt() == "bo3"
    pt.fmt_labels["bo1"].mouseReleaseEvent(_click_event())                                            # the label picks the radio, too
    assert pt.fmt() == "bo1"
    pt.fmt_bo3.click()
    fake_api._models["TST"]["n_games"] = {"draft": 1, "bo1": 100, "bo3": 20_000}
    win.sets_tab.refresh()
    wait_until(lambda: idle(win), what="refresh")
    assert pt.fmt() == "bo3" and "preselected" not in pt.fmt_note.text()


def _click_event():
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(3, 3), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                       Qt.KeyboardModifier.NoModifier)


def test_default_format_helper_and_unavailable_format(win):
    from sealed_lab.gui.pool_tab import default_format

    class S:
        formats = ["bo1", "bo3"]
        n_games = {"bo1": 5, "bo3": 5}
    assert default_format(S()) == "bo3" and default_format(None) is None
    S.n_games = {"bo1": 9, "bo3": 5}
    assert default_format(S()) == "bo1"
    S.formats = ["bo3"]
    assert default_format(S()) == "bo3"
    pt = win.pool_tab
    fake_api._models["TST"]["formats"] = ["bo1"]
    win.sets_tab.refresh()
    wait_until(lambda: idle(win), what="refresh")
    paste(win, fake_api.sample_pool_text())
    assert not pt.fmt_bo3.isEnabled() and pt.fmt_bo1.isEnabled() and pt.fmt() == "bo1"
    build(win, fmt="bo1")
    assert pt.result.fmt == "bo1" and pt.views[0].m_match.value.text() == "—"


def _typo_pool() -> str:
    txt = fake_api.sample_pool_text()
    return txt.replace("1 Dawn Knight\n", "1 Dawn Knigth\n", 1).replace("1 Dawn Drake\n", "1 Dawn Drak\n", 1)


def test_guessed_card_names_are_flagged_and_can_be_changed(win):
    pt = win.pool_tab
    assert pt.guess_box.isHidden()
    paste(win, _typo_pool())
    assert pt.guess_box.isVisibleTo(pt) and "2 card names were guessed — please confirm" in pt.guess_title.text()
    raws = [r for r, _l, _c in pt.guess_rows]
    assert raws == ["Dawn Knigth", "Dawn Drak", "Unknown Card Name"]                    # guesses first, then the unrecognised name
    assert "1 name is not a card of this set" in pt.guess_title.text()
    by_raw = {r: (lab, cb) for r, lab, cb in pt.guess_rows}
    lab, cb = by_raw["Dawn Knigth"]
    assert lab.text().startswith("‘Dawn Knigth’ →") and cb.currentData() == "Dawn Knight" and cb.currentText().startswith("Dawn Knight  (")
    names = [cb.itemData(i) for i in range(cb.count())]
    assert names[-1] == "" and cb.itemText(cb.count() - 1) == "Remove this card" and 2 <= len(names) - 1 <= 5     # top candidates + remove
    assert pt.overrides() == {"Dawn Knigth": "Dawn Knight", "Dawn Drak": "Dawn Drake"}      # the unrecognised one stays out unless matched by hand
    # builds proceed with the guesses; the result is not out of date
    pt.fmt_bo3.click()
    pt.build_btn.click()
    wait_until(lambda: pt.result is not None and idle(win), what="build")
    assert pt.stale_banner.isHidden()
    assert "Dawn Knigth" not in " ".join(pt.top_warn.lines) and "Unknown Card Name" in pt.top_warn.lines[0]
    n_pool = sum(pt.result.pool.values())
    # changing a guess (remove this card) updates the pool and invalidates the result: a rebuild prompt
    cb.setCurrentIndex(cb.count() - 1)
    cb.activated.emit(cb.count() - 1)
    assert pt.overrides()["Dawn Knigth"] == "" and pt.stale_banner.isVisibleTo(pt)
    assert "a card-name choice" in pt.stale_label.text() and pt.stale_btn.isEnabled()
    pt.stale_btn.click()
    wait_until(lambda: pt.stale_banner.isHidden() and idle(win), what="rebuilt")
    assert sum(pt.result.pool.values()) == n_pool - 1
    # editing the pasted list makes it out of date as well
    pt.text.appendPlainText("1 Dawn Goblin")
    wait_until(lambda: pt.stale_banner.isVisibleTo(pt), what="stale after edit")
    assert "pasted list" in pt.stale_label.text()
    grab(win, "pool_guessed_names.png")


def test_name_choices_are_forgotten_for_a_new_paste(win):
    pt = win.pool_tab
    paste(win, _typo_pool())
    _l, cb = pt.guess_rows[0][1], pt.guess_rows[0][2]
    cb.setCurrentIndex(cb.count() - 1)
    cb.activated.emit(cb.count() - 1)
    assert "" in pt.overrides().values()
    other = fake_api.sample_pool_text().replace("1 Iron Warden\n", "1 Iron Wardn\n", 1)
    pt.text.setPlainText("1 Tide Ghoul\n" + "\n".join(fake_api.sample_pool_text().splitlines()[30:]) + "\n1 Iron Wardn\n")
    wait_until(lambda: pt.detector.idle() and not pt.check_task.running, what="new check")
    assert "Dawn Knigth" not in pt.choices and other


def test_unrecognised_name_can_be_matched_by_hand(win):
    pt = win.pool_tab
    paste(win, fake_api.sample_pool_text())                       # contains "Unknown Card Name": nothing close enough to guess
    rows = {r: (lab, cb) for r, lab, cb in pt.guess_rows}
    lab, cb = rows["Unknown Card Name"]
    assert cb.currentData() == "" and cb.itemText(cb.count() - 1) == "Remove this card" and cb.count() == 6
    assert "Unknown Card Name" not in pt.overrides()                # ignored unless the user picks a card
    assert "guessed" not in pt.guess_title.text() and "1 name is not a card of this set" in pt.guess_title.text()
    cb.setCurrentIndex(0)
    cb.activated.emit(0)
    assert pt.overrides() == {"Unknown Card Name": "Dawn Drake"}


# --------------------------------------------------------------------------- pool: results

def test_build_pool_results(win):
    pt = win.pool_tab
    win.tabs.setCurrentWidget(pt)
    build(win)
    assert pt.fmt() == "bo3"
    res = pt.result
    assert res is not None and len(res.decks) == 3 == len(res.families)
    assert pt.tabs.count() == 3 and pt.user_builder is None                          # a plain pool: no "Your deck"
    assert pt.tabs.tabText(0) == "#1 WU+r · 57.3% · P(best) 56%"
    assert pt.tabs.tabText(1).startswith("#2 WB · 56.8% · P(best) 31%")
    v = pt.views[0]
    n_spells = sum(c.count for c in res.decks[0].cards if not c.name in {"Plains", "Island", "Swamp", "Mountain", "Forest"}
                   and c.name != "Scoured Crossroads")
    assert n_spells == 23
    # three groups + their header rows
    assert v.table.group_rows == [CREATURES, SPELLS, LANDS]
    assert v.table.n_card_rows == len(res.decks[0].cards)
    assert v.table.rowCount() == len(res.decks[0].cards) + 3
    assert v.m_game.value.text() == "57.3%"
    assert v.m_match.value.text() != "—"
    assert v.m_score.value.text() == f"{res.decks[0].score:.0f}"
    assert v.m_pbest.value.text() == "56%"
    assert "Top family" in v.diff_label.text()
    assert "12 nonland cards different from #1" in pt.views[1].diff_label.text()
    assert "gap" in pt.views[1].m_pbest.sub.text()
    # deck family box + flex chips on the builder
    b0, b1 = pt.builders[0], pt.builders[1]
    assert b0.family_box.isVisibleTo(b0) and "Family #1" in b0.family_label.text() and "P(best) 56%" in b0.family_label.text()
    assert "Gap vs #1" in b1.family_label.text() and "80% interval" in b1.family_label.text()
    assert "Core (" in b0.family_label.text() and "Flex slots" in b0.family_label.text()
    flex_ids = {x["card_id"] for x in res.families[0]["flex"]}
    assert flex_ids and set(b0.deck_pile.chips) == flex_ids and all(v == "flex" for v in b0.deck_pile.chips.values())
    assert all("Flex card" in t for t in b0.deck_pile.tips.values())
    assert "refits" in b0.family_box.toggle.toolTip() and "10 cards" not in b0.family_box.toggle.toolTip()
    assert v._mana_labels["p_screw"].text().endswith("%")
    assert pt.top_warn.isVisible() and "Unknown Card Name" in pt.top_warn.lines[0]
    # value column colored by sign and low-data marker present
    rows = v.table.rows_text()
    assert any(r[3].startswith("⚠") for r in rows) or any("low data" in r[3] for r in rows)
    vals = [r[2] for r in rows if r[2]]
    assert all(x[0] in "+-" for x in vals)
    # copy export goes to the clipboard
    v.copy_btn.click()
    txt = QApplication.clipboard().text()
    assert txt.startswith("Deck\n") and "Plains" in txt or "Island" in txt
    grab(win, "pool_tab.png")


def test_error_dialog_and_traceback(win, monkeypatch):
    from sealed_lab.gui import workers

    def boom(*a, **k):
        raise RuntimeError("model file is corrupt")
    monkeypatch.setattr(fake_api, "build_pool", boom)
    build(win)
    assert win.last_error is not None
    assert win.last_error[0].endswith("model file is corrupt")
    log = win.sets_tab.log.toPlainText()
    assert "Traceback" in log and "model file is corrupt" in log
    assert win.pool_tab.build_btn.isEnabled()
    text = workers.error_log_path().read_text(encoding="utf-8")                          # the operation and set are named in errors.log
    assert "build pool TST bo3" in text and "model file is corrupt" in text


def test_empty_pool_message(win):
    pt = win.pool_tab
    paste(win, fake_api.sample_pool_text())
    pt.text.setPlainText("   ")
    assert not pt.build_btn.isEnabled()
    pt.set_code = "TST"                                               # the button is disabled; the guard is still there for other callers
    pt.build()
    assert win.last_error and "Paste" in win.last_error[0]


def test_optional_card_info(win):
    class NoInfo:
        pass
    assert fetch_card_info(NoInfo(), "TST") is None

    class Raises:
        def card_info(self, s):
            raise AttributeError
    assert fetch_card_info(Raises(), "TST") is None

    cards = [CardLine(1, "A", 1, -0.5, False), CardLine(2, "B", 2, 1.2, True), CardLine(None, "Plains", 5, None, False)]
    flat = group_cards(cards, None)
    assert len(flat) == 1 and flat[0][0] == "" and [c.name for c in flat[0][1]] == ["B", "A", "Plains"]
    grouped = group_cards(cards, {1: {"name": "A", "mana_value": 3, "is_creature": True, "is_land": False},
                                  2: {"name": "B", "mana_value": 1, "is_creature": False, "is_land": False}})
    assert [g for g, _ in grouped] == [CREATURES, SPELLS, LANDS]


def test_flat_list_without_card_info(win, monkeypatch):
    monkeypatch.delattr(fake_api, "card_info")
    build(win)
    v = win.pool_tab.views[0]
    assert v.table.group_rows == []
    assert v.table.rowCount() == len(win.pool_tab.result.decks[0].cards)
    vals = [float(r[2]) for r in v.table.rows_text() if r[2]]
    assert vals == sorted(vals, reverse=True)


# --------------------------------------------------------------------------- pool: the user's own deck (merged Deck score tab)

def own_deck_text() -> str:
    return fake_api.sample_deck_text(with_side=True)


def test_deck_export_with_sideboard_opens_your_deck_first(win):
    pt = win.pool_tab
    paste(win, own_deck_text())
    assert pt.deck_note.isVisibleTo(pt) and "Deck + Sideboard export detected" in pt.deck_note.text()
    assert pt.pool_check["has_deck"] is True
    pt.fmt_bo3.click()
    pt.build_btn.click()
    wait_until(lambda: pt.result is not None and idle(win), what="build")
    res = pt.result
    assert res.user_deck is not None and pt.tabs.count() == 4                           # Your deck + the three families
    assert pt.tabs.tabText(0).startswith("Your deck · 54.8%") and "pp vs #1" in pt.tabs.tabText(0)
    assert pt.tabs.tabText(1).startswith("#1 ")
    ub = pt.user_builder
    assert ub is pt.builders[0] and ub.orig_eval is res.user_deck and sum(ub.deck.values()) == 40
    assert ub.origin_label == "your pasted deck" and ub.m_delta.caption.text() == "vs your pasted deck"
    # the pool is deck + sideboard, so the deck is fully editable with the sideboard cards
    assert sum(ub.side().values()) == 2
    cid = next(c for c in ub.deck if not ub._is_basic(c))
    assert ub.move(cid, "deck")
    ub.wait_idle()
    assert "edited" in pt.tabs.tabText(0)
    # compared with the best family from the start
    assert ub.compare_combo.currentData() == "alt0" and "pp" in ub.compare_label.text()
    ub.undo()
    ub.wait_idle()
    # the family builders do not carry the user's deck as an alternative
    assert all(not t.startswith("Your") for t in [pt.builders[1].compare_combo.itemText(i) for i in range(pt.builders[1].compare_combo.count())])
    grab(win, "pool_your_deck.png")


def test_score_information_lives_in_the_builder_panel(win):
    pt = win.pool_tab
    paste(win, own_deck_text())
    pt.fmt_bo3.click()
    pt.build_btn.click()
    wait_until(lambda: pt.result is not None and idle(win), what="build")
    ub = pt.user_builder
    ub.wait_idle()
    d = ub.current
    # the old Deck score tab's numbers, live for whatever deck is being edited, with their tooltips
    assert ub.m_score.value.text() == f"{d.score:.0f}" and "percentile among real BO3 decks" in ub.m_score.sub.text()
    assert f"{d.score_lo:.0f}–{d.score_hi:.0f}" in ub.m_score.sub.text() and "median real deck" in ub.m_score.toolTip()
    assert ub.m_game.value.text() == f"{d.game_wr * 100:.1f}%" and "average 17lands pilot" in ub.m_game.toolTip()
    assert ub.m_match.value.text() != "—" and "p²" in ub.m_match.toolTip()
    assert ub.m_wins.value.text() == f"{d.exp_wins:.2f}" and "wins" in ub.m_wins.toolTip()
    assert ub.m_pmax.value.text() == f"{d.p_max_wins * 100:.1f}%" and "P(4 wins)" in ub.m_pmax.sub.text() and "maximum" in ub.m_pmax.toolTip()
    assert ub.rule_label.text() == "Bo3: run ends at 4 wins or 2 losses"
    gap = 100 * (d.game_wr - pt.result.decks[0].game_wr)
    assert "Efficiency vs the best family (#1)" in ub.eff_label.text() and f"{gap:+.2f} pp" in ub.eff_label.text() and gap < -1
    assert ub.eff_label.isVisibleTo(ub)
    # live: editing the deck moves the numbers
    wins0 = ub.m_wins.value.text()
    cid = next(c for c in ub.deck if not ub._is_basic(c))
    ub.move(cid, "deck")
    ub.wait_idle()
    assert ub.m_game.value.text() != f"{d.game_wr * 100:.1f}%" and "pp" in ub.eff_label.text()
    ub.reset()
    ub.wait_idle()
    assert ub.m_wins.value.text() == wins0
    # the family builders show the same panel, efficiency against #1
    b1 = pt.builders[2]
    b1.wait_idle()
    assert "Efficiency vs the best family (#1)" in b1.eff_label.text()
    assert pt.builders[1].eff_label.text().endswith("(this is the best build)")
    # bo1: its own rule text
    assert "7 wins or 3 losses" in __import__("sealed_lab.gui.widgets", fromlist=["x"]).run_rule_text("bo1")


def test_score_only_is_the_fast_path(win):
    pt = win.pool_tab
    paste(win, own_deck_text().replace("1 Iron Knight", "1 Iron Knigth", 1) if "Iron Knight" in own_deck_text() else own_deck_text())
    pt.fmt_bo3.click()
    calls = []
    real = fake_api.score_deck

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)
    fake_api.score_deck = spy
    try:
        pt.score_btn.click()
        wait_until(lambda: pt.result is not None and idle(win), what="score")
    finally:
        fake_api.score_deck = real
    assert calls and calls[0].get("search") is False                                       # no search for a better build
    assert pt.tabs.count() == 1 and pt.tabs.tabText(0).startswith("Your deck · 54.8% · score ")
    assert pt.user_builder is pt.builders[0] and sum(pt.user_builder.deck.values()) == 40
    assert sum(pt.user_builder.side().values()) == 2 and not pt.user_builder.eff_label.isVisibleTo(pt.user_builder)
    assert pt.result.best_alternative is None
    # the scored deck is editable like any other
    cid = next(c for c in pt.user_builder.deck if not pt.user_builder._is_basic(c))
    pt.user_builder.move(cid, "deck")
    pt.user_builder.wait_idle()
    assert sum(pt.user_builder.deck.values()) == 39


def test_score_only_buttons_follow_the_set(win):
    pt = win.pool_tab
    assert not pt.score_btn.isEnabled() and not pt.build_btn.isEnabled()
    paste(win, own_deck_text())
    assert pt.score_btn.isEnabled() and pt.build_btn.isEnabled()
    assert "without searching" in pt.score_btn.toolTip()


# --------------------------------------------------------------------------- per-set locking

def test_calculating_a_set_blocks_only_that_set(win, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.002")
    calculate(win, "NEO")                                   # NEO gets a model too (identical card names: the chooser picks the set)
    wait_until(lambda: set(win.pool_tab.statuses) == {"NEO", "TST"}, what="two sets")
    pt = win.pool_tab
    rows = [f"1 {d['name']} (TST) {i}" for i, d in fake_api._CARDS.items() if not d["is_land"]][:50]
    tst_text = "Deck\n" + "\n".join(rows) + "\n"
    neo_text = tst_text.replace("(TST)", "(NEO)")
    fake_api._models["NEO"]["stale"] = True
    win.sets_tab.refresh()
    wait_until(lambda: idle(win), what="refresh")
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.25")
    select_set(win, "NEO")
    win.sets_tab.calc_btn.click()
    wait_until(lambda: win.sets_tab.running == "NEO", what="NEO running")
    # TST is untouched: its pool builds, the Sets tab itself and the dataset panel stay usable
    paste(win, tst_text, "TST")
    assert pt.build_btn.isEnabled() and pt.lock_note.isHidden()
    pt.fmt_bo3.click()
    pt.build_btn.click()
    wait_until(lambda: pt.result is not None, what="TST build while NEO calculates")
    assert pt.result.set_code == "TST"
    assert win.sets_tab.running == "NEO" and not win.datasets.locked_sets - {"NEO"}
    b = pt.builders[0]
    b.wait_idle()
    assert not b.frozen and b.current is not None
    # NEO is blocked with a note, and a result of NEO that is open is frozen
    paste(win, neo_text, "NEO")
    assert not pt.build_btn.isEnabled() and not pt.score_btn.isEnabled()
    assert pt.lock_note.isVisibleTo(pt) and "NEO</b> is being recalculated — available when it finishes" in pt.lock_note.text()
    win.sets_tab.cancel_all_btn.click() if win.sets_tab.queued() else win.sets_tab.cancel_btn.click()
    wait_until(lambda: idle(win), 30, "NEO done")
    assert pt.build_btn.isEnabled() and pt.lock_note.isHidden()                     # the lock is gone with the calculation
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.002")
    pt.build_btn.click()
    wait_until(lambda: pt.result is not None and pt.result.set_code == "NEO" and idle(win), what="NEO build after it finished")


def test_open_builders_of_a_calculating_set_are_frozen_and_resume(win, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.06")
    build(win)
    b = win.pool_tab.builders[0]
    b.wait_idle()
    fake_api._models["TST"]["stale"] = True
    win.sets_tab.refresh()
    wait_until(lambda: idle(win), what="refresh")
    select_set(win, "TST")
    win.sets_tab.calc_btn.click()
    wait_until(lambda: win.sets_tab.running == "TST", what="running")
    assert b.frozen and "TST is being recalculated" in b.status.text()
    cid = next(c for c in b.deck if not b._is_basic(c))
    seq0 = b._seq
    b.move(cid, "deck")                                          # edits are still possible; the evaluation waits
    assert "being recalculated" in b.status.text()
    time.sleep(0.15)
    QApplication.processEvents()
    assert b._shown_seq < b._seq and b.frozen
    wait_until(lambda: idle(win), 30, "done")
    wait_until(lambda: not b.frozen, what="unfrozen")
    b.wait_idle(5000)
    assert b._shown_seq == b._seq and b._seq > seq0
    assert win.pool_tab.busy_set() == ""


def test_pool_build_holds_back_the_same_sets_calculation(win, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.05")
    fake_api._models["TST"]["stale"] = True
    win.sets_tab.refresh()
    wait_until(lambda: idle(win), what="refresh")
    pt = win.pool_tab
    paste(win, fake_api.sample_pool_text())
    pt.fmt_bo3.click()
    pt.build_btn.click()
    wait_until(lambda: pt.task.running, what="build running")
    assert pt.busy_set() == "TST" and win.sets_tab.pool_busy_set == "TST"
    select_set(win, "TST")
    win.sets_tab.calc_btn.click()
    assert win.sets_tab.running is None and win.sets_tab.queued() == ["TST"]            # waits for the deck builder
    assert "Waiting for the deck builder" in win.sets_tab.progress_label.text()
    wait_until(lambda: win.sets_tab.running == "TST", 30, "calculation starts after the build")
    wait_until(lambda: idle(win), 30, "done")


def test_api_guard_raises_set_busy():
    from sealed_lab.types import SetBusy

    fake_api._calc["TST"] = 1
    try:
        for fn, args in ((fake_api.evaluate_deck, ("TST", {1: 1})), (fake_api.build_pool, ("TST", "1 A")),
                         (fake_api.card_values, ("TST",)), (fake_api.score_deck, ("TST", "1 A"))):
            with pytest.raises(SetBusy, match="TST is being recalculated"):
                fn(*args)
        fake_api.evaluate_deck("NEO", {1: 1})                                      # other sets are not affected
    finally:
        fake_api._calc.clear()


# --------------------------------------------------------------------------- docked card preview

def test_card_preview_is_docked_and_follows_the_mouse(win):
    pt = win.pool_tab
    assert pt.preview.isVisibleTo(pt) and pt.preview.parent() is pt.left
    build(win)
    b = pt.builders[0]
    assert b.preview.parent() is b and not b.preview.isVisibleTo(b)                  # the builder has no preview of its own any more
    cid = next(c for c in b.deck if not b._is_basic(c))
    b._on_hover(cid, "deck")
    assert pt.preview.name == b.info[cid]["name"] and b.info[cid]["name"] in pt.preview.caption.text()
    cid2 = next(iter(b.side()))
    b._on_hover(cid2, "side")
    assert pt.preview.name == b.info[cid2]["name"]
    assert pt.preview.isVisibleTo(pt) and pt.left.sizes()[1] > 100                       # still there while the results are shown
    # hovering a card line of the pasted pool shows that card too
    pt.text.nameHovered.emit("Tide Ghoul (TST) 12")
    pt._on_text_hover("Tide Ghoul")
    assert pt.preview.name == "Tide Ghoul" and "Tide Ghoul" in pt.preview.caption.text()
    pt._on_text_hover("Tide  ghoul")                                                      # spelling variants resolve like the builder's
    assert pt.preview.name == "Tide Ghoul"
    assert pt.preview.art.height() >= pt.preview.art.READABLE_H                            # big enough to read the rules text
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from sealed_lab.gui.widgets import _CardPopup

    QTest.mouseClick(pt.preview.art, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))       # click: the card at full size
    pops = [w for w in pt.preview.art.findChildren(_CardPopup) if w.isVisible()]
    assert len(pops) == 1 and pops[0].name == "Tide Ghoul" and pops[0].width() == _CardPopup.W
    pops[0].close()
    win.tabs.setCurrentWidget(pt)
    grab(win, "pool_preview_docked.png")


def test_paste_edit_reports_the_card_line_under_the_mouse(win):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    pt = win.pool_tab
    pt.text.setPlainText("Deck\n2 Lightning Strike (M21) 159\n1 Fire // Ice\n")
    got = []
    pt.text.nameHovered.connect(got.append)
    rect = pt.text.cursorRect(pt.text.document().findBlockByNumber(1).__class__ and pt.text.textCursor())
    cur = pt.text.textCursor()
    cur.setPosition(pt.text.document().findBlockByNumber(1).position() + 2)
    r = pt.text.cursorRect(cur)
    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(r.center()), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    pt.text.mouseMoveEvent(ev)
    assert got == ["Lightning Strike"] and rect is not None


def test_pool_input_can_be_hidden(win):
    pt = win.pool_tab
    win.tabs.setCurrentWidget(pt)
    assert pt.input_visible()
    pt._toggle_input()
    assert not pt.input_visible() and "Show" in pt.toggle_input.text()
    pt._toggle_input()
    assert pt.input_visible()


# --------------------------------------------------------------------------- Pool tab layout: strip instead of the input, floating preview, banner

def test_pool_input_hides_when_the_result_arrives_and_edit_pool_brings_it_back(win):
    pt = win.pool_tab
    win.tabs.setCurrentWidget(pt)
    assert pt.input_visible() and pt.pool_strip.isHidden() and not pt.toggle_input.isHidden()
    build(win)
    assert not pt.input_visible()                                                    # the result gets the room
    assert pt.pool_strip.isVisibleTo(pt) and pt.toggle_input.isHidden()
    n = sum(pt.result.pool.values())
    assert pt.pool_strip_label.text() == f"TST  ·  Bo3  ·  {n} cards"
    assert pt.edit_pool_btn.text() == "Edit pool" and pt.strip_import_btn.text() == "Import from Arena"
    pt.edit_pool_btn.click()                                                         # the input again; it stays while the user works on it
    assert pt.input_visible() and pt.pool_strip.isHidden() and "Hide" in pt.toggle_input.text()
    QApplication.processEvents()
    assert pt.input_visible() and pt.text.hasFocus() or not win.isActiveWindow()
    # a new result hides it again
    pt.build_btn.click()
    wait_until(lambda: pt.task.running or not pt.input_visible(), what="build start")
    wait_until(lambda: idle(win) and not pt.input_visible(), what="second result")
    assert pt.pool_strip.isVisibleTo(pt)
    # the strip's Import from Arena reads the log like the button of the input does
    pt.strip_import_btn.click()
    wait_until(lambda: idle(win) and pt.import_msg.text().startswith("Imported"), what="import")
    # a score-only result does the same
    pt.edit_pool_btn.click()
    pt.score_btn.click()
    wait_until(lambda: idle(win) and not pt.input_visible(), what="score")
    assert pt.pool_strip.isVisibleTo(pt)


def test_arena_banner_is_on_top_of_the_tab_and_visible_while_the_input_is_hidden(win, tmp_path, monkeypatch):
    pt = win.pool_tab
    win.tabs.setCurrentWidget(pt)
    build(win)
    assert not pt.input_visible()
    add = _watch_log(tmp_path, monkeypatch, pt)
    add("Sealed_TST_20261114", 84)
    wait_until(lambda: not pt.watch_banner.isHidden(), what="banner")
    assert pt.watch_banner.isVisibleTo(pt) and pt.watch_banner.parent() is pt        # not inside the (hidden) input column
    assert pt.watch_banner.geometry().bottom() < pt.split.geometry().top()           # above everything else
    assert "Sealed_TST_20261114" in pt.watch_label.text() and pt.watch_import_btn.isVisibleTo(pt) and pt.watch_dismiss_btn.isVisibleTo(pt)
    assert pt.watch_banner.width() > pt.width() // 2
    pt.watch_import_btn.click()                                                      # Import works from the banner with the input hidden
    wait_until(lambda: pt.text.toPlainText() == fake_api.sample_pool_text() and idle(win), what="import by click")
    assert pt.watch_banner.isHidden()


class _CursorAt:
    """Stand-in for QCursor: the pointer is where the test says (the offscreen platform has no real one)."""
    pos_ = None

    @staticmethod
    def pos():
        return _CursorAt.pos_


def test_floating_card_preview_follows_the_hover_while_the_input_is_hidden(win, monkeypatch):
    from PySide6.QtCore import QPoint
    from sealed_lab.gui import pool_tab
    from sealed_lab.gui.widgets import FloatingCardPreview

    pt = win.pool_tab
    win.tabs.setCurrentWidget(pt)
    monkeypatch.setattr(pool_tab, "QCursor", _CursorAt)
    build(win)
    pop = pt.hover_popup
    assert isinstance(pop, FloatingCardPreview) and not pop.isVisible() and not pt.input_visible()
    b = pt.builders[0]
    cid = next(c for c in b.deck if not b._is_basic(c))
    name = b.info[cid]["name"]
    scr = pt.screen().availableGeometry()
    _CursorAt.pos_ = QPoint(scr.left() + 300, scr.top() + 400)
    b._on_hover(cid, "deck")
    assert pop.isVisible() and pop.card.name == name and name in pop.card.caption.text()     # the image, or the text preview as fallback
    assert 280 <= pop.width() <= 320
    fl = pop.windowFlags()
    assert fl & Qt.WindowType.Tool and fl & Qt.WindowType.FramelessWindowHint and fl & Qt.WindowType.WindowDoesNotAcceptFocus
    assert pop.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating) and not pop.isActiveWindow()
    assert pop.x() >= _CursorAt.pos_.x()                                              # right of the pointer, not under it
    assert scr.contains(pop.geometry())
    # near the right and bottom edges of the screen it flips to the left and stays on the screen
    _CursorAt.pos_ = QPoint(scr.right() - 5, scr.bottom() - 5)
    b._on_hover(next(iter(b.side())), "side")
    assert pop.x() + pop.width() <= _CursorAt.pos_.x() and scr.contains(pop.geometry())
    # the pointer leaves the cards: gone
    b._on_hover(-1)
    assert not pop.isVisible()
    # with the input shown the docked preview does the job: no floating window
    pt.show_input()
    b._on_hover(cid, "deck")
    assert not pop.isVisible() and pt.preview.name == name
    # hiding the input again (button) with the popup showing is not needed; leaving the tab hides it
    pt._toggle_input()
    b._on_hover(cid, "deck")
    assert pop.isVisible()
    win.tabs.setCurrentWidget(win.sets_tab)
    assert not pop.isVisible()


def test_pool_notes_are_one_line_until_clicked(win):
    pt = win.pool_tab
    win.tabs.setCurrentWidget(pt)
    build(win)
    tw = pt.top_warn
    assert tw.isVisibleTo(pt) and tw.toggle is not None and "Pool notes (2)" in tw.toggle.text()
    assert "Unknown Card Name" in tw.toggle.text() and tw.label.isHidden() and not tw.is_expanded()
    h_closed = tw.height()
    tw.toggle.click()
    assert tw.label.isVisibleTo(tw) and tw.is_expanded() and "UG build is weak" in tw.label.text()
    QApplication.processEvents()
    assert tw.height() > h_closed
    tw.toggle.click()
    assert tw.label.isHidden()


def test_builder_toolbar_is_one_row_and_the_headers_are_not_cut(win):
    pt = win.pool_tab
    win.tabs.setCurrentWidget(pt)
    build(win)
    b = pt.builders[0]
    QApplication.processEvents()
    ys = {w.y() for w in (b.group_combo, b.sort_combo, b.split_cb, b.values_cb, b.badge_combo, b.size_slider,
                          b.undo_btn, b.reset_btn, b.rebuild_btn, b.copy_btn)
          if w.window() is win}
    tops = {b.mapFromGlobal(w.mapToGlobal(w.rect().topLeft())).y() for w in (b.group_combo, b.sort_combo, b.badge_combo, b.undo_btn, b.copy_btn)}
    bar = b.toolbar
    items = [bar.itemAt(i) for i in range(bar.count()) if not bar.itemAt(i).isEmpty()]
    need = sum(it.sizeHint().width() for it in items) + bar._spacing * (len(items) - 1)
    # 1400 px: every control on the same row (with Segoe UI it fits); a wider system font (Linux) may wrap, but only when it must
    assert ys and (max(tops) - min(tops) < 12 or need > bar.geometry().width()), (tops, need, bar.geometry().width())
    for t in (b.deck_title, b.side_title):
        assert not t.is_elided(), t.full_text()
    assert b.deck_title.full_text().startswith("Deck (") and b.side_title.full_text().startswith("Sideboard (")
    # a narrow window wraps the toolbar instead of cutting it
    win.resize(1000, 800)
    QApplication.processEvents()
    for t in (b.deck_title, b.side_title):
        assert not t.is_elided(), t.full_text()
    for btn in (b.undo_btn, b.reset_btn, b.rebuild_btn, b.copy_btn):
        assert btn.geometry().right() <= btn.parentWidget().width()
    win.resize(1400, 900)


# --------------------------------------------------------------------------- window geometry, shortcuts

def test_window_geometry_is_saved_and_restored(qapp, data_dir):
    # (the offscreen screen is 800x800 and Qt fits a restored window to the screen: a size and place that fit it; the layout's own
    # minimum, which would grow the window on show, is overridden so that the test sees the restored size itself)
    def window() -> MainWindow:
        w = MainWindow(fake_api, auto_refresh=False)
        w.setMinimumSize(300, 300)
        return w

    fake_api.reset()
    new_settings().clear()
    w = window()
    assert (w.width(), w.height()) == (1400, 900)                                    # nothing saved yet: the default size
    w.show()
    w.resize(700, 640)
    w.move(23, 31)
    QApplication.processEvents()
    pos = w.pos()
    w.close()
    assert new_settings().value("window/geometry") is not None
    w2 = window()
    w2.show()
    QApplication.processEvents()
    assert (w2.width(), w2.height()) == (700, 640) and w2.pos() == pos
    w2.close()
    new_settings().setValue("window/geometry", b"garbage")                           # a damaged value is ignored
    w3 = window()
    assert (w3.width(), w3.height()) == (1400, 900)
    w3.close()
    new_settings().clear()


def test_keyboard_shortcuts(win, monkeypatch):
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest

    pt = win.pool_tab
    win.tabs.setCurrentWidget(pt)
    win.activateWindow()
    wait_until(lambda: "TST" in pt.statuses, what="set known")
    paste(win, fake_api.sample_pool_text())
    pt.fmt_bo3.click()
    # Ctrl+Enter builds (the tooltip says so)
    assert "Ctrl+Enter" in pt.build_btn.toolTip()
    assert {pt.build_shortcut.key().toString(), pt.build_shortcut2.key().toString()} == {"Ctrl+Return", "Ctrl+Enter"}
    pt.text.setFocus()
    QTest.keyClick(pt.text, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    wait_until(lambda: pt.task.running or pt.result is not None, what="build started by Ctrl+Enter")
    wait_until(lambda: idle(win), what="build")
    assert pt.result is not None and pt.text.toPlainText() == fake_api.sample_pool_text()      # no newline typed into the pool
    b = pt.builders[0]
    # Ctrl+Z / Ctrl+Y / Ctrl+Shift+C / Ctrl+H
    assert b.undo_shortcut.key() == QKeySequence("Ctrl+Z") and b.redo_shortcut.key() == QKeySequence("Ctrl+Y")
    assert b.copy_shortcut.key() == QKeySequence("Ctrl+Shift+C") and b.sample_shortcut.key() == QKeySequence("Ctrl+H")
    assert "Ctrl+Z" in b.undo_btn.toolTip() and "Ctrl+Shift+C" in b.copy_btn.toolTip()
    start = dict(b.deck)
    cid = next(iter(b.side()))
    assert b.move(cid, "side")
    moved = dict(b.deck)
    assert moved != start and b.undo_btn.isEnabled()
    b.undo_shortcut.activated.emit()
    assert b.deck == start and "Ctrl+Y" in b.undo_btn.toolTip() or b.deck == start
    b.redo_shortcut.activated.emit()
    assert b.deck == moved
    b.undo()
    assert b.move(next(iter(b.side())), "side") and not b._future                      # a new edit drops the redo stack
    b.redo()
    assert b._future == []
    QApplication.clipboard().setText("")
    b.copy_shortcut.activated.emit()
    assert QApplication.clipboard().text().startswith("Deck\n")
    b.sample_shortcut.activated.emit()
    dlg = b._sample_dlg
    assert dlg.isVisible()
    dlg.close()
    # the keys themselves reach the shortcuts while the builder is on screen
    b.deck_pile.setFocus()
    QApplication.clipboard().setText("")
    QTest.keyClick(b.deck_pile, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    if win.isActiveWindow():
        assert QApplication.clipboard().text().startswith("Deck\n")
        before = dict(b.deck)
        b.move(next(iter(b.side())), "side")
        QTest.keyClick(b.deck_pile, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert b.deck == before
        QTest.keyClick(b.deck_pile, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert b.deck != before
        QTest.keyClick(b.deck_pile, Qt.Key.Key_H, Qt.KeyboardModifier.ControlModifier)
        assert b._sample_dlg.isVisible()
        b._sample_dlg.close()


def test_app_stylesheet_follows_the_theme_and_keeps_the_semantic_colours(qapp, win):
    from sealed_lab.gui.widgets import ACCENT, app_qss, is_dark, semantic

    assert win.tabs.objectName() == "mainTabs" and "QTabWidget#mainTabs" in qapp.styleSheet()
    assert ACCENT in qapp.styleSheet() and "#primary" in qapp.styleSheet()               # one accent: primary buttons, selected tab, focus
    assert qapp.palette().color(qapp.palette().ColorRole.Highlight).name() == ACCENT
    light_green = semantic()["good"].name()
    light_qss = qapp.styleSheet()
    assert light_qss == app_qss(False) and app_qss(True) != app_qss(False)
    assert win.sets_tab.set_table.verticalHeader().defaultSectionSize() >= 30             # comfortable rows in the Sets table
    apply_theme(qapp, "dark")
    try:
        assert is_dark() and qapp.styleSheet() == app_qss(True) and ACCENT in qapp.styleSheet()
        assert semantic()["good"].name() != light_green                                    # the green / red values have their own dark variants
        assert win.tabs.tabBar().sizeHint().height() >= 2 * win.tabs.fontMetrics().height()   # large main tabs
    finally:
        apply_theme(qapp, "light")
    assert qapp.styleSheet() == light_qss


# --------------------------------------------------------------------------- settings & screenshots

def test_settings_remembered(win):
    win.sets_tab.workers_spin.setValue(1)
    win.sets_tab.boot_spin.setValue(32)
    win.sets_tab.fetch_cb.setChecked(False)
    win.sets_tab.advanced_btn.setChecked(True)
    s = new_settings()
    assert int(s.value("workers")) == 1
    assert int(s.value("n_boot")) == 32
    assert str(s.value("fetch_scryfall")).lower() in ("false", "0")
    assert str(s.value("sets/advanced_open")).lower() in ("true", "1")
    assert s.value("last_set") in ("TST", "NEO")
    # reset for the other tests
    win.sets_tab.workers_spin.setValue(os.cpu_count() or 1)
    win.sets_tab.boot_spin.setValue(16)
    win.sets_tab.fetch_cb.setChecked(True)
    win.sets_tab.advanced_btn.setChecked(False)


def test_quick_calculation_checkbox(win):
    st = win.sets_tab
    assert st.quick_cb.isChecked() and st._options()["quick"] is True                       # on by default
    combos = (st.recency_combo, st.attributes_combo, st.manasim_combo, st.extra_combo)
    assert not any(c.isEnabled() for c in combos) and all("Quick calculation" in c.toolTip() for c in combos)
    assert st.cast_combo.isEnabled() and st.matchups_combo.isEnabled()                      # the other settings stay editable
    st.recency_combo.setCurrentIndex(st.recency_combo.findData("on"))                       # a stored choice survives the ticking
    st.quick_cb.setChecked(False)
    assert all(c.isEnabled() for c in combos) and st._options()["quick"] is False and st._options()["recency"] == "on"
    assert str(new_settings().value("quick_calc")).lower() in ("false", "0")
    st.quick_cb.setChecked(True)
    assert str(new_settings().value("quick_calc")).lower() in ("true", "1") and not st.recency_combo.isEnabled()
    assert st.recency_combo.currentData() == "on" and "Quick calculation" in st.recency_combo.toolTip()


def test_screenshots_sets_tab(win):
    calculate(win, "NEO")
    win.sets_tab.log.append_line("Example error trace would appear here.")
    select_set(win, "TST")
    wait_until(lambda: "TST" in win.sets_tab.overviews, what="statistics")
    win.tabs.setCurrentWidget(win.sets_tab)
    grab(win, "sets_tab.png")
    assert (SHOTS / "sets_tab.png").stat().st_size > 5000


def test_report_tab_renders_and_switches_format(win):
    rt = win.report_tab
    assert win.tabs.tabText(2) == "Report" and win.tabs.widget(2) is rt
    win.tabs.setCurrentWidget(rt)
    wait_until(lambda: rt.current is not None and not rt.task.running, what="report")
    assert rt.set_code() == "TST" and rt.current["set_code"] == "TST" and rt.fmt() == rt.current["fmt"]
    assert rt.fmt_bo1.isEnabled() and rt.fmt_bo3.isEnabled() and not rt.message.isVisible()
    from sealed_lab.gui.report_tab import Badge, CardLink, EffectBar

    assert rt.findChildren(EffectBar) and rt.findChildren(Badge) and rt.findChildren(CardLink)
    texts = " ".join(lab.text() for lab in rt.findChildren(QLabel))
    for needle in ("The set at a glance", "Color pairs", "Cards: where players misjudge", "Underrated", "Overrated", "Trap rares: auto-included, but they lose", "Best commons by color",
                   "observed only", "How sure?", "Archetypes: how each color pair is played", "Winning deck shape", "strong results",
                   "same in every archetype", "handles curve and land count through the model"):
        assert needle.lower() in texts.lower(), needle
    # one block per color pair, a pair with a sub-split shows its styles (names, shape words, signature cards); blocks open and close
    from sealed_lab.gui.report_tab import Chip

    names = [lab.text() for lab in rt.findChildren(QLabel)]
    assert any("aggro" in t for t in names) and any("control" in t for t in names) and rt.findChildren(Chip)
    from sealed_lab.gui.report_tab import SpectrumBar
    from sealed_lab.gui.widgets import ElidedLabel

    assert len(rt.findChildren(SpectrumBar)) >= 2                                      # at least two pairs show a style spectrum
    low = texts.lower()
    assert "a spectrum, not separate archetypes" in low and "no clear difference in win rate between the two styles" in low
    assert "points more often than decks at the" in low
    titles = [lab.full_text() for lab in rt.findChildren(ElidedLabel)]
    assert any("+r/g" in t for t in titles) and not any(t.split(" ")[0].endswith("x") and len(t.split(" ")[0]) == 3 for t in titles if t[:2].isupper())
    blocks = [t for t in rt.findChildren(QToolButton) if t.toolTip().startswith("Show or hide")]
    assert len(blocks) == len(rt.current["archetypes"]["pairs"]) >= 5 and not any(t.isChecked() for t in blocks)
    blocks[0].setChecked(True)
    assert blocks[0].arrowType() == Qt.ArrowType.DownArrow and rt.current["archetypes"]["pairs"][0]["key"] in rt._open
    blocks[0].setChecked(False)
    assert not rt._open
    assert "p-value" not in texts.lower() and "confidence interval" not in texts.lower()
    first = rt.fmt()
    other = "bo1" if first == "bo3" else "bo3"
    (rt.fmt_bo1 if other == "bo1" else rt.fmt_bo3).click()
    wait_until(lambda: rt.current is not None and rt.current["fmt"] == other and not rt.task.running, what="other format")
    assert rt._loaded_key == ("TST", other)
    # a card name hovered or clicked goes to the floating preview (there is no docked one on this tab)
    link = rt.findChildren(CardLink)[0]
    link.shown.emit(link.name, link.caption)
    assert rt.hover_popup.card.name == link.name and rt.hover_popup.card.set_code == "TST"
    # a rebuilt model drops the cached reports
    n = len(rt.reports)
    assert n >= 2
    calculate(win, "TST")
    wait_until(lambda: len([k for k in rt.reports if k[0] == "TST"]) < n or rt.task.running or rt.current is not None, what="refresh")
    win.tabs.setCurrentWidget(rt)
    wait_until(lambda: not rt.task.running and rt.current is not None, what="reload")


def test_report_tab_more_detail_blocks_and_card_explorer(win):
    from sealed_lab.gui.report_explorer import CardExplorer
    from sealed_lab.gui.widgets import Collapsible

    rt = win.report_tab
    win.tabs.setCurrentWidget(rt)
    wait_until(lambda: rt.current is not None and not rt.task.running, what="report")
    texts = " ".join(lab.text() for lab in rt.findChildren(QLabel)).lower()
    for needle in ("how the format changed", "best of 1 against best of 3", "about these numbers", "got better", "got worse", "played more by the end",
                   "for reference: the builder handles curve and land count through the model"):
        assert needle in texts, needle
    blocks = {c.toggle.text(): c for c in rt.findChildren(Collapsible)}
    for t in ("More detail: all cards", "More detail: each color pair", "More detail: win rate by land count and creatures", "More detail: both formats side by side"):
        assert t in blocks and not blocks[t].toggle.isChecked() and not blocks[t].content.isVisibleTo(rt), t          # collapsed at first
    blocks["More detail: all cards"].toggle.setChecked(True)
    assert "detail:cards" in rt._open and blocks["More detail: all cards"].content.isVisibleTo(rt)
    ex = rt.findChildren(CardExplorer)[0]
    n = ex.model.rowCount()
    assert n == len(rt.current["cards"]["explorer"]["rows"]) and ex.proxy.rowCount() == n
    # sorted by value, best first; a click on a header re-sorts; the search and the filters narrow the table
    vals = [ex.proxy.data(ex.proxy.index(i, 4), Qt.ItemDataRole.UserRole) for i in range(n)]
    assert vals == sorted(vals, reverse=True)
    ex.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    names = [ex.proxy.data(ex.proxy.index(i, 0)) for i in range(n)]
    assert names == sorted(names, key=str.lower)
    ex.search.setText(names[3][:6])
    assert 1 <= ex.proxy.rowCount() < n and all(names[3][:6].lower() in ex.proxy.data(ex.proxy.index(i, 0)).lower() for i in range(ex.proxy.rowCount()))
    ex.search.setText("")
    ex.group.setCurrentIndex(1)
    g1 = ex.group.currentData()
    assert 0 < ex.proxy.rowCount() < n and all(ex.model.rows[ex.proxy.mapToSource(ex.proxy.index(i, 0)).row()]["group"] == g1 for i in range(ex.proxy.rowCount()))
    ex.group.setCurrentIndex(0)
    assert ex.proxy.rowCount() == n and "of" in ex.count.text()
    # hovering / clicking a row shows the card in the floating preview; leaving the table hides it
    ex.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    ex._show(ex.proxy.index(0, 0))
    assert rt.hover_popup.isVisible() and rt.hover_popup.card.name == names[0]
    QApplication.sendEvent(ex.table.viewport(), QEvent(QEvent.Type.Leave))
    assert not rt.hover_popup.isVisible()
    # the evolution and comparison tables carry a verdict / difference only where the data say so
    for p in rt.current["evolution"]["pairs"]:
        assert p["verdict"] == "stable" or p["change_badge"] in ("likely", "strong")


def test_screenshot_report_tab(win):
    rt = win.report_tab
    win.tabs.setCurrentWidget(rt)
    wait_until(lambda: rt.current is not None and not rt.task.running, what="report")
    QApplication.processEvents()
    grab(win, "report_tab.png")
    assert (SHOTS / "report_tab.png").stat().st_size > 5000


def test_dark_theme_screenshot(qapp, win):
    apply_theme(qapp, "dark")
    try:
        build(win)
        win.tabs.setCurrentWidget(win.pool_tab)
        QApplication.processEvents()
        grab(win, "pool_tab_dark.png")
    finally:
        apply_theme(qapp, "light")


def _drop(widget, paths):
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier)
    widget.dropEvent(ev)  # synthetic drops cannot be routed through Qt's drag manager


def test_drop_on_window(win, tmp_path):
    f1 = tmp_path / "game_data_public.AAA.Sealed.csv.gz"
    f1.write_bytes(b"a" * 100)
    f2 = tmp_path / "game_data_public.BBB.TradSealed.csv.gz"
    f2.write_bytes(b"b" * 100)
    win.tabs.setCurrentWidget(win.pool_tab)
    _drop(win, [f1, f2])  # anywhere on the window -> switches to Sets and registers the files
    wait_until(lambda: idle(win) and win.sets_tab.set_table.rowCount() == 4, what="window drop")
    assert win.tabs.currentWidget() is win.sets_tab
    assert {"AAA", "BBB"} <= {s.set_code for s in win.sets_tab.statuses}


# --------------------------------------------------------------------------- visual deck builder

def test_deck_builder_shows_deck_and_rest_of_pool(win):
    from sealed_lab.gui.deckbuilder import PileView, card_group

    build(win)
    pt = win.pool_tab
    res = pt.result
    assert len(pt.builders) == 3 and pt.tabs.widget(0).tabText(0) == "Deck builder"
    b = pt.builders[0]
    n_deck = sum(c.count for c in res.decks[0].cards)
    assert sum(b.deck.values()) == n_deck == len(b.deck_pile.ids) == 40
    # sideboard = pool minus the non-basic deck cards
    side = b.side()
    assert sum(side.values()) == sum(b.pool.values()) - sum(n for c, n in b.deck.items() if not b._is_basic(c))
    assert sorted(b.side_pile.ids) == sorted(c for c, n in side.items() for _ in range(n))
    # grouping: every card lands in exactly one column, rows split creatures / the rest
    for mode in ("cmc", "color", "type", "rarity"):
        b.group_combo.setCurrentIndex(b.group_combo.findData(mode))
        cols = b.deck_pile.columns()
        assert sum(len(ids) for _, row in cols for _, ids in row) == 40
    b.group_combo.setCurrentIndex(b.group_combo.findData("cmc"))
    b.split_cb.setChecked(True)
    assert [r for r, _ in b.deck_pile.columns()] == ["Creatures", "Non-creatures"]
    assert card_group({"is_land": True}, "cmc")[1] == "Lands"
    assert card_group({"mana_value": 9}, "cmc")[1] == "7+"
    assert card_group({"colors": "WU"}, "color")[1] == "Multicolor"
    assert isinstance(b.deck_pile, PileView) and b.deck_pile.card_at(b.deck_pile._items[0][0].center()) >= 0


def test_pool_tab_archetype_line_links_to_the_analysis_tab(win):
    build(win, fmt="bo1")
    pt = win.pool_tab
    b = pt.builders[0]
    line = b.archetype_line
    line.wait_idle()
    wait_until(lambda: not line.isHidden(), what="archetype line")
    assert line.label.wordWrap() and line.label.text().startswith("Color pair: ")
    assert "Decks in this pair:" in line.label.text() and "the builder does not use archetypes" in line.label.text()
    assert line.link.text() == "See in Report" and line.label.parent() is line and line.parent() is b
    key = line.result["key"]
    assert key in line.label.text() or line.result["code"] in line.label.text()
    # an edit of the deck asks again (the line follows the deck on screen)
    n0 = line._seq
    cid = next(c for c in b.deck if not b._is_basic(c))
    assert b.move(cid, "deck")
    b.wait_idle()
    line.wait_idle()
    assert line._seq > n0
    # the link: Report tab, same set and format, that pair's block opened and scrolled to
    win.tabs.setCurrentWidget(pt)
    line.link.click()
    rt = win.report_tab
    wait_until(lambda: win.tabs.currentWidget() is rt and rt.current is not None and key in rt._pair_widgets
               and rt._pair_widgets[key][1].isChecked() and rt._pending_focus is None, what="report at the pair")
    assert rt.set_code() == "TST" and rt.fmt() == "bo1" and rt.fmt_bo1.isChecked()
    wrap, toggle = rt._pair_widgets[key]
    wait_until(lambda: wrap.mapTo(rt.body, wrap.rect().topLeft()).y() - rt.scroll.verticalScrollBar().value() < 60, what="scrolled to the block")
    assert wrap.isVisibleTo(rt) and key in rt._open


def test_deck_builder_edits_reevaluate(win):
    build(win)
    pt = win.pool_tab
    b = pt.builders[0]
    wr0 = b.current.game_wr
    cid = next(c for c in b.deck if not b._is_basic(c))
    n_side = sum(b.side().values())
    assert b.move(cid, "deck")
    b.wait_idle()
    assert sum(b.deck.values()) == 39 and sum(b.side().values()) == n_side + 1
    assert "39 / 40" in b.counts.text()
    assert b.current.game_wr != wr0 and b.m_delta.value.text().endswith("pp")
    assert "edited" in pt.tabs.tabText(0)
    # back from the sideboard (double-click path), then basics
    b.side_pile.activated.emit(cid)
    b.wait_idle()
    assert sum(b.deck.values()) == 40
    plains = b._basic_ids["W"]
    before = b.deck.get(plains, 0)
    assert b.add_basic("W", +1) and b.deck[plains] == before + 1
    b.undo()
    assert b.deck.get(plains, 0) == before
    b.move(cid, "deck")
    b.reset()
    assert b.deck == b.orig and "edited" not in pt.tabs.tabText(0)
    # Arena export: deck + the rest of the pool as sideboard
    b.copy_btn.click()
    txt = QApplication.clipboard().text()
    assert txt.startswith("Deck\n") and "\nSideboard\n" in txt
    grab(win, "pool_builder.png")


def test_deck_builder_drop_between_piles(win):
    from PySide6.QtCore import QMimeData, QPointF, Qt
    from PySide6.QtGui import QDropEvent

    from sealed_lab.gui.deckbuilder import MIME

    build(win)
    b = win.pool_tab.builders[0]
    cid = next(c for c in b.deck if not b._is_basic(c))
    md = QMimeData()
    md.setData(MIME, f"deck:{cid}".encode())
    ev = QDropEvent(QPointF(20, 20), Qt.DropAction.MoveAction, md, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    b.side_pile.dropEvent(ev)
    assert sum(b.deck.values()) == 39
    md2 = QMimeData()
    md2.setData(MIME, f"side:{cid}".encode())
    b.side_pile.dropEvent(QDropEvent(QPointF(20, 20), Qt.DropAction.MoveAction, md2, Qt.MouseButton.LeftButton,
                                     Qt.KeyboardModifier.NoModifier))
    assert sum(b.deck.values()) == 39      # same-zone drops are ignored


# --------------------------------------------------------------------------- explanation, swaps, locks, compare, Arena import

def _builder(win, k: int = 0):
    build(win)
    b = win.pool_tab.builders[k]
    b.wait_idle(5000, swaps=True)
    wait_until(lambda: b.swaps_ready(), what="swap analysis")
    return b


def test_builder_waterfall_and_swaps(win):
    from sealed_lab.gui.deckbuilder import PileView

    b = _builder(win)
    ex = b.expl
    assert ex is not None and ex.steps[0].kind == "start" and ex.steps[-1].kind == "end"
    assert sum(s.d_pp for s in ex.steps) == pytest.approx(100 * (ex.end_wr - ex.start_wr), abs=1e-9)
    assert len(b.waterfall.steps) == len(ex.steps) and b.waterfall.height() >= len(ex.steps) * b.waterfall.ROW
    assert "→ this deck" in b.why_caption.text()
    assert b.waterfall.tip(1) and "pp" in b.waterfall.tip(1)
    # suggested swaps: at most 3 rows with gain / P(better) and an Apply button each
    sw = b.swaps
    assert sw is not None and sw.complete and sw.suggestions
    assert len(b.swap_list.rows) == len(sw.suggestions) <= 3
    assert "pp" in b.swap_list.rows[0][1].text() and "P(better)" in b.swap_list.rows[0][1].text()
    # hover captions
    out_id = next(iter(sw.deck_best))
    b._on_hover(out_id, "deck")
    cap = b.preview_caption.text()
    assert "best swap" in cap and "value above replacement" in cap and "P(better)" in cap
    in_id = next(iter(sw.side_best))
    b._on_hover(in_id, "side")
    assert "best slot" in b.preview_caption.text()
    # badge modes
    assert b.badge_combo.currentData() == "avg"
    b.badge_combo.setCurrentIndex(b.badge_combo.findData("var"))
    assert b.deck_pile.values[out_id][0] == pytest.approx(sw.var_pp[out_id])
    assert b.side_pile.values[in_id][0] == pytest.approx(sw.side_best[in_id].gain_pp)
    b.badge_combo.setCurrentIndex(b.badge_combo.findData("avg"))
    assert out_id in b.deck_pile.values and b.deck_pile.values[out_id][0] != pytest.approx(sw.var_pp[out_id])
    # applying a suggestion swaps the cards, is one undo step and recomputes everything
    s0 = sw.suggestions[0]
    n_hist = len(b._history)
    wr0 = b.current.game_wr
    b.swap_list.rows[0][2].click()
    assert b.deck.get(s0.in_id, 0) >= 1 and len(b._history) == n_hist + 1 and sum(b.deck.values()) == 40
    assert b.swaps is None                                          # stale analysis is dropped at once
    b.wait_idle(5000, swaps=True)
    wait_until(lambda: b.swaps_ready(), what="swap analysis after apply")
    assert b.current.game_wr != wr0 and b.swaps is not None
    assert b.undo_btn.isEnabled()
    b.undo()
    assert sum(b.deck.values()) == 40 and b.deck == b.orig
    assert isinstance(b.deck_pile, PileView)


def test_builder_swap_analysis_latest_wins_and_never_blocks(win, monkeypatch):
    import threading as th

    b = _builder(win)
    calls = []
    gate = th.Event()
    real = fake_api.swap_analysis

    def slow(set_code, deck, pool, fmt="bo3", locked=None, excluded=None, cancel=None, on_partial=None):
        calls.append(sum(deck.values()))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.4 and not gate.is_set():
            if cancel is not None and cancel.is_set():
                from sealed_lab.types import Cancelled
                raise Cancelled()
            time.sleep(0.005)
        return real(set_code, deck, pool, fmt, locked, excluded, cancel, on_partial)
    monkeypatch.setattr(fake_api, "swap_analysis", slow)
    cid = next(c for c in b.deck if not b._is_basic(c))
    b.move(cid, "deck")                 # starts a slow analysis after the debounce ...
    wait_until(lambda: len(calls) >= 1, what="first analysis started")
    t0 = time.monotonic()
    b.move(cid, "side")                 # ... the next edit cancels it; the GUI thread is never blocked
    assert time.monotonic() - t0 < 0.2
    gate.set()
    b.wait_idle(5000, swaps=True)
    wait_until(lambda: b.swaps_ready(), what="final analysis")
    assert sum(b.deck.values()) == 40 and len(calls) >= 2 and calls[-1] == 40


def test_background_failures_reach_errors_log(win, monkeypatch):
    from sealed_lab.gui import workers

    b = _builder(win)

    def boom(*a, **k):
        raise RuntimeError("swap model broke")
    monkeypatch.setattr(fake_api, "swap_analysis", boom)
    cid = next(c for c in b.deck if not b._is_basic(c))
    b.move(cid, "deck")
    wait_until(lambda: "swap model broke" in workers.error_log_path().read_text(encoding="utf-8"), what="logged swap failure")
    assert "swap analysis TST" in workers.error_log_path().read_text(encoding="utf-8")
    b._swap_worker.cancel()


def test_builder_locks_exclusions_and_rebuild(win, monkeypatch):
    b = _builder(win)
    deck_sp = [c for c in b.deck if not b._is_basic(c)]
    side = [c for c in b.side()]
    lock_id, keep_id = deck_sp[0], deck_sp[1]
    ex_id = side[0]
    # lock: marker on the pile, info label, sideboard card is pulled into the deck
    b.lock_action(lock_id, "lock")
    assert b.locked[lock_id] == b.deck[lock_id] and b.deck_pile.locks == b.locked
    b.lock_action(ex_id, "lock")                                    # a sideboard card: enters the deck and is locked
    assert b.deck.get(ex_id, 0) >= 1 and ex_id in b.locked
    b.lock_action(ex_id, "exclude")                                 # excluding wins over the lock and removes it
    assert ex_id in b.excluded and ex_id not in b.locked and ex_id not in b.deck
    assert b.side_pile.excluded == {ex_id} and "locked" in b.lock_info.text() and "excluded" in b.lock_info.text()
    b._on_hover(lock_id, "deck")
    assert "locked" in b.preview_caption.text()
    # a manual move of an excluded card into the deck clears the exclusion; moving a locked card out clears its lock
    b.move(ex_id, "side")
    assert ex_id not in b.excluded
    b.move(lock_id, "deck", all_copies=True)
    assert lock_id not in b.locked
    b.lock_action(lock_id, "lock")
    assert lock_id in b.deck and b.locked[lock_id] >= 1
    b.exclude_card(keep_id)
    assert keep_id not in b.deck and keep_id in b.excluded
    b.wait_idle(5000, swaps=True)
    # rebuild around the locks in a worker
    b.rebuild_btn.click()
    wait_until(lambda: b._rebuild.running, what="rebuild start")
    assert b.rebuild_btn.text() == "Cancel rebuild"
    wait_until(lambda: not b._rebuild.running, what="rebuild end")
    b.wait_idle(5000, swaps=True)
    assert b.rebuild_btn.text() == "Rebuild around locks"
    assert b.deck.get(lock_id, 0) >= b.locked[lock_id] and keep_id not in b.deck and sum(b.deck.values()) == 40
    assert lock_id in b.locked and keep_id in b.excluded                # locks survive the rebuild
    assert "Rebuilt" in b.copy_msg.text() or "best deck" in b.copy_msg.text()
    # undo restores the previous deck, locks stay
    b.undo()
    assert sum(b.deck.values()) == 40
    b.clear_lock(lock_id)
    b.clear_lock(keep_id)
    assert not b.locked and not b.excluded and not b.lock_info.isVisible()
    # a conflicting build is reported, not crashed
    errs = []
    b.errorOccurred.connect(lambda m, t: errs.append(m))

    def conflict(*a, **k):
        raise ValueError("no color combination can play all the locked cards")
    monkeypatch.setattr(fake_api, "build_from_pool", conflict)
    b.lock_card(deck_sp[2])
    b.rebuild_btn.click()
    wait_until(lambda: errs, what="rebuild error")
    assert "locked cards" in errs[0]
    wait_until(lambda: b.rebuild_btn.isEnabled(), what="button enabled again")


def test_builder_rebuild_can_be_cancelled(win, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.08")
    b = _builder(win)
    b.rebuild_btn.click()
    wait_until(lambda: b._rebuild.running, what="rebuild start")
    deck0 = dict(b.deck)
    b.rebuild_btn.click()                                           # second click = cancel
    wait_until(lambda: not b._rebuild.running, what="rebuild cancelled")
    assert b.status.text() == "Rebuild cancelled." and b.deck == deck0


def test_builder_compare_with_other_builds(win):
    b = _builder(win)
    texts = [b.compare_combo.itemText(i) for i in range(b.compare_combo.count())]
    assert texts[0] == "Generated build" and texts[-1] == "(none)" and len(texts) == 4        # generated, #2, #3, none
    assert texts[1].startswith("#2") and texts[2].startswith("#3")
    assert "Same cards" in b.compare_label.text() and not b.deck_pile.marks
    cid = next(c for c in b.deck if not b._is_basic(c))
    b.move(cid, "deck")
    b.wait_idle(5000)
    assert b.deck_pile.marks == {} and b.side_pile.marks == {cid: "-"}
    assert "only in the generated build" in b.compare_label.text() and "pp" in b.compare_label.text()
    b.compare_combo.setCurrentIndex(1)                              # compare with build #2
    other = {c.card_id: c.count for c in win.pool_tab.result.decks[1].cards}
    add, rem = b.diff_against(other)
    assert add or rem
    assert set(b.deck_pile.marks.values()) <= {"+"} and set(b.side_pile.marks.values()) <= {"-"}
    expected = 100 * (b.current.game_wr - win.pool_tab.result.decks[1].game_wr)
    assert f"{expected:+.2f} pp" in b.compare_label.text()
    b.compare_combo.setCurrentIndex(b.compare_combo.findData("none"))
    assert b.compare_label.text() == "" and not b.deck_pile.marks
    b.compare_combo.setCurrentIndex(0)
    b.lock_card(next(c for c in b.deck if not b._is_basic(c) and c != cid))
    b.exclude_card(next(iter(b.side())))
    b.wait_idle(5000, swaps=True)
    win.tabs.setCurrentWidget(win.pool_tab)
    grab(win, "pool_builder_explain.png")


def test_pool_import_from_arena_log(win):
    pt = win.pool_tab
    wait_until(lambda: "TST" in pt.statuses, what="set known")
    pt.text.setPlainText("keep me")
    for mode, needle in (("not_found", "not found"), ("detailed_off", "Detailed Logs"), ("no_pool", "No sealed pool"),
                         ("unknown_cards", "card ids")):
        fake_api.arena_log_mode = mode
        try:
            pt.import_btn.click()
            wait_until(lambda: not pt.import_task.running and needle in pt.import_msg.text(), what=mode)
        finally:
            fake_api.arena_log_mode = "ok"
        assert pt.text.toPlainText() == "keep me"                   # failures never touch the paste box
    pt.import_btn.click()
    wait_until(lambda: not pt.import_task.running and pt.text.toPlainText() != "keep me", what="import")
    assert pt.text.toPlainText() == fake_api.sample_pool_text()
    assert "Sealed_TST_20261024" in pt.import_msg.text() and "2026-10-24 11:03:43" in pt.import_msg.text()
    assert "Sealed_TST_20261024" in win.sets_tab.log.toPlainText()
    # the imported text is detected and builds
    wait_until(lambda: pt.set_code == "TST" and pt.detector.idle() and not pt.check_task.running, what="detected")
    pt.build_btn.click()
    wait_until(lambda: idle(win) and pt.result is not None, what="build after import")
    assert len(pt.result.decks) == 3


def _watch_log(tmp_path, monkeypatch, pt):
    """Point the Pool tab's Arena watcher at an empty detailed log in a scratch folder and wait for its first (empty) poll."""
    d = tmp_path / "MTGA"
    d.mkdir()
    monkeypatch.setenv("SEALED_LAB_ARENA_LOG_DIR", str(d))
    log = d / "Player.log"
    log.write_text("Mono path[0] = 'C:/x'\nDETAILED LOGS: ENABLED\n", encoding="utf-8")
    pt.watch_timer.setInterval(20)
    wait_until(lambda: pt._watcher is not None and pt._watcher.state == "ok" and not pt.watch_task.running, what="first poll")

    def add(event: str, n: int = 60) -> None:
        import json
        from datetime import datetime

        doc = {"Courses": [{"CourseId": "c", "InternalEventName": event, "CurrentModule": "DeckSelect",
                            "CourseDeckSummary": {"Attributes": []}, "CourseDeck": {"MainDeck": [], "Sideboard": []},
                            "CardPool": list(range(1, n + 1))}]}
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"[UnityCrossThreadLogger]{datetime.now():%Y-%m-%d %H:%M:%S}\n<== EventGetCoursesV2(1)\n{json.dumps(doc)}\n")
    return add


def test_arena_watch_imports_into_empty_box(win, tmp_path, monkeypatch):
    pt = win.pool_tab
    wait_until(lambda: "TST" in pt.statuses, what="set known")
    assert pt.watch_check.isChecked() and pt.text.toPlainText() == ""
    add = _watch_log(tmp_path, monkeypatch, pt)
    add("Sealed_TST_20261024")
    wait_until(lambda: pt.text.toPlainText() == fake_api.sample_pool_text(), what="auto import")
    assert pt.watch_banner.isHidden()
    wait_until(lambda: idle(win), what="import done")


def test_arena_watch_banner_never_overwrites(win, tmp_path, monkeypatch):
    pt = win.pool_tab
    wait_until(lambda: "TST" in pt.statuses, what="set known")
    pt.text.setPlainText("keep me")
    add = _watch_log(tmp_path, monkeypatch, pt)
    add("Sealed_TST_20261024", 84)
    wait_until(lambda: not pt.watch_banner.isHidden(), what="banner")
    assert "Sealed_TST_20261024" in pt.watch_label.text() and "84 cards" in pt.watch_label.text()
    assert pt.text.toPlainText() == "keep me"
    # dismiss: banner goes away, nothing imported, the same pool is not offered again
    pt.watch_dismiss_btn.click()
    assert pt.watch_banner.isHidden()
    for _ in range(10):
        wait_until(lambda: not pt.watch_task.running, what="poll")
        time.sleep(0.03)
    assert pt.watch_banner.isHidden() and pt.text.toPlainText() == "keep me"
    # a second pool: banner, then Import replaces the text
    add("Sealed_TST_20261031", 70)
    wait_until(lambda: not pt.watch_banner.isHidden(), what="second banner")
    assert pt.text.toPlainText() == "keep me"
    pt.watch_import_btn.click()
    wait_until(lambda: pt.text.toPlainText() == fake_api.sample_pool_text(), what="import by click")
    assert pt.watch_banner.isHidden()


def test_arena_watch_setting_off(win, tmp_path, monkeypatch):
    pt = win.pool_tab
    add = _watch_log(tmp_path, monkeypatch, pt)
    pt.watch_check.setChecked(False)
    assert str(new_settings().value("pool/watch_arena")).lower() == "false"
    add("Sealed_TST_20261024")
    for _ in range(10):
        wait_until(lambda: not pt.watch_task.running, what="poll")
        time.sleep(0.03)
    assert pt.text.toPlainText() == "" and pt.watch_banner.isHidden()
    pt.watch_check.setChecked(True)


# --------------------------------------------------------------------------- archetype, synergies, matchup sideboard plan

def test_builder_archetype_synergies_and_sideboard_chips(win):
    b = _builder(win)
    wait_until(lambda: b.plan is not None and b.plan.status == "ok", what="sideboard plan")
    # archetype line
    assert not b.arch_label.isHidden() and "Archetype:" in b.arch_label.text() and "game win rate" in b.arch_label.text()
    assert "signature cards you do not play" in b.arch_label.text()
    # synergy section lists reliable pairs with pp and P; threshold stated
    assert b.syn is not None and b.syn.status == "ok" and not b.syn_box.isHidden()
    assert "pp" in b.syn_label.text() and "P(sign)" in b.syn_label.text()
    some = next(iter(b.syn.by_card))
    b._on_hover(some, "deck" if some in b.deck else "side")
    assert "ynerg" in b.preview_caption.text() or "synergy" in b.preview_caption.text().lower()
    # premium sideboard cards carry an "SB vs XY" chip with a tooltip naming the card to take out
    assert b.side_pile.chips and all(t.startswith("SB vs ") for t in b.side_pile.chips.values())
    cid = next(iter(b.side_pile.chips))
    assert "take out" in b.side_pile.tips[cid] and "P(better)" in b.side_pile.tips[cid]
    b._on_hover(cid, "side")
    assert "SB vs" in b.preview_caption.text()
    # plan section: pick an opponent, apply a swap (one undo step)
    assert not b.plan_box.isHidden() and b.plan_combo.count() == 1 + len(b.plan.opponents)
    b.plan_combo.setCurrentIndex(b.plan_combo.findData("UB"))
    assert b.plan_swaps.rows and "vs UB" in b.plan_swaps.rows[0][1].text()
    sw = b.plan.swaps["UB"][0]
    n_hist = len(b._history)
    b.plan_swaps.rows[0][2].click()
    assert b.deck.get(sw.in_id, 0) >= 1 and len(b._history) == n_hist + 1 and b.plan is None
    b.undo()
    # Bo1: no sideboarding, no section
    b.fmt = "bo1"
    b._refresh_plan_ui()
    assert b.plan_box.isHidden() and not b.side_pile.chips


def test_builder_says_plainly_when_data_do_not_support_matchups(win, monkeypatch):
    monkeypatch.setattr(fake_api, "matchups_mode", "not_supported")
    monkeypatch.setattr(fake_api, "synergy_mode", "off")
    b = _builder(win)
    wait_until(lambda: b.plan is not None, what="sideboard plan")
    assert b.plan.status == "not_supported" and not b.side_pile.chips and not b.plan.swaps
    assert "do not support opponent-specific plans" in b.plan_note.text() and "Nothing is marked" in b.plan_note.text()
    assert not b.plan_combo.isVisibleTo(b) and not b.plan_swaps.rows
    assert b.syn.status == "off" and "not used" in b.syn_label.text()


# --------------------------------------------------------------------------- new 17lands data panel

@pytest.fixture(autouse=True)
def _no_announce_popup(monkeypatch):
    """The 'new 17lands data' box must never block a test: by default it answers 'Later' (tests patch it to look at it)."""
    monkeypatch.setattr(datasets_ui, "ask", lambda parent, title, text: False)


@pytest.fixture()
def fresh_datasets(data_dir):
    def clear():
        fake_api.FAKE_NEW_SET = ""
        fake_api.datasets_items.clear()
        fake_api.datasets_dismissed.clear()
        fake_api.datasets_error = ""
        fake_api.datasets_fail.clear()
    clear()
    yield data_dir.parent
    clear()


def _two_sets_of_items():
    return [fake_api.make_dataset_item("TST", "game", "TradSealed", "updated", 700_000),
            fake_api.make_dataset_item("TST", "replay", "TradSealed", "new", 41_000_000),
            fake_api.make_dataset_item("TST", "draft", "PremierDraft", "new", 900_000_000),
            fake_api.make_dataset_item("NEO", "game", "Sealed", "new", 12_000_000)]


def test_dataset_panel_hidden_when_nothing_new(win, fresh_datasets):
    win.check_datasets(manual=False)
    wait_until(lambda: not win.datasets.check_task.running and win.datasets.check is not None, what="check")
    assert win.datasets.isHidden()
    win.check_datasets(manual=True)
    wait_until(lambda: not win.datasets.check_task.running, what="manual check")
    assert "up to date" in win.statusBar().currentMessage()


def test_dataset_panel_lists_selects_and_downloads(win, fresh_datasets):
    fake_api.datasets_items.extend(_two_sets_of_items())
    win.check_datasets(manual=False)
    wait_until(lambda: win.datasets.isVisible(), what="panel")
    dp = win.datasets
    assert "3 new, 1 updated" in dp.title.text()
    tree = dp.tree
    assert [tree.topLevelItem(i).text(0).split(" ")[0] for i in range(tree.topLevelItemCount())] == ["NEO", "TST"]
    assert "has a model" in tree.topLevelItem(1).text(0)
    tst = tree.topLevelItem(1)
    assert [tst.child(i).text(1) for i in range(3)] == ["Updated", "New", "New"]
    assert tst.child(0).text(2) == "684 KB" and tst.child(0).text(3) == "2026-10-20"
    # draft data (optional) starts unticked
    checked = [it.name for it in dp.selected()]
    assert len(checked) == 3 and not any("draft_data" in n for n in checked)
    assert "3 files" in dp.download_btn.text()
    tst.child(1).setCheckState(0, Qt.CheckState.Unchecked)         # drop the replay file
    assert "2 files" in dp.download_btn.text()
    # download into a chosen folder
    new_settings().setValue("datasets/data_dir", str(fresh_datasets / "mine"))
    fake_api._models["TST"]["stale"] = False
    dp.download_btn.click()
    wait_until(lambda: not dp.dl_task.running and "Downloaded" in dp.result_label.text(), what="download")
    assert (fresh_datasets / "mine" / "TST" / "game_data_public.TST.TradSealed.csv.gz").exists()
    assert (fresh_datasets / "mine" / "NEO" / "game_data_public.NEO.Sealed.csv.gz").exists()
    assert not list((fresh_datasets / "mine").rglob("*.part"))
    assert "Calculate" in dp.result_label.text() and "TST" in dp.result_label.text()
    assert fake_api._models["TST"]["stale"]                          # registered -> stale
    # what was not downloaded is still offered
    left = [dp.tree.topLevelItem(0).child(i).text(0) for i in range(dp.tree.topLevelItem(0).childCount())]
    assert len(left) == 2 and "replay_data_public.TST.TradSealed.csv.gz" in left
    assert all(i.name != "game_data_public.TST.TradSealed.csv.gz" for i in fake_api.datasets_items)


def test_dataset_download_failure_cancel_and_not_now(win, fresh_datasets, monkeypatch):
    fake_api.datasets_items.extend(_two_sets_of_items()[:2])
    fake_api.datasets_fail.add("replay_data_public.TST.TradSealed.csv.gz")
    win.check_datasets(manual=False)
    wait_until(lambda: win.datasets.isVisible(), what="panel")
    dp = win.datasets
    dp.download_btn.click()
    wait_until(lambda: not dp.dl_task.running and dp.result_label.text(), what="download")
    assert "1 failed" in dp.result_label.text() and "simulated failure" in dp.result_label.text()
    assert dp.tree.topLevelItemCount() == 1 and dp.tree.topLevelItem(0).childCount() == 1   # the failed one stays
    # cancel in the middle
    wait_until(lambda: idle(win) and not dp.dl_task.running, what="sets refreshed after the registration")
    fake_api.datasets_fail.clear()
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.05")
    dp.populate(_two_sets_of_items()[:2])
    dp.download_btn.click()
    wait_until(lambda: dp.dl_task.running, what="running")
    assert not dp.download_btn.isEnabled()
    assert win.pool_tab.import_btn.isEnabled()                       # downloading does not lock the Pool tab any more
    dp.strip.cancel_btn.click()
    wait_until(lambda: not dp.dl_task.running, what="cancelled")
    # not now
    dp.later_btn.click()
    assert dp.isHidden()


def test_dataset_download_of_a_set_being_calculated_waits(win, fresh_datasets, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.03")
    fake_api.datasets_items.extend(_two_sets_of_items())
    win.check_datasets(manual=False)
    wait_until(lambda: win.datasets.isVisible(), what="panel")
    dp = win.datasets
    assert dp.download_btn.isEnabled()
    fake_api._models["NEO"] = dict(fake_api._models["TST"])
    select_set(win, "NEO")
    win.sets_tab.calc_btn.click()
    wait_until(lambda: win.sets_tab.running == "NEO", what="NEO running")
    assert dp.locked_sets == {"NEO"} and not dp.download_btn.isEnabled() and "NEO is being recalculated" in dp.download_btn.toolTip()
    for k in range(dp.tree.topLevelItemCount()):                       # only TST files selected: nothing blocks
        top = dp.tree.topLevelItem(k)
        for j in range(top.childCount()):
            if top.text(0).startswith("NEO"):
                top.child(j).setCheckState(0, Qt.CheckState.Unchecked)
    assert dp.download_btn.isEnabled()
    wait_until(lambda: idle(win), 30, "calculation")
    assert dp.locked_sets == set()


def test_dataset_dont_ask_again_and_offline(win, fresh_datasets):
    fake_api.datasets_items.extend(_two_sets_of_items())
    win.check_datasets(manual=False)
    wait_until(lambda: win.datasets.isVisible(), what="panel")
    win.datasets.never_btn.click()
    assert win.datasets.isHidden() and len(fake_api.datasets_dismissed) == 4
    win.check_datasets(manual=False)
    wait_until(lambda: not win.datasets.check_task.running, what="check")
    assert win.datasets.isHidden()                                    # remembered
    win.check_datasets(manual=True)                                   # a manual check shows everything again
    wait_until(lambda: win.datasets.isVisible(), what="panel again")
    win.datasets.hide_panel()
    fake_api.datasets_error = "offline or 17lands unreachable (ConnectionError)"
    win.check_datasets(manual=True)
    wait_until(lambda: "Could not check" in win.statusBar().currentMessage(), what="offline message")
    assert win.datasets.isHidden()


def test_data_menu_and_startup_check(qapp, fresh_datasets):
    fake_api.reset()
    new_settings().clear()
    fake_api.datasets_items.extend(_two_sets_of_items())
    w = MainWindow(fake_api, auto_refresh=True, auto_check=True, check_delay_ms=20)
    w.show()
    try:
        wait_until(lambda: w.datasets.isVisible(), what="startup panel")
        assert [a.text() for a in w.menuBar().actions()[0].menu().actions()] == ["Check for new 17lands data", "17lands data folder…"]
        grab(w, "new_data_panel.png")
    finally:
        w.sets_tab.wait(3000)
        w.pool_tab.wait(3000)
        w.close()
    w2 = MainWindow(fake_api, auto_refresh=False)                       # no auto check without auto_refresh
    w2.show()
    QApplication.processEvents()
    assert w2.datasets.check is None
    w2.close()


def test_changing_the_data_folder_updates_both_panels(win, tmp_path):
    other = tmp_path / "other_root"
    other.mkdir()
    write_dataset(other / "TST", "game_data_public.TST.TradSealed.csv.gz")
    n = len(fake_api.scan_calls)
    win.sets_tab.set_data_dir(str(other))
    wait_until(lambda: len(fake_api.scan_calls) > n and idle(win), what="scan of the new folder")
    assert str(other) in win.sets_tab.folder_label.text() and win.datasets.data_dir() == str(other)
    assert fake_api.scan_calls[-1]["data_dir"] == str(other)
    assert any(f.path.endswith("TradSealed.csv.gz") and str(other) in f.path for f in fake_api.set_status("TST").files)
    win.datasets.set_data_dir(str(tmp_path), notify=True)                  # picked in the other panel: the Sets tab follows
    wait_until(lambda: idle(win), what="scan")
    assert str(tmp_path) in win.sets_tab.folder_label.text()


def test_open_folder_button(win, monkeypatch):
    opened = []
    from sealed_lab.gui import sets_tab as module

    monkeypatch.setattr(module.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    win.sets_tab.open_btn.click()
    assert opened == [fake_api.fake_data_dir] or os.path.normpath(opened[0]) == os.path.normpath(fake_api.fake_data_dir)


# --------------------------------------------------------------------------- basic lands strip, number explanations

def _drop_on(widget, payload: str):
    from PySide6.QtCore import QMimeData, QPointF
    from PySide6.QtGui import QDragEnterEvent, QDropEvent

    from sealed_lab.gui.deckbuilder import MIME

    md = QMimeData()
    md.setData(MIME, payload.encode())
    enter = QDragEnterEvent(QPointF(20, 20).toPoint(), Qt.DropAction.MoveAction, md, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    widget.dragEnterEvent(enter)
    accepted = enter.isAccepted()
    if accepted:
        widget.dropEvent(QDropEvent(QPointF(20, 20), Qt.DropAction.MoveAction, md, Qt.MouseButton.LeftButton,
                                    Qt.KeyboardModifier.NoModifier))
    return accepted


def test_basic_land_strip_and_header_steppers(win):
    from PySide6.QtTest import QTest

    build(win)
    b = win.pool_tab.builders[0]
    b.wait_idle()
    strip = b.basic_strip
    assert strip.isVisibleTo(b)
    assert set(strip.arts) == set("WUBRG") and all(strip.arts[c].cid == b._basic_ids[c] for c in "WUBRG")
    plains = b._basic_ids["W"]
    n0 = b.deck.get(plains, 0)
    assert strip.counts["W"].text() == str(n0) == b._color_rows["W"][2].text()
    assert f" {n0}" in b.steppers.counts["W"].text()
    wr0 = b.current.game_wr
    seq0 = b._seq
    # double-click on the card adds one; the deck is re-evaluated at once
    QTest.mouseDClick(strip.arts["W"], Qt.MouseButton.LeftButton)
    assert b.deck[plains] == n0 + 1 and b._seq > seq0 and strip.counts["W"].text() == str(n0 + 1)
    b.wait_idle()
    assert sum(b.deck.values()) == 41 and b.current.game_wr != wr0 and "41 / 40" in b.counts.text()
    # + / - buttons on the strip and in the deck header
    b.steppers.plus["W"].click()
    assert b.deck[plains] == n0 + 2 and f" {n0 + 2}" in b.steppers.counts["W"].text()
    b.steppers.minus["W"].click()
    strip.minus["W"].click()
    assert b.deck.get(plains, 0) == n0
    # dragging a basic from the strip into the deck adds one (a drop on the sideboard is refused)
    assert not _drop_on(b.side_pile, f"basic:{plains}")
    assert _drop_on(b.deck_pile, f"basic:{plains}")
    assert b.deck.get(plains, 0) == n0 + 1
    # dragging a basic from the deck onto the strip takes it out; a non-basic card is refused
    assert _drop_on(strip, f"deck:{plains}") and b.deck.get(plains, 0) == n0
    spell = next(c for c in b.deck if not b._is_basic(c))
    assert not _drop_on(strip, f"deck:{spell}") and b.deck.get(spell, 0) >= 1
    # the deck pile accepts side / basic cards, the side pile only deck cards
    assert b.deck_pile.accepts == {"side", "basic"} and b.side_pile.accepts == {"deck"}
    # double-click on a basic in the deck removes one copy (as for any card)
    b.add_basic("U", +1)
    nu = b.deck[b._basic_ids["U"]]
    b.deck_pile.activated.emit(b._basic_ids["U"])
    assert b.deck[b._basic_ids["U"]] == nu - 1
    # undo: every move above is one step, all of it comes back and the deck is re-evaluated
    while b._history:
        b.undo()
    assert b.deck == b.orig and "edited" not in win.pool_tab.tabs.tabText(0)
    b.wait_idle()
    assert b.m_delta.value.text() == "+0.0 pp" or abs(b.current.game_wr - b.orig_eval.game_wr) < 0.02
    grab(win, "pool_builder_basics.png")


def test_number_tooltips_in_the_builder(win):
    b = _builder(win)
    tip = b.m_game.toolTip()
    assert "average 17lands pilot" in tip and "50/50" in tip and "80%" in tip and "bootstrap" in tip
    assert b.m_game.value.toolTip() == b.m_game.sub.toolTip() == tip
    assert "avg real deck" in b.m_game.sub.text() and "avg pilot" in b.m_game.caption.text()
    assert b.expl is not None and f"{b.expl.start_wr * 100:.1f}%" in b.m_game.sub.text()
    assert "p² (3 − 2p)" in b.m_match.toolTip() and "p²" in b.m_match.sub.text()
    st = b.m_score.toolTip()
    assert "median real deck" in st and "percentile" in b.m_score.sub.text() and "ranking" in st
    dt = b.m_delta.toolTip()
    assert "generated build" in dt and "percentage points" in dt and "Caution" not in dt
    assert not b.support_box.isVisibleTo(b)
    # counts, curve, color table headers
    assert "Colored mana symbols" in b._color_rows["W"][0].toolTip()
    assert "dual land counts once" in b._color_rows["W"][1].toolTip()
    assert "basic lands of this color" in b._color_rows["W"][2].toolTip()
    assert "40-card" in b.counts.toolTip() and "mana value" in b.curve.toolTip().lower()
    # mana simulation: plain language, units, how to read, regime note
    for key in ("p_screw", "p_flood", "p_color_screw", "mana_spent", "mull_rate"):
        mt = b._mana_labels[key].toolTip()
        assert b._mana_names[key].toolTip() == mt and "Read:" in mt and "% of games" in mt and "Monte-Carlo" in mt
        assert "Bo3 note" in mt and "plain shuffles" in mt
    assert "Bo3" in b.mana_note.text()
    # waterfall rows and the swap rows
    assert "Average build" in b.waterfall.tip(0) or "average real build" in b.waterfall.tip(0)
    assert "log-odds" in b.waterfall.tip(1) and "average spell slot" in b.waterfall.tip(1)
    row = b.swap_list.rows[0][1]
    assert "P(better)" in row.text() and "/16 refits" in row.text()
    assert "bootstrap refits" in row.toolTip() and "not a probability" in row.toolTip()
    # family box: each part has its own tooltip
    tips = b.family_label.tips
    assert {"head", "pbest", "gap", "core", "flex", "why"} <= set(tips)
    assert "1/16" in tips["pbest"] and "coarse" in tips["pbest"]
    assert "tip:pbest" in b.family_label.text()
    assert "Archetype" in b.arch_label.toolTip() and "cosine" in b.arch_label.toolTip()
    assert "P ≥ 90%" in b.syn_label.toolTip() and "Sideboard plan" in b.plan_box.toggle.toolTip()


def test_breakdown_in_percentage_points(win):
    b = _builder(win)
    bars = b.bars
    d = b.current
    assert bars._exact
    assert sum(bars._pp) == pytest.approx(100 * (d.game_wr - 0.5), abs=1e-6)
    assert all(bars.row_text(i).endswith("pp") for i in range(1, len(bars._items)))
    assert bars.row_text(0).startswith("50% →") and bars.label_of("intercept") == "Baseline"
    tp = bars.tip(1)
    assert "log-odds" in tp and "percentage points" in tp
    assert "log-odds" not in b.sections["breakdown"].toggle.text()
    # a breakdown that does not add up falls back to the local slope and says so
    from sealed_lab.gui.widgets import BreakdownBars

    w2 = BreakdownBars()
    w2.set_items({"intercept": 0.2, "cards": 0.3}, 0.6)
    assert not w2._exact and "local slope" in w2.tip(1)


def test_castable_chip_and_support_caution(win):
    b = _builder(win)
    cast = {c.card_id: c.castable for c in b.current.cards if c.castable is not None}
    assert cast and b.deck_pile.castable == cast
    low = [cid for cid, p in cast.items() if p < 0.8]
    assert low, "the fake generated deck has splash cards the mana struggles to cast"
    cid = low[0]
    assert "Castable on time" in b.deck_pile.tip_for(cid) and "warning mark" in b.deck_pile.tip_for(cid)
    b._on_hover(cid, "deck")
    assert "castable on time: <b" in b.preview_caption.text() and f"{cast[cid] * 100:.0f}%" in b.preview_caption.text()
    ok = next(c for c, p in cast.items() if p >= 0.8)
    assert "Castable on time" not in b.deck_pile.tip_for(ok)
    grab(win, "pool_builder_castable.png")
    # a deck outside what real decks look like: the support reasons are shown near the win rate and in the tooltips
    for _ in range(7):
        b.add_basic("W", -1)
        b.add_basic("U", -1)
    b.wait_idle()
    assert b.current.support and "lands" in b.current.support[0]
    assert b.support_box.isVisibleTo(b) and "lands" in b.support_box.lines[0]
    assert "less reliable" in b.m_game.sub.text() and "less reliable" in b.m_delta.sub.text()
    assert "Caution" in b.m_delta.toolTip() and b.current.support[0] in b.m_delta.toolTip()
    assert "does not" in b.m_game.toolTip()
    grab(win, "pool_builder_support.png")
    b.reset()
    b.wait_idle()
    assert not b.support_box.isVisibleTo(b) and "Caution" not in b.m_delta.toolTip()


def test_details_tab_explains_its_numbers(win):
    build(win)
    v = win.pool_tab.views[0]
    assert "average 17lands pilot" in v.m_game.toolTip() and "avg real deck" in v.m_game.sub.text()
    assert "median real deck" in v.m_score.toolTip() and "P(best)" in v.m_pbest.toolTip()
    assert "Read:" in v._mana_labels["p_screw"].toolTip() and "pp" in v.bars_note.text()
    heads = [v.table.horizontalHeaderItem(k).text() for k in range(v.table.columnCount())]
    assert heads == ["Count", "Name", "Value (pp)", "Castable", "Notes"]
    assert any(r[3].endswith("%") for r in v.table.rows_text())
    assert any(r[3].startswith("⚠") for r in v.table.rows_text())          # the splash cards
    assert "Castable" in v.table.horizontalHeaderItem(3).toolTip()


def test_a_deck_outside_real_decks_is_flagged_when_scored(win):
    pt = win.pool_tab
    text = fake_api.sample_deck_text(with_side=True).replace("9 Plains\n8 Island\n", "5 Plains\n5 Island\n")
    paste(win, text)
    pt.fmt_bo3.click()
    pt.score_btn.click()
    wait_until(lambda: pt.result is not None and idle(win), what="score")
    ub = pt.user_builder
    ub.wait_idle()
    assert pt.result.deck.support and "lands" in ub.support_box.lines[0] and ub.support_box.isVisibleTo(ub)
    assert "less reliable" in ub.m_game.sub.text() and "does not" in ub.m_game.toolTip()
    assert pt.user_view.support_box.isVisibleTo(pt.user_view)
    grab(win, "pool_your_deck_support.png")


def test_bo1_regime_note_and_collapsible_sections(win):
    b = _builder(win)
    assert b.sections["family"].toggle.isChecked() and not b.sections["mana"].toggle.isChecked()
    assert "screw" in b.sections["mana"].toggle.text()
    b.sections["mana"].set_expanded(True)
    assert b._mana_labels["p_screw"].isVisibleTo(b)
    b.sections["mana"].set_expanded(False)
    assert not b._mana_labels["p_screw"].isVisibleTo(b)
    b.fmt = "bo1"
    b._smoothing_modelled = False
    b._refresh_mana_tips()
    assert "does <b>not</b> model" in b._mana_labels["p_flood"].toolTip() and "Bo1" in b.mana_note.text()
    b._smoothing_modelled = True
    b._refresh_mana_tips()
    assert "includes that" in b._mana_labels["p_flood"].toolTip()


def test_stop_tracking_a_set(win, data_dir, monkeypatch):
    write_dataset(data_dir / "XYZ", "game_data_public.XYZ.TradSealed.csv.gz")
    st = win.sets_tab
    st.track_set("XYZ")
    wait_until(lambda: idle(win) and "XYZ" in [s.set_code for s in st.statuses], what="tracked")
    select_set(win, "XYZ")
    assert st.untrack_btn.isEnabled()
    monkeypatch.setattr(st, "_confirm_untrack", lambda code, status: False)          # Cancel: nothing changes
    st.untrack_btn.click()
    assert "XYZ" in [s.set_code for s in st.statuses] and not st._io_jobs
    monkeypatch.setattr(st, "_confirm_untrack", lambda code, status: True)
    st.untrack_btn.click()
    wait_until(lambda: idle(win) and "XYZ" not in [s.set_code for s in st.statuses], what="untracked")
    assert row_state(win, "XYZ").endswith("Not tracked") and (data_dir / "XYZ" / "game_data_public.XYZ.TradSealed.csv.gz").exists()
    select_set(win, "XYZ")
    assert not st.untrack_btn.isEnabled()











# --------------------------------------------------------------------------- the Report tab: summary first, then the analysis

def report_ready(win):
    rt = win.report_tab
    win.tabs.setCurrentWidget(rt)
    wait_until(lambda: rt.current is not None and not rt.task.running and rt.summary is not None and rt._loaded_key == (rt.set_code(), rt.fmt()), what="report")
    wait_until(lambda: rt.summary.view.height() > 150 and rt.summary.view.viewport().width() > 300, what="summary laid out")
    return rt


def set_format(rt, fmt):
    (rt.fmt_bo1 if fmt == "bo1" else rt.fmt_bo3).click()
    wait_until(lambda: rt.current is not None and rt.current["fmt"] == fmt and rt._loaded_key == (rt.set_code(), fmt) and not rt.task.running, what=f"format {fmt}")
    wait_until(lambda: rt.summary.view.height() > 150 and rt.summary.view.viewport().width() > 300, what="summary laid out")


def sentence_point(rt, text, off=3):
    """A point on a character of the sentence containing `text` in the summary (document `text` found from its start)."""
    v = rt.summary.view
    return v.sentence_rect_center(v.document().find(text).selectionStart() + off)


def hover(rt, p):
    """Move the pointer onto `p` of the summary (from a corner first: the offscreen pointer keeps its place between tests, and a move to
    where it already is sends nothing)."""
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest

    v = rt.summary.view
    QTest.mouseMove(v.viewport(), QPoint(v.viewport().width() - 3, 2))
    QApplication.processEvents()
    QTest.mouseMove(v.viewport(), p)


def click_sentence(rt, text, off=3):
    from PySide6.QtTest import QTest

    v = rt.summary.view
    p = sentence_point(rt, text, off)
    hover(rt, p)
    QTest.mouseClick(v.viewport(), Qt.MouseButton.LeftButton, pos=p)


def section_top(rt, key):
    from PySide6.QtCore import QPoint

    return rt._secs[key].mapTo(rt.body, QPoint(0, 0)).y()


def test_one_report_tab_with_the_summary_as_first_section(win):
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Sets", "Deck builder", "Report", "Deck search"]
    assert not hasattr(win, "written_tab")
    rt = report_ready(win)
    codes = lambda c: [c.itemData(i) for i in range(c.count())]
    assert codes(rt.set_combo) == ["ALL", "TST"] and rt.set_code() == "TST"            # "All sets" first, never the default
    keys = list(rt._secs)
    assert keys[0] == "summary" and keys[1:4] == ["glance", "pairs", "cards"]
    sec = rt._secs["summary"]
    assert sec.title == "Summary" and rt.summary.parent() is sec and rt.body_lay.itemAt(1).widget() is sec      # (item 0 is the message)
    v = rt.summary.view
    assert "White-Blue is the best pair" in v.toPlainText() and "<!--" not in v.toPlainText()
    assert v.maximumWidth() <= 820 and v.font().pointSizeF() > rt.font().pointSizeF()      # a readable column, a little larger text
    assert v.height() > 150 and v.verticalScrollBar().maximum() == 0                         # as high as its text: the page scrolls, not the view
    assert rt.export_btn.isEnabled() and rt.export_btn.parent() is rt                        # the export button sits in the header
    pos = v.document().find("White-Blue is the best pair").selectionStart() + 1
    from PySide6.QtGui import QTextCursor

    cur = QTextCursor(v.document())
    cur.setPosition(pos)
    assert "White-Blue: +2.2 points" in cur.charFormat().toolTip()                           # the hover tooltip with the figures
    grab(win, "report_tab_summary.png")


def test_summary_outdated_banner_and_missing_report(win):
    rt = report_ready(win)
    p = rt.summary
    assert p.view.isVisibleTo(rt) and not p.banner.isVisibleTo(rt) and not p.message.isVisibleTo(rt)
    fake_api.FAKE_WRITTEN_OUTDATED = True
    try:
        p.load("TST")
        assert p.banner.isVisibleTo(rt) and "Outdated" in p.banner.text()
    finally:
        fake_api.FAKE_WRITTEN_OUTDATED = False
    fake_api.FAKE_WRITTEN_NONE = True
    try:
        p.load("TST")
        assert p.message.isVisibleTo(rt) and "No written report yet" in p.message.text() and not p.view.isVisibleTo(rt)
        assert "summary" in rt._secs                                                          # the analysis below is still there
    finally:
        fake_api.FAKE_WRITTEN_NONE = False
    p.load("TST")
    assert p.view.isVisibleTo(rt)


@pytest.mark.parametrize("sentence, fmt, section, extra", [
    ("playing first wins about", "bo1", "glance", None),
    ("White-Blue is the best pair", "bo1", "pairs", None),
    ("is overrated: players love it", "bo3", "cards", None),
    ("Blue-Black decks range", "bo1", "archetypes", "UB"),
    ("decks with 17 lands", "bo3", "shape", "detail:shape"),
    ("White-Blue decks got better", "bo3", "evolution", None),
    ("The builder plays 0.5 fewer lands", "bo1", "shape", None),
    ("The typical game lasts 9 turns", "bo3", "glance", None),
])
def test_click_on_a_tagged_sentence_jumps_to_its_section(win, sentence, fmt, section, extra):
    from PySide6.QtCore import QPoint

    rt = report_ready(win)
    rt.FLASH_MS = 350
    set_format(rt, "bo3" if fmt == "bo1" else "bo1")                       # the other format: the click has to switch it
    rt.scroll.verticalScrollBar().setValue(0)
    click_sentence(rt, sentence)
    wait_until(lambda: rt._pending_focus is None and rt.fmt() == fmt and rt._loaded_key == ("TST", fmt) and rt._secs[section].is_flashing(), what="jump")
    sec = rt._secs[section]
    assert rt.fmt() == fmt and (rt.fmt_bo1 if fmt == "bo1" else rt.fmt_bo3).isChecked()
    assert rt.nav.currentItem().data(Qt.ItemDataRole.UserRole) == section
    bar = rt.scroll.verticalScrollBar()
    if extra == "UB":                                                      # the color pair's archetype block is opened and scrolled to
        wrap, toggle = rt._pair_widgets["UB"]
        assert toggle.isChecked()
        wait_until(lambda: abs(wrap.mapTo(rt.body, QPoint(0, 0)).y() - bar.value()) < 40, what="scrolled to the block")
    else:
        wait_until(lambda: abs(section_top(rt, section) - bar.value()) < 40, what="scrolled to the section")
    if extra == "detail:shape":
        assert rt._details["shape"].toggle.isChecked()
    assert bar.value() > 0 or section == "glance"
    wait_until(lambda: not sec.is_flashing(), timeout=3, what="highlight gone")      # a short highlight only


def test_click_on_a_sentence_and_hover_shows_a_hand(win):
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest

    rt = report_ready(win)
    fmt = rt.fmt()
    v = rt.summary.view
    p = sentence_point(rt, "The typical game lasts 9 turns")                      # (a Best of 3 sentence)
    hover(rt, p)
    wait_until(lambda: v.viewport().cursor().shape() == Qt.CursorShape.PointingHandCursor, what="hand cursor")
    assert v._hover is not None and v.extraSelections()                            # the sentence under the pointer is highlighted
    QTest.mouseMove(v.viewport(), QPoint(v.viewport().width() - 3, 3))
    wait_until(lambda: v.viewport().cursor().shape() != Qt.CursorShape.PointingHandCursor and v._hover is None, what="no hand")
    other = "bo3" if fmt == "bo1" else "bo1"
    set_format(rt, other)
    n = rt.task.running
    click_sentence(rt, "The typical game lasts 9 turns" if other == "bo3" else "playing first wins about")
    wait_until(lambda: rt._pending_focus is None and rt._secs["glance"].is_flashing(), what="jump")
    assert rt.fmt() == other and not n


def test_a_drag_over_the_summary_selects_text_and_does_not_jump(win):
    from PySide6.QtTest import QTest

    rt = report_ready(win)
    v = rt.summary.view
    a, b = sentence_point(rt, "is overrated: players love it", 2), sentence_point(rt, "is overrated: players love it", 20)
    QTest.mousePress(v.viewport(), Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(v.viewport(), b)
    QTest.mouseRelease(v.viewport(), Qt.MouseButton.LeftButton, pos=b)
    QApplication.processEvents()
    assert v.textCursor().hasSelection() and rt._pending_focus is None and not any(s.is_flashing() for s in rt._secs.values())


def test_section_list_jumps_and_follows_the_scrolling(win):
    import copy

    rt = report_ready(win)
    n = rt.nav.count()
    keys = [rt.nav.item(i).data(Qt.ItemDataRole.UserRole) for i in range(n)]
    assert keys == list(rt._secs) and rt.nav.isVisibleTo(rt)
    shorts = [rt.nav.item(i).text() for i in range(n)]
    assert shorts[:4] == ["Summary", "Set at a glance", "Color pairs", "Cards"] and "About the numbers" in shorts
    assert rt.nav.currentItem().data(Qt.ItemDataRole.UserRole) == "summary" and rt.nav.width() < 220
    bar = rt.scroll.verticalScrollBar()
    # a click on a name jumps there
    for key in ("archetypes", "cards"):
        item = rt.nav.item(keys.index(key))
        rt.nav.itemClicked.emit(item)
        wait_until(lambda: abs(section_top(rt, key) - bar.value()) < 40, what=f"jump to {key}")
        assert rt.nav.currentItem().data(Qt.ItemDataRole.UserRole) == key
    # scrolling moves the highlight
    for key in ("pairs", "shape", "summary"):
        bar.setValue(max(0, section_top(rt, key) + 3))
        wait_until(lambda: rt.nav.currentItem().data(Qt.ItemDataRole.UserRole) == key, what=f"highlight {key}")
    bar.setValue(bar.maximum())
    wait_until(lambda: rt.nav.currentItem().data(Qt.ItemDataRole.UserRole) == keys[-1], what="last section at the bottom")
    # sections that are not there are not listed
    rep = copy.deepcopy(rt.current)
    rep.pop("evolution")
    rep["archetypes"]["pairs"] = []
    rt.current = rep
    rt.render()
    keys2 = [rt.nav.item(i).data(Qt.ItemDataRole.UserRole) for i in range(rt.nav.count())]
    assert "evolution" not in keys2 and "archetypes" not in keys2 and keys2[0] == "summary" and "pairs" in keys2 and keys2 == list(rt._secs)
    # nothing to list without a report
    rt._show_message("x")
    assert rt.nav.count() == 0 and not rt.nav.isVisibleTo(rt)


def test_no_docked_card_preview_and_card_hover_uses_the_floating_one(win):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QEnterEvent
    from sealed_lab.gui.report_tab import CardLink
    from sealed_lab.gui.widgets import CardPreview, FloatingCardPreview

    rt = report_ready(win)
    assert not hasattr(rt, "preview") and not hasattr(rt, "split")
    assert not [w for w in rt.findChildren(CardPreview) if not isinstance(w.parent(), FloatingCardPreview)]
    pop = rt.hover_popup
    assert isinstance(pop, FloatingCardPreview) and not pop.isVisible()
    assert rt.scroll.width() > rt.width() - rt.nav.width() - 60                              # the analysis has the full width
    link = rt.findChildren(CardLink)[0]
    QApplication.sendEvent(link, QEnterEvent(QPointF(3, 3), QPointF(3, 3), QPointF(3, 3)))
    assert pop.isVisible() and pop.card.name == link.name and pop.card.set_code == "TST"
    QApplication.sendEvent(link, QEvent(QEvent.Type.Leave))
    assert not pop.isVisible()
    link.shown.emit(link.name, link.caption)
    assert pop.isVisible()
    rt.scroll.verticalScrollBar().setValue(rt.scroll.verticalScrollBar().value() + 40)         # scrolling away hides it
    assert not pop.isVisible()
    link.shown.emit(link.name, link.caption)
    win.tabs.setCurrentWidget(win.sets_tab)
    assert not pop.isVisible()


def test_card_names_in_the_summary_show_the_card_on_hover(win):
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest

    rt = report_ready(win)
    v = rt.summary.view
    html = v.toHtml()
    assert "card:Tide%20Ghoul" in html and "card:Dawn%20Knight" in html
    pop = rt.hover_popup
    assert not pop.isVisible()
    on_name = sentence_point(rt, "Dawn Knight", 2)
    hover(rt, on_name)
    wait_until(lambda: pop.isVisible() and pop.card.name == "Dawn Knight", what="card preview")
    assert v.viewport().cursor().shape() == Qt.CursorShape.PointingHandCursor
    QTest.mouseMove(v.viewport(), QPoint(v.viewport().width() - 3, on_name.y()))               # off the name: gone
    wait_until(lambda: not pop.isVisible(), what="preview gone")


def test_summary_card_links_cover_the_set_cards_longest_first_and_not_basic_lands(win):
    rt = report_ready(win)
    pat, esc = rt.summary._cards_of("TST")
    names = set(esc.values())
    assert "Tide Ghoul" in names and "Plains" not in names and "Island" not in names                 # basic lands are not linked
    assert pat.match("Tide Bolt of Aldra is good").group(0) == "Tide Bolt of Aldra"
    assert pat.search("Tide Ghouls") is None                                                       # whole words only


def test_weak_rows_are_dimmed_and_folded_behind_a_toggle(win):
    import copy

    from sealed_lab.gui.report_tab import Badge
    from sealed_lab.gui.widgets import ElidedLabel

    from sealed_lab.gui.widgets import Collapsible

    def table_badges(sec):                                                    # (not those of the collapsed "More detail" block)
        def inside_block(w):
            while w is not None and w is not sec:
                if isinstance(w, Collapsible):
                    return True
                w = w.parentWidget()
            return False

        return [b for b in sec.findChildren(Badge) if not inside_block(b)]

    def visible(sec):
        return [b for b in table_badges(sec) if b.isVisibleTo(sec)]

    rt = report_ready(win)
    rep = copy.deepcopy(rt.current)
    rows = [r for r in rep["pairs"]["rows"] if not r["hidden"]]
    n = len(rows)
    assert n >= 8
    # every row weak: the 3 biggest effects stay, the rest folds
    for r in rows:
        r["badge"] = "weak"
    rt._fold_open.clear()
    sec = rt._pairs(rep)
    btn = sec.findChild(QToolButton, "foldWeak")
    assert btn is not None and btn.text() == f"Show {n - 3} more with no clear difference" and not btn.isChecked()
    assert len(visible(sec)) == 3 and len(table_badges(sec)) == n
    assert all(b.graphicsEffect() is not None for b in table_badges(sec))                          # dimmed
    top = sorted(rows, key=lambda r: -r["effect_pp"])[:3]
    vis_names = {w.full_text() for w in sec.findChildren(ElidedLabel) if w.isVisibleTo(sec) and w.full_text() in {r["label"] for r in rows}}
    assert vis_names == {r["label"] for r in top}
    btn.click()
    assert btn.isChecked() and "Hide" in btn.text() and len(visible(sec)) == n
    assert "pairs" in rt._fold_open                                                                  # remembered for the session
    sec2 = rt._pairs(rep)                                                                            # drawn again: still open
    assert len(visible(sec2)) == n and sec2.findChild(QToolButton, "foldWeak").isChecked()
    btn.click()
    assert not btn.isChecked() and len(visible(sec)) == 3 and "pairs" not in rt._fold_open
    # some rows clear: those always show, only the weak ones fold
    rt._fold_open.clear()
    for r in rows[:4]:
        r["badge"] = "strong" if r is rows[0] else "likely"
    sec3 = rt._pairs(rep)
    btn3 = sec3.findChild(QToolButton, "foldWeak")
    assert btn3.text() == f"Show {n - 4} more with no clear difference"
    vis = visible(sec3)
    assert len(vis) == 4 and {b.text() for b in vis} == {"Strong", "Likely"}
    # nothing weak: no toggle
    for r in rows:
        r["badge"] = "likely"
    assert rt._pairs(rep).findChild(QToolButton, "foldWeak") is None
    # the same on the live tab: the deck-shape tables (creatures: two weak rows fold, one Likely row shows)
    rt.render()
    det = rt._details["shape"]
    det.set_expanded(True)
    folds = [b for b in rt.findChildren(QToolButton, "foldWeak")]
    assert any(b.text() == "Show 2 more with no clear difference" for b in folds)


def test_all_sets_summary_and_a_click_jumps_to_the_lesson(win):
    from PySide6.QtGui import QTextCursor

    rt = report_ready(win)
    rt.FLASH_MS = 350
    i = rt.set_combo.findData("ALL")
    rt.set_combo.setCurrentIndex(i)
    rt.set_combo.activated.emit(i)
    wait_until(lambda: rt.cross is not None and rt._loaded_key == ("ALL", "") and rt.summary is not None, what="all sets")
    v = rt.summary.view
    assert v.isVisibleTo(rt) and "playing first did better" in v.toPlainText() and "<!--" not in v.toPlainText()
    cur = QTextCursor(v.document())
    cur.setPosition(v.document().find("playing first did better").selectionStart() + 1)
    assert "clear in 3 of 3 sets" in cur.charFormat().toolTip()
    assert not rt.export_btn.isEnabled() and "not available for all sets" in rt.export_btn.toolTip().lower()
    # a click on a sentence goes to the lesson that holds it (L1: playing first; L3: splash)
    for text, lesson in (("playing first did better", "cross_L1"), ("Splashing gave mixed results", "cross_L3")):
        rt.scroll.verticalScrollBar().setValue(0)
        click_sentence(rt, text)
        wait_until(lambda: rt._pending_focus is None and rt._secs[lesson].is_flashing(), what=f"jump to {lesson}")
        assert rt.nav.currentItem().data(Qt.ItemDataRole.UserRole) == lesson and rt.scroll.verticalScrollBar().value() > 0
        wait_until(lambda: not rt._secs[lesson].is_flashing(), timeout=3, what="highlight gone")
    p = rt.summary
    fake_api.FAKE_WRITTEN_OUTDATED = True
    try:
        p.load("ALL")
        assert p.banner.isVisibleTo(rt) and "Outdated" in p.banner.text()
    finally:
        fake_api.FAKE_WRITTEN_OUTDATED = False
    fake_api.FAKE_WRITTEN_NONE = True
    try:
        p.load("ALL")
        assert p.message.isVisibleTo(rt) and "No written report yet" in p.message.text() and not p.view.isVisibleTo(rt)
    finally:
        fake_api.FAKE_WRITTEN_NONE = False
    j = rt.set_combo.findData("TST")
    rt.set_combo.setCurrentIndex(j)
    rt.set_combo.activated.emit(j)
    wait_until(lambda: rt._loaded_key == ("TST", rt.fmt()) and rt.current is not None, what="back to TST")
    assert rt.export_btn.isEnabled()


# --------------------------------------------------------------------------- "new 17lands data" popup

def _it(code, kind="game", event="PremierDraft", status="new", **kw):
    return fake_api.make_dataset_item(code, kind, event, status, 10_000_000, **kw)


def test_announcements_new_set_and_first_sealed():
    new = [_it("NEWX"), _it("NEWX", event="Sealed"), _it("NEWX", "draft", "PremierDraft")]
    a = datasets_ui.announcements(new, set())
    assert len(a) == 1 and a[0]["reason"] == "new_set" and a[0]["set_code"] == "NEWX"
    assert a[0]["keys"] == ["NEWX:new_set", "NEWX:sealed"] and len(a[0]["items"]) == 2      # optional draft data is not part of it
    assert "new set: NEWX" in datasets_ui.announcement_text(a[0]) and "19 MB" in datasets_ui.announcement_text(a[0])
    known = [_it("TST", event="Sealed", set_in_app=True), _it("TST", "replay", "Sealed", set_in_app=True)]
    b = datasets_ui.announcements(known, set())
    assert [(x["key"], x["reason"], len(x["items"])) for x in b] == [("TST:sealed", "sealed", 1)]
    assert "Sealed data for TST" in datasets_ui.announcement_text(b[0])
    c = datasets_ui.announcements([_it("TST", event="TradSealed", set_in_app=True)], set())
    assert [x["key"] for x in c] == ["TST:sealed"]


def test_announcements_ignored_and_not_repeated():
    sealed = _it("TST", event="Sealed", set_in_app=True)
    assert datasets_ui.announcements([sealed], {"TST:sealed"}) == []
    assert datasets_ui.announcements([_it("NEWX"), _it("NEWX", event="Sealed")], {"NEWX:new_set"}) != []   # sealed key still open
    assert datasets_ui.announcements([_it("NEWX")], {"NEWX:new_set"}) == []
    for code in ("CUBE", "TMTCube", "Y_CUBE", "AB_C"):
        assert datasets_ui.announcements([_it(code), _it(code, event="Sealed")], set()) == []
    assert datasets_ui.announcements([_it("NEWX", status="updated"), _it("NEWX", status="unregistered")], set()) == []
    assert datasets_ui.announcements([_it("TST", event="PremierDraft", set_in_app=True)], set()) == []      # draft-only known set
    assert datasets_ui.announcements([_it("TST", "replay", "Sealed", set_in_app=True)], set()) == []


def test_announce_popup_download_later_and_once(win, fresh_datasets, monkeypatch):
    asked = []
    answer = {"v": False}
    monkeypatch.setattr(datasets_ui, "ask", lambda parent, title, text: (asked.append((parent, title, text)), answer["v"])[1])
    fake_api.FAKE_NEW_SET = "NEWX"
    dp = win.datasets
    win.check_datasets(manual=False)                                  # "Later": everything stays listed
    wait_until(lambda: dp.isVisible() and asked, what="popup")
    assert asked[0][0] is win and asked[0][1] == "New 17lands data" and "NEWX" in asked[0][2]
    assert len(dp.items) == 3 and not dp.dl_task.running
    win.check_datasets(manual=False)                                  # asked only once ever
    wait_until(lambda: not dp.check_task.running, what="second check")
    assert len(asked) == 1
    new_settings().setValue("datasets/announced", [])                 # forget -> ask again, this time "Download"
    answer["v"] = True
    win.check_datasets(manual=False)
    wait_until(lambda: len(asked) == 2, what="second popup")
    wait_until(lambda: not dp.dl_task.running and "Downloaded" in dp.result_label.text(), what="download")
    got = sorted(p.name for p in (Path(dp.data_dir()) / "NEWX").glob("*"))
    assert got == ["game_data_public.NEWX.PremierDraft.csv.gz", "game_data_public.NEWX.Sealed.csv.gz"]       # not the optional draft file


def test_export_button_writes_html_without_blocking(win, tmp_path, monkeypatch):
    from sealed_lab.gui import export_ui

    tab = win.report_tab
    win.tabs.setCurrentWidget(tab)
    wait_until(lambda: tab.set_code() == "TST", what="set chosen")
    target = tmp_path / "out" / "mine.html"
    target.parent.mkdir()
    asked, opened = [], []
    monkeypatch.setattr(export_ui, "get_save_file_name", lambda parent, title, start, flt: (asked.append(start) or str(target), flt))
    monkeypatch.setattr(export_ui, "ask_open", lambda parent, path: True)
    monkeypatch.setattr(export_ui, "open_file", lambda path: opened.append(path) or True)
    assert tab.export_btn.isEnabled()
    tab.export_btn.click()
    assert asked and asked[0].endswith("TST_sealed_report.html")
    html = target.read_text(encoding="utf-8")
    assert "TST Sealed report" in html and "Analysis: Best of 1" in html and "Analysis: Best of 3" in html
    assert opened == [str(target)]
    assert "Exported TST" in win.statusBar().currentMessage()
    # the folder is remembered for the next time; a cancelled dialog writes nothing
    assert new_settings().value("export/folder") == str(target.parent)
    monkeypatch.setattr(export_ui, "get_save_file_name", lambda parent, title, start, flt: (asked.append(start) or "", flt))
    tab.export_btn.click()
    assert asked[-1] == str(target.parent / "TST_sealed_report.html")








def test_all_sets_report_tab_shows_the_lessons(win):
    from PySide6.QtWidgets import QFrame, QLabel

    rt = win.report_tab
    win.tabs.setCurrentWidget(rt)
    wait_until(lambda: not rt.task.running and rt.current is not None, what="report")
    i = rt.set_combo.findData("ALL")
    rt.set_combo.setCurrentIndex(i)
    rt.set_combo.activated.emit(i)
    wait_until(lambda: rt.cross is not None and rt._loaded_key == ("ALL", ""), what="cross-set lessons")
    assert not rt.fmt_bo1.isEnabled() and not rt.fmt_bo3.isEnabled() and not rt.export_btn.isEnabled() and not rt.refresh_btn.isEnabled()
    assert "not available for all sets" in rt.export_btn.toolTip().lower()
    names = {f.objectName() for f in rt.body.findChildren(QFrame) if f.objectName().startswith("cross_")}
    assert names == {"cross_intro"} | {f"cross_L{n}" for n in range(1, 10)}
    assert list(rt._secs) == ["summary", "cross_intro"] + [f"cross_L{n}" for n in range(1, 10)] and rt.nav.count() == 11      # the summary is first here too
    texts = [lab.text() for lab in rt.body.findChildren(QLabel)]
    joined = " ".join(texts)
    assert "Playing first against playing second:" in joined and "Best of 1: PATTERN, playing first did better (clear in 3 of 3)" in joined
    assert "Best of 3: RARE (clear in 1 of 3)" in joined and "MIXED" in joined
    assert "AAA" in texts and "BBB" in texts and "Best of 1" in texts and "Best of 3" in texts           # the set x format table
    assert any(t.startswith("+5.0 Strong") for t in texts) and any("WU +2.0 Likely" in t for t in texts)
    grab(win, "report_all_sets.png")
    j = rt.set_combo.findData("TST")
    rt.set_combo.setCurrentIndex(j)
    rt.set_combo.activated.emit(j)
    wait_until(lambda: rt.current is not None and rt._loaded_key == ("TST", rt.fmt()), what="report of TST")
    assert rt.fmt_bo3.isEnabled() and rt.export_btn.isEnabled() and not rt.body.findChildren(QFrame, "cross_L1")


def test_sample_hand_dialog(win):
    build(win)
    b = win.pool_tab.builders[0]
    b.sample_btn.click()
    dlg = b._sample_dlg
    assert dlg.isVisible() and not dlg.isModal()
    assert len(dlg.widgets) == 7 and len(dlg.deal.library) == 33 and "Opening hand" in dlg.status.text()
    assert all(w.card.name for w in dlg.widgets) and "land" in dlg.summary.text()
    dlg.rb_draw.setChecked(True)                                   # a new hand on the draw
    assert not dlg.deal.on_play and len(dlg.widgets) == 7
    dlg.btn_draw.click()
    assert len(dlg.widgets) == 8 and "Turn 1" in dlg.status.text() and "on the draw" in dlg.status.text()
    dlg.btn_new.click()
    assert len(dlg.widgets) == 7 and dlg.deal.turn == 0
    # London mulligan: 7 new cards, click one, confirm -> 6
    dlg.btn_mull.click()
    assert len(dlg.widgets) == 7 and "Mulligan to 6" in dlg.status.text() and not dlg.btn_confirm.isEnabled()
    dlg.widgets[3].click()
    assert dlg.btn_confirm.isEnabled()
    dlg.btn_confirm.click()
    assert len(dlg.widgets) == 6 and len(dlg.deal.library) == 34 and dlg.deal.to_bottom == 0
    # the deck changes: the next New hand uses it
    cid = next(c for c in b.deck if not b._is_basic(c))
    assert b.move(cid, "deck")
    b.wait_idle()
    dlg.btn_new.click()
    assert len(dlg.widgets) == 7 and len(dlg.deal.library) == 32
    b.sample_btn.click()
    assert b._sample_dlg is dlg
    dlg.close()


# --------------------------------------------------------------------------- the Report tab knows your pool: section, marks, cheat sheet

def pool_names(text: str) -> set[str]:
    import re

    return {m.group(1) for ln in text.splitlines() if (m := re.match(r"^\d+\s+(.+)$", ln.strip()))}


def expected_pool(text: str) -> dict[int, int]:
    """{card_id: copies} of the non-land cards of a pasted fake pool (names the fake universe knows)."""
    import re

    out: dict[int, int] = {}
    by = {d["name"]: i for i, d in fake_api._CARDS.items()}
    for ln in text.splitlines():
        m = re.match(r"^(\d+)\s+(.+)$", ln.strip())
        if m and m.group(2) in by and not fake_api._CARDS[by[m.group(2)]]["is_land"]:
            out[by[m.group(2)]] = out.get(by[m.group(2)], 0) + int(m.group(1))
    return out


def pool_report(win):
    """The Report tab with the report of TST on screen and the sample pool pasted in the Deck builder."""
    rt = report_ready(win)
    paste(win, fake_api.sample_pool_text())
    wait_until(lambda: "pool" in rt._secs and rt._pool_sec is not None, what="the pool section")
    return rt


def test_pool_tab_emits_the_pool_and_the_best_deck(win):
    pt = win.pool_tab
    pools, best = [], []
    pt.poolChanged.connect(lambda code, pool: pools.append((code, dict(pool))))
    pt.bestDeckChanged.connect(lambda *a: best.append(a))
    text = fake_api.sample_pool_text()
    paste(win, text)
    assert pools and pools[-1] == ("TST", expected_pool(text)) and len(pools[-1][1]) == 58
    assert all(not fake_api._CARDS[i]["is_land"] for i in pools[-1][1])                      # basics and lands are not part of it
    n = len(pools)
    paste(win, text.replace("1 Tide Ghoul\n", "3 Tide Ghoul\n", 1))                          # more copies: the new pool is announced
    ghoul = next(i for i, d in fake_api._CARDS.items() if d["name"] == "Tide Ghoul")
    assert len(pools) > n and pools[-1][1][ghoul] == 3 and pools[-1][0] == "TST"
    n = len(pools)
    pt._emit_pool()
    assert len(pools) == n                                                                    # nothing is said twice
    pt.fmt_bo3.click()
    pt.build_btn.click()
    wait_until(lambda: pt.task.running or pt.result is not None, what="build start")
    wait_until(lambda: idle(win), what="build")
    col = "".join(c for c in "WUBRG" if c in pt.result.decks[0].colors)
    assert best and best[-1] == ("TST", "bo3", col)
    pt.text.clear()
    wait_until(lambda: pt.set_code is None, what="cleared")
    assert pools[-1] == ("", {}) and best[-1] == ("", "", "")


def test_for_your_pool_section_appears_only_for_the_pools_set(win):
    rt = report_ready(win)
    from sealed_lab.gui.report_tab import POOL_HINT, CardLink

    assert "pool" not in rt._secs and rt._pool_sec is None
    assert [rt.nav.item(r).text() for r in range(rt.nav.count())][:3] == ["Summary", "Set at a glance", "Color pairs"]
    assert rt.pool_hint is not None and rt.pool_hint.isVisibleTo(rt) and rt.pool_hint.text() == POOL_HINT
    assert rt._secs["summary"].isAncestorOf(rt.pool_hint)
    assert not any(l.owned() for l in rt.findChildren(CardLink))                          # no pool: no marks
    paste(win, fake_api.sample_pool_text())
    wait_until(lambda: rt._pool_sec is not None, what="the pool section")
    keys = list(rt._secs)
    assert keys[:3] == ["summary", "pool", "glance"]
    assert [rt.nav.item(r).text() for r in range(3)] == ["Summary", "Your pool", "Set at a glance"]
    assert rt._secs["pool"].title == "For your pool" and not rt.pool_hint.isVisibleTo(rt)
    ys = [section_top(rt, k) for k in keys[:3]]
    assert ys[0] < ys[1] < ys[2]                                                              # right after the Summary
    assert rt.body_lay.itemAt(1).widget() is rt._secs["summary"] and rt.body_lay.itemAt(2).widget() is rt._secs["pool"]
    # the pool is of TST: another set's report has none of it
    calculate(win, "NEO")
    wait_until(lambda: "NEO" in [rt.set_combo.itemData(i) for i in range(rt.set_combo.count())], what="NEO in the report's sets")
    rt.select_set("NEO")
    wait_until(lambda: rt._loaded_key is not None and rt._loaded_key[0] == "NEO" and not rt.task.running, what="NEO report")
    assert "pool" not in rt._secs and rt._pool_sec is None and rt.pool_hint.isVisibleTo(rt)
    assert not any(l.owned() for l in rt.findChildren(CardLink))
    rt.select_set("TST")
    wait_until(lambda: rt._loaded_key is not None and rt._loaded_key[0] == "TST" and rt._pool_sec is not None, what="TST report with the pool again")
    # All sets: no pool part, no hint
    rt.select_set("ALL")
    wait_until(lambda: rt._loaded_key == ("ALL", "") and not rt.task.running, what="All sets")
    assert "pool" not in rt._secs and rt.pool_hint is None
    # the pool is cleared: the section goes
    rt.select_set("TST")
    wait_until(lambda: rt._pool_sec is not None, what="back")
    win.pool_tab.text.clear()
    wait_until(lambda: rt._pool_sec is None and "pool" not in rt._secs, what="section removed")
    assert rt.pool_hint.isVisibleTo(rt) and [rt.nav.item(r).text() for r in range(rt.nav.count())][:2] == ["Summary", "Set at a glance"]


def test_for_your_pool_section_contents(win):
    from PySide6.QtWidgets import QLabel

    from sealed_lab.gui.widgets import ColorPips, ElidedLabel

    rt = pool_report(win)
    rep, fmt = rt.current, rt.fmt()
    sec = rt._pool_sec
    info = fake_api.card_info("TST")
    vals = fake_api.card_values("TST", fmt)
    pool = expected_pool(fake_api.sample_pool_text())
    # a. color pairs: castable cards (colors within the pair, colorless included) and the summed value of the best 23, in points (beta x 25)
    ps = rt.pool_summary
    want = []
    for pair in ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]:
        cards = [(cid, n) for cid, n in pool.items() if set(info[cid]["colors"]) <= set(pair)]
        vs = sorted((vals[cid]["beta"] * 25.0 for cid, n in cards for _ in range(n)), reverse=True)[:23]
        want.append((pair, sum(n for _c, n in cards), sum(vs)))
    want.sort(key=lambda t: (-t[2], -t[1], t[0]))
    assert [(p["key"], p["n"]) for p in ps["pairs"]] == [(k, n) for k, n, _v in want]
    assert all(abs(p["value"] - v) < 1e-9 for p, (_k, _n, v) in zip(ps["pairs"], want))
    assert [p["value"] for p in ps["pairs"]] == sorted((p["value"] for p in ps["pairs"]), reverse=True)
    labels = {r["key"]: r for r in rep["pairs"]["rows"] if len(r["key"]) == 2}
    shown = [p["key"] for p in ps["pairs"][:4]]
    in_order = [l.full_text() for l in sec.findChildren(ElidedLabel) if l.full_text() in {r["label"] for r in labels.values()}]
    assert in_order == [labels[k]["label"] for k in shown]                                   # the top four, best first
    texts = [l.text() for l in sec.findChildren(QLabel)]
    for k in shown:                                                                           # the whole-set result sits beside each
        assert any(t == f"{labels[k]['effect_pp']:+.1f}".replace("-", "−") for t in texts), k
    assert len(sec.findChildren(ColorPips)) >= 4
    assert not ps["best_pair"] and sec.findChild(QLabel, "poolBest") is None                  # nothing built yet: no "best deck" line
    # b. your cards named in this report
    cs = rep["cards"]
    assert ps["good"] == [r["name"] for r in cs["underrated"] if r["card_id"] in pool]
    trap_ids = {r["card_id"] for r in cs["traps"]}
    assert ps["watch"] == [r["name"] for r in cs["traps"] if r["card_id"] in pool] + \
        [r["name"] for r in cs["overrated"] if r["card_id"] in pool and r["card_id"] not in trap_ids]
    assert ps["good"] and ps["watch"]
    body = " ".join(texts)
    assert "Good news" in body and "Watch out" in body and "Underrated" in body and "Trap rare" in body and "Overrated" in body
    # c. best cards: the eight best by value, without the cards with little data
    mine = sorted(((vals[c]["beta"] * 25.0, info[c]["name"]) for c in pool if not info[c]["is_land"] and not vals[c]["low_data"]), key=lambda t: (-t[0], t[1]))
    assert ps["best"] == [n for _v, n in mine[:8]] and len(ps["best"]) == 8
    assert "left out" in body and "little data" in body
    # the builder's best deck from this pool
    build(win)
    wait_until(lambda: rt.pool_summary["best_pair"] != "", what="best deck line")
    line = rt._pool_sec.findChild(QLabel, "poolBest")
    assert line is not None and "The builder's best deck from this pool is" in line.text()
    col = "".join(c for c in "WUBRG" if c in win.pool_tab.result.decks[0].colors)
    assert labels[col[:2]]["label"] in line.text()
    # a card you do not own is not listed: a pool without the trap rare drops it from "watch out"
    trap = next(r for r in cs["traps"] if r["card_id"] in pool)
    paste(win, fake_api.sample_pool_text().replace(f"1 {trap['name']}\n", "", 1))
    wait_until(lambda: trap["name"] not in rt.pool_summary["watch"], what="updated section")


def test_owned_cards_are_marked_everywhere_in_the_report(win, qapp):
    from PySide6.QtGui import QTextCursor

    from sealed_lab.gui.report_tab import CardLink

    rt = report_ready(win)
    assert rt.findChildren(CardLink) and not any(l.owned() for l in rt.findChildren(CardLink))
    assert "★" not in rt.summary.view.toPlainText() and rt.explorer is not None
    assert "★" not in rt.explorer.model.data(rt.explorer.model.index(0, 0))
    paste(win, fake_api.sample_pool_text())
    wait_until(lambda: rt._pool_sec is not None, what="the pool section")
    pool = expected_pool(fake_api.sample_pool_text())
    owned = {fake_api._CARDS[i]["name"]: n for i, n in pool.items()}
    assert rt._owned == owned
    # cards section, archetype signature and best cards: every card name of the report that you own has the star and the tooltip
    marked = [l for l in rt.findChildren(CardLink) if l.owned()]
    assert marked and all(l.name in owned and l.full_text() == "★ " + l.name and f"In your pool ({owned[l.name]} cop" in l.toolTip() for l in marked)
    assert [l for l in marked if rt._secs["cards"].isAncestorOf(l)]                           # the Cards section
    assert [l for l in marked if rt._secs["archetypes"].isAncestorOf(l)]                      # signature / best cards of an archetype
    outside = [l for l in rt.findChildren(CardLink) if not rt._secs["pool"].isAncestorOf(l)]
    assert all(l.owned() == owned.get(l.name, 0) for l in outside)
    # the explorer table
    ex = rt.explorer
    names = [ex.proxy.data(ex.proxy.index(i, 0)) for i in range(ex.proxy.rowCount())]
    r0 = next(i for i, n in enumerate(names) if n.startswith("★ "))
    nm = names[r0][2:]
    assert nm in owned and f"In your pool ({owned[nm]} cop" in ex.proxy.data(ex.proxy.index(r0, 0), Qt.ItemDataRole.ToolTipRole)
    ex.search.setText(nm)                                                                    # (searching still goes by the plain name)
    assert ex.proxy.rowCount() >= 1
    ex.search.setText("")
    # the summary: stars before the names, the sentences still find their tooltips
    v = rt.summary.view
    assert "★ Tide Ghoul" in v.toPlainText() and "★ Dawn Knight" in v.toPlainText()
    assert len(v.ranges) >= 5 and v.owned == owned
    cur = QTextCursor(v.document())
    cur.setPosition(v.document().find("White-Blue is the best pair").selectionStart() + 1)
    assert "White-Blue: +2.2 points" in cur.charFormat().toolTip()
    # dark theme: still marked after the page is drawn again
    apply_theme(qapp, "dark")
    try:
        wait_until(lambda: rt._pool_sec is not None and not rt._rendering and rt.summary is not None, what="redrawn")
        QApplication.processEvents()
        assert [l for l in rt.findChildren(CardLink) if l.owned()] and "★" in rt.summary.view.toPlainText()
        grab(win, "report_pool_dark.png")
    finally:
        apply_theme(qapp, "light")
    wait_until(lambda: not rt._rendering and rt._pool_sec is not None, what="light again")


def test_pool_changes_update_the_open_report_without_recalculating(win, monkeypatch):
    rt = pool_report(win)
    cur = rt.current
    calls = []
    real = fake_api.set_report
    monkeypatch.setattr(fake_api, "set_report", lambda *a, **k: calls.append(a) or real(*a, **k))
    first = list(rt.pool_summary["best"])
    assert len(first) == 8
    text = fake_api.sample_pool_text()
    lines = text.splitlines()
    small = "\n".join(lines[:1] + lines[1:20])                                                 # a smaller pool
    paste(win, small)
    wait_until(lambda: rt.pool_summary["best"] != first, what="the section follows the pool")
    assert rt.current is cur and not calls and not rt.task.running
    assert set(rt.pool_summary["best"]) <= pool_names(small)
    paste(win, text)
    wait_until(lambda: rt.pool_summary["best"] == first, what="and back")
    assert not calls
    # keeping the place on the page: a pool change while scrolled down does not make the page jump
    rt.focus_section("archetypes")
    wait_until(lambda: rt.scroll.verticalScrollBar().value() > 200, what="scrolled")
    QApplication.processEvents()
    top = section_top(rt, "archetypes") - rt.scroll.verticalScrollBar().value()
    paste(win, small)
    wait_until(lambda: rt.pool_summary["best"] != first, what="again")
    QApplication.processEvents()
    assert abs((section_top(rt, "archetypes") - rt.scroll.verticalScrollBar().value()) - top) < 40


def test_cheat_sheet_button_saves_and_opens_the_page(win, tmp_path, monkeypatch):
    from sealed_lab.gui import export_ui

    rt = report_ready(win)
    assert rt.cheat_btn.text().startswith("Cheat sheet") and rt.cheat_btn.isEnabled() and "one sheet of paper" in rt.cheat_btn.toolTip()
    assert rt.cheat_btn.parent() is rt and rt.cheat_btn.toolTip() != rt.export_btn.toolTip()
    target = tmp_path / "sheet" / "mine.html"
    target.parent.mkdir()
    asked, opened = [], []
    monkeypatch.setattr(export_ui, "get_save_file_name", lambda parent, title, start, flt: (asked.append(start) or str(target), flt))
    monkeypatch.setattr(export_ui, "open_file", lambda path: opened.append(path) or True)
    fmt = rt.fmt()
    rt.cheat_btn.click()
    assert asked and Path(asked[0]).name == f"TST_{fmt}_cheatsheet.html"
    wait_until(lambda: opened and not rt.cheat_task.running, what="cheat sheet saved and opened")
    html = target.read_text(encoding="utf-8")
    assert f"TST Sealed cheat sheet — Best of {fmt[-1]}" in html and "Best color pairs" in html and "@media print" in html
    assert opened == [str(target)] and "Saved the cheat sheet" in win.statusBar().currentMessage()
    assert new_settings().value("export/folder") == str(target.parent)
    # a cancelled dialog writes nothing and opens nothing
    monkeypatch.setattr(export_ui, "get_save_file_name", lambda parent, title, start, flt: (asked.append(start) or "", flt))
    target.unlink()
    rt.cheat_btn.click()
    QApplication.processEvents()
    assert not target.exists() and len(opened) == 1 and asked[-1] == str(target.parent / f"TST_{fmt}_cheatsheet.html")
    # the other format gets its own name and page
    other = "bo1" if fmt == "bo3" else "bo3"
    set_format(rt, other)
    monkeypatch.setattr(export_ui, "get_save_file_name", lambda parent, title, start, flt: (asked.append(start) or str(target), flt))
    rt.cheat_btn.click()
    wait_until(lambda: len(opened) == 2 and not rt.cheat_task.running, what="second page")
    assert Path(asked[-1]).name == f"TST_{other}_cheatsheet.html" and f"Best of {other[-1]}" in target.read_text(encoding="utf-8")


def test_cheat_sheet_button_needs_a_report_of_one_set(win):
    rt = report_ready(win)
    assert rt.cheat_btn.isEnabled()
    rt.select_set("ALL")
    wait_until(lambda: rt._loaded_key == ("ALL", "") and not rt.task.running, what="All sets")
    assert not rt.cheat_btn.isEnabled() and "Not available for All sets" in rt.cheat_btn.toolTip()
    rt.select_set("TST")
    wait_until(lambda: rt._loaded_key == ("TST", rt.fmt()) and rt.cheat_btn.isEnabled(), what="a set again")
    rt._show_message("The report could not be computed (see the error message).")                # no report on screen
    assert not rt.cheat_btn.isEnabled() and "once the report is shown" in rt.cheat_btn.toolTip()


# --------------------------------------------------------------------------- draft mode (draft formats of the Deck builder)

def enable_draft(win: MainWindow, code: str = "TST", calc: bool = True) -> None:
    """Switch "Also build draft models" on in the set's panel (and calculate, which builds the fake draft formats)."""
    select_set(win, code)
    cb = win.sets_tab.panel.draft_cb
    wait_until(lambda: win.sets_tab.panel.draft_box.isVisibleTo(win.sets_tab.panel) and idle(win), what="the draft option in the panel")
    cb.setChecked(True)
    wait_until(lambda: fake_api.get_option(code, "draft_formats") is True and idle(win), what="the option saved")
    if calc:
        win.sets_tab.calc_btn.click()
        wait_until(lambda: idle(win) and "dbo1" in fake_api.set_status(code).formats, what="calculate with draft models")
        wait_until(lambda: "dbo1" in win.pool_tab.statuses[code].formats, what="pool tab status")


def test_draft_option_in_the_set_panel_roundtrips_marks_the_set_pending_and_reaches_the_calculation(win):
    sp = win.sets_tab.panel
    select_set(win, "TST")
    wait_until(lambda: sp.draft_box.isVisibleTo(sp), what="draft box")
    assert not sp.draft_cb.isChecked() and sp.draft_cb.text() == "Also build draft models"
    assert "Fits extra models on the 17Lands Premier Draft and Traditional Draft games" in sp.draft_cb.toolTip()
    assert "Sealed results do not change" in sp.draft_cb.toolTip() and "takes longer" in sp.draft_cb.toolTip()
    assert row_state(win, "TST").endswith("Ready")
    sp.draft_cb.setChecked(True)
    wait_until(lambda: fake_api.get_option("TST", "draft_formats") is True and idle(win), what="option saved")
    wait_until(lambda: "Recalculation pending" in row_state(win, "TST"), what="pending hint")
    assert any("draft models" in p for p in fake_api.set_status("TST").pending)
    select_set(win, "NEO")
    select_set(win, "TST")
    assert sp.draft_cb.isChecked()                                                     # remembered per set
    win.sets_tab.calc_btn.click()
    wait_until(lambda: idle(win) and row_state(win, "TST").endswith("Ready"), what="calculate")
    assert fake_api.calc_options[-1]["draft_formats"] is True
    assert fake_api.set_status("TST").formats == ["bo3", "bo1", "dbo3", "dbo1"]
    sp.draft_cb.setChecked(False)
    wait_until(lambda: fake_api.get_option("TST", "draft_formats") is False and idle(win), what="option off")
    win.sets_tab.calc_btn.click()
    wait_until(lambda: idle(win), what="calculate off")
    assert fake_api.calc_options[-1]["draft_formats"] is False and fake_api.set_status("TST").formats == ["bo3", "bo1"]


def test_deck_builder_lists_draft_formats_only_when_the_model_has_them(win):
    pt = win.pool_tab
    wait_until(lambda: "TST" in pt.statuses, what="set known")
    paste(win, fake_api.sample_pool_text())
    assert not pt.fmt_rows["dbo1"].isVisibleTo(pt) and not pt.fmt_rows["dbo3"].isVisibleTo(pt)
    assert pt.fmt_labels["bo1"].text() == "Bo1 — Arena Sealed: single games, no sideboarding"           # today's labels
    enable_draft(win)
    assert pt.fmt_rows["dbo1"].isVisibleTo(pt) and pt.fmt_rows["dbo3"].isVisibleTo(pt)
    assert pt.fmt_labels["dbo1"].text().startswith("Draft (Bo1)") and pt.fmt_labels["dbo3"].text().startswith("Draft (Bo3)")
    assert pt.fmt_labels["bo1"].text().startswith("Sealed (Bo1)") and pt.fmt_labels["bo3"].text().startswith("Sealed (Bo3)")
    assert pt.fmt() in ("bo1", "bo3")                                                  # the preselection stays Sealed


def test_a_draft_deck_says_draft_everywhere_and_degrades_clearly(win):
    pt = win.pool_tab
    enable_draft(win)
    for fmt, tag in (("dbo1", "Premier Draft"), ("dbo3", "Traditional Draft")):
        build(win, fmt=fmt)
        assert pt.result.fmt == fmt and pt.fmt() == fmt
        v, b = pt.views[0], pt.builders[0]
        assert tag in v.m_game.caption.text() and "Predicted game win rate" in v.m_game.caption.text()
        assert tag in b.m_game.caption.text()
        wait_until(lambda: b.expl is not None, what="explanation")
        assert tag in b.why_caption.text() and tag in b.m_score.sub.text()
        assert (v.m_match.value.text() != "—") == (fmt == "dbo3")                      # Bo1 has no match win rate
        assert "Draft" in pt.pool_strip_label.text()
        if fmt == "dbo3":
            assert b.plan_box.isVisibleTo(b) and "Sealed (Bo3) only" in b.plan_note.text()      # sideboard advice: Sealed only
        else:
            assert not b.plan_box.isVisibleTo(b)
    build(win, fmt="bo1")
    assert pt.views[0].m_game.caption.text() == "Game win rate (avg pilot)"             # Sealed captions are unchanged


def test_sample_hand_of_a_draft_deck_and_its_archetype_line(win):
    pt = win.pool_tab
    enable_draft(win)
    build(win, fmt="dbo3")
    b = pt.builders[0]
    b.open_sample_hand()
    assert "Draft" in b._sample_dlg.windowTitle() and b._sample_dlg.rb_bo3.isChecked()
    b._sample_dlg.close()
    r = fake_api.deck_archetype("TST", "dbo3", {})
    assert r["available"] is False and "Sealed-only" in r["reason"]


def test_report_tab_stays_sealed_only_with_draft_formats_in_the_status(win):
    enable_draft(win)
    rt = win.report_tab
    assert set(win.pool_tab.statuses["TST"].formats) >= {"dbo1", "dbo3"}
    win.tabs.setCurrentWidget(rt)
    wait_until(lambda: rt.current is not None and not rt.task.running, what="report")
    assert rt.fmt() in ("bo1", "bo3") and rt.current["fmt"] in ("bo1", "bo3")
    assert not hasattr(rt, "fmt_dbo1")
    from sealed_lab.gui import pool_tab, report_tab, written_tab

    assert report_tab.FORMATS == written_tab.FORMATS == pool_tab.FORMATS == ("bo1", "bo3")


def test_remembered_draft_format_falls_back_when_the_set_loses_it(win):
    pt = win.pool_tab
    enable_draft(win)
    paste(win, fake_api.sample_pool_text())
    pt.fmt_dbo1.click()
    assert pt.fmt() == "dbo1" and pt._fmt_user == ("TST", "dbo1")
    fake_api._models["TST"]["formats"] = ["bo3", "bo1"]                                 # the model is rebuilt without the draft formats
    win.sets_tab.refresh()
    wait_until(lambda: idle(win) and "dbo1" not in pt.statuses["TST"].formats, what="refresh")
    assert pt.fmt() in ("bo1", "bo3") and pt.fmt_radios[pt.fmt()].isEnabled()
    assert not pt.fmt_rows["dbo1"].isVisibleTo(pt)
    assert pt.fmt_labels["bo1"].text().startswith("Bo1 —")                              # and today's labels again


def test_arena_import_of_a_draft_course_preselects_the_draft_format(win):
    pt = win.pool_tab
    wait_until(lambda: "TST" in pt.statuses, what="set known")
    try:
        fake_api.arena_log_mode = "draft"                                                # no draft models yet: the format is kept
        pt.import_btn.click()
        wait_until(lambda: not pt.import_task.running and pt.text.toPlainText() != "", what="import")
        assert "draft models are not built for TST" in pt.import_msg.text()
        assert pt.fmt() in ("bo1", "bo3")
        pt.text.clear()
        enable_draft(win)
        pt.import_btn.click()
        wait_until(lambda: not pt.import_task.running and pt.text.toPlainText() != "", what="import")
        wait_until(lambda: pt.set_code == "TST" and pt.fmt() == "dbo1", what="Draft (Bo1) preselected")
        assert "Draft (Bo1) selected" in pt.import_msg.text()
        fake_api.arena_log_mode = "draft_trad"
        pt.import_btn.click()
        wait_until(lambda: not pt.import_task.running and pt.fmt() == "dbo3", what="Draft (Bo3) preselected")
    finally:
        fake_api.arena_log_mode = "ok"


def test_every_set_dropdown_lists_the_newest_release_first(win):
    """The owner's rule: set lists are in release order, newest on top (the Deck builder, Report and Deck search tabs too)."""
    from sealed_lab.types import SetStatus

    def st(code):
        return SetStatus(code, [], True, True, False, ["bo1"], {"bo1": 1000}, "2026-11-01T00:00:00")

    win.sets_tab.release = {"AFR": "2021-07-23", "TDM": "2025-04-11", "BLB": "2024-08-02"}
    win.sets_tab.set_statuses([st("AFR"), st("BLB"), st("TDM"), st("ZZZ")])     # alphabetical in, undated last out

    def items(combo):
        return [combo.itemData(i) for i in range(combo.count())]

    want = ["TDM", "BLB", "AFR", "ZZZ"]
    assert [s.set_code for s in win.sets_tab.statuses] == want
    assert items(win.pool_tab.set_combo)[1:] == want                 # (after "choose the set")
    assert items(win.report_tab.set_combo)[1:] == want               # (after "All sets")
    assert items(win.deck_search_tab.set_combo) == want


def test_calculate_tooltip_says_game_data_is_missing(win, monkeypatch):
    """A tracked set with pick (draft) files that are not a Premier Draft one: Calculate stays disabled and its tooltip says a game data file
    is needed (a Premier Draft pick file alone gives a provisional model: tests/test_gui_provisional.py)."""
    import dataclasses

    st = win.sets_tab
    code = st.selected_set() or st.set_table.item(0, 0).text()
    real = fake_api.set_status(code)
    drafts_only = dataclasses.replace(real, files=[dataclasses.replace(f, kind="draft", event="TradDraft") for f in real.files])
    monkeypatch.setattr(st, "status_for", lambda sc: drafts_only)
    monkeypatch.setattr(st, "selected_set", lambda: code)
    st._update_buttons()
    assert not st.calc_btn.isEnabled()
    assert "no 17Lands game data file" in st.calc_btn.toolTip()
    monkeypatch.setattr(st, "status_for", lambda sc: real)
    st._update_buttons()
    assert "Fit the model" in st.calc_btn.toolTip()
