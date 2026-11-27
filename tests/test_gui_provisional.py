"""GUI of the provisional model (a set with 17Lands pick files only): offscreen Qt + the fake backend."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
os.environ["SEALED_LAB_FAKE_API"] = "1"
os.environ["SEALED_LAB_FAKE_DELAY"] = "0.002"
_CFG = tempfile.mkdtemp(prefix="sealed_lab_prov_cfg_")
os.environ.setdefault("SEALED_LAB_SETTINGS", os.path.join(_CFG, "settings.ini"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from sealed_lab import provisional as prov  # noqa: E402
from sealed_lab.gui import fake_api, workers  # noqa: E402
from sealed_lab.gui.app import MainWindow  # noqa: E402
from sealed_lab.gui.widgets import apply_theme, new_settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_theme(app, "light")
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
    old = fake_api.fake_data_dir
    d = tmp_path / "17lands"
    d.mkdir()
    fake_api.fake_data_dir = str(d)
    yield d
    fake_api.fake_data_dir = old


@pytest.fixture()
def make_win(qapp, data_dir, monkeypatch):
    monkeypatch.setattr(workers, "_closing", False)        # a main window closed by an earlier GUI test stops all new background jobs
    made = []

    def make(calculated: bool) -> MainWindow:
        fake_api.reset()
        fake_api.add_pick_only_set("PRV", calculated=calculated)
        new_settings().clear()
        w = MainWindow(fake_api, auto_refresh=True)
        w.sets_tab.rescan_timer.setInterval(30)
        w.show()
        wait_until(lambda: w.sets_tab.set_table.rowCount() == 3 and idle(w), what="initial set list")
        made.append(w)
        return w

    yield make
    for w in made:
        w.sets_tab.wait(3000)
        w.pool_tab.wait(3000)
        w.close()
    QApplication.processEvents()
    monkeypatch.setattr(workers, "_closing", False)


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


# --------------------------------------------------------------------------- Sets tab
def test_calculate_is_enabled_for_a_pick_only_set_and_says_provisional(make_win):
    win = make_win(calculated=False)
    st = win.sets_tab
    select_set(win, "PRV")
    assert st.calc_btn.isEnabled()
    assert "Provisional model from pick data (no game data yet)" in st.calc_btn.toolTip()
    assert "cannot be calculated" not in st.calc_btn.toolTip()
    assert "Provisional" in row_state(win, "PRV")
    assert list(win.pool_tab.statuses) == ["TST"]                    # no model yet: not in the Deck builder's list
    st.calc_btn.click()
    wait_until(lambda: idle(win) and fake_api.set_status("PRV").has_model, what="the provisional calculation")
    assert fake_api.set_status("PRV").provisional
    wait_until(lambda: "Provisional (pick data)" in row_state(win, "PRV"), what="row state")
    assert list(win.pool_tab.statuses) == ["PRV", "TST"] or sorted(win.pool_tab.statuses) == ["PRV", "TST"]
    t = win.sets_tab.set_table
    r = [i for i in range(t.rowCount()) if t.item(i, 0).text() == "PRV"][0]
    assert "built from 17Lands pick data only" in t.item(r, 1).toolTip()


def test_a_pick_file_other_than_premier_draft_cannot_be_calculated(make_win):
    win = make_win(calculated=False)
    with fake_api._lock:
        fake_api._files["PRV"] = [fake_api._fi("/data/draft_data_public.PRV.TradDraft.csv.gz", "draft", "PRV", "TradDraft", "draft", 3.4)]
    win.sets_tab.refresh()
    wait_until(lambda: idle(win), what="refresh")
    select_set(win, "PRV")
    assert not win.sets_tab.calc_btn.isEnabled()
    assert "no 17Lands game data file" in win.sets_tab.calc_btn.toolTip()


def test_panel_of_a_provisional_set_says_so(make_win):
    win = make_win(calculated=True)
    select_set(win, "PRV")
    wait_until(lambda: "Provisional model" in win.sets_tab.panel.model_label.text(), what="the panel's model section")
    assert "pick data only" in win.sets_tab.panel.model_label.text()


def test_game_file_makes_the_provisional_set_pending(make_win, data_dir):
    win = make_win(calculated=True)
    assert "Provisional (pick data)" in row_state(win, "PRV")
    fake_api.add_files([str(_write(data_dir / "PRV", "game_data_public.PRV.Sealed.csv.gz"))])
    win.sets_tab.refresh()
    wait_until(lambda: idle(win) and "Recalculation pending" in row_state(win, "PRV"), what="pending state")
    st = fake_api.set_status("PRV")
    assert st.stale and st.provisional and "game_data_public.PRV.Sealed" in st.pending[0]
    select_set(win, "PRV")
    assert win.sets_tab.calc_btn.isEnabled()
    win.sets_tab.calc_btn.click()
    wait_until(lambda: idle(win) and not fake_api.set_status("PRV").provisional, what="the normal calculation")
    assert "Ready" in row_state(win, "PRV")


def _write(folder: Path, name: str, size: int = 2048) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"x" * size)
    return p


# --------------------------------------------------------------------------- Deck builder
def test_deck_builder_banner_for_a_provisional_set_only(make_win):
    win = make_win(calculated=True)
    pt = win.pool_tab
    assert not pt.prov_banner.isVisibleTo(pt)                                     # no set chosen
    pt._set_current("PRV", "chosen")
    QApplication.processEvents()
    assert pt.prov_banner.isVisibleTo(pt)
    text = pt.prov_label.text()
    assert "Provisional model built from 17Lands pick data only (no game results yet)" in text
    assert "win rates read high" in text and "replaced automatically when game data arrives" in text
    pt._set_current("TST", "chosen")
    QApplication.processEvents()
    assert not pt.prov_banner.isVisibleTo(pt)
    assert pt.trainer_set_combo.findData("PRV") < 0 and pt.trainer_set_combo.findData("TST") >= 0      # the trainer needs real pools


def test_deck_builder_builds_for_a_provisional_set(make_win):
    win = make_win(calculated=True)
    pt = win.pool_tab
    pt._set_current("PRV", "chosen")
    pt.text.setPlainText(fake_api.sample_pool_text())
    pt.show_input()
    wait_until(lambda: pt.build_btn.isEnabled(), what="build button")
    pt.build_btn.click()
    wait_until(lambda: idle(win) and pt.result is not None, what="the build")
    assert pt.prov_banner.isVisibleTo(pt)


# --------------------------------------------------------------------------- Report and Deck search
def test_report_tab_says_not_available_for_a_provisional_set(make_win, monkeypatch):
    win = make_win(calculated=True)
    real = fake_api.set_report

    def guarded(code, *a, **k):
        if code == "PRV":
            pytest.fail("a provisional set has no report to compute")
        return real(code, *a, **k)

    monkeypatch.setattr(fake_api, "set_report", guarded)
    rt = win.report_tab
    win.tabs.setCurrentWidget(rt)
    rt.select_set("PRV")
    wait_until(lambda: rt.message.isVisible(), what="the message")
    assert rt.message.text() == prov.UNAVAILABLE
    assert "Not available yet: this set has only pick data. The Deck builder works with a provisional model." in rt.message.text()
    assert not rt.refresh_btn.isEnabled()
    rt.select_set("TST")                                                             # a normal set still loads its report
    wait_until(lambda: not rt.message.isVisible() and rt.current is not None, what="the normal report")


def test_deck_search_lists_the_provisional_set_with_a_plain_message(make_win, monkeypatch):
    win = make_win(calculated=True)
    monkeypatch.setattr(fake_api, "search_decks", lambda *a, **k: pytest.fail("no deck search on a provisional set"))
    monkeypatch.setattr(fake_api, "deck_search_cards", lambda *a, **k: pytest.fail("no card names of a provisional set"))
    dt = win.deck_search_tab
    assert dt.set_combo.findData("PRV") >= 0                                         # the set list includes it ...
    win.tabs.setCurrentWidget(dt)
    i = dt.set_combo.findData("PRV")
    dt.set_combo.setCurrentIndex(i)
    dt._apply_set()
    wait_until(lambda: dt.message.isVisible(), what="the message")
    assert dt.message.text().startswith("Not available yet: this set has only pick data.")      # ... with a plain message
    assert not dt.search_btn.isEnabled()
    QApplication.processEvents()
    assert not dt.task.running
