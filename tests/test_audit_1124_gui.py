"""Audit 1124, GUI side: closing, page leaks, model-change staleness, report / deck-search edge cases, image store, MILP time limit."""
from __future__ import annotations

import os
import tempfile
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["SEALED_LAB_FAKE_API"] = "1"
os.environ.setdefault("SEALED_LAB_SETTINGS", os.path.join(tempfile.mkdtemp(prefix="sealed_lab_aud_cfg_"), "settings.ini"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sealed_lab.gui import fake_api, workers  # noqa: E402
from sealed_lab.gui.widgets import apply_theme, new_settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_theme(app, "light")
    return app


@pytest.fixture(autouse=True)
def _clean(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    new_settings().clear()
    fake_api.reset()
    workers.set_closing(False)
    yield
    workers.set_closing(False)
    QApplication.processEvents()


def wait_until(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        QApplication.processEvents()
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("timeout")


# ---------------------------------------------------------------------------------------------- E2 closing

def test_closing_refuses_new_work_and_detaches_a_stuck_thread():
    tc = workers.TaskController()
    release = threading.Event()
    assert tc.run(lambda progress, cancel: release.wait(10), label="stuck")      # ignores the cancel event
    w = tc._worker
    tc.shutdown(50)                                                            # still running after the wait: detached, not destroyed
    assert not tc.running and w in workers._detached and w.parent() is None and w.isRunning()
    release.set()
    assert w.wait(5000)
    workers._detached.remove(w)
    workers.set_closing(True)
    assert tc.run(lambda progress, cancel: 1) is False
    workers.set_closing(False)


def test_main_window_close_stops_timers_and_controllers():
    from sealed_lab.gui.app import MainWindow

    w = MainWindow(fake_api, auto_refresh=False, auto_check=False)
    w.show()
    release = threading.Event()
    rebuild = workers.TaskController(w.pool_tab)                      # like a Deck builder's _rebuild controller
    rebuild.run(lambda progress, cancel: cancel.wait(10) or release.set(), label="rebuild")
    assert rebuild.running
    w.close()
    assert workers.is_closing()
    assert not w.pool_tab.watch_timer.isActive()
    wait_until(lambda: not rebuild.running)                                   # cancelled and waited for
    assert not workers._detached or all(not t.isRunning() for t in workers._detached)
    assert rebuild.run(lambda progress, cancel: 1) is False                   # no new work while closing


def test_calc_ended_does_not_refresh_while_closing():
    from sealed_lab.gui.sets_tab import SetsTab

    t = SetsTab(fake_api)
    calls = []
    t.refresh = lambda: calls.append(1)
    t.running = "TST"
    t._on_calc_ended()
    assert calls == [1]
    t.running = "TST"
    workers.set_closing(True)
    t._on_calc_ended()
    assert calls == [1]
    t.wait(100)


# ---------------------------------------------------------------------------------------------- E3 / B5 pool tab

def _pool_tab():
    from sealed_lab.gui.pool_tab import PoolTab

    t = PoolTab(fake_api)
    t.refresh_sets(fake_api.list_sets())
    t._set_current("TST", "chosen")
    return t


def test_reset_results_deletes_pages_and_stops_builders():
    t = _pool_tab()
    t.text.setPlainText(fake_api.sample_pool_text())
    t.build()
    wait_until(lambda: t.result is not None and not t.task.running)
    assert t.tabs.count() >= 1 and t.builders
    pages = [t.tabs.widget(i) for i in range(t.tabs.count())]
    stopped = []
    for b in t.builders:
        b.shutdown = lambda b=b: stopped.append(b)
    t._reset_results()
    assert len(stopped) >= 1 and t.tabs.count() == 0
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    from shiboken6 import isValid
    assert all(not isValid(p) for p in pages)
    t.wait(2000)


def test_model_rebuilt_marks_result_stale_and_reloads_card_info():
    t = _pool_tab()
    t.text.setPlainText(fake_api.sample_pool_text())
    t.build()
    wait_until(lambda: t.result is not None and not t.task.running)
    assert not t.stale_banner.isVisibleTo(t)
    loads = []
    orig = t._load_set_info
    t._load_set_info = lambda: (loads.append(1), orig())[1]
    fake_api._models["TST"]["built_at"] = "2026-11-01 10:00"                    # a recalculation finished
    t.refresh_sets(fake_api.list_sets())
    assert loads == [1]
    assert t.stale_banner.isVisibleTo(t) and "model" in t.stale_label.text()
    t.refresh_sets(fake_api.list_sets())                                       # nothing changed again: no second reload
    assert loads == [1]
    t.wait(2000)


def test_watcher_offer_passes_its_course_key_to_the_import():
    t = _pool_tab()
    seen = []

    def import_arena_log(set_code=None, log_dir=None, course_key=None):
        seen.append(course_key)
        return fake_api.import_arena_log(set_code)

    t.api = type("A", (), {"__getattr__": lambda self, n: getattr(fake_api, n), "import_arena_log": staticmethod(import_arena_log)})()
    t._watch_offer = dict(key=("Sealed_X", (1, 2)), event="Sealed_X", n=2)
    t._watch_import()
    wait_until(lambda: not t.import_task.running)
    assert seen == [("Sealed_X", (1, 2))]
    t.wait(2000)


# ---------------------------------------------------------------------------------------------- E4 report / deck search

def test_report_pool_data_failure_is_not_retried_and_foreign_error_is_not_shown():
    from sealed_lab.gui.report_tab import ReportTab

    calls = []

    class Api:
        def __getattr__(self, n):
            return getattr(fake_api, n)

        def card_info(self, sc):
            calls.append(sc)
            raise RuntimeError("boom")

    t = ReportTab(Api())
    t.refresh_sets(fake_api.list_sets())
    t.show()
    runs = []
    t.pool_task.started.connect(lambda: runs.append(1))
    wait_until(lambda: t.current is not None)
    t.set_pool("TST", {1: 2})
    wait_until(lambda: not t.pool_task.running and runs)
    n = len(runs)
    for _ in range(10):
        QApplication.processEvents()
        time.sleep(0.01)
    t._ensure_pool_data()
    wait_until(lambda: not t.pool_task.running)
    assert len(runs) == n == 1                                                # remembered: no endless retry
    # an error of a report started for another key is not shown as the current page's message
    t._task_key = ("OTHER", "bo3")
    t._show_message("")
    t.message.setVisible(False)
    t._on_error("Traceback...\nRuntimeError: x")
    assert not t.message.isVisible() or "could not be computed" not in t.message.text()
    t.wait(2000)


def test_show_more_with_a_pending_filter_change_runs_the_new_search():
    from sealed_lab.gui.deck_search_tab import DeckSearchTab

    t = DeckSearchTab(fake_api)
    t.refresh_sets(fake_api.list_sets())
    t.searched = ("TST", "x")                                                  # a search is on screen
    t.card_edit.setText("x")
    t.debounce.start()                                                         # the filter was just changed
    t.request_search("more")
    assert t._pending == "new" and not t.debounce.isActive()
    t.wait(2000)


# ---------------------------------------------------------------------------------------------- E5 fakes, images

def test_fake_api_has_the_isolated_calc_and_the_cosmetic_calls():
    import inspect

    from sealed_lab import api

    for name in ("calculate_set_isolated", "calibration_note", "prefetch_images", "set_release_dates"):
        fake, real = getattr(fake_api, name), getattr(api, name)
        fp, rp = inspect.signature(fake).parameters, inspect.signature(real).parameters
        assert [p for p in rp if p != "options"] == [p for p in fp if p != "options"], name
    st = fake_api.calculate_set_isolated("NEO", n_jobs=1, n_boot=4)
    assert st.set_code == "NEO" and st.has_model


def test_image_store_is_bounded_per_set_and_keeps_one_thread_alive_correctly():
    from sealed_lab.gui import deckbuilder as D

    class Api:
        def card_image(self, set_code, name, fetch=True):
            return None

    store = D.ImageStore(Api())
    img = QImage(4, 4, QImage.Format.Format_RGB32)
    img.fill(QColor("red"))
    for i in range(D.MAX_IMAGES + 25):
        store._on_loaded("TST", f"Card {i}", img)
    assert len(store._images) == D.MAX_IMAGES and ("TST", "Card 0") not in store._images
    store._on_loaded("NEO", "Plains", img)
    assert store.pixmap("NEO", "Plains", 20) is not None and store.pixmap("TST", "Plains", 20) is None   # keyed by set too
    # an item queued after the idle thread decided to exit starts a new thread
    store._thread = None
    store._q.put(("TST", "Late"))
    store._ensure_thread()
    wait_until(lambda: store._q.empty())
    store.pixmap("TST", "Zed", 10)
    wait_until(lambda: ("TST", "Zed") in store._missing)


# ---------------------------------------------------------------------------------------------- MILP time limit

def test_milp_time_limit_is_logged_and_flagged(monkeypatch, caplog):
    from types import SimpleNamespace

    from sealed_lab.deck import milp as M

    spec = M.MilpSpec(coef=np.array([1.0]), copies=np.array([1]), is_land=np.array([False]), produces=np.zeros((1, 5), dtype=bool),
                      splash=np.array([False]), basic_colors=[0], floors={}, n_lands=1, deck_size=2)
    real = M.milp
    assert M.solve_milp(spec)[0].timed_out is False

    def limited(*a, **k):
        r = real(*a, **k)
        return SimpleNamespace(x=r.x, fun=r.fun, status=1)
    monkeypatch.setattr(M, "milp", limited)
    with caplog.at_level("WARNING"):
        sols = M.solve_milp(spec, k=3)
    assert len(sols) == 1 and sols[0].timed_out and "time limit" in caplog.text
