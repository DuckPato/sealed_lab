"""Deck search tab: offscreen Qt + the fake backend."""
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
_CFG = tempfile.mkdtemp(prefix="sealed_lab_ds_cfg_")
os.environ.setdefault("SEALED_LAB_SETTINGS", os.path.join(_CFG, "settings.ini"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sealed_lab.gui import fake_api  # noqa: E402
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


@pytest.fixture()
def calls(monkeypatch):
    log: list[dict] = []
    orig = fake_api.search_decks

    def rec(set_code, card, **kw):
        log.append(dict(set_code=set_code, card=card, **kw))
        return orig(set_code, card, **kw)

    monkeypatch.setattr(fake_api, "search_decks", rec)
    return log


@pytest.fixture()
def win(qapp, tmp_path, calls):
    fake_api.reset()
    new_settings().clear()
    old = fake_api.fake_data_dir
    fake_api.fake_data_dir = str(tmp_path)
    w = MainWindow(fake_api, auto_refresh=True, auto_check=False)
    w.show()
    w.tabs.setCurrentWidget(w.deck_search_tab)
    t = w.deck_search_tab
    wait_until(lambda: t.set_combo.count() >= 1 and t._meta_ok() and not t.task.running, what="names loaded")
    yield w
    t.wait(3000)
    w.sets_tab.wait(3000)
    w.close()
    QApplication.processEvents()
    fake_api.fake_data_dir = old


def idle(t) -> bool:
    return not t.task.running and t._pending is None


def search(t, card: str = "Dawn Knight") -> None:
    t.card_edit.setText(card)
    t.search_btn.click()
    wait_until(lambda: idle(t) and t.searched is not None, what="search")


def test_tab_exists_and_set_combo(win):
    t = win.deck_search_tab
    assert win.tabs.tabText(win.tabs.indexOf(t)) == "Deck search"
    assert [t.set_combo.itemText(i) for i in range(t.set_combo.count())] == ["TST"]       # only sets with a corpus
    t.refresh_sets([])
    assert t.set_combo.count() == 0 and "Calculate a set" in t.message.text()


def test_completer_gets_the_names_and_formats_are_built(win):
    t = win.deck_search_tab
    names = fake_api.deck_search_cards("TST")
    assert t.cards == names and t.completer.model().rowCount() == len(names)
    assert "Dawn Knight" in names
    assert len(t._format_boxes) == 4 and all(b.isChecked() for b in t._format_boxes.values())
    assert "Premier Draft (Bo1) (40)" == t._format_boxes["PremierDraft"].text()
    assert t.completer.filterMode().name == "MatchContains"


def test_search_fills_list_and_selection_shows_the_deck(win, calls):
    t = win.deck_search_tab
    search(t)
    assert t.tree.topLevelItemCount() == len(t.results) > 0
    assert "decks with Dawn Knight" in t.summary.full_text()
    assert calls[-1]["limit"] == 200 and calls[-1]["offset"] == 0 and calls[-1]["events"] is None
    assert t.tree.currentItem() is not None and t.copy_btn.isEnabled()
    from PySide6.QtWidgets import QLabel

    from sealed_lab.gui.deck_search_tab import ReadOnlyPile
    texts = lambda: [w.text() for w in t.deck_box.findChildren(QLabel)]

    def pick(pred):
        for i, r in enumerate(t.results):
            if pred(r):
                t.tree.setCurrentItem(t.tree.topLevelItem(i))
                return r
        raise AssertionError("no such run")

    def check_main(r):
        QApplication.processEvents()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        piles = t.deck_box.findChildren(ReadOnlyPile)
        assert piles and piles == t.piles
        main = piles[0]
        spells = sum(c["n"] for c in r["main"] if not c["is_basic"])
        assert len(main.ids) == spells > 0                       # one id per copy; basics are on a text line
        basics = [c for c in r["main"] if c["is_basic"]]
        if basics:
            assert any(x.startswith("Basic lands: ") and f"{basics[0]['n']} {basics[0]['name']}" in x for x in texts())
        target = next(c for c in r["main"] if c["name"] == "Dawn Knight")
        assert main.chips == {target["card_id"]: "searched"} and main.ids.count(target["card_id"]) == target["n"]
        assert main.accepts == set() and not main.acceptDrops()
        return piles

    # a draft run (no sideboard pile, a note) and a Sealed run (a sideboard pile)
    r = pick(lambda r: r["event"] == "PremierDraft")
    assert len(check_main(r)) == 1
    assert any("Draft sideboards are not stored" in x for x in texts())
    assert not any(x.startswith("Sideboard (") for x in texts())
    r = pick(lambda r: r["event"] == "Sealed")
    piles = check_main(r)
    assert len(piles) == 2 and any(x.startswith("Sideboard (6)") for x in texts())
    assert len(piles[1].ids) == sum(c["n"] for c in r["sideboard"]) == 6
    assert piles[1].opts["card_w"] < piles[0].opts["card_w"]
    # the size slider resizes the piles and is remembered
    t.size_slider.setValue(150)
    assert t.piles[0].opts["card_w"] == 150 and new_settings().value("decksearch/card_w", type=int) == 150
    assert any(x.startswith("Main deck (") for x in texts())


def test_format_filter_and_sort_requery(win, calls):
    t = win.deck_search_tab
    search(t)
    n = len(calls)
    for e in ("PremierDraft", "TradDraft"):
        t._format_boxes[e].setChecked(False)
    wait_until(lambda: len(calls) > n and idle(t), what="requery")
    assert calls[-1]["events"] == ["Sealed", "TradSealed"]
    assert {r["event"] for r in t.results} <= {"Sealed", "TradSealed"}
    n = len(calls)
    t.sort_combo.setCurrentIndex(t.sort_combo.findData("newest"))
    t._on_sort_chosen(0)
    t.min_games.setValue(5)
    t.share_box.setChecked(False)
    wait_until(lambda: len(calls) > n and idle(t), what="requery 2")
    last = calls[-1]
    assert last["sort"] == "newest" and last["min_games"] == 5 and last["min_card_share"] == 0.0
    dates = [r["date"] for r in t.results]
    assert dates == sorted(dates, reverse=True)


def test_show_more_appends(win, calls):
    t = win.deck_search_tab
    search(t)
    total = t.data["total"]
    assert total > 40
    import sealed_lab.gui.deck_search_tab as m
    m.PAGE = 20
    try:
        t.request_search("new")
        wait_until(lambda: idle(t) and len(t.results) == 20, what="page 1")
        assert t.more_btn.isEnabled()
        t.more_btn.click()
        wait_until(lambda: idle(t) and len(t.results) == 40, what="page 2")
        assert calls[-1]["offset"] == 20 and t.tree.topLevelItemCount() == 40
        assert len({r["draft_id"] for r in t.results}) == 40
    finally:
        m.PAGE = 200


def test_unknown_card_is_an_inline_message(win):
    t = win.deck_search_tab
    t.card_edit.setText("Nonexistent Thing")
    t.search_btn.click()
    wait_until(lambda: idle(t) and not t.inline.isHidden(), what="inline message")
    assert "No card called" in t.inline.text() and win.last_error is None


def test_no_match_message(win):
    t = win.deck_search_tab
    t.min_games.setValue(20)
    search(t)
    assert "No deck with Dawn Knight matches these filters." == t.message.text() and t.message.isVisible()


def test_copy_deck_puts_arena_text_on_the_clipboard(win):
    t = win.deck_search_tab
    search(t)
    seen = []
    t.statusMessage.connect(seen.append)
    t.copy_btn.click()
    text = QApplication.clipboard().text()
    assert text.startswith("Deck\n") and "1 Dawn Knight" in text
    assert seen and "Arena's Import" in seen[-1]


def test_set_calculating_disables_search(win):
    t = win.deck_search_tab
    t.set_calculating({"TST"})
    assert not t.search_btn.isEnabled() and not t.card_edit.isEnabled()
    assert "recalculated" in t.message.text()
    t.card_edit.setText("Dawn Knight")
    t.request_search("new")
    assert t._pending is None
    t.set_calculating(set())
    assert t.search_btn.isEnabled()
