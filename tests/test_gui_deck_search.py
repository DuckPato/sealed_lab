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

from PySide6.QtCore import QEvent, Qt  # noqa: E402
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


def test_browse_mode_insights_chips_and_filters(win, calls):
    t = win.deck_search_tab
    # Browse mode (empty card box) works and renders Corpus insights
    t.card_edit.setText("")
    t.search_btn.click()
    wait_until(lambda: idle(t) and t.searched is not None, what="browse search")
    assert t.results and t.insights_box.isVisible()
    assert calls[-1]["card"] is None

    # Adding a card chip + trophy filter requeries cleanly
    n = len(calls)
    t.card_edit.setText("Dawn Knight")
    t.add_inc_btn.click()
    wait_until(lambda: len(calls) > n and idle(t), what="chip requery")
    assert calls[-1]["include_cards"] == [("Dawn Knight", 1)]
    assert t.chip_row.isVisible()

    n = len(calls)
    t.trophy_box.setChecked(True)
    wait_until(lambda: len(calls) > n and idle(t), what="trophy requery")
    assert calls[-1]["trophy_only"] is True
    assert all(r["trophy"] for r in t.results)


def test_model_values_overlay_and_open_in_builder(win):
    t = win.deck_search_tab
    search(t)
    # Toggle Model values overlay
    t.values_box.setChecked(True)
    wait_until(lambda: not t.val_task.running and bool(t.piles and t.piles[0].values), what="model values overlay")
    assert t.piles[0].opts["show_values"] is True

    # Open in Deck builder switches to pool_tab and populates the text
    t.open_builder_btn.click()
    QApplication.processEvents()
    assert win.tabs.currentWidget() is win.pool_tab
    assert "Deck" in win.pool_tab.text.toPlainText()
    win.pool_tab.wait(3000)


def test_explorer_modes_similar_rebuilds_and_sideboarding(win):
    t = win.deck_search_tab
    # 1. Rebuilds mode
    t.set_mode("rebuilds")
    wait_until(lambda: idle(t) and t.searched is not None, what="rebuilds search")
    assert t.results and t.insights_box.isVisible()

    # 2. Bo3 sideboarding mode
    t.set_mode("sideboard")
    wait_until(lambda: idle(t) and t.searched is not None, what="sideboard search")
    assert t.results and t.insights_box.isVisible()

    # 3. Similar pools mode (feed a current pool via set_current_pool)
    info = fake_api.card_info("TST")
    sample_pool = {cid: 1 for cid, d in list(info.items())[:20] if not d.get("is_basic")}
    t.set_current_pool("TST", sample_pool)
    t.set_mode("similar")
    wait_until(lambda: idle(t) and t.searched is not None, what="similar pools search")
    assert t.results and t.insights_box.isVisible()
    assert win.last_error is None


def test_date_toggles_and_advanced_search_panel(win, calls):
    t = win.deck_search_tab
    # Advanced search panel starts collapsed and toggles cleanly
    assert not t.advanced_panel.isVisible()
    assert t.date_days() is None and t.date_buttons[None].isChecked()
    t.advanced_btn.click()
    assert t.advanced_panel.isVisible() and "▴" in t.advanced_btn.text()
    t.advanced_btn.click()
    assert not t.advanced_panel.isVisible()

    # Date window toggles pass `days` and display cutoff in summary
    from PySide6.QtWidgets import QLabel

    search(t)
    ins_txt = " ".join(w.text() for w in t.insights_box.findChildren(QLabel))
    assert "1st-hand keep" in ins_txt
    all_total = t.data["total"]
    n = len(calls)
    t.date_buttons[14].click()
    wait_until(lambda: len(calls) > n and idle(t), what="date 14d requery")
    assert calls[-1]["days"] == 14
    assert t.data["total"] <= all_total
    if t.results:
        assert "since " in t.summary.full_text()

    n = len(calls)
    t.date_buttons[7].click()
    wait_until(lambda: len(calls) > n and idle(t), what="date 7d requery")
    assert calls[-1]["days"] == 7

    # Reset filters restores date window to All and clears active badges
    t.trophy_box.setChecked(True)
    assert "(1)" in t.advanced_btn.text()
    t.reset_filters()
    wait_until(lambda: idle(t), what="reset requery")
    assert t.date_days() is None and t.date_buttons[None].isChecked() and not t.trophy_box.isChecked()
    assert "(1)" not in t.advanced_btn.text()


def test_colour_and_rank_tick_lists_and_model_score_sort(win, calls):
    t = win.deck_search_tab
    t.advanced_btn.setChecked(True)
    t.card_edit.setText("")
    t.search_btn.click()
    wait_until(lambda: idle(t) and t.searched is not None, what="browse search")

    # ticking colors: several at once, exact by default, then "including"
    n = len(calls)
    t.color_pick.boxes["U"].setChecked(True)
    t.color_pick.boxes["R"].setChecked(True)
    wait_until(lambda: len(calls) > n and idle(t), what="color requery")
    assert calls[-1]["colors"] == "UR" and calls[-1]["color_match"] == "exact"
    assert t.color_pick.text() == "UR"
    assert all(r["colors"].split("+")[0] == "UR" for r in t.results)
    n = len(calls)
    t.color_pick.radios["include"].setChecked(True)
    wait_until(lambda: len(calls) > n and idle(t), what="include requery")
    assert calls[-1]["color_match"] == "include" and t.color_pick.text() == "Including UR"

    # ranks: any set of tiers
    n = len(calls)
    t.rank_pick.boxes["Diamond"].setChecked(True)
    t.rank_pick.boxes["Mythic"].setChecked(True)
    wait_until(lambda: len(calls) > n and idle(t), what="rank requery")
    assert calls[-1]["ranks"] == ["Diamond", "Mythic"]
    assert all(r["rank"] in ("Diamond", "Mythic") for r in t.results)
    assert t._active_advanced_count() >= 2

    # the chip clears it; Reset clears everything
    t._clear_one("rank")
    assert t.rank_pick.values() == [] and t.rank_pick.text() == "Any rank"
    t.reset_filters()
    assert t.color_pick.values() == [] and t.color_pick.mode() == "exact"

    # Model score sort: scores the decks first, then lists Sealed runs by score (draft runs, unscored, last)
    n = len(calls)
    t.sort_combo.setCurrentIndex(t.sort_combo.findData("model"))
    t._on_sort_chosen(0)
    wait_until(lambda: len(calls) > n and idle(t), what="model sort")
    assert calls[-1]["sort"] == "model" and fake_api.has_deck_model_scores(t.set_code())
    scores = [r["model_score"] for r in t.results]
    known = [s for s in scores if s is not None]
    assert known and known == sorted(known, reverse=True)
    assert scores[:len(known)] == known
    assert t.tree.topLevelItem(0).text(8)

    # Clickable color pill in Corpus insights -> Colors: filters straight to that exact color pair
    from PySide6.QtWidgets import QLabel, QPushButton

    color_pills = [b for b in t.insights_box.findChildren(QPushButton) if b.toolTip().endswith("Click to filter to this exact color pair.")]
    assert color_pills
    n = len(calls)
    color_pills[0].click()
    wait_until(lambda: len(calls) > n and idle(t), what="color pill filter")
    assert t.color_pick.values() and t.color_pick.mode() == "exact"

    # Rebuilds mode: default min_changes=5, unbiased Model score (before -> after) instead of WR, Model score column
    t.set_mode("rebuilds")
    wait_until(lambda: idle(t) and t.searched is not None, what="rebuilds mode")
    assert t.min_changes.value() == 5
    assert all(r["cards_changed"] >= 5 for r in t.results)
    reb_txt = " ".join(w.text() for w in t.insights_box.findChildren(QLabel))
    assert "Did rebuilding improve the deck? (Sealed only" in reb_txt and "Before rebuild WR" not in reb_txt
    assert "Draft runs are not scored" in reb_txt
    assert any("→" in t.tree.topLevelItem(i).text(2) for i in range(t.tree.topLevelItemCount()))

    # Clicking column headers in Rebuilds sorts across the corpus (Model score desc, then asc, then Changed)
    t._on_header_clicked(2)
    wait_until(lambda: idle(t) and "sorted by Model score ↓" in t.summary.full_text(), what="rebuilds sort model desc")
    deltas_desc = [r["model_score_delta"] for r in t.results]
    known_desc = [d for d in deltas_desc if d is not None]
    assert known_desc and known_desc == sorted(known_desc, reverse=True) and deltas_desc[:len(known_desc)] == known_desc

    t._on_header_clicked(2)
    wait_until(lambda: idle(t) and "sorted by Model score ↑" in t.summary.full_text(), what="rebuilds sort model asc")
    deltas_asc = [r["model_score_delta"] for r in t.results]
    known_asc = [d for d in deltas_asc if d is not None]
    assert known_asc and known_asc == sorted(known_asc) and deltas_asc[:len(known_asc)] == known_asc

    t._on_header_clicked(7)
    wait_until(lambda: idle(t) and "sorted by Skill ↓" in t.summary.full_text(), what="rebuilds sort skill desc")
    skills = [r["skill"] for r in t.results if r.get("skill") is not None]
    assert skills == sorted(skills, reverse=True)

    # Sideboard mode: pool filter checkbox enables once pool is loaded & passes pool_cards
    info = fake_api.card_info("TST")
    sample_pool = {cid: 1 for cid, d in list(info.items())[:25] if not d.get("is_basic")}
    t.set_current_pool("TST", sample_pool)
    t.set_mode("sideboard")
    wait_until(lambda: idle(t) and t.searched is not None, what="sideboard mode")
    assert t.sb_pool_box.isEnabled()
    assert "strong-pilot" in t.insights_headline.text()
    t.sb_pool_box.setChecked(True)
    wait_until(lambda: idle(t) and bool((t.data or {}).get("pool_filtered")), what="sideboard pool filter")
    assert t.data["pool_filtered"] is True

    # Similar pools mode shows pilot's cached Model score column & supports clickable column sorting
    t.set_mode("similar")
    wait_until(lambda: idle(t) and t.searched is not None, what="similar pools model score")
    assert t.tree.topLevelItem(0).text(7)
    t._on_header_clicked(7)
    QApplication.processEvents()
    sim_order = [t.results[t.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)]["model_score"]
                 for i in range(t.tree.topLevelItemCount())]
    sim_known = [s for s in sim_order if s is not None]
    assert sim_known and sim_known == sorted(sim_known, reverse=True)
