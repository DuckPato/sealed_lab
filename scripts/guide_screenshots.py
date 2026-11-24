"""Screenshots for GETTING_STARTED.md: the first-time walkthrough (download a set, calculate it, build a deck from a pool, look a
card up), taken offscreen on the built-in example backend.  The example set "TST" and its cards are invented, so no real data,
folder or name of anyone's appears.  Never touches the real app home or settings.

    python scripts/guide_screenshots.py [OUTDIR]          (default guide_images)
"""
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
os.environ["SEALED_LAB_NO_DATASET_CHECK"] = "1"
os.environ["SEALED_LAB_NO_UPDATE_CHECK"] = "1"
_TMP = Path(tempfile.mkdtemp(prefix="sealed_lab_guide_"))
os.environ["SEALED_LAB_SETTINGS"] = str(_TMP / "settings.ini")
os.environ["SEALED_LAB_HOME"] = str(_TMP / "home")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIZE = (1360, 860)
SHOWN_DATA_DIR = r"C:\Users\you\Documents\17lands"     # what the screenshots show instead of the scratch folder


def pump(app, seconds: float = 0.4) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.004)


def until(app, pred, timeout: float = 30.0, what: str = "") -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if pred():
            return
        time.sleep(0.004)
    raise SystemExit(f"timeout: {what}")


def scrub(win, data_dir: Path) -> None:
    """Show a neutral data folder instead of the scratch one, wherever a label names it."""
    from PySide6.QtWidgets import QLabel

    variants = {str(data_dir), str(data_dir).replace("\\", "/"), data_dir.as_posix()}
    for lab in win.findChildren(QLabel):
        full = lab.full_text() if hasattr(lab, "full_text") else lab.text()
        new = full
        for v in variants:
            new = new.replace(v, SHOWN_DATA_DIR)
        if new != full:
            lab.setText(new)
        tip = lab.toolTip()
        if any(v in tip for v in variants):
            for v in variants:
                tip = tip.replace(v, SHOWN_DATA_DIR)
            lab.setToolTip(tip)


def pool_text(fake_api) -> str:
    """A sealed pool as Arena exports it before you build: every card under "Deck" (all invented TST cards)."""
    cards = [(i, d) for i, d in fake_api._CARDS.items() if not d["is_land"]]
    return "\n".join(["Deck", *[f"1 {d['name']} (TST) {i + 1}" for i, d in cards]]) + "\n"


def tidy_example(fake_api) -> None:
    """The example backend is made for tests: give its placeholder cards a creature type and drop a test-only sentence."""
    orig = fake_api.card_info

    def card_info(set_code: str) -> dict:
        out = orig(set_code)
        for e in out.values():
            if e.get("types") == "Creature - Fake":
                e["types"] = "Creature - " + e["name"].split()[1]
        return out
    fake_api.card_info = card_info
    family = fake_api._fake_family

    def fake_family(*a, **kw) -> dict:
        d = family(*a, **kw)
        if d.get("archetype") == "Fake Fliers":
            d["archetype"] = "Fliers"
        return d
    fake_api._fake_family = fake_family
    calc = fake_api.calculate_set

    def calculate_set(*a, **kw):
        st = calc(*a, **kw)
        for m in fake_api._models.values():
            m["warnings"] = []                             # (the example's own note about its quick bootstrap)
        return st
    fake_api.calculate_set = calculate_set
    fake_api.FAKE_WRITTEN_TEXT = fake_api.FAKE_WRITTEN_TEXT.replace(
        "; *this paragraph* has no numbers, so it needs no tag.", ".")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "guide_images"
    out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtWidgets import QApplication

    from sealed_lab.gui import fake_api
    from sealed_lab.gui.app import MainWindow
    from sealed_lab.gui.widgets import Collapsible, apply_theme

    app = QApplication([])
    apply_theme(app, "light")
    tidy_example(fake_api)
    data = _TMP / "17lands"
    data.mkdir()
    fake_api.fake_data_dir = str(data)
    # first start: nothing tracked yet; 17lands lists the example set TST (and another one)
    fake_api._files.clear()
    fake_api._models.clear()
    fake_api.untracked_remote[:] = [dict(set_code="TST", events=["PremierDraft", "Sealed", "TradSealed"]),
                                    dict(set_code="ABC", events=["PremierDraft", "Sealed"])]
    win = MainWindow(fake_api, auto_refresh=True, auto_check=False)
    win._status.setText("")                                # not "fake backend"
    win.pool_tab.detector.timer.setInterval(5)
    win.resize(*SIZE)
    win.show()
    st = win.sets_tab
    until(app, lambda: st.set_table.rowCount() >= 2 and not st.busy, what="sets listed")

    def shot(name: str) -> None:
        win.resize(*SIZE)
        win.statusBar().clearMessage()
        pump(app, 0.4)
        scrub(win, data)
        pump(app, 0.1)
        p = out / f"{name}.png"
        win.grab().save(str(p))
        print(p)

    def pick(code: str) -> None:
        t = st.set_table
        for row in range(t.rowCount()):
            if t.item(row, 0).text() == code:
                t.selectRow(row)
        pump(app, 0.6)

    # 1. Sets tab at first start: TST is greyed (17lands has it, you don't yet) -> Download...
    win.tabs.setCurrentWidget(st)
    pick("TST")
    shot("01_sets_first_start")
    # 2. the download panel at the top of the window
    st.panel.download_btn.click()
    until(app, lambda: win.datasets.isVisible(), what="download panel")
    pump(app, 0.4)
    shot("02_download_panel")
    win.datasets.download_btn.click()
    until(app, lambda: "TST" in fake_api._files, 30, "download")
    pump(app, 1.0)
    if win.datasets.isVisible():                           # (it may stay up with a "done" line)
        win.datasets.setVisible(False)
    st.refresh()
    until(app, lambda: not st.busy, what="rescan")
    # 3. TST is tracked: Calculate
    pick("TST")
    shot("03_sets_calculate")
    os.environ["SEALED_LAB_FAKE_DELAY"] = "0.08"
    st.calc_btn.click()
    until(app, lambda: st.running is not None and st.run_frac > 0.45, what="calculating")
    shot("04_sets_calculating")
    os.environ["SEALED_LAB_FAKE_DELAY"] = "0.002"
    until(app, lambda: not st.busy, 120, "calculated")
    pick("TST")
    until(app, lambda: "TST" in st.overviews, what="overview")
    shot("05_sets_ready")

    # 4. Deck builder: paste the pool, the set is detected
    win.tabs.setCurrentWidget(win.pool_tab)
    pt = win.pool_tab
    text = pool_text(fake_api)
    pt.text.setPlainText(text)
    until(app, lambda: pt.set_code == "TST" and pt._check_snapshot == ("TST", text) and not pt.check_task.running, what="pool checked")
    pt.fmt_bo3.click()
    shot("06_builder_pool")
    pt.build_btn.click()
    until(app, lambda: pt.result is not None and not pt.task.running, 60, "built")
    pt.builders[0].wait_idle(5000, swaps=True)
    pump(app, 0.6)
    shot("07_builder_best_deck")
    pt.tabs.widget(0).setCurrentIndex(1)                   # "Details (as generated)": why this deck
    pump(app, 0.6)
    shot("08_builder_details")
    pt.tabs.widget(0).setCurrentIndex(0)

    # 5. Report: the written summary first
    win.tabs.setCurrentWidget(win.report_tab)
    rt = win.report_tab
    until(app, lambda: rt.current is not None and not rt.task.running, 60, "report")
    pump(app, 0.6)
    shot("09_report")
    # 6. a card: the card explorer (More detail: all cards), searched by name, with the card shown
    rt._open |= {"detail:cards"}
    rt.render()
    pump(app, 0.5)
    blk = next((w for w in rt.findChildren(Collapsible) if w.toggle.text().startswith("More detail: all cards")), None)
    if blk is None:
        raise SystemExit("no card explorer in the report")
    rt._scroll_to(blk)
    pump(app, 0.4)
    card = "Dawn Knight"
    if rt.explorer is not None:
        rt.explorer.search.setText(card)
        pump(app, 0.4)
    shot("10_report_card")

    # 7. Deck search: real decks (here: invented ones) that played the card
    win.tabs.setCurrentWidget(win.deck_search_tab)
    ds = win.deck_search_tab
    until(app, lambda: ds.set_combo.count() >= 1 and ds._meta_ok() and not ds.task.running, 30, "deck search ready")
    ds.card_edit.setText(card)
    ds.search_btn.click()
    until(app, lambda: not ds.task.running and ds._pending is None and ds.searched is not None, 30, "searched")
    pump(app, 0.6)
    shot("11_deck_search")

    win.close()
    pump(app, 0.2)


if __name__ == "__main__":
    main()
