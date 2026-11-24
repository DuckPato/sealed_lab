"""Offscreen screenshots and a text-clipping audit of the Sets, Pool and Set report tabs (fake backend, scratch home / settings).

    python scripts/ui_screenshots.py [OUTDIR]               (default docs/screenshots/ui)
        --scale S         QT_SCALE_FACTOR (1.25, 1.5 ...): the physical sizes below are divided by it, like a high-DPI laptop
        --font-scale F    enlarge the application font by F (Windows "make text bigger")
        --prefix P        file name prefix ("scale150_")
        --sizes LIST      physical screen sizes (default 1280x800,1920x1080; with --scale 1920x1080,2560x1440)
        --stress          longest real card names, 8-digit counts, long file paths
        --no-shots        audit only
        --json FILE       write the audit findings as JSON (the tests read it)
        --splitters       also drag every splitter to both extremes in the key states and audit each

Every state is audited with ``sealed_lab.gui.clipcheck.audit`` (text that does not fit its box).  Never touches the real app home
or settings.  The offscreen platform has no fonts of its own: the Windows font folder is used.
"""
from __future__ import annotations

import argparse
import json
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
os.environ.setdefault("SEALED_LAB_NO_DATASET_CHECK", "1")
_TMP = Path(tempfile.mkdtemp(prefix="sealed_lab_shots_"))
os.environ["SEALED_LAB_SETTINGS"] = str(_TMP / "settings.ini")
os.environ["SEALED_LAB_HOME"] = str(_TMP / "home")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIZES = [(1280, 800), (1920, 1080)]          # physical pixels of the screen (scale 1)
SCALED_SIZES = [(1920, 1080), (2560, 1440)]  # with --scale: laptops / monitors that run at 125-150%
LONG_NAMES = ["Asmoranomardicadaistinaculdacar",
              "Rakdos, Patron of Chaos // Rakdos, Spinner of Gluttony",
              "Our Market Research Shows That Players Like Really Long Card Names",
              "Kaya, Ghost Assassin // Kaya, the Inexorable Phantom",
              "Lim-Dûl's Vault of the Forgotten Archmage"]


def pump(app, seconds: float = 0.5) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.004)


def until(app, pred, timeout: float = 20.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if pred():
            return
        time.sleep(0.004)
    raise SystemExit("timeout")


def write(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"x" * 2048)
    return p


def misspell(name: str) -> str:
    w = name.split(" ")
    first = w[0]
    if len(first) > 3:
        first = first[:-2] + first[-1] + first[-2]
    return " ".join([first] + w[1:])


def apply_stress(fake_api) -> None:
    """Longest real card names, 8-digit game counts, long file paths and warnings (before the window is built)."""
    cards = fake_api._CARDS
    for i, n in enumerate(LONG_NAMES):
        cards[i]["name"] = n
    fake_api._BY_NAME.clear()
    fake_api._BY_NAME.update({d["name"].casefold(): i for i, d in cards.items()})
    fake_api._fake_index = None
    long_dir = "C:/Users/someone-with-a-rather-long-user-name/Documents/17Lands exports/Season 2026 - Quick Draft and Sealed/games"
    fake_api._files["TST"] = [
        fake_api._fi(f"{long_dir}/game_data_public.TST.PremierDraft.csv.gz", "game", "TST", "PremierDraft", "draft", 412300.5),
        fake_api._fi(f"{long_dir}/game_data_public.TST.TradSealed.csv.gz", "game", "TST", "TradSealed", "bo3", 18400.4),
        fake_api._fi(f"{long_dir}/game_data_public.TST.Sealed.csv.gz", "game", "TST", "Sealed", "bo1", 46900.9),
        fake_api._fi(f"{long_dir}/draft_data_public.TST.PremierDraft.csv.gz", "draft", "TST", "PremierDraft", "draft", 98000.0,
                     "registered, not used by the model because the replay columns are missing from this export"),
    ]
    fake_api._models["TST"]["n_games"] = {"draft": 12_601_204, "bo1": 4_549_977, "bo3": 1_020_310}
    fake_api._models["TST"]["warnings"] = [
        "bo3: only 20,310 games - card values lean on the Bo1/draft prior, so every number on this page is less certain than usual",
        "bo1: the data ends 41 days ago; later balance patches are not in it"]
    orig = fake_api.set_overview

    def stressed(code: str) -> dict:
        d = orig(code)
        d["games"] = {k: v * 137 for k, v in d.get("games", {}).items()}
        d["events"] = {k: v * 1000 + 99_999 for k, v in d.get("events", {}).items()}
        if d.get("replay_games"):
            d["replay_games"] *= 1000
        return d
    fake_api.set_overview = stressed


class Run:
    """Drives the states; `shot()` audits (and saves) the current state at every size."""

    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.outdir)
        self.findings: list[dict] = []
        self.counts: dict[str, int] = {}
        self.win = None
        self.app = None
        self.sizes = [tuple(int(v) for v in sz.lower().split("x")) for sz in args.sizes.split(",")] if args.sizes else (
            SIZES if args.scale == 1.0 else SCALED_SIZES)

    def shot(self, name: str, shots: bool = True) -> None:
        from sealed_lab.gui.clipcheck import audit

        a = self.args
        for w, h in self.sizes:
            lw, lh = round(w / a.scale), round(h / a.scale)
            self.win.resize(lw, lh)
            pump(self.app, 0.4)
            tag = f"{w}x{h}"
            if shots and not a.no_shots:
                self.out.mkdir(parents=True, exist_ok=True)
                p = self.out / f"{a.prefix}{name}_{tag}.png"
                self.win.grab().save(str(p))
                print(p)
            self.audit_here(name, tag, audit)

    def audit_here(self, name: str, tag: str, audit=None) -> None:
        if audit is None:
            from sealed_lab.gui.clipcheck import audit
        fs = audit(self.win)
        key = f"{name}@{tag}"
        self.counts[key] = len(fs)
        win = self.win
        for f in fs:
            self.findings.append(dict(state=name, size=tag, window=f"{win.width()}x{win.height()}", kind=f.kind, cls=f.cls,
                                      path=f.path, text=f.text, needed=round(f.needed, 1), available=round(f.available, 1)))
        print(f"AUDIT {key}: window {win.width()}x{win.height()} (min {win.minimumSizeHint().width()}x{win.minimumSizeHint().height()}) -> {len(fs)} finding(s)")
        for f in fs[:60]:
            print("   ", f)

    def splitters(self, name: str) -> None:
        """Drag every visible splitter handle to both ends; audit each result; restore."""
        from PySide6.QtWidgets import QSplitter

        from sealed_lab.gui.clipcheck import audit

        w, h = self.sizes[0]
        self.win.resize(round(w / self.args.scale), round(h / self.args.scale))
        pump(self.app, 0.3)
        splits = [s for s in self.win.findChildren(QSplitter) if s.isVisible() and s.count() > 1]
        for k, sp in enumerate(splits):
            saved = sp.sizes()
            total = sum(saved) or 1
            for i in range(sp.count()):
                sizes = [0] * sp.count()
                sizes[i] = total * 10                       # everything else to its minimum
                sp.setSizes(sizes)
                pump(self.app, 0.25)
                self.audit_here(f"{name}:split{k}-grow{i}", f"{w}x{h}", audit)
            sp.setSizes(saved)
            pump(self.app, 0.2)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default=str(ROOT / "docs" / "screenshots" / "ui"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--font-scale", type=float, default=1.0)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--sizes", default="", help="physical screen sizes, e.g. 1920x1080,2560x1440")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--splitters", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.scale != 1.0:
        os.environ["QT_SCALE_FACTOR"] = str(args.scale)

    from PySide6.QtWidgets import QApplication

    from sealed_lab.gui import fake_api
    from sealed_lab.gui.app import MainWindow
    from sealed_lab.gui.clipcheck import scaled_font_app
    from sealed_lab.gui.widgets import apply_theme

    app = QApplication([])
    apply_theme(app, "light")
    if args.font_scale != 1.0:
        scaled_font_app(app, args.font_scale)
    r = Run(args)
    r.app = app
    if args.stress:
        apply_stress(fake_api)
    data = _TMP / "17lands"
    data.mkdir()
    fake_api.fake_data_dir = str(data)
    write(data / "XYZ", "game_data_public.XYZ.TradSealed.csv.gz")
    write(data / "XYZ", "game_data_public.XYZ.Sealed.csv.gz")
    fake_api.untracked_remote.extend([dict(set_code="QQQ", events=["PremierDraft", "Sealed"]), dict(set_code="RRR", events=["TradSealed"])])
    win = MainWindow(fake_api, auto_refresh=True)
    r.win = win
    win.pool_tab.detector.timer.setInterval(5)
    win.show()
    until(app, lambda: win.sets_tab.set_table.rowCount() >= 2 and not win.sets_tab.busy)
    st = win.sets_tab

    def pick(code: str) -> None:
        t = st.set_table
        for row in range(t.rowCount()):
            if t.item(row, 0).text() == code:
                t.selectRow(row)
        pump(app, 0.6)
    # --- Sets: statistics of a ready set, with some other states in the list
    fake_api.add_files([str(write(_TMP / "loose", "game_data_public.AAA.TradSealed.csv.gz")),
                        str(write(_TMP / "loose", "game_data_public.BBB.Sealed.csv.gz"))])
    fake_api.calc_fail["BBB"] = "ValueError: could not convert string to float: 'n/a' (column 'opp_rank')"
    st.refresh()
    until(app, lambda: not st.busy)
    for code in ("BBB", "AAA", "NEO"):
        pick(code)
        st.calc_btn.click()
    os.environ["SEALED_LAB_FAKE_DELAY"] = "0.05"
    until(app, lambda: st.running is not None and st.run_frac > 0.2)
    pick("TST")
    until(app, lambda: "TST" in st.overviews)
    r.shot("sets_statistics_queue")
    if args.splitters:
        r.splitters("sets_statistics_queue")
    os.environ["SEALED_LAB_FAKE_DELAY"] = "0.002"
    until(app, lambda: not st.busy, 60)
    pick("BBB")
    r.shot("sets_failed")
    pick("QQQ")
    r.shot("sets_untracked_download")
    pick("XYZ")
    st.panel.files.set_expanded(True)
    pick("TST")
    st.panel.files.set_expanded(True)
    st.advanced_btn.setChecked(True)
    st.log_btn.setChecked(True)
    r.shot("sets_advanced_log_files")
    st.advanced_btn.setChecked(False)
    st.log_btn.setChecked(False)
    st.panel.files.set_expanded(False)
    # new 17lands data panel on top of the window
    fake_api.datasets_items[:] = [fake_api.make_dataset_item("TST", "game", "PremierDraft", "new"),
                                  fake_api.make_dataset_item("QQQ", "game", "TradSealed", "new"),
                                  fake_api.make_dataset_item("NEO", "game", "Sealed", "updated")]
    win.datasets.show_items(list(fake_api.datasets_items))
    pump(app, 0.5)
    if args.stress:                  # a long status-bar message in a narrow window
        win.statusBar().showMessage("Could not check for new 17lands data: HTTPSConnectionPool(host='data.17landscapes.com', "
                                    "port=443): Max retries exceeded with url: /analysis_data/game_data/ (Caused by NewConnectionError)", 0)
    r.shot("datasets_panel")
    win.statusBar().clearMessage()
    fake_api.datasets_items.clear()
    win.datasets.setVisible(False)
    pump(app, 0.2)

    # --- Pool: detected set, guessed names, own deck, docked preview
    win.tabs.setCurrentWidget(win.pool_tab)
    pt = win.pool_tab
    cards = fake_api._CARDS
    main_ = [f"1 {cards[i]['name']} (TST) {i}" for i in range(0, 60) if cards[i]["colors"] in ("W", "U", "")][:23]
    main_[0] = f"1 {misspell(cards[0]['name'])} (TST) 1" if args.stress else "1 Dawn Knigth (TST) 1"       # a misspelt name: guessed and flagged
    side = ["1 Ember Golem (TST) 4", "1 Grave Ward (TST) 5"]
    text = "\n".join(["Deck", *main_, "9 Plains", "8 Island", "", "Sideboard", *side]) + "\n"
    pt.text.setPlainText(text)
    until(app, lambda: pt.set_code == "TST" and pt._check_snapshot == ("TST", text) and not pt.check_task.running)
    pump(app, 0.4)
    r.shot("pool_detected_guesses")
    pt.fmt_bo3.click()
    pt.build_btn.click()
    until(app, lambda: pt.result is not None and not pt.task.running)
    ub = pt.user_builder
    ub.wait_idle(5000, swaps=True)
    cid = next(iter(ub.side()))
    ub._on_hover(cid, "side")
    pump(app, 0.5)
    r.shot("pool_your_deck")
    if args.splitters:
        r.splitters("pool_your_deck")
    for sec in ub.sections.values():
        sec.set_expanded(True)
    pump(app, 0.5)
    r.shot("pool_builder_panel")
    pt.tabs.widget(0).setCurrentIndex(1)                  # "Details (as generated)" of the first page
    pump(app, 0.5)
    r.shot("pool_details")
    if args.splitters:
        r.splitters("pool_details")
    pt.tabs.widget(0).setCurrentIndex(0)
    pt.tabs.setCurrentIndex(1)
    pump(app, 0.5)
    pt.builders[1].wait_idle(5000, swaps=True)
    r.shot("pool_family")
    for b in pt.builders:
        for sec in b.sections.values():
            sec.set_expanded(True)
    pump(app, 0.5)
    r.shot("pool_family_panel_open")
    if args.splitters:
        r.splitters("pool_family")
    # --- Report tab: the written summary first, both formats, a card in the floating preview, the card lists at the bottom
    win.tabs.setCurrentWidget(win.report_tab)
    rt = win.report_tab
    until(app, lambda: rt.current is not None and not rt.task.running)
    pump(app, 0.5)
    r.shot("report_bo1")
    rt.fmt_bo3.click()
    until(app, lambda: rt.current is not None and rt.current["fmt"] == "bo3" and not rt.task.running)
    pump(app, 0.5)
    r.shot("report_bo3")
    top = (rt.current["cards"]["top"].get("common") or {}).get("W") or rt.current["cards"]["underrated"]
    if top:
        rt._show_card(top[0]["name"], f"<b>{top[0]['name']}</b><br>{top[0]['reason']}")
    bar = rt.scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    pump(app, 0.4)
    r.shot("report_bo3_bottom")
    rt.hover_popup.hide()
    if args.splitters:
        r.splitters("report_bo3_bottom")
    # the summary: its outdated banner, a set without a written report
    from sealed_lab.gui import fake_api as _fa

    bar.setValue(0)
    _fa.FAKE_WRITTEN_OUTDATED = True
    rt.summary.load(rt.set_code())
    pump(app, 0.3)
    r.shot("report_summary_outdated")
    _fa.FAKE_WRITTEN_OUTDATED = False
    _fa.FAKE_WRITTEN_NONE = True
    rt.summary.load(rt.set_code())
    pump(app, 0.3)
    r.shot("report_summary_none")
    _fa.FAKE_WRITTEN_NONE = False
    rt.summary.load(rt.set_code())
    win.tabs.setCurrentWidget(win.report_tab)
    # --- phase 2: the archetype blocks (pairs with sub-archetypes opened) and the deck-shape table
    from PySide6.QtWidgets import QLabel

    arch = rt.current["archetypes"]["pairs"]
    rt._open = {a["key"] for a in arch if a["subs"] or a["spectrum"].get("shown")} | {arch[0]["key"]}
    rt.render()
    pump(app, 0.4)

    def scroll_to(prefix: str) -> None:
        for lab in rt.findChildren(QLabel):
            if lab.text().startswith(prefix):
                bar.setValue(max(0, lab.mapTo(rt.body, lab.rect().topLeft()).y() - 6))
                break
        pump(app, 0.4)

    scroll_to("Archetypes: how each colour")
    r.shot("report_bo3_archetypes")
    scroll_to("Winning deck shape")
    r.shot("report_bo3_deck_shape")
    if args.splitters:
        r.splitters("report_bo3_deck_shape")
    # --- "More detail" blocks opened: the card explorer, per-pair details, shape details, evolution, Bo1 vs Bo3
    rt._open |= {"detail:cards", "detail:pairs", "detail:shape", "detail:compare"}
    rt.render()
    pump(app, 0.5)
    for prefix, name in (("More detail: all cards", "report_bo3_explorer"), ("More detail: each colour pair", "report_bo3_pair_details"),
                         ("More detail: win rate by land", "report_bo3_shape_details"), ("How the format changed", "report_bo3_evolution"),
                         ("Best of 1 against Best of 3", "report_bo3_compare")):
        from sealed_lab.gui.widgets import Collapsible

        target = next((w for w in rt.findChildren(Collapsible) if w.toggle.text().startswith(prefix)), None)
        if target is not None:
            bar.setValue(max(0, target.mapTo(rt.body, target.rect().topLeft()).y() - 6))
            pump(app, 0.4)
        else:
            scroll_to(prefix)
        r.shot(name)
    if args.splitters:
        r.splitters("report_bo3_evolution")
    win.close()
    pump(app, 0.2)
    total = sum(r.counts.values())
    print(f"TOTAL findings: {total} in {len(r.counts)} audited states")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(findings=r.findings, counts=r.counts, total=total), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
