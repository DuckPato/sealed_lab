"""Text-clipping audit of the whole GUI (sealed_lab.gui.clipcheck): no label, button, tab, header, cell or painted text may be cut off.

* unit tests of the audit itself (it must find a clipped label and accept deliberate "…" elision that carries a tooltip);
* the real screens, driven by scripts/ui_screenshots.py in a subprocess (one per Qt scale factor: the factor must be set before the
  QApplication exists) with the fake backend: every state of the Sets / Pool tabs, with ordinary data and with STRESS data (the longest
  real card names, 8-digit game counts, long file paths and warnings), at 1280x800 / 1920x1080 and at 125 % / 150 % screen scaling;
* every splitter dragged to both ends in the key states.

Each subprocess uses its own scratch settings file and app home: nothing of the owner's profile is touched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sealed_lab.gui.clipcheck import audit, format_findings  # noqa: E402
from sealed_lab.gui.widgets import (  # noqa: E402
    Collapsible,
    ElidedCombo,
    ElidedLabel,
    FlowLayout,
    FullTextTipDelegate,
    elide_wrapped,
    with_full_text,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ui_screenshots.py"


@pytest.fixture(scope="module")
def app():
    from sealed_lab.gui.widgets import apply_theme

    a = QApplication.instance() or QApplication([])
    apply_theme(a, "light")
    return a


def shown(widget: QWidget, w: int, h: int, app) -> QWidget:
    widget.resize(w, h)
    widget.show()
    app.processEvents()
    app.processEvents()
    return widget


def kinds(findings) -> set[str]:
    return {f.kind for f in findings}


# --------------------------------------------------------------------------- the audit itself

def test_audit_finds_a_label_that_is_too_narrow(app):
    host = QWidget()
    lay = QVBoxLayout(host)
    lab = QLabel("A label that is much longer than the box it sits in")
    lab.setFixedWidth(60)
    lay.addWidget(lab)
    shown(host, 200, 100, app)
    fs = audit(host)
    assert "label-width" in kinds(fs), format_findings(fs)
    host.close()


def test_audit_finds_a_wrapped_label_without_the_height_it_needs(app):
    host = QWidget()
    lay = QVBoxLayout(host)
    lab = QLabel("word " * 40)
    lab.setWordWrap(True)
    lab.setFixedSize(150, 16)
    lay.addWidget(lab)
    shown(host, 200, 100, app)
    assert "label-height" in kinds(audit(host))
    lab.setFixedHeight(400)
    app.processEvents()
    assert not audit(host)
    host.close()


def test_audit_finds_clipped_buttons_combos_tabs_and_headers(app):
    host = QWidget()
    lay = QVBoxLayout(host)
    btn = QPushButton("Build best decks")
    btn.setFixedWidth(40)
    combo = QComboBox()
    combo.addItem("Asmoranomardicadaistinaculdacar and then some")
    combo.setFixedWidth(70)
    tabs = QTabWidget()
    tabs.setUsesScrollButtons(False)
    tabs.setElideMode(Qt.TextElideMode.ElideRight)
    tabs.addTab(QWidget(), "A rather long tab title that cannot fit")
    tabs.addTab(QWidget(), "Another long tab title that cannot fit")
    table = QTableWidget(1, 2)
    table.setHorizontalHeaderLabels(["A long column title", "Other"])
    table.setColumnWidth(0, 30)
    for w in (btn, combo, tabs, table):
        lay.addWidget(w)
    shown(host, 260, 400, app)
    ks = kinds(audit(host))
    assert {"button", "combo", "tab", "header"} <= ks, ks
    # a tab that elides is fine once its tooltip shows the whole title
    for i in range(tabs.count()):
        tabs.setTabToolTip(i, tabs.tabText(i))
    assert "tab" not in kinds(audit(host))
    host.close()


def test_elided_label_is_accepted_only_with_its_tooltip(app):
    host = QWidget()
    lay = QVBoxLayout(host)
    lab = ElidedLabel("C:/Users/someone/Documents/17Lands exports/Season 2026/games/game_data_public.TST.PremierDraft.csv.gz")
    lay.addWidget(lab)
    shown(host, 180, 60, app)
    assert lab.is_elided() and "game_data_public.TST" in lab.toolTip()
    assert not audit(host)
    QLabel.setToolTip(lab, "")                     # the safety net: elided text without its tooltip is a finding
    assert "elided-no-tip" in kinds(audit(host))
    host.close()


def test_elided_combo_and_collapsible_title(app):
    host = QWidget()
    lay = QVBoxLayout(host)
    combo = ElidedCombo()
    combo.addItem("Rakdos, Patron of Chaos // Rakdos, Spinner of Gluttony  (91%)", "x")
    combo.setToolTip("Pick the card")
    sec = Collapsible("Mana simulation  ·  screw 8% · flood 7% · color 5% · and a lot of other words", QLabel("x"), expanded=False)
    sec.toggle.setToolTip("Help text")
    lay.addWidget(combo)
    lay.addWidget(sec)
    shown(host, 190, 120, app)
    assert combo.is_elided() and "Rakdos, Patron" in combo.toolTip() and "Pick the card" in combo.toolTip()
    assert sec.toggle.is_elided() and "Mana simulation" in sec.toggle.toolTip() and "Help text" in sec.toggle.toolTip()
    assert not audit(host), format_findings(audit(host))
    host.close()


def test_cells_elide_with_a_full_text_tooltip(app):
    long = "Our Market Research Shows That Players Like Really Long Card Names"
    plain = QTableWidget(1, 1)
    plain.setItem(0, 0, QTableWidgetItem(long))
    plain.setColumnWidth(0, 80)
    smart = QTableWidget(1, 1)
    smart.setItemDelegate(FullTextTipDelegate(smart))
    smart.setItem(0, 0, QTableWidgetItem(long))
    smart.setColumnWidth(0, 80)
    for t in (plain, smart):
        shown(t, 200, 100, app)
    assert "cell" in kinds(audit(plain))                 # cut off, nothing says what it was
    assert not audit(smart)                              # elided on purpose, the delegate shows the whole name on hover
    plain.item(0, 0).setToolTip(long)                    # a tooltip with the text also does it
    assert not audit(plain)
    plain.close()
    smart.close()


def test_custom_painted_widgets_report_through_the_hook(app):
    class Painted(QWidget):
        def clipped_texts(self):
            return [("a painted text", 120.0, 80.0)]

    w = Painted()
    shown(w, 100, 40, app)
    fs = audit(w)
    assert [f.kind for f in fs] == ["custom"] and fs[0].needed == 120.0 and fs[0].available == 80.0
    w.close()


def test_audit_finds_overlapping_siblings(app):
    host = QWidget()                                       # no layout: widgets placed by hand on top of each other
    a, b = QLabel("first", host), QWidget(host)
    a.setGeometry(0, 0, 80, 30)
    b.setGeometry(40, 5, 80, 30)
    shown(host, 200, 100, app)
    assert "overlap" in kinds(audit(host))
    b.move(100, 5)
    app.processEvents()
    assert "overlap" not in kinds(audit(host))
    host.close()


def test_queue_strip_wraps_a_long_queue(app):
    from sealed_lab.gui.sets_tab import QueueStrip

    strip = QueueStrip()
    strip.set_items(("AAA", 40), [f"S{k:02d}" for k in range(14)])
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.addWidget(strip)
    lay.addStretch(1)
    shown(host, 420, 300, app)
    assert strip.minimumSizeHint().width() < 200                 # a long queue does not widen the window ...
    assert strip.height() > strip.fontMetrics().height() * 2     # ... it takes a second row
    assert not audit(host), format_findings(audit(host))
    host.close()


def test_status_bar_elides_a_long_message_with_a_tooltip(app):
    from PySide6.QtWidgets import QMainWindow

    from sealed_lab.gui.widgets import ElidedStatusBar

    win = QMainWindow()
    win.setStatusBar(ElidedStatusBar())
    win.statusBar().addPermanentWidget(QLabel("fake backend"))
    msg = "Could not check for new 17lands data: " + "connection refused " * 12
    win.statusBar().showMessage(msg, 0)
    shown(win, 400, 100, app)
    bar = win.statusBar()
    assert bar.currentMessage().endswith("…") and msg.startswith(bar.currentMessage()[:-1])
    assert msg in bar.toolTip() and not audit(win)
    win.resize(bar.fontMetrics().horizontalAdvance(msg) + 400, 100)     # room for the whole message in any font
    app.processEvents()
    assert bar.currentMessage() == msg and not bar.toolTip()
    win.close()


def test_helpers(app):
    assert with_full_text("", "Name") == "Name"
    assert with_full_text("plain tip", "Name") == "Name\nplain tip"
    assert with_full_text("Name is here", "Name") == "Name is here"
    rich = "<html><table width='400'><tr><td><p>hi</p></td></tr></table></html>"
    assert "<b>Name</b>" in with_full_text(rich, "Name") and "<p>hi</p>" in with_full_text(rich, "Name")
    from PySide6.QtGui import QFontMetrics

    fm = QFontMetrics(app.font())
    out = elide_wrapped(fm, "word " * 60, 90, fm.lineSpacing() * 2)
    assert out.endswith("…") and len(out) < 300
    assert elide_wrapped(fm, "short", 200, 100) == "short"


def test_flow_layout_wraps_instead_of_widening(app):
    host = QWidget()
    flow = FlowLayout(host)
    for t in ("Track this folder", "Track a folder for this set…", "Download…"):
        flow.addWidget(QPushButton(t))
    wide = flow.heightForWidth(2000)
    narrow = flow.heightForWidth(220)
    assert narrow > wide                                   # two or three rows instead of one
    assert flow.minimumSize().width() < 220                # the layout itself can be as narrow as its widest widget
    host.deleteLater()


# --------------------------------------------------------------------------- the real screens

def run_screens(tmp_path: Path, *args: str, timeout: int = 900) -> dict:
    out = tmp_path / "audit.json"
    env = dict(os.environ)
    env.pop("QT_SCALE_FACTOR", None)
    cmd = [sys.executable, str(SCRIPT), str(tmp_path / "shots"), "--no-shots", "--json", str(out), *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env, cwd=str(ROOT))
    assert out.exists(), f"audit run failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return json.loads(out.read_text(encoding="utf-8"))


def describe(res: dict) -> str:
    lines = [f"{f['state']} @ {f['size']} (window {f['window']}) [{f['kind']}] {f['cls']} {f['path'][-80:]}: "
             f"{f['text'][:60]!r} needs {f['needed']}px, has {f['available']}px" for f in res["findings"][:40]]
    return "\n".join(lines) + (f"\n... and {res['total'] - 40} more" if res["total"] > 40 else "")


@pytest.mark.parametrize("name,args", [
    ("normal+stress+splitters", ["--stress", "--splitters"]),
    ("scale125", ["--scale", "1.25", "--stress"]),
    ("scale150", ["--scale", "1.5", "--stress", "--splitters"]),
], ids=lambda v: v if isinstance(v, str) else "")
def test_no_text_is_clipped(tmp_path, name, args):
    res = run_screens(tmp_path, *args)
    assert len(res["counts"]) >= 20, "the audit did not visit the screens"           # Sets (7 states) + Pool (6) at two sizes
    assert res["total"] == 0, f"{res['total']} clipped text(s) in '{name}':\n{describe(res)}"


def test_big_system_font_is_clipping_free_at_full_hd(tmp_path):
    """Windows 'make text bigger' (125 %) on a 1920x1080 screen."""
    res = run_screens(tmp_path, "--font-scale", "1.25", "--sizes", "1920x1080")
    assert res["total"] == 0, f"{res['total']} clipped text(s):\n{describe(res)}"
