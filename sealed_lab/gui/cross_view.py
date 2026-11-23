"""The Report tab's read-only view of "All sets": one section per cross-set lesson (L1-L9, see `sealed_lab.cross_set`).

Each section has the question, a plain-words verdict per item and format (PATTERN / MIXED / RARE with "clear in N of M"), and a table
set x format with the clear results (effect and badge); a blank cell means no clear result.  Nothing is computed here: `api.cross_set` has it all.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from .widgets import muted, sized_font

LEVEL_NAME = {"bo1": "Best of 1", "bo3": "Best of 3", "both": "together"}
VERDICT_NAME = {"pattern": "PATTERN", "mixed": "MIXED", "rare": "RARE"}
INTRO = ("Each line counts how many sets show a clear result (a Likely or Strong badge in that set), in the same direction or the other way. "
         "Effects are never averaged across sets.  PATTERN: at least 3 clear results the same way and none the other way.  MIXED: clear results go both "
         "ways.  RARE: too few clear results to say anything.  A blank cell below means no clear result in that set and format.")


def verdict_line(item: dict) -> str:
    """One item in plain words, e.g. "Playing first: Best of 1: PATTERN, playing first did better (clear in 11 of 11 sets) | Best of 3: RARE (...)"."""
    bits = []
    for level in ("bo1", "bo3", "both"):
        t = (item.get("levels") or {}).get(level)
        if not t:
            continue
        what = VERDICT_NAME[t["verdict"]]
        if t["verdict"] == "pattern":
            what += ", " + (item_phrase(item, t["n_up"] > 0))
        bits.append(f"{LEVEL_NAME[level]}: {what} (clear in {t['n']} of {t['m']})")
    return f"{item['label']}:  " + "  |  ".join(bits)


def item_phrase(item: dict, up: bool) -> str:
    return item["up"] if up else item["down"]


def _cell(text: str, bold: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lab.setMinimumWidth(80)
    return lab


def _table(lesson: dict, sets: list[str], font) -> QWidget:
    grid = QGridLayout()
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(3)
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(2, 1)
    for c, h in enumerate(("Set", "Best of 1", "Best of 3")):
        lab = muted(QLabel(h))
        lab.setFont(sized_font(font, -1))
        grid.addWidget(lab, 0, c)
    by_cell: dict[tuple[str, str], list[str]] = {}
    if lesson["id"] == "L9":
        for r in lesson.get("rows", []):
            by_cell[(r["set"], r["fmt"])] = [f"{r['mulligan']:.1f}% of games with a mulligan, {r['turns']:.1f} turns typical"]
    else:
        for it in lesson["items"]:
            for ob in it["obs"]:
                if ob["clear"]:
                    by_cell.setdefault((ob["set"], ob["fmt"]), []).append(ob["cell"])
    for r, code in enumerate(sets, start=1):
        grid.addWidget(_cell(code), r, 0)
        for c, fmt in enumerate(("bo1", "bo3"), start=1):
            grid.addWidget(_cell("\n".join(by_cell.get((code, fmt), []))), r, c)
    host = QWidget()
    host.setLayout(grid)
    return host


def build_sections(data: dict, font, section_cls) -> list[QWidget]:
    """The sections for the Report tab (`section_cls` = its Section panel class)."""
    facts, lessons = data.get("facts") or {}, data.get("lessons") or []
    sets = list(facts.get("sets") or [])
    out: list[QWidget] = []
    head = section_cls(f"All sets: what repeats from set to set?  ({len(sets)} sets: {' '.join(sets)})")
    head.setObjectName("cross_intro")
    head.sentence(INTRO, bullet=False)
    out.append(head)
    for lesson in lessons:
        sec = section_cls(f"{lesson['id']}  {lesson['title']}")
        sec.setObjectName(f"cross_{lesson['id']}")
        sec.sentence(lesson["question"], bullet=False)
        for it in lesson["items"]:
            if it.get("levels"):
                sec.sentence(verdict_line(it))
        if lesson["id"] == "L9":
            for g in lesson.get("glance", []):
                sec.sentence(g["text"])
        sec.lay.addWidget(_table(lesson, sets, font))
        out.append(sec)
    return out
