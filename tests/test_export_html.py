"""The HTML export of a set (sealed_lab.export_html): built from the fake backend's synthetic report and written report."""
from __future__ import annotations

import copy
from html.parser import HTMLParser

import pytest

from sealed_lab import export_html
from sealed_lab.gui import fake_api

VOID = {"meta", "br", "hr", "img", "input", "link"}
TRACKED = {"main", "section", "table", "details", "ul", "p", "div", "tr", "td", "th", "h2", "h3", "header", "script", "style"}


class Check(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.urls: list[str] = []
        self.text: list[str] = []
        self.titles: list[str] = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ("src", "href") and v and not v.startswith("#"):
                self.urls.append(v)
            if k == "title" and v:
                self.titles.append(v)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()

    def handle_data(self, data):
        self.text.append(data)


def parse(html: str) -> Check:
    c = Check()
    c.feed(html)
    c.close()
    return c


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0")


@pytest.fixture()
def page():
    return fake_api.export_set_html("TST")


def test_sections_both_formats_and_written_report(page):
    for needle in ("TST Sealed report", "The written report", "Analysis: Best of 1", "Analysis: Best of 3", "The set at a glance", "Colour pairs: which decks win?",
                   "Cards: where players misjudge", "Underrated: players leave them out", "Overrated: players love them", "Trap rares", "Best commons by colour",
                   "Best uncommons by colour", "Archetypes: how each colour pair is played", "Winning deck shape", "How the format changed",
                   "Best of 1 against Best of 3", "About these numbers", "More detail: all cards", "How sure?", "Strong", "Likely", "Weak",
                   "does not use archetypes", "Data: 2026-06-23 to 2026-07-21", "Generated:", "Model built:"):
        assert needle in page, needle
    assert "White-Blue is the best pair" in page and "<!--" not in page and "F:PAIR" not in page
    # the tooltip of a tagged sentence carries the fact text and badge
    assert "White-Blue: +2.2 points (range +0.9 to +3.5)  [Likely]" in page
    assert "the other pairs: no clear difference  [no clear difference]" in page
    assert "Outdated" not in page


def test_outdated_and_missing_written_report():
    w = copy.deepcopy(fake_api.written_report("TST"))
    w["outdated"] = True
    reps = {"bo1": fake_api.set_report("TST", "bo1"), "bo3": None}
    h = export_html.export_set_html("TST", w, reps)
    assert export_html.OUTDATED in h and "Not computed" in h and "Analysis: Best of 3" in h
    h = export_html.export_set_html("TST", {"text": None, "meta": None, "outdated": False}, reps)
    assert export_html.NO_REPORT in h


def test_escapes_card_names_and_tooltips():
    rep = fake_api.set_report("TST", "bo3")
    rep = copy.deepcopy(rep)
    nasty = 'Bob\'s "Best" <b>Card</b> & Co'
    rep["cards"]["underrated_by"]["W"][0]["name"] = nasty
    rep["cards"]["explorer"]["rows"][0][1] = nasty
    w = {"text": 'A "quoted" <script>x</script> sentence & more.<!--F:X-->\n', "meta": {"facts": {"facts": {"X": {"text": 'say "hi" <i>&', "badge": "strong"}}}}, "outdated": False}
    h = export_html.export_set_html("TST", w, {"bo3": rep})
    assert nasty not in h and export_html.e(nasty) in h
    assert "<b>Card</b>" not in h and "<script>x</script>" not in h
    assert "&lt;script&gt;x&lt;/script&gt;" in h
    c = parse(h)
    assert not c.errors and any(t.startswith('say "hi" <i>&') for t in c.titles)


def test_well_formed_and_no_external_urls(page):
    c = parse(page)
    assert c.errors == [] and c.stack == []
    from sealed_lab.attribution import DATASETS_URL, LICENSE_URL       # the only outside URLs: the 17Lands credit's two links (nothing loaded)
    assert sorted(c.urls) == sorted([DATASETS_URL, LICENSE_URL])
    assert "http" not in page.lower().replace("<!doctype html>", "").replace(DATASETS_URL, "").replace(LICENSE_URL, "")
    assert " src=" not in page
    assert "@media print" in page and "max-width:900px" in page
    assert page.count("<main>") == 1 and page.count("</main>") == 1 and page.count("<table") == page.count("</table>")
    assert page.count("<details") == page.count("</details>")


def test_missing_formats_only_listed_ones():
    h = export_html.export_set_html("tst", None, {"bo3": fake_api.set_report("TST", "bo3")})
    assert "Analysis: Best of 3" in h and "Analysis: Best of 1" not in h and "TST Sealed report" in h
    assert export_html.NO_REPORT in h


def test_export_credits_17lands():
    """CC BY 4.0 + the 17Lands usage guidelines: the export names 17Lands with links to the datasets and the license."""
    from sealed_lab.attribution import DATASETS_URL, LICENSE_URL
    h = export_html.export_set_html("TST", None, {"bo3": fake_api.set_report("TST", "bo3")})
    head = h[:h.index("</header>")]
    assert DATASETS_URL in head and LICENSE_URL in head and "not affiliated" in head
