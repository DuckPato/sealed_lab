"""The one-page cheat sheet (sealed_lab.cheatsheet): structure, content rules and the fake backend's assembly of it."""
from __future__ import annotations

import os
import re
from html.parser import HTMLParser

os.environ.setdefault("SEALED_LAB_FAKE_DELAY", "0.001")

from sealed_lab import cheatsheet as C
from sealed_lab import report
from sealed_lab.attribution import DATASETS_URL, LICENSE_URL, credit_html
from sealed_lab.gui import fake_api


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self.srcs: list[str] = []
        self.tags: list[str] = []
        self.text: list[str] = []
        self.h2: list[str] = []
        self._in_h2 = False

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for k, v in attrs:
            if k == "href" and v:
                self.urls.append(v)
            if k == "src" and v:
                self.srcs.append(v)
        self._in_h2 = tag == "h2"

    def handle_endtag(self, tag):
        if tag == "h2":
            self._in_h2 = False

    def handle_data(self, data):
        self.text.append(data)
        if self._in_h2:
            self.h2.append(data.strip())


def parse(html: str) -> Page:
    p = Page()
    p.feed(html)
    return p


# --------------------------------------------------------------------------- synthetic inputs

def info_and_values(n_per: int = 9):
    """Every colour group with `n_per` commons and uncommons (values descend with the index), some low-data cards, lands and basics."""
    info: dict[int, dict] = {}
    vals: dict[int, dict] = {}
    cid = 0
    for colors in ("W", "U", "B", "R", "G", "WU", "BR", ""):
        for rarity in ("common", "uncommon", "rare"):
            for i in range(n_per):
                name = f"{colors or 'Colourless'} {rarity} {i}"
                info[cid] = dict(name=name, colors=colors, rarity=rarity, is_land=False, is_basic=False, mana_value=2.0)
                vals[cid] = dict(beta=0.04 - 0.004 * i, low_data=(i == 0 and rarity == "common" and colors == "W"))
                cid += 1
    info[cid] = dict(name="Some Land", colors="", rarity="common", is_land=True, is_basic=False, mana_value=0.0)
    vals[cid] = dict(beta=0.5, low_data=False)
    cid += 1
    info[cid] = dict(name="Plains", colors="W", rarity="common", is_land=True, is_basic=True, mana_value=0.0)
    return info, vals


def pair_row(key, eff, badge, hidden=False, share=0.07):
    return dict(key=key, label=report.group_label(key), colors=key, share=share, n_decks=500, n_games=2000, splash_share=0.3, effect_pp=eff, se_pp=0.5,
                z=eff / 0.5, badge=badge, range="between", hidden=hidden)


def make_report(pair_rows=None, on_play=None, with_shape=True, under=None, traps=None):
    rows = pair_rows if pair_rows is not None else [pair_row("WU", 2.2, "strong"), pair_row("BR", 1.0, "likely"), pair_row("WB", 0.4, "weak"),
                                                    pair_row("RG", -1.5, "likely"), pair_row("UG", 3.0, "strong", hidden=True)]
    rng = lambda lo, med, hi: dict(lo=lo, med=med, hi=hi)
    shape_row = dict(key="all", label="All decks", n_decks=2000, n_strong=400, metrics={
        "lands": dict(all=rng(16, 17, 17), strong=rng(16, 17, 17)), "creatures": dict(all=rng(13, 15, 16), strong=rng(14, 15, 16)),
        "two_drops": dict(all=rng(3, 4, 5), strong=None), "avg_mv": dict(all=rng(2.7, 2.95, 3.2), strong=rng(2.6, 2.8, 3.0))})
    card = lambda n, v, r="rare": dict(card_id=n, name=f"Pick {n}", rarity=r, colors="W", group="W", value_pp=v, badge="strong", range="x", reason="r")
    return dict(set_code="TST", fmt="bo3", glance=dict(n_decks=2900, n_games=7500, on_play=on_play if on_play is not False else None),
                pairs=dict(rows=rows), shape=dict(rows=[shape_row], strong_def="won 3 or more matches") if with_shape else {},
                cards=dict(underrated=under if under is not None else [card(i, 3.0 - 0.1 * i) for i in range(9)],
                           traps=traps if traps is not None else [card(100 + i, -2.0 + 0.1 * i) for i in range(8)]))


PLAY = dict(wr_play=0.585, wr_draw=0.549, diff_pp=3.6, se_pp=0.5, z=7.2, badge="strong", range="x")


# --------------------------------------------------------------------------- content

def test_points_conversion_is_the_reports():
    assert C.POINTS_PER_LOGIT == report.PT == 25.0
    assert C.points(0.04) == 1.0


def test_structure_sections_and_title():
    info, vals = info_and_values()
    html = C.cheat_sheet_html("tst", "bo3", make_report(on_play=PLAY), info, vals, generated_at="2026-01-02")
    pg = parse(html)
    assert html.startswith("<!doctype html>") and "<title>TST Sealed cheat sheet — Best of 3</title>" in html
    assert "<h1>TST Sealed cheat sheet — Best of 3</h1>" in html
    assert pg.h2 == ["Best colour pairs", "Best commons and uncommons by colour", "Deck shape and play or draw", "Cards to pick up and cards to avoid"]
    assert C.cheat_sheet_html("TST", "bo1", make_report(on_play=PLAY), info, vals).count("Best of 1") >= 2
    assert "@media print" in html and "@page" in html and "size: portrait" in html                       # one portrait page, A4 or Letter
    assert "Generated 2026-01-02" in html


def test_only_the_two_credit_links_and_nothing_external():
    info, vals = info_and_values()
    html = C.cheat_sheet_html("TST", "bo3", make_report(on_play=PLAY), info, vals, generated_at="x")
    pg = parse(html)
    assert sorted(pg.urls) == sorted([DATASETS_URL, LICENSE_URL]) and not pg.srcs
    assert credit_html() in html
    assert not {"script", "img", "link", "iframe", "object"} & set(pg.tags)
    assert len(re.findall(r"https?://", html)) == 2 and "@import" not in html and "url(" not in html


def test_deterministic_with_generated_at():
    info, vals = info_and_values()
    a = C.cheat_sheet_html("TST", "bo3", make_report(on_play=PLAY), info, vals, generated_at="2026-03-04")
    b = C.cheat_sheet_html("TST", "bo3", make_report(on_play=PLAY), info, vals, generated_at="2026-03-04")
    assert a == b
    c = C.cheat_sheet_html("TST", "bo3", make_report(on_play=PLAY), info, vals)
    import datetime

    assert datetime.date.today().isoformat() in c                                                    # the default is today


def test_per_colour_limits_and_order_and_low_data():
    info, vals = info_and_values()
    top = C.top_cards(info, vals)
    assert list(top) == ["W", "U", "B", "R", "G", "other"]
    for g, d in top.items():
        assert len(d["common"]) == C.N_COMMONS == 5 and len(d["uncommon"]) == C.N_UNCOMMONS == 3, g
        for lst in d.values():
            assert [r["pts"] for r in lst] == sorted((r["pts"] for r in lst), reverse=True)
    # the white common 0 (the best by value) has little data: it is not listed, the next one leads
    assert top["W"]["common"][0]["name"] == "W common 1" and all("W common 0" != r["name"] for r in top["W"]["common"])
    assert all(r["rarity"] == "common" for g in top.values() for r in g["common"]) and not any(r["rarity"] == "rare" for g in top.values() for k in g.values() for r in k)
    # gold and colourless cards share one column; lands and basics never appear
    other = {r["name"] for r in top["other"]["common"]}
    assert any(n.startswith("WU") for n in other) or any(n.startswith("Colourless") for n in other)
    assert not any(r["name"] in ("Some Land", "Plains") for g in top.values() for k in g.values() for r in k)
    html = C.cheat_sheet_html("TST", "bo3", make_report(on_play=PLAY), info, vals, generated_at="x")
    assert "W common 0<" not in html and "W common 1<" in html and "Gold &amp; colourless" in html
    # points = beta x 25: the best card of a colour (beta 0.04, index 0 and not low-data) reads +1.0
    assert "+1.0" in html


def test_low_data_and_missing_values_are_excluded_and_a_thin_colour_says_so():
    info, vals = info_and_values()
    for cid, d in info.items():
        if d["colors"] == "G":
            vals[cid]["low_data"] = True
    for cid, d in info.items():
        if d["colors"] == "R":
            del vals[cid]
    top = C.top_cards(info, vals)
    assert top["G"] == {"common": [], "uncommon": []} and top["R"] == {"common": [], "uncommon": []}
    html = C.cheat_sheet_html("TST", "bo3", make_report(on_play=PLAY), info, vals, generated_at="x")
    assert html.count("none with enough data") >= 4


def test_best_pairs_clear_ones_else_top_three():
    rep = make_report()
    rows, clear = C.best_pairs(rep)
    assert clear and [r["key"] for r in rows] == ["WU", "BR"]                                         # Strong / Likely and positive; hidden and weak out
    flat = make_report(pair_rows=[pair_row("WU", 0.9, "weak"), pair_row("BR", 0.5, "weak"), pair_row("WB", 0.4, "weak"), pair_row("RG", -0.3, "weak"),
                                  pair_row("UG", -0.5, "weak")])
    rows, clear = C.best_pairs(flat)
    assert not clear and [r["key"] for r in rows] == ["WU", "BR", "WB"]
    info, vals = info_and_values()
    html = C.cheat_sheet_html("TST", "bo3", flat, info, vals, generated_at="x")
    assert "No pair stands out clearly" in html and "Weak" in html
    html2 = C.cheat_sheet_html("TST", "bo3", rep, info, vals, generated_at="x")
    assert "White-Blue" in html2 and "Strong" in html2 and "Likely" in html2 and "Only the pairs with a clear" in html2
    many = make_report(pair_rows=[pair_row(k, 3.0 - i * 0.1, "strong") for i, k in enumerate(["WU", "WB", "WR", "WG", "UB", "UR", "UG"])])
    assert len(C.best_pairs(many)[0]) == C.N_PAIRS_MAX


def test_shape_playdraw_and_lists():
    info, vals = info_and_values()
    html = C.cheat_sheet_html("TST", "bo3", make_report(on_play=PLAY), info, vals, generated_at="x")
    assert "16–17 (typical 17)" in html and "14–16 (typical 15)" in html            # the decks with good results (strong ranges)
    assert "3–5 (typical 4)" in html                                                      # two-drops: too few strong decks, all decks instead
    assert "2.6–3.0 (typical 2.8)" in html and "400 decks that won 3 or more matches" in html
    assert "Choose to play first: 58.5% of games won on the play against 54.9% on the draw (+3.6 points)." in html
    assert "Choose to draw" in C.play_draw_note(make_report(on_play=dict(PLAY, diff_pp=-2.0, wr_play=0.5, wr_draw=0.52)))
    assert "No clear edge" in C.play_draw_note(make_report(on_play=dict(PLAY, badge="weak", diff_pp=0.3)))
    assert "No play-or-draw figure" in C.play_draw_note(make_report(on_play=False))
    # underrated and traps: six each, best (most negative) first, with their rarity
    pg = parse(html)
    text = " ".join(pg.text)
    for i in range(6):
        assert f"Pick {i}" in text and f"Pick {100 + i}" in text
    assert "Pick 6" not in text and "Pick 106" not in text
    assert text.index("Pick 100") < text.index("Pick 105")
    # nothing to list, no shape: the page still has every section
    bare = C.cheat_sheet_html("TST", "bo3", make_report(on_play=False, with_shape=False, under=[], traps=[]), info, vals, generated_at="x")
    assert "Nothing clear enough to list." in bare and "No rare or mythic is clearly a trap." in bare and "No deck-shape figures" in bare
    assert parse(bare).h2 == pg.h2


def test_escaping():
    info, vals = info_and_values()
    info[0]["name"] = 'Bad <b>"Name"</b> & Co'
    vals[0] = dict(beta=0.9, low_data=False)
    html = C.cheat_sheet_html("TST", "bo3", make_report(on_play=PLAY), info, vals, generated_at="<x>")
    assert "&lt;b&gt;" in html and "<b>&quot;Name" not in html and "<b>\"Name\"</b>" not in html and "Generated &lt;x&gt;" in html


# --------------------------------------------------------------------------- with the fake backend

def test_fake_backend_cheat_sheet_matches_its_report():
    fake_api.reset()
    for fmt in ("bo1", "bo3"):
        html = fake_api.cheat_sheet("TST", fmt)
        rep = fake_api.set_report("TST", fmt)
        pg = parse(html)
        assert pg.h2 and f"TST Sealed cheat sheet — Best of {fmt[-1]}" in html
        rows, _ = C.best_pairs(rep)
        assert all(r["label"] in html for r in rows)
        for r in (rep["cards"]["traps"] or [])[:C.N_LISTED]:
            assert r["name"].replace("&", "&amp;") in html or r["name"] in html
        assert sorted(pg.urls) == sorted([DATASETS_URL, LICENSE_URL])
    assert "TST" in fake_api.cheat_sheet("tst", "bo3")
