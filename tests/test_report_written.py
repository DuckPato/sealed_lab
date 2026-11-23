"""The written report: `cli report-install` (refuses a text the checker rejects), the api, and the Markdown -> HTML conversion."""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sealed_lab import api, cli, config, written_report as W
from sealed_lab.gui import fake_api
from sealed_lab.gui.written_tab import markdown_to_html

FACTS = fake_api._FAKE_WRITTEN_FACTS


def _files(tmp_path, text: str):
    r, f = tmp_path / "TST_report.md", tmp_path / "facts.json"
    r.write_text(text, encoding="utf-8")
    f.write_text(json.dumps(FACTS), encoding="utf-8")
    return r, f


def test_install_refuses_a_report_failing_the_checker(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    r, f = _files(tmp_path, "Playing first wins 9.9 points more often.<!--F:PLAY.bo1-->\n")        # 9.9 is not a number of the fact
    assert cli.main(["report-install", "TST", str(r), str(f)]) == 1
    assert "does not match" in capsys.readouterr().out
    assert not (config.app_home() / "written_reports").exists()
    r2, f2 = _files(tmp_path, "White-Blue wins more often.\n")                                       # untagged direction word
    assert cli.main(["report-install", "TST", str(r2), str(f2)]) == 1
    assert W.available() == []


def test_install_writes_text_and_meta_and_api_reads_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    r, f = _files(tmp_path, fake_api.FAKE_WRITTEN_TEXT)
    assert cli.main(["report-install", "TST", str(r), str(f)]) == 0
    d = config.app_home() / "written_reports"
    assert (d / "TST.md").read_text(encoding="utf-8") == fake_api.FAKE_WRITTEN_TEXT
    meta = json.loads((d / "TST.json").read_text(encoding="utf-8"))
    assert meta["stamp"] == {"bo1": "fake|bo1"} and meta["written_at"] and meta["facts"]["facts"]["PLAY.bo1"]
    res = api.written_report("TST")
    assert res["text"] == fake_api.FAKE_WRITTEN_TEXT and res["available"] == ["TST"] and res["outdated"] is False   # no cached analysis: unknown
    assert api.written_report("XXX")["text"] is None


def test_outdated_compares_the_stamps(tmp_path, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    meta = {"stamp": {"bo1": "a|v1"}}
    assert W.outdated(meta, {"bo1": "b|v1"}) and not W.outdated(meta, {"bo1": "a|v1"}) and not W.outdated(meta, {})
    cache = config.cache_dir("TST")
    cache.mkdir(parents=True)
    (cache / "report_bo1.json").write_text(json.dumps({"stamp": "b|v1"}), encoding="utf-8")
    assert W.current_stamps("TST") == {"bo1": "b|v1"}


def test_markdown_tooltips_on_tagged_sentences():
    h = markdown_to_html(fake_api.FAKE_WRITTEN_TEXT, FACTS)
    assert "<h3>Headlines</h3>" in h and "<li><span title=" in h
    assert "playing first wins 3.6 points more often" in h and "[Strong]" in h and "no clear difference" in h
    assert "<i>this paragraph</i>" in h
    assert "<!--" not in h


def test_fact_ids_map_to_the_sections_of_the_report_tab():
    from sealed_lab.gui.written_tab import fact_target

    cases = {
        "GLANCE.bo1": ("bo1", "glance", None, None), "PLAY.bo3": ("bo3", "glance", None, None),
        "PAIR.bo1.WU": ("bo1", "pairs", None, None), "PAIR.bo1.rest": ("bo1", "pairs", None, None), "SPLASH.bo3": ("bo3", "pairs", None, None),
        "PAIRDET.bo3.WU.PLAY": ("bo3", "pairs", "pairs", None), "PAIRDET.bo1.WU.SPLASH.R": ("bo1", "pairs", "pairs", None),
        "CARD.bo3.tide_ghoul": ("bo3", "cards", None, None), "SPEC.bo1.UB": ("bo1", "archetypes", None, "UB"),
        "SHAPE.bo1.all": ("bo1", "shape", None, None), "SHAPEDET.bo3.lands.L17": ("bo3", "shape", "shape", None),
        "BUILDER.bo1.lands": ("bo1", "shape", None, None), "BUILDERCHECK.bo3": ("bo3", "shape", None, None),
        "EVOL.bo1.PAIR.WU": ("bo1", "evolution", None, None), "EVOL.bo3.CARD.x": ("bo3", "evolution", None, None),
        "XSET.bo1.PLAY": (None, "xset", None, None),
    }
    for fid, want in cases.items():
        assert fact_target(fid) == want, fid
    assert fact_target("NOPE.bo1") is None


def test_card_names_become_anchors_longest_first_and_sentences_are_listed():
    from sealed_lab.gui.written_tab import card_pattern, render_markdown

    cards = card_pattern(["Dawn Charm", "Dawn Charm of Aldra", "Ox", "Jötun's Rage, the Unbowed"])
    text = ("- Dawn Charm of Aldra and Dawn Charm win, as does Jötun's Rage, the Unbowed, not Ox or Dawn Charms.<!--F:PLAY.bo1-->\n"
            "Plain *words* about Dawn Charm.\n")
    html, sents = render_markdown(text, FACTS, cards)
    assert '<a href="card:Dawn%20Charm%20of%20Aldra">Dawn Charm of Aldra</a>' in html
    assert '<a href="card:Dawn%20Charm">Dawn Charm</a> win' in html and "Dawn Charms" in html and 'href="card:Dawn%20Charms"' not in html
    assert "card:J%C3%B6tun%27s%20Rage%2C%20the%20Unbowed" in html and "card:Ox" not in html         # (names under 4 letters are not looked for)
    assert "<i>words</i>" in html and html.count("<a ") == 4
    assert sents == [("Dawn Charm of Aldra and Dawn Charm win, as does Jötun's Rage, the Unbowed, not Ox or Dawn Charms.", ["PLAY.bo1"])]
    assert "<a " not in markdown_to_html(text, FACTS)                                           # no cards given: no links
