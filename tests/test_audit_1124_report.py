"""Audit 1124 A3: the written-report checker must reject wrong-sign, wrong-role, spelled-out and untagged claims, and see every rendered line."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sealed_lab import cheatsheet, export_html
from sealed_lab.report_check import check, split_sentences

FACTS = {"facts": {
    "PAIR.bo1.BG": {"text": "Black-Green (12.3% of decks) -2.1 points against an average deck (range -3.5 to -0.7), Likely",
                    "nums": [-3.5, -2.1, -0.7, 12.3, 40.0], "value": -2.1},
    "PAIR.bo1.UB": {"text": "Blue-Black (8.0% of decks) +2.0 points against an average deck (range +0.8 to +3.2), Strong",
                    "nums": [0.8, 2.0, 3.2, 8.0, 30.0], "value": 2.0},
    "SPLASH.bo1": {"text": "30% of decks splash; splashing costs -1.2 points (range -2.0 to -0.4), Likely", "nums": [-2.0, -1.2, -0.4, 30.0], "value": -1.2},
    "SPLASH.bo3": {"text": "splashing pays off +4.6 points (range +1.0 to +8.2), Strong", "nums": [1.0, 4.6, 8.2], "value": 4.6},
    "SHAPEDET.bo1.lands.L17": {"text": "decks with 17 lands (50% of decks) +0.5 points against an average deck (range +0.1 to +0.9), Likely",
                               "nums": [0.1, 0.5, 0.9, 50.0], "value": 0.5},
}, "noclear": {"PLAY.bo1": {"text": "playing first or second"}, "PAIR.bo1.RG": {"text": "Red-Green"}}}


def bad(text: str) -> list[str]:
    return check(text, FACTS)


@pytest.mark.parametrize("text", [
    "Black-Green decks win about 2.1 points more often than an average deck.<!--F:PAIR.bo1.BG-->",       # sign
    "Black-Green decks do best: 12.3 points better than an average deck.<!--F:PAIR.bo1.BG-->",           # a share, wrong sign
    "Blue-Black decks do best: 8.0 points better than an average deck.<!--F:PAIR.bo1.UB-->",             # a share as the effect
    "Splashing pays off: about 1.2 points.<!--F:SPLASH.bo1-->",
    "Splashing pays off handsomely, adding about five points.<!--F:SPLASH.bo1-->",
    "Splashing in Best of 3 adds about nine points.<!--F:SPLASH.bo3-->",
    "Blue-Black is the strongest color pair in Bo1.",
    "Playing first is clearly the stronger choice.<!--F:PLAY.bo1-->",
    "Playing first has no clear edge, but it is better.<!--F:PLAY.bo1-->",
    "<!-- note --> Red decks win 9 points more often. <!-- end -->",
    "## Red-Green wins by 9.9 points",
    "Blue-Black decks win about 3 points less often.<!--F:PAIR.bo1.UB-->",
])
def test_rejects(text):
    assert bad(text), text


@pytest.mark.parametrize("text", [
    "Black-Green decks win about 2.1 points less often than an average deck.<!--F:PAIR.bo1.BG-->",
    "Black-Green decks are 12.3% of all decks and do worse by 2.1 points.<!--F:PAIR.bo1.BG-->",
    "Blue-Black decks win about two points more often than an average deck.<!--F:PAIR.bo1.UB-->",
    "Blue-Black decks win about 2 points more, probably between +0.8 and +3.2 points.<!--F:PAIR.bo1.UB-->",
    "Splashing costs about 1.2 points.<!--F:SPLASH.bo1-->",
    "Splashing pays off by about five points in Best of 3.<!--F:SPLASH.bo3-->",
    "Splashing costs 1.2 points in Best of 1, but pays off 4.6 points in Best of 3.<!--F:SPLASH.bo1,SPLASH.bo3-->",
    "The data show no clear difference between playing first or second in Bo1.<!--F:PLAY.bo1-->",
    "Decks with three or more main colors do worse by 1.2 points.<!--F:SPLASH.bo1-->",
    "Decks with seventeen lands probably do 0.5 points better.<!--F:SHAPEDET.bo1.lands.L17-->",
    "A \"point\" is one percentage point of the chance to win a game.",
    "## Best archetypes and how to build them\n\n<!-- a note -->\n",
])
def test_accepts(text):
    assert bad(text) == [], text


def test_comment_wrapped_text_and_headings_are_split_like_the_renderer():
    text = "<!-- hidden -->\n<!-- note --> Red wins 9 points more. <!-- end -->\n## Red-Green wins by 9.9 points\n## Headlines\n"
    sentences = [s for _, s in split_sentences(text)]
    assert sentences[0] == "<!-- note --> Red wins 9 points more. <!-- end -->"
    assert any(s.startswith("Red-Green wins") for s in sentences)
    assert "Headlines" not in sentences
    rendered = export_html.markdown_to_html("<!-- note --> Red wins 9 points more. <!-- end -->", {})
    assert "Red wins 9 points more" in rendered          # what is rendered is what is checked


def test_installed_reports_pass_except_known_untagged_intro():
    root = Path(__file__).resolve().parents[1] / "docs" / "report_writer" / "reports"
    for rep in sorted(root.glob("*_report.md")):
        facts = json.loads((root / f"facts_{rep.stem.split('_')[0]}.json").read_text(encoding="utf-8"))
        problems = check(rep.read_text(encoding="utf-8"), facts)
        assert all("untagged claim (direction word 'lose')" in p for p in problems), (rep.name, problems)


def test_cheatsheet_says_which_scale_each_list_uses():
    assert "not measured against the average card" in cheatsheet.LOGIT_NOTE and "against the average card of their color" in cheatsheet.LOGIT_NOTE
