"""The facts file for the written report (sealed_lab.report_facts), the CLI, and the deterministic checker (scripts/check_report.py)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from sealed_lab import cli, report as R, report_facts as RF

_SPEC = importlib.util.spec_from_file_location("check_report", Path(__file__).resolve().parents[1] / "scripts" / "check_report.py")
CK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CK)


@pytest.fixture(scope="module")
def reports():
    from sealed_lab.gui import fake_api

    return {f: fake_api.set_report("TST", f) for f in RF.FORMATS}


@pytest.fixture(scope="module")
def facts(reports):
    return RF.facts_from_reports("TST", reports)


# --------------------------------------------------------------------------- trap rares in the report

def test_trap_rares_rule(reports):
    for fmt, rep in reports.items():
        cs = rep["cards"]
        assert len(cs["traps"]) <= R.TRAP_MAX
        vals = [r["value_pp"] for r in cs["traps"]]
        assert vals == sorted(vals)                                           # most negative first
        for r in cs["traps"]:
            assert r["rarity"] in R.TRAP_RARITIES and r["value_pp"] < 0 and r["badge"] in ("likely", "strong")
            assert r["play_rate"] >= cs["play_rate_cuts"][1] and "loses" in r["reason"]
    assert reports["bo1"]["cards"]["traps"]                                   # the fake backend plants some


# --------------------------------------------------------------------------- facts content

def test_facts_only_claim_likely_or_strong_and_ids_are_stable(reports, facts):
    assert json.loads(json.dumps(facts)) == facts
    assert facts == RF.facts_from_reports("TST", reports)                   # deterministic
    assert facts["facts"] and facts["noclear"]
    for fid, f in facts["facts"].items():
        assert f["id"] == fid and f["badge"] in ("likely", "strong", "descriptive")             # never a Weak claim
        assert f["builder_uses"] in ("yes", "observed") and f["nums"] and f["text"]
        assert fid.split(".")[1] in RF.FORMATS
    assert not set(facts["facts"]) & set(facts["noclear"])
    assert "PLAY.bo1" in facts["facts"] and any(i.startswith("PAIR.bo1.") for i in facts["facts"])
    assert any(i.startswith("SPEC.") for i in facts["facts"]) and "SHAPE.bo1.all" in facts["facts"]
    kinds = {f["kind"] for f in facts["facts"].values()}
    assert {"card_trap", "card_under", "card_over"} <= kinds
    for f in facts["facts"].values():
        if f["kind"].startswith("card_"):
            assert f["builder_uses"] == "yes"
        elif f["kind"] in ("pair", "splash", "spectrum", "play"):
            assert f["builder_uses"] == "observed"
    ids = ["PAIR.bo1.BR", "SPLASH.bo1", "SPEC.bo1.WR", "CARD.bo1.dawn_knight", "SHAPE.bo1.all", "PLAY.bo1"]          # (stable names)
    assert all(i in facts["facts"] for i in ids)


def test_weak_items_are_listed_as_no_clear_difference(reports, facts):
    weak_pairs = [r["key"] for r in reports["bo1"]["pairs"]["rows"] if not r["hidden"] and r["badge"] == "weak"]
    assert weak_pairs
    for k in weak_pairs:
        assert f"PAIR.bo1.{k}" in facts["noclear"] and f"PAIR.bo1.{k}" not in facts["facts"]
    shown_weak = [a["key"] for a in reports["bo1"]["archetypes"]["pairs"] if a["spectrum"].get("shown") and not a["spectrum"]["win"]["finding"]]
    for k in shown_weak:
        assert f"SPEC.bo1.{k}" in facts["noclear"]


def test_spectrum_fact_names_the_winning_end(facts):
    f = facts["facts"]["SPEC.bo1.WR"]
    assert f["ends"]["winner"] in (f["ends"]["a"], f["ends"]["b"]) and f["nums"] and min(f["nums"]) > 0       # seen from the winning end


def test_text_is_compact_and_lists_every_id(facts):
    t = RF.render_text(facts)
    assert len(t.encode("utf-8")) < 10000                                  # (the synthetic set has far more clear results than a real one)
    for fid in facts["facts"]:
        assert f"[{fid}]" in t
    for fid in facts["noclear"]:
        assert fid in t
    assert "no stored comparison" in t                                      # no builder results stored for TST


def test_builder_facts_from_stored_results(reports, tmp_path):
    comp = {"set": "TST", "formats": {"bo1": {"n_pools": 100, "n_real": 400, "n_top": 50,
            "lands": dict(builder_mean=16.0, players_mean=17.0, diff_players=-1.0, diff_players_se=0.05),
            "creatures": dict(builder_mean=14.1, players_mean=14.0, diff_players=0.1, diff_players_se=0.2)}}}
    p = tmp_path / "builder_composition_TST.json"
    p.write_text(json.dumps(comp), encoding="utf-8")
    assert RF.load_builder_composition("TST", p)["set"] == "TST" and RF.load_builder_composition("TST", tmp_path / "none.json") is None
    f = RF.facts_from_reports("TST", reports, RF.load_builder_composition("TST", p))
    assert f["builder_available"] and f["facts"]["BUILDER.bo1.lands"]["builder_uses"] == "yes"
    assert "BUILDER.bo1.creatures" in f["noclear"] and "BUILDER.bo1.lands" not in f["noclear"]
    assert "[BUILDER.bo1.lands]" in RF.render_text(f)


def test_cli_report_facts(tmp_path, monkeypatch, capsys, facts):
    monkeypatch.setattr(RF, "build_facts", lambda sc, refresh=False, progress=None: facts)
    out = tmp_path / "f.json"
    assert cli.main(["report-facts", "TST", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["set"] == "TST" and out.with_suffix(".txt").read_text(encoding="utf-8").startswith("FACTS TST")
    assert cli.main(["report-facts", "TST"]) == 0
    assert "NOCLEAR" in capsys.readouterr().out


# --------------------------------------------------------------------------- the checker

def sample(facts) -> str:
    pb = facts["facts"]["PAIR.bo1.BR"]
    spec = facts["facts"]["SPEC.bo1.WR"]
    trap = facts["facts"]["CARD.bo1.dawn_knight"]
    play = facts["facts"]["PLAY.bo1"]
    weak = next(i for i in facts["noclear"] if i.startswith("PAIR.bo1."))
    wk = weak.split(".")[-1]
    return "\n".join([
        "# Report",
        "## Headlines",
        f"- Black-Red decks do {'better' if pb['value'] > 0 else 'worse'} than an average deck, by about {abs(pb['value']):.1f} points.<!--F:PAIR.bo1.BR-->",
        f"- Playing first is worth about {play['nums'][-2]:.1f} points in Best of 1.<!--F:PLAY.bo1-->",
        f"The {spec['ends']['winner']} end of White-Red wins {spec['nums'][0]:.1f} points more.<!--F:SPEC.bo1.WR-->",
        f"Dawn Knight is a trap rare: it loses about {abs(trap['nums'][1]):.1f} points per copy.<!--F:CARD.bo1.dawn_knight-->",
        f"The data show no clear difference for the {wk} colour group.<!--F:{weak}-->",
        "Drafting is a game of choices.",
    ])


def test_checker_accepts_a_good_sample(facts):
    assert CK.check(sample(facts), facts) == []


def test_checker_rejects_an_untagged_claim(facts):
    p = CK.check(sample(facts) + "\nBlue decks win more often in the late game.", facts)
    assert len(p) == 1 and "untagged" in p[0]
    p2 = CK.check(sample(facts) + "\nAbout 55 decks were studied.", facts)
    assert len(p2) == 1 and "untagged" in p2[0]


def test_checker_rejects_a_wrong_number(facts):
    pb = facts["facts"]["PAIR.bo1.BR"]
    wrong = f"Black-Red decks do worse than an average deck, by about {abs(pb['value']) + 1.7:.1f} points.<!--F:PAIR.bo1.BR-->"
    p = CK.check(wrong, facts)
    assert len(p) == 1 and "does not match" in p[0]
    assert CK.check(f"Black-Red decks do worse, by about {abs(pb['value']):.1f} points.<!--F:PAIR.bo1.BR-->", facts) == []
    sgn = "+" if pb["value"] < 0 else "-"                                           # the opposite of the real sign
    assert CK.check(f"Black-Red decks are at {sgn}{abs(pb['value']):.1f} points.<!--F:PAIR.bo1.BR-->", facts)
    assert CK.check("Black-Red decks are 9% of the decks.<!--F:PAIR.bo1.BR-->", facts)


def test_checker_rejects_an_unknown_id(facts):
    p = CK.check("Black-Red decks win more.<!--F:PAIR.bo1.ZZ-->", facts)
    assert len(p) == 1 and "unknown fact id PAIR.bo1.ZZ" in p[0]
    assert CK.check("Black-Red decks win more.<!--F:PAIR.bo1.BR,PAIR.bo1.ZZ-->", facts)


def test_checker_rejects_a_direction_claim_on_a_no_clear_fact(facts):
    weak = next(i for i in facts["noclear"] if i.startswith("PAIR.bo1."))
    k = weak.split(".")[-1]
    p = CK.check(f"{k} decks win more than the others.<!--F:{weak}-->", facts)
    assert len(p) == 1 and "no-clear" in p[0]
    assert CK.check(f"There is no clear difference between {k} and the others.<!--F:{weak}-->", facts) == []
    assert CK.check(f"{k} decks show no difference in who wins more.<!--F:{weak}-->", facts) == []


def test_checker_number_rules_and_length(facts):
    assert CK.numbers_of("Bo1 and Best of 3 and 11,050 decks and 12% and +1.4")[0][0] == 11050.0
    assert [(v, s) for v, _d, s, _t in CK.numbers_of("13-16 and -1.4")] == [(13.0, False), (16.0, False), (-1.4, True)]
    assert CK.number_ok(3.0, 0, False, [2.6]) and not CK.number_ok(3.0, 1, False, [2.6]) and CK.number_ok(1.4, 1, False, [-1.36])
    long = " ".join(["word"] * 300)
    assert any("length" in p for p in CK.check(long, facts, check_length=True)) and not CK.check(long, facts)


def test_checker_cli_exit_codes(facts, tmp_path, capsys):
    fj = tmp_path / "f.json"
    fj.write_text(json.dumps(facts), encoding="utf-8")
    good, bad = tmp_path / "g.md", tmp_path / "b.md"
    good.write_text(sample(facts), encoding="utf-8")
    bad.write_text(sample(facts) + "\nRed is the best colour.", encoding="utf-8")
    assert CK.main([str(good), str(fj)]) == 0 and "OK" in capsys.readouterr().out
    assert CK.main([str(bad), str(fj)]) == 1 and "untagged" in capsys.readouterr().out


def test_builder_check_facts():
    """BUILDERCHECK.<fmt>: a claim only with a Likely/Strong combined result, otherwise a NOCLEAR item; skipped formats give nothing."""
    from sealed_lab import report_facts as F
    chk = {"formats": {"bo1": {"n_games": 50000, "combined": {"pp": 0.85, "se_total": 0.39, "badge": "Likely"}},
                       "bo3": {"n_games": 7000, "combined": {"pp": 2.4, "se_total": 1.3, "badge": "Weak"}}}}
    facts, noclear = [], []
    F._builder_check_fact(chk["formats"]["bo1"], "bo1", facts, noclear)
    F._builder_check_fact(chk["formats"]["bo3"], "bo3", facts, noclear)
    F._builder_check_fact({"skipped": "too few games"}, "bo3", facts, noclear)
    assert [f["id"] for f in facts] == ["BUILDERCHECK.bo1"] and facts[0]["badge"] == "likely"
    assert 0.85 in facts[0]["nums"] and "+0.1 to +1.6" in facts[0]["text"] and "not a test of the builder" in facts[0]["text"]
    assert [n["id"] for n in noclear] == ["BUILDERCHECK.bo3"]
