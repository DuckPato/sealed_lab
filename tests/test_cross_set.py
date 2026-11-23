"""Cross-set lessons (sealed_lab/cross_set.py): the pre-registered rules R1-R4 on synthetic per-set facts, the facts structure the checker needs,
the CLI / install path for the pseudo-set "ALL", and the stamps."""
from __future__ import annotations

import json
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from sealed_lab import api, cli, config, cross_set as X, report_check, report_facts, written_report as W
from sealed_lab.gui import fake_api
from sealed_lab.gui.fake_api import synthetic_facts as SF


def sets_of(**per_set):
    return {c: SF(c, claims, noclear) for c, (claims, noclear) in per_set.items()}


def lesson(res: dict, key: str, level: str) -> dict:
    for ls in res["lessons"]:
        for it in ls["items"]:
            if it["key"] == key:
                return it["levels"][level]
    raise KeyError(key)


def play_sets(values: dict[str, tuple[str, float] | None], fmt="bo1") -> dict:
    """{set: (badge, value) | None (no clear result: a NOCLEAR item)} -> per-set facts with only PLAY."""
    return {c: (SF(c, [(f"PLAY.{fmt}", v[0], v[1])]) if v else SF(c, [], [f"PLAY.{fmt}"])) for c, v in values.items()}


# ------------------------------------------------------------------------------------------ R3: pattern / mixed / rare

def test_verdict_rule():
    assert X.verdict_of(3, 0) == "pattern" and X.verdict_of(0, 5) == "pattern" and X.verdict_of(11, 0) == "pattern"
    assert X.verdict_of(3, 1) == "mixed" and X.verdict_of(1, 1) == "mixed" and X.verdict_of(2, 1) == "mixed"
    assert X.verdict_of(2, 0) == "rare" and X.verdict_of(1, 0) == "rare" and X.verdict_of(0, 0) == "rare"


def test_pattern_is_a_likely_fact_with_in_n_of_m():
    res = X.cross_from_facts(play_sets({"AAA": ("strong", 5.0), "BBB": ("likely", 3.0), "CCC": ("likely", 4.0), "DDD": None}))
    f = res["facts"]["facts"]["XSET.bo1.PLAY"]
    assert f["badge"] == "likely" and f["verdict"] == "pattern" and (f["n_clear"], f["m"]) == (3, 4)
    assert "clear in 3 of 4 sets" in f["text"] and "A pattern" in f["text"] and "+3.0 to +5.0" in f["text"]
    assert {3, 4, 5} <= set(f["nums"]) and "AAA, BBB, CCC" in f["text"]


def test_mixed_is_a_fact_that_says_mixed():
    res = X.cross_from_facts(play_sets({"AAA": ("strong", 5.0), "BBB": ("likely", 3.0), "CCC": ("likely", 4.0), "DDD": ("likely", -2.0)}))
    f = res["facts"]["facts"]["XSET.bo1.PLAY"]
    assert f["verdict"] == "mixed" and f["badge"] == "likely" and "Mixed" in f["text"] and "-2.0" in f["text"]
    assert "XSET.bo1.PLAY" not in res["facts"]["noclear"]


def test_rare_goes_to_noclear_only():
    res = X.cross_from_facts(play_sets({"AAA": ("strong", 5.0), "BBB": ("likely", 3.0), "CCC": None}))
    assert "XSET.bo1.PLAY" not in res["facts"]["facts"] and "XSET.bo1.PLAY" in res["facts"]["noclear"]
    assert res["facts"]["noclear"]["XSET.bo1.PLAY"]["fmt"] == "bo1"
    assert lesson(res, "PLAY", "bo1")["verdict"] == "rare"
    res0 = X.cross_from_facts(play_sets({"AAA": None, "BBB": None}))
    assert "XSET.bo1.PLAY" in res0["facts"]["noclear"]


def test_weak_is_never_counted():
    per = play_sets({"AAA": ("strong", 5.0), "BBB": ("likely", 3.0), "CCC": ("weak", 4.0), "DDD": ("weak", -4.0)})
    res = X.cross_from_facts(per)
    t = lesson(res, "PLAY", "bo1")
    assert (t["n"], t["m"], t["verdict"]) == (2, 4, "rare")          # weak claims exist (M) but are not clear
    assert "XSET.bo1.PLAY" not in res["facts"]["facts"]
    per["EEE"] = SF("EEE", [("PLAY.bo1", "likely", 2.0)])
    assert lesson(X.cross_from_facts(per), "PLAY", "bo1")["verdict"] == "pattern"          # three Likely+ now; the weak -4.0 does not make it mixed


def test_counts_per_format_and_together():
    per = {"AAA": SF("AAA", [("PLAY.bo1", "strong", 5.0), ("PLAY.bo3", "likely", 3.0)]),
           "BBB": SF("BBB", [("PLAY.bo1", "likely", 4.0)], ["PLAY.bo3"]),
           "CCC": SF("CCC", [("PLAY.bo1", "likely", 6.0), ("PLAY.bo3", "likely", 2.5)]),
           "DDD": SF("DDD", [], ["PLAY.bo1"], formats=("bo1",))}        # DDD has no Bo3
    res = X.cross_from_facts(per)
    b1, b3, both = (lesson(res, "PLAY", k) for k in ("bo1", "bo3", "both"))
    assert (b1["n"], b1["m"]) == (3, 4) and (b3["n"], b3["m"]) == (2, 3) and (both["n"], both["m"]) == (5, 7)
    assert b1["verdict"] == "pattern" and b3["verdict"] == "rare" and both["verdict"] == "pattern"
    assert both["up"]["sets"] == ["AAA (Bo1)", "AAA (Bo3)", "BBB (Bo1)", "CCC (Bo1)", "CCC (Bo3)"]
    assert (both["up"]["lo"], both["up"]["hi"]) == (2.5, 6.0)


def test_no_effect_is_averaged_or_tested():
    f = X.cross_from_facts(play_sets({"AAA": ("strong", 5.0), "BBB": ("likely", 3.0), "CCC": ("likely", 4.0)}))["facts"]["facts"]["XSET.bo1.PLAY"]
    assert 4.0 not in f["nums"] and 3.0 in f["nums"] and 5.0 in f["nums"]          # the range ends only: no mean
    assert not any(k in f["text"].lower() for k in ("average of", "mean", "p-value", "significan"))


# ------------------------------------------------------------------------------------------ the other lessons

def _spec(w, o):
    return dict(ends=dict(a="x", b="y", winner="x", winner_avg_mv=w, other_avg_mv=o))


def test_spec_lower_curve_end_is_decided_by_average_mana_value():
    per = {"AAA": SF("AAA", [("SPEC.bo1.WU", "likely", 2.0, _spec(2.8, 3.1)), ("SPEC.bo1.UB", "strong", 3.0, _spec(3.5, 3.2))], ["SPEC.bo1.BG"]),
           "BBB": SF("BBB", [("SPEC.bo1.WU", "likely", 1.5, _spec(2.0, 2.5)), ("SPEC.bo1.UB", "weak", 9.0, _spec(1.0, 4.0))]),
           "CCC": SF("CCC", [("SPEC.bo1.WU", "likely", 1.0, _spec(2.6, 3.0))])}
    res = X.cross_from_facts(per)
    t = lesson(res, "SPEC_LOWCURVE", "bo1")
    assert t["m"] == 6 and (t["n_up"], t["n_down"]) == (3, 1) and t["verdict"] == "mixed"       # per pair-set-format; the weak one is not clear
    assert t["up"]["sets"] == ["AAA WU", "BBB WU", "CCC WU"] and t["down"]["sets"] == ["AAA UB"]
    assert "XSET.bo1.SPEC_LOWCURVE" in res["facts"]["facts"]


def test_land_and_creature_bands_low_and_high():
    def sd(code, claims, nc=()):
        return SF(code, claims, nc)
    per = {"AAA": sd("AAA", [("SHAPEDET.bo1.lands.L16", "likely", 1.0), ("SHAPEDET.bo1.lands.L18", "likely", -1.0), ("SHAPEDET.bo1.creatures.C3", "likely", 1.0)]),
           "BBB": sd("BBB", [("SHAPEDET.bo1.lands.L16", "strong", 0.5), ("SHAPEDET.bo1.lands.L17", "likely", 3.0)], ["SHAPEDET.bo1.creatures"]),
           "CCC": sd("CCC", [("SHAPEDET.bo1.lands.L16", "likely", 0.8), ("SHAPEDET.bo1.lands.L18", "weak", -2.0)]),
           "DDD": sd("DDD", [], ["SHAPEDET.bo1.lands"])}                       # DDD has lands bands but nothing clear; no creatures band at all
    res = X.cross_from_facts(per)
    lo = lesson(res, "LANDS_LOW", "bo1")
    assert (lo["n_up"], lo["n_down"], lo["m"], lo["verdict"]) == (3, 0, 4, "pattern")
    hi = lesson(res, "LANDS_HIGH", "bo1")
    assert (hi["n_up"], hi["n_down"], hi["m"], hi["verdict"]) == (0, 1, 4, "rare")           # the middle band L17 is never L16 / L18
    cl, ch = lesson(res, "CREATURES_LOW", "bo1"), lesson(res, "CREATURES_HIGH", "bo1")
    assert cl["m"] == 2 and cl["n"] == 0 and ch["n_up"] == 1 and ch["m"] == 2
    assert 16 in res["facts"]["facts"]["XSET.bo1.LANDS_LOW"]["nums"]


def test_trap_rares_count_sets_and_cards():
    tr = lambda slug, v: (f"CARD.bo1.{slug}", "likely", v, dict(name=slug))
    per = {"AAA": SF("AAA", [tr("a", -1.0), tr("b", -0.5)]), "BBB": SF("BBB", [tr("c", -0.8)]), "CCC": SF("CCC", [tr("d", -2.0)]), "DDD": SF("DDD", [])}
    res = X.cross_from_facts(per)
    t = lesson(res, "TRAPS", "bo1")
    assert (t["n"], t["m"], t["verdict"], t["cards"], t["n_sets"]) == (3, 4, "pattern", 4, 3)
    f = res["facts"]["facts"]["XSET.bo1.TRAPS"]
    assert "4 trap rares listed in 3 sets" in f["text"] and {4, 3, -2.0, -0.5} <= set(f["nums"])


def test_colour_pair_pattern_needs_three_and_no_opposite():
    def pairs(vals):
        return {c: SF(c, [("PAIR.bo1.WU", b, v)] if b else [], [] if b else ["PAIR.bo1.WU"]) for c, (b, v) in vals.items()}
    ok = X.cross_from_facts(pairs({"A": ("likely", 2.0), "B": ("strong", 3.0), "C": ("likely", 1.0), "D": (None, 0)}))
    assert lesson(ok, "PAIR.WU", "bo1")["verdict"] == "pattern"
    bad = X.cross_from_facts(pairs({"A": ("likely", 2.0), "B": ("strong", 3.0), "C": ("likely", 1.0), "D": ("likely", -1.0)}))
    assert lesson(bad, "PAIR.WU", "bo1")["verdict"] == "mixed"
    assert "XSET.bo1.PAIR.WU" in X.cross_from_facts(pairs({"A": ("likely", 2.0)}))["facts"]["noclear"]


def test_builder_and_check_counts():
    per = {"AAA": SF("AAA", [("BUILDER.bo1.lands", "likely", -0.6), ("BUILDER.bo1.mean_mv", "likely", 0.12), ("BUILDERCHECK.bo1", "strong", 1.2)]),
           "BBB": SF("BBB", [("BUILDER.bo1.lands", "likely", -0.4), ("BUILDER.bo1.mean_mv", "likely", -0.05)], ["BUILDERCHECK.bo1"]),
           "CCC": SF("CCC", [("BUILDER.bo1.lands", "likely", -0.2)], ["BUILDER.bo1.mean_mv", "BUILDERCHECK.bo1"])}
    res = X.cross_from_facts(per)
    lands = lesson(res, "BUILDER.lands", "bo1")
    assert (lands["n_down"], lands["m"], lands["verdict"]) == (3, 3, "pattern")
    mv = lesson(res, "BUILDER.mean_mv", "bo1")
    assert (mv["n_up"], mv["n_down"], mv["m"], mv["verdict"]) == (1, 1, 3, "mixed")
    assert "+0.12" in res["facts"]["facts"]["XSET.bo1.BUILDER.mean_mv"]["text"]
    chk = lesson(res, "BUILDERCHECK", "bo1")
    assert (chk["n"], chk["m"]) == (1, 3) and "XSET.bo1.BUILDERCHECK" in res["facts"]["noclear"]


def test_glance_ranges_are_descriptive():
    per = {"AAA": SF("AAA", mulligan=(10.0, 21.0), turns=(8.0, 9.0)), "BBB": SF("BBB", mulligan=(12.5, 23.0), turns=(9.5, 9.0))}
    f = X.cross_from_facts(per)["facts"]["facts"]
    g = f["XSET.bo1.GLANCE"]
    assert g["badge"] == "descriptive" and "from 10.0% (AAA) to 12.5% (BBB)" in g["text"] and "from 8.0 turns (AAA) to 9.5 turns (BBB)" in g["text"]
    assert {2, 10.0, 12.5, 8.0, 9.5} <= set(g["nums"])
    assert "9.0 turns in every one" in f["XSET.bo3.GLANCE"]["text"]


def test_old_per_set_facts_are_refused():
    f = SF("AAA", [("PLAY.bo1", "likely", 2.0)])
    del f["facts"]["PLAY.bo1"]["value"]
    with pytest.raises(X.OldFacts):
        X.cross_from_facts({"AAA": f})


# ------------------------------------------------------------------------------------------ structure, nums, determinism

NUM_IN_TEXT = re.compile(r"(?<![\w.])[+\-]?\d+(?:\.\d+)?")


def test_every_number_of_every_text_is_in_nums_and_n_of_m_matches():
    res = X.cross_from_facts(fake_api.fake_cross_per_set())
    facts = res["facts"]
    assert facts["set"] == "ALL" and set(facts) >= {"version", "set", "formats", "stamps", "facts", "noclear"} and facts["formats"] == ["bo1", "bo3"]
    assert facts["stamps"] == {"AAA.bo1": "AAA|bo1", "AAA.bo3": "AAA|bo3", "BBB.bo1": "BBB|bo1", "BBB.bo3": "BBB|bo3", "CCC.bo1": "CCC|bo1", "CCC.bo3": "CCC|bo3"}
    assert facts["facts"]
    for fid, f in facts["facts"].items():
        assert fid.startswith("XSET.") and f["fmt"] in ("bo1", "bo3", "both") and f["badge"] in ("likely", "descriptive")
        for v, dec, signed, raw in report_check.numbers_of(report_check.plain(f["text"])):
            assert report_check.number_ok(v, dec, signed, f["nums"]), (fid, raw, f["text"])
        if f["badge"] == "likely":
            m = re.search(r"in (\d+) of (\d+)", f["text"])
            assert m and int(m.group(1)) == f["n_clear"] and int(m.group(2)) == f["m"] and {f["n_clear"], f["m"]} <= set(f["nums"])
            assert f["n_clear"] == f["n_up"] + f["n_down"] and f["verdict"] in ("pattern", "mixed")
    for n in facts["noclear"].values():
        assert n["fmt"] in ("bo1", "bo3", "both") and n["id"].startswith("XSET.")


def test_deterministic_and_independent_of_input_order():
    per = fake_api.fake_cross_per_set()
    a = json.dumps(X.cross_from_facts(per), sort_keys=False)
    b = json.dumps(X.cross_from_facts(dict(reversed(list(per.items())))), sort_keys=False)
    assert a == b == json.dumps(X.cross_from_facts(fake_api.fake_cross_per_set()), sort_keys=False)


def test_render_text_lists_facts_and_noclear():
    facts = X.cross_from_facts(fake_api.fake_cross_per_set())["facts"]
    t = report_facts.render_text(facts)                      # dispatches to the cross-set digest for "ALL"
    assert t.startswith("FACTS ALL") and "[XSET.bo1.PLAY]" in t and "== Best of 1 and Best of 3 together ==" in t
    assert "NOCLEAR" in t and "XSET.bo3.PLAY" in t and "[XSET.bo3.PLAY]" not in t


# ------------------------------------------------------------------------------------------ the checker and the install path

def test_checker_accepts_a_tiny_text_tagged_with_xset_ids():
    facts = X.cross_from_facts(fake_api.fake_cross_per_set())["facts"]
    assert report_check.check(fake_api.FAKE_CROSS_TEXT, facts) == []
    good = "In Best of 1, playing first did better in 3 of 3 sets, a pattern.<!--F:XSET.bo1.PLAY-->\n"
    assert report_check.check(good, facts) == []
    bad_num = "In Best of 1, playing first did better in 2 of 3 sets, a pattern.<!--F:XSET.bo1.PLAY-->\n"
    assert any("does not match" in p for p in report_check.check(bad_num, facts))
    untagged = "In Best of 1, playing first did better in 3 of 3 sets.\n"
    assert report_check.check(untagged, facts)
    rare = "Playing first wins more often in Best of 3.<!--F:XSET.bo3.PLAY-->\n"                # a RARE lesson is NOCLEAR: no direction allowed
    assert any("no-clear" in p for p in report_check.check(rare, facts))
    ok_rare = "There is no clear pattern for playing first in Best of 3.<!--F:XSET.bo3.PLAY-->\n"
    assert report_check.check(ok_rare, facts) == []


def test_cli_report_facts_all_and_install(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    per = fake_api.fake_cross_per_set()
    monkeypatch.setattr(X, "build_cross", lambda sets=None, refresh=False, progress=None: X.cross_from_facts(per))
    out = tmp_path / "facts_ALL.json"
    assert cli.main(["report-facts", "ALL", "--out", str(out)]) == 0
    facts = json.loads(out.read_text(encoding="utf-8"))
    assert facts["set"] == "ALL" and out.with_suffix(".txt").read_text(encoding="utf-8").startswith("FACTS ALL")
    r = tmp_path / "ALL_report.md"
    r.write_text(fake_api.FAKE_CROSS_TEXT, encoding="utf-8")
    assert cli.main(["report-install", "ALL", str(r), str(out)]) == 0
    assert W.available() == ["ALL"]
    got = api.written_report("ALL")
    assert got["text"] == fake_api.FAKE_CROSS_TEXT and got["meta"]["stamp"] == facts["stamps"] and got["outdated"] is False
    r.write_text("In Best of 1, playing first did better in 9 of 3 sets.<!--F:XSET.bo1.PLAY-->\n", encoding="utf-8")
    assert cli.main(["report-install", "ALL", str(r), str(out)]) == 1


def test_all_is_outdated_when_any_per_set_stamp_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    meta = {"stamp": {"AAA.bo1": "s1", "BBB.bo1": "t1", "BBB.bo3": "t3"}}
    for code, fmt, st in (("AAA", "bo1", "s1"), ("BBB", "bo1", "t1"), ("BBB", "bo3", "t3")):
        d = config.cache_dir(code)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"report_{fmt}.json").write_text(json.dumps({"stamp": st}), encoding="utf-8")
    cur = W.current_stamps("ALL", like=meta["stamp"])
    assert cur == meta["stamp"] and not W.outdated(meta, cur)
    (config.cache_dir("BBB") / "report_bo3.json").write_text(json.dumps({"stamp": "t3-new"}), encoding="utf-8")
    assert W.outdated(meta, W.current_stamps("ALL", like=meta["stamp"]))
    assert not W.outdated(meta, W.current_stamps("ALL", like={"CCC.bo1": "x"}))          # nothing known: not outdated
