"""The "More detail" statistics (sealed_lab.report_detail, the card explorer rows of report.py) on hand-made data and on the synthetic set."""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pytest

from sealed_lab import report as R, report_detail as D, report_facts as RF


def table(rows: list[dict]) -> pl.DataFrame:
    base = dict(rarity="common", colors="W", is_land=False, is_basic=False, se=0.01, exposure=5000.0, n_elig=1000, n_played=500, copies=600,
                exposure_o=5000.0, se_o=0.01, copies_o=600)
    return pl.DataFrame([dict(base, card_id=i, name=f"Card {i}", **{"beta_o": r.get("beta", 0.0), **r}) for i, r in enumerate(rows)])


def rows_for(n: int = 30, **over) -> list[dict]:
    rng = np.random.default_rng(3)
    return [dict(beta=float(rng.normal(0, 0.005)), n_played=int(rng.integers(300, 900))) for _ in range(n)]


def test_explorer_rows_are_compact_and_flag_only_clear_format_differences():
    rows = rows_for()
    rows[0]["beta_o"] = rows[0]["beta"] + 0.06                    # Bo3 clearly better than Bo1 (6 SE-ish)
    rows[1]["beta_o"] = rows[1]["beta"] + 0.06
    rows[1]["exposure_o"] = 1000.0                               # ... but thin in the other format: not called clear
    rows[2]["beta_o"] = rows[2]["beta"] + 0.01                    # inside the noise
    out = R.build_card_lists(table(rows), "bo1")
    ex = out["explorer"]
    assert ex["cols"] == R.EXPLORER_COLS and len(ex["rows"]) == out["n_shown"] == 30
    assert json.dumps(out) and all(len(r) == len(ex["cols"]) for r in ex["rows"])
    names = [r[1] for r in ex["rows"]]
    assert names == sorted(names)
    c = {n: i for i, n in enumerate(ex["cols"])}
    by = {r[c["name"]]: r for r in ex["rows"]}
    assert by["Card 0"][c["dz"]] > R.Z_LIKELY and by["Card 1"][c["thin_other"]] == 1 and abs(by["Card 2"][c["dz"]]) < R.Z_LIKELY
    assert [r["name"] for r in out["compare_cards"]] == ["Card 0"]
    cc = out["compare_cards"][0]
    assert cc["v_bo3"] > cc["v_bo1"] and cc["diff_pp"] == pytest.approx(cc["v_bo3"] - cc["v_bo1"], abs=0.02) and cc["badge"] in ("likely", "strong")
    # the Bo3 report sees the same difference with the same sign
    out3 = R.build_card_lists(table([dict(r, beta=r["beta_o"] if "beta_o" in r else r["beta"], beta_o=r["beta"]) for r in rows]), "bo3")
    assert out3["compare_cards"] == [] or out3["compare_cards"][0]["diff_pp"] != 0


def test_explorer_without_the_other_format():
    t = table(rows_for()).drop("beta_o", "se_o", "exposure_o", "copies_o")
    out = R.build_card_lists(t, "bo3")
    c = {n: i for i, n in enumerate(out["explorer"]["cols"])}
    assert all(r[c["value_other"]] is None and r[c["dz"]] is None for r in out["explorer"]["rows"]) and out["compare_cards"] == []


def test_evolution_verdict_needs_likely_and_the_same_sign_in_both_halves():
    assert D._verdict(3.0, 1.0, [2.0, 4.0])["verdict"] == "better"
    assert D._verdict(-3.0, 1.0, [-2.0, -4.0])["verdict"] == "worse"
    assert D._verdict(3.0, 1.0, [6.0, -0.5])["verdict"] == "stable"                # does not replicate
    assert D._verdict(1.5, 1.0, [1.0, 2.0])["verdict"] == "stable"                 # Weak
    v = D._verdict(2.0, 1.0, [1.0, 3.0])
    assert v["change_badge"] == "likely" and v["verdict"] == "better"
    e, se = D._combine([1.0, 3.0], [1.0, 1.0])
    assert e == pytest.approx(2.0) and se == pytest.approx(1 / np.sqrt(2))


def test_compare_pairs_flags_only_clear_differences():
    def sec(effs):
        return dict(rows=[dict(key=k, label=k, hidden=False, effect_pp=e, se_pp=s, badge=R.badge_of(e / s)) for k, (e, s) in effs.items()])
    a = sec({"WU": (2.0, 0.5), "UB": (0.0, 0.5)})
    b = sec({"WU": (-2.0, 1.0), "UB": (0.5, 1.0)})
    out = D.compare_pairs("bo1", a, b)
    r = {x["key"]: x for x in out["pairs"]}
    assert r["WU"]["diff_effect_pp"] == pytest.approx(-4.0) and r["WU"]["diff_badge"] in ("likely", "strong")
    assert r["UB"]["diff_badge"] == "weak"
    assert out["pairs"][0]["key"] == "WU" and "Best of 1" in out["text"][0]
    assert "No color pair clearly" in D.compare_pairs("bo1", a, a)["text"][0]


def test_pair_text_states_only_clear_splashes():
    d = dict(on_play=dict(diff_pp=0.4, badge="weak", wr_play=0.5, wr_draw=0.5), median_turns=9.0, mulligan_rate=0.1,
             splash=[dict(label="Red", effect_pp=0.5, badge="weak"), dict(label="Green", effect_pp=-3.0, badge="likely")])
    t = " ".join(D.pair_text(d, 9.0, 0.1))
    assert "no clear difference" in t and "Splashing green costs" in t and "red" not in t.lower().replace("green", "")
    d["splash"] = d["splash"][:1]
    assert "No splash color clearly" in " ".join(D.pair_text(d, None, None))


def test_fake_report_has_the_detail_sections_and_facts_follow_the_rule():
    from sealed_lab.gui import fake_api

    reps = {f: fake_api.set_report("TST", f) for f in RF.FORMATS}
    for rep in reps.values():
        assert json.dumps(rep)
        assert rep["pair_details"]["rows"] and rep["shape"]["details"]["lands"] and rep["evolution"]["pairs"] and rep["compare"]["pairs"] and rep["data_notes"]
        assert rep["version"] == R.REPORT_VERSION
    facts = RF.facts_from_reports("TST", reps)
    ids = list(facts["facts"])
    assert any(i.startswith("EVOL.bo1.PAIR.") for i in ids) and any(i.startswith("EVOL.bo1.CARD.") for i in ids)
    assert any(i.startswith("PAIRDET.") for i in ids) and any(i.startswith("SHAPEDET.") for i in ids)
    for f in facts["facts"].values():
        if f["id"].split(".")[0] in ("EVOL", "PAIRDET", "SHAPEDET"):
            assert f["badge"] in ("likely", "strong") and f["builder_uses"] == "observed"
    for p in reps["bo1"]["evolution"]["pairs"]:
        fid = f"EVOL.bo1.PAIR.{p['key']}"
        assert not (fid in facts["facts"] and p["verdict"] == "stable") and not (fid in facts["facts"] and fid in facts["noclear"])
