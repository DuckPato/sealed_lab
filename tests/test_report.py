"""The Set report (sealed_lab.report): helpers, the card-list rules on hand-made tables, and the whole report on the synthetic set."""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from sealed_lab import api, cli, report as R


# --------------------------------------------------------------------------- helpers

def test_badges_and_plain_words():
    assert [R.badge_of(z) for z in (3.0, -3.5, 2.0, -2.5, 1.99, 0.0, None, float("nan"))] == \
        ["strong", "strong", "likely", "likely", "weak", "weak", "weak", "weak"]
    assert R.signed(1.234) == "+1.2" and R.signed(-0.04, 2) == "-0.04"
    assert R.range_text(1.0, 0.5) == "likely between +0.0 and +2.0 points"
    txt = R.range_text(0.0, 1.0)
    assert "CI" not in txt and "p-value" not in txt


def test_colour_groups():
    assert R.pair_group("WU") == "WU" and R.pair_group("UW") == "WU"       # always in WUBRG order
    assert R.pair_group("G") == "mono" and R.pair_group("WUB") == "tri" and R.pair_group("WBRG") == "tri"
    assert R.pair_group("") is None and R.pair_group(None) is None
    assert R.group_label("WU") == "White-Blue" and R.group_label("tri").startswith("Three")
    assert R.card_group("G", False) == "G" and R.card_group("BG", False) == "gold" and R.card_group("", False) == "colorless"
    assert R.card_group("", True) == "land"


# --------------------------------------------------------------------------- the card lists on a hand-made table

def table(rows: list[dict]) -> pl.DataFrame:
    base = dict(rarity="common", colors="W", is_land=False, is_basic=False, se=0.01, exposure=5000.0, n_elig=1000, n_played=500, copies=600)
    return pl.DataFrame([dict(base, card_id=i, name=f"Card {i}", **r) for i, r in enumerate(rows)])


def test_card_lists_rules():
    rng = np.random.default_rng(1)
    rows = [dict(beta=float(rng.normal(0, 0.01)), n_played=int(rng.integers(300, 900))) for _ in range(60)]
    rows[0] = dict(beta=0.08, n_played=100)                     # played by 10 %, clearly good  -> underrated
    rows[1] = dict(beta=-0.08, n_played=950)                    # played by 95 %, clearly bad   -> overrated
    rows[2] = dict(beta=0.08, n_played=100, n_elig=100)         # too few decks that could play it -> hidden
    rows[3] = dict(beta=0.08, n_played=100, exposure=100.0)     # too little data in the format -> hidden
    rows[4] = dict(beta=0.08, n_played=950)                     # good AND played a lot: neither list
    rows[5] = dict(beta=0.012, n_played=100, se=0.02)           # a gap inside the noise: not listed
    out = R.build_card_lists(table(rows), "bo1")
    assert json.dumps(out)                                      # plain data
    under = [r["card_id"] for r in out["underrated"]]
    over = [r["card_id"] for r in out["overrated"]]
    assert under[0] == 0 and 2 not in under and 3 not in under and 4 not in under and 5 not in under
    assert over[0] == 1 and 4 not in over
    assert out["n_hidden"] == 2 and out["n_shown"] == 58
    for r in out["underrated"] + out["overrated"]:
        assert abs(r["z"]) >= R.Z_LIKELY and r["badge"] in ("strong", "likely") and 0 <= r["play_rate"] <= 1 and r["reason"]
    assert out["underrated"][0]["value_pp"] > 0 > out["overrated"][0]["value_pp"]
    assert [r["value_pp"] for r in out["underrated"]] == sorted((r["value_pp"] for r in out["underrated"]), reverse=True)
    tops = out["top"]["common"]["W"]
    assert len(tops) == R.TOP_N and tops[0]["value_pp"] >= tops[-1]["value_pp"] and all(r["badge"] in ("strong", "likely", "weak") for r in tops)
    assert "uncommon" in out["top"] and out["top"]["uncommon"] == {}


def test_card_lists_value_is_relative_to_the_colour_group():
    # blue cards are all better than white ones; inside each color the average PLAYED card is the zero point
    rows = [dict(beta=0.05 + 0.001 * i, colors="U") for i in range(20)] + [dict(beta=-0.05 + 0.001 * i, colors="W") for i in range(20)]
    out = R.build_card_lists(table(rows), "bo3")
    top_w = out["top"]["common"]["W"][0]
    assert abs(top_w["value_pp"]) < 1.5                         # a white card at -0.04 logit is NOT -4 points: its zero is the white average
    assert out["top"]["common"]["U"][0]["group"] == "U"


# --------------------------------------------------------------------------- the whole report on the synthetic set

@pytest.mark.parametrize("fmt", ["bo1", "bo3"])
def test_compute_report_on_the_synthetic_set(calc, model, fmt):
    rep = R.compute_report("TST", fmt, model=model)
    json.dumps(rep)
    assert rep["set_code"] == "TST" and rep["fmt"] == fmt and rep["built_at"] == model.meta["built_at"] and rep["stamp"].endswith(f"v{R.REPORT_VERSION}")
    gl = rep["glance"]
    assert gl["n_games"] > 1000 and gl["n_decks"] > 100 and 3 <= gl["median_turns"] <= 30 and 0 < gl["mulligan_rate"] < 0.5
    op = gl["on_play"]                                          # planted: +0.08 logit on the play = about +2 points
    assert abs(op["diff_pp"] - 2.0) < 3.5 * op["se_pp"] and 0.4 < op["wr_draw"] < 0.7 and 0.4 < op["wr_play"] < 0.7 and gl["text"]   # (small samples: judged against its own SE)
    pr = rep["pairs"]
    assert pr["rows"], "no color groups"
    assert sum(r["share"] for r in pr["rows"]) == pytest.approx(1.0, abs=1e-6)
    for r in pr["rows"]:
        assert 0 < r["share"] <= 1 and 0 <= r["splash_share"] <= 1 and r["se_pp"] > 0 and r["badge"] in ("strong", "likely", "weak")
        assert abs(r["effect_pp"]) < 40
    assert sum(r["share"] * r["effect_pp"] for r in pr["rows"]) == pytest.approx(0.0, abs=2.0 * max(r["se_pp"] for r in pr["rows"]))
    assert pr["splash"]["se_pp"] > 0 and 0 <= pr["splash"]["share"] <= 1
    cs = rep["cards"]
    assert cs["n_shown"] + cs["n_hidden"] == cs["n_cards"] and (fmt == "bo3" or cs["n_shown"] >= 5)      # (Bo3 of the tiny set has too little data per card)
    for lst in (cs["underrated"], cs["overrated"]):
        for r in lst:
            assert 0 <= r["play_rate"] <= 1 and r["n_elig"] >= R.MIN_ELIGIBLE and abs(r["z"]) >= R.Z_LIKELY
    for rar in cs["top"].values():
        for rows in rar.values():
            for r in rows:
                assert 0 <= r["play_rate"] <= 1
    assert rep["notes"]["cards"].startswith("Yes") and "Observed only" in rep["notes"]["pairs"]


def test_card_values_track_the_model(calc, model):
    """The reported value of a card is the model's own beta less its color group's played average (in points = logit x 25)."""
    rep = R.compute_report("TST", "bo1", model=model)
    beta = model.card_beta("bo1")
    rows = [r for rar in rep["cards"]["top"].values() for g in rar.values() for r in g]
    assert rows
    by_group: dict[str, list[float]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r["value_pp"] - 25.0 * beta[r["card_id"]])
    for g, offs in by_group.items():
        assert np.ptp(offs) < 1e-6, g                           # one constant (the group's played average) per color group


def test_cache_roundtrip_and_stamp(calc, model):
    p = R.cache_path("TST", "bo1")
    p.unlink(missing_ok=True)
    a = R.get_report("TST", "bo1", model=model)
    assert a["cached"] is False and p.exists()
    b = R.get_report("TST", "bo1", model=model)
    assert b["cached"] is True and b["pairs"]["rows"] == a["pairs"]["rows"]
    c = R.get_report("TST", "bo1", model=model, refresh=True)
    assert c["cached"] is False
    d = json.loads(p.read_text(encoding="utf-8"))
    d["stamp"] = "an older model|v1"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert R.read_cached("TST", "bo1", R.model_stamp(model)) is None          # a different built-at stamp invalidates the cache
    assert R.get_report("TST", "bo1", model=model)["cached"] is False
    p.write_text("{not json", encoding="utf-8")                                # a damaged cache file is just a miss
    assert R.get_report("TST", "bo1", model=model)["cached"] is False


def test_api_and_cli(calc, tmp_path, capsys):
    rep = api.set_report("TST", "bo1")
    assert rep["fmt"] == "bo1" and rep["cards"]["n_shown"] > 0
    assert cli.main(["report", "TST", "--fmt", "bo3"]) == 0
    out = capsys.readouterr().out
    assert "Set report TST bo3" in out and "Color pairs" in out and "Underrated" in out
    f = tmp_path / "r.json"
    assert cli.main(["report", "TST", "--fmt", "bo1", "--out", str(f)]) == 0
    assert json.loads(f.read_text(encoding="utf-8"))["fmt"] == "bo1"
    with pytest.raises(ValueError):
        R.compute_report("TST", "draft")


def test_card_lists_by_colour_group_three_then_up_to_five_when_strong():
    rng = np.random.default_rng(2)
    rows = [dict(beta=float(rng.normal(0, 0.005)), n_played=int(rng.integers(300, 900))) for _ in range(60)]
    rows += [dict(beta=0.080 + 0.001 * i, n_played=100) for i in range(4)]            # 4 Strong underrated white cards
    rows += [dict(beta=0.030, n_played=100, se=0.009) for _ in range(3)]               # 3 Likely ones (smaller, so ranked after)
    out = R.build_card_lists(table(rows), "bo1")
    w = out["underrated_by"]["W"]
    assert len(w) == 4 and all(r["badge"] == "strong" for r in w)                       # 3, plus the 4th only because it is Strong
    assert out["underrated"] == [r for g in out["underrated_by"].values() for r in g]
    assert all(len(v) <= R.LIST_MAX for v in out["underrated_by"].values())


def test_trap_rares_are_negative_popular_rares_and_mythics():
    rng = np.random.default_rng(3)
    rows = [dict(beta=float(rng.normal(0, 0.005)), n_played=int(rng.integers(300, 900))) for _ in range(60)]
    rows[0] = dict(beta=-0.08, n_played=950, rarity="rare")                # popular, clearly bad rare        -> trap
    rows[1] = dict(beta=-0.10, n_played=950, rarity="mythic")              # worse mythic                     -> trap, listed first
    rows[2] = dict(beta=-0.08, n_played=950, rarity="common")              # overrated but a common           -> not a trap
    rows[3] = dict(beta=-0.08, n_played=100, rarity="rare")                # clearly bad rare nobody plays    -> not a trap
    rows[4] = dict(beta=0.08, n_played=950, rarity="rare")                 # popular and good                 -> not a trap
    out = R.build_card_lists(table(rows), "bo1")
    assert [r["card_id"] for r in out["traps"]] == [1, 0]
    assert all(r["value_pp"] < 0 and r["badge"] in ("likely", "strong") and "loses" in r["reason"] for r in out["traps"])
    assert R.REPORT_VERSION >= 6 and len(out["traps"]) <= R.TRAP_MAX


def test_splash_play_stats_exposed_when_present():
    rows = [dict(beta=0.05, n_played=400, n_elig=800, n_splash_elig=200, n_splashed=50) for _ in range(15)]
    rows[0] = dict(beta=0.10, se=0.01, n_played=700, n_elig=800, n_splash_elig=200, n_splashed=40)       # 20% splash, Strong winner -> splashed
    rows[1] = dict(beta=-0.10, se=0.01, n_played=300, n_elig=800, n_splash_elig=200, n_splashed=30)      # 15% splash, Strong loser  -> splash_traps
    rows[2] = dict(beta=0.10, se=0.01, n_played=700, n_elig=800, n_splash_elig=20, n_splashed=10)        # < MIN_SPLASH_ELIG -> omitted from explorer splash
    out = R.build_card_lists(table(rows), "bo1")
    top_w = out["top"]["common"]["W"][0]
    assert top_w["n_splash_elig"] in (200, 20)
    assert [r["card_id"] for r in out["splashed"]] == [0]
    assert [r["card_id"] for r in out["splash_traps"]] == [1]
    c = {n: i for i, n in enumerate(out["explorer"]["cols"])}
    by_id = {r[c["id"]]: r for r in out["explorer"]["rows"]}
    assert by_id[0][c["splash_rate"]] == pytest.approx(0.20)
    assert by_id[2][c["splash_rate"]] is None
