"""scripts/draft_split_half.py, draft_sealed_identity.py and the draft side of builder_scoreboard.py: the pure parts (cuts, halves, pool
eligibility, decision lines, identity comparison, G1 line) on synthetic tables; no model is fitted."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.special import logit

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dsh = _load("draft_split_half")
dsi = _load("draft_sealed_identity")
bsb = _load("builder_scoreboard")
T0 = datetime(2024, 1, 1)


def _games(n_sealed=60, n_prem=80, n_trad=40, n_other=20, games_per=8):
    """A games table: Sealed Bo1 events, PremierDraft, TradDraft and QuickDraft events, event i starting i hours after T0."""
    rows = []
    for fmt, ev, n in (("bo1", "Sealed", n_sealed), ("draft", "PremierDraft", n_prem), ("draft", "TradDraft", n_trad),
                       ("draft", "QuickDraft", n_other)):
        for i in range(n):
            t = T0 + timedelta(hours=i)
            for g in range(games_per):
                rows.append(dict(draft_id=f"{ev}-{i}", fmt=fmt, event=ev, game_time=t + timedelta(minutes=g), draft_time=t))
    return pl.DataFrame(rows)


# ---- cuts and training groups ---------------------------------------------------------------------------------------------------------

def test_cuts_are_the_start_of_the_last_quarter_of_each_format():
    g = _games()
    cuts = dsh.cuts_of(g)
    assert cuts["dbo1"] == T0 + timedelta(hours=60)            # 80 events, the last 20 are held out
    assert cuts["dbo3"] == T0 + timedelta(hours=30)            # 40 events, the last 10
    assert cuts["sealed"] == T0 + timedelta(hours=45)          # 60 events (Bo1 only: the target is bo1), the last 15
    ev = dsh.event_table(g)
    groups = dsh.train_groups(ev, cuts)
    assert [len(groups[k]) for k in ("sealed", "dbo1", "dbo3", "other")] == [45, 60, 30, 20]
    assert all(s < cuts["dbo1"] for _, s in groups["other"])
    assert {d for d, _ in groups["dbo1"]} == {f"PremierDraft-{i}" for i in range(60)}


def test_other_draft_events_use_the_dbo1_cut_and_fall_back_to_dbo3():
    g = _games(n_other=80)                                      # QuickDraft events starting up to 79 h after T0
    ev = dsh.event_table(g)
    cuts = dsh.cuts_of(g)
    other = dsh.train_groups(ev, cuts)["other"]
    assert len(other) == 60 and max(s for _, s in other) < cuts["dbo1"]
    no_dbo1 = dict(cuts, dbo1=None)
    assert dsh.other_cut(no_dbo1) == cuts["dbo3"]
    assert dsh.train_groups(ev, dict(cuts, dbo1=None, dbo3=None))["other"] == []


def test_a_format_without_enough_events_has_no_cut():
    g = _games(n_trad=10)
    cuts = dsh.cuts_of(g)
    assert cuts["dbo3"] is None and cuts["dbo1"] is not None
    assert dsh.train_groups(dsh.event_table(g), cuts)["dbo3"] == []


# ---- halves ---------------------------------------------------------------------------------------------------------------------------

def test_halves_split_the_draft_training_events_by_format_and_cover_them_once():
    g = _games()
    cuts = dsh.cuts_of(g)
    groups = dsh.train_groups(dsh.event_table(g), cuts)
    a, b = dsh.split_draft_halves(groups, "TST", 0)
    for f in ("dbo1", "dbo3"):
        both = a[f] + b[f]
        assert sorted(both) == sorted(d for d, _ in groups[f]) and not set(a[f]) & set(b[f])
        assert abs(len(a[f]) - len(b[f])) <= 1
    a2, b2 = dsh.split_draft_halves(groups, "TST", 0)
    assert (a, b) == (a2, b2)                                   # deterministic
    assert dsh.split_draft_halves(groups, "OTHER", 0)[0] != a   # per-set seed


def test_half_keep_is_sealed_plus_other_plus_own_half():
    sp = dict(train=dict(sealed=["s1", "s2"], other=["o1"]), A=dict(dbo1=["a1"], dbo3=["a3"]), B=dict(dbo1=["b1"], dbo3=[]))
    assert dsh.half_keep(sp, "A") == {"s1", "s2", "o1", "a1", "a3"}
    assert dsh.half_keep(sp, "B") == {"s1", "s2", "o1", "b1"}


# ---- pools ----------------------------------------------------------------------------------------------------------------------------

def test_pool_eligibility_and_draw():
    g = _games()
    ev, cuts = dsh.event_table(g), dsh.cuts_of(g)
    rows = []
    for i in range(80):                                         # events with i % 10 == 9 hold 10 + 15 = 25 cards (too small), the others 40
        rows.append(dict(draft_id=f"PremierDraft-{i}", card_id=1, n=10 if i % 10 == 9 else 25))
        rows.append(dict(draft_id=f"PremierDraft-{i}", card_id=2, n=15))
    rows.append(dict(draft_id="TradDraft-35", card_id=1, n=45))
    pools = pl.DataFrame(rows)
    el = dsh.eligible_pools(pools, ev, cuts, "dbo1")
    assert all(int(d.split("-")[1]) >= 60 for d in el) and len(el) == 18                 # held-out events (60..79) with >= 40 cards: 69 and 79 are out
    assert all(int(d.split("-")[1]) % 10 != 9 for d in el)
    assert dsh.eligible_pools(pools, ev, cuts, "dbo3") == ["TradDraft-35"]
    pick = dsh.draw_pools(el, 10, dsh.pool_seed(0, "TST", "dbo1"))
    assert len(pick) == 10 and set(pick) <= set(el) and pick == dsh.draw_pools(el, 10, dsh.pool_seed(0, "TST", "dbo1"))
    assert dsh.draw_pools(el[:4], 10, 1) == sorted(el[:4])                               # a cell with fewer pools takes all of them
    assert dsh.pool_seed(0, "TST", "dbo1") != dsh.pool_seed(0, "TST", "dbo3")


def test_pools_json_is_never_overwritten(tmp_path):
    (tmp_path / "pools.json").write_text("{}")
    with pytest.raises(SystemExit):
        dsh.cmd_pools(tmp_path, tmp_path, ["TST"])


# ---- gaps and decision lines ----------------------------------------------------------------------------------------------------------

def _row(sc, fmt, k, gap_out_dl, gap_in_dl=0.1, same=False):
    """A pool with the draft deck's logit gap (out / in) over the Sealed deck at level 0.0 in both directions."""
    d = dict(ls=0.0, lo=0.0, seconds=2.0, deck={"1": 40}, n_creatures=14, n_lands=17)
    dd = dict(d, ls=gap_in_dl, lo=gap_out_dl, seconds=3.0, deck={"1": 39, "2": 1} if not same else {"1": 40}, n_creatures=15, n_lands=16)
    return dict(set=sc, fmt=fmt, draft_id=f"{fmt}-{k}", dirs={"A": dict(draft=dd, sealed=d), "B": dict(draft=dict(dd), sealed=dict(d))})


def test_gap_conversion_is_split_half_search_s():
    r = _row("TST", "dbo1", 0, 0.4, 0.5)
    recs = dsh.draft_records([r])
    assert len(recs) == 2
    from scipy.special import expit
    assert recs[0]["gap_out"] == pytest.approx(100 * (expit(0.4) - expit(0.0)))
    assert recs[0]["gap_in"] == pytest.approx(100 * (expit(0.5) - expit(0.0)))
    assert recs[0]["n_cre_d"] == 15 and recs[0]["n_land_s"] == 17 and recs[0]["pool"] == "TST|dbo1|dbo1-0"


def test_decision_passes_with_a_clear_gain_and_fails_otherwise():
    gain = [_row("TST", f, k, 0.05 + 0.01 * (k % 3)) for f in ("dbo1", "dbo3") for k in range(15)]
    d = dsh.decision(dsh.draft_records(gain), n_boot=300)
    assert d["line1"]["status"] == "PASS" and d["line2"]["status"] == "PASS" and d["verdict"] == "PASS"
    assert d["line1"]["n"] == 60 and d["line1"]["n_pools"] == 30                          # both directions of a pool are one cluster
    zero = [_row("TST", f, k, 0.0) for f in ("dbo1", "dbo3") for k in range(15)]
    d0 = dsh.decision(dsh.draft_records(zero), n_boot=300)
    assert d0["line1"]["status"] == "FAIL" and d0["verdict"] == "FAIL"                    # lower bound is not above 0


def test_a_harmed_format_fails_line_2_even_when_the_pooled_gain_is_positive():
    rows = [_row("TST", "dbo1", k, 0.2 + 0.01 * (k % 4)) for k in range(30)]
    rows += [_row("TST", "dbo3", k, -0.08 - 0.005 * (k % 4)) for k in range(10)]          # about -2 pp, tight interval
    d = dsh.decision(dsh.draft_records(rows), n_boot=300)
    assert d["line1"]["status"] == "PASS"
    assert d["line2"]["formats"]["dbo3"]["status"] == "FAIL" and d["line2"]["status"] == "FAIL" and d["verdict"] == "FAIL"
    # a small, noisy loss is not "harmed": mean above -0.05 pp
    mild = [_row("TST", "dbo3", k, -0.001 * (1 + k % 2)) for k in range(10)]
    dm = dsh.decision(dsh.draft_records(rows[:30] + mild), n_boot=300)
    assert dm["line2"]["formats"]["dbo3"]["status"] == "PASS"


def test_no_records_is_inconclusive_and_skipped_directions_have_no_record():
    assert dsh.decision([])["verdict"] == "INCONCLUSIVE"
    r = _row("TST", "dbo1", 0, 0.3)
    r["dirs"]["B"] = dict(skipped="model B lacks bo1")
    assert len(dsh.draft_records([r])) == 1
    rep = dsh.full_report([r], n_boot=100)
    assert rep["skipped_directions"][0]["why"] == "model B lacks bo1"
    assert rep["per_cell"]["TST dbo1"]["share_pools_changed"] == 1.0 and rep["per_cell"]["TST dbo1"]["cre_draft"] == 15


def test_report_writes_markdown_and_json(tmp_path):
    rows = [_row("TST", f, k, 0.06) for f in ("dbo1", "dbo3") for k in range(6)]
    (tmp_path / "search_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    md = dsh.write_report(tmp_path, n_boot=200)
    assert "Verdict" in md and "TST dbo1" in md
    assert (tmp_path / "draft_split_half.json").exists() and (tmp_path / "draft_split_half.md").exists()


# ---- G2: the identity comparison ------------------------------------------------------------------------------------------------------

def test_compare_draws_looks_only_at_sealed_keys():
    a = {"bo1__beta": np.arange(4.0), "bo3__ref": np.ones((2, 3)), "dbo1__beta": np.zeros(3), "psi": np.ones(2)}
    b = {"bo1__beta": np.arange(4.0), "bo3__ref": np.ones((2, 3)), "dbo1__beta": np.ones(3), "dbo3__beta": np.ones(3), "psi": np.zeros(2)}
    c = dsi.compare_draws(a, b)
    assert c["identical"] and c["n_keys"] == 2 and c["n_equal"] == 2
    b["bo3__ref"] = b["bo3__ref"] + 1e-12
    c = dsi.compare_draws(a, b)
    assert not c["identical"] and [x["key"] for x in c["differ"]] == ["bo3__ref"]
    del b["bo1__beta"]
    assert {x["key"] for x in dsi.compare_draws(a, b)["differ"]} == {"bo3__ref", "bo1__beta"}
    assert not dsi.compare_draws({}, {})["identical"]                                      # nothing compared is no proof
    nan = {"bo1__x": np.array([1.0, np.nan])}
    assert dsi.compare_draws(nan, dict(nan))["identical"]


def test_compare_logits_tolerance_and_verdict():
    ok = dict(deck="d", fmt="bo1", d0_off=0.1, d0_on=0.1 + 5e-10, draws_off=[0.1, 0.2], draws_on=[0.1, 0.2 + 1e-10])
    bad = dict(deck="e", fmt="bo3", d0_off=0.1, d0_on=0.1, draws_off=[0.1, 0.2], draws_on=[0.1, 0.2 + 1e-6])
    assert dsi.compare_logits([ok])["identical"]
    r = dsi.compare_logits([ok, bad])
    assert not r["identical"] and r["bad"][0]["deck"] == "e" and r["max_abs_diff_draws"] == pytest.approx(1e-6, rel=1e-3)
    assert not dsi.compare_logits([])["identical"]
    good = dict(draws=dict(identical=True), logits=dict(identical=True))
    assert dsi.verdict(good) == "PASS" and dsi.verdict(dict(good, logits=dict(identical=False))) == "BLOCK"


def test_identity_report_aggregates_sets(tmp_path):
    def res(sc, ok):
        return dict(set=sc, status="PASS" if ok else "BLOCK", draws=dict(n_equal=8, n_keys=8 if ok else 9, differ=[] if ok else [dict(key="bo1__ref")]),
                    logits=dict(max_abs_diff_draw0=0.0, max_abs_diff_draws=0.0, n=12), seconds_off=100.0, seconds_on=160.0)
    for sc, ok in (("AAA", True), ("BBB", False)):
        (tmp_path / f"identity_{sc}.json").write_text(json.dumps(res(sc, ok)))
    md = dsi.report(tmp_path)
    assert "G2: BLOCK" in md and "bo1__ref" in md and "1.60" in md
    assert json.loads((tmp_path / "draft_sealed_identity.json").read_text())["g2"] == "BLOCK"


# ---- G1: the scoreboard's draft side --------------------------------------------------------------------------------------------------

def _dc(b, half_width=0.3, n_events=100):
    return dict(n_events=n_events, n_games=900, a=1.0, a_ci=[0.5, 1.5], b=b, b_ci=[b - half_width, b + half_width], mean_predicted_gain_pp=3.0,
                event_ids=["1"], b_boots=[b])


def test_scoreboard_pools_draft_formats_apart_and_g1_blocks_on_a_weak_slope():
    res = {"AAA": dict(set="AAA", formats=dict(bo1=_dc(-1.0), dbo1=_dc(-1.0), dbo3=_dc(-0.9))),
           "BBB": dict(set="BBB", formats=dict(bo1=_dc(-1.2), dbo1=_dc(-0.9), dbo3=_dc(-0.4)))}
    rows = bsb.cells(res)
    assert {r["fmt"] for r in rows} == {"bo1", "dbo1", "dbo3"}
    pooled = bsb.pooled_rows(rows)
    assert pooled["all"]["k"] == 2 and pooled["dbo1"]["k"] == 2 and pooled["bo1"]["k"] == 2          # "all" stays the Sealed rows
    g1 = bsb.g1_lines(pooled)
    assert g1["dbo1"]["status"] == "PASS" and g1["dbo3"]["status"] == "BLOCK" and g1["status"] == "BLOCK"
    assert "G1: **BLOCK**" in bsb.markdown(rows, pooled)
    sealed_only = bsb.pooled_rows(bsb.cells({"AAA": res["AAA"] | dict(formats=dict(bo1=_dc(-1.0)))}))
    assert set(sealed_only) == {"bo1", "all"} and bsb.g1_lines(sealed_only)["status"] == "n/a"
    assert "G1" not in bsb.markdown(bsb.cells({"AAA": dict(set="AAA", formats=dict(bo1=_dc(-1.0)))}), sealed_only)


# ---- the second test's exclusions (docs/draft_mode2/RULE.md) -------------------------------------------------------------------------------

def test_second_test_constants_follow_the_rule():
    r1, r2 = dsh.RULES["draft_mode"], dsh.RULES["draft_mode2"]
    assert (r1["exclude_first"], r1["pool_seed"], r1["n_pools"], r1["sets"]) == (False, 0, 10, None)         # the first test is the default
    assert (r2["exclude_first"], r2["pool_seed"], r2["n_pools"]) == (True, 1, 20) and r2["sets"] == ["FRA", "HOB", "MSH", "SOS", "TMT"]
    assert (dsh.FIRST_G1_MAX_EVENTS, dsh.FIRST_G1_SEED, dsh.SECOND_G1_MAX_EVENTS) == (300, 0, 2000)


def test_new_pools_exclude_the_first_draw_exactly_and_never_overlap():
    el = [f"P{i:04d}" for i in range(300)]
    first = dsh.first_test_pools(el, "TST", "dbo1")
    assert first == dsh.draw_pools(el, 10, dsh.pool_seed(0, "TST", "dbo1")) and len(first) == 10      # the first test's own call
    new = dsh.draw_new_pools(el, first, "TST", "dbo1")
    assert len(new) == 20 and not set(new) & set(first) and set(new) <= set(el)
    assert new == dsh.draw_new_pools(el, first, "TST", "dbo1")                                         # deterministic
    assert new == dsh.draw_pools([d for d in el if d not in first], 20, dsh.pool_seed(1, "TST", "dbo1"))   # seed 1 among the rest
    assert new != dsh.draw_new_pools(el, first, "TST", "dbo1", seed=2)
    assert dsh.draw_new_pools(el, first, "TST", "dbo3") != new                                        # cells draw independently
    few = el[:25]                                                                                       # fewer left than asked: all of the rest
    f2 = dsh.first_test_pools(few, "TST", "dbo1")
    assert dsh.draw_new_pools(few, f2, "TST", "dbo1") == sorted(set(few) - set(f2))
    assert dsh.draw_new_pools(el[:10], dsh.first_test_pools(el[:10], "TST", "dbo1"), "TST", "dbo1") == []   # nothing left


def test_new_pools_overlap_is_asserted(monkeypatch):
    el = [f"P{i}" for i in range(50)]
    monkeypatch.setattr(dsh, "draw_pools", lambda ids, k, seed: ["P1"])       # a broken draw that returns an excluded pool
    with pytest.raises(AssertionError):
        dsh.draw_new_pools(el, ["P1", "P2"], "TST", "dbo1")


def test_first_g1_events_are_the_deviation_cost_choice():
    ids = [f"E{i}" for i in range(1000)][::-1]                    # event_starts order: arbitrary, not sorted
    rng = np.random.default_rng(0)                                # deviation_cost.py's literal code of the first test (--max-events 300)
    old = sorted(rng.choice(ids, 300, replace=False).tolist())
    assert dsh.first_g1_events(ids) == old and len(old) == 300
    small = ids[:200]
    assert dsh.first_g1_events(small) == small                    # not more than 300: all of them (and so none left for the second test)


def test_new_g1_events_exclude_the_first_300_take_2000_and_ignore_the_table_order():
    ids = [f"E{i:05d}" for i in range(5000)]
    first = dsh.first_g1_events(ids)
    new = dsh.new_g1_events(ids, first)
    assert len(new) == 2000 and new == sorted(new) and not set(new) & set(first) and set(new) <= set(ids)
    assert new == dsh.new_g1_events(ids[::-1], first)                                  # same list whatever the order of the event table
    rest = sorted(set(ids) - set(first))
    assert new == sorted(np.random.default_rng(1).choice(rest, 2000, replace=False).tolist())      # default_rng(1) among the remaining
    assert new != dsh.new_g1_events(ids, first, seed=0)
    few = ids[:700]                                                                       # 700 held-out: 400 left, all taken
    f2 = dsh.first_g1_events(few)
    assert dsh.new_g1_events(few, f2) == sorted(set(few) - set(f2)) and len(dsh.new_g1_events(few, f2)) == 400
    assert dsh.new_g1_events(ids[:250], dsh.first_g1_events(ids[:250])) == []          # all 250 were used by the first G1
    ints = list(range(1000))                                                              # integer ids work too
    assert not set(dsh.new_g1_events(ints, dsh.first_g1_events(ints))) & set(dsh.first_g1_events(ints))


def test_new_g1_events_overlap_is_asserted(monkeypatch):
    ids = [f"E{i}" for i in range(3000)]
    class Fake:                                                    # a broken draw that returns an excluded event
        def choice(self, a, size, replace):
            return np.array(["E1"] * size)

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: Fake())
    with pytest.raises(AssertionError):
        dsh.new_g1_events(ids, ["E0"] + ["E1"], max_events=2000)


def _write_corpus(src, code="TST", n_prem=400, n_trad=300):
    g = _games(n_sealed=300, n_prem=n_prem, n_trad=n_trad, n_other=20)
    d = src / "sets" / code / "corpus"
    d.mkdir(parents=True)
    g.write_parquet(d / "games.parquet")
    ids = g["draft_id"].unique().to_list()
    pl.DataFrame([dict(draft_id=i, card_id=c, n=20) for i in ids for c in (1, 2)]).write_parquet(d / "pools.parquet")


def test_cmd_pools_second_rule_excludes_the_first_draw_and_reports_a_comparison(tmp_path):
    src = tmp_path / "src"
    _write_corpus(src)
    o1, o2, o3 = tmp_path / "o1", tmp_path / "o2", tmp_path / "o3"
    for o in (o1, o2, o3):
        o.mkdir()
    dsh.cmd_pools(src, o1, ["TST"])                                                      # the first test's call: defaults unchanged
    first = json.loads((o1 / "pools.json").read_text())
    assert first["n_per_cell"] == 10 and "exclude_first" not in first and len(first["pools"]) == 20
    dsh.cmd_pools(src, o2, ["TST"], exclude_first=True, n_pools=20, pool_seed_value=1, first_pools_json=o1 / "pools.json")
    new = json.loads((o2 / "pools.json").read_text())

    def key(e):
        return (e["set"], e["fmt"], e["draft_id"])

    assert len(new["pools"]) == 40 and not {key(e) for e in new["pools"]} & {key(e) for e in first["pools"]}
    assert new["seed"] == 1 and new["halves_seed"] == 0 and new["overlap_with_first_test"] == 0
    assert all(c["drawn"] == 20 and c["eligible_after_exclusion"] == c["eligible"] - 10 for c in new["cells"].values())
    assert all(v["same"] for v in new["first_test_comparison"].values())                  # recomputed draw == the first test's saved file
    assert {c: sorted(ids) for c, ids in new["first_test_pools"].items()} == {
        f"TST {f}": sorted(e["draft_id"] for e in first["pools"] if e["fmt"] == f) for f in ("dbo1", "dbo3")}
    other = json.loads(json.dumps(first))
    other["pools"][0]["draft_id"] = "something-else"
    (tmp_path / "other.json").write_text(json.dumps(other))
    dsh.cmd_pools(src, o3, ["TST"], exclude_first=True, n_pools=20, pool_seed_value=1, first_pools_json=tmp_path / "other.json")
    doc3 = json.loads((o3 / "pools.json").read_text())
    assert not all(v["same"] for v in doc3["first_test_comparison"].values())              # a difference is reported ...
    assert any(v["only_recomputed"] for v in doc3["first_test_comparison"].values())
    assert doc3["pools"] == new["pools"]                                                  # ... and the pools themselves are the recomputed draw's
