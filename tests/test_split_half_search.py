"""scripts/split_half_search.py: the event split and the summary arithmetic (no model fits)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("split_half_search", _SCRIPTS / "split_half_search.py")
sh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sh)


def _events(n_by_fmt=(("bo1", 101), ("bo3", 60), ("draft", 33)), seed=1):
    rng = np.random.default_rng(seed)
    ids, starts, fmts = [], [], []
    for f, n in n_by_fmt:
        for k in range(n):
            ids.append(f"{f}-{k}")
            starts.append(float(rng.uniform(0, 1000)))
            fmts.append(f)
    return ids, starts, fmts


# ---- the event split -------------------------------------------------------------------------------------------------------------

def test_split_is_disjoint_and_covers_every_event_once():
    ids, starts, fmts = _events()
    a, b = sh.split_events(ids, starts, fmts, seed=3)
    assert not set(a) & set(b)
    assert sorted(a + b) == sorted(ids)                    # every event whole, in exactly one half
    assert len(a) + len(b) == len(ids)


def test_split_is_balanced_by_format():
    ids, starts, fmts = _events()
    a, b = sh.split_events(ids, starts, fmts, seed=3)
    for f in ("bo1", "bo3", "draft"):
        na, nb = sum(i.startswith(f) for i in a), sum(i.startswith(f) for i in b)
        assert abs(na - nb) <= 1, (f, na, nb)


def test_split_is_balanced_in_time():
    ids, starts, fmts = _events((("bo1", 400),))
    t = dict(zip(ids, starts))
    a, b = sh.split_events(ids, starts, fmts, seed=0)
    # every consecutive pair in start order is split: the k-th order statistics of the halves can differ by at most one rank
    order = {i: r for r, i in enumerate(sorted(ids, key=lambda i: (t[i], i)))}
    ra, rb = np.sort([order[i] for i in a]), np.sort([order[i] for i in b])
    assert np.max(np.abs(ra - rb)) <= 1
    assert abs(np.mean([t[i] for i in a]) - np.mean([t[i] for i in b])) < 25          # of a 1000-wide period


def test_split_is_random_but_reproducible():
    ids, starts, fmts = _events()
    a1, _ = sh.split_events(ids, starts, fmts, seed=5)
    a2, _ = sh.split_events(ids, starts, fmts, seed=5)
    a3, _ = sh.split_events(ids, starts, fmts, seed=6)
    assert a1 == a2 and set(a1) != set(a3)
    assert sh.set_seed(0, "MSH") != sh.set_seed(0, "SOS") and sh.set_seed(0, "MSH") == sh.set_seed(0, "MSH")


def test_split_edge_cases():
    assert sh.split_events([], [], []) == ([], [])
    a, b = sh.split_events(["x"], [1.0], ["bo1"])
    assert len(a) + len(b) == 1
    with pytest.raises(ValueError):
        sh.split_events(["x", "x"], [1.0, 2.0], ["bo1", "bo1"])
    with pytest.raises(ValueError):
        sh.split_events(["x"], [1.0, 2.0], ["bo1"])


# ---- bootstrap arithmetic --------------------------------------------------------------------------------------------------------

def test_cluster_boot_point_ci_and_clusters():
    v = np.array([1.0, 3.0, 2.0, 2.0])
    pt, lo, hi = sh.cluster_boot(v, np.ones(4), ["a", "a", "b", "b"], n_boot=500, seed=1)
    assert pt == pytest.approx(2.0) and lo <= pt <= hi
    # constant data: zero-width interval
    pt, lo, hi = sh.cluster_boot(np.full(10, 0.7), np.ones(10), list("aabbccddee"), n_boot=200)
    assert (pt, lo, hi) == (pytest.approx(0.7), pytest.approx(0.7), pytest.approx(0.7))
    # the interval is wider when the same values sit in fewer clusters
    rng = np.random.default_rng(0)
    x = np.repeat(rng.normal(0.5, 1.0, 10), 20) + rng.normal(0, 0.1, 200)            # strong cluster effect
    few = sh.cluster_boot(x, np.ones(200), np.repeat(np.arange(10), 20), n_boot=500, seed=2)
    many = sh.cluster_boot(x, np.ones(200), np.arange(200), n_boot=500, seed=2)
    assert (few[2] - few[1]) > (many[2] - many[1])


def test_cluster_boot_ratio():
    num = np.array([1.0, 2.0, 3.0, 4.0])
    den = np.array([2.0, 4.0, 6.0, 8.0])
    pt, lo, hi = sh.cluster_boot(num, den, [0, 1, 2, 3], n_boot=200)
    assert pt == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    assert np.isnan(sh.cluster_boot([], [], [])[0])


# ---- records and summaries -------------------------------------------------------------------------------------------------------

def _row(sc, fmt, did, a, b):
    """a, b: {variant: (ls, lo, deck)} of direction A and B."""
    def blk(d):
        return {v: dict(ls=ls, lo=lo, deck=deck, seconds=2.0 if v != "default" else 1.0) for v, (ls, lo, deck) in d.items()}
    return dict(set=sc, fmt=fmt, draft_id=did, dirs={"A": blk(a), "B": blk(b)})


def test_pool_direction_gaps_in_percentage_points():
    row = _row("MSH", "bo1", "d1",
               {"default": (0.0, 0.0, {"1": 40}), "d0_fast": (0.2, 0.1, {"1": 39, "2": 1}), "d0_mid": (0.2, -0.1, {"1": 39, "2": 1})},
               {"default": (0.0, 0.0, {"1": 40}), "d0_fast": (0.0, 0.0, {"1": 40}), "d0_mid": (0.4, 0.0, {"1": 38, "2": 2})})
    recs = sh.pool_direction_records([row])
    assert len(recs) == 4                                              # 2 directions x 2 variants
    r = next(x for x in recs if x["dir"] == "A" and x["variant"] == "d0_fast")
    assert r["gap_in"] == pytest.approx(100 * (expit(0.2) - 0.5)) and r["gap_out"] == pytest.approx(100 * (expit(0.1) - 0.5))
    assert r["dl_in"] == pytest.approx(0.2) and r["dl_out"] == pytest.approx(0.1)
    assert not r["same_deck"]
    r = next(x for x in recs if x["dir"] == "B" and x["variant"] == "d0_fast")
    assert r["same_deck"] and r["gap_in"] == 0.0
    r = next(x for x in recs if x["dir"] == "A" and x["variant"] == "d0_mid")
    assert r["gap_out"] < 0 < r["gap_in"]


def test_summary_means_ratio_and_shares():
    rows = []
    for k in range(40):                                                 # in-sample gap 1 pp-ish, out-of-sample half of it, one negative
        gi = 0.04
        go = 0.02 if k % 8 else -0.02
        rows.append(_row("ECL", "bo3" if k % 2 else "bo1", f"p{k}",
                         {"default": (0.0, 0.0, {"1": 40}), "d0_fast": (gi, go, {"2": 40}), "d0_mid": (gi, go, {"2": 40})},
                         {"default": (0.0, 0.0, {"1": 40}), "d0_fast": (gi, go, {"2": 40}), "d0_mid": (gi, go, {"2": 40})}))
    summ = sh.full_summary(rows, n_boot=300)
    s = summ["variants"]["d0_fast"]["overall"]
    assert s["n"] == 80 and s["n_pools"] == 40
    exp_in = 100 * (expit(0.04) - 0.5)
    assert s["mean_gap_in"] == pytest.approx(exp_in)
    n_neg = sum(1 for k in range(40) if k % 8 == 0)
    exp_out = (100 * (expit(0.02) - 0.5) * (40 - n_neg) + 100 * (expit(-0.02) - 0.5) * n_neg) / 40
    assert s["mean_gap_out"] == pytest.approx(exp_out)
    assert s["ratio"] == pytest.approx(exp_out / exp_in)
    assert s["share_out_pos"] == pytest.approx(1 - n_neg / 40) and s["share_out_neg"] == pytest.approx(n_neg / 40)
    assert s["gap_out_ci"][0] <= s["mean_gap_out"] <= s["gap_out_ci"][1]
    assert s["ratio_ci"][0] <= s["ratio"] <= s["ratio_ci"][1]
    assert summ["variants"]["d0_fast"]["bo1"]["n"] + summ["variants"]["d0_fast"]["bo3"]["n"] == 80
    assert set(summ["variants"]["d0_mid"]["per_set"]) == {"ECL"}
    assert s["share_same_deck"] == 0.0 and s["mean_seconds"] == 2.0 and s["mean_seconds_default"] == 1.0


def test_self_preference():
    a = {"default": (logit(0.60), logit(0.55), {"1": 1}), "d0_fast": (0, 0, {}), "d0_mid": (0, 0, {})}     # deck A: 60 under A, 55 under B
    b = {"default": (logit(0.58), logit(0.50), {"2": 1}), "d0_fast": (0, 0, {}), "d0_mid": (0, 0, {})}     # deck B: 58 under B, 50 under A
    # row dirs are keyed by searching model: A's block holds deck A (ls = A's logit, lo = B's logit), B's block holds deck B
    row = dict(set="TDM", fmt="bo1", draft_id="x", dirs={
        "A": {k: dict(ls=v[0], lo=v[1], deck=v[2], seconds=1.0) for k, v in a.items()},
        "B": {k: dict(ls=v[0], lo=v[1], deck=v[2], seconds=1.0) for k, v in b.items()}})
    pr = sh.self_preference([row])[0]
    assert pr["pref_a"] == pytest.approx(10.0)         # model A: own deck 60 vs the other's deck 50
    assert pr["pref_b"] == pytest.approx(3.0)          # model B: own deck 58 vs deck A 55
    assert pr["pref"] == pytest.approx(6.5) and not pr["same_deck"]
    s = sh.summarise_pref([pr], n_boot=50)
    assert s["mean_self_preference_pp"] == pytest.approx(6.5) and s["n_pools"] == 1


def test_markdown_renders_and_progress_file(tmp_path):
    rows = [_row("MSH", "bo1", f"p{k}", {"default": (0.0, 0.0, {"1": 1}), "d0_fast": (0.05, 0.02, {"2": 1}), "d0_mid": (0.05, 0.03, {"2": 1})},
                 {"default": (0.0, 0.0, {"1": 1}), "d0_fast": (0.05, 0.02, {"2": 1}), "d0_mid": (0.05, 0.03, {"2": 1})}) for k in range(6)]
    md = sh.markdown(sh.full_summary(rows, n_boot=50), dict(fits={}, model_opts={}, n_boot=4, search_wall_seconds=60))
    assert "d0_mid against default" in md and "| ALL |" in md and "| MSH |" in md
    p = sh.Progress(tmp_path, "PLAN: 22 fits")
    p.done("fit 1/22 done (ECL A)")
    p.set_status("searching")
    lines = (tmp_path / "progress.txt").read_text().splitlines()
    assert lines[0] == "PLAN: 22 fits" and lines[1].startswith("fit 1/22 done (ECL A) ") and "searching" in lines[-1]
    assert sh.Progress(tmp_path, "PLAN: 22 fits").units == p.units            # resumes from disk


def test_gaps_are_converted_at_the_searching_models_level():
    # the other model sits 0.8 logit higher: its logit difference must not be turned into points at its own (flatter) level
    row = _row("OM1", "bo1", "z",
               {"default": (0.0, 0.8, {"1": 1}), "d0_fast": (0.2, 1.0, {"2": 1}), "d0_mid": (0.2, 1.0, {"2": 1})},
               {"default": (0.8, 0.0, {"1": 1}), "d0_fast": (1.0, 0.2, {"2": 1}), "d0_mid": (1.0, 0.2, {"2": 1})})
    for r in sh.pool_direction_records([row]):
        assert r["dl_in"] == pytest.approx(0.2) and r["dl_out"] == pytest.approx(0.2)
        l0 = 0.0 if r["dir"] == "A" else 0.8
        assert r["gap_in"] == pytest.approx(r["gap_out"]) == pytest.approx(100 * (expit(l0 + 0.2) - expit(l0)))
    s = sh.summarise_records(sh.pool_direction_records([row]), n_boot=50)
    assert s["ratio"] == pytest.approx(1.0) and s["ratio_logit"] == pytest.approx(1.0)


# ---- forcing model options / selecting pools (matchups acceptance check) --------------------------------------------------------

def test_parse_model_opts():
    assert sh.parse_model_opts(["matchups=on", "extra_draft=on"]) == {"matchups": "on", "extra_draft": "on"}
    assert sh.parse_model_opts([]) == {} and sh.parse_model_opts(None) == {}
    assert sh.parse_model_opts(["manasim=v1"]) == {"manasim": "v1"}
    for bad in ("matchups", "sigma_cv=random", "matchups=maybe", "manasim=on"):
        with pytest.raises(ValueError):
            sh.parse_model_opts([bad])


def test_select_tasks_by_set_and_cell():
    tasks = [dict(set=s, fmt=f) for s in ("EOE", "FRA", "MSH") for f in ("bo1", "bo3")]
    assert sh.select_tasks(tasks, None) == tasks and sh.select_tasks(tasks, []) == tasks
    got = {(t["set"], t["fmt"]) for t in sh.select_tasks(tasks, ["eoe", "FRA:bo3"])}
    assert got == {("EOE", "bo1"), ("EOE", "bo3"), ("FRA", "bo3")}
