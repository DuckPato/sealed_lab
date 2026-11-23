"""scripts/matchups_accept.py: the acceptance arithmetic of the matchups pooled-rule check (no model fits)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("matchups_accept", _SCRIPTS / "matchups_accept.py")
ma = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ma)

NAMES = ("A_cur", "B_cur", "A_on", "B_on")


def _zero_L():
    return {d: {m: 0.0 for m in NAMES} for d in NAMES}


def _row(sc="EOE", fmt="bo1", did="d", L=None, decks=None):
    return dict(set=sc, fmt=fmt, draft_id=did, decks=decks or {d: {"1": 1} for d in NAMES}, L=L or _zero_L(), chk_stored=0.0)


def test_gap_out_scored_by_the_other_half_in_pp():
    L = _zero_L()
    L["A_on"]["B_on"], L["A_cur"]["B_on"] = 0.30, 0.10        # B_on prefers A's matchups deck
    L["A_on"]["B_cur"], L["A_cur"]["B_cur"] = 0.05, 0.10      # B_cur prefers A's current deck
    decks = {d: {"1": 1} for d in NAMES}
    decks["A_on"] = {"1": 2}
    recs, _ = ma.pool_records(_row(L=L, decks=decks))
    a = next(r for r in recs if r["dir"] == "A")
    assert a["gap_out"] == pytest.approx(100 * (expit(0.30) - expit(0.10)))
    assert a["gap_out_cur"] == pytest.approx(100 * (expit(0.05) - expit(0.10)))
    assert a["changed"] is True
    b = next(r for r in recs if r["dir"] == "B")
    assert b["changed"] is False and b["gap_out"] == 0.0 and b["gap_in"] == 0.0


def test_gap_in_is_scored_by_the_searching_matchups_model():
    L = _zero_L()
    L["B_on"]["B_on"], L["B_cur"]["B_on"] = 0.4, 0.1
    recs, _ = ma.pool_records(_row(L=L))
    b = next(r for r in recs if r["dir"] == "B")
    assert b["gap_in"] == pytest.approx(100 * (expit(0.4) - expit(0.1)))


def test_T_is_the_mean_self_preference_in_pp():
    L = _zero_L()
    # current: A scores its own deck above B's, B scores its own above A's  (L[deck][model])
    L["A_cur"]["A_cur"], L["B_cur"]["A_cur"] = 0.5, 0.3
    L["B_cur"]["B_cur"], L["A_cur"]["B_cur"] = 0.2, 0.1
    _, t = ma.pool_records(_row(L=L))
    want = 100 * 0.5 * ((expit(0.5) - expit(0.3)) + (expit(0.2) - expit(0.1)))
    assert t["T_cur"] == pytest.approx(want)
    assert t["T_on"] == 0.0 and t["T_diff"] == pytest.approx(-want)


def test_identical_arms_give_zero_gap_and_pass():
    rows = [_row("EOE", "bo1", f"d{i}") for i in range(6)] + [_row("FRA", "bo3", f"f{i}") for i in range(4)]
    s = ma.check_a_summary(rows, n_boot=200)
    g = s["pooled"]["gap"]
    assert g["gap_out"]["mean"] == 0.0 and g["share_changed"] == 0.0
    assert s["pass_a"] is True and s["n_pools"] == 10


def test_only_switch_cells_count():
    rows = [_row("FRA", "bo1", "x"), _row("ECL", "bo3", "y"), _row("FRA", "bo3", "z"), _row("EOE", "bo1", "w")]
    s = ma.check_a_summary(rows, n_boot=50)
    assert s["n_pools"] == 2 and s["cells"] == ["EOE bo1", "FRA bo3"]
    assert ma.in_switch_cells("MSH", "bo1") and not ma.in_switch_cells("FRA", "bo1") and not ma.in_switch_cells("OM1", "bo3")


def test_pass_a_requires_ci_upper_bound_at_or_above_zero():
    rows = []
    for i in range(40):                                       # matchups decks clearly worse under the other half's matchups model
        L = _zero_L()
        L["A_on"]["B_on"], L["B_on"]["A_on"] = -0.3, -0.3
        decks = {d: {"1": 1} for d in NAMES}
        decks["A_on"], decks["B_on"] = {"1": 2}, {"1": 3}
        rows.append(_row("EOE", "bo1", f"d{i}", L=L, decks=decks))
    s = ma.check_a_summary(rows, n_boot=200)
    assert s["pooled"]["gap"]["gap_out"]["mean"] < 0 and s["pooled"]["gap"]["gap_out"]["ci"][1] < 0
    assert s["pass_a"] is False and s["pooled"]["gap"]["share_changed"] == 1.0
    assert ma.pass_a({"gap_out": {"ci": [-0.2, 0.0]}}) and not ma.pass_a({"gap_out": {"ci": [-0.2, -0.01]}}) and not ma.pass_a({})


def test_both_on_sensitivity_excludes_sets():
    rows = [_row("EOE", "bo1", "a"), _row("TDM", "bo1", "b")]
    s = ma.check_a_summary(rows, n_boot=50, both_on_sets=["TDM"])
    assert s["pooled_excl_both_on"]["gap"]["n_pools"] == 1


# ---- Check B ------------------------------------------------------------------------------------------------------------------

def _cell(b, se, boots):
    return dict(b=b, b_ci=[b - 1.96 * se, b + 1.96 * se], b_boots=list(boots))


def test_pooled_b_is_inverse_variance_and_replicates_pool_the_same_way():
    cells = [_cell(-1.0, 0.1, [-1.1, -0.9, -1.0]), _cell(-0.5, 0.2, [-0.6, -0.4, -0.5])]
    p = ma.pooled_b(cells)
    w = np.array([1 / 0.1 ** 2, 1 / 0.2 ** 2])
    assert p["b"] == pytest.approx((w * [-1.0, -0.5]).sum() / w.sum(), abs=1e-6)
    assert p["se"] == pytest.approx(1 / np.sqrt(w.sum()), rel=1e-4)
    assert p["reps"][1] == pytest.approx((w * [-0.9, -0.4]).sum() / w.sum())
    assert "reps" not in ma.pooled_b([dict(b=-1.0, b_ci=[-1.2, -0.8])])


def test_paired_abs_dev_sign_and_significance():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.05, size=(200, 3))                   # shared noise -> paired
    cur = [_cell(-0.8, 0.05, -0.8 + base[:, k]) for k in range(3)]
    worse = [_cell(-0.5, 0.05, -0.5 + base[:, k]) for k in range(3)]            # from -0.8 to -0.5: abs(b+1) 0.2 -> 0.5
    r = ma.paired_abs_dev(cur, worse)
    assert r["diff_abs_dev"] == pytest.approx(0.3, abs=1e-6)
    assert r["diff_abs_dev_ci"][0] > 0 and r["significantly_worse"] is True        # paired: the shared noise cancels
    better = [_cell(-1.0, 0.05, -1.0 + base[:, k]) for k in range(3)]               # to -1: abs(b+1) 0.2 -> ~0
    r2 = ma.paired_abs_dev(cur, better)
    assert r2["diff_abs_dev"] < 0 and r2["significantly_worse"] is False
    same = ma.paired_abs_dev(cur, cur)
    assert same["diff_abs_dev"] == 0.0 and same["diff_abs_dev_ci"] == [0.0, 0.0] and same["significantly_worse"] is False
    with pytest.raises(ValueError):
        ma.paired_abs_dev(cur, worse[:2])
    with pytest.raises(ValueError):
        ma.paired_abs_dev([dict(b=-1, b_ci=[-1.1, -0.9])], [dict(b=-1, b_ci=[-1.1, -0.9])])


def test_cell_pairs_flags_identical_and_unpaired_arms():
    def res(sc, b, ids, boots):
        return {sc: dict(set=sc, formats={"bo1": dict(n_events=3, b=b, b_ci=[b - .1, b + .1], event_ids=ids, b_boots=boots)})}
    cur = res("EOE", -0.9, ["1", "2"], [-0.9, -0.8, -1.0])
    rows = ma.cell_pairs(cur, res("EOE", -0.9, ["1", "2"], [-0.9, -0.8, -1.0]))
    assert rows[0]["identical"] is True and rows[0]["diff_abs_dev"] == 0.0 and rows[0]["diff_abs_dev_ci"] == [0.0, 0.0]
    rows = ma.cell_pairs(cur, res("EOE", -0.7, ["1", "2"], [-0.7, -0.6, -0.8]))
    assert rows[0]["identical"] is False and rows[0]["diff_abs_dev"] == pytest.approx(0.2)
    rows = ma.cell_pairs(cur, res("EOE", -0.7, ["1", "3"], [-0.7, -0.6, -0.8]))
    assert rows[0]["same_events"] is False and "diff_abs_dev_ci" not in rows[0]


def test_pass_b_and_markdown_render():
    assert ma.pass_b({"bo1": {"significantly_worse": False}, "bo3": {"significantly_worse": False}})
    assert not ma.pass_b({"bo1": {"significantly_worse": False}, "bo3": {"significantly_worse": True}})
    rows = [_row("EOE", "bo1", f"d{i}") for i in range(4)]
    a = ma.check_a_summary(rows, n_boot=50)
    md = ma.markdown(a, None, {"k": "v"})
    assert "PASS A" in md and "EOE" in md and "- k: v" in md


def test_model_logit_shift_detects_any_change_between_arms():
    assert ma.model_logit_shift([_row()]) == 0.0
    L = _zero_L()
    L["B_cur"]["B_on"] = 0.25
    assert ma.model_logit_shift([_row(), _row(L=L)]) == pytest.approx(0.25)
    assert ma.check_a_summary([_row(L=L)], n_boot=20)["max_logit_shift"] == pytest.approx(0.25)


def test_model_dir_diff_flags_changed_arrays_and_expected_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for d in (a, b):
        np.savez(d / "draws.npz", x=np.arange(5.0))
        np.savez(d / "reference.npz", y=np.ones(3))
        (d / "archetypes.json").write_text(json.dumps({"r": 1}), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"m": 1}), encoding="utf-8")
    np.savez(b / "opp.npz", delta=np.zeros(2))                                   # the matchup layer exists only in the on model
    (b / "archetypes.json").write_text(json.dumps({"r": 2}), encoding="utf-8")   # descriptive intervals differ
    d = ma.model_dir_diff(a, b)
    assert d["draws.npz"] == "identical" and d["opp.npz"] == "only in B" and d["archetypes.json"] == "DIFFERS" and d["meta.json"] == "(not compared)"
    assert ma.unexpected_diffs(d) == {}
    np.savez(b / "reference.npz", y=np.ones(3) + 1e-12)                          # a changed builder array must be flagged
    assert ma.unexpected_diffs(ma.model_dir_diff(a, b)) == {"reference.npz": "DIFFERS in y"}
    s = ma.model_diff_summary({"p": (a, b)})
    assert s["n_pairs"] == 1 and s["n_unexpected"] == 1


def test_check_b_summary_from_saved_runs(tmp_path):
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.05, size=(30, 2))

    def write(root, arm, sc, shift, n_boot=30):
        d = root / arm / sc
        d.mkdir(parents=True)
        fm = {}
        for j, fmt in enumerate(("bo3", "bo1")):
            boots = (-0.8 + shift + noise[:n_boot, j]).tolist()
            fm[fmt] = dict(n_events=10, n_games=100, b=-0.8 + shift, b_ci=[-0.9 + shift, -0.7 + shift], a=1.0, a_ci=[0.9, 1.1],
                           event_ids=["1", "2"], b_boots=boots, build_seconds=1.0, mean_predicted_gain_pp=1.0)
        (d / "deviation_cost.json").write_text(json.dumps(dict(set=sc, fit_seconds=2.0, formats=fm)), encoding="utf-8")

    full, small = tmp_path / "full", tmp_path / "small"
    for sc in ("EOE", "TMT"):
        write(full, "cur", sc, 0.0)
        write(full, "on", sc, 0.0)
    write(small, "cur", "FRA", 0.0, 20)
    write(small, "on", "FRA", 0.3, 20)                            # this arm moves b away from -1: significantly worse
    s = ma.check_b_summary([full, small])
    assert set(s["groups"]) == {"full", "small"} and len(s["cells"]) == 6
    assert s["groups"]["full"]["pass_b"] is True and s["groups"]["small"]["pass_b"] is False and s["pass_b"] is False
    v = s["groups"]["full"]["by_format"]["bo3"]
    assert v["diff_abs_dev"] == 0.0 and v["diff_abs_dev_ci"] == [0.0, 0.0] and v["cur"]["k"] == 2 and v["n_boot"] == 30
    assert s["groups"]["small"]["by_format"]["bo1"]["diff_abs_dev"] == pytest.approx(0.3)
    md = ma.markdown(None, s, {})
    assert "PASS B" in md and "FAIL" in md
