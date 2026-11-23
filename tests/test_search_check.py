"""scripts/search_check.py: the budget variants' patching / restoring and the summary arithmetic (no model fits)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit

from sealed_lab.deck import builder, families

_SPEC = importlib.util.spec_from_file_location("search_check", Path(__file__).resolve().parents[1] / "scripts" / "search_check.py")
sc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sc)


class _Ctx:
    def __init__(self, draw):
        self.draw = draw


def _consts():
    return {k: getattr(builder, k) for k in ("N_SCREEN", "N_CONFIG_FULL", "N_LS_STARTS", "SWAP_K", "N_PAIR_STARTS", "MAX_LS_ITER")} | \
           {k: getattr(families, k) for k in ("N_SEED_MAINS", "MAX_SEEDS", "SEED_MAX_GAP_PP", "SEED_APART", "KEEP_DRAW0", "KEEP_OTHER")}


# ---- patching ------------------------------------------------------------------------------------------------------------------

def test_every_variant_names_real_constants_and_restores_everything():
    before, run0, search0 = _consts(), builder.Searcher.run, builder.run_search
    pick0, map0 = families._pick_seeds, families._Runner.map
    for name, spec in sc.VARIANTS.items():
        for k in (spec or {}).get("builder", {}):
            assert hasattr(builder, k), (name, k)          # a misspelt constant would be set on the module and silently do nothing
        for k in (spec or {}).get("families", {}):
            assert hasattr(families, k), (name, k)
        with sc.patched_search(spec):
            pass
        assert _consts() == before and builder.Searcher.run is run0 and builder.run_search is search0
        assert families._pick_seeds is pick0 and families._Runner.map is map0, name


def test_restores_after_an_exception():
    before, run0 = _consts(), builder.Searcher.run
    with pytest.raises(RuntimeError):
        with sc.patched_search(sc.VARIANTS["large"]):
            assert families.KEEP_DRAW0 == 200 and builder.Searcher.run is not run0
            raise RuntimeError("boom")
    assert _consts() == before and builder.Searcher.run is run0


def test_unknown_constant_is_an_error_not_silently_ignored(monkeypatch):
    _record_search(monkeypatch)
    with pytest.raises(AttributeError):
        with sc.patched_search(dict(builder=dict(N_SCREEEN=1))):
            builder.run_search(_Ctx(0), None, 1)
    with pytest.raises(AttributeError):
        with sc.patched_search(dict(families=dict(KEEP_NOTHING=1))):
            pass


def test_large_matches_the_original_large_budget():
    spec = sc.VARIANTS["large"]
    assert spec["builder"] == sc.LARGE["builder"] and spec["families"] == sc.LARGE["families"]
    assert spec["ls_iter"] == 60 and spec["n_decks"] == 20 and spec["pairs_all"] and not spec.get("only_draw0")


def _record_search(monkeypatch):
    seen = []

    def fake_search(ctx, prep, n_decks=1, *a, **k):
        seen.append(dict(draw=ctx.draw, n_decks=n_decks, screen=builder.N_SCREEN, cfg=builder.N_CONFIG_FULL, starts=builder.N_LS_STARTS))
        return "ranked"
    monkeypatch.setattr(builder, "run_search", fake_search)
    return seen


def _record_run(monkeypatch):
    seen = []

    def fake_run(self, deck, max_iter=None, deadline=None, pairs=True):
        seen.append(dict(draw=self.ctx.draw, max_iter=max_iter, pairs=pairs, swap_k=builder.SWAP_K))
        return deck, 0.0
    monkeypatch.setattr(builder.Searcher, "run", fake_run)
    return seen


def test_draw0_large_patches_only_draw_zero(monkeypatch):
    seen_s, seen_r = _record_search(monkeypatch), _record_run(monkeypatch)
    searcher = builder.Searcher.__new__(builder.Searcher)
    with sc.patched_search(sc.VARIANTS["draw0_large"]):
        builder.run_search(_Ctx(0), None, 2)
        builder.run_search(_Ctx(3), None, 1)
        searcher.ctx = _Ctx(0)
        builder.Searcher.run(searcher, {}, pairs=False)
        searcher.ctx = _Ctx(3)
        builder.Searcher.run(searcher, {}, pairs=False)
    assert seen_s[0] == dict(draw=0, n_decks=20, screen=48, cfg=20, starts=20)
    assert seen_s[1] == dict(draw=3, n_decks=1, screen=builder.N_SCREEN, cfg=builder.N_CONFIG_FULL, starts=builder.N_LS_STARTS)
    assert seen_r[0] == dict(draw=0, max_iter=60, pairs=True, swap_k=48)
    assert seen_r[1] == dict(draw=3, max_iter=None, pairs=False, swap_k=builder.SWAP_K)      # None: the Searcher's own limit
    assert builder.N_SCREEN == 16 and builder.SWAP_K == 12


def test_ls_variant_forces_starts_but_not_the_number_of_configurations(monkeypatch):
    seen = _record_search(monkeypatch)
    base = builder.N_CONFIG_FULL
    with sc.patched_search(sc.VARIANTS["ls"]):
        builder.run_search(_Ctx(0), None, 2)
        builder.run_search(_Ctx(1), None, 1)
    assert [s["n_decks"] for s in seen] == [20, 20]
    assert seen[0]["cfg"] == base and seen[1]["cfg"] == base - 3          # what n_decks 2 / 1 would have used
    assert seen[0]["screen"] == builder.N_SCREEN and seen[0]["starts"] == 20


def test_cfg_variant_changes_only_screening_and_configurations(monkeypatch):
    seen = _record_search(monkeypatch)
    with sc.patched_search(sc.VARIANTS["cfg"]):
        builder.run_search(_Ctx(0), None, 2)
    assert seen[0] == dict(draw=0, n_decks=2, screen=48, cfg=20, starts=builder.N_LS_STARTS)


def test_pairs_variant_turns_pairs_on_everywhere_but_keeps_iterations(monkeypatch):
    seen = _record_run(monkeypatch)
    searcher = builder.Searcher.__new__(builder.Searcher)
    searcher.ctx = _Ctx(2)
    with sc.patched_search(sc.VARIANTS["pairs"]):
        builder.Searcher.run(searcher, {}, pairs=False)
    assert seen[0]["pairs"] is True and seen[0]["max_iter"] is None and seen[0]["swap_k"] == builder.SWAP_K


def test_seed_polishing_beyond_the_default_seeds_is_draw_zero_only(monkeypatch):
    def fake_pick(rows0, st):
        return [(((k, 1),), "WU") for k in range(families.N_SEED_MAINS)]
    seen = []

    def fake_map(self, tasks, progress=None):
        seen.append(list(tasks))
        return [[] for _ in tasks]
    monkeypatch.setattr(families, "_pick_seeds", fake_pick)
    monkeypatch.setattr(families._Runner, "map", fake_map)
    with sc.patched_search(sc.VARIANTS["draw0_large"]):
        seeds = families._pick_seeds(None, None)
        assert len(seeds) == 8 and families.N_SEED_MAINS == 8
        tasks = [("polish", (0, 1, 2), [sd]) for sd in seeds] + [("polish", (1, 2), [seeds[7]]), ("search", 1)]
        families._Runner.map(None, tasks)
    assert families.N_SEED_MAINS == 4 and families._Runner.map is not None
    out = seen[0]
    keep = [t for t in out if t[0] == "polish" and t[2][0] == seeds[0]]
    assert keep == [("polish", (0, 1, 2), [seeds[0]])]                       # a default seed keeps all draws
    extra = [t for t in out if t[0] == "polish" and t[2][0] == seeds[7]]
    assert extra == [("polish", (0,), [seeds[7]])]                           # an extra seed: draw 0 only; its draws-1..2 task is dropped
    assert ("search", 1) in out


# ---- summary arithmetic --------------------------------------------------------------------------------------------------------

def _row(s, fmt, did, probs, deck, seconds=10.0, **kw):
    L = logit(np.asarray(probs, dtype=float))
    return dict(set=s, fmt=fmt, draft_id=did, logit_default=float(L[0]), logit=float(L[0]), L=L.tolist(), deck=deck, seconds=seconds,
                n_candidates=50, **kw)


def _data():
    d = [_row("A", "bo3", "1", [0.50, 0.50, 0.50, 0.50], {"1": 17}), _row("B", "bo1", "2", [0.55, 0.50, 0.60, 0.50], {"2": 17}, 20.0)]
    v = [_row("A", "bo3", "1", [0.50, 0.50, 0.50, 0.50], {"1": 17}, 30.0),                     # same deck
         _row("B", "bo1", "2", [0.57, 0.52, 0.58, 0.54], {"3": 17}, 60.0)]                     # +2 pp at draw 0; +2, -2, +4 on the draws
    return {"default": d, "large": v, "half": [dict(r) for r in v]}


def test_pool_gaps_and_aggregate_arithmetic():
    data = _data()
    per = sc.pool_gaps(data["large"], data["default"])
    assert len(per) == 2
    p = {x["set"]: x for x in per}
    assert p["A"]["same"] and p["A"]["gap0"] == 0 and p["A"]["share_ge"] == 1.0
    assert p["B"]["gap0"] == pytest.approx(2.0)
    assert p["B"]["gap_rep"] == pytest.approx((2 - 2 + 4) / 3)
    assert p["B"]["share_ge"] == pytest.approx(2 / 3)
    a = sc.aggregate(per, large_mean_gap=1.0)
    assert a["pools"] == 2 and a["n_better"] == 1 and a["share_better"] == 0.5 and a["n_worse"] == 0 and a["n_same_deck"] == 1
    assert a["mean_gap_pp"] == pytest.approx(1.0) and a["captured"] == pytest.approx(1.0)
    assert a["mean_gap_rep_pp"] == pytest.approx((0 + 4 / 3) / 2)
    assert a["share_draws_ge"] == pytest.approx((1 + 2 / 3) / 2) and a["share_draws_ge_diff"] == pytest.approx(2 / 3)
    assert a["time_ratio"] == pytest.approx(45.0 / 15.0)
    assert a["mean_gap_pp_when_better"] == pytest.approx(2.0)


def test_a_worse_variant_counts_as_worse_and_ties_are_not_better():
    d = [_row("A", "bo3", "1", [0.50, 0.50, 0.50], {"1": 1})]
    v = [_row("A", "bo3", "1", [0.4999999999, 0.50, 0.50], {"2": 1})]
    a = sc.aggregate(sc.pool_gaps(v, d))
    assert a["n_better"] == 0 and a["n_worse"] == 0                       # below the 0.0001 pp numerical-noise threshold
    v = [_row("A", "bo3", "1", [0.49, 0.50, 0.45], {"2": 1})]
    a = sc.aggregate(sc.pool_gaps(v, d))
    assert a["n_worse"] == 1 and a["mean_gap_pp"] == pytest.approx(-1.0)
    assert a["share_draws_ge"] == pytest.approx(0.5)


def test_summarize_variants_share_of_large_and_per_set_and_markdown():
    data = _data()
    data["half"][1] = _row("B", "bo1", "2", [0.56, 0.52, 0.58, 0.54], {"4": 17}, 25.0)         # +1 pp: half of large's 2 pp
    summ = sc.summarize_variants(data, breakdown=["large", "half"])
    assert summ["variants"]["large"]["captured"] == pytest.approx(1.0)
    assert summ["variants"]["half"]["captured"] == pytest.approx(0.5)
    assert summ["variants"]["default"]["mean_gap_pp"] == 0 and summ["variants"]["default"]["time_ratio"] == pytest.approx(1.0)
    assert set(summ["per_set"]["large"]) == {"A", "B"} and summ["per_set"]["large"]["B"]["mean_gap_pp"] == pytest.approx(2.0)
    assert summ["checks"]["default_vs_saved_max_abs_logit_diff"] == 0
    md = sc.variants_markdown(summ, sc.ORDER)
    assert "| default |" in md and "| large |" in md and "## Per set: large" in md and "share of large gap" in md


def test_summarize_dir_roundtrip(tmp_path):
    for name, rows in _data().items():
        (tmp_path / f"var_{name}.json").write_text(json.dumps(dict(variant=name, spec=None, rows=rows)), encoding="utf-8")
    (tmp_path / "inapp.json").write_text(json.dumps(dict(runs=[
        dict(variant="default", set="A", fmt="bo3", wall=10.0), dict(variant="large", set="A", fmt="bo3", wall=50.0)])), encoding="utf-8")
    md = sc.summarize_dir(tmp_path)
    assert (tmp_path / "variants.json").exists() and (tmp_path / "variants.md").exists()
    j = json.loads((tmp_path / "variants.json").read_text(encoding="utf-8"))
    assert j["inapp"]["mean_by_variant"] == {"default": 10.0, "large": 50.0}
    assert "In-app timing" in md and "| large |" in md


def test_set_structure_uses_default_info_and_large_candidates():
    data = _data()
    data["default"][0]["info"] = dict(pool_cards=80, pool_distinct=60, n_configs=40, n_viable=30, n_main_colors=10, n_land_counts=2)
    st = sc.set_structure(data)
    assert st["A"]["n_configs"] == 40 and st["A"]["n_candidates_large"] == 50 and st["A"]["seconds_large"] == 30.0
    assert st["B"]["n_configs"] is None
    assert expit(0) == 0.5
