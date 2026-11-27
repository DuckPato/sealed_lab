"""scripts/plat_draft_check.py: the rank filter (arm P), the rank indicators (arm R) and the event sampler (arm N); synthetic frames only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("plat_draft_check", _SCRIPTS / "plat_draft_check.py")
pd_ = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pd_)


def _games():
    rows = [
        ("d1", "draft", "PremierDraft", "Platinum"),
        ("d2", "draft", "PremierDraft", "DIAMOND"),
        ("d3", "draft", "PickTwoDraft", "mythic"),
        ("d4", "draft", "PremierDraft", "gold"),
        ("d5", "draft", "PremierDraft", "Bronze"),
        ("d6", "draft", "PremierDraft", None),
        ("d7", "draft", "TradDraft", None),
        ("b1", "bo1", "Sealed", None),
        ("b2", "bo3", "TradSealed", None),
    ]
    return pl.DataFrame(dict(draft_id=[r[0] for r in rows], fmt=[r[1] for r in rows], event=[r[2] for r in rows],
                             rank=[r[3] for r in rows]), schema_overrides=dict(rank=pl.String))


def test_rank_filter_keeps_platinum_up_case_insensitive():
    out = pd_.rank_filter(_games())
    assert out.filter(pl.col("fmt") == "draft")["draft_id"].to_list() == ["d1", "d2", "d3"]


def test_rank_filter_drops_null_rank_and_traddraft_but_not_sealed():
    out = pd_.rank_filter(_games())
    d = set(out["draft_id"].to_list())
    assert {"d6", "d7"}.isdisjoint(d)                    # null rank, TradDraft (no rank)
    assert {"b1", "b2"} <= d                             # other formats untouched


def test_rank_filter_keep_all_is_identity():
    g = _games()
    out = pd_.rank_filter(g, keep=pd_.RANKS, keep_null=True)
    assert out.equals(g)


def test_rank_indicators_bronze_zero_unranked_own_column():
    r = ["Bronze", "silver", "GOLD", "Platinum", "diamond", "mythic", None, "TradDraft", ""]
    m = pd_.rank_indicators6(r)
    assert m.shape == (9, 6)
    assert pd_.RANK_COLS == ["silver", "gold", "platinum", "diamond", "mythic", "unranked"]
    assert m[0].tolist() == [0, 0, 0, 0, 0, 0]           # bronze = reference
    for i, c in enumerate(["silver", "gold", "platinum", "diamond", "mythic"], start=1):
        row = np.zeros(6)
        row[pd_.RANK_COLS.index(c)] = 1
        assert m[i].tolist() == row.tolist()
    for i in (6, 7, 8):                                   # null / other value: only the unranked column
        assert m[i].tolist() == [0, 0, 0, 0, 0, 1]
    assert (m.sum(axis=1) <= 1).all()
    assert pd_.rank_indicators6([]).shape == (0, 6)


def test_make_rank_stage_appends_columns_and_sigma():
    from sealed_lab.model import design, glm
    from dataclasses import dataclass

    n = 4
    prob = glm.Problem(y=np.array([1.0, 0, 1, 0]), C=sp.csr_matrix(np.ones((n, 2))), D=sp.csr_matrix(np.eye(n)), idx=np.arange(n, dtype=np.int32))
    enc = design.CovEncoder(events=["e"], ref_event="e", skill_bins=[], use_g2=False, t0=None, umax=0.0, use_time=False,
                            names=["a", "b"], sigma=np.array([10.0, 10.0]))
    st = design.StageData(name="draft", games=pl.DataFrame(dict(rank=["gold", None, "Bronze", "mythic"])), enc=enc, prob=prob,
                          group=np.arange(n, dtype=np.int32), group_ids=list("abcd"), group_start=np.zeros(n, dtype="datetime64[us]"),
                          deck_rows=np.arange(n), exposure=np.ones(n), ref_row=np.zeros(2), n_games=n, n_groups=n)
    st2, k0 = pd_.make_rank_stage(st)
    assert k0 == 2 and st2.prob.k == 8 and st2.prob.D is prob.D
    assert st2.enc.sigma.tolist() == [10.0, 10.0] + [1.0] * 6 and len(st2.enc.names) == 8
    assert st.prob.k == 2 and len(st.enc.sigma) == 2     # the original stage is untouched
    C = st2.prob.C.toarray()
    assert C[:, 2:].tolist() == [[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0] * 6, [0, 0, 0, 0, 1, 0]]
    z, _ = pd_.make_rank_stage(st, zero=True)
    assert not z.prob.C.toarray()[:, 2:].any()


def _counts(n_events=400, seed=0):
    rng = np.random.default_rng(seed)
    return {f"e{i:04d}": int(rng.integers(3, 14)) for i in range(n_events)}


def test_sampler_whole_events_count_within_2pct():
    cnt = _counts()
    target = 1500
    ids = pd_.sample_events(cnt, target, seed=7)
    tot = sum(cnt[i] for i in ids)
    assert abs(tot - target) <= 0.02 * target
    assert len(set(ids)) == len(ids) and set(ids) <= set(cnt)     # each event once, whole (counted by its full game count)


def test_sampler_same_seed_same_result_different_seed_differs():
    cnt = _counts()
    a = pd_.sample_events(cnt, 1500, seed=11)
    b = pd_.sample_events(cnt, 1500, seed=11)
    c = pd_.sample_events(cnt, 1500, seed=12)
    assert a == b
    assert set(a) != set(c)


def test_sampler_unreachable_target_raises():
    with pytest.raises(ValueError):
        pd_.sample_events({"a": 5, "b": 5}, 1000, seed=1)
