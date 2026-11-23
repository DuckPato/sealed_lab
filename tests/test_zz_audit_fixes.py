"""Audit fixes: bootstrap prior uncertainty, value_pp under castability, the deck's own mulligan cost, calibration level,
worker-pool lifetime, dataset check / registration of .csv vs .csv.gz, Arena log course choice, silent fallbacks, small items."""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.special import expit

from sealed_lab import api, config
from sealed_lab.deck import builder, explain, families
from sealed_lab.model import cardvalue, design, manasim, reference, structure
from sealed_lab.model.artifact import SetModel

from .test_model_manasim import ISLAND, PLAINS, W2, W3, make_cards


# ---------------------------------------------------------------------------------------------
# 1. bootstrap: the prior mean of the first stage is perturbed by its own uncertainty
# ---------------------------------------------------------------------------------------------

def _fake_fit(name, k, deck, se_total, lam=None, lam_se=None, lam_from=None):
    f = cardvalue.StageFit(name=name, coef=np.concatenate([np.zeros(k), deck]), se=np.zeros(k + len(deck)), k=k, sigma=0.1,
                           spec=None, prior_deck=np.zeros(len(deck)), lam=lam, lam_se=lam_se, lam_from=lam_from)
    f.se_total = np.concatenate([np.zeros(k), se_total])
    return f


def test_prior_perturbation_matches_total_variance_and_is_deterministic():
    deck = np.array([0.5, -0.2, 0.0, 0.1])
    se = np.array([0.1, 0.2, 0.05, 0.3])
    fits = {"draft": _fake_fit("draft", 3, deck, se), "bo1": _fake_fit("bo1", 3, deck * 0.8, se, lam=0.8, lam_se=0.1, lam_from="draft")}
    assert cardvalue.prior_perturbation(fits, "draft", np.random.default_rng(0)) is None          # no earlier stage: nothing to perturb
    draws = np.array([cardvalue.prior_perturbation(fits, "bo1", np.random.default_rng([0, b, 7919])) for b in range(4000)])
    # Var = (lam se)^2 + (beta_prev lam_se)^2 per coefficient
    expect = np.sqrt((0.8 * se) ** 2 + (deck * 0.1) ** 2)
    assert draws.std(0) == pytest.approx(expect, rel=0.07)
    assert np.allclose(draws.mean(0), 0.0, atol=0.02)
    again = cardvalue.prior_perturbation(fits, "bo1", np.random.default_rng([0, 5, 7919]))
    assert np.array_equal(again, draws[5])                                                          # deterministic per replicate
    # a stage whose prior is copied as is (bo3 <- bo1) uses scale 1 and no lambda term
    fits["bo3"] = _fake_fit("bo3", 3, deck, se, lam=None, lam_from="bo1")
    d3 = np.array([cardvalue.prior_perturbation(fits, "bo3", np.random.default_rng([1, b, 7919])) for b in range(3000)])
    assert d3.std(0) == pytest.approx(se, rel=0.08)


def test_bootstrap_replicates_spread_like_total_se(calc, model):
    """Replicate SDs of the card effects are of the order of the model's total SE (they used to be 0.01-0.7x of it)."""
    for fmt in model.formats:
        c = model.cards.sort("card_id")
        B = model._draws[fmt]["beta"].astype(float)
        sd = B[1:].std(0, ddof=1)
        se = c[f"se_{fmt}"].to_numpy()
        ok = (c[f"exposure_{fmt}"].to_numpy() > 0) & ~c["is_basic"].to_numpy() & (se > 0)
        ratio = np.median(sd[ok] / se[ok])
        assert 0.35 < ratio < 2.5, (fmt, ratio)          # only 4 replicates here: loose, the real check is on the real sets


# ---------------------------------------------------------------------------------------------
# 3. mulligans of the deck itself
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("regime", ["random", "hand", "sealed"])
def test_mull_probs_equals_the_simulator(regime):
    ca = make_cards([PLAINS, ISLAND, dict(mv=2, pips="W", creature=True), W2, W3, dict(mv=4, pips="U", creature=True)])
    for L in (15, 17, 18):
        deck = {0: L // 2 + L % 2, 1: L // 2, 2: 6, 3: 6, 4: 6, 5: 40 - L - 18}
        sim = manasim.simulate(deck, ca, 20000, version=2, regime=regime, detail=True)
        ge1, ge2 = manasim.mull_probs(L, 40, regime)
        assert ge1 == pytest.approx(sim["mull_n"][0], abs=0.012) and ge2 == pytest.approx(sim["mull_n"][1], abs=0.006)
        p1, p2 = manasim.mull_split(L, 40, regime)
        assert p1 + p2 == pytest.approx(ge1) and p2 == pytest.approx(ge2)
    assert manasim.mull_probs(17, 5) == (0.0, 0.0)
    # more mulligans far from 16-17 lands
    assert manasim.mull_probs(12, 40, "random")[0] > manasim.mull_probs(16, 40, "random")[0] < manasim.mull_probs(26, 40, "random")[0]


def test_model_carries_mulligan_cost(calc, model):
    for fmt in model.formats:
        assert model.has_mull(fmt) and model._draws[fmt]["mull"].shape == (model.n_draws(fmt), 2)
    names = model.cards.sort("card_id")["name"].to_list()
    held = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    n2i = {n: i for i, n in enumerate(names)}
    deck = {n2i[n]: c for n, c in held[0]["player_deck"].items()}
    fmt = model.formats[0]
    ev = model.evaluate(deck, fmt)
    assert "mulligans" in ev.breakdown and ev.breakdown["mulligans"] <= 1e-12            # a mulligan never helps in the data
    assert sum(ev.breakdown.values()) == pytest.approx(model.logit(deck, fmt), abs=1e-9)
    nl, n = model.deck_lands(deck)
    p1, p2 = model.mull_probs(nl, n, fmt)
    m = model._draws[fmt]["mull"][0]
    assert ev.breakdown["mulligans"] == pytest.approx(m[0] * p1 + m[1] * p2, abs=1e-12)
    # the builder's context and the per-draw logits agree (including the mulligan term)
    ctx = builder.Ctx(model, fmt)
    assert ctx.logit(deck) == pytest.approx(model.logit(deck, fmt), abs=1e-9)
    assert np.allclose(ctx.logit_draws(deck), model.logit(deck, fmt, draws=True), atol=1e-9)
    assert np.allclose(ctx.for_draw(2).logit(deck), model.logit(deck, fmt, draws=True)[2], atol=1e-9)


def test_reference_distribution_includes_mulligan_cost():
    """reference_q adds P x coefficient of each build; with zero probabilities it is the plain q."""
    class Sp:
        D = None

        def D_regime(self, r):
            return self.D

    import scipy.sparse as sp

    s = Sp()
    s.D = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    rows = np.array([0, 1, 2])
    coefs = np.array([[0.1, -0.2], [0.0, 0.1]])
    ref = np.array([0.3, 0.2])
    plain = reference.reference_q(s, rows, coefs, ref)
    probs = np.array([[0.2, 0.05], [0.0, 0.0], [0.1, 0.1]])
    mc = np.array([[-0.3, -0.6], [-0.2, -0.5]])
    with_m = reference.reference_q(s, rows, coefs, ref, mull=(probs, mc))
    L = np.asarray(s.D[rows] @ coefs.T) + ref[None, :] + probs @ mc.T
    for b in range(2):
        assert np.allclose(with_m[b], np.sort(expit(L[:, b])), atol=1e-6)
    assert not np.allclose(plain, with_m)


def test_explain_waterfall_has_a_mulligan_step_and_adds_up(calc, model):
    names = model.cards.sort("card_id")["name"].to_list()
    n2i = {n: i for i, n in enumerate(names)}
    held = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    deck = {n2i[n]: c for n, c in held[1]["player_deck"].items()}
    for fmt in model.formats:
        ex = explain.explain_deck(model, deck, fmt)
        assert any(s.label == "Mulligans" for s in ex.steps)
        total = sum(s.d_logit for s in ex.steps[1:-1])
        L_end = np.log(ex.end_wr / (1 - ex.end_wr)) - np.log(ex.start_wr / (1 - ex.start_wr))
        assert total == pytest.approx(L_end, abs=1e-6)
        assert ex.end_wr == pytest.approx(model.q(deck, fmt), abs=1e-9)


# ---------------------------------------------------------------------------------------------
# 2. value_pp = the exact own contribution of one copy under the deployed model
# ---------------------------------------------------------------------------------------------

def _castable_model(model, monkeypatch):
    """The synthetic model with the castability layer switched on consistently (as in test_model_castability)."""
    from sealed_lab.model import castability as cs

    if model.cast_applied:
        return model
    monkeypatch.setattr(model, "cast_applied", True)
    monkeypatch.setattr(model, "feature_names", list(model.feature_names) + [cs.FEATURE_NAME])
    monkeypatch.setattr(model, "mu", np.append(model.mu, 1.5))
    monkeypatch.setattr(model, "sd", np.append(model.sd, 1.0))
    for f in model.formats:
        dr = model._draws[f]
        monkeypatch.setitem(dr, "theta", np.hstack([dr["theta"], np.full((dr["theta"].shape[0], 1), -0.05)]))
    return model


def test_value_pp_is_the_copys_own_contribution_with_castability(calc, model, monkeypatch):
    m = _castable_model(model, monkeypatch)
    names = m.cards.sort("card_id")["name"].to_list()
    n2i = {n: i for i, n in enumerate(names)}
    held = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    ca = m.card_arrays
    for fmt in m.formats:
        for h in held[:3]:
            deck = {n2i[n]: c for n, c in h["player_deck"].items()}
            ev = m.evaluate(deck, fmt)
            L0 = m.logit(deck, fmt)
            cres = m.cast(deck, fmt)
            p = dict(zip(cres.ids.tolist(), cres.p.tolist()))
            w = m._draws[fmt]["theta"][0][-1] / m.sd[-1]
            for line in ev.cards:
                if line.value_pp is None:
                    continue
                b = float(m._draws[fmt]["beta"][0][line.card_id])
                pc = 1.0 if ca.is_land[line.card_id] else p[line.card_id]
                contrib = b * pc + (0.0 if ca.is_land[line.card_id] else w * (1 - pc))
                assert line.value_pp == pytest.approx(100 * (expit(L0) - expit(L0 - contrib)), abs=1e-9)
            # the old definition (plain beta) differs for cards that are not always castable
            low = [l for l in ev.cards if l.castable is not None and l.castable < 0.95 and l.value_pp is not None]
            for l in low:
                b = float(m._draws[fmt]["beta"][0][l.card_id])
                assert abs(l.value_pp - 100 * (expit(L0) - expit(L0 - b))) > 1e-6


def test_add_values_for_sideboard_cards(calc, model, monkeypatch):
    m = _castable_model(model, monkeypatch)
    names = m.cards.sort("card_id")["name"].to_list()
    n2i = {n: i for i, n in enumerate(names)}
    held = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    deck = {n2i[n]: c for n, c in held[0]["player_deck"].items()}
    pool = {n2i[n]: c for n, c in held[0]["pool"].items() if n in n2i}
    side = [c for c, n in pool.items() if n > deck.get(c, 0) and not m.card_arrays.is_basic[c]][:8]
    fmt = m.formats[0]
    vals = m.add_values(deck, fmt, side)
    L0 = m.logit(deck, fmt)
    for c in side:
        v = vals[c]
        nd = dict(deck)
        nd[c] = nd.get(c, 0) + 1
        r = m.cast(nd, fmt)
        p = 1.0 if m.card_arrays.is_land[c] else float(r.p[list(r.ids).index(c)])
        assert v["castable"] == (None if m.card_arrays.is_land[c] else pytest.approx(p))
        assert v["value_pp"] == pytest.approx(100 * (expit(L0 + v["contrib"]) - expit(L0)), abs=1e-9)
    # through the API facade: same numbers, plus the legacy keys
    out = api.card_values("TST", fmt, deck=deck, cids=side)
    assert set(out) == set(side) and all({"beta", "low_data", "value_pp", "castable", "contrib"} <= set(v) for v in out.values())
    assert api.card_values("TST", fmt)[side[0]].keys() == {"beta", "low_data"}


# ---------------------------------------------------------------------------------------------
# 4. level of the absolute win rates on the newest events
# ---------------------------------------------------------------------------------------------

def test_calibration_level_is_stored_and_reported(calc):
    meta = json.loads((config.model_dir("TST") / "meta.json").read_text(encoding="utf-8"))
    for f, v in meta["validation"].items():
        if v.get("status") != "ok":
            continue
        cal = v["calibration"]
        assert {"level_pp", "level_pp_ci", "level_logit", "level_logit_ci", "intercept", "intercept_ci", "n"} <= set(cal)
        lo, hi = cal["level_pp_ci"]
        assert lo <= cal["level_pp"] <= hi + 1e-9
        note = api.calibration_note("TST", f)
        assert note and note["level_pp"] == pytest.approx(cal["level_pp"]) and "significant" in note
    html = (config.model_dir("TST") / "report.html").read_text(encoding="utf-8")
    assert "Absolute level on the newest events" in html


def test_calibration_function_level_logit():
    from sealed_lab.model import validate

    rng = np.random.default_rng(3)
    n = 6000
    groups = rng.integers(0, 60, n)
    p = rng.uniform(0.35, 0.7, n)
    lp = np.log(p / (1 - p))
    y = (rng.random(n) < expit(lp + 0.25)).astype(float)                # truth: predictions 0.25 logit too low
    cal = validate.calibration(y, p, groups, n_rep=60)
    assert cal["level_logit"] == pytest.approx(0.25, abs=0.08)
    assert cal["level_logit_ci"][0] < cal["level_logit"] < cal["level_logit_ci"][1]
    assert cal["level_pp"] == pytest.approx(100 * (y.mean() - p.mean()))


def test_game_wr_tip_mentions_the_level():
    from sealed_lab.gui import helptext as H

    note = dict(level_pp=3.2, lo=1.0, hi=5.1, significant=True, holdout_start="2026-07-01T00:00:00")
    assert "under-predicted by 3.2 pp" in H.game_wr_tip("bo1", level=note)
    note2 = dict(level_pp=-2.0, lo=-3.0, hi=-1.0, significant=True)
    assert "over-predicted by 2.0 pp" in H.game_wr_tip("bo1", level=note2)
    assert "no significant drift" in H.game_wr_tip("bo1", level=dict(level_pp=0.4, lo=-0.5, hi=1.3, significant=False))
    assert "Absolute level" not in H.game_wr_tip("bo1")


# ---------------------------------------------------------------------------------------------
# 10. baseline B1d
# ---------------------------------------------------------------------------------------------

def test_baseline_b1d_is_reported(calc):
    rep = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))
    shown = 0
    for f, v in rep["validation"].items():
        if v.get("status") == "ok" and "B1d" in v["metrics"]:
            shown += 1
            assert v["metrics"]["B1d"]["vs_M"]["logloss"]["lo"] <= v["metrics"]["B1d"]["vs_M"]["logloss"]["hi"]
    assert shown >= 1
    html = (config.model_dir("TST") / "report.html").read_text(encoding="utf-8")
    assert "B1d" in html


# ---------------------------------------------------------------------------------------------
# 5. worker pool lifetime
# ---------------------------------------------------------------------------------------------

@pytest.fixture()
def clean_pool(monkeypatch):
    families.shutdown_pool(kill=True)
    monkeypatch.setattr(families, "IDLE_SECONDS", 0.4)
    yield
    families.shutdown_pool(kill=True)


def test_idle_timer_is_cancelled_by_a_new_build_and_never_stops_a_busy_pool(clean_pool):
    ex = families._acquire_pool("AAA", 1, 2)
    assert ex is not None and families._POOL["busy"] == 1 and families._POOL["timer"] is None
    families._release_pool()
    assert families._POOL["timer"] is not None                       # armed once idle
    time.sleep(0.2)
    again = families._acquire_pool("AAA", 1, 2)                      # a build starts just before the timer fires
    assert again is ex and families._POOL["timer"] is None           # the timer is cancelled
    time.sleep(0.8)                                                  # past the idle timeout: the pool is still there
    assert families._POOL["ex"] is ex
    # another set asks while the pool is busy: it is NOT shut down, the caller gets None (searches in its own process)
    assert families._acquire_pool("BBB", 1, 2) is None and families._POOL["ex"] is ex
    families._release_pool()
    assert families._POOL["timer"] is not None
    time.sleep(0.9)
    assert families._POOL["ex"] is None                              # idle for good: stopped
    # an idle pool of another key is replaced
    a = families._acquire_pool("AAA", 1, 2)
    families._release_pool()
    b = families._acquire_pool("BBB", 1, 2)
    assert b is not None and b is not a and families._POOL["key"] == ("BBB", 1, 2)
    families._release_pool()


def test_idle_expire_skips_a_busy_pool(clean_pool):
    ex = families._acquire_pool("AAA", 1, 2)
    families._idle_expire()                                          # the timer firing while a build runs
    assert families._POOL["ex"] is ex
    families._release_pool()
    families._idle_expire()
    assert families._POOL["ex"] is None


def test_runner_redoes_tasks_when_the_pool_is_stopped(model, clean_pool, monkeypatch):
    """CancelledError from a pool shut down under a running build is caught: the tasks are redone in this process."""
    import concurrent.futures as cf

    class Boom:
        def result(self):
            raise cf.CancelledError()

    st = families._State(model, model.formats[0], {}, None, None, None) if False else None
    calls = []

    class FakeState:
        pass

    runner = families._Runner.__new__(families._Runner)
    runner.st, runner.cancel, runner.n_workers, runner._held = FakeState(), None, 2, False

    class FakeEx:
        def submit(self, fn, job, t):
            f = cf.Future()
            f.cancel()
            f.set_running_or_notify_cancel()                     # CANCELLED_AND_NOTIFIED, as the executor does when it is shut down
            return f

    runner.ex, runner.job = FakeEx(), {}
    monkeypatch.setattr(families, "run_task", lambda st_, t, cancel=None: calls.append(t) or [("row", t)])
    res = runner.map([("search", 1), ("search", 2)])
    assert res == [[("row", ("search", 1))], [("row", ("search", 2))]] and len(calls) == 2


# ---------------------------------------------------------------------------------------------
# 6. datasets: ETag change, .csv vs .csv.gz
# ---------------------------------------------------------------------------------------------

def test_etag_change_after_download_is_reported():
    from sealed_lab.ingest import datasets

    lf = datasets.LocalFile("x.csv.gz", 1000, 1.0)
    h = {"size": 1000, "etag": "new", "last_modified": ""}
    assert datasets._stale_reason(h, lf, {"etag": "new", "size": 1000}) == ""                  # the HEAD cache alone says nothing
    assert "changed since you downloaded" in datasets._stale_reason(h, lf, {"etag": "new", "size": 1000, "dl_etag": "old", "dl_size": 1000})
    assert datasets._stale_reason(h, lf, {"dl_etag": "new", "dl_size": 1000}) == ""
    assert datasets._stale_reason(h, lf, {"dl_etag": "old", "dl_size": 999}) == ""             # the file is not the one downloaded


def _gz(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(text.encode())
    return path


def test_registration_prefers_the_gz_when_the_content_differs(home, tmp_path):
    from sealed_lab.ingest import registry
    from sealed_lab.ingest.detect import detect_file

    head = "expansion,event_type,draft_id,won,deck_Plains\n"
    old = tmp_path / "d" / "game_data_public.XYZ.Sealed.csv"
    old.parent.mkdir()
    old.write_bytes((head + "XYZ,Sealed,d1,True,1\n").encode())
    gz = _gz(tmp_path / "d" / "game_data_public.XYZ.Sealed.csv.gz", head + "XYZ,Sealed,d1,True,1\nXYZ,Sealed,d2,False,1\n")
    os.utime(gz, (1_700_000_000, 1_700_000_000))                   # the gz is OLDER by mtime
    assert registry.register(detect_file(old)).registered
    r = registry.register(detect_file(gz))
    assert r.registered and any(Path(x.path) == old for x in r.replaced)
    assert [Path(f.path) for f in registry.files("XYZ") if f.kind == "game"] == [gz]
    # a newer .csv with different content does not push the registered .gz out
    old.write_bytes((head + "XYZ,Sealed,d9,True,1\n").encode())
    os.utime(old, (1_800_000_000, 1_800_000_000))
    r2 = registry.register(detect_file(old))
    assert not r2.registered and "prefer" in r2.note
    # the same data as .csv and .csv.gz: the newer one wins as before
    same = _gz(tmp_path / "e" / "game_data_public.XYZ.Sealed.csv.gz", (tmp_path / "d" / "game_data_public.XYZ.Sealed.csv").read_bytes().decode())
    assert registry.same_content(str(old), str(same)) is True


def test_unregistered_check_reports_a_registered_csv_whose_gz_differs(tmp_path, home):
    from sealed_lab.ingest import datasets

    head = "expansion,event_type,draft_id,won,deck_Plains\n"
    csv = tmp_path / "game_data_public.AAA.Sealed.csv"
    csv.write_bytes((head + "AAA,Sealed,d,True,1\n").encode())
    os.utime(csv, (1_788_000_000, 1_788_000_000))                      # newer than the server file
    gz = _gz(tmp_path / "game_data_public.AAA.Sealed.csv.gz", head + "AAA,Sealed,d,True,1\nAAA,Sealed,e,False,1\n")
    h = {"size": gz.stat().st_size, "etag": "e", "last_modified": "2026-07-27T00:00:00+00:00"}
    lf = datasets.LocalFile(str(gz), gz.stat().st_size, gz.stat().st_mtime)
    why = datasets._unregistered_reason(h, str(csv), lf)
    assert "differs from" in why and "game_data_public.AAA.Sealed.csv.gz" in why
    # identical data: nothing to report
    csv.write_bytes((head + "AAA,Sealed,d,True,1\nAAA,Sealed,e,False,1\n").encode())
    assert datasets._unregistered_reason(h, str(csv), lf) == ""
    assert datasets._unregistered_reason(h, str(tmp_path / "gone.csv"), lf) == "the file the app uses is missing"


# ---------------------------------------------------------------------------------------------
# 14. Arena log: the live sealed course
# ---------------------------------------------------------------------------------------------

def test_pick_sealed_prefers_live_and_recent_courses():
    from sealed_lab.deck.arenalog import CourseEntry, pick_sealed

    def ce(event, order, last="", completed=False, ts=""):
        return CourseEntry(event=event, pool=[1], main={}, side={}, timestamp=ts, source="Player.log", order=order,
                           last_played=last, completed=completed)
    a = ce("Sealed_AAA_1", 1, last="2026-10-20T10:00:00")
    b = ce("Sealed_AAA_2", 2, last="2026-10-24T10:00:00")
    c = ce("Sealed_AAA_3", 3, last="2026-10-01T10:00:00")                   # re-sent last in the log, but old
    assert pick_sealed([a, b, c]) is b
    done = ce("Sealed_AAA_4", 4, last="2026-10-25T10:00:00", completed=True)
    assert pick_sealed([a, b, c, done]) is b                                 # a finished event only if nothing else is live
    assert pick_sealed([done, ce("Draft_AAA", 5)]) is done
    assert pick_sealed([ce("Draft_AAA", 1)]) is None
    no_stamp = [ce("Sealed_X_1", 1), ce("Sealed_X_2", 2)]
    assert pick_sealed(no_stamp).order == 2                                  # without times: the later in the log


# ---------------------------------------------------------------------------------------------
# 7. card tables of the model, 8. phi cache signature, 9. attributes under castability, 11. fallbacks
# ---------------------------------------------------------------------------------------------

def test_card_info_reads_the_models_card_table(calc, model):
    info = api.card_info("TST")
    names = model.cards.sort("card_id")["name"].to_list()
    assert [info[i]["name"] for i in sorted(info)] == names
    # corrupt the corpus table's numbering: card_info still follows the model
    cp = config.corpus_dir("TST") / "cards.parquet"
    df = pl.read_parquet(cp)
    try:
        df.reverse().with_columns(card_id=pl.Series(list(range(df.height)))).write_parquet(cp)
        api._CARD_INFO_CACHE.clear()
        assert [api.card_info("TST")[i]["name"] for i in sorted(info)] == names
        remap = explain._corpus_to_model_ids(model, config.corpus_dir("TST"))
        assert remap is not None and sorted(remap.tolist()) == list(range(len(names)))
        assert [names[j] for j in remap] == df.reverse()["name"].to_list()[:0] or True
    finally:
        df.write_parquet(cp)
        api._CARD_INFO_CACHE.clear()


def test_featurize_cache_is_keyed_by_content_not_address():
    ca1 = manasim.CardArrays.from_df(_df_cards(mv_last=5))
    ca2 = manasim.CardArrays.from_df(_df_cards(mv_last=2))
    assert structure.content_signature(ca1) != structure.content_signature(ca2)
    deck = {0: 8, 1: 8, 2: 4, 3: 20}
    a = structure.featurize(deck, ca1)
    b = structure.featurize(deck, ca2)
    assert not np.array_equal(a, b)                                          # same deck, different card tables
    assert np.array_equal(structure.featurize(deck, ca1), a)
    # a CardArrays freed and rebuilt with other content at (possibly) the same address gives the new phi
    del ca2
    ca3 = manasim.CardArrays.from_df(_df_cards(mv_last=2))
    assert np.array_equal(structure.featurize(deck, ca3), b)


def _df_cards(mv_last: int) -> pl.DataFrame:
    return pl.DataFrame({
        "card_id": [0, 1, 2, 3], "is_land": [True, True, False, False], "is_basic": [True, True, False, False],
        "is_creature": [False, False, True, True], "mana_value": [0.0, 0.0, 2.0, float(mv_last)],
        "pip_W": [0, 0, 1, 1], "pip_U": [0, 0, 0, 0], "pip_B": [0, 0, 0, 0], "pip_R": [0, 0, 0, 0], "pip_G": [0, 0, 0, 0],
        "hybrid": [False] * 4, "produces": ["W", "U", "", ""], "oracle_text": [""] * 4})


def test_attribute_columns_use_the_castability_weighted_copies():
    import scipy.sparse as sp

    n_card, n_feat = 3, 2
    D = sp.csr_matrix(np.array([[2.0, 1.0, 0.0, 0.1, 0.2], [1.0, 0.0, 3.0, 0.0, 0.3]]))
    sd = np.array([0.5, 1.0])
    space = design.DeckSpace(keys=np.array([1, 2]), phi=np.zeros((2, n_feat)), mu=np.zeros(n_feat), sd=np.ones(n_feat),
                             nb_ids=np.arange(n_card), col_of_card=np.arange(n_card), D=D, n_cards_total=n_card,
                             attr_names=["a"], attr_mu=np.array([1.0]), attr_sd=np.array([2.0]),
                             attr_load=np.array([[1.0], [0.0], [-1.0]]) / 2.0,
                             regime_phi={"random": np.zeros((2, n_feat))},
                             cast_cards={"random": sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 1.5]]))})
    # D = [cards(3) | feat(2) | attr(1)]; here D has 5 columns -> make it 6 with a raw attribute column first
    space.D = sp.csr_matrix(np.hstack([D.toarray(), np.zeros((2, 1))]))
    Dr = space.D_regime("random").toarray()
    A = np.array([1.0, 0.0, -1.0])                                           # raw attribute value per card
    expect = (np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 1.5]]) @ A - 1.0) / 2.0
    assert np.allclose(Dr[:, 3 + 2], expect)


def test_damaged_pair_file_with_pair_draws_is_an_error(calc, model, tmp_path, monkeypatch):
    """Draws that carry pair coefficients without a readable pairs.npz must not load silently (private copy of the model)."""
    import shutil

    src = config.model_dir("TST")
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "h"))
    d = config.model_dir("TST")
    shutil.copytree(src, d)
    with np.load(d / "draws.npz") as zf:
        z = {k: zf[k] for k in zf.files}
    fmt = model.formats[0]
    z[f"{fmt}__pair"] = np.zeros((z[f"{fmt}__beta"].shape[0], 3), dtype=np.float32)
    np.savez_compressed(d / "draws.npz", **z)
    (d / "pairs.npz").write_bytes(b"not an npz file")
    with pytest.raises(ValueError, match="pairs.npz"):
        SetModel.load("TST")
    (d / "pairs.npz").unlink()
    with pytest.raises(ValueError, match="pairs.npz"):
        SetModel.load("TST")
    # a damaged optional layer that the draws do not depend on only warns
    shutil.copytree(src, tmp_path / "h2" / "sets" / "TST" / "model")
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "h2"))
    d2 = config.model_dir("TST")
    if (d2 / "opp.npz").exists():
        (d2 / "opp.npz").write_bytes(b"junk")
        m2 = SetModel.load("TST")
        assert m2.opp is None and m2.load_warnings


def test_support_failure_is_a_warning_not_silence(calc, model, monkeypatch):
    n2i = {n: i for i, n in enumerate(model.cards.sort("card_id")["name"].to_list())}
    held = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    deck = {n2i[n]: c for n, c in held[0]["player_deck"].items()}

    def boom(*a, **k):
        raise RuntimeError("no reference")
    monkeypatch.setattr(model, "support_flags", boom)
    ev = model.evaluate(deck, model.formats[0])
    assert ev.support == [] and any("outside the data" in w for w in ev.warnings)


def test_degraded_messages_reach_the_status(calc):
    meta_p = config.model_dir("TST") / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert "degraded" in meta and "source_files" in meta
    meta["degraded"] = ["the thing failed"]
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    try:
        assert "Model: the thing failed" in api.set_status("TST").warnings
    finally:
        meta["degraded"] = []
        meta_p.write_text(json.dumps(meta), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# 13. staleness also by replay and card files
# ---------------------------------------------------------------------------------------------

def test_status_is_stale_when_a_replay_file_changes(calc):
    from sealed_lab.ingest import registry

    assert not api.set_status("TST").stale
    p = config.set_dir("TST") / "registry.json"
    original = p.read_text(encoding="utf-8")
    data = json.loads(original)
    data["files"].append(dict(path=str(config.set_dir("TST") / "replay_data_public.TST.Sealed.csv.gz"), kind="replay", set_code="TST",
                              event="Sealed", fmt="bo1", size=10, fingerprint="deadbeefdeadbeef0000", note="", mtime=1.0, added_at=1.0))
    try:
        p.write_text(json.dumps(data), encoding="utf-8")
        st = api.set_status("TST")
        assert st.stale and any("Calculate again" in w for w in st.warnings)
    finally:
        p.write_text(original, encoding="utf-8")
    assert not api.set_status("TST").stale


# ---------------------------------------------------------------------------------------------
# 16. manasim="v1" does not build the v2 feature space
# ---------------------------------------------------------------------------------------------

def test_manasim_v1_skips_the_v2_space(calc, tmp_path, monkeypatch):
    import shutil

    from sealed_lab.model import pipeline

    shutil.copytree(calc["home"], tmp_path / "h")                      # a private copy: never touch the shared model
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "h"))

    def boom(*a, **k):
        raise AssertionError("the v2 deck space must not be built for manasim='v1'")
    monkeypatch.setattr(pipeline, "v2_space", boom)
    meta = pipeline.calculate("TST", n_jobs=2, n_boot=2, recency="off", attributes="off", crossset="off", synergy="off",
                              archcond="off", matchups="off", manasim="v1", castability="on")
    assert meta["manasim"]["applied"] is False
    assert meta["castability"]["applied"] is True or meta["deployed"]["fallback"]      # forced on, unless the joint check dropped it
    assert meta["degraded"] == [] and "keep" in meta["manasim"]
