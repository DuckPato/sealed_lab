"""Deck builder and scorer on the synthetic set (scale 0.3 model shared with the model tests via `calc`)."""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from sealed_lab import api, cli
from sealed_lab.deck import arena, builder, score
from sealed_lab.deck.milp import MilpSpec, solve_milp
from sealed_lab.deck.names import NameIndex
from sealed_lab.synth.generate import truth_strength
from sealed_lab.types import Cancelled

ROOT = Path(__file__).resolve().parents[1]
BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")


# ------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def names(model):
    return model.cards.sort("card_id")["name"].to_list()


@pytest.fixture(scope="module")
def name2id(names):
    return {n: i for i, n in enumerate(names)}


@pytest.fixture(scope="module")
def held(calc):
    return json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))


def to_ids(by_name: dict[str, int], name2id) -> dict[int, int]:
    return {name2id[n]: c for n, c in by_name.items()}


def pool_ids(h: dict, name2id) -> dict[int, int]:
    return {name2id[n]: c for n, c in h["pool"].items() if n not in BASICS}


def deck_by_name(ev) -> dict[str, int]:
    return {c.name: c.count for c in ev.cards}


def text_of(by_name: dict[str, int], side: dict[str, int] | None = None) -> str:
    t = "Deck\n" + "".join(f"{n} {k}\n" for k, n in by_name.items())
    if side:
        t += "\nSideboard\n" + "".join(f"{n} {k}\n" for k, n in side.items())
    return t


_WORKER_MODEL = {}


def _build_task(task):
    """Top-level (picklable) worker: build one pool in a spawned process."""
    import time

    set_code, pool_by_name, n_decks, min_diff = task
    m = _WORKER_MODEL.get(set_code)
    if m is None:
        m = _WORKER_MODEL[set_code] = builder.load_model(set_code)
    nm = {n: i for i, n in enumerate(m.cards.sort("card_id")["name"].to_list())}
    t0 = time.time()
    res = builder.build(m, {nm[n]: c for n, c in pool_by_name.items() if n not in BASICS}, "bo3", n_decks=n_decks, min_diff=min_diff,
                        families=False)
    return res, time.time() - t0


@pytest.fixture(scope="module")
def builds(calc, held):
    """30 one-deck builds (quality) + 3 three-deck builds (diversity), run in 2 spawned worker processes."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    tasks = [("TST", h["pool"], 1, 10) for h in held[:30]] + [("TST", h["pool"], 3, 10) for h in held[:3]]
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as ex:
        out = list(ex.map(_build_task, tasks, chunksize=1))
    return dict(one=out[:30], three=out[30:])


@pytest.fixture(scope="module")
def quality(model, name2id, held, calc, builds):
    """Builder (n_decks=1) vs the synthetic player's deck on 30 held-out pools."""
    rows = []
    for h, (res, _) in zip(held[:30], builds["one"]):
        ev = res.decks[0]
        d = deck_by_name(ev)
        human = to_ids(h["player_deck"], name2id)
        rows.append(dict(h=h, ev=ev, ts_builder=truth_strength(calc["truth"], d),
                         ts_human=truth_strength(calc["truth"], h["player_deck"]), q_builder=ev.game_wr,
                         q_human=float(model.q(human, "bo3"))))
    return rows


@pytest.fixture(scope="module")
def multi(held, builds):
    return [(h, res, t) for h, (res, t) in zip(held[:3], builds["three"])]


# ------------------------------------------------------------------------------------------ exactness

def test_fast_logit_equals_model_logit(model, name2id, held):
    for fmt in ("bo3", "bo1"):
        ctx = builder.Ctx(model, fmt)
        for h in held[:8]:
            d = to_ids(h["player_deck"], name2id)
            assert ctx.logit(d) == pytest.approx(model.logit(d, fmt), abs=1e-9)
            d2 = dict(d)
            k = next(i for i in d2 if not ctx.is_land[i])
            d2[k] += 1
            d2[name2id["Plains"]] = d2.get(name2id["Plains"], 0) - 1 if d2.get(name2id["Plains"], 0) > 1 else d2.get(name2id["Plains"], 0)
            d2 = {i: n for i, n in d2.items() if n > 0}
            assert ctx.logit(d2) == pytest.approx(model.logit(d2, fmt), abs=1e-9)


# ------------------------------------------------------------------------------------------ MILP

def _brute(spec: MilpSpec) -> list[float]:
    """All feasible objectives (sorted desc) by exhaustive enumeration over distinct-card selections."""
    n = len(spec.coef)
    objs: dict[tuple, float] = {}
    for counts in itertools.product(*[range(int(c) + 1) for c in spec.copies]):
        counts = np.array(counts)
        if counts.sum() + spec.n_lands - counts[spec.is_land].sum() != spec.deck_size:
            continue
        nb = spec.n_lands - counts[spec.is_land].sum()
        if nb < 0 or counts[spec.splash & ~spec.is_land].sum() > spec.splash_cap:
            continue
        ok = False
        cols = spec.basic_colors
        for split in itertools.product(range(nb + 1), repeat=len(cols)):
            if sum(split) != nb:
                continue
            src = (counts[spec.is_land, None] * spec.produces[spec.is_land]).sum(0).astype(int)
            for c, s in zip(cols, split):
                src[c] += s
            if all(src[c] >= f for c, f in spec.floors.items()):
                ok = True
                break
        if ok:
            objs[tuple(counts)] = float(counts @ spec.coef)
    return sorted(objs.values(), reverse=True)


def test_milp_matches_bruteforce_synthetic_arrays():
    rng = np.random.default_rng(3)
    n = 10
    is_land = np.zeros(n, dtype=bool)
    is_land[[8, 9]] = True
    prod = np.zeros((n, 5), dtype=bool)
    prod[8, [0, 1]] = True
    prod[9, 1] = True
    splash = np.zeros(n, dtype=bool)
    splash[[4, 5]] = True
    for trial in range(6):
        spec = MilpSpec(coef=rng.normal(0, 1, n), copies=rng.integers(1, 3, n), is_land=is_land, produces=prod,
                        splash=splash, basic_colors=[0, 1, 2], floors={0: 2, 1: 2, 2: 1}, n_lands=4, deck_size=11,
                        splash_cap=1)
        bf = _brute(spec)
        sols = solve_milp(spec, k=3)
        assert len(sols) == 3
        assert [s.obj for s in sols] == pytest.approx(bf[:3], abs=1e-7)
        sels = {tuple(s.counts) for s in sols}
        assert len(sels) == 3                                    # no-good cuts give distinct decks
        for s in sols:                                           # reported solutions are feasible
            assert s.counts.sum() + sum(s.basics.values()) == 11
            assert s.counts[splash & ~is_land].sum() <= 1


def test_milp_distance_constraint():
    rng = np.random.default_rng(5)
    n = 12
    spec = MilpSpec(coef=rng.normal(0, 1, n), copies=np.ones(n, dtype=int), is_land=np.zeros(n, dtype=bool),
                    produces=np.zeros((n, 5), dtype=bool), splash=np.zeros(n, dtype=bool), basic_colors=[0],
                    floors={}, n_lands=3, deck_size=9)
    best = solve_milp(spec, k=1)[0]
    alt = solve_milp(spec, k=1, apart=[(best.counts, 4)])[0]
    assert (alt.counts > best.counts).sum() >= 4 and alt.obj < best.obj
    # and it is the best such deck: brute force over 6-subsets of 12
    bf = max(float(spec.coef[list(c)].sum()) for c in itertools.combinations(range(n), 6)
             if len(set(c) - set(np.nonzero(best.counts)[0])) >= 4)
    assert alt.obj == pytest.approx(bf, abs=1e-7)


def test_milp_with_model_betas_matches_bruteforce(model, name2id, names):
    """Structure zeroed (linear=False -> beta only): MILP optimum == exhaustive optimum on a tiny real pool."""
    ctx = builder.Ctx(model, "bo3")
    ca = ctx.ca
    # tiny WU pool: 26 spells (one copy each) + the first W/U dual land if any
    spells = [i for i in range(ctx.n_cards) if not ca.is_land[i] and ca.pips[i, [2, 3, 4]].sum() == 0
              and ca.pips[i, [0, 1]].sum() > 0 and not ca.hybrid[i]]
    spells = spells[:26]
    duals = [i for i in range(ctx.n_cards) if ca.is_land[i] and not ca.is_basic[i] and ca.produces[i, 0] and ca.produces[i, 1]]
    pool = {i: 1 for i in spells + duals[:1]}
    cfg = next(c for c in builder.enumerate_configs(ctx, pool) if c.main == "WU" and not c.splash)
    for nl in (16, 17):
        spec, ids = builder.make_spec(ctx, cfg, pool, nl, linear=False)
        assert not spec.splash.any()
        got = solve_milp(spec, k=1)[0]
        bf = _brute_real(spec, nl)
        assert got.obj == pytest.approx(bf, abs=1e-7)


def _brute_real(spec: MilpSpec, nl: int) -> float:
    spell_idx = np.nonzero(~spec.is_land)[0]
    land_idx = np.nonzero(spec.is_land)[0]
    best = -1e18
    need = spec.deck_size - nl
    for k_l in range(len(land_idx) + 1):
        for lands in itertools.combinations(land_idx, k_l):
            ns = need
            if ns > len(spell_idx):
                continue
            nb = nl - k_l
            if nb < 0:
                continue
            src = np.zeros(5, dtype=int)
            for l in lands:
                src += spec.produces[l].astype(int)
            if not any(all(src[c] + (s if c == spec.basic_colors[0] else nb - s) >= f for c, f in spec.floors.items())
                       for s in range(nb + 1)):
                continue
            top = np.sort(spec.coef[spell_idx])[::-1][:ns].sum()     # best ns spells (one copy each)
            best = max(best, float(top + spec.coef[list(lands)].sum()))
    return best


# ------------------------------------------------------------------------------------------ builder: legality

def _assert_legal(model, ev, pool: dict[int, int], name2id):
    ctx = builder.Ctx(model, "bo3")
    deck = {c.card_id: c.count for c in ev.cards}
    assert sum(deck.values()) == 40
    nl = sum(n for i, n in deck.items() if ctx.is_land[i])
    assert 15 <= nl <= 18, nl
    for i, n in deck.items():
        if not ctx.is_basic[i]:
            assert n <= pool.get(i, 0), (model.card_name(i), n)
    # the deck must be explainable by one legal configuration (castable colors, splash <= 3, source floors)
    ok = False
    for cfg in builder.enumerate_configs(ctx, pool):
        if any((not ctx.is_land[i] and not ctx.is_basic[i] and cfg.cls[i] == 0) or (ctx.is_land[i] and not ctx.is_basic[i] and not cfg.land_ok[i])
               for i in deck):
            continue
        if sum(n for i, n in deck.items() if cfg.cls[i] == 2) > 3:
            continue
        src = np.zeros(5)
        for i, n in deck.items():
            if ctx.is_land[i]:
                src += n * ctx.ca.produces[i]
        if all(src[c] >= f for c, f in cfg.floors.items()):
            ok = True
            break
    assert ok, "deck is not legal in any color configuration"
    # every colored spell has at least 3 sources of each of its colors (castable)
    src = sum(n * ctx.ca.produces[i] for i, n in deck.items() if ctx.is_land[i])
    for i, n in deck.items():
        if not ctx.is_land[i]:
            for col in np.nonzero(ctx.ca.pips[i] > 0)[0]:
                if not (ctx.ca.hybrid[i] and (src[np.nonzero(ctx.ca.pips[i] > 0)[0]] >= 3).any()):
                    assert src[col] >= 3, (model.card_name(i), col)


def test_builder_legality_on_30_pools(model, name2id, quality):
    for r in quality:
        _assert_legal(model, r["ev"], pool_ids(r["h"], name2id), name2id)


def test_builder_legality_multi(model, name2id, multi):
    for h, res, _ in multi:
        for ev in res.decks:
            _assert_legal(model, ev, pool_ids(h, name2id), name2id)


def test_classic_diverse_alternatives_and_probabilities(model, name2id, multi, names):
    for h, res, _ in multi:
        assert len(res.decks) == 3
        decks = [deck_by_name(e) for e in res.decks]

        def nonland_diff(a, b):
            na = {k: v for k, v in a.items() if k not in BASICS}
            nb = {k: v for k, v in b.items() if k not in BASICS}
            return max(sum(max(0, v - nb.get(k, 0)) for k, v in na.items()), sum(max(0, v - na.get(k, 0)) for k, v in nb.items()))

        lands = {i for i, nm in enumerate(names) if model.card_arrays.is_land[i]}
        land_names = {names[i] for i in lands}

        def nl_diff(a, b):
            return nonland_diff({k: v for k, v in a.items() if k not in land_names}, {k: v for k, v in b.items() if k not in land_names})

        assert res.diffs[0] == 0
        for i in range(1, 3):
            assert res.diffs[i] >= 10 and nl_diff(decks[0], decks[i]) == res.diffs[i]
        for a, b in itertools.combinations(range(3), 2):
            assert nl_diff(decks[a], decks[b]) >= 10
        q = [e.game_wr for e in res.decks]
        assert q[0] >= max(q[1:]) - 1e-12                       # best first
        assert sum(res.p_best) == pytest.approx(1.0) and all(0 <= p <= 1 for p in res.p_best)
        assert res.p_best[0] >= max(res.p_best[1:])
        for e in res.decks:
            assert e.game_wr_lo <= e.game_wr <= e.game_wr_hi and 0 <= e.score <= 100 and e.match_wr is not None
            assert all(c.value_pp is None for c in e.cards if c.name in BASICS)
            assert all(c.value_pp is not None for c in e.cards if c.name not in BASICS)


# ------------------------------------------------------------------------------------------ builder: quality

def test_builder_beats_synthetic_player_on_true_strength(quality, capsys):
    gain = np.array([r["ts_builder"] - r["ts_human"] for r in quality])
    wins = float((gain > 0).mean())
    dq = np.array([r["q_builder"] - r["q_human"] for r in quality])
    with capsys.disabled():
        print(f"\n[builder vs synthetic player, {len(gain)} held-out pools] true-strength win rate {wins:.0%}, "
              f"mean gain {gain.mean():+.3f} logit (median {np.median(gain):+.3f}, min {gain.min():+.3f}); "
              f"model-q gain mean {100 * dq.mean():+.1f}pp")
    assert len(gain) == 30
    assert wins >= 0.8
    assert gain.mean() > 0
    assert (dq > 0).mean() >= 0.9          # the optimiser beats the human in the model's own eyes


def test_builder_bo1_progress_and_unknown_format(model, name2id, held):
    log = []
    res = builder.build(model, pool_ids(held[1], name2id), "bo1", n_decks=1, progress=lambda f, m: log.append((f, m)))
    assert res.fmt == "bo1" and res.decks[0].match_wr is None
    fr = [f for f, _ in log]
    assert fr == sorted(fr) and fr[-1] == 1.0 and 0 <= fr[0] <= 0.05 and len(fr) > 10
    with pytest.raises(ValueError):
        builder.build(model, pool_ids(held[1], name2id), "draft", n_decks=1)


def test_build_time_for_90_card_pool(model, held, name2id, capsys):
    import time

    pool = pool_ids(held[7], name2id)
    builder.build(model, pool, "bo3", n_decks=1)                 # warm the simulator tables
    t0 = time.time()
    res = builder.build(model, pool, "bo3", n_decks=3, families=False)
    dt = time.time() - t0
    t1 = time.time()
    fam = builder.build(model, pool, "bo3", n_decks=5, n_jobs=1)
    dt_fam = time.time() - t1
    with capsys.disabled():
        print(f"\n[build time, {sum(pool.values())}-card pool, single process] classic 3 decks {dt:.1f}s; families "
              f"({model.n_draws('bo3') - 1} draws) {dt_fam:.1f}s")
    assert len(res.decks) == 3 and dt < 20 and fam.families and dt_fam < 120


def test_tiny_pool_and_empty_pool(model, name2id, held):
    h = held[0]
    spells = [n for n in h["pool"] if n not in BASICS][:10]
    pool = {name2id[n]: 1 for n in spells}
    res = builder.build(model, pool, "bo3", n_decks=2, families=False)
    assert sum(c.count for c in res.decks[0].cards) == 40
    assert any("lands" in w or "apart" in w for w in res.warnings)
    with pytest.raises(ValueError):
        builder.build(model, {}, "bo3")


def test_cancel(model, name2id, held):
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        builder.build(model, pool_ids(held[0], name2id), "bo3", cancel=ev)


# ------------------------------------------------------------------------------------------ names in pools

def test_pool_text_name_variants(calc, model, names):
    """Front faces, accents stripped, set suffixes, basics, junk -> resolved / reported, never a crash."""
    idx = NameIndex(model.cards)
    split = [n for n in names if " // " in n]
    accent = [n for n in names if any(ord(ch) > 127 for ch in n)]
    comma = [n for n in names if "," in n and " // " not in n]
    apos = [n for n in names if "'" in n]
    assert split and accent and comma and apos
    import unicodedata

    plain_target = accent[0]
    plain = "".join(c for c in unicodedata.normalize("NFKD", accent[0]) if not unicodedata.combining(c))
    lines = [f"1 {split[0].split(' // ')[0]} (TST) 12", f"1 {plain}", f"2 {comma[0]} (TST) 99", f"1 {apos[0].replace(chr(39), chr(0x2019))}",
             "3 Plains", "1 Definitely Not A Card"]
    text = "﻿" + "\r\n".join(lines) + "\r\n"
    # build from a pool that is mostly the held-out pool plus these cards
    held0 = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))[0]
    pool_text = text_of(held0["pool"]).replace("Deck\n", "Deck\n" + text.lstrip("﻿").replace("\r\n", "\n"), 1)
    res = builder.build_pool("TST", "﻿" + pool_text.replace("\n", "\r\n"), "bo3", n_decks=1)
    assert res.pool_unmatched == ["Definitely Not A Card"]
    parsed = arena.parse_arena_full(text)
    r = __import__("sealed_lab.deck.names", fromlist=["resolve_names"]).resolve_names(
        arena.pool_of(parsed.main, parsed.side), idx, drop_basics=True)
    expected = {split[0], plain_target, comma[0], apos[0]}
    assert r.unmatched == ["Definitely Not A Card"] and {names[i] for i in r.counts} == expected | set()


# ------------------------------------------------------------------------------------------ score

def test_score_monotone_with_truth(calc, model, held, name2id):
    ts, sc, wr = [], [], []
    for h in held:
        res = score.score_deck("TST", text_of(h["player_deck"]), "bo3")
        assert res.efficiency_pp is None and res.best_alternative is None and not res.unmatched
        ts.append(truth_strength(calc["truth"], h["player_deck"]))
        sc.append(res.deck.score)
        wr.append(res.deck.game_wr)
    assert spearmanr(sc, ts)[0] >= 0.8
    assert spearmanr(wr, ts)[0] >= 0.8


def test_score_efficiency_with_sideboard(model, held, name2id, capsys):
    effs = []
    for h in held[:2]:
        side = {n: c - h["player_deck"].get(n, 0) for n, c in h["pool"].items() if c - h["player_deck"].get(n, 0) > 0}
        res = score.score_deck("TST", text_of(h["player_deck"], side), "bo3")
        assert res.efficiency_pp is not None and res.efficiency_pp <= 1e-9
        if res.efficiency_pp < -1e-9:
            assert res.best_alternative is not None and res.best_alternative.game_wr > res.deck.game_wr
            assert any(w.startswith("Tip:") for w in res.warnings)
        effs.append(res.efficiency_pp)
    with capsys.disabled():
        print(f"\n[score efficiency vs best build from deck+sideboard] {np.round(effs, 2).tolist()} pp")


def test_score_short_deck_unmatched_and_errors(model, held):
    h = held[0]
    d = dict(list(h["player_deck"].items())[:12])
    res = score.score_deck("TST", text_of(d) + "1 Nonexistent Wizard\n", "bo3")
    assert res.unmatched == ["Nonexistent Wizard"]
    assert any("cards" in w for w in res.deck.warnings)          # not 40 cards -> still scored, with a warning
    assert 0 <= res.deck.score <= 100
    with pytest.raises(ValueError):
        score.score_deck("TST", "", "bo3")
    with pytest.raises(ValueError):
        score.score_deck("TST", "1 Nonexistent Wizard\n", "bo3")
    with pytest.raises(ValueError):
        score.score_deck("TST", text_of(h["player_deck"]), "draft")
    with pytest.raises(FileNotFoundError):
        score.score_deck("NOPE", text_of(h["player_deck"]), "bo3")


# ------------------------------------------------------------------------------------------ api + cli

def test_api_end_to_end(calc, held, name2id):
    h = held[4]
    res = api.build_pool("tst", text_of(h["pool"]), "bo3", n_decks=2, min_diff=10)
    assert 1 <= len(res.decks) <= 2 and res.set_code == "TST" and len(res.families) == len(res.decks)
    exp = api.export_arena(res.decks[0], None, "TST")
    assert exp.startswith("Deck\n")
    main, side = arena.parse_arena(exp)
    assert sum(main.values()) == 40 and not side
    assert all(" // " not in k for k in main)
    # exported deck scores identically when pasted back (front faces resolve to the same cards)
    back = api.score_deck("TST", exp, "bo3")
    assert back.deck.game_wr == pytest.approx(res.decks[0].game_wr, abs=1e-12)
    # export with a sideboard
    side_names = {n: c for n, c in h["pool"].items()}
    exp2 = api.export_arena(res.decks[0], side_names, "TST")
    assert "\n\nSideboard\n" in exp2
    m2, s2 = arena.parse_arena(exp2)
    assert m2 == main and sum(s2.values()) == sum(h["pool"].values())
    # the builder's own best deck with the rest of the pool as sideboard: efficiency ~ 0 (nothing better to find)
    best = deck_by_name(res.decks[0])
    side = {n: c - best.get(n, 0) for n, c in h["pool"].items() if c - best.get(n, 0) > 0}
    sc = api.score_deck("TST", text_of(best, side), "bo3")
    assert sc.efficiency_pp is not None and -1.0 < sc.efficiency_pp <= 1e-9


def test_cli_score(calc, held, tmp_path, capsys):
    h = held[5]
    pool = tmp_path / "pool.txt"
    pool.write_text(text_of(h["pool"]), encoding="utf-8")
    deck = tmp_path / "deck.txt"
    deck.write_text(text_of(h["player_deck"]), encoding="utf-8")
    assert cli.main(["score", "TST", str(deck)]) == 0
    assert "Deck 1:" in capsys.readouterr().out


def test_cli_subprocess_build(calc, held, tmp_path):
    pool = tmp_path / "pool.txt"
    pool.write_text(text_of(held[6]["pool"]), encoding="utf-8")
    env = dict(os.environ, SEALED_LAB_HOME=str(calc["home"]), PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "sealed_lab", "cli", "build", "TST", str(pool), "--n-decks", "1"],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Deck 1:" in r.stdout and "game WR" in r.stdout


# ------------------------------------------------------------------------------------------ locks / exclusions

def _spells_of(model, deck: dict[int, int]) -> list[int]:
    return [i for i in deck if not model.card_arrays.is_land[i]]


def test_milp_locked_lower_bounds():
    coef = np.array([1.0, 0.5, -3.0, -4.0, 0.0])
    spec = MilpSpec(coef=coef, copies=np.array([2, 2, 3, 3, 0]), is_land=np.zeros(5, bool), produces=np.zeros((5, 5), bool),
                    splash=np.zeros(5, bool), basic_colors=[], floors={}, n_lands=0, deck_size=8, splash_cap=3)
    base = solve_milp(spec, k=1)[0]
    assert base.counts[3] == 0 or base.counts[2] > 0           # the bad cards are taken only as fillers
    spec.locked = np.array([0, 0, 0, 2, 0])
    sol = solve_milp(spec, k=3)
    assert sol and all(s.counts[3] >= 2 for s in sol)           # every solution (also the no-good cut ones) keeps the lock
    assert sol[0].obj < base.obj


def test_build_honours_locks_and_exclusions(model, name2id, held):
    h = held[3]
    pool = pool_ids(h, name2id)
    base = builder.build(model, pool, "bo3", n_decks=1)
    deck0 = {c.card_id: c.count for c in base.decks[0].cards}
    beta = model.card_beta("bo3")
    sp = _spells_of(model, deck0)
    weakest = sorted(sp, key=lambda c: beta[c])[:2]
    strongest = sorted(sp, key=lambda c: -beta[c])[:2]
    res = builder.build(model, pool, "bo3", n_decks=3, locked={c: 1 for c in weakest}, excluded=set(strongest))
    assert res.decks and any("locked" in w for w in res.warnings)
    for ev in res.decks:
        d = {c.card_id: c.count for c in ev.cards}
        assert sum(d.values()) == 40
        assert all(d.get(c, 0) >= 1 for c in weakest)
        assert not any(c in d for c in strongest)
    assert res.decks[0].game_wr <= base.decks[0].game_wr + 1e-12       # constraints can only cost
    # diversity still measured on the deck lists
    for a, b in itertools.combinations(res.decks, 2):
        da, db = ({c.card_id: c.count for c in e.cards} for e in (a, b))
        assert da != db


def test_locked_cards_never_removed_by_local_search(model, name2id, held):
    pool = pool_ids(held[4], name2id)
    base = builder.build(model, pool, "bo3", n_decks=1)
    deck0 = {c.card_id: c.count for c in base.decks[0].cards}
    ctx = builder.Ctx(model, "bo3")
    locked = {c: deck0[c] for c in _spells_of(model, deck0)[:3]}
    ctx.locked = locked
    cfg = builder._config_of_deck(ctx, deck0, builder.enumerate_configs(ctx, pool), pool)
    s = builder.Searcher(ctx, cfg, pool, (15, 18))
    for mv in s.moves(deck0):
        for c, n in mv.items():
            if c in locked:
                assert n >= 0
    # a deck that drops a locked card is illegal
    d = dict(deck0)
    c0 = next(iter(locked))
    d[c0] -= locked[c0]
    if d[c0] == 0:
        del d[c0]
    assert not s.valid(d)


def test_build_pool_by_names_and_ids_and_errors(calc, model, name2id, held, names):
    h = held[5]
    text = text_of(h["pool"])
    pool = pool_ids(h, name2id)
    base = api.build_pool("TST", text, n_decks=1)
    deck0 = deck_by_name(base.decks[0])
    sp = [n for n in deck0 if n not in BASICS and not model.card_arrays.is_land[name2id[n]]]
    beta = model.card_beta("bo3")
    lock_name = min(sp, key=lambda n: beta[name2id[n]])
    ex_name = max(sp, key=lambda n: beta[name2id[n]])
    by_name = api.build_pool("TST", text, n_decks=1, locked={lock_name: 1}, excluded=[ex_name])
    by_id = api.build_pool("TST", text, n_decks=1, locked={name2id[lock_name]: 1}, excluded={name2id[ex_name]})
    for r in (by_name, by_id):
        d = deck_by_name(r.decks[0])
        assert d.get(lock_name, 0) >= 1 and ex_name not in d
    assert deck_by_name(by_name.decks[0]) == deck_by_name(by_id.decks[0])
    assert by_name.pool == pool                                       # the excluded card is still part of the pool
    # unknown names are reported, not fatal
    r = api.build_pool("TST", text, n_decks=1, locked=["No Such Card"])
    assert any("not recognised" in w for w in r.warnings)
    # locked and excluded at once, or too many locks, are clear errors
    with pytest.raises(ValueError, match="both locked and excluded"):
        api.build_pool("TST", text, n_decks=1, locked=[lock_name], excluded=[lock_name])
    with pytest.raises(ValueError, match="too many locked"):
        builder.build(model, pool, "bo3", n_decks=1, locked={c: 1 for c in list(pool)[:30] if not model.card_arrays.is_land[c]})
    # ids route used by the visual builder's rebuild
    r2 = api.build_from_pool("TST", pool, "bo3", locked={lock_name: 1}, excluded=[ex_name])
    # the rebuild uses the family search (commit "rebuild uses the family search"), which is at least as good as the classic search
    d2 = deck_by_name(r2.decks[0])
    assert d2.get(lock_name, 0) >= 1 and ex_name not in d2 and r2.pool == pool and r2.families
    assert r2.decks[0].game_wr >= by_name.decks[0].game_wr - 0.002


def test_locks_that_need_too_many_colours(model, name2id, held):
    pool = pool_ids(held[2], name2id)
    ca = model.card_arrays
    by_col: dict[int, int] = {}
    for c in pool:
        if not ca.is_land[c] and (ca.pips[c] > 0).sum() == 1:
            by_col.setdefault(int(np.argmax(ca.pips[c] > 0)), c)
    if len(by_col) < 5:
        pytest.skip("pool lacks mono-colored cards of every color")
    with pytest.raises(ValueError, match="no color combination"):
        builder.build(model, pool, "bo3", n_decks=1, locked={c: 1 for c in by_col.values()})


# ------------------------------------------------------------------------------------------ candidate 1 switches (docs/builder_wide/RULE.md)

def test_wide_configs_and_deficit_both(model, name2id, held, monkeypatch):
    """WIDE_CONFIGS adds 3+1 and 2+2 configurations (gated by fixing), DEFICIT_BOTH fixes the sign; builds stay legal and exact."""
    ctx = builder.Ctx(model, "bo3")
    pools = [pool_ids(h, name2id) for h in held[:6]]
    base = [[c.key for c in builder.enumerate_configs(ctx, p)] for p in pools]
    monkeypatch.setattr(builder, "WIDE_CONFIGS", True)
    monkeypatch.setattr(builder, "DEFICIT_BOTH", True)
    for p, keys in zip(pools, base):
        cfgs = builder.enumerate_configs(ctx, p)
        wide = [c.key for c in cfgs]
        assert set(keys) <= set(wide) and len(wide) >= len(keys)
        for c in cfgs:
            if len(c.main) == 3 and c.splash:
                assert c.floors == {**{"WUBRG".index(x): 4 for x in c.main}, "WUBRG".index(c.splash): 3}
            if len(c.main) == 2 and len(c.splash) == 2:
                assert set(c.floors.values()) == {5, 2} and len(c.sc) == 2
                # a splash card's off-main color is a splash color (a hybrid card needs only one of its colors)
                for i in np.nonzero(c.cls == 2)[0]:
                    need = set(np.nonzero(ctx.ca.pips[i] > 0)[0]) - {"WUBRG".index(x) for x in c.main}
                    assert need and (need & set(c.sc) if ctx.ca.hybrid[i] else need <= set(c.sc))
        res = builder.build(model, p, "bo3", n_decks=1, families=False, n_jobs=1)
        ev = res.decks[0]
        deck = {cl.card_id: cl.count for cl in ev.cards}
        assert sum(deck.values()) == 40
        assert abs(builder.Ctx(model, "bo3").logit(deck) - float(model.logit(deck, "bo3"))) < 1e-9
        ok = False
        for c in cfgs:
            if any(not ctx.is_basic[i] and ((ctx.is_land[i] and not c.land_ok[i]) or (not ctx.is_land[i] and c.cls[i] == 0)) for i in deck):
                continue
            if sum(n for i, n in deck.items() if c.cls[i] == 2) > 3:
                continue
            src = sum(n * ctx.ca.produces[i] for i, n in deck.items() if ctx.is_land[i])
            if all(src[k] >= f for k, f in c.floors.items()):
                ok = True
                break
        assert ok
    # the deficit has its documented sign: a color with no sources but demand needs more (> 0)
    cfg = next(c for c in builder.enumerate_configs(ctx, pools[0]) if len(c.main) == 2 and not c.splash)
    s = builder.Searcher(ctx, cfg, pools[0], (16, 17))
    a, b = cfg.colors
    spell_a = next(int(i) for i in np.nonzero(cfg.cls == 1)[0] if ctx.ca.pips[i, a] > 0)
    deck = {spell_a: 1, ctx.basic_id[b]: 5}
    assert s._deficit(deck, a) > 0 > s._deficit(deck, b)
    assert ctx.splash_term(np.array([spell_a]), [a])[0] == pytest.approx(ctx.splash_term(np.array([spell_a]), a)[0])
