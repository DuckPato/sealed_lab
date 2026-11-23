"""Replay ingest, id resolution and statistics on a synthetic replay file derived from the synthetic game data."""
from __future__ import annotations

import csv
import gzip
import json
import shutil
import threading
from math import comb
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sealed_lab import api
from sealed_lab.config import BASICS
from sealed_lab.ingest import replay
from sealed_lab.model import replaystats as rs
from sealed_lab.types import Cancelled

N_GAMES = 700
K_TURNS = 8


def _expand(row: dict, prefix: str, names: list[str], ids: dict[str, int]) -> list[int]:
    out = []
    for n in names:
        c = int(row.get(prefix + n) or 0)
        out += [ids[n]] * c
    return out


def make_replay_file(synth: Path, out: Path, n: int = N_GAMES) -> dict:
    """Write a replay-format CSV for the first `n` TradSealed games of the synthetic set; return the truth per game."""
    src = synth / "game_data_public.TST.TradSealed.csv.gz"
    with gzip.open(src, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        rows = [r for _, r in zip(range(n), rd)]
        names = [h[len("deck_"):] for h in rd.fieldnames if h.startswith("deck_")]
    ids = {nm: 1000 + i for i, nm in enumerate(names)}
    cols = ["expansion", "event_type", "draft_id", "draft_time", "build_index", "match_number", "game_number", "game_time", "rank",
            "opp_rank", "main_colors", "splash_colors", "on_play", "num_mulligans", "opp_num_mulligans", "opp_colors", "num_turns",
            "won", "candidate_hand_1", "opening_hand"]
    stems = ["cards_drawn", "cards_tutored", "lands_played", "creatures_cast", "non_creatures_cast", "user_instants_sorceries_cast",
             "user_mana_spent", "eot_user_lands_in_play", "eot_user_cards_in_hand", "eot_user_life"]
    for k in range(1, 13):
        cols += [f"user_turn_{k}_{s}" for s in stems] + [f"oppo_turn_{k}_user_instants_sorceries_cast", f"oppo_turn_{k}_eot_user_life"]
    cols += ["user_n_games_bucket", "user_game_win_rate_bucket"] + [f"deck_{n}" for n in names]
    truth = {}
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            opening = _expand(r, "opening_hand_", names, ids)
            draws = _expand(r, "drawn_", names, ids)
            on_play = r["on_play"] == "True"
            done = max(min(int(r["num_turns"]), K_TURNS) - 1, 1)
            hand = list(opening)
            lands_play: list[int] = []
            rec = dict(lands=[], mana=[], cre=[], seen=[], cast=[])
            d = {c: "" for c in cols}
            for c in ("expansion", "event_type", "draft_id", "draft_time", "build_index", "match_number", "game_number", "game_time",
                      "rank", "opp_rank", "main_colors", "splash_colors", "on_play", "num_mulligans", "opp_num_mulligans",
                      "opp_colors", "num_turns", "won"):
                d[c] = r[c]
            d["user_n_games_bucket"] = r["user_n_games_bucket"]
            d["user_game_win_rate_bucket"] = r["user_game_win_rate_bucket"]
            for nm in names:
                d[f"deck_{nm}"] = r[f"deck_{nm}"]
            d["candidate_hand_1"] = d["opening_hand"] = "|".join(map(str, opening))
            di = 0
            for k in range(1, 13):
                if k > done:
                    for s in stems:
                        d[f"user_turn_{k}_{s}"] = ""
                    d[f"user_turn_{k}_user_mana_spent"] = "0.0"
                    d[f"user_turn_{k}_eot_user_life"] = "0.0"
                    d[f"oppo_turn_{k}_eot_user_life"] = "0.0"
                    continue
                if (k > 1 or not on_play) and di < len(draws):
                    hand.append(draws[di])
                    d[f"user_turn_{k}_cards_drawn"] = str(draws[di])
                    rec["seen"].append((k, draws[di]))
                    di += 1
                land = next((c for c in hand if c in {ids[b] for b in names if b in BASICS}), None)
                if land is not None:
                    hand.remove(land)
                    lands_play.append(land)
                    d[f"user_turn_{k}_lands_played"] = str(land)
                spell = next((c for c in hand if c not in {ids[b] for b in names if b in BASICS}), None)
                mana = 0
                cre = 0
                if spell is not None and len(lands_play) >= 2:
                    hand.remove(spell)
                    d[f"user_turn_{k}_creatures_cast"] = str(spell)
                    mana, cre = 2, 1
                    rec["cast"].append((k, spell))
                d[f"user_turn_{k}_user_mana_spent"] = f"{float(mana)}"
                d[f"user_turn_{k}_eot_user_lands_in_play"] = "|".join(map(str, lands_play))
                d[f"user_turn_{k}_eot_user_cards_in_hand"] = "|".join(map(str, hand))
                d[f"user_turn_{k}_eot_user_life"] = "20.0"
                d[f"oppo_turn_{k}_eot_user_life"] = "20.0"
                rec["lands"].append(len(lands_play))
                rec["mana"].append(mana)
                rec["cre"].append(cre)
            rec["done"] = done
            rec["opening"] = opening
            truth[(r["draft_id"], int(r["match_number"]), int(r["game_number"]))] = rec
            w.writerow([d[c] for c in cols])
    return dict(truth=truth, ids=ids, names=names)


@pytest.fixture(scope="module")
def renv(built, tmp_path_factory):
    """A copy of the built synthetic home with one registered replay file."""
    mp = pytest.MonkeyPatch()
    home = tmp_path_factory.mktemp("replay_home") / "home"
    shutil.copytree(built["home"], home)
    mp.setenv("SEALED_LAB_HOME", str(home))
    d = tmp_path_factory.mktemp("replay_in")
    info = make_replay_file(built["synth"], d / "replay_data_public.TST.TradSealed.csv")
    res = api.add_files([str(d)])
    assert [r.kind for r in res] == ["replay"]
    mp.setattr(replay, "RANGE_BYTES", 40_000)         # many byte ranges even for a small file
    yield dict(home=home, **info, file=d / "replay_data_public.TST.TradSealed.csv")
    mp.undo()


def test_ingest_matches_truth_and_is_incremental(renv):
    r = replay.ingest_replays("TST", n_jobs=2)
    assert r["ingested"] == 1 and r["cached"] == 0
    g = replay.load_games("TST", ["TradSealed"])
    assert g.height == N_GAMES and g["gid"].n_unique() == N_GAMES
    assert Path(renv["file"]).stat().st_size > 3 * 40_000       # several ranges were really used
    bad = 0
    for row in g.iter_rows(named=True):
        t = renv["truth"][(row["draft_id"], row["match_number"], row["game_number"])]
        assert row["turns_done"] == t["done"]
        for k in range(1, replay.K_WIDE + 1):
            if k <= t["done"]:
                bad += (row[f"lands_{k}"], row[f"mana_{k}"], row[f"cre_{k}"]) != (t["lands"][k - 1], t["mana"][k - 1], t["cre"][k - 1])
            else:
                assert row[f"lands_{k}"] is None and row[f"mana_{k}"] is None
    assert bad == 0
    # id tables
    dr = replay.load_draws("TST", ["TradSealed"])
    ca = replay.load_casts("TST", ["TradSealed"])
    assert dr.filter(pl.col("kind") == 0)["n"].sum() == sum(len(t["opening"]) for t in renv["truth"].values())
    assert ca["n"].sum() == sum(len(t["cast"]) for t in renv["truth"].values())
    # second run: nothing to do
    r2 = replay.ingest_replays("TST", n_jobs=2)
    assert r2["ingested"] == 0 and r2["cached"] == 1
    assert replay.status("TST")[0]["cached"]


def test_range_size_does_not_change_the_result(renv, monkeypatch):
    base = replay.load_games("TST", ["TradSealed"]).drop("gid", "src").sort("draft_id", "match_number", "game_number")
    fp = replay.status("TST")[0]["fingerprint"]
    shutil.rmtree(replay.shard_path("TST", fp))
    monkeypatch.setattr(replay, "RANGE_BYTES", 1 << 30)           # one single range
    replay.ingest_replays("TST", n_jobs=2)
    again = replay.load_games("TST", ["TradSealed"]).drop("gid", "src").sort("draft_id", "match_number", "game_number")
    assert base.equals(again)


def test_idmap_is_exact(renv):
    m = replay.resolve_idmap("TST", refresh=True)
    from sealed_lab.ingest.corpus import load_corpus

    names = load_corpus("TST")["cards"].sort("card_id")["name"].to_list()
    inv = {v: k for k, v in renv["ids"].items()}
    wrong = [(a, names[c], inv[a]) for a, c in m["map"].items() if names[c] != inv[a]]
    assert not wrong, wrong[:5]
    assert m["mass_mapped"] > 0.999 and m["n_mapped"] >= 0.9 * m["n_ids"]


def test_summary_card_stats_and_html(renv):
    cancel = threading.Event()
    s = rs.replay_summary("TST", n_jobs=2, refresh=True)
    assert s["available"] and s["events"][0]["event"] == "TradSealed" and s["events"][0]["n_games"] == N_GAMES
    json.dumps(s)                                           # JSON-able
    assert rs.cached_summary("TST") is not None
    assert s["simulator"]["rows"] and s["simulator"]["rows"][0]["mull"]["n"] > 0
    html = rs.render_replay_section(s)
    assert "<h2>" in html and "<table" in html and "nan" not in html.lower().replace("finance", "")
    # per-card statistics against the truth
    cs = rs.replay_card_stats("TST", "bo3", n_jobs=2, refresh=True)
    assert cs and all(isinstance(k, int) for k in cs)
    from sealed_lab.ingest.corpus import load_corpus

    cards = load_corpus("TST")["cards"].sort("card_id")
    names = cards["name"].to_list()
    lands = set(cards.filter(pl.col("is_land"))["name"].to_list())
    inv = {v: k for k, v in renv["ids"].items()}
    mapped_ids = set(replay.resolve_idmap("TST")["map"])
    casts: dict[str, int] = {}
    for t in renv["truth"].values():
        for (_k, i) in {(0, c) for _, c in t["cast"]}:
            if inv[i] not in lands and i in mapped_ids:
                casts[inv[i]] = casts.get(inv[i], 0) + 1
    checked = 0
    for cid, v in cs.items():
        assert 0 <= v["cast_rate"] <= 1 if v["cast_rate"] is not None else True
        if names[cid] in casts:
            assert v["n_cast"] == casts[names[cid]]
            checked += 1
    assert checked > 10
    # unavailable / cancelled paths
    cancel.set()
    shutil.rmtree(replay.replay_dir("TST"))
    with pytest.raises(Cancelled):
        replay.ingest_replays("TST", n_jobs=2, cancel=cancel)
    assert rs.replay_summary("NOPE")["available"] is False
    assert "Not available" in rs.render_replay_section({"available": False, "reason": "x"})


# ---------------------------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------------------------

def test_cluster_ratio_and_logit_fit():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 200, 20000)
    u = rng.normal(0, 0.5, 200)[g]
    x = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.3 + 0.8 * x + u)))).astype(float)
    f = rs.logit_fit(np.column_stack([np.ones(20000), x]), y, g)
    assert abs(f["beta"][1] - 0.8) < 0.1 and f["se"][1] > 0
    est, lo, hi = rs.cluster_ratio(y, np.ones_like(y), g, 200)
    assert lo < est < hi and abs(est - y.mean()) < 1e-12
    X1 = np.column_stack([np.ones(20000), x + 1])
    X0 = np.column_stack([np.ones(20000), x])
    a = rs.ame(f, X1, X0)
    assert a[0] > 0 and a[2] < a[0] < a[3]
    o = rs.ols_fit(np.column_stack([np.ones(20000), x]), 2 * x + rng.normal(size=20000), g)
    assert abs(o["beta"][1] - 2) < 0.05


def test_hypergeometric_helpers():
    # 2 lands in hand, 17 lands in 40, two draws: P(no land) = (18/33)(17/32)
    p = rs._screw_given_hand(np.array([40]), np.array([17]), np.array([2]), np.array([2]))[0]
    assert abs(p - 18 * 17 / (33 * 32)) < 1e-12
    assert rs._screw_given_hand(np.array([40]), np.array([17]), np.array([3]), np.array([2]))[0] == 0.0
    f = rs._flood_given_hand(np.array([40]), np.array([17]), np.array([5]))[0]
    exact = sum(comb(12, k) * comb(21, 9 - k) for k in range(4, 10)) / comb(33, 9)       # >= 4 lands in 9 draws
    assert abs(f - exact) < 1e-12
    t = rs._hyper_per_game(np.array([40]), np.array([17]), 7, "tail7")[0]
    assert 0.1 < t < 0.14


def test_first_cards_lands():
    d = pl.DataFrame({"gid": [1] * 5 + [2], "turn": [0, 0, 1, 2, 3, 0], "kind": [3, 3, 1, 1, 1, 3], "id": [10, 11, 10, 11, 11, 10],
                      "n": [3, 4, 1, 1, 2, 7]}, schema={"gid": pl.Int64, "turn": pl.Int8, "kind": pl.Int8, "id": pl.Int32, "n": pl.Int8})
    ii = pl.DataFrame({"id": [10, 11], "is_land": [True, False]})
    out = rs.first_cards_lands(d, ii).sort("gid")
    r1 = out.row(0, named=True)
    assert (r1["n7"], r1["l7"], r1["isl1"]) == (7, 3, 1) and r1["lands9"] is None      # only 4 draws recorded
    assert out.row(1, named=True)["l7"] == 7


def test_wilson_and_weights():
    lo, hi = rs.wilson(50, 100)
    assert 0.4 < lo < 0.5 < hi < 0.6
    w = rs._play_draw_weights(np.array([True] * 30 + [False] * 70))
    assert abs(w[:30].sum() - 50) < 1e-9 and abs(w[30:].sum() - 50) < 1e-9


def test_fake_api_replay_roundtrip():
    from sealed_lab.gui import fake_api

    s = fake_api.replay_summary("TST")
    html = rs.render_replay_section(s)
    assert "Replay data" in html and html.count("<table") >= 8
    cs = fake_api.replay_card_stats("TST", "bo3")
    assert cs and all(0 <= v["cast_rate"] <= 1 for v in cs.values())
