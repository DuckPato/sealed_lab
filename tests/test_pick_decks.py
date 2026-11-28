"""Deck search of a pick-file set (provisional model): the pick-deck tables, the routing of the api, filters, model scores and their cache,
the lazy rebuild, and what stays unavailable (docs: sealed_lab/provisional.py `build_pick_decks`, sealed_lab/deck_search.py `_load_pick`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

from sealed_lab import api, config, deck_search, provisional as prov

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_provisional import SET, N_RUNS, calculated, pick_dir  # noqa: F401  (fixtures)


@pytest.fixture()
def fresh(calculated):  # noqa: F811
    deck_search.clear_cache()
    yield calculated
    deck_search.clear_cache()


def _tables():
    d = config.model_dir(SET)
    return (pl.read_parquet(d / prov.PICK_DECKS), pl.read_parquet(d / prov.PICK_DECK_CARDS), pl.read_parquet(d / prov.PICK_POOLS))


# ------------------------------------------------------------------------------------------------ the tables
def test_calculation_writes_the_pick_deck_tables(fresh):
    runs, dc, pools = _tables()
    meta = fresh["meta"]
    assert prov.pick_decks_stamp(SET) is not None
    assert runs.height == meta["provisional"]["n_runs"] > 0                       # only rebuilt runs (the model's rules)
    assert runs.height + sum(meta["provisional"]["dropped"].values()) == N_RUNS
    assert runs["draft_id"].n_unique() == runs.height
    per_deck = dc.group_by("deck_key").agg(pl.col("n").sum().alias("n"))
    assert per_deck["n"].to_list() == [40] * per_deck.height                        # 40-card decks, basics included
    assert set(runs["deck_key"].to_list()) == set(per_deck["deck_key"].to_list())
    assert (runs["wins"] + runs["losses"]).min() > 0
    assert runs["main_colors"].str.len_chars().min() >= 1 and runs["rank"].drop_nulls().unique().to_list() == ["gold"]
    assert runs["splash_colors"].null_count() == 0 and runs["n_cards"].unique().to_list() == [40]
    # the pool holds every non-basic pick of the run, so it covers the main deck's non-basic cards
    basics = set(pl.read_parquet(config.model_dir(SET) / "cards.parquet").filter(pl.col("is_basic"))["card_id"].to_list())
    r0 = runs.row(0, named=True)
    deck = {c: n for c, n in dc.filter(pl.col("deck_key") == r0["deck_key"]).select("card_id", "n").iter_rows()}
    pool = {c: n for c, n in pools.filter(pl.col("draft_id") == r0["draft_id"]).select("card_id", "n").iter_rows()}
    assert sum(pool.values()) <= 45 and sum(pool.values()) >= sum(n for c, n in deck.items() if c not in basics)
    assert all(pool.get(c, 0) >= n for c, n in deck.items() if c not in basics)


def test_stamp_follows_the_model_and_tables_are_made_again_when_missing(fresh):
    d = config.model_dir(SET)
    before = _tables()[0]
    (d / prov.PICK_POOLS).unlink()
    assert prov.pick_decks_stamp(SET) is None
    assert deck_search.events(SET)                                                  # lazy: rebuilt from the registered pick file
    assert prov.pick_decks_stamp(SET) is not None and _tables()[0].equals(before)
    meta = json.loads((d / prov.PICK_DECKS_META).read_text(encoding="utf-8"))
    (d / prov.PICK_DECKS_META).write_text(json.dumps(dict(meta, built_at="1999-01-01T00:00:00+00:00")), encoding="utf-8")
    assert prov.pick_decks_stamp(SET) is None                                       # tables of another model are not used
    assert prov.ensure_pick_decks(SET) and prov.pick_decks_stamp(SET) is not None


def test_no_tables_and_no_pick_file_is_a_plain_error(fresh, monkeypatch):
    d = config.model_dir(SET)
    saved = {fn: (d / fn).read_bytes() for fn in (prov.PICK_DECKS, prov.PICK_DECK_CARDS, prov.PICK_POOLS, prov.PICK_DECKS_META)}
    try:
        (d / prov.PICK_DECKS_META).unlink()
        monkeypatch.setattr(prov, "pick_files", lambda sc: [])
        assert not prov.ensure_pick_decks(SET)
        with pytest.raises(FileNotFoundError, match="draft decks of this set cannot be rebuilt"):
            api.deck_search_events(SET)
    finally:
        for fn, b in saved.items():
            (d / fn).write_bytes(b)
        deck_search.clear_cache()


# ------------------------------------------------------------------------------------------------ the api routing
def test_events_cards_and_search_read_the_pick_decks(fresh):
    runs = _tables()[0]
    ev = api.deck_search_events(SET)
    assert [e["event"] for e in ev] == ["PremierDraft"] and ev[0]["label"] == "Premier Draft (Bo1)" and ev[0]["n_runs"] == runs.height
    assert ev[0]["n_games"] == int((runs["wins"] + runs["losses"]).sum())
    names = api.deck_search_cards(SET)
    assert names and "Plains" not in names
    res = api.search_decks(SET, min_games=1, limit=0)
    assert res["total"] == runs.height and res["shown"] == runs.height
    r = res["results"][0]
    assert r["event"] == "PremierDraft" and r["fmt"] == "draft" and r["drafted_pool"] and not r["bo3"]
    assert sum(c["n"] for c in r["main"]) == 40 and r["n_cards"] == 40 and r["n_builds"] == 1
    assert isinstance(r["sideboard"], list)                                        # the rest of the drafted pool
    assert sum(c["n"] for c in r["main"] if not c["is_basic"]) + sum(c["n"] for c in r["sideboard"]) <= 45
    assert res["analytics"]["pool_kind"] == "drafted" and res["analytics"]["n_runs"] == runs.height
    assert api.deck_search_text(SET, r).count("\n") > 5


def test_card_filters_in_deck_and_in_pool(fresh):
    _runs, dc, pools = _tables()
    cards = pl.read_parquet(config.model_dir(SET) / "cards.parquet")
    cid = int(dc.join(cards.filter(~pl.col("is_basic")).select("card_id"), on="card_id")["card_id"].mode()[0])
    name = cards.filter(pl.col("card_id") == cid)["name"][0]
    res = api.search_decks(SET, name, min_games=1, limit=0, min_card_share=0.5)
    keys = set(dc.filter(pl.col("card_id") == cid)["deck_key"].to_list())
    assert res["total"] > 0 and all(r["deck_key"] in keys and r["copies"] >= 1 for r in res["results"])
    assert all(any(c["name"] == name for c in r["main"]) for r in res["results"])
    # a card in the pool but not in the deck: "pool only"
    po = api.search_decks(SET, None, min_games=1, limit=0, in_pool_only=[name])
    assert po["total"] > 0
    for r in po["results"]:
        assert all(c["name"] != name for c in r["main"]) and any(c["name"] == name for c in r["sideboard"])
    ex = api.search_decks(SET, None, min_games=1, limit=0, exclude_cards=[name])
    assert ex["total"] == api.search_decks(SET, None, min_games=1, limit=0)["total"] - len(keys_runs(_runs, keys))


def keys_runs(runs: pl.DataFrame, keys: set) -> list:
    return [k for k in runs["deck_key"].to_list() if k in keys]


def test_colors_rank_skill_date_and_sorts(fresh):
    runs = _tables()[0]
    pair = runs["main_colors"].mode()[0]
    ex = api.search_decks(SET, min_games=1, limit=0, colors=pair, color_match="exact")
    assert ex["total"] == (runs["main_colors"] == pair).sum() and {r["colors"].split("+")[0] for r in ex["results"]} == {pair}
    inc = api.search_decks(SET, min_games=1, limit=0, colors=pair[0], color_match="include")
    assert inc["total"] == runs["main_colors"].str.contains(pair[0], literal=True).sum()
    assert api.search_decks(SET, min_games=1, ranks=["Gold"])["total"] == runs.height
    assert api.search_decks(SET, min_games=1, ranks=["Mythic"])["total"] == 0
    assert api.search_decks(SET, min_games=1, min_rank="Platinum")["total"] == 0
    sk = api.search_decks(SET, min_games=1, limit=0, min_skill=0.55)
    assert sk["total"] == (runs["skill"] >= 0.55).sum() > 0
    old = api.search_decks(SET, min_games=1, limit=0, days=1)                         # the synthetic runs span ~10 days
    assert 0 < old["total"] < runs.height and old["date_cutoff"]
    wr = [r["win_rate"] for r in api.search_decks(SET, min_games=1, limit=0, sort="win_rate")["results"]]
    assert wr == sorted(wr, reverse=True)
    best = api.search_decks(SET, min_games=1, limit=0, sort="result")["results"]
    assert best[0]["score"] >= best[-1]["score"]
    assert api.search_decks(SET, min_games=1, limit=0, trophy_only=True)["total"] == sum(1 for r in best if r["trophy"])
    assert api.search_decks(SET, min_games=1, limit=0, splash="none")["total"] == (runs["splash_colors"] == "").sum()


def test_color_pair_insights_have_games_skill_and_wins(fresh):
    runs = _tables()[0]
    res = api.search_decks(SET, min_games=1, limit=1)
    bc = res["analytics"]["by_color"]
    assert bc and sum(c["runs"] for c in bc) <= runs.height
    for c in bc:
        sub = runs.filter(pl.col("main_colors") == c["colors"])
        g = int((sub["wins"] + sub["losses"]).sum())
        assert c["games"] == g and c["raw_wr"] == pytest.approx(int(sub["wins"].sum()) / g)
        assert c["avg_skill"] is not None


def test_unavailable_views_keep_raising_the_provisional_message(fresh):
    for fn, args in ((api.search_similar_pools, (SET, {1: 1})), (api.search_rebuilds, (SET,)), (api.search_sideboarding, (SET,))):
        with pytest.raises(ValueError, match="Not available yet: this set has only pick data"):
            fn(*args)
    for fn in (deck_search.similar_pools, deck_search.search_rebuilds, deck_search.search_sideboarding):
        with pytest.raises(ValueError, match="Not available yet"):
            fn(SET, {1: 1}) if fn is deck_search.similar_pools else fn(SET)
    with pytest.raises(ValueError, match="Not available yet"):
        api.set_report(SET, "bo1")


# ------------------------------------------------------------------------------------------------ model scores
def test_model_scores_are_cached_with_the_model(fresh, monkeypatch):
    d = config.model_dir(SET)
    for fn in (deck_search._MS_FILE, deck_search._MS_STAMP):
        (d / fn).unlink(missing_ok=True)
    deck_search.clear_cache()
    assert not api.has_deck_model_scores(SET)
    res = api.search_decks(SET, min_games=1, limit=0)
    assert all(r["model_score"] is None for r in res["results"])                     # never computes on its own
    seen: list[float] = []
    n = api.deck_model_scores(SET, progress=lambda f, m: seen.append(f))
    runs, dc, _pools = _tables()
    assert n == dc["deck_key"].n_unique() and seen and (d / deck_search._MS_FILE).exists() and api.has_deck_model_scores(SET)
    ms = pl.read_parquet(d / deck_search._MS_FILE)
    assert ms["fmt"].unique().to_list() == ["draft"] and ms["model_score"].min() >= 0 and ms["model_score"].max() <= 100
    top = api.search_decks(SET, min_games=1, limit=0, sort="model")
    sc = [r["model_score"] for r in top["results"]]
    assert None not in sc and sc == sorted(sc, reverse=True) and top["results"][0]["model_q"] is not None
    bc = top["analytics"]["by_color"]
    assert all(c["model_median"] is not None and c["model_best"] >= c["model_median"] for c in bc)
    # the scores are exactly the model's own number for a deck (the path of normal sets: model.q + percentile)
    from sealed_lab.deck import builder

    m = builder.load_model(SET)
    r0 = top["results"][3]
    deck = {c["card_id"]: c["n"] for c in r0["main"]}
    q = float(m.q(deck, "bo1"))
    assert r0["model_q"] == pytest.approx(q, abs=1e-4) and r0["model_score"] == pytest.approx(float(m.percentile(q, "bo1")), abs=1e-3)
    # a second run, with the in-memory cache gone, reads the file and does not load the model
    deck_search._MS_CACHE.clear()
    monkeypatch.setattr(builder, "load_model", lambda *a, **k: pytest.fail("scores are cached"))
    assert api.deck_model_scores(SET) == n


def test_model_scores_are_stale_after_a_new_model(fresh):
    d = config.model_dir(SET)
    api.deck_model_scores(SET)
    assert api.has_deck_model_scores(SET)
    meta = json.loads((d / prov.PICK_DECKS_META).read_text(encoding="utf-8"))
    (d / prov.PICK_DECKS_META).write_text(json.dumps(dict(meta, n_runs=meta["n_runs"] + 1)), encoding="utf-8")
    deck_search.clear_cache()
    assert not api.has_deck_model_scores(SET)                                         # another tables stamp: scored again
    (d / prov.PICK_DECKS_META).write_text(json.dumps(meta), encoding="utf-8")
    deck_search.clear_cache()
    assert api.has_deck_model_scores(SET)


def test_cast_counts_say_a_pick_only_set_has_no_games(fresh):
    name = api.deck_search_cards(SET)[0]
    res = api.search_decks(SET, name, min_games=1, min_card_share=0, limit=5)
    assert res["results"] and all(x["cast"] is None for x in res["results"])
    assert res["cast_summary"] is None
    assert res["cast_note"] == "This set has only pick data: its decks were never played in recorded games."
