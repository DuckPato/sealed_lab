"""Card deck search (sealed_lab/deck_search.py + the api facade): real runs of the corpus that played a card."""
from __future__ import annotations

import os

import polars as pl
import pytest

from sealed_lab import api, deck_search
from sealed_lab.config import corpus_dir


@pytest.fixture(scope="module")
def tst(built):
    deck_search.clear_cache()
    cdir = corpus_dir("TST")
    games = pl.read_parquet(cdir / "games.parquet")
    builds = pl.read_parquet(cdir / "builds.parquet")
    dc = pl.read_parquet(cdir / "deck_cards.parquet")
    cards = pl.read_parquet(cdir / "cards.parquet")
    # a card played in many decks of every event
    nonbasic = cards.filter(~pl.col("is_basic") & ~pl.col("is_land"))
    counts = dc.join(nonbasic.select("card_id"), on="card_id").group_by("card_id").len().sort(["len", "card_id"], descending=[True, False])
    cid = int(counts["card_id"][len(counts) // 4])
    name = nonbasic.filter(pl.col("card_id") == cid)["name"][0]
    return dict(cdir=cdir, games=games, builds=builds, dc=dc, cards=cards, cid=cid, name=name)


def _runs_with_card(t, cid):
    keys = t["dc"].filter(pl.col("card_id") == cid)["deck_key"].to_list()
    return set(t["builds"].filter(pl.col("deck_key").is_in(keys)).select("event", "draft_id").iter_rows())


def test_card_names_and_events(tst):
    names = deck_search.card_names("TST")
    basics = set(tst["cards"].filter(pl.col("is_basic"))["name"].to_list())
    assert names and not (set(names) & basics)
    assert names == sorted(names, key=str.lower)
    assert tst["name"] in names
    evs = deck_search.events("TST")
    assert [e["event"] for e in evs] == ["PremierDraft", "TradDraft", "Sealed", "TradSealed"]
    by = {e["event"]: e for e in evs}
    assert by["TradDraft"]["bo3"] and by["TradSealed"]["bo3"] and not by["PremierDraft"]["bo3"] and not by["Sealed"]["bo3"]
    assert by["PremierDraft"]["label"] == "Premier Draft (Bo1)" and by["TradDraft"]["label"] == "Traditional Draft (Bo3)"
    assert all(e["max_wins"] >= 1 and e["n_runs"] > 0 for e in evs)
    assert api.deck_search_cards("TST") == names
    assert api.deck_search_events("TST") == evs


def test_results_are_runs_that_played_the_card(tst):
    r = deck_search.search("TST", tst["name"], min_games=1, min_card_share=0, limit=0)
    want = _runs_with_card(tst, tst["cid"])
    got = {(x["event"], x["draft_id"]) for x in r["results"]}
    assert got == want and r["total"] == r["total_all"] == len(want)
    assert r["card"] == tst["name"] and r["card_id"] == tst["cid"]
    for x in r["results"][:200]:
        main = {d["card_id"]: d["n"] for d in x["main"]}
        assert main.get(tst["cid"]) == x["copies"] >= 1
        assert sum(main.values()) == x["n_cards"]
        assert x["games_with_card"] <= x["games"] and x["build_games"] <= x["games_with_card"]


def test_records_match_the_games(tst):
    r = deck_search.search("TST", tst["name"], min_games=1, min_card_share=0, limit=300)
    g = tst["games"]
    for x in r["results"]:
        gg = g.filter((pl.col("draft_id") == x["draft_id"]) & (pl.col("event") == x["event"]))
        w = int(gg["won"].sum())
        assert (x["wins"], x["losses"], x["games"]) == (w, gg.height - w, gg.height)
        if x["bo3"]:
            m = gg.group_by("match_number").agg(pl.col("won").sum().alias("w"), (~pl.col("won")).sum().alias("l"))
            mw = int((m["w"] > m["l"]).sum())
            ml = int((m["w"] < m["l"]).sum())
            inc = int((m["w"] == m["l"]).sum())
            assert (x["match_wins"], x["match_losses"], x["incomplete_matches"]) == (mw, ml, inc)
            assert x["record"].startswith(f"{mw}–{ml} (")
        else:
            assert x["record"] == f"{w}–{gg.height - w}"
        assert x["trophy"] == ((x["match_wins"] if x["bo3"] else x["wins"]) >= x["max_wins"])
        assert x["date"][:4].isdigit()


def test_filters(tst):
    full = deck_search.search("TST", tst["name"], min_games=1, min_card_share=0, limit=0)
    r = deck_search.search("TST", tst["name"], events=["Sealed", "TradSealed"], min_games=1, min_card_share=0, limit=0)
    assert r["results"] and {x["event"] for x in r["results"]} <= {"Sealed", "TradSealed"}
    assert r["total"] == full["by_event"].get("Sealed", 0) + full["by_event"].get("TradSealed", 0)
    assert r["by_event_all"] == full["by_event"]
    r5 = deck_search.search("TST", tst["name"], min_games=5, min_card_share=0, limit=0)
    assert all(x["games"] >= 5 for x in r5["results"])
    assert r5["total"] == sum(1 for x in full["results"] if x["games"] >= 5)
    rs = deck_search.search("TST", tst["name"], min_games=1, min_card_share=0.5, limit=0)
    assert rs["total"] == sum(1 for x in full["results"] if x["games_with_card"] >= 0.5 * x["games"])
    assert all(x["games_with_card"] * 2 >= x["games"] for x in rs["results"])


def test_sorts_and_paging(tst):
    best = deck_search.search("TST", tst["name"], sort="result", limit=0)["results"]
    keys = [(-x["score"], x["losses"] if not x["bo3"] else x["match_losses"], -x["win_rate"]) for x in best]
    assert keys == sorted(keys)
    assert best[0]["trophy"] or best[0]["score"] == max(x["score"] for x in best)
    wr = deck_search.search("TST", tst["name"], sort="win_rate", limit=0)["results"]
    assert [x["win_rate"] for x in wr] == sorted((x["win_rate"] for x in wr), reverse=True)
    new = deck_search.search("TST", tst["name"], sort="newest", limit=0)["results"]
    assert [x["date"] for x in new] == sorted((x["date"] for x in new), reverse=True)
    old = deck_search.search("TST", tst["name"], sort="oldest", limit=0)["results"]
    assert [x["date"] for x in old] == sorted(x["date"] for x in old)
    p1 = deck_search.search("TST", tst["name"], limit=7)
    p2 = deck_search.search("TST", tst["name"], limit=7, offset=7)
    assert p1["shown"] == 7 and p2["offset"] == 7
    assert [x["draft_id"] for x in p1["results"] + p2["results"]] == [x["draft_id"] for x in best[:14]]
    with pytest.raises(ValueError):
        deck_search.search("TST", tst["name"], sort="random")


def test_sealed_sideboard_is_pool_minus_deck(tst):
    r = deck_search.search("TST", tst["name"], events=["Sealed", "TradSealed"], min_games=1, min_card_share=0, limit=40)
    pools = pl.read_parquet(tst["cdir"] / "pools.parquet")
    assert r["results"]
    for x in r["results"]:
        assert isinstance(x["sideboard"], list)
        pool = {int(c): int(n) for c, n in pools.filter(pl.col("draft_id") == x["draft_id"]).select("card_id", "n").iter_rows()}
        both: dict[int, int] = {}
        for d in x["main"] + x["sideboard"]:
            if not d["is_basic"]:
                both[d["card_id"]] = both.get(d["card_id"], 0) + d["n"]
        assert all(both.get(c, 0) >= n for c, n in pool.items())               # pool = main + sideboard (non-basics)
        assert all(d["n"] > 0 and not d["is_basic"] for d in x["sideboard"])


def test_draft_results_have_no_sideboard(tst):
    r = deck_search.search("TST", tst["name"], events=["PremierDraft", "TradDraft"], limit=20)
    assert r["results"]
    for x in r["results"]:
        assert x["sideboard"] is None and "not stored" in x["sideboard_note"]
        assert x["fmt"] == "draft"


def test_deck_text_is_arena_import(tst):
    x = deck_search.search("TST", tst["name"], events=["Sealed"], min_games=1, min_card_share=0, limit=1)["results"][0]
    text = api.deck_search_text("TST", x)
    lines = text.splitlines()
    assert lines[0] == "Deck"
    main_part = lines[1:lines.index("")] if "" in lines else lines[1:]
    assert sum(int(ln.split(" ", 1)[0]) for ln in main_part) == x["n_cards"]
    if x["sideboard"]:
        assert "Sideboard" in lines
    d = deck_search.search("TST", tst["name"], events=["PremierDraft"], limit=1)["results"][0]
    assert "Sideboard" not in api.deck_search_text("TST", d)


def test_card_lookup(tst):
    nm = tst["name"]
    assert deck_search.resolve_card("TST", nm.upper()) == tst["cid"]
    assert deck_search.resolve_card("TST", "  " + nm.lower() + " ") == tst["cid"]
    assert deck_search.search("TST", tst["cid"], limit=1)["card_id"] == tst["cid"]
    assert deck_search.resolve_card("TST", "") is None
    with pytest.raises(ValueError):
        deck_search.search("TST", "No Such Card Anywhere")


def test_busy_set_and_missing_corpus(tst, tmp_path, monkeypatch):
    from sealed_lab.types import SetBusy

    with api._calculating("TST"):
        with pytest.raises(SetBusy):
            api.search_decks("TST", tst["name"])
        with pytest.raises(SetBusy):
            api.deck_search_cards("TST")
    with pytest.raises(FileNotFoundError):
        deck_search.card_names("NOPE")


def test_read_only(tst):
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tst["cdir"].iterdir() if p.is_file()}
    deck_search.clear_cache()
    deck_search.search("TST", tst["name"], events=["Sealed"], limit=50)
    api.deck_search_text("TST", deck_search.search("TST", tst["name"], limit=1)["results"][0])
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tst["cdir"].iterdir() if p.is_file()}
    assert before == after


def test_cache_follows_the_corpus(tst, tmp_path, monkeypatch):
    """A rebuilt corpus (new file stamps) is read again; the same stamps reuse the loaded index."""
    a = deck_search._load("TST")
    assert deck_search._load("TST") is a
    p = tst["cdir"] / "builds.parquet"
    st = p.stat()
    try:
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        assert deck_search._load("TST") is not a
    finally:
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_browse_mode_multi_card_and_advanced_filters(tst):
    # Browse mode (card=None) returns decks across the corpus
    br = deck_search.search("TST", None, min_games=1, limit=50)
    assert br["card"] is None and br["card_id"] is None and br["total"] > 0 and br["results"]
    assert isinstance(br["analytics"], dict) and br["analytics"]["n_runs"] == br["total"]
    assert br["analytics"]["by_color"]

    # Pick a second card that co-occurs in the first result of tst["name"]
    r1 = deck_search.search("TST", tst["name"], min_games=1, min_card_share=0, limit=10)
    assert r1["results"]
    sample_main = r1["results"][0]["main"]
    other = next(c for c in sample_main if not c["is_land"] and c["card_id"] != tst["cid"])

    # Include + Exclude constraints
    both = deck_search.search("TST", tst["name"], include_cards=[(other["name"], 1)], min_games=1, min_card_share=0, limit=50)
    assert both["total"] > 0
    for x in both["results"]:
        ids = {c["card_id"] for c in x["main"]}
        assert tst["cid"] in ids and other["card_id"] in ids

    excl = deck_search.search("TST", tst["name"], exclude_cards=[other["name"]], min_games=1, min_card_share=0, limit=50)
    for x in excl["results"]:
        ids = {c["card_id"] for c in x["main"]}
        assert tst["cid"] in ids and other["card_id"] not in ids

    # Sealed in_pool_only (cut from main deck)
    sealed_r = deck_search.search("TST", tst["name"], events=["Sealed", "TradSealed"], min_games=1, min_card_share=0, limit=10)
    sb_card = next((c for x in sealed_r["results"] for c in (x["sideboard"] or []) if not c["is_basic"]), None)
    if sb_card is not None:
        cut_r = deck_search.search("TST", None, events=["Sealed", "TradSealed"], in_pool_only=[sb_card["name"]], min_games=1, limit=20)
        assert cut_r["total"] > 0
        for x in cut_r["results"]:
            assert sb_card["card_id"] not in {c["card_id"] for c in x["main"]}
            assert sb_card["card_id"] in {c["card_id"] for c in (x["sideboard"] or [])}

    # Trophy, skill, color, splash, lands filters
    tr = deck_search.search("TST", tst["name"], trophy_only=True, min_games=1, min_card_share=0, limit=50)
    assert all(x["trophy"] for x in tr["results"])
    c0 = r1["results"][0]["colors"].split("+")[0]
    if len(c0) == 2:
        cf = deck_search.search("TST", tst["name"], colors=c0, splash="none", min_lands=15, max_lands=18,
                                min_games=1, min_card_share=0, limit=50)
        assert all(x["colors"] == c0 and 15 <= x["n_lands"] <= 18 for x in cf["results"])

    # Analytics includes companions & hands.parquet play_profile
    an = r1["analytics"]
    assert an is not None and an["n_runs"] == r1["total"]
    assert an["by_copies"]
    if an["play_profile"] is not None:
        assert an["play_profile"]["n_games"] >= 10

    # Date window filter (days relative to latest game in the set)
    d7 = deck_search.search("TST", None, min_games=1, days=7, limit=50)
    d30 = deck_search.search("TST", None, min_games=1, days=30, limit=50)
    assert d7["date_cutoff"] is not None and d30["date_cutoff"] is not None
    assert d7["total"] <= d30["total"] <= br["total"]
    assert all(x["date"][:10] >= d7["date_cutoff"] for x in d7["results"])


def test_similar_pools_rebuilds_and_bo3_sideboarding(tst):
    # Similar pools using a real Sealed run's pool (should find itself near the top with high similarity)
    sealed = deck_search.search("TST", tst["name"], events=["Sealed", "TradSealed"], min_games=1, min_card_share=0, limit=1)
    assert sealed["results"]
    sample = sealed["results"][0]
    pool_counts: dict[str, int] = {}
    for c in sample["main"] + (sample["sideboard"] or []):
        if not c["is_basic"]:
            pool_counts[c["name"]] = pool_counts.get(c["name"], 0) + c["n"]
    sim = api.search_similar_pools("TST", pool_counts, min_games=1, limit=10)
    assert sim["total"] > 0 and sim["results"] and sim["color_breakdown"]
    assert sim["results"][0]["similarity"] >= sim["results"][-1]["similarity"]
    assert sim["results"][0]["similarity"] == pytest.approx(1.0)

    # Mid-run rebuilds API runs cleanly and returns well-formed structure
    reb = api.search_rebuilds("TST", limit=20)
    assert "analytics" in reb and "results" in reb
    for x in reb["results"]:
        assert x["n_builds"] >= 2 and isinstance(x["builds"], list)

    # Bo3 sideboarding API runs cleanly in both matchup mode and card mode
    sb_all = api.search_sideboarding("TST", limit=20)
    assert "total_postboard" in sb_all and "matches" in sb_all
    sb_card = api.search_sideboarding("TST", card=tst["name"], limit=20)
    assert sb_card["card"] == tst["name"] and "matches" in sb_card


def test_bo3_rebuilds_are_changes_of_the_game1_deck_between_matches(tst):
    """17lands numbers a new build for every sideboarded game: in Bo3 a rebuild is a different game-1 deck in a later match."""
    g = tst["games"].filter(pl.col("event") == "TradSealed").sort(["draft_id", "match_number", "game_number"])
    g1 = g.group_by(["draft_id", "match_number"], maintain_order=True).agg(pl.col("deck_key").first())
    want = {d for d, n in g1.group_by("draft_id").agg(pl.col("deck_key").n_unique()).iter_rows() if n > 1}
    r = deck_search.search_rebuilds("TST", events=["TradSealed"], limit=0)
    got = {x["draft_id"] for x in r["results"]}
    assert got == want and r["analytics"]["n_multi"] == len(want)
    for x in r["results"]:
        assert sum(b["games"] for b in x["builds"]) == x["games"]          # every game, sideboarded ones included, belongs to a build


def test_similar_pool_weights_come_from_the_model(tst, tmp_path, monkeypatch):
    """The saved model's per-format `<fmt>__beta` draws weight the pool similarity (posterior mean |beta|)."""
    import numpy as np

    names = tst["cards"]["name"].to_list()
    pl.DataFrame({"card_id": list(range(len(names))), "name": names}).write_parquet(tmp_path / "cards.parquet")
    beta = np.zeros((4, len(names)))
    beta[:, tst["cid"]] = 0.1                                               # one card valued: weight 1 + 20 * 0.1 = 3
    np.savez(tmp_path / "draws.npz", bo3__beta=beta)
    monkeypatch.setattr(deck_search, "model_dir", lambda sc: tmp_path)
    idx = deck_search._load("TST")
    idx._card_weights = None
    try:
        w = deck_search._pool_card_weights(idx)
        cid = idx.by_name[tst["name"].lower()]
        assert w[cid] == pytest.approx(3.0)
        assert all(v == pytest.approx(1.0) for c, v in w.items() if c != cid)  # model weights replace the rarity fallback
    finally:
        idx._card_weights = None


def test_companions_use_strong_pilots_and_a_colour_matched_baseline(tst):
    r = deck_search.search("TST", tst["name"], min_games=1, min_card_share=0, limit=0)
    an = r["analytics"]
    for c in an["by_color"]:
        assert "adj_wr_delta" not in c and (c["avg_skill"] is None or 0 <= c["avg_skill"] <= 1)
    for c in an["companions"]:
        assert 0 <= c["base_pct"] <= 1 and c["lift"] == pytest.approx(c["pct_in_matches"] / max(c["base_pct"], 0.005))


def test_replay_card_info_candidate_hand_keep_rate(tst, tmp_path, monkeypatch):
    """_replay_card_info reads candidate_hand_1 (kind 3) and num_mulligans from replay shards."""
    import json

    rdir = tmp_path / "replay"
    sh = rdir / "shard0"
    sh.mkdir(parents=True)
    cid = tst["cid"]
    (rdir / "idmap.json").write_text(json.dumps({"map": {"9001": cid}}), encoding="utf-8")
    # 40 replay games: 36 kept 1st hand (90% base keep), 4 mulliganed; card 9001 in 25 1st-hands, only 1 mulliganed (96% keep)
    gids = list(range(1, 41))
    mulls = [1 if g <= 4 else 0 for g in gids]
    pl.DataFrame({"gid": pl.Series(gids, dtype=pl.Int64), "num_mulligans": pl.Series(mulls, dtype=pl.Int8)}).write_parquet(sh / "games.parquet")
    draws = []
    for g in range(4, 29):          # g=4 mulliganed, g=5..28 kept -> 24/25 = 0.96 keep rate
        draws.append((g, 0, 3, 9001, 1))
        if g >= 5:
            draws.append((g, 0, 0, 9001, 1))
    pl.DataFrame(draws, schema={"gid": pl.Int64, "turn": pl.Int8, "kind": pl.Int8, "id": pl.Int32, "n": pl.Int8}, orient="row").write_parquet(sh / "draws.parquet")
    casts = [(g, 3, 0, 9001, 1) for g in range(5, 20)]
    pl.DataFrame(casts, schema={"gid": pl.Int64, "turn": pl.Int8, "kind": pl.Int8, "id": pl.Int32, "n": pl.Int8}, orient="row").write_parquet(sh / "casts.parquet")

    monkeypatch.setattr(deck_search, "cache_dir", lambda sc: tmp_path)
    idx = deck_search._load("TST")
    idx._replay_hist_cache.clear()
    idx._replay_mull_cache = None
    try:
        rep = deck_search._replay_card_info(idx, cid)
        assert rep is not None
        assert rep["n_cand1"] == 25
        assert rep["base_keep_rate"] == pytest.approx(0.90)
        assert rep["cand1_keep_rate"] == pytest.approx(24 / 25)
        assert rep["keep_delta"] == pytest.approx(24 / 25 - 0.90)
        assert rep["n_cast"] == 15 and rep["cast_rate"] == pytest.approx(15 / 24)
    finally:
        idx._replay_hist_cache.clear()
        idx._replay_mull_cache = None


def test_model_score_sort_colour_include_and_rank_list(calc):
    """Model scores of real Sealed decks (cached next to the model), the "including" color match and the rank list."""
    import polars as pl

    from sealed_lab import api, config
    from sealed_lab.deck import builder

    deck_search.clear_cache()                                     # the corpus index of another fixture home may be cached as TST
    deck_search._MS_CACHE.clear()
    for f in ("deck_scores.parquet", "deck_scores.json"):
        (config.model_dir("TST") / f).unlink(missing_ok=True)
    assert not api.has_deck_model_scores("TST")
    seen = []
    n = api.deck_model_scores("TST", progress=lambda fr, m: seen.append(fr))
    assert n > 0 and seen and seen[-1] == 1.0
    assert api.has_deck_model_scores("TST")
    deck_search._MS_CACHE.clear()
    assert api.has_deck_model_scores("TST")                       # read back from the file

    res = deck_search.search("TST", None, sort="model", limit=50, with_analytics=True)
    sc = [r["model_score"] for r in res["results"] if r["model_score"] is not None]
    assert sc and sc == sorted(sc, reverse=True)
    top = res["results"][0]
    model = builder.load_model("TST")
    deck = {}
    for line in top["main"]:
        deck[int(line["card_id"])] = int(line["n"])
    assert top["model_q"] == pytest.approx(model.q(deck, top["fmt"]), abs=1e-5)
    assert top["model_score"] == pytest.approx(model.percentile(model.q(deck, top["fmt"]), top["fmt"]), abs=1e-3)
    assert any(c.get("model_median") is not None for c in res["analytics"]["by_color"])
    assert all(r["model_score"] is None for r in res["results"] if r["fmt"] not in ("bo1", "bo3"))

    # a recalculated model (meta.json touched) invalidates the cache
    meta = config.model_dir("TST") / "meta.json"
    os.utime(meta, ns=(meta.stat().st_atime_ns, meta.stat().st_mtime_ns + 10_000_000))
    deck_search._MS_CACHE.clear()
    assert not api.has_deck_model_scores("TST")

    # colors: "include" keeps every deck whose main colors contain the ticked ones
    allr = deck_search.search("TST", None, limit=0, with_analytics=False)["results"]
    c = next(r["colors"].split("+")[0] for r in allr if len(r["colors"].split("+")[0]) == 2)
    exact = deck_search.search("TST", None, colors=c, limit=0, with_analytics=False)["results"]
    incl = deck_search.search("TST", None, colors=c[0], color_match="include", limit=0, with_analytics=False)["results"]
    assert exact and all(r["colors"].split("+")[0] == c for r in exact)
    assert incl and all(c[0] in r["colors"].split("+")[0] for r in incl)
    assert {r["draft_id"] for r in exact} <= {r["draft_id"] for r in incl}
    with pytest.raises(ValueError):
        deck_search.search("TST", None, color_match="nope")

    # ranks: only the listed tiers
    idx = deck_search._load("TST")
    tiers = idx.runs.filter(pl.col("rank_tier") > 0)["rank"].drop_nulls().to_list()
    if tiers:
        want = tiers[0].split("-")[0].strip()
        rr = deck_search.search("TST", None, ranks=[want], limit=0, min_games=1, with_analytics=False)["results"]
        assert rr and all(r["rank"].split("-")[0].strip().lower() == want.lower() for r in rr)
    deck_search.clear_cache()
    deck_search._MS_CACHE.clear()
