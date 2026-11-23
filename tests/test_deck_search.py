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
            assert (x["match_wins"], x["match_losses"]) == (mw, m.height - mw)
            assert x["record"].startswith(f"{mw}–{m.height - mw} (")
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
