"""Tests for Sealed in-color play rates and the Deck builder's "rarely played" note."""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import polars as pl
import pytest

from sealed_lab import api, config, play_rates
from sealed_lab.gui import fake_api


def _write_synthetic_corpus(home, sc: str = "SYN") -> None:
    """Create a small synthetic Sealed corpus with known in-color, colorless, splashed, land, Bo1, and Bo3 runs."""
    cdir = home / "sets" / sc / "corpus"
    mdir = home / "sets" / sc / "model"
    cdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    # Cards:
    # 0: RareW    (W,  nonland) -> opened in 60 WU runs (played in 5 -> 5/60 = 8.3% < 15% -> FLAGGED)
    #                              + opened & played as splash in 10 BG runs (excluded from BOTH top and bottom!)
    # 1: BorderU  (U,  nonland) -> opened in 60 WU runs (played in 9 -> 9/60 = 15.0% -> NOT flagged)
    # 2: BelowMin (W,  nonland) -> opened in 49 WU runs (played in 2 -> 2/49 = 4.1%, but < 50 runs -> NOT flagged)
    # 3: Artifact ("", nonland) -> colorless: opened in 60 WU + 10 BG = 70 runs (played in 6 -> 6/70 = 8.6% < 15% -> FLAGGED)
    # 4: GoldWU   (WU, nonland) -> opened in 60 WU runs + 10 BG runs + 5 W-mono runs: only the 60 WU runs are eligible!
    #                              played in 8 of 60 WU runs (13.3% < 15% -> FLAGGED), plus played in 3 BG splashes (ignored)
    # 5: Nonbasic ("", land)    -> opened in 60 WU runs, played in 0 -> land, NEVER flagged
    # 6: Bo3Card  (B,  nonland) -> opened in 50 Bo3 BG runs where game-1 deck does NOT play it (only boarded in game 2 -> 0/50 -> FLAGGED)
    cards = pl.DataFrame({
        "card_id": [0, 1, 2, 3, 4, 5, 6],
        "name": ["RareW", "BorderU", "BelowMin", "Artifact", "GoldWU", "Nonbasic", "Bo3Card"],
        "colors": ["W", "U", "W", "", "WU", "", "B"],
        "is_land": [False, False, False, False, False, True, False],
        "is_basic": [False, False, False, False, False, False, False],
    })
    cards.write_parquet(cdir / "cards.parquet")
    cards.write_parquet(mdir / "cards.parquet")

    games_rows: list[dict] = []
    pool_rows: list[dict] = []
    deck_rows: list[dict] = []
    dk = 1

    # 60 Bo1 WU runs
    for i in range(60):
        did = f"wu_{i:03d}"
        games_rows.append(dict(draft_id=did, fmt="bo1", deck_key=dk, build_index=0, match_number=1, game_number=1, main_colors="WU"))
        # Also add a Draft game with a different draft_id to make sure Draft runs are ignored
        pool_rows.append(dict(draft_id=did, card_id=0, n=1))
        pool_rows.append(dict(draft_id=did, card_id=1, n=1))
        if i < 49:
            pool_rows.append(dict(draft_id=did, card_id=2, n=1))
        pool_rows.append(dict(draft_id=did, card_id=3, n=1))
        pool_rows.append(dict(draft_id=did, card_id=4, n=1))
        pool_rows.append(dict(draft_id=did, card_id=5, n=1))
        if i < 5:
            deck_rows.append(dict(deck_key=dk, card_id=0, n=1))
        if i < 9:
            deck_rows.append(dict(deck_key=dk, card_id=1, n=1))
        if i < 2:
            deck_rows.append(dict(deck_key=dk, card_id=2, n=1))
        if i < 4:
            deck_rows.append(dict(deck_key=dk, card_id=3, n=1))
        if i < 8:
            deck_rows.append(dict(deck_key=dk, card_id=4, n=1))
        dk += 1

    # 10 Bo1 BG runs (off-color splashes for RareW and GoldWU; colorless Artifact IS eligible!)
    for i in range(10):
        did = f"bg_{i:03d}"
        games_rows.append(dict(draft_id=did, fmt="bo1", deck_key=dk, build_index=0, match_number=1, game_number=1, main_colors="BG"))
        pool_rows.append(dict(draft_id=did, card_id=0, n=1))
        pool_rows.append(dict(draft_id=did, card_id=3, n=1))
        pool_rows.append(dict(draft_id=did, card_id=4, n=1))
        # Splashed RareW in all 10 BG runs, splashed GoldWU in 3 BG runs -> must NOT enter top or bottom
        deck_rows.append(dict(deck_key=dk, card_id=0, n=1))
        if i < 3:
            deck_rows.append(dict(deck_key=dk, card_id=4, n=1))
        if i < 2:
            deck_rows.append(dict(deck_key=dk, card_id=3, n=1))
        dk += 1

    # 5 Bo1 mono-W runs: GoldWU (WU) is NOT within main_colors="W", so NOT eligible
    for i in range(5):
        did = f"w_{i:03d}"
        games_rows.append(dict(draft_id=did, fmt="bo1", deck_key=dk, build_index=0, match_number=1, game_number=1, main_colors="W"))
        pool_rows.append(dict(draft_id=did, card_id=4, n=1))
        deck_rows.append(dict(deck_key=dk, card_id=4, n=1))
        dk += 1

    # 50 Bo3 BG runs: game 1 uses dk_g1 (without Bo3Card), games 2-3 use dk_sb (with Bo3Card).
    # Bo3 uses the registered game-1 deck, so Bo3Card must have played=0, eligible=50 (NOT played=50!).
    for i in range(50):
        did = f"bo3_{i:03d}"
        dk_g1, dk_sb = dk, dk + 1
        dk += 2
        games_rows.append(dict(draft_id=did, fmt="bo3", deck_key=dk_g1, build_index=0, match_number=1, game_number=1, main_colors="BG"))
        games_rows.append(dict(draft_id=did, fmt="bo3", deck_key=dk_sb, build_index=1, match_number=1, game_number=2, main_colors="BG"))
        games_rows.append(dict(draft_id=did, fmt="bo3", deck_key=dk_sb, build_index=1, match_number=1, game_number=3, main_colors="BG"))
        pool_rows.append(dict(draft_id=did, card_id=6, n=1))
        deck_rows.append(dict(deck_key=dk_sb, card_id=6, n=1))

    # 10 Draft runs (fmt="draft") + 5 null-main_colors runs -> both must be ignored
    for i in range(10):
        did = f"dr_{i:03d}"
        games_rows.append(dict(draft_id=did, fmt="draft", deck_key=dk, build_index=0, match_number=1, game_number=1, main_colors="WU"))
        pool_rows.append(dict(draft_id=did, card_id=0, n=1))
        deck_rows.append(dict(deck_key=dk, card_id=0, n=1))
        dk += 1
    for i in range(5):
        did = f"null_{i:03d}"
        games_rows.append(dict(draft_id=did, fmt="bo1", deck_key=dk, build_index=0, match_number=1, game_number=1, main_colors=None))
        pool_rows.append(dict(draft_id=did, card_id=0, n=1))
        dk += 1

    pl.DataFrame(games_rows).write_parquet(cdir / "games.parquet")
    pl.DataFrame(pool_rows).write_parquet(cdir / "pools.parquet")
    pl.DataFrame(deck_rows).write_parquet(cdir / "deck_cards.parquet")
    (cdir / "manifest.json").write_text(json.dumps({"built_at": "2026-11-26T00:00:00Z", "tables": {"pools": len(pool_rows)}}), encoding="utf-8")


def test_play_rates_counting_thresholds_and_cache(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("SEALED_LAB_HOME", str(home))
    play_rates.forget()
    _write_synthetic_corpus(home, "SYN")

    df = play_rates.play_rates_df("SYN")
    by_id = {int(cid): (int(e), int(k)) for cid, e, k in df.iter_rows()}

    # 0: RareW -> 60 in-color WU runs, 5 played (10 BG splashes + 10 Draft + 5 null main_colors excluded)
    assert by_id[0] == (60, 5)
    # 1: BorderU -> exactly 9/60 = 15.0%
    assert by_id[1] == (60, 9)
    # 2: BelowMin -> 49 eligible runs (< 50 minimum)
    assert by_id[2] == (49, 2)
    # 3: Artifact (colorless) -> 60 WU + 10 BG = 70 eligible, 4 + 2 = 6 played
    assert by_id[3] == (70, 6)
    # 4: GoldWU -> only 60 WU runs eligible (10 BG splashes & 5 mono-W excluded), 8 played
    assert by_id[4] == (60, 8)
    # 5: Nonbasic land -> excluded completely
    assert 5 not in by_id
    # 6: Bo3Card -> 50 Bo3 runs, game-1 deck never played it (only boarded in G2/G3)
    assert by_id[6] == (50, 0)

    notes = api.card_play_rates("SYN")
    # Flagged strictly when eligible >= 50 and played/eligible < 0.15
    assert set(notes) == {0, 3, 4, 6}
    assert 1 not in notes   # 9/60 == 15.0% -> NOT flagged
    assert 2 not in notes   # 49 eligible < 50 -> NOT flagged
    assert 5 not in notes   # land -> NOT flagged

    # Exact factual wording, no forbidden editorial words
    assert notes[0]["marker"] == "rarely played"
    assert notes[0]["tip"] == "Played in 8% of the 60 Sealed pools that opened it in its colors."
    assert notes[3]["tip"] == "Played in 9% of the 70 Sealed pools that opened it in its colors."
    assert notes[6]["tip"] == "Played in 0% of the 50 Sealed pools that opened it in its colors."
    for v in notes.values():
        low = (v["marker"] + " " + v["tip"]).lower()
        for bad in ("trap", "bad", "avoid", "overrated", "lose", "worse"):
            assert bad not in low

    # Cached next to the model as play_rates.parquet with columns card_id, eligible, played
    cache_p = config.model_dir("SYN") / "play_rates.parquet"
    assert cache_p.exists()
    cached = pl.read_parquet(cache_p)
    assert cached.columns == ["card_id", "eligible", "played"]

    # Second read uses disk cache after clearing memory cache
    play_rates.forget("SYN")
    df2 = play_rates.play_rates_df("SYN")
    assert df2.to_dicts() == df.to_dicts()


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from sealed_lab.gui.widgets import apply_theme

    app = QApplication.instance() or QApplication([])
    apply_theme(app, "light")
    return app


def test_gui_deck_builder_shows_rarely_played_marker_and_tooltip(qapp):
    """DeckBuilder displays the 'rarely played' pill and exact factual tooltip via fake_api without altering values or sort."""
    from sealed_lab.gui.deckbuilder import DeckBuilder

    fake_api.reset()
    info = fake_api.card_info("TST")
    betas = fake_api.card_values("TST", "bo3")
    rare_cid = 3
    assert rare_cid in fake_api.card_play_rates("TST")

    # Build a valid deck & pool that includes the rarely-played card (cid=3) and a normal card (cid=1)
    deck_counts = {cid: 1 for cid in range(23)}
    plains = next(i for i, d in info.items() if d["name"] == "Plains")
    island = next(i for i, d in info.items() if d["name"] == "Island")
    deck_counts[plains] = 9
    deck_counts[island] = 8
    pool_counts = {cid: 1 for cid in range(30)}

    ev = fake_api.evaluate_deck("TST", deck_counts, "bo3")
    b = DeckBuilder(fake_api)
    try:
        b.load("TST", "bo3", ev, pool_counts, info, betas)
        assert rare_cid in b.deck_pile.rarely_played
        assert 1 not in b.deck_pile.rarely_played

        # Marker pill planned on the card
        r = b.deck_pile._rect_of(rare_cid)
        assert r is not None
        chips = b.deck_pile._plan_chips(r, rare_cid)
        assert any(it["text"] == "rarely played" and it["label"] == "rarely played" for it in chips)

        # Normal card has no rarely-played marker
        r_norm = b.deck_pile._rect_of(1)
        assert r_norm is not None
        assert not any(it["label"] == "rarely played" for it in b.deck_pile._plan_chips(r_norm, 1))

        # Tooltip contains the exact factual sentence and no forbidden words
        tip = b.deck_pile.tip_for(rare_cid)
        assert "Played in 9% of the 505 Sealed pools that opened it in its colors." in tip
        for bad in ("trap", "bad", "avoid", "overrated"):
            assert bad not in tip.lower()

        # Sideboard card with rarely_played (cid=23) also shows marker & tooltip
        assert 23 in b.side_pile.rarely_played
        r_sb = b.side_pile._rect_of(23)
        assert r_sb is not None
        assert any(it["label"] == "rarely played" for it in b.side_pile._plan_chips(r_sb, 23))
        assert "Played in 9% of the 505 Sealed pools that opened it in its colors." in b.side_pile.tip_for(23)

        # Layout clipping audit passes cleanly
        assert b.deck_pile.clipped_texts() == []
        assert b.side_pile.clipped_texts() == []

        # Blind mode hides the rarely-played marker & tooltip
        b.load_blind("TST", "bo3", pool_counts, info)
        assert b.side_pile.rarely_played == {}
        assert "Played in" not in b.side_pile.tip_for(rare_cid)
    finally:
        b.shutdown()
        b.close()
