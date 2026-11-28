"""Draft target formats `dbo1` / `dbo3` (docs/draft_mode/RULE.md): ingest of draft pools, the switch of `pipeline.calculate`, the Sealed
chain unchanged by it, the builder in the draft formats, and an old corpus (no draft pools) that still loads and builds.

The shared synthetic set (`calc`: PremierDraft, TradDraft, Sealed and TradSealed games) is copied to a second home, which is then
recalculated with the switch on (4 Sealed draws, 3 draft draws)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import polars as pl
import pytest

from sealed_lab import api, config
from sealed_lab.deck import builder, samplehand
from sealed_lab.deck.builder import build_ids
from sealed_lab.ingest.corpus import load_corpus
from sealed_lab.model import draftfmt, pipeline
from sealed_lab.model.artifact import SetModel

BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")
DRAFT_BOOT = 3


def _recalc(src_home, dst_home, monkeypatch_ctx, **kw) -> dict:
    shutil.copytree(src_home, dst_home)                       # a COPY (never a link): the shared home stays as `calc` left it
    with monkeypatch_ctx() as mp:
        mp.setenv("SEALED_LAB_HOME", str(dst_home))
        return pipeline.calculate("TST", n_jobs=2, n_boot=4, **kw)


@pytest.fixture(scope="session")
def draft_calc(calc, tmp_path_factory):
    home = tmp_path_factory.mktemp("draft_home") / "h"
    meta = _recalc(calc["home"], home, pytest.MonkeyPatch.context, draft_formats=True, draft_boot=DRAFT_BOOT)
    return dict(home=home, meta=meta)


@pytest.fixture()
def in_draft_home(draft_calc, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_HOME", str(draft_calc["home"]))
    return draft_calc["home"]


@pytest.fixture()
def dmodel(in_draft_home):
    return SetModel.load("TST")


def _draws(home) -> dict:
    with np.load(home / "sets" / "TST" / "model" / "draws.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def held(calc):
    return json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------ ingest: draft pools

def test_draft_pools_are_stored_and_sealed_pool_rows_are_unchanged(built):
    c = load_corpus("TST")
    games = c["games"].collect()
    pools, audit = c["pools"].collect(), c["pool_audit"]
    basics = set(c["cards"].filter(pl.col("is_basic"))["card_id"])
    assert not (set(pools["card_id"]) & basics)                                   # basics excluded, as for Sealed
    draft_ids = set(games.filter(pl.col("fmt") == "draft")["draft_id"])
    sealed_ids = set(games.filter(pl.col("fmt").is_in(["bo1", "bo3"]))["draft_id"])
    got = set(pools["draft_id"])
    assert got & sealed_ids == sealed_ids                                          # every Sealed event keeps its pool
    assert got & draft_ids                                                         # draft events have pools now
    assert got <= sealed_ids | draft_ids
    assert set(audit["draft_id"]) == got and audit["invariant"].all()              # pool_audit covers them
    # the draft pool of a build holds every non-basic maindeck card (deck + sideboard, the same max-over-games rule)
    dc, builds = c["deck_cards"].collect(), c["builds"]
    some = sorted(draft_ids & got)[:30]
    j = dc.join(builds.select("deck_key", "draft_id"), on="deck_key").filter(pl.col("draft_id").is_in(some))
    j = j.join(c["cards"].select("card_id", "is_basic"), on="card_id").filter(~pl.col("is_basic"))
    mx = j.group_by(["draft_id", "card_id"]).agg(pl.col("n").max().alias("deckmax")).join(pools, on=["draft_id", "card_id"], how="left")
    assert (mx["n"] >= mx["deckmax"]).all()
    # the audit keeps the Sealed lines as they were and reports the draft pools apart
    a = built["manifest"]["audit"]
    assert set(a["pool_invariance_rate"]) <= {"bo1", "bo3"}
    assert a["draft_pool_events"] and all(v == 1.0 for v in a["draft_pool_invariance_rate"].values())


def test_old_draft_shard_without_pools_is_reingested_only_while_its_source_exists(tmp_path):
    from sealed_lab.ingest import game_data

    cd = tmp_path
    fp = "abc123"
    d = game_data.shard_path(cd, fp)
    d.mkdir(parents=True)
    src = tmp_path / "src.csv"
    src.write_text("x")
    meta = dict(version=game_data.SHARD_VERSION, ok=True, fmt="draft", path=str(src))
    (d / "meta.json").write_text(json.dumps(meta))
    assert game_data.shard_valid(cd, fp) is None                                   # source there: rebuilt with draft pools
    src.unlink()
    assert game_data.shard_valid(cd, fp) is not None                               # source gone: the cached shard is all there is
    (d / "meta.json").write_text(json.dumps(dict(meta, draft_pools_v=1)))
    src.write_text("x")
    assert game_data.shard_valid(cd, fp) is not None                               # current draft shard
    (d / "meta.json").write_text(json.dumps(dict(meta, fmt="bo1")))
    assert game_data.shard_valid(cd, fp) is not None                               # Sealed shards are untouched by the change


# ------------------------------------------------------------------------------------------------------ the helpers

def test_relabel_and_present_leave_the_input_alone():
    g = pl.DataFrame(dict(event=["PremierDraft", "TradDraft", "PickTwoDraft", "Sealed", "TradSealed"],
                          fmt=["draft", "draft", "draft", "bo1", "bo3"]))
    assert draftfmt.relabel(g)["fmt"].to_list() == ["bo1", "bo3", "draft"]
    assert draftfmt.relabel(g, named=True)["fmt"].to_list() == ["dbo1", "dbo3", "draft"]
    assert g["fmt"].to_list() == ["draft", "draft", "draft", "bo1", "bo3"]
    assert draftfmt.present(g) == ["dbo3", "dbo1"]
    assert draftfmt.present(g.filter(pl.col("event") != "TradDraft")) == ["dbo1"]
    assert draftfmt.present(g.filter(pl.col("fmt") != "draft")) == []


def test_format_helpers():
    assert config.format_regime("dbo1") == "hand" and config.format_regime("dbo3") == "random"
    assert config.format_regime("bo1") == "sealed" and config.format_regime("bo3") == "random"
    assert config.is_draft_format("dbo1") and not config.is_draft_format("bo1") and not config.is_draft_format("draft")
    assert config.is_bo3_format("bo3") and config.is_bo3_format("dbo3") and not config.is_bo3_format("dbo1")
    assert config.RUN_RULES["dbo1"] == config.RUN_RULES["bo1"] and config.RUN_RULES["dbo3"] == config.RUN_RULES["bo3"]


# ------------------------------------------------------------------------------------------------------ the switch

def test_switch_off_has_no_draft_formats(calc):
    assert calc["meta"]["formats"] == ["bo3", "bo1"] and "draft_formats" not in calc["meta"]
    assert not any(k.startswith("dbo") for k in calc["meta"]["n_games"])


def test_switch_on_adds_the_draft_formats(draft_calc, dmodel):
    meta = draft_calc["meta"]
    assert meta["formats"] == ["bo3", "bo1", "dbo3", "dbo1"] and dmodel.formats == meta["formats"]
    assert meta["n_draws"] == 5 and meta["draft_formats"]["n_draws"] == DRAFT_BOOT + 1
    for f in ("dbo1", "dbo3"):
        assert dmodel.n_draws(f) == DRAFT_BOOT + 1 and dmodel.n_draws("bo1") == 5
        assert dmodel._draws[f]["beta"].shape == dmodel._draws["bo1"]["beta"].shape[:0] + (DRAFT_BOOT + 1, dmodel._draws["bo1"]["beta"].shape[1])
        assert dmodel._draws[f]["theta"].shape == (DRAFT_BOOT + 1, dmodel._draws["bo1"]["theta"].shape[1])
        assert "mull" in dmodel._draws[f] and "pair" not in dmodel._draws[f] and "arch" not in dmodel._draws[f]
        assert meta["n_games"][f] > 0 and meta["stages"][f]["n_games"] == meta["n_games"][f]
        assert meta["validation"][f]["status"] == "ok" and meta["ref"][f]["ref"] is not None
        assert meta["manasim"]["regimes"][f] == config.format_regime(f)
        assert f in meta["castability"]["support"] and f in meta["land_support"]
        ref = dmodel._ref[f]
        assert ref.shape[0] == DRAFT_BOOT + 1 and (np.diff(ref, axis=1) >= 0).all()
        cards = dmodel.cards.sort("card_id")
        assert cards[f"exposure_{f}"].sum() > 0 and cards[f"se_{f}"].null_count() == 0
    assert meta["manasim"]["regimes"]["dbo1"] == "hand" and meta["manasim"]["regimes"]["dbo3"] == "random"
    assert meta["draft_formats"]["formats"] == ["dbo3", "dbo1"]
    # the dbo1 chain: its prior is the stage of the other draft events (none in this set: zero prior) and dbo3's prior is dbo1
    assert meta["stages"]["dbo3"]["lam_from"] == "dbo1" and meta["stages"]["dbo1"]["lam_from"] in (None, "dprior")


def test_sealed_chain_is_untouched_by_the_switch(calc, draft_calc):
    off, on = _draws(calc["home"]), _draws(draft_calc["home"])
    for f in ("bo3", "bo1"):
        for part in ("beta", "theta", "ref", "mull"):
            assert np.array_equal(off[f"{f}__{part}"], on[f"{f}__{part}"]), (f, part)
    with np.load(calc["home"] / "sets" / "TST" / "model" / "reference.npz") as a, \
            np.load(draft_calc["home"] / "sets" / "TST" / "model" / "reference.npz") as b:
        for f in ("bo3", "bo1"):
            assert np.array_equal(a[f"{f}__q"], b[f"{f}__q"])
    a0 = json.loads((calc["home"] / "sets" / "TST" / "model" / "meta.json").read_text(encoding="utf-8"))
    b0 = draft_calc["meta"]
    for key in ("feature_names", "phi_mean", "phi_sd", "prior_strength", "recency", "deployed", "n_boot", "n_draws"):
        if key == "prior_strength":
            assert {n: v for n, v in b0[key]["stages"].items() if n in ("draft", "bo1", "bo3")} == a0[key]["stages"]
        else:
            assert b0[key] == a0[key], key
    assert {n: s for n, s in b0["stages"].items() if n in ("draft", "bo1", "bo3")} == a0["stages"]
    assert {f: b0["validation"][f] for f in ("bo3", "bo1")} == {f: a0["validation"][f] for f in ("bo3", "bo1")}


def test_sealed_logits_agree_with_and_without_the_switch(calc, draft_calc, held, monkeypatch):
    # SetModel.logit of the same decks under both homes
    with monkeypatch.context() as mp:
        mp.setenv("SEALED_LAB_HOME", str(calc["home"]))
        m0 = SetModel.load("TST")
    with monkeypatch.context() as mp:
        mp.setenv("SEALED_LAB_HOME", str(draft_calc["home"]))
        m1 = SetModel.load("TST")
    names = m0.cards.sort("card_id")["name"].to_list()
    n2i = {n: i for i, n in enumerate(names)}
    decks = [{n2i[n]: c for n, c in h["player_deck"].items()} for h in held[:6]]
    assert decks
    for d in decks:
        for f in ("bo3", "bo1"):
            assert abs(m0.logit(d, f) - m1.logit(d, f)) <= 1e-9
            assert np.array_equal(m0.logit(d, f, draws=True), m1.logit(d, f, draws=True))


# ------------------------------------------------------------------------------------------------------ the builder in the draft formats

def _pool(h, name2id):
    return {name2id[n]: c for n, c in h["pool"].items() if n not in BASICS}


@pytest.mark.parametrize("fmt", ["dbo1", "dbo3"])
def test_builder_builds_a_deck_in_each_draft_format(dmodel, held, fmt):
    n2i = {n: i for i, n in enumerate(dmodel.cards.sort("card_id")["name"].to_list())}
    res = builder.build(dmodel, _pool(held[0], n2i), fmt, n_decks=1, families=False)
    deck = res.decks[0]
    assert sum(c.count for c in deck.cards) == 40 and res.fmt == fmt
    twin = builder.build(dmodel, _pool(held[0], n2i), "bo3" if fmt == "dbo3" else "bo1", n_decks=1, families=False).decks[0]
    assert 0.05 < deck.game_wr < 0.99 and abs(deck.game_wr - twin.game_wr) < 0.25, (fmt, deck.game_wr, twin.game_wr)
    fam = builder.build(dmodel, _pool(held[3], n2i), fmt, n_decks=3, n_jobs=1)              # the family search under the draft draws
    assert fam.decks and len(fam.p_best) == len(fam.decks)
    assert (deck.match_wr is not None) == (fmt == "dbo3")
    # the safe ranges come from the format's own decks (draft events), the land counts too
    assert builder.supported_lands(dmodel, fmt)
    cs = builder.creature_support(dmodel, fmt)
    assert cs is not None and cs[0] <= cs[1]
    ev = dmodel.evaluate({c.card_id: c.count for c in deck.cards if c.card_id is not None}, fmt)
    assert abs(ev.game_wr - deck.game_wr) < 1e-9
    assert dmodel.sim_spec(fmt)[1] is not None or dmodel.sim_version == 1


def test_builder_by_name_text_and_ids_accept_the_draft_formats(in_draft_home, dmodel, held):
    n2i = {n: i for i, n in enumerate(dmodel.cards.sort("card_id")["name"].to_list())}
    out = build_ids("TST", _pool(held[1], n2i), fmt="dbo1", n_decks=1)
    assert out.fmt == "dbo1" and out.decks
    text = "\n".join(f"{c} {n}" for n, c in held[1]["pool"].items())
    res = api.build_pool("TST", text, "dbo3", n_decks=1)
    assert res.fmt == "dbo3" and res.decks
    with pytest.raises(ValueError):
        api.build_pool("TST", text, "nope", n_decks=1)


def test_sample_hands_deal_the_draft_queues_like_their_sealed_twins():
    import random

    from sealed_lab.deck.samplehand import SCard, shuffle_deal
    cards = [SCard(1, "Land", True)] * 17 + [SCard(2, "Spell", False, 2, "1W")] * 23
    rng = random.Random(0)
    h1 = [sum(c.is_land for c in shuffle_deal(cards, rng, "dbo1")[0]) for _ in range(400)]
    rng = random.Random(0)
    h3 = [sum(c.is_land for c in shuffle_deal(cards, rng, "dbo3")[0]) for _ in range(400)]
    assert np.var(h1) < np.var(h3)                       # the smoother narrows the land count of Premier hands, Trad is a plain shuffle
    assert samplehand.format_regime("dbo1") == "hand"


def test_the_set_report_is_sealed_only(in_draft_home):
    from sealed_lab import report

    with pytest.raises(ValueError, match="format must be one of"):          # (docs/draft_mode/RULE.md: no Report sections in v1)
        report.get_report("TST", "dbo1")


def test_other_sets_cross_prior_ignores_draft_formats(draft_calc, calc):
    from sealed_lab.model import crossset

    a = crossset.other_set_effects("ZZZ", ["x"], home=calc["home"])
    b = crossset.other_set_effects("ZZZ", ["x"], home=draft_calc["home"])
    assert [e["formats"] for e in b] == [e["formats"] for e in a] == [["bo3", "bo1"]]


# ------------------------------------------------------------------------------------------------------ an old corpus

def test_old_corpus_without_draft_pools_loads_and_builds(calc, held, tmp_path_factory):
    home = tmp_path_factory.mktemp("old_home") / "h"
    shutil.copytree(calc["home"], home)
    cd = home / "sets" / "TST" / "corpus"
    games = pl.read_parquet(cd / "games.parquet", columns=["fmt", "draft_id"])
    sealed_ids = games.filter(pl.col("fmt").is_in(["bo1", "bo3"]))["draft_id"].unique()
    for t in ("pools", "pool_audit"):                         # what a corpus built before draft pools looked like
        df = pl.read_parquet(cd / f"{t}.parquet")
        df.filter(pl.col("draft_id").is_in(sealed_ids.implode())).write_parquet(cd / f"{t}.parquet")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SEALED_LAB_HOME", str(home))
        c = load_corpus("TST")
        assert set(c["pools"].collect()["draft_id"]) <= set(sealed_ids)
        meta = pipeline.calculate("TST", n_jobs=2, n_boot=2, draft_formats=True, draft_boot=2)
        assert meta["formats"] == ["bo3", "bo1", "dbo3", "dbo1"]
        m = SetModel.load("TST")
        n2i = {n: i for i, n in enumerate(m.cards.sort("card_id")["name"].to_list())}
        for f in ("dbo1", "dbo3", "bo1"):
            assert builder.build(m, _pool(held[2], n2i), f, n_decks=1, families=False).decks
        from sealed_lab import report
        assert report._sealed_pool_count("TST") == len(set(sealed_ids))


def test_a_set_without_traddraft_games_has_no_dbo3(calc):
    g = pl.read_parquet(config.corpus_dir("TST") / "games.parquet", columns=["fmt", "event"])
    assert draftfmt.present(g) == ["dbo3", "dbo1"]
    assert draftfmt.present(g.filter(pl.col("event") != "TradDraft")) == ["dbo1"]
