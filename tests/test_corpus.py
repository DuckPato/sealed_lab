from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sealed_lab import api
from sealed_lab.config import BASICS
from sealed_lab.ingest import game_data, registry
from sealed_lab.ingest.corpus import build_corpus, load_corpus, read_manifest
from sealed_lab.synth.generate import generate
from sealed_lab.types import Cancelled

EVENTS = {"PremierDraft": "draft", "TradDraft": "draft", "Sealed": "bo1", "TradSealed": "bo3"}


def _csv_rows(synth, ev):
    return pl.read_csv(synth / f"game_data_public.TST.{ev}.csv.gz", columns=["won"]).height


def test_row_counts_match(built):
    c = load_corpus("TST")
    games = c["games"].collect()
    syn = built["synth"]
    for ev, fmt in EVENTS.items():
        assert games.filter(pl.col("event") == ev).height == _csv_rows(syn, ev)
        assert (games.filter(pl.col("event") == ev)["fmt"] == fmt).all()
    man = built["manifest"]
    assert man["tables"]["games"] == games.height
    assert sum(man["audit"]["n_games_by_fmt"].values()) == games.height
    assert games["game_key"].n_unique() == games.height
    assert games.schema["game_key"] == pl.Int64 and games.schema["won"] == pl.Boolean
    assert games.schema["skill"] == pl.Float32 and games.schema["draft_time"] == pl.Datetime("us")
    assert games["draft_time"].null_count() == 0 and games["won"].null_count() == 0
    # values survive: wins and plays
    raw = pl.read_csv(syn / "game_data_public.TST.TradSealed.csv.gz", columns=["won", "on_play", "num_mulligans"],
                      schema_overrides={"won": pl.String, "on_play": pl.String})
    g = games.filter(pl.col("event") == "TradSealed")
    assert g["won"].sum() == (raw["won"] == "True").sum()
    assert g["on_play"].sum() == (raw["on_play"] == "True").sum()
    assert g["num_mulligans"].sum() == raw["num_mulligans"].sum()


def test_cards_table(built):
    c = load_corpus("TST")
    cards = c["cards"]
    truth = json.loads((built["synth"] / "truth.json").read_text(encoding="utf-8"))
    assert cards["name"].to_list() == sorted(truth["cards"])
    assert cards["card_id"].to_list() == list(range(cards.height))
    assert set(cards["meta_source"]) == {"scryfall"}
    r = {x["name"]: x for x in cards.iter_rows(named=True)}
    for nm, t in truth["cards"].items():  # metadata parsed from the Scryfall-like file equals the planted truth
        row = r[nm]
        assert row["is_land"] == t["is_land"] and row["is_basic"] == t["is_basic"], nm
        assert [row[f"pip_{c}"] for c in "WUBRG"] == t["pips"], nm
        assert row["produces"] == "".join(ch for ch in "WUBRG" if ch in t["produces"]), nm
        assert row["hybrid"] == t["hybrid"], nm
    for nm in ("Aldric, Dawn's Herald", "Jötun Tidecaller", "Searing Rift // Frozen Hold", "Zürn the Smelter"):
        assert nm in r
    assert r["Searing Rift // Frozen Hold"]["front_name"] == "Searing Rift"
    assert r["Moonlit Scout // Howling Pack"]["is_creature"]
    assert r["Moonlit Scout // Howling Pack"]["pip_G"] == 1
    assert r["Plains"]["rarity"] == "basic" and r["Plains"]["is_basic"]


def test_decks_and_pools(built):
    c = load_corpus("TST")
    decks, dc, cards = c["decks"], c["deck_cards"].collect(), c["cards"]
    assert (decks["n_cards"] == 40).all()
    assert decks["n_lands"].min() >= 16 and decks["n_lands"].max() <= 18
    assert decks["deck_hash"].n_unique() == decks.height == decks["deck_key"].n_unique()
    assert dc.group_by("deck_key").agg(pl.col("n").sum()).get_column("n").eq(40).all()
    games = c["games"].collect()
    assert set(games["deck_key"]).issubset(set(decks["deck_key"]))
    builds = c["builds"]
    assert builds["games"].sum() == games.height
    assert builds.group_by(["draft_id", "build_index"]).len()["len"].max() == 1
    # pools: invariant, basics excluded, six boosters
    audit = c["pool_audit"]
    assert audit["invariant"].all() and audit.height > 100
    pools = c["pools"].collect()
    basics = set(cards.filter(pl.col("is_basic"))["card_id"])
    assert not (set(pools["card_id"]) & basics)
    tot = pools.group_by("draft_id").agg(pl.col("n").sum())
    assert (tot["n"] == 84).all()
    assert set(tot["draft_id"]) == set(audit["draft_id"])
    assert set(tot["draft_id"]) <= set(games.filter(pl.col("fmt").is_in(["bo1", "bo3"]))["draft_id"].to_list())
    # no pools for draft formats
    assert not set(pools["draft_id"]) & set(games.filter(pl.col("fmt") == "draft")["draft_id"])
    # pool contains every maindeck non-basic card
    j = dc.join(builds.select("deck_key", "draft_id"), on="deck_key").filter(
        pl.col("draft_id").is_in(tot["draft_id"].head(30).to_list()))
    j = j.join(cards.select("card_id", "is_basic"), on="card_id").filter(~pl.col("is_basic"))
    mx = j.group_by(["draft_id", "card_id"]).agg(pl.col("n").max().alias("deckmax"))
    chk = mx.join(pools, on=["draft_id", "card_id"], how="left")
    assert (chk["n"] >= chk["deckmax"]).all()


def test_hands_consistent(built):
    c = load_corpus("TST")
    hands = c["hands"].collect()
    games = c["games"].collect().select("game_key", "deck_key", "num_mulligans")
    dc = c["deck_cards"].collect()
    h = hands.join(games, on="game_key").join(dc, on=["deck_key", "card_id"], how="left")
    assert h["n"].null_count() == 0  # every card in a hand is in the maindeck
    assert (h["opening"].cast(pl.Int16) + h["drawn"] <= h["n"]).all()
    per = hands.group_by("game_key").agg(pl.col("opening").sum().alias("op")).join(games, on="game_key")
    assert (per["op"] == 7 - per["num_mulligans"]).all()
    assert hands["game_key"].n_unique() == games.height
    assert built["manifest"]["audit"]["hand_violation_rate"] == 0.0


def test_names_with_odd_characters_roundtrip(built):
    c = load_corpus("TST")
    names = set(c["cards"]["name"])
    assert any("," in n for n in names) and any("'" in n for n in names)
    assert any(" // " in n for n in names) and any(any(ord(ch) > 127 for ch in n) for n in names)
    dc = c["deck_cards"].collect().join(c["cards"].select("card_id", "name"), on="card_id")
    used = set(dc["name"])
    assert "Searing Rift // Frozen Hold" in used or "Moonlit Scout // Howling Pack" in used
    assert any("," in n for n in used)


def test_strength_correlates_with_results(built):
    """Builds with higher planted strength win more (end-to-end sanity of keys and joins)."""
    from scipy.stats import spearmanr

    from sealed_lab.synth import generate as G

    truth = json.loads((built["synth"] / "truth.json").read_text(encoding="utf-8"))
    arr = G._arrays_from_truth(truth)
    names = arr["names"]
    c = load_corpus("TST")
    dc = c["deck_cards"].collect().join(c["cards"].select("card_id", "name"), on="card_id")
    keys = dc["deck_key"].unique().sort().to_list()
    pos = {k: i for i, k in enumerate(keys)}
    D = np.zeros((len(keys), len(names)))
    ni = {n: i for i, n in enumerate(names)}
    for k, n, cnt in zip(dc["deck_key"].to_list(), dc["name"].to_list(), dc["n"].to_list()):
        D[pos[k], ni[n]] = cnt
    ts = G.strength_matrix(truth, arr, D)
    b = c["builds"].group_by("deck_key").agg(pl.col("games").sum(), pl.col("wins").sum()).filter(pl.col("games") >= 8)
    st = np.array([ts[pos[k]] for k in b["deck_key"]])
    wr = (b["wins"] / b["games"]).to_numpy()
    assert spearmanr(st, wr)[0] > 0.15


def test_manifest(built):
    m = read_manifest("TST")
    assert m["set"] == "TST" and m["tables"]["cards"] == 156
    assert {f["kind"] for f in m["files"]} >= {"game", "draft", "cards", "scryfall"}
    assert all(f["status"] == "ingested" for f in m["files"] if f["kind"] == "game")
    assert m["audit"]["pool_invariance_rate"]["bo3"] == 1.0
    assert m["audit"]["deck_size_hist"] == {"40": sum(m["audit"]["deck_size_hist"].values())}
    assert m["card_meta_sources"] == {"scryfall": 156}
    assert not m["warnings"]
    assert Path(built["home"], "sets", "TST", "corpus", "cards.parquet").exists()


def test_chunking_and_plain_csv_equivalence(synth_dir, tmp_path):
    src = synth_dir / "game_data_public.TST.TradSealed.csv.gz"
    from sealed_lab.ingest.detect import detect_file, open_binary

    plain = tmp_path / "plain" / "game_data_public.TST.TradSealed.csv"
    plain.parent.mkdir()
    plain.write_bytes(open_binary(src).read())
    a = tmp_path / "a"
    b = tmp_path / "b"
    ia, ib = detect_file(src), detect_file(plain)
    ma = game_data.process_game_file(str(src), ia.fingerprint, "TradSealed", "TST", a)
    mb = game_data.process_game_file(str(plain), ib.fingerprint, "TradSealed", "TST", b, block_bytes=64 * 1024)
    assert ma["rows"] == mb["rows"] > 0
    assert ma["names"] == mb["names"]
    for t in ("deck_cards", "pools", "pool_audit", "hands", "builds"):
        da = pl.read_parquet(a / "_shards" / ia.fingerprint / f"{t}.parquet").drop("game_key", strict=False)
        db = pl.read_parquet(b / "_shards" / ib.fingerprint / f"{t}.parquet").drop("game_key", strict=False)
        assert da.height == db.height > 0, t
        assert da.sort(da.columns).equals(db.sort(db.columns)), t
    ga = pl.read_parquet(a / "_shards" / ia.fingerprint / "games.parquet").drop("src", "game_key")
    gb = pl.read_parquet(b / "_shards" / ib.fingerprint / "games.parquet").drop("src", "game_key")
    assert ga.equals(gb)


def test_incremental_rebuild_skips_unchanged(synth_dir, home, tmp_path):
    api.add_files([str(synth_dir)])
    m1 = build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    assert m1["stats"] == {"files_ingested": 4, "files_cached": 0} and not m1["up_to_date"]
    shards = Path(home, "sets", "TST", "corpus", "_shards")
    mt = {p.name: (p / "meta.json").stat().st_mtime_ns for p in shards.iterdir()}
    # nothing changed -> quick path
    t = time.time()
    m2 = build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    assert m2["up_to_date"] and m2["stats"]["files_ingested"] == 0
    assert all(f["status"] == "cached" for f in m2["files"] if f["kind"] == "game")
    assert time.time() - t < 3
    # a new, newer TradSealed file replaces the old one: only it is ingested
    d2 = tmp_path / "second"
    generate(d2, scale=0.05, seed=5, events=("TradSealed",))
    new = d2 / "game_data_public.TST.TradSealed.csv.gz"
    os.utime(new, (time.time() + 100, time.time() + 100))
    infos = api.add_files([str(new)])
    assert "replaces" in infos[0].note
    m3 = build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    st = {f["event"]: f["status"] for f in m3["files"] if f["kind"] == "game"}
    assert st == {"PremierDraft": "cached", "Sealed": "cached", "TradDraft": "cached", "TradSealed": "ingested"}
    assert m3["stats"] == {"files_ingested": 1, "files_cached": 3}
    now = {p.name: (p / "meta.json").stat().st_mtime_ns for p in shards.iterdir()}
    kept = [k for k in now if k in mt]
    assert len(kept) == 3 and all(now[k] == mt[k] for k in kept)  # untouched shards
    assert len(now) == 4  # replaced file's shard garbage-collected
    games = load_corpus("TST")["games"].collect()
    assert games.filter(pl.col("event") == "TradSealed").height == _csv_rows(d2, "TradSealed")
    # removing a file drops its rows
    fp = next(f.fingerprint for f in registry.files("TST") if f.event == "Sealed" and f.kind == "game")
    api.remove_file("TST", fp)
    m4 = build_corpus("TST", n_jobs=1, fetch_scryfall=False)
    assert "bo1" not in m4["audit"]["n_games_by_fmt"] and m4["stats"]["files_ingested"] == 0
    # status reflects staleness
    st = api.set_status("TST")
    assert st.has_corpus and not st.has_model and st.n_games.keys() == {"draft", "bo3"}


def test_fallback_cards_without_scryfall(synth_dir, home, tmp_path):
    # only 17lands cards.csv
    sub = tmp_path / "s1"
    sub.mkdir()
    shutil.copy(synth_dir / "game_data_public.TST.Sealed.csv.gz", sub)
    shutil.copy(synth_dir / "game_data_public.TST.TradSealed.csv.gz", sub)
    shutil.copy(synth_dir / "cards.csv", sub)
    api.add_files([str(sub)])
    m = build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    assert m["card_meta_sources"] == {"17lands": 156}
    assert any("17lands cards.csv fallback" in w for w in m["warnings"])
    cards = load_corpus("TST")["cards"]
    truth = json.loads((synth_dir / "truth.json").read_text(encoding="utf-8"))
    r = {x["name"]: x for x in cards.iter_rows(named=True)}
    ok = sum(r[n]["colors"] == t["colors"] or r[n]["colors"] == "".join(c for c in "WUBRG" if c in t["colors"])
             for n, t in truth["cards"].items() if not t["is_land"])
    assert ok / sum(1 for t in truth["cards"].values() if not t["is_land"]) > 0.95
    assert all(r[n]["is_land"] for n, t in truth["cards"].items() if t["is_land"])


def test_fallback_inferred_only(synth_dir, home, tmp_path):
    sub = tmp_path / "s2"
    sub.mkdir()
    shutil.copy(synth_dir / "game_data_public.TST.PremierDraft.csv.gz", sub)
    api.add_files([str(sub)])
    m = build_corpus("TST", n_jobs=1, fetch_scryfall=False)
    assert m["card_meta_sources"].get("inferred") == 156
    assert any("inferred" in w for w in m["warnings"])
    cards = load_corpus("TST")["cards"]
    truth = json.loads((synth_dir / "truth.json").read_text(encoding="utf-8"))
    r = {x["name"]: x for x in cards.iter_rows(named=True)}
    assert all(r[b]["is_land"] and r[b]["is_basic"] and r[b]["produces"] == c for b, c in BASICS.items() if b in r and len(c) == 1)
    mono = {n: t for n, t in truth["cards"].items() if len(t["colors"]) == 1 and not t["is_land"]}
    right = sum(r[n]["colors"] == t["colors"] for n, t in mono.items())
    assert right / len(mono) > 0.9
    colorless = {n for n, t in truth["cards"].items() if t["colors"] == "" and not t["is_land"] and not t["is_basic"]}
    assert sum(r[n]["colors"] == "" for n in colorless) >= len(colorless) - 1


def test_cancel_before_start_and_missing_game_files(synth_dir, home):
    with pytest.raises(ValueError):
        build_corpus("TST")
    api.add_files([str(synth_dir / "game_data_public.TST.TradSealed.csv.gz")])
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        build_corpus("TST", cancel=ev, n_jobs=1, fetch_scryfall=False)
    # not cancelled: works, progress is monotone and ends at 1
    seen = []
    build_corpus("TST", progress=lambda f, m: seen.append(f), n_jobs=1, fetch_scryfall=False)
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_cancel_during_ingest(synth_dir, home):
    api.add_files([str(synth_dir / "game_data_public.TST.PremierDraft.csv.gz")])
    ev = threading.Event()
    threading.Timer(0.05, ev.set).start()
    with pytest.raises(Cancelled):
        build_corpus("TST", cancel=ev, n_jobs=1, fetch_scryfall=False)
    # a later run succeeds
    assert build_corpus("TST", n_jobs=1, fetch_scryfall=False)["tables"]["games"] > 1000


def test_scryfall_fetch_mocked_through_build(synth_dir, home, tmp_path):
    import json as _json

    from tests.test_cards_scryfall import FakeSession  # noqa: F401  (same fake HTTP session)

    sc = _json.loads((synth_dir / "scryfall_TST.json").read_text(encoding="utf-8"))
    sub = tmp_path / "s3"
    sub.mkdir()
    shutil.copy(synth_dir / "game_data_public.TST.TradSealed.csv.gz", sub)
    api.add_files([str(sub)])
    m = build_corpus("TST", n_jobs=1, fetch_scryfall=True, session=FakeSession(sc, per_page=50))
    assert m["card_meta_sources"] == {"scryfall": 156}
    assert m["scryfall"]["complete"]
    # fully cached afterwards: quick path even with fetch enabled
    assert build_corpus("TST", n_jobs=1, fetch_scryfall=True, session=FakeSession([], fail=RuntimeError("x")))["up_to_date"]


def test_tar_inside_gzip_old_header_and_missing_result(synth_dir, tmp_path):
    """Older 17lands downloads (AFR) are a .tar inside the .csv.gz, call the skill bucket "user_win_rate_bucket", and can carry a row
    without a result: the same games come out as from the plain file, the row without a result is dropped."""
    import gzip
    import io
    import tarfile

    from sealed_lab.ingest.detect import detect_file, open_binary, read_head_rows
    from sealed_lab.ingest import replay

    src = synth_dir / "game_data_public.TST.TradSealed.csv.gz"
    data = open_binary(src).read()
    header, rest = data.split(b"\n", 1)
    assert b"user_game_win_rate_bucket" in header
    n_cols = header.count(b",") + 1
    old = header.replace(b"user_game_win_rate_bucket", b"user_win_rate_bucket") + b"\n" + rest + b"," * (n_cols - 1) + b"\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("game_data_public.TST.TradSealed.csv")
        info.size = len(old)
        tf.addfile(info, io.BytesIO(old))
    wrapped = tmp_path / "tar" / "game_data_public.TST.TradSealed.csv.gz"
    wrapped.parent.mkdir()
    wrapped.write_bytes(gzip.compress(buf.getvalue()))
    assert read_head_rows(wrapped)[0][0] == header.decode().split(",")[0]
    ia, ib = detect_file(src), detect_file(wrapped)
    ma = game_data.process_game_file(str(src), ia.fingerprint, "TradSealed", "TST", tmp_path / "a")
    mb = game_data.process_game_file(str(wrapped), ib.fingerprint, "TradSealed", "TST", tmp_path / "b")
    assert ma["rows"] == mb["rows"] > 0
    ga = pl.read_parquet(tmp_path / "a" / "_shards" / ia.fingerprint / "games.parquet").drop("src", "game_key")
    gb = pl.read_parquet(tmp_path / "b" / "_shards" / ib.fingerprint / "games.parquet").drop("src", "game_key")
    assert ga.equals(gb) and ga["skill"].null_count() < ga.height

    rdata = b"draft_id,user_win_rate_bucket,won\nd1,0.6,True\n"
    rbuf = io.BytesIO()
    with tarfile.open(fileobj=rbuf, mode="w") as tf:
        info = tarfile.TarInfo("replay_data_public.TST.TradSealed.csv")
        info.size = len(rdata)
        tf.addfile(info, io.BytesIO(rdata))
    rw = tmp_path / "tar" / "replay_data_public.TST.TradSealed.csv.gz"
    rw.write_bytes(gzip.compress(rbuf.getvalue()))
    assert replay._read_header(str(rw)) == (["draft_id", "user_game_win_rate_bucket", "won"], 0, True)


def test_card_counts_written_as_floats(synth_dir, tmp_path):
    """Some files (EOE) write card counts as "0.0" / "1.0": they are read like integer counts."""
    from sealed_lab.ingest.detect import detect_file, open_binary

    src = synth_dir / "game_data_public.TST.TradSealed.csv.gz"
    df = pl.read_csv(open_binary(src).read(), infer_schema_length=0)
    counts = [c for c in df.columns if c.startswith(("deck_", "opening_hand_", "drawn_", "sideboard_"))]
    df = df.with_columns(pl.col(c).cast(pl.Float64).cast(pl.String) for c in counts)
    assert df[counts[0]].str.contains(r"\.").all()
    floats = tmp_path / "f" / "game_data_public.TST.TradSealed.csv"
    floats.parent.mkdir()
    df.write_csv(floats)
    ia, ib = detect_file(src), detect_file(floats)
    ma = game_data.process_game_file(str(src), ia.fingerprint, "TradSealed", "TST", tmp_path / "a")
    mb = game_data.process_game_file(str(floats), ib.fingerprint, "TradSealed", "TST", tmp_path / "b")
    assert ma["rows"] == mb["rows"] > 0
    for t in ("deck_cards", "hands"):
        da = pl.read_parquet(tmp_path / "a" / "_shards" / ia.fingerprint / f"{t}.parquet").drop("game_key", strict=False)
        db = pl.read_parquet(tmp_path / "b" / "_shards" / ib.fingerprint / f"{t}.parquet").drop("game_key", strict=False)
        assert da.sort(da.columns).equals(db.sort(db.columns)), t
