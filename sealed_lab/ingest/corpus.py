"""Corpus builder: registered files -> parquet tables in `sets/<SET>/corpus/` (incremental, parallel per file)."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import multiprocessing
import os
import queue as queue_mod
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from sealed_lab import CODE_VERSION
from sealed_lab.config import BASICS
from sealed_lab.config import corpus_dir as _corpus_dir
from sealed_lab.config import ensure_set_dirs
from sealed_lab.ingest import cards as cards_mod
from sealed_lab.ingest import game_data, registry, scryfall
from sealed_lab.ingest.detect import detect_file
from sealed_lab.types import Cancelled, FileInfo

log = logging.getLogger(__name__)

TABLES = ["cards", "games", "decks", "deck_cards", "builds", "pools", "pool_audit", "hands"]
LAZY_TABLES = {"games", "deck_cards", "hands", "pools"}

# ---------------------------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------------------------
_Q = None
_CANCEL = None


def _init_worker(q, cancel_ev) -> None:
    global _Q, _CANCEL
    from sealed_lab.procguard import watch_parent

    watch_parent()
    _Q, _CANCEL = q, cancel_ev


def ingest_worker(task: dict) -> dict:
    """Top-level (picklable) worker: ingest one game file into a shard."""
    os.environ["POLARS_MAX_THREADS"] = str(max(1, int(task.get("threads", 1))))
    fp = task["fingerprint"]

    def prog(frac: float) -> None:
        if _Q is not None:
            try:
                _Q.put((fp, float(frac)))
            except Exception:  # noqa: BLE001
                pass

    def cancelled() -> bool:
        return _CANCEL is not None and _CANCEL.is_set()

    return game_data.process_game_file(task["path"], fp, task["event"], task["set_code"], task["corpus_dir"],
                                       progress=prog, cancelled=cancelled)


def run_ingest_tasks(tasks: list[dict], n_jobs: int, progress: Callable[[float, str], None] | None,
                     cancel: threading.Event | None) -> dict[str, dict]:
    """Run ingest tasks in a spawn ProcessPoolExecutor. Returns {fingerprint: shard meta}."""
    if not tasks:
        return {}
    ctx = multiprocessing.get_context("spawn")
    n_workers = max(1, min(len(tasks), n_jobs))
    threads = max(1, n_jobs // n_workers)
    for t in tasks:
        t["threads"] = threads
    tasks = sorted(tasks, key=lambda t: -t.get("size", 0))
    q = ctx.Queue()
    cancel_ev = ctx.Event()
    sizes = {t["fingerprint"]: max(t.get("size", 1), 1) for t in tasks}
    total = sum(sizes.values())
    frac = {fp: 0.0 for fp in sizes}
    out: dict[str, dict] = {}
    prev_threads = os.environ.get("POLARS_MAX_THREADS")
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    ex = cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_worker,
                                initargs=(q, cancel_ev))
    try:
        futs = {ex.submit(ingest_worker, t): t for t in tasks}
        pending = set(futs)
        while pending:
            if cancel is not None and cancel.is_set():
                cancel_ev.set()
                for f in pending:
                    f.cancel()
            try:
                while True:
                    fp, fr = q.get_nowait()
                    frac[fp] = max(frac.get(fp, 0.0), fr)
            except queue_mod.Empty:
                pass
            if progress:
                done_frac = sum(frac[f] * sizes[f] for f in sizes) / total
                progress(done_frac, f"Ingesting {len(tasks)} file(s)")
            done, pending = cf.wait(pending, timeout=0.2, return_when=cf.FIRST_COMPLETED)
            for f in done:
                if f.cancelled():
                    continue
                t = futs[f]
                meta = f.result()  # re-raises worker exceptions (incl. Cancelled)
                out[t["fingerprint"]] = meta
                frac[t["fingerprint"]] = 1.0
        if cancel is not None and cancel.is_set():
            raise Cancelled()
    except BaseException:
        cancel_ev.set()
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        ex.shutdown(wait=True)
    finally:
        if prev_threads is None:
            os.environ.pop("POLARS_MAX_THREADS", None)
        else:
            os.environ["POLARS_MAX_THREADS"] = prev_threads
        q.close()
    return out


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def _stage(df: pl.DataFrame, path: Path) -> Path:
    """Write `df` next to `path` under a staging name (the live table is untouched until `_commit_tables`)."""
    staged = path.with_name(path.name + ".stage")
    df.write_parquet(staged, compression="zstd")
    return staged


def _commit_tables(staged: dict[Path, Path], manifest: Path) -> None:
    """Replace the live tables by their staged versions, all together.  The old manifest goes first (no manifest = the tables are
    not trusted: `tables_ok` is false and the quick path does not apply), so a crash half way is detected; the new manifest, written
    by the caller afterwards, is the commit marker."""
    manifest.unlink(missing_ok=True)
    for live, new in staged.items():
        os.replace(new, live)


def manifest_path(set_code: str) -> Path:
    return _corpus_dir(set_code) / "manifest.json"


def read_manifest(set_code: str) -> dict | None:
    p = manifest_path(set_code)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stat_sig(path: Path) -> str:
    try:
        st = path.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return "none"


def _signature(set_code: str, game_fps: list[str], src_fps: list[str], fetch: bool) -> str:
    cache = scryfall.cache_path(set_code)
    payload = json.dumps([game_data.SHARD_VERSION, CODE_VERSION, cards_mod.CARDS_VERSION, sorted(game_fps), sorted(src_fps),
                          _stat_sig(cache), fetch])
    return hashlib.sha1(payload.encode()).hexdigest()


def _refresh(f: FileInfo) -> FileInfo:
    """Re-detect a registered file if it changed on disk since registration."""
    p = Path(f.path)
    if not p.exists():
        return f
    try:
        from sealed_lab.ingest.detect import fingerprint

        if fingerprint(p) == f.fingerprint:
            return f
    except OSError:
        return f
    info = detect_file(p)
    if info.kind == f.kind and info.set_code and info.set_code == f.set_code:
        registry.register(info)
        return info
    return f


# ---------------------------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------------------------

def build_corpus(
    set_code: str,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
    n_jobs: int | None = None,
    fetch_scryfall: bool = True,
    session=None,
) -> dict:
    """Build/refresh the corpus tables for `set_code`. Incremental per source file. Returns the manifest dict."""
    t_start = time.time()
    set_code = set_code.upper()
    ensure_set_dirs(set_code)
    cdir = _corpus_dir(set_code)
    n_jobs = int(n_jobs or os.cpu_count() or 1)

    def prog(frac: float, msg: str) -> None:
        if progress:
            progress(float(min(max(frac, 0.0), 1.0)), msg)

    prog(0.0, "Checking source files")
    files = registry.files(set_code)
    games_files = [_refresh(f) for f in files if f.kind == "game"]
    if not games_files:
        raise ValueError(f"No game data files registered for set {set_code}")
    csv_files = [f for f in files if f.kind == "cards"]
    sf_files = [f for f in files if f.kind == "scryfall"]
    games_files.sort(key=lambda f: (f.event or "", f.path))
    warnings: list[str] = []

    # --- quick path: nothing changed
    sig = _signature(set_code, [f.fingerprint for f in games_files], [f.fingerprint for f in csv_files + sf_files], fetch_scryfall)
    prev = read_manifest(set_code)
    if (prev and prev.get("signature") == sig and all((cdir / f"{t}.parquet").exists() for t in TABLES)
            and (not fetch_scryfall or prev.get("scryfall", {}).get("complete", False))
            and all(game_data.shard_valid(cdir, f.fingerprint) for f in games_files)):
        prev = dict(prev)
        prev["up_to_date"] = True
        prev["stats"] = {"files_ingested": 0, "files_cached": len(games_files)}
        for fi in prev["files"]:
            if fi.get("status") == "ingested":
                fi["status"] = "cached"
        prog(1.0, "Corpus is up to date")
        return prev

    # --- which files need ingesting
    tasks, statuses = [], {}
    metas: dict[str, dict] = {}
    for f in games_files:
        meta = game_data.shard_valid(cdir, f.fingerprint)
        if meta:
            metas[f.fingerprint] = meta
            statuses[f.fingerprint] = "cached"
        elif Path(f.path).exists():
            tasks.append(dict(path=f.path, fingerprint=f.fingerprint, event=f.event, set_code=set_code,
                              corpus_dir=str(cdir), size=f.size))
            statuses[f.fingerprint] = "ingested"
        else:
            warnings.append(f"source file missing and not cached: {f.path}")
    _check(cancel)
    if tasks:
        new = run_ingest_tasks(tasks, n_jobs, lambda fr, m: prog(0.02 + 0.78 * fr, m), cancel)
        metas.update(new)
    prog(0.80, "Building card table")
    _check(cancel)
    used = [f for f in games_files if f.fingerprint in metas]
    if not used:
        raise ValueError(f"No usable game data for set {set_code}")

    # --- global card ids
    names = sorted(set().union(*[set(metas[f.fingerprint]["names"]) for f in used]))
    gid = {n: i for i, n in enumerate(names)}
    luts = {f.fingerprint: np.array([gid[n] for n in metas[f.fingerprint]["names"]], dtype=np.int32) for f in used}

    # --- cards table
    stats_frames = []
    for f in used:
        sp = game_data.shard_path(cdir, f.fingerprint) / "colorstats.parquet"
        cs = pl.read_parquet(sp)
        if cs.height:
            nm_list = metas[f.fingerprint]["names"]
            stats_frames.append(cs.with_columns(pl.Series("name", [nm_list[i] for i in cs["card"].to_list()], dtype=pl.String)))
    color_stats: dict[str, dict[str, int]] = {}
    if stats_frames:
        agg = pl.concat(stats_frames).group_by(["name", "main_colors"]).agg(pl.col("n").sum())
        for nm, mc, n in agg.iter_rows():
            color_stats.setdefault(nm, {})[mc] = n
    sf_cards, sf_warn = scryfall.get_scryfall_cards(
        set_code, names, [(f.path, f.fingerprint) for f in sf_files if Path(f.path).exists()], fetch=fetch_scryfall,
        session=session, progress=lambda fr, m: prog(0.80 + 0.04 * fr, m), cancel=cancel)
    warnings += sf_warn
    csv_sorted = sorted([f for f in csv_files if Path(f.path).exists()], key=lambda f: Path(f.path).stat().st_mtime)
    csv_rows = cards_mod.read_cards_csv([f.path for f in csv_sorted], set_code) if csv_sorted else {}
    cards = cards_mod.build_cards_table(names, sf_cards, csv_rows, color_stats)
    staged: dict[Path, Path] = {}          # live table path -> staged file; swapped in together just before the manifest
    staged[cdir / "cards.parquet"] = _stage(cards, cdir / "cards.parquet")
    by_src = cards.group_by("meta_source").len()
    src_counts = {r[0]: r[1] for r in by_src.iter_rows()}
    nonbasic_total = int((~cards["is_basic"]).sum())
    n_inferred = src_counts.get("inferred", 0) - int(((cards["meta_source"] == "inferred") & cards["is_basic"]).sum())
    if n_inferred > 0:
        warnings.append(f"{n_inferred} card(s) have inferred metadata only (no Scryfall / cards.csv match); "
                        "colors/mana values are approximate")
    elif src_counts.get("17lands", 0) and not src_counts.get("scryfall", 0):
        warnings.append("Scryfall data unavailable: using the 17lands cards.csv fallback (approximate pips, no oracle text)")
    _check(cancel)

    # --- merge shards
    prog(0.85, "Merging tables")
    games_l, builds_l, dc_l, pools_l, audit_l, hands_l = [], [], [], [], [], []
    for f in used:
        sd = game_data.shard_path(cdir, f.fingerprint)
        lut = luts[f.fingerprint]
        games_l.append(pl.read_parquet(sd / "games.parquet"))
        builds_l.append(pl.read_parquet(sd / "builds.parquet"))
        dc = pl.read_parquet(sd / "deck_cards.parquet")
        dc_l.append(pl.DataFrame({"deck_key": dc["deck_key"], "card_id": lut[dc["card"].to_numpy()],
                                  "n": dc["n"]}, schema={"deck_key": pl.Int64, "card_id": pl.Int32, "n": pl.Int8}))
        pl_ = pl.read_parquet(sd / "pools.parquet")
        pools_l.append(pl.DataFrame({"draft_id": pl_["draft_id"], "card_id": lut[pl_["card"].to_numpy()],
                                     "n": pl_["n"]}, schema={"draft_id": pl.String, "card_id": pl.Int32, "n": pl.Int8}))
        audit_l.append(pl.read_parquet(sd / "pool_audit.parquet"))
        h = pl.read_parquet(sd / "hands.parquet")
        hands_l.append(pl.DataFrame({"game_key": h["game_key"], "card_id": lut[h["card"].to_numpy()],
                                     "opening": h["opening"], "drawn": h["drawn"]},
                                    schema={"game_key": pl.Int64, "card_id": pl.Int32, "opening": pl.Int8, "drawn": pl.Int8}))
        _check(cancel)
    games = pl.concat(games_l)
    builds = pl.concat(builds_l)
    dc_all = pl.concat(dc_l)
    if len(dc_l) > 1 and dc_all.height:
        # Guard against cross-shard 63-bit deck_key collisions: two shards sharing a deck_key must carry identical (card_id, n) rows
        per_shard = [df.sort(["deck_key", "card_id"]).group_by("deck_key").agg(
            (pl.col("card_id").cast(pl.String) + ":" + pl.col("n").cast(pl.String)).str.join(",").alias("sig"))
            for df in dc_l if df.height]
        if len(per_shard) > 1:
            coll = pl.concat(per_shard).group_by("deck_key").agg(pl.col("sig").n_unique().alias("nu")).filter(pl.col("nu") > 1)
            if coll.height:
                warnings.append(f"{coll.height} deck_key hash collision(s) detected across shards; first shard kept")
                log.warning("deck_key hash collision across shards for %s: %d key(s)", set_code, coll.height)
    deck_cards = dc_all.unique(subset=["deck_key", "card_id"], keep="first").sort(["deck_key", "card_id"])
    pools = pl.concat(pools_l)
    audit = pl.concat(audit_l)
    hands = pl.concat(hands_l)

    prog(0.90, "Hashing decks")
    is_land_df = cards.select("card_id", "is_land")
    j = deck_cards.join(is_land_df, on="card_id", how="left")
    agg = (j.with_columns((pl.col("card_id").cast(pl.String) + ":" + pl.col("n").cast(pl.String)).alias("t"))
           .group_by("deck_key", maintain_order=True)
           .agg(pl.col("t").str.join(","), pl.col("n").sum().cast(pl.Int16).alias("n_cards"),
                (pl.col("n") * pl.col("is_land").cast(pl.Int16)).sum().cast(pl.Int16).alias("n_lands")))
    hashes = [hashlib.sha1(s.encode()).hexdigest() for s in agg["t"].to_list()]
    decks = pl.DataFrame({"deck_key": agg["deck_key"], "deck_hash": hashes, "n_cards": agg["n_cards"],
                          "n_lands": agg["n_lands"]},
                         schema={"deck_key": pl.Int64, "deck_hash": pl.String, "n_cards": pl.Int16, "n_lands": pl.Int16})
    prog(0.94, "Writing tables")
    for name, df in [("games", games), ("decks", decks), ("deck_cards", deck_cards), ("builds", builds),
                     ("pools", pools), ("pool_audit", audit), ("hands", hands)]:
        staged[cdir / f"{name}.parquet"] = _stage(df, cdir / f"{name}.parquet")

    # --- audits
    fmt_games = {r[0]: r[1] for r in games.group_by("fmt").len().iter_rows()}
    pool_ev = {}
    hist: dict[str, int] = {}
    viol = rows = dropped = 0
    for f in used:
        m = metas[f.fingerprint]
        if m["fmt"] in ("bo1", "bo3"):
            e = pool_ev.setdefault(m["fmt"], [0, 0])
            e[0] += m["pool_events"]
            e[1] += m["pool_invariant"]
        for k, v in m["deck_size_hist"].items():
            hist[k] = hist.get(k, 0) + v
        viol += m["hand_violations"]
        rows += m["rows"]
        dropped += m["rows_dropped"]
        if m["missing_columns"]:
            warnings.append(f"{Path(m['path']).name}: missing columns {m['missing_columns']}")
    pool_rate = {k: (v[1] / v[0] if v[0] else None) for k, v in pool_ev.items()}
    for k, r in pool_rate.items():
        if r is not None and r < 0.98:
            warnings.append(f"{k}: only {r:.1%} of events have an invariant deck+sideboard pool")
    n40 = hist.get("40", 0)
    tot = sum(hist.values())
    if tot and n40 / tot < 0.99:
        warnings.append(f"{1 - n40 / tot:.1%} of games have a deck that is not 40 cards")
    if dropped:
        warnings.append(f"{dropped} game row(s) dropped (missing/invalid 'won')")
    for fmt, n in fmt_games.items():
        if n < 1000:
            warnings.append(f"only {n} {fmt} games: model for this format will lean on priors")
    audit_d = dict(
        n_games_by_fmt=fmt_games, pool_invariance_rate=pool_rate,
        pool_events={k: v[0] for k, v in pool_ev.items()}, deck_size_hist=hist,
        hand_violation_rate=(viol / rows if rows else 0.0), rows_dropped=dropped,
        unique_decks=int(decks.height), unique_builds=int(builds.height),
    )
    unused = [f for f in files if f.kind in ("draft", "replay")]
    man_files = []
    for f in games_files:
        m = metas.get(f.fingerprint)
        man_files.append(dict(path=f.path, fingerprint=f.fingerprint, kind=f.kind, event=f.event, fmt=f.fmt,
                              rows=(m["rows"] if m else None), status=statuses.get(f.fingerprint, "missing")))
    for f in unused:
        man_files.append(dict(path=f.path, fingerprint=f.fingerprint, kind=f.kind, event=f.event, fmt=f.fmt,
                              rows=None, status="registered (not used in v1)"))
    for f in csv_files + sf_files:
        man_files.append(dict(path=f.path, fingerprint=f.fingerprint, kind=f.kind, event=None, fmt=None,
                              rows=None, status="card metadata"))
    cache = scryfall.load_cache(set_code)
    idx = scryfall.build_index(cache["cards"])
    miss = [n for n in names if n not in BASICS and scryfall.lookup(idx, n) is None and n not in cache["not_found"]]
    manifest = dict(
        set=set_code, built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), code_version=CODE_VERSION,
        shard_version=game_data.SHARD_VERSION, signature=None, files=man_files,
        tables={"cards": cards.height, "games": games.height, "decks": decks.height, "deck_cards": deck_cards.height,
                "builds": builds.height, "pools": pools.height, "pool_audit": audit.height, "hands": hands.height},
        audit=audit_d, card_meta_sources=src_counts, n_cards=cards.height, warnings=warnings,
        scryfall=dict(requested=bool(fetch_scryfall), searched=bool(cache.get("searched")), complete=bool(cache.get("searched")) and not miss,
                      cards_cached=len(cache["cards"])),
        stats={"files_ingested": sum(1 for s in statuses.values() if s == "ingested"),
               "files_cached": sum(1 for s in statuses.values() if s == "cached")},
        seconds=round(time.time() - t_start, 2), up_to_date=False,
    )
    manifest["signature"] = _signature(set_code, [f.fingerprint for f in games_files],
                                       [f.fingerprint for f in csv_files + sf_files], fetch_scryfall)
    tmp = manifest_path(set_code).with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        _check(cancel)                     # the only cancel point after the first table is staged: nothing live has changed yet
    except Cancelled:
        for new in staged.values():
            new.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        raise
    _commit_tables(staged, manifest_path(set_code))
    os.replace(tmp, manifest_path(set_code))
    # garbage-collect shards of files no longer registered
    keep = {f.fingerprint for f in games_files}
    root = game_data.shard_root(cdir)
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and d.name not in keep:
                shutil.rmtree(d, ignore_errors=True)
    prog(1.0, "Corpus ready")
    return manifest


def load_corpus(set_code: str) -> dict[str, pl.LazyFrame | pl.DataFrame]:
    """Open the corpus tables. Big tables (games, deck_cards, hands, pools) are LazyFrames, the rest DataFrames."""
    cdir = _corpus_dir(set_code.upper())
    out: dict[str, pl.LazyFrame | pl.DataFrame] = {}
    for t in TABLES:
        p = cdir / f"{t}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"corpus table missing: {p} (run calculate / build_corpus first)")
        out[t] = pl.scan_parquet(p) if t in LAZY_TABLES else pl.read_parquet(p)
    return out
