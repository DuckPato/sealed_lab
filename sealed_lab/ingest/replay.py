"""Replay (turn-by-turn) data: streaming ingest of 17lands `replay_data_public.*` files into compact per-file shards.

Semantics of the replay columns (verified on MSH, see `scripts/replay_check.py`):

* One row per game; the rows are the SAME games as the sibling `game_data_public.*` file and join to the corpus on
  (draft_id, match_number, game_number) with 100% agreement of on_play / won / num_turns / mulligans.
* Card references are Arena ids (`grpId`), pipe-separated lists: ``"104942|105151|75025"``. There is no id -> name table
  in the files; `resolve_idmap` infers it from the decks (an id can only appear in games whose deck contains the card).
* `user_turn_<k>_*` = the user's own k-th turn (``k = 1..30``), `oppo_turn_<k>_*` = the opponent's k-th turn. The
  column stem names whose action: ``user_turn_2_creatures_cast`` = creatures the *user* cast in his own turn 2,
  ``user_turn_2_oppo_instants_sorceries_cast`` = instants the *opponent* cast during the user's turn 2,
  ``oppo_turn_2_user_instants_sorceries_cast`` = instants the *user* cast during the opponent's turn 2.
* ``cards_drawn`` / ``cards_tutored``: ids (first own turn on the play has no draw). ``lands_played`` and
  ``creatures_cast`` / ``non_creatures_cast`` (artifacts, enchantments, planeswalkers) / ``user_instants_sorceries_cast``
  are lists of ids. ``user_mana_spent`` is a float (mana paid during that turn, abilities included).
  ``eot_user_lands_in_play`` / ``eot_user_cards_in_hand`` are lists of ids at end of that turn; empty -> null.
* A turn is *complete* when ``eot_user_life`` > 0; the turn in which the game ends carries no end-of-turn data
  (life 0.0 placeholder, no draw recorded), so the data only describe completed own turns.
* ``opening_hand`` = the final 7 candidate cards (after a London mulligan the bottomed card is not identified);
  ``candidate_hand_1..7`` list the successive 7-card hands shown to the player (each mulligan shows a fresh one; in Bo1 queues
  they are the already smoothed hands); the shards keep the first three.  ``num_turns`` ~ number of rounds.

Shard (per file fingerprint, in ``sets/<SET>/cache/replay/<fp>/``): ``games.parquet`` (one row per game, per-turn wide
columns ``lands_k mana_k hand_k cre_k ncre_k inst_k drawn_k`` for own turns 1..10, null for incomplete turns),
``draws.parquet`` (gid, turn, kind 0 opening hand / 1 draw / 2 tutor / 3, 4, 5 the 7-card hands shown before the 1st, after the 1st, after the 2nd mulligan, id, n) and ``casts.parquet``
(gid, turn, kind 0 creature / 1 other permanent / 2 instant-sorcery / 3 instant-sorcery cast in the opponent's turn,
id, n). ``gid`` is globally unique across shards. Ingest is incremental by file fingerprint.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import logging
import multiprocessing
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from sealed_lab import config
from sealed_lab.ingest import registry
from sealed_lab.procguard import watch_parent
from sealed_lab.types import Cancelled, FileInfo

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

REPLAY_VERSION = 3
K_WIDE = 10          # own turns stored as wide per-turn columns
K_LIST = 12          # own turns scanned for id lists (draws, casts) and for "turn complete" flags
RANGE_BYTES = 40 << 20
GZ_BLOCK_BYTES = 24 << 20
CAND_COLS = ("candidate_hand_1", "candidate_hand_2", "candidate_hand_3")   # draws kinds 3, 4, 5
WIDE_STEMS = ("lands", "mana", "hand", "cre", "ncre", "inst", "drawn")
_TRUE = ["true", "1", "t", "yes"]
_FALSE = ["false", "0", "f", "no"]

META_COLS = ["event_type", "draft_id", "build_index", "match_number", "game_number", "game_time", "on_play", "won",
             "num_turns", "num_mulligans", "opp_num_mulligans", "main_colors", "opp_colors",
             "user_n_games_bucket", "user_game_win_rate_bucket"]


def wide_cols() -> list[str]:
    return [f"{s}_{k}" for s in WIDE_STEMS for k in range(1, K_WIDE + 1)]


# ---------------------------------------------------------------------------------------------
# paths / shard bookkeeping
# ---------------------------------------------------------------------------------------------

def replay_dir(set_code: str) -> Path:
    return config.cache_dir(set_code.upper()) / "replay"


def shard_path(set_code: str, fp: str) -> Path:
    return replay_dir(set_code) / fp


def shard_meta(set_code: str, fp: str) -> dict | None:
    """Meta of a complete shard of the current version, else None."""
    mp = shard_path(set_code, fp) / "meta.json"
    if not mp.exists():
        return None
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("version") != REPLAY_VERSION or not meta.get("ok"):
        return None
    return meta


def replay_files(set_code: str) -> list[FileInfo]:
    """Replay files registered for the set (existing on disk), sorted by event name."""
    fs = [f for f in registry.files(set_code.upper(), include_shared=False) if f.kind == "replay" and Path(f.path).exists()]
    return sorted(fs, key=lambda f: (f.event or "", f.path))


def _current_fingerprint(f: FileInfo) -> str:
    from sealed_lab.ingest.detect import fingerprint

    try:
        return fingerprint(f.path)
    except OSError:
        return f.fingerprint


def status(set_code: str) -> list[dict]:
    """Per registered replay file: event, size, fingerprint and whether a valid shard exists."""
    out = []
    for f in replay_files(set_code):
        fp = _current_fingerprint(f)
        m = shard_meta(set_code, fp)
        out.append(dict(event=f.event, path=f.path, size=f.size, fingerprint=fp, cached=m is not None,
                        rows=(m or {}).get("rows")))
    return out


# ---------------------------------------------------------------------------------------------
# parsing of one block of rows
# ---------------------------------------------------------------------------------------------

def _q(field: str) -> str:
    if any(ch in field for ch in ',"\n\r'):
        return '"' + field.replace('"', '""') + '"'
    return field


def _header_fields(header: list[str]) -> tuple[list[str], set[str]]:
    """Columns to read from the file (subset of the header) and the header as a set."""
    hset = set(header)
    want: list[str] = [c for c in META_COLS if c in hset]
    want += [c for c in ("opening_hand",) if c in hset]
    for c in CAND_COLS:
        if c in hset:
            want.append(c)
    for k in range(1, K_LIST + 1):
        for stem in ("cards_drawn", "cards_tutored", "creatures_cast", "non_creatures_cast", "user_instants_sorceries_cast",
                     "user_mana_spent", "eot_user_lands_in_play", "eot_user_life", "eot_user_cards_in_hand"):
            full = f"user_turn_{k}_{stem}" if stem in ("cards_drawn", "cards_tutored", "creatures_cast", "non_creatures_cast") \
                else f"user_turn_{k}_{stem}"
            if full in hset:
                want.append(full)
        full = f"oppo_turn_{k}_user_instants_sorceries_cast"
        if full in hset:
            want.append(full)
    seen, out = set(), []
    for c in want:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out, hset


def _extract(df: pl.DataFrame, hset: set[str], event: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, int]:
    """One parsed block (all columns String) -> (games, draws, casts, rows_read). `g` is the local row index."""
    R = df.height
    df = df.with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("_g"))

    def S(c: str) -> pl.Expr:
        return pl.col(c) if c in hset else pl.lit(None, dtype=pl.String)

    def F(c: str) -> pl.Expr:
        return pl.col(c).cast(pl.Float64, strict=False) if c in hset else pl.lit(None, dtype=pl.Float64)

    def count(c: str) -> pl.Expr:
        if c not in hset:
            return pl.lit(0, dtype=pl.Int16)
        col = pl.col(c)
        return (pl.when(col.is_null() | (col == "")).then(0).otherwise(col.str.count_matches("|", literal=True) + 1)
                .cast(pl.Int16))

    won_s = S("won").str.to_lowercase()
    valid = won_s.is_in(_TRUE + _FALSE).fill_null(False)
    op_s = S("on_play").str.to_lowercase()
    df = df.with_columns(op_s.is_in(_TRUE).alias("_op"), valid.alias("_valid"))

    # ---- completed own turns
    def done(k: int) -> pl.Expr:
        life = f"user_turn_{k}_eot_user_life"
        if life in hset:
            return (pl.col(life).cast(pl.Float64, strict=False) > 0).fill_null(False)
        return (count(f"user_turn_{k}_eot_user_cards_in_hand") + count(f"user_turn_{k}_eot_user_lands_in_play")) > 0

    D = df.select([done(k).alias(f"d{k}") for k in range(1, K_LIST + 1)]).to_numpy().astype(np.int8)
    turns_done = np.cumprod(D, axis=1).sum(axis=1).astype(np.int8)

    exprs: list[pl.Expr] = [
        pl.col("_g").alias("g"),
        S("draft_id").alias("draft_id"),
        F("build_index").cast(pl.Int16, strict=False).alias("build_index"),
        F("match_number").cast(pl.Int16, strict=False).alias("match_number"),
        F("game_number").cast(pl.Int16, strict=False).alias("game_number"),
        S("game_time").str.to_datetime(strict=False, time_unit="us").alias("game_time"),
        pl.col("_op").alias("on_play"),
        won_s.is_in(_TRUE).alias("won"),
        F("num_turns").cast(pl.Int16, strict=False).alias("num_turns"),
        F("num_mulligans").fill_null(0).cast(pl.Int8, strict=False).alias("num_mulligans"),
        F("opp_num_mulligans").fill_null(0).cast(pl.Int8, strict=False).alias("opp_num_mulligans"),
        S("main_colors").fill_null("").alias("main_colors"),
        S("opp_colors").fill_null("").alias("opp_colors"),
        F("user_game_win_rate_bucket").cast(pl.Float32, strict=False).alias("skill"),
        F("user_n_games_bucket").cast(pl.Float32, strict=False).alias("n_games_bucket"),
        pl.Series("turns_done", turns_done),
        count("opening_hand").cast(pl.Int8).alias("open_n"),
    ]
    for k in range(1, K_WIDE + 1):
        dk = pl.Series(f"_d{k}", D[:, k - 1].astype(bool))

        def w(e: pl.Expr, name: str, dk=dk) -> pl.Expr:
            return pl.when(dk).then(e).otherwise(None).cast(pl.Int8).alias(name)

        exprs += [
            w(count(f"user_turn_{k}_eot_user_lands_in_play"), f"lands_{k}"),
            w(F(f"user_turn_{k}_user_mana_spent").fill_null(0).clip(0, 127), f"mana_{k}"),
            w(count(f"user_turn_{k}_eot_user_cards_in_hand"), f"hand_{k}"),
            w(count(f"user_turn_{k}_creatures_cast"), f"cre_{k}"),
            w(count(f"user_turn_{k}_non_creatures_cast"), f"ncre_{k}"),
            w(count(f"user_turn_{k}_user_instants_sorceries_cast"), f"inst_{k}"),
            w(count(f"user_turn_{k}_cards_drawn") + count(f"user_turn_{k}_cards_tutored"), f"drawn_{k}"),
        ]
    games = df.select(exprs)
    keep = df["_valid"].to_numpy()

    # ---- long id tables
    base = df.select("_g", "_op", *[c for c in df.columns if c.startswith(("user_turn_", "oppo_turn_")) or c == "opening_hand" or c in CAND_COLS])
    lf = base.lazy()

    def one(col: str, kind: int, turn: pl.Expr) -> pl.LazyFrame | None:
        if col not in hset:
            return None
        return (lf.filter(pl.col(col).is_not_null() & (pl.col(col) != ""))
                .select(pl.col("_g").alias("g"), turn.cast(pl.Int8).alias("turn"), pl.lit(kind, dtype=pl.Int8).alias("kind"),
                        pl.col(col).str.split("|").alias("id"))
                .explode("id"))

    def finish(frames: list[pl.LazyFrame | None]) -> pl.DataFrame:
        fr = [f for f in frames if f is not None]
        schema = {"g": pl.Int32, "turn": pl.Int8, "kind": pl.Int8, "id": pl.Int32, "n": pl.Int8}
        if not fr:
            return pl.DataFrame(schema=schema)
        parts = pl.collect_all(fr)
        out = pl.concat(parts)
        if out.is_empty():
            return pl.DataFrame(schema=schema)
        return (out.with_columns(pl.col("id").cast(pl.Int32, strict=False)).drop_nulls("id")
                .group_by(["g", "turn", "kind", "id"]).len().rename({"len": "n"})
                .with_columns(pl.col("n").clip(0, 127).cast(pl.Int8)).select(list(schema)).sort(["g", "turn", "kind", "id"]))

    # kind 0 = opening_hand (final 7 candidates), kind 3 = candidate_hand_1 (the first 7 cards of the shuffle, before any mulligan)
    # kinds 4 and 5 = the 7-card hands shown after the first and second mulligan (candidate_hand_2 / _3)
    draw_frames = [one("opening_hand", 0, pl.lit(0))] + [one(c, 3 + i, pl.lit(0)) for i, c in enumerate(CAND_COLS)]
    cast_frames = []
    for k in range(1, K_LIST + 1):
        draw_frames.append(one(f"user_turn_{k}_cards_drawn", 1, pl.lit(k)))
        draw_frames.append(one(f"user_turn_{k}_cards_tutored", 2, pl.lit(k)))
        cast_frames.append(one(f"user_turn_{k}_creatures_cast", 0, pl.lit(k)))
        cast_frames.append(one(f"user_turn_{k}_non_creatures_cast", 1, pl.lit(k)))
        cast_frames.append(one(f"user_turn_{k}_user_instants_sorceries_cast", 2, pl.lit(k)))
        # the opponent's k-th turn comes after our k-th turn when we are on the play, before it when on the draw
        cast_frames.append(one(f"oppo_turn_{k}_user_instants_sorceries_cast", 3,
                               pl.lit(k) - (~pl.col("_op")).cast(pl.Int8)))
    draws = finish(draw_frames)
    casts = finish(cast_frames)
    if not keep.all():
        gk = pl.Series(np.flatnonzero(keep).astype(np.int32))
        games = games.filter(pl.Series(keep))
        draws = draws.filter(pl.col("g").is_in(gk))
        casts = casts.filter(pl.col("g").is_in(gk))
    return games, draws, casts, R


# ---------------------------------------------------------------------------------------------
# worker: one byte range of a file (or a whole gzip stream)
# ---------------------------------------------------------------------------------------------

def _read_header(path: str) -> tuple[list[str], int, bool]:
    import csv

    from sealed_lab.ingest.detect import csv_stream, normalize_header

    with open(path, "rb") as raw:
        stream = csv_stream(raw)
        line = stream.readline()
        header = normalize_header(next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")])))
        if stream is not raw:          # gzip / tar: read as one stream (byte ranges need a plain file)
            return header, 0, True
        return header, raw.tell(), False


def _write_part(outdir: Path, ri: int, si: int, games, draws, casts, rows_read: int) -> dict:
    stem = f"{ri:05d}_{si:03d}"
    games.write_parquet(outdir / f"{stem}.games.parquet", compression="zstd")
    draws.write_parquet(outdir / f"{stem}.draws.parquet", compression="zstd")
    casts.write_parquet(outdir / f"{stem}.casts.parquet", compression="zstd")
    return dict(ri=ri, si=si, rows_read=rows_read, rows=games.height)


def parse_range_task(task: dict) -> dict:
    """Top-level (picklable) worker. task: path, ri, start, end (None for gzip), header, parts_dir, event."""
    header: list[str] = task["header"]
    want, hset = _header_fields(header)
    header_line = (",".join(_q(h) for h in want) + "\n").encode("utf-8")
    # re-quote only the selected columns' header: polars needs a header that matches the row layout, so keep the full one
    full_header = (",".join(_q(h) for h in header) + "\n").encode("utf-8")
    overrides = {c: pl.String for c in want}
    outdir = Path(task["parts_dir"])
    ri = task["ri"]
    parts = []

    def parse(buf: bytes) -> tuple:
        df = pl.read_csv(io.BytesIO(full_header + buf), columns=want, schema_overrides=overrides, infer_schema_length=0)
        return _extract(df, hset, task["event"])

    if task.get("gz"):
        from sealed_lab.ingest.detect import csv_stream

        with open(task["path"], "rb") as raw:
            stream = csv_stream(raw)
            stream.readline()
            carry, si, eof = b"", 0, False
            while not eof:
                block = stream.read(GZ_BLOCK_BYTES)
                if not block:
                    eof, buf, carry = True, carry, b""
                else:
                    buf = carry + block
                    cut = buf.rfind(b"\n")
                    if cut < 0:
                        carry = buf
                        continue
                    carry, buf = buf[cut + 1:], buf[:cut + 1]
                if not buf.strip():
                    continue
                g, d, c, rr = parse(buf)
                parts.append(_write_part(outdir, ri, si, g, d, c, rr))
                si += 1
        return dict(ri=ri, parts=parts, bytes=task.get("size", 0))

    start, end = task["start"], task["end"]
    with open(task["path"], "rb") as f:
        if start > 0:
            f.seek(start - 1)
            f.readline()                       # skip the line that started in the previous range
        p = f.tell()
        data = f.read(max(end - p, 0)) if p < end else b""
        if data and not data.endswith(b"\n"):
            data += f.readline()               # finish the last line that started inside this range
    if data.strip():
        g, d, c, rr = parse(data)
        parts.append(_write_part(outdir, ri, 0, g, d, c, rr))
    return dict(ri=ri, parts=parts, bytes=end - start)


# ---------------------------------------------------------------------------------------------
# parent: planning, merging
# ---------------------------------------------------------------------------------------------

def _plan(f: FileInfo, fp: str, parts_dir: Path, ri0: int) -> tuple[list[dict], int]:
    header, data_start, gz = _read_header(f.path)
    size = os.path.getsize(f.path)
    base = dict(path=f.path, header=header, parts_dir=str(parts_dir), event=f.event or "", size=size)
    if gz:
        return [dict(base, ri=ri0, gz=True, start=0, end=None)], ri0 + 1
    tasks, s, ri = [], data_start, ri0
    while s < size:
        e = min(s + RANGE_BYTES, size)
        tasks.append(dict(base, ri=ri, gz=False, start=s, end=e))
        s, ri = e, ri + 1
    return tasks, ri


def _file_tag(fp: str) -> int:
    return int(fp[:6], 16)


def _merge(set_code: str, f: FileInfo, fp: str, parts_dir: Path, results: list[dict], seconds: float) -> dict:
    parts = sorted((p for r in results for p in r["parts"]), key=lambda p: (p["ri"], p["si"]))
    tag = np.int64(_file_tag(fp)) << np.int64(32)
    offset = 0
    G, Dr, Ca = [], [], []
    rows_read = 0
    for p in parts:
        stem = f"{p['ri']:05d}_{p['si']:03d}"
        g = pl.read_parquet(parts_dir / f"{stem}.games.parquet")
        d = pl.read_parquet(parts_dir / f"{stem}.draws.parquet")
        c = pl.read_parquet(parts_dir / f"{stem}.casts.parquet")
        add = pl.lit(int(tag) + offset, dtype=pl.Int64)
        G.append(g.with_columns((pl.col("g").cast(pl.Int64) + add).alias("gid")).drop("g"))
        Dr.append(d.with_columns((pl.col("g").cast(pl.Int64) + add).alias("gid")).drop("g"))
        Ca.append(c.with_columns((pl.col("g").cast(pl.Int64) + add).alias("gid")).drop("g"))
        offset += p["rows_read"]
        rows_read += p["rows_read"]
    games = pl.concat(G) if G else pl.DataFrame()
    draws = pl.concat(Dr) if Dr else pl.DataFrame()
    casts = pl.concat(Ca) if Ca else pl.DataFrame()
    games = games.select(["gid"] + [c for c in games.columns if c != "gid"])
    draws = draws.select("gid", "turn", "kind", "id", "n")
    casts = casts.select("gid", "turn", "kind", "id", "n")
    final = shard_path(set_code, fp)
    tmp = final.with_name(fp + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, df in (("games", games), ("draws", draws), ("casts", casts)):
        df.write_parquet(tmp / f"{name}.parquet", compression="zstd")
    meta = dict(version=REPLAY_VERSION, ok=True, fingerprint=fp, path=f.path, event=f.event, set_code=set_code.upper(),
                rows=int(games.height), rows_read=int(rows_read), n_draw_rows=int(draws.height), n_cast_rows=int(casts.height),
                k_wide=K_WIDE, k_list=K_LIST, seconds=round(seconds, 2))
    (tmp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return meta


def ingest_replays(set_code: str, progress: Progress | None = None, cancel: threading.Event | None = None,
                   n_jobs: int | None = None, events: list[str] | None = None) -> dict:
    """Ingest the registered replay files of a set (only those without a valid shard). Parallel over byte ranges of the
    files (spawn pool). Returns {"files": [meta...], "ingested": n, "cached": n, "seconds": s}. Writes only to
    ``sets/<SET>/cache/replay/``."""
    set_code = set_code.upper()
    t0 = time.time()
    files = replay_files(set_code)
    if events is not None:
        files = [f for f in files if f.event in events]
    rdir = replay_dir(set_code)
    rdir.mkdir(parents=True, exist_ok=True)

    def prog(fr: float, msg: str) -> None:
        if progress:
            progress(float(min(max(fr, 0.0), 1.0)), msg)

    metas: dict[str, dict] = {}
    todo: list[tuple[FileInfo, str]] = []
    for f in files:
        fp = _current_fingerprint(f)
        m = shard_meta(set_code, fp)
        if m:
            metas[fp] = m
        else:
            todo.append((f, fp))
    n_cached = len(metas)
    if todo:
        n_jobs = int(n_jobs or os.cpu_count() or 1)
        tasks: list[dict] = []
        owner: dict[int, tuple[FileInfo, str]] = {}
        ri = 0
        for f, fp in todo:
            pdir = rdir / f"{fp}.parts"
            if pdir.exists():
                shutil.rmtree(pdir)
            pdir.mkdir(parents=True)
            ts, ri2 = _plan(f, fp, pdir, ri)
            for t in ts:
                owner[t["ri"]] = (f, fp)
            tasks += ts
            ri = ri2
        total = sum(max(t["end"] - t["start"], 1) if not t.get("gz") else max(t["size"], 1) for t in tasks)
        order = sorted(tasks, key=lambda t: (-t["size"], t["ri"]))
        ctx = multiprocessing.get_context("spawn")
        prev = os.environ.get("POLARS_MAX_THREADS")
        os.environ["POLARS_MAX_THREADS"] = "1"
        ex = cf.ProcessPoolExecutor(max_workers=max(1, min(n_jobs, len(tasks))), mp_context=ctx, initializer=watch_parent)
        results: dict[str, list[dict]] = {fp: [] for _, fp in todo}
        remaining = {fp: sum(1 for t in tasks if owner[t["ri"]][1] == fp) for _, fp in todo}
        done_bytes = 0
        try:
            futs = {ex.submit(parse_range_task, t): t for t in order}
            pending = set(futs)
            while pending:
                if cancel is not None and cancel.is_set():
                    for p in pending:
                        p.cancel()
                    raise Cancelled()
                done, pending = cf.wait(pending, timeout=0.25, return_when=cf.FIRST_COMPLETED)
                for fu in done:
                    if fu.cancelled():
                        continue
                    r = fu.result()
                    t = futs[fu]
                    f, fp = owner[t["ri"]]
                    results[fp].append(r)
                    done_bytes += r["bytes"]
                    remaining[fp] -= 1
                    prog(0.92 * done_bytes / total, f"Replay ingest {Path(f.path).name}")
                    if remaining[fp] == 0:
                        pdir = rdir / f"{fp}.parts"
                        prog(0.92 * done_bytes / total, f"Merging {Path(f.path).name}")
                        metas[fp] = _merge(set_code, f, fp, pdir, results[fp], time.time() - t0)
                        shutil.rmtree(pdir, ignore_errors=True)
        except BaseException:
            ex.shutdown(wait=True, cancel_futures=True)
            for _, fp in todo:
                shutil.rmtree(rdir / f"{fp}.parts", ignore_errors=True)
            raise
        else:
            ex.shutdown(wait=True)
        finally:
            if prev is None:
                os.environ.pop("POLARS_MAX_THREADS", None)
            else:
                os.environ["POLARS_MAX_THREADS"] = prev
    # garbage-collect shards of files that are no longer registered (only when all registered files were considered)
    if events is None:
        keep = set(metas)
        for d in rdir.iterdir():
            if d.is_dir() and len(d.name) == 20 and d.name not in keep and all(ch in "0123456789abcdef" for ch in d.name):
                shutil.rmtree(d, ignore_errors=True)
    prog(1.0, "Replay data ready")
    return dict(files=[metas[_current_fingerprint(f)] for f in files if _current_fingerprint(f) in metas],
                ingested=len(todo), cached=n_cached, seconds=round(time.time() - t0, 2))


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------

def _shards(set_code: str, events: list[str] | None) -> list[dict]:
    out = []
    for f in replay_files(set_code):
        if events is not None and f.event not in events:
            continue
        m = shard_meta(set_code, _current_fingerprint(f))
        if m:
            out.append(m)
    return out


def available_events(set_code: str) -> list[str]:
    return [m["event"] for m in _shards(set_code, None)]


def _load(set_code: str, name: str, events: list[str] | None, lazy: bool):
    ms = _shards(set_code, events)
    if not ms:
        raise FileNotFoundError(f"no replay shard for {set_code} (run ingest_replays first)")
    frames = []
    for m in ms:
        lf = pl.scan_parquet(shard_path(set_code, m["fingerprint"]) / f"{name}.parquet")
        if name == "games":
            lf = lf.with_columns(pl.lit(m["event"]).alias("event"), pl.lit(m["fingerprint"]).alias("src"))
        frames.append(lf)
    lf = pl.concat(frames)
    return lf if lazy else lf.collect()


def load_games(set_code: str, events: list[str] | None = None, lazy: bool = False):
    """Per-game replay table of the given events (all if None), with `event` and `src` columns."""
    return _load(set_code, "games", events, lazy)


def load_draws(set_code: str, events: list[str] | None = None, lazy: bool = False):
    return _load(set_code, "draws", events, lazy)


def load_casts(set_code: str, events: list[str] | None = None, lazy: bool = False):
    return _load(set_code, "casts", events, lazy)


# ---------------------------------------------------------------------------------------------
# Arena id -> corpus card_id
# ---------------------------------------------------------------------------------------------

def link_corpus(games: pl.DataFrame, corpus_games: pl.DataFrame) -> pl.DataFrame:
    """Add the corpus columns (game_key, deck_key, fmt, skill_c) to replay games. Join on (draft_id, match_number,
    game_number); rows still unmatched (no match number) fall back to (draft_id, game_time)."""
    cg = corpus_games.select("game_key", "deck_key", "fmt", "draft_id", "match_number", "game_number", "game_time",
                             pl.col("skill").alias("skill_c"))
    k3 = ["draft_id", "match_number", "game_number"]
    a = games.join(cg.drop("game_time").unique(subset=k3), on=k3, how="left", maintain_order="left")
    if a["game_key"].null_count():
        k2 = ["draft_id", "game_time"]
        b = (cg.drop("match_number", "game_number").unique(subset=k2)
             .rename({"game_key": "gk2", "deck_key": "dk2", "fmt": "fmt2", "skill_c": "sk2"}))
        a = a.join(b, on=k2, how="left", maintain_order="left").with_columns(
            pl.coalesce("game_key", "gk2").alias("game_key"), pl.coalesce("deck_key", "dk2").alias("deck_key"),
            pl.coalesce("fmt", "fmt2").alias("fmt"), pl.coalesce("skill_c", "sk2").alias("skill_c")).drop(
            "gk2", "dk2", "fmt2", "sk2")
    return a


def resolve_idmap(set_code: str, progress: Progress | None = None, refresh: bool = False,
                  max_games: int = 250_000) -> dict:
    """Infer {arena_id: corpus card_id} from the decks: an id can only appear in games whose main deck contains its card.
    For each id the candidate cards are those that contain it in every game where it appears (tolerance 0.5%);
    ties go to the card with the highest P(id appears | card in deck). Cached in cache/replay/idmap.json.

    Returns dict(map={arena_id: card_id}, n_ids, n_mapped, mass_mapped, ambiguous, ...)."""
    import scipy.sparse as sp

    from sealed_lab.ingest.corpus import load_corpus

    set_code = set_code.upper()
    ms = _shards(set_code, None)
    sig = json.dumps([REPLAY_VERSION, sorted(m["fingerprint"] for m in ms)])
    cpath = replay_dir(set_code) / "idmap.json"
    corpus_man = config.corpus_dir(set_code) / "manifest.json"
    try:
        csig = json.loads(corpus_man.read_text(encoding="utf-8")).get("built_at", "")
    except (OSError, ValueError):
        csig = ""
    if cpath.exists() and not refresh:
        try:
            d = json.loads(cpath.read_text(encoding="utf-8"))
            if d.get("sig") == sig and d.get("corpus") == csig:
                d["map"] = {int(k): int(v) for k, v in d["map"].items()}
                return d
        except (OSError, ValueError):
            pass
    c = load_corpus(set_code)
    cards = c["cards"].sort("card_id")
    n_cards = cards.height
    games = load_games(set_code).select("gid", "draft_id", "match_number", "game_number", "game_time")
    cg = c["games"].select("game_key", "deck_key", "draft_id", "match_number", "game_number", "game_time", "fmt", "skill").collect()
    linked = link_corpus(games, cg).filter(pl.col("deck_key").is_not_null())
    if linked.height > max_games:
        linked = linked.sample(max_games, seed=0)
    dc = c["deck_cards"].collect()
    dk = np.sort(dc["deck_key"].unique().to_numpy())
    dpos = np.searchsorted(dk, dc["deck_key"].to_numpy())
    Dec = sp.csr_matrix((np.ones(dc.height, dtype=np.float32), (dpos, dc["card_id"].to_numpy())), shape=(len(dk), n_cards))
    row_deck = np.searchsorted(dk, linked["deck_key"].to_numpy())
    gid = linked["gid"].to_numpy()
    order = np.argsort(gid)
    gid_s = gid[order]
    Gsel = sp.csr_matrix((np.ones(len(row_deck), dtype=np.float32), (np.arange(len(row_deck)), row_deck[order])),
                         shape=(len(row_deck), len(dk)))
    Dg = (Gsel @ Dec).tocsr()                       # games x cards, copies > 0
    Dg.data[:] = 1.0
    Dn = np.asarray(Dg.sum(axis=0)).ravel()
    gid_pl = pl.Series(gid_s)

    def assign(ids_df: pl.DataFrame, min_a: int, tol: float, skip: set[int]) -> tuple[dict[int, int], int, float, int]:
        """Map the ids of `ids_df` (gid, id) to the cards consistent with every game where they appear."""
        ids_df = ids_df.filter(pl.col("gid").is_in(gid_pl.implode()))
        gi = np.searchsorted(gid_s, ids_df["gid"].to_numpy())
        uid, iinv = np.unique(ids_df["id"].to_numpy(), return_inverse=True)
        S = sp.csr_matrix((np.ones(len(gi), dtype=np.float32), (gi, iinv)), shape=(len(gid_s), len(uid)))
        C = (S.T @ Dg).toarray()                    # ids x cards: games where id appears and the deck has the card
        A = np.asarray(S.sum(axis=0)).ravel()
        out: dict[int, int] = {}
        amb = 0
        for i, a in enumerate(uid):
            if int(a) in skip or A[i] < min_a:
                continue
            ok = C[i] >= A[i] - max(1.0, tol * A[i])
            if ok.any():
                cand = np.flatnonzero(ok)
                if len(cand) > 1:
                    amb += 1
                out[int(a)] = int(cand[int(np.argmax(A[i] / np.maximum(Dn[cand], 1)))])
            else:
                j = int(np.argmax(C[i] / np.maximum(Dn, 1)))
                if C[i, j] >= 0.98 * A[i]:
                    out[int(a)] = j
        mass = float(sum(A[i] for i, a in enumerate(uid) if int(a) in out) / max(A.sum(), 1))
        return out, amb, mass, int(len(uid))

    # evidence 1 = cards seen from the library (opening hand / draws); cast lists also contain copied or stolen cards
    seen = load_draws(set_code, lazy=True).filter(pl.col("kind") < 2).select("gid", "id").unique().collect()
    mp, ambiguous, mass, n_ids = assign(seen, 3, 0.005, set())
    # evidence 2 = ids that only ever appear in cast lists (the cast face of double-faced / adventure / alternate-cast cards)
    cast_ids = load_casts(set_code, lazy=True).select("gid", "id").unique().collect()
    cast_ids = cast_ids.filter(~pl.col("id").is_in(pl.Series(list(mp.keys()) or [0], dtype=pl.Int32).implode()))
    mp2, amb2, mass2, n2 = assign(cast_ids, 8, 0.01, set(mp))
    mp.update(mp2)
    names = cards["name"].to_list()
    out = dict(sig=sig, corpus=csig, n_ids=n_ids, n_mapped=len(mp) - len(mp2), n_cast_only_ids=n2, n_mapped_cast_only=len(mp2),
               mass_mapped=mass, ambiguous=ambiguous + amb2,
               n_games_used=int(len(gid_s)), map={int(k): int(v) for k, v in mp.items()},
               sample=[[int(k), names[v]] for k, v in list(mp.items())[:10]])
    try:
        cpath.write_text(json.dumps(out), encoding="utf-8")
    except OSError:
        log.warning("could not write id map cache")
    return out


__all__ = ["ingest_replays", "load_games", "load_draws", "load_casts", "resolve_idmap", "link_corpus", "status",
           "replay_files", "available_events", "replay_dir", "wide_cols", "K_WIDE", "K_LIST", "REPLAY_VERSION"]
