"""Streaming ingest of one wide 17lands game_data file into a parquet "shard".

The wide CSV (one column per card per block) is read in ~16 MiB blocks (gzip or plain, decided by magic bytes),
parsed with Polars (only the needed columns, explicit dtypes), the card blocks are converted to numpy int8, decks
are hashed with a vectorised 64-bit linear hash keyed by card *name* (so shards stay valid when other files add
cards), and long tables are accumulated. Shards store card references as indices into the shard's own `names`
list; the corpus merge maps them to global `card_id`s.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from sealed_lab.config import BASICS, event_format
from sealed_lab.ingest.detect import csv_stream, normalize_header, open_binary
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)

SHARD_VERSION = 3
BLOCK_BYTES = 16 << 20

BLOCKS = ("opening_hand_", "drawn_", "deck_", "sideboard_")
STR_META = ["expansion", "event_type", "draft_id", "draft_time", "game_time", "rank", "opp_rank", "main_colors",
            "splash_colors", "on_play", "opp_colors", "won"]
NUM_META = ["build_index", "match_number", "game_number", "num_mulligans", "opp_num_mulligans", "num_turns",
            "user_n_games_bucket", "user_game_win_rate_bucket"]

_TRUE = ["true", "1", "t", "yes"]


def name_weight(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "little") | 1


def read_header(stream) -> list[str]:
    line = stream.readline()
    if not line:
        raise ValueError("empty file")
    text = line.decode("utf-8-sig").rstrip("\r\n")
    return next(csv.reader([text]))


def shard_root(corpus_dir: Path) -> Path:
    return corpus_dir / "_shards"


def shard_path(corpus_dir: Path, fingerprint: str) -> Path:
    return shard_root(corpus_dir) / fingerprint


def shard_valid(corpus_dir: Path, fingerprint: str) -> dict | None:
    """Return shard meta if a complete shard of the current version exists."""
    mp = shard_path(corpus_dir, fingerprint) / "meta.json"
    if not mp.exists():
        return None
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("version") != SHARD_VERSION or not meta.get("ok"):
        return None
    return meta


def _group_reduce(codes: np.ndarray, X: np.ndarray):
    """Group rows of X by integer codes -> (unique codes, max, min, counts)."""
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    Xs = X[order]
    mx = np.maximum.reduceat(Xs, starts, axis=0)
    mn = np.minimum.reduceat(Xs, starts, axis=0)
    cnt = np.diff(np.r_[starts, len(cs)])
    return cs[starts], mx, mn, cnt


def process_game_file(
    path: str,
    fingerprint: str,
    event: str | None,
    set_code: str,
    corpus_dir: str | Path,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    block_bytes: int = BLOCK_BYTES,
) -> dict:
    """Ingest one game file into `corpus_dir/_shards/<fingerprint>/`. Returns the shard meta dict."""
    t0 = time.time()
    corpus_dir = Path(corpus_dir)
    size = max(os.path.getsize(path), 1)
    raw = open(path, "rb")
    try:
        stream = csv_stream(raw)
        header = normalize_header(read_header(stream))
        header_line = (",".join(_q(h) for h in header) + "\n").encode("utf-8")

        hset = set(header)
        deck_cols = [h for h in header if h.startswith("deck_")]
        names = [h[len("deck_"):] for h in deck_cols]
        C = len(names)
        name_idx = {n: i for i, n in enumerate(names)}
        weights = np.array([name_weight(n) for n in names], dtype=np.uint64)
        nonbasic = np.array([n not in BASICS for n in names])
        nonbasic_idx = np.flatnonzero(nonbasic)
        block_cols: dict[str, list[str | None]] = {}
        for pre in BLOCKS:
            block_cols[pre] = [(pre + n) if (pre + n) in hset else None for n in names]
        present_meta = [c for c in STR_META + NUM_META if c in hset]
        missing_meta = [c for c in STR_META + NUM_META if c not in hset]
        select = list(present_meta)
        for pre in BLOCKS:
            select += [c for c in block_cols[pre] if c is not None]
        overrides: dict[str, pl.DataType] = {}
        for c in present_meta:
            overrides[c] = pl.String if c in STR_META else pl.Float64
        for pre in BLOCKS:
            for c in block_cols[pre]:
                if c is not None:
                    overrides[c] = pl.Int16
        ev = event or ""
        file_tag = int(fingerprint[:6], 16)
        fmt = event_format(ev) if ev else None

        games_parts: list[pl.DataFrame] = []
        deck_h_parts: list[np.ndarray] = []
        deck_trip_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        n_decks_seen = 0
        hands_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        pool_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        color_acc: dict[str, np.ndarray] = {}
        size_hist = np.zeros(128, dtype=np.int64)
        row_base = 0
        rows_total = rows_dropped = hand_viol = 0
        carry = b""
        eof = False
        want_pools = fmt in ("bo1", "bo3")
        mulligan_total = 0

        while not eof:
            if cancelled and cancelled():
                raise Cancelled()
            block = stream.read(block_bytes)
            if not block:
                eof = True
                buf = carry
                carry = b""
            else:
                buf = carry + block
                cut = buf.rfind(b"\n")
                if cut < 0:
                    carry = buf
                    continue
                carry = buf[cut + 1:]
                buf = buf[:cut + 1]
            if not buf.strip():
                continue
            if progress:
                progress(min(raw.tell() / size, 0.999))
            try:
                df = pl.read_csv(io.BytesIO(header_line + buf), columns=select, schema_overrides=overrides,
                                 infer_schema_length=0)
            except pl.exceptions.ComputeError:
                # some files write card counts as "0.0" / "1.0" (EOE): read this block's counts as floats and round
                df = pl.read_csv(io.BytesIO(header_line + buf), columns=select, infer_schema_length=0,
                                 schema_overrides={c: (pl.Float32 if t == pl.Int16 else t) for c, t in overrides.items()})
                df = df.with_columns(pl.col(c).round(0).cast(pl.Int16, strict=False) for c, t in overrides.items() if t == pl.Int16)
            n0 = df.height
            if n0 == 0:
                continue
            # --- numpy blocks
            def mat(pre: str) -> np.ndarray:
                cols = block_cols[pre]
                out = np.zeros((n0, C), dtype=np.int8)
                have = [(i, c) for i, c in enumerate(cols) if c is not None]
                if have:
                    sub = df.select([c for _, c in have])
                    if sub.null_count().sum_horizontal()[0]:
                        sub = sub.fill_null(0)
                    arr = sub.to_numpy()
                    if len(have) == C:
                        out[:] = np.clip(arr, 0, 127)
                    else:
                        out[:, [i for i, _ in have]] = np.clip(arr, 0, 127)
                return out

            Dm = mat("deck_")
            Om = mat("opening_hand_")
            Am = mat("drawn_")
            Sm = mat("sideboard_") if want_pools else None
            # --- games frame
            def sc(c: str) -> pl.Expr:
                return pl.col(c) if c in hset else pl.lit(None, dtype=pl.String)

            def nc(c: str, default=None) -> pl.Expr:
                return pl.col(c) if c in hset else pl.lit(default, dtype=pl.Float64)

            won = sc("won").str.to_lowercase().is_in(_TRUE) if "won" in hset else pl.lit(None, dtype=pl.Boolean)
            valid_expr = (sc("won").str.to_lowercase().is_in(_TRUE + ["false", "0", "f", "no"])
                          if "won" in hset else pl.lit(False)).fill_null(False)     # a row without a result is dropped
            h = (Dm.astype(np.uint64) @ weights)
            deck_key = (h >> np.uint64(1)).astype(np.int64)
            gk = (np.int64(file_tag) << np.int64(36)) + row_base + np.arange(n0, dtype=np.int64)
            g = df.select(
                pl.Series("game_key", gk),
                pl.lit(fingerprint).alias("src"),
                (sc("event_type") if not ev and "event_type" in hset else pl.lit(ev)).alias("event"),
                pl.lit(fmt or "").alias("fmt"),
                sc("draft_id").alias("draft_id"),
                nc("build_index", 0.0).cast(pl.Int16, strict=False).alias("build_index"),
                nc("match_number").cast(pl.Int16, strict=False).alias("match_number"),
                nc("game_number", 1.0).cast(pl.Int16, strict=False).alias("game_number"),
                pl.Series("deck_key", deck_key),
                sc("draft_time").str.to_datetime(strict=False, time_unit="us").alias("draft_time"),
                sc("game_time").str.to_datetime(strict=False, time_unit="us").alias("game_time"),
                sc("rank").alias("rank"),
                sc("opp_rank").alias("opp_rank"),
                sc("main_colors").fill_null("").alias("main_colors"),
                sc("splash_colors").fill_null("").alias("splash_colors"),
                sc("on_play").str.to_lowercase().is_in(_TRUE).alias("on_play"),
                nc("num_mulligans", 0.0).fill_null(0).cast(pl.Int8, strict=False).alias("num_mulligans"),
                nc("opp_num_mulligans", 0.0).fill_null(0).cast(pl.Int8, strict=False).alias("opp_num_mulligans"),
                sc("opp_colors").fill_null("").alias("opp_colors"),
                nc("num_turns").cast(pl.Int16, strict=False).alias("num_turns"),
                won.alias("won"),
                nc("user_game_win_rate_bucket").cast(pl.Float32, strict=False).alias("skill"),
                nc("user_n_games_bucket").cast(pl.Float32, strict=False).alias("n_games_bucket"),
                valid_expr.alias("_valid"),
            )
            valid = g["_valid"].to_numpy()
            rows_total += n0
            row_base += n0
            if not valid.all():
                rows_dropped += int((~valid).sum())
                g = g.filter(pl.Series(valid))
                Dm, Om, Am = Dm[valid], Om[valid], Am[valid]
                if Sm is not None:
                    Sm = Sm[valid]
                h = h[valid]
                gk = gk[valid]
            g = g.drop("_valid")
            n = g.height
            if n == 0:
                continue
            games_parts.append(g)
            mulligan_total += int(g["num_mulligans"].sum() or 0)
            # --- decks (unique within chunk)
            uh, first = np.unique(h, return_index=True)
            U = Dm[first]
            r, c = np.nonzero(U)
            deck_trip_parts.append((r.astype(np.int64) + n_decks_seen, c.astype(np.int16), U[r, c]))
            deck_h_parts.append(uh)
            n_decks_seen += len(uh)
            sizes = np.clip(Dm.sum(axis=1, dtype=np.int64), 0, 127)
            size_hist += np.bincount(sizes, minlength=128)
            # --- hands
            hand_mask = (Om > 0) | (Am > 0)
            hr, hc = np.nonzero(hand_mask)
            hands_parts.append((gk[hr], hc.astype(np.int16), Om[hr, hc], Am[hr, hc]))
            hand_viol += int(((Om.astype(np.int16) + Am) > Dm).any(axis=1).sum())
            # --- colour stats for the inferred-metadata fallback
            mc = g["main_colors"].to_numpy()
            umc, inv = np.unique(mc, return_inverse=True)
            B = (Dm > 0)
            for k, key in enumerate(umc):
                cnt = B[inv == k].sum(axis=0, dtype=np.int64)
                color_acc[key] = color_acc.get(key, 0) + cnt
            # --- pools
            if want_pools and Sm is not None:
                P = (Dm[:, nonbasic_idx].astype(np.int16) + Sm[:, nonbasic_idx])
                did = g["draft_id"].to_numpy()
                ud, dinv = np.unique(did, return_inverse=True)
                codes, mx, mn, cnt = _group_reduce(dinv, P)
                pool_parts.append((ud[codes], mx, mn, cnt))
            if progress:
                progress(min(raw.tell() / size, 0.999))
    finally:
        raw.close()

    # ------------------------------------------------------------------ assemble the shard
    tmp = shard_path(corpus_dir, fingerprint).with_name(fingerprint + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    zs = dict(compression="zstd")

    if games_parts:
        games = pl.concat(games_parts)
    else:
        games = pl.DataFrame()
    games.write_parquet(tmp / "games.parquet", **zs)

    # decks: dedupe by hash over the whole file
    if deck_h_parts:
        all_h = np.concatenate(deck_h_parts)
        uq, keep_idx = np.unique(all_h, return_index=True)
        keep = np.zeros(len(all_h), dtype=bool)
        keep[keep_idx] = True
        didx = np.concatenate([t[0] for t in deck_trip_parts])
        dcard = np.concatenate([t[1] for t in deck_trip_parts])
        dn = np.concatenate([t[2] for t in deck_trip_parts])
        m = keep[didx]
        dkey_all = (all_h >> np.uint64(1)).astype(np.int64)
        deck_cards = pl.DataFrame({"deck_key": dkey_all[didx[m]], "card": dcard[m], "n": dn[m]},
                                  schema={"deck_key": pl.Int64, "card": pl.Int16, "n": pl.Int8})
    else:
        deck_cards = pl.DataFrame(schema={"deck_key": pl.Int64, "card": pl.Int16, "n": pl.Int8})
    deck_cards.write_parquet(tmp / "deck_cards.parquet", **zs)

    # builds
    if games.height:
        builds = (games.group_by(["fmt", "event", "draft_id", "build_index", "deck_key"], maintain_order=True)
                  .agg(pl.len().cast(pl.Int32).alias("games"), pl.col("won").sum().cast(pl.Int32).alias("wins"),
                       pl.col("game_time").min().alias("first_game_time")))
    else:
        builds = pl.DataFrame(schema={"fmt": pl.String, "event": pl.String, "draft_id": pl.String,
                                      "build_index": pl.Int16, "deck_key": pl.Int64, "games": pl.Int32,
                                      "wins": pl.Int32, "first_game_time": pl.Datetime("us")})
    builds.write_parquet(tmp / "builds.parquet", **zs)

    # hands
    if hands_parts:
        hands = pl.DataFrame({
            "game_key": np.concatenate([p[0] for p in hands_parts]),
            "card": np.concatenate([p[1] for p in hands_parts]),
            "opening": np.concatenate([p[2] for p in hands_parts]),
            "drawn": np.concatenate([p[3] for p in hands_parts]),
        }, schema={"game_key": pl.Int64, "card": pl.Int16, "opening": pl.Int8, "drawn": pl.Int8})
    else:
        hands = pl.DataFrame(schema={"game_key": pl.Int64, "card": pl.Int16, "opening": pl.Int8, "drawn": pl.Int8})
    hands.write_parquet(tmp / "hands.parquet", **zs)

    # pools + audit
    pool_schema = {"draft_id": pl.String, "card": pl.Int16, "n": pl.Int8}
    audit_schema = {"draft_id": pl.String, "n_games": pl.Int32, "invariant": pl.Boolean}
    pools = pl.DataFrame(schema=pool_schema)
    audit = pl.DataFrame(schema=audit_schema)
    if pool_parts:
        ids = np.concatenate([p[0] for p in pool_parts])
        MX = np.concatenate([p[1] for p in pool_parts])
        MN = np.concatenate([p[2] for p in pool_parts])
        CT = np.concatenate([p[3] for p in pool_parts])
        uid, inv = np.unique(ids, return_inverse=True)
        codes, mx, mn, _ = _group_reduce(inv, MX)
        _, _, mn2, _ = _group_reduce(inv, MN)
        cnt = np.bincount(inv, weights=CT, minlength=len(uid)).astype(np.int64)
        invariant = (mx == mn2).all(axis=1)
        rr, cc = np.nonzero(mx > 0)
        pools = pl.DataFrame({"draft_id": uid[codes][rr], "card": nonbasic_idx[cc].astype(np.int16),
                              "n": np.clip(mx[rr, cc], 0, 127).astype(np.int8)}, schema=pool_schema)
        audit = pl.DataFrame({"draft_id": uid[codes], "n_games": cnt[codes].astype(np.int32), "invariant": invariant},
                             schema=audit_schema)
    pools.write_parquet(tmp / "pools.parquet", **zs)
    audit.write_parquet(tmp / "pool_audit.parquet", **zs)

    # colour stats
    cs_rows = [(k, int(ci), int(v[ci])) for k, v in color_acc.items() for ci in np.flatnonzero(v)]
    cs = pl.DataFrame({"main_colors": [r[0] for r in cs_rows], "card": [r[1] for r in cs_rows],
                       "n": [r[2] for r in cs_rows]}, schema={"main_colors": pl.String, "card": pl.Int32, "n": pl.Int64})
    cs.write_parquet(tmp / "colorstats.parquet", **zs)

    meta = dict(
        version=SHARD_VERSION, ok=True, fingerprint=fingerprint, path=str(path), set_code=set_code, event=ev, fmt=fmt,
        names=names, rows=int(games.height), rows_read=int(rows_total), rows_dropped=int(rows_dropped),
        n_decks=int(deck_cards["deck_key"].n_unique()) if deck_cards.height else 0,
        deck_size_hist={str(i): int(v) for i, v in enumerate(size_hist) if v},
        hand_violations=int(hand_viol), missing_columns=missing_meta,
        pool_games=int(audit["n_games"].sum()) if audit.height else 0,
        pool_events=int(audit.height), pool_invariant=int(audit["invariant"].sum()) if audit.height else 0,
        seconds=round(time.time() - t0, 3),
    )
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    final = shard_path(corpus_dir, fingerprint)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    if progress:
        progress(1.0)
    return meta


def _q(field: str) -> str:
    if any(ch in field for ch in ',"\n\r'):
        return '"' + field.replace('"', '""') + '"'
    return field
