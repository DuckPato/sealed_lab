"""Cache of the hold-out validation results (component tests, recency split, deployed combination).

A re-calculation with unchanged data, code and settings repeats the hold-out tests of every component bit for bit (they are
deterministic functions of their inputs).  Each test result is therefore stored under a key that covers EVERYTHING the result
depends on, so a stale result can never be reused:

    * the code: a hash of the source of every module of `sealed_lab/model`, of `config.py`, and the Python / numpy / scipy /
      polars versions;
    * the data: a content hash of the game table used (all columns the model reads), of the corpus' hands table, of the card
      table, and of every deck-space matrix the test fits (card copies, structure features per regime, castability blocks,
      attribute loadings, pair / archetype columns, ...);
    * the settings of that test: format, seed, bootstrap replicates, half-life, opponent encodings, deployed component list and
      their prior arrays, ...;
    * for a component, the key of the plain-chain result it is paired with.

A hit returns the stored numbers and the date they were computed; the report says so ("cached from <date>").  Any
difference in any input is a different key, i.e. a miss and a full recomputation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import scipy
import scipy.sparse as sp

log = logging.getLogger(__name__)

CACHE_VERSION = 1          # bump to invalidate every stored result by hand
KEEP_PER_NAME = 6          # stored versions kept per (kind, name): different settings / data states can alternate without thrashing

_code_sig: str | None = None


# ---------------------------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------------------------

def code_signature() -> str:
    """Hash of the model source code + library versions: any edit of the numerics invalidates every stored result."""
    global _code_sig
    if _code_sig is None:
        h = hashlib.sha256()
        root = Path(__file__).resolve().parent
        files = sorted(root.glob("*.py")) + [root.parent / "config.py"]
        for p in files:
            h.update(p.name.encode())
            h.update(p.read_bytes())
        h.update(json.dumps([sys.version, np.__version__, scipy.__version__, pl.__version__, CACHE_VERSION]).encode())
        _code_sig = h.hexdigest()
    return _code_sig


def feed(h, obj) -> None:
    """Add an object (arrays, sparse matrices, frames, containers, scalars) to the hash `h` in a canonical, typed way."""
    if obj is None:
        h.update(b"N")
    elif isinstance(obj, (bool, np.bool_)):
        h.update(b"b1" if obj else b"b0")
    elif isinstance(obj, (int, np.integer)):
        h.update(b"i" + str(int(obj)).encode())
    elif isinstance(obj, (float, np.floating)):
        h.update(b"f" + repr(float(obj)).encode())
    elif isinstance(obj, str):
        h.update(b"s" + str(len(obj)).encode() + b":" + obj.encode("utf-8", "surrogatepass"))
    elif isinstance(obj, bytes):
        h.update(b"y" + str(len(obj)).encode() + b":" + obj)
    elif isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        if a.dtype == object:
            h.update(b"O" + json.dumps([str(x) for x in a.ravel().tolist()]).encode())
            h.update(str(a.shape).encode())
        else:
            h.update(b"A" + str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.reshape(-1).view(np.uint8) if a.size else b"")
    elif sp.issparse(obj):
        m = obj.tocsr()
        h.update(b"S" + str(m.shape).encode())
        feed(h, m.indptr)
        feed(h, m.indices)
        feed(h, m.data)
    elif isinstance(obj, pl.DataFrame):
        feed_frame(h, obj)
    elif isinstance(obj, dict):
        h.update(b"D" + str(len(obj)).encode())
        for k in sorted(obj, key=str):
            feed(h, str(k))
            feed(h, obj[k])
    elif isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
        h.update(b"L" + str(len(items)).encode())
        for x in items:
            feed(h, x)
    else:
        feed(h, repr(obj))


def feed_frame(h, df: pl.DataFrame) -> None:
    """Content hash of a polars frame (column names, dtypes, every value)."""
    h.update(b"F" + str(df.shape).encode() + json.dumps([[c, str(t)] for c, t in df.schema.items()]).encode())
    if df.height == 0:
        return
    try:
        feed(h, df.hash_rows(seed=0).to_numpy())
    except Exception:  # noqa: BLE001 - nested dtypes: hash a string rendering column by column
        for c in df.columns:
            col = df[c]
            try:
                feed(h, col.cast(pl.String).fill_null("\x00null").to_numpy().astype("U"))
            except Exception:  # noqa: BLE001
                feed(h, [str(x) for x in col.to_list()])


def digest(*objs) -> str:
    h = hashlib.sha256()
    for o in objs:
        feed(h, o)
    return h.hexdigest()


def file_digest(path: Path | str) -> str:
    """SHA-256 of a file's bytes (streamed)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
    except OSError:
        return "missing"
    return h.hexdigest()


def space_digest(space) -> str:
    """Content hash of a DeckSpace: every array a fit on it reads (memoised on the object; spaces are never mutated)."""
    if space is None:
        return "none"
    hit = getattr(space, "_vc_digest", None)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    for name in ("keys", "phi", "mu", "sd", "nb_ids", "col_of_card", "D", "w", "attr_names", "attr_mu", "attr_sd", "attr_load",
                 "pair_ab", "arch_ca", "deck_arch", "n_cards_total"):
        feed(h, name)
        feed(h, getattr(space, name, None))
    feed(h, "regime_phi")
    feed(h, getattr(space, "regime_phi", None))
    feed(h, "cast_cards")
    feed(h, getattr(space, "cast_cards", None))
    out = h.hexdigest()
    try:
        space._vc_digest = out
    except Exception:  # noqa: BLE001
        pass
    return out


def stage_ids_digest(df: pl.DataFrame) -> str:
    return digest(df)


# ---------------------------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------------------------

def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def normalise(payload):
    """What a stored-and-reloaded payload looks like (JSON round trip): used for fresh results too, so a hit and a miss hand the
    rest of the pipeline exactly the same kind of object."""
    return json.loads(json.dumps(payload, default=_default))


class ValCache:
    """Key -> payload store in `<cache dir>/validation/`.  `base` = the parts of the key shared by every entry of this run
    (code, data hashes, ...)."""

    def __init__(self, directory: Path | None, base: dict, prefix: str = ""):
        self.dir = None if directory is None else Path(directory) / "validation"
        self.base = base
        self.prefix = prefix          # names of this cache's entries start with it (the draft formats' entries never evict the Sealed ones)
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def key(self, kind: str, name: str, **parts) -> str:
        h = hashlib.sha256()
        feed(h, dict(version=CACHE_VERSION, code=code_signature(), base=self.base, kind=kind, name=self.prefix + name, parts=parts))
        return h.hexdigest()

    def _stem(self, kind: str, name: str, key: str) -> Path:
        return self.dir / f"{kind}_{self.prefix}{name}_{key[:24]}"

    def load(self, kind: str, name: str, key: str) -> tuple[dict, dict] | None:
        """(payload, arrays) of a stored entry whose FULL key equals `key`, else None."""
        if self.dir is None:
            return None
        stem = self._stem(kind, name, key)
        try:
            d = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            if d.get("key") != key or d.get("version") != CACHE_VERSION:
                raise ValueError("key mismatch")
            arrays = {}
            if d.get("arrays"):
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
            payload = d["payload"]
            payload["_created"] = d["created"]
            self.hits += 1
            return payload, arrays
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - a damaged entry is a miss
            log.warning("validation cache entry %s unreadable; recomputing", stem.name)
        self.misses += 1
        return None

    def save(self, kind: str, name: str, key: str, payload: dict, arrays: dict | None = None) -> None:
        if self.dir is None:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            stem = self._stem(kind, name, key)
            if arrays:
                tmp = stem.with_name(stem.name + ".tmp.npz")
                np.savez(tmp, **arrays)
                os.replace(tmp, stem.with_suffix(".npz"))
            doc = dict(version=CACHE_VERSION, key=key, kind=kind, name=name, arrays=bool(arrays),
                       created=datetime.now(timezone.utc).isoformat(timespec="seconds"), payload=payload)
            tmp = stem.with_name(stem.name + ".tmp.json")
            tmp.write_text(json.dumps(doc, default=_default), encoding="utf-8")
            os.replace(tmp, stem.with_suffix(".json"))
            self._prune(kind, name, keep=stem.name)
        except OSError:
            log.warning("could not write the validation cache")

    def _prune(self, kind: str, name: str, keep: str) -> None:
        try:
            old = sorted((p for p in self.dir.glob(f"{kind}_{self.prefix}{name}_*.json") if p.stem != keep and not p.name.endswith(".tmp.json")),
                         key=lambda p: p.stat().st_mtime, reverse=True)
            for p in old[max(0, KEEP_PER_NAME - 1):]:
                p.unlink(missing_ok=True)
                p.with_suffix(".npz").unlink(missing_ok=True)
        except OSError:
            pass


def nice_date(iso: str | None) -> str:
    """'2026-10-26 01:30' (local time) of an ISO timestamp."""
    if not iso:
        return "?"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(iso)
