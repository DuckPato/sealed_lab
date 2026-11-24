"""Build the folder of ready-made sets that goes into a release zip (maintainer's computer, before `release_sign.py sign --bundle`).

    python tools/bundle_sets.py --out DIR [--sets FRA HOB ...] [--n 5] [--home APP_HOME] [--skip-written] [--overwrite]

Takes the `--n` newest sets (by release date, from the cached Scryfall set list) the app home has, or exactly `--sets`, and writes

    DIR/bundle.json                         {"bundle_version", "created_at", "sets": {CODE: {"built_at", "release_date"}}}
    DIR/<CODE>/model/*                      the model
    DIR/<CODE>/corpus/*.parquet, manifest.json   the data tables (the manifest with every path reduced to a file name; no _shards)
    DIR/<CODE>/cache/report_*.json scryfall.json creature_support*.json archetype_proj_*.json   what the tabs need without a calculation
    DIR/<CODE>/written/<CODE>.md|json       the written report

It REFUSES (exit 1) when a chosen set is stale, has no model, or lacks a written report that is current and passes the checker
(`--skip-written` ships such a set without a written report and says so), and when any file holds a personal string
(`export_public.py`'s scan).  It only reads the app home.  `--home` defaults to the app's own home (SEALED_LAB_HOME or the usual folder).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEXT_EXT = {".json", ".md", ".html", ".txt", ".csv"}


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _export_public():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dev_export_public", ROOT / "tools" / "export_public.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_sets(codes: list[str] | None, n: int, known: list[str], dates: dict[str, str]) -> list[str]:
    """`codes` as given (must exist), else the `n` newest of `known` by release date (an undated set is never picked unasked)."""
    if codes:
        codes = [c.upper() for c in codes]
        missing = [c for c in codes if c not in known]
        if missing:
            _die(f"not tracked in this app home: {', '.join(missing)}")
        return codes
    dated = sorted((c for c in known if c in dates), key=lambda c: (dates[c], c), reverse=True)
    if len(dated) < min(n, len(known)):
        _die("release dates are unknown for " + ", ".join(sorted(set(known) - set(dated))) + " (start the app once online, or name the sets with --sets)")
    return dated[:n]


def problems_of(code: str, skip_written: bool) -> tuple[list[str], list[str]]:
    """(refusals, notes) for one set: stale / no model / calculating, and the written report."""
    from sealed_lab import api, report_check, written_report

    bad: list[str] = []
    notes: list[str] = []
    st = api.set_status(code)
    if not st.has_model:
        bad.append(f"{code}: no model")
    if st.stale:
        bad.append(f"{code}: stale ({'; '.join(st.pending) or 'recalculate it'})")
    if api._calculating_elsewhere(code) is not None:
        bad.append(f"{code}: is being calculated right now")
    if not st.has_corpus:
        bad.append(f"{code}: no data tables")
    w = written_report.load(code)
    why = ""
    if w is None:
        why = "no written report"
    else:
        meta = w["meta"]
        if written_report.outdated(meta, written_report.current_stamps(code)):
            why = "the written report is outdated (the model or its statistics changed since it was written)"
        else:
            probs = report_check.check(w["text"], meta.get("facts") or {})
            if probs:
                why = f"the written report fails the checker ({len(probs)} problem(s), first: {probs[0]})"
    if why:
        (notes if skip_written else bad).append(f"{code}: {why}" + ("; shipped without a written report" if skip_written else ""))
    return bad, notes


def _copy_set(code: str, out: Path, with_written: bool) -> dict:
    from sealed_lab import bundled, config, written_report

    dest = out / code
    model = config.model_dir(code)
    shutil.copytree(model, dest / "model")
    (dest / "corpus").mkdir(parents=True)
    cd = config.corpus_dir(code)
    for p in sorted(cd.glob("*.parquet")):
        shutil.copy2(p, dest / "corpus" / p.name)
    man = json.loads((cd / "manifest.json").read_text(encoding="utf-8"))
    (dest / "corpus" / "manifest.json").write_text(json.dumps(bundled.sanitize_manifest(man), indent=1, ensure_ascii=False), encoding="utf-8")
    (dest / "cache").mkdir()
    for pat in bundled.CACHE_PATTERNS:
        for p in sorted(config.cache_dir(code).glob(pat)):
            if p.is_file():
                shutil.copy2(p, dest / "cache" / p.name)
    if with_written:
        (dest / "written").mkdir()
        for ext in ("md", "json"):
            shutil.copy2(written_report.folder() / f"{code}.{ext}", dest / "written" / f"{code}.{ext}")
    meta = json.loads((model / "meta.json").read_text(encoding="utf-8"))
    return dict(built_at=meta.get("built_at"))


def scan_folder(out: Path) -> list[str]:
    """Personal strings in any file of the bundle: text files through export_public.scan, every other file as bytes (in overlapping
    chunks, so a path split between two chunks is still found)."""
    ep = _export_public()
    pats = ep.deny_patterns()
    files = sorted(p for p in out.rglob("*") if p.is_file())
    text = [p.relative_to(out).as_posix() for p in files if p.suffix.lower() in TEXT_EXT]
    hits = ep.scan(text, root=out, patterns=pats)
    for p in files:
        if p.suffix.lower() in TEXT_EXT:
            continue
        rel = p.relative_to(out).as_posix()
        with open(p, "rb") as fh:
            tail = b""
            while True:
                chunk = fh.read(8 << 20)
                if not chunk:
                    break
                hits += ep._blob_hits(rel, "bytes", tail + chunk, pats)
                tail = chunk[-4096:]
        hits += _decoded_hits(p, rel, ep, pats)
    return hits


def _decoded_hits(p: Path, rel: str, ep, pats) -> list[str]:
    """Compression hides text from the byte scan: the string columns of a .parquet and the string / object arrays of a .npz are
    decoded and scanned too."""
    texts: list[str] = []
    try:
        if p.suffix.lower() == ".parquet":
            import polars as pl

            df = pl.read_parquet(p)
            for c in df.columns:
                if df[c].dtype == pl.String:
                    texts += [v for v in df[c].unique().to_list() if v]
            texts += list(df.columns)
        elif p.suffix.lower() == ".npz":
            import numpy as np

            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    texts.append(k)
                    a = z[k]
                    if a.dtype.kind in "US":
                        texts += [str(v) for v in a.ravel().tolist()]
    except Exception as e:  # noqa: BLE001 - an unreadable file is itself a reason to stop
        return [f"{rel}: could not be decoded for the personal-string scan ({e})"]
    if not texts:
        return []
    return ep._blob_hits(rel, "decoded", "\n".join(texts).encode("utf-8"), pats)


def build(out: Path, codes: list[str] | None, n: int, skip_written: bool, overwrite: bool = False) -> dict:
    from sealed_lab import __version__
    from sealed_lab.ingest import registry, setinfo

    known = registry.known_sets()

    dates = setinfo.release_dates(known, fetch=False)          # the cached list only: this tool never writes to the app home
    chosen = pick_sets(codes, n, known, dates)
    bad: list[str] = []
    notes: dict[str, str] = {}
    for c in chosen:
        b, nt = problems_of(c, skip_written)
        bad += b
        if nt:
            notes[c] = "; ".join(nt)
    if bad:
        _die("not building the bundle:\n  " + "\n  ".join(bad))
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            _die(f"{out} is not empty (use another folder, or --overwrite to replace it)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    sets: dict[str, dict] = {}
    try:
        for c in chosen:
            info = _copy_set(c, out, with_written=c not in notes)
            info["release_date"] = dates.get(c, "")
            sets[c] = info
        (out / "bundle.json").write_text(json.dumps(dict(bundle_version=__version__, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                         sets=sets), indent=1), encoding="utf-8")
        hits = scan_folder(out)
        if hits:
            _die("personal strings found in the bundle, nothing kept:\n  " + "\n  ".join(hits[:30]))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return dict(sets=sets, notes=notes)


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", nargs="+")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--home", type=Path, help="app home to read from (default: the app's own)")
    ap.add_argument("--skip-written", action="store_true", help="ship a set without a written report instead of refusing")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.home:
        os.environ["SEALED_LAB_HOME"] = str(a.home)
    res = build(a.out, a.sets, a.n, a.skip_written, a.overwrite)
    total = 0
    for c, info in res["sets"].items():
        s = _size(a.out / c)
        total += s
        print(f"{c}: built {info['built_at']}, released {info['release_date'] or '?'}, {s / 1e6:.1f} MB"
              + (f"  (no written report: {res['notes'][c]})" if c in res["notes"] else ""))
    print(f"bundle: {len(res['sets'])} sets, {total / 1e6:.1f} MB in {a.out}; scan clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
