"""Draft mode guardrails G2 (Sealed unchanged) and G3 (calculation time): docs/draft_mode/RULE.md.

    python scripts/draft_sealed_identity.py SET SRC OUT [--jobs N] [--keep-homes]     (one set; OUT/identity_SET.json + a summary line)
    python scripts/draft_sealed_identity.py --report DIR                               (all identity_*.json of DIR -> DIR/draft_sealed_identity.md/.json)

SRC = a scratch home with sets/SET/{corpus,cache,registry.json} (only read). For the set, two full `pipeline.calculate` runs in two scratch homes
OUT/SET/home_off and home_on (each a COPY of SRC's set; same seed, the app's defaults; `draft_formats=False` / `draft_formats=True` with the default
draft_boot = 16), each in its own process. Then:
    * every array of draws.npz whose key belongs to a Sealed format (bo1__*, bo3__*) must be identical (np.array_equal, NaN == NaN); the keys that
      differ are listed (a Sealed key missing on one side differs);
    * `SetModel.logit` of 6 decks of the set must agree within 1e-9 for bo1 / bo3, at draw 0 and over all draws. The decks are the registered decks
      (first build) of 6 Sealed events of pools.parquet, drawn at random with seed 0 (a pool is no 40-card deck; the deck the player built from it is).
PASS only if nothing differs; any difference BLOCKs. The wall time of both calculations is G3's calculation time (reported, not blocking).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SEALED_FORMATS = ("bo1", "bo3")
TOL = 1e-9
N_POOLS = 6


# =====================================================================================================================
# pure comparisons (tested without any model)
# =====================================================================================================================

def is_sealed_key(key: str) -> bool:
    """A draws.npz key of a Sealed format: `bo1__beta`, `bo3__ref`, ..."""
    return key.split("__", 1)[0] in SEALED_FORMATS


def compare_draws(a: dict, b: dict) -> dict:
    """Sealed arrays of two draws.npz contents (dict key -> array): identical or not, per key. A Sealed key present on one side only differs."""
    keys = sorted({k for k in list(a) + list(b) if is_sealed_key(k)})
    differ, n_equal = [], 0
    for k in keys:
        if k not in a or k not in b:
            differ.append(dict(key=k, why="missing in the " + ("switch-off" if k not in a else "switch-on") + " run"))
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        eq = x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" and y.dtype.kind in "fc" else np.array_equal(x, y))
        if eq:
            n_equal += 1
        else:
            d = float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.shape == y.shape and x.dtype.kind in "fiub" else None
            differ.append(dict(key=k, why="values differ" if x.shape == y.shape else f"shape {x.shape} vs {y.shape}", max_abs_diff=d))
    return dict(n_keys=len(keys), n_equal=n_equal, differ=differ, identical=bool(keys) and not differ)


def compare_logits(rows: list[dict], tol: float = TOL) -> dict:
    """rows: per (deck, fmt) {deck, fmt, d0_off, d0_on, draws_off, draws_on}. identical = every |difference| <= tol (draw 0 and every draw)."""
    worst0, worstd, bad = 0.0, 0.0, []
    for r in rows:
        d0 = abs(float(r["d0_off"]) - float(r["d0_on"]))
        dd = float(np.max(np.abs(np.asarray(r["draws_off"], float) - np.asarray(r["draws_on"], float)))) if len(r["draws_off"]) == len(r["draws_on"]) else float("inf")
        worst0, worstd = max(worst0, d0), max(worstd, dd)
        if d0 > tol or dd > tol:
            bad.append(dict(deck=r["deck"], fmt=r["fmt"], diff0=d0, diff_draws=dd))
    return dict(n=len(rows), max_abs_diff_draw0=worst0, max_abs_diff_draws=worstd, tol=tol, bad=bad, identical=bool(rows) and not bad)


def verdict(res: dict) -> str:
    """PASS if the Sealed draws are identical and the logits agree; BLOCK otherwise (also when nothing could be compared)."""
    return "PASS" if res["draws"]["identical"] and res["logits"]["identical"] else "BLOCK"


# =====================================================================================================================
# the two calculations
# =====================================================================================================================

def _copy_set(src: Path, home: Path, set_code: str) -> None:
    dst = home / "sets" / set_code
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for sub in ("corpus", "cache"):
        shutil.copytree(src / "sets" / set_code / sub, dst / sub)          # a COPY, never a link
    shutil.copy(src / "sets" / set_code / "registry.json", dst / "registry.json")


def _calc_worker(set_code: str, home: str, draft: bool, result: str, jobs: int | None) -> int:
    os.environ["SEALED_LAB_HOME"] = home
    from sealed_lab.model import pipeline

    t0 = time.time()
    meta = pipeline.calculate(set_code, progress=lambda f, m: None, n_jobs=jobs, draft_formats=draft)
    Path(result).write_text(json.dumps(dict(wall_seconds=time.time() - t0, meta_seconds=meta.get("seconds"), formats=meta.get("formats"),
                                            n_boot=meta.get("n_boot"), seed=meta.get("seed"))), encoding="utf-8")
    return 0


def _run_calc(set_code: str, home: Path, draft: bool, result: Path, jobs: int | None, log: Path) -> dict:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--calc", set_code, str(home), "on" if draft else "off", str(result)]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    if rc != 0:
        raise SystemExit(f"calculation ({'on' if draft else 'off'}) failed, rc {rc}: see {log}")
    return json.loads(result.read_text(encoding="utf-8"))


def _load_draws(home: Path, set_code: str) -> dict:
    with np.load(home / "sets" / set_code / "model" / "draws.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sample_decks(src_home: Path, set_code: str, n: int = N_POOLS, seed: int = 0) -> list[dict]:
    """n registered decks (first build) of Sealed events drawn at random (seed 0) among the events of pools.parquet: [{draft_id, deck: {card_id: n}}]."""
    import polars as pl

    cd = src_home / "sets" / set_code / "corpus"
    games = pl.read_parquet(cd / "games.parquet", columns=["draft_id", "fmt"])
    sealed = set(games.filter(pl.col("fmt").is_in(list(SEALED_FORMATS)))["draft_id"].unique().to_list())
    pools = set(pl.read_parquet(cd / "pools.parquet", columns=["draft_id"])["draft_id"].unique().to_list())
    builds = pl.read_parquet(cd / "builds.parquet")
    ok = sorted(sealed & pools & set(builds["draft_id"].to_list()), key=str)
    rng = np.random.default_rng(seed)
    pick = [ok[i] for i in sorted(rng.choice(len(ok), size=min(n, len(ok)), replace=False))] if ok else []
    dc = pl.read_parquet(cd / "deck_cards.parquet")
    out = []
    for d in pick:
        key = builds.filter(pl.col("draft_id") == d).sort("deck_key")["deck_key"][0]
        sub = dc.filter(pl.col("deck_key") == key)
        out.append(dict(draft_id=str(d), deck={int(c): int(k) for c, k in zip(sub["card_id"].to_list(), sub["n"].to_list())}))
    return out


def _logit_rows(homes: dict, set_code: str, decks: list[dict]) -> list[dict]:
    import split_half_search as sh

    models = {}
    for tag, h in homes.items():
        with sh._home(str(h)):
            from sealed_lab.model.artifact import SetModel

            models[tag] = SetModel.load(set_code)
    rows = []
    for d in decks:
        for f in SEALED_FORMATS:
            if f not in models["off"].formats or f not in models["on"].formats:
                continue
            vals = {}
            for tag, h in homes.items():
                with sh._home(str(h)):
                    vals[tag] = (float(models[tag].logit(d["deck"], f)), np.asarray(models[tag].logit(d["deck"], f, draws=True), float).tolist())
            rows.append(dict(deck=d["draft_id"], fmt=f, d0_off=vals["off"][0], d0_on=vals["on"][0], draws_off=vals["off"][1], draws_on=vals["on"][1]))
    return rows


def run_set(set_code: str, src: Path, out: Path, jobs: int | None, keep_homes: bool = False) -> dict:
    set_code = set_code.upper()
    root = out / set_code
    root.mkdir(parents=True, exist_ok=True)
    homes, calc = {}, {}
    for tag, draft in (("off", False), ("on", True)):
        homes[tag] = root / f"home_{tag}"
        _copy_set(src, homes[tag], set_code)
        calc[tag] = _run_calc(set_code, homes[tag], draft, root / f"calc_{tag}.json", jobs, root / f"calc_{tag}.log")
        print(f"{set_code} switch {tag}: {calc[tag]['wall_seconds']:.0f}s", flush=True)
    draws = compare_draws(_load_draws(homes["off"], set_code), _load_draws(homes["on"], set_code))
    decks = sample_decks(src, set_code)
    logits = compare_logits(_logit_rows(homes, set_code, decks))
    res = dict(set=set_code, draws=draws, logits=logits, n_decks=len(decks), calc=calc, seconds_off=calc["off"]["wall_seconds"],
               seconds_on=calc["on"]["wall_seconds"], formats_on=calc["on"].get("formats"), seed_same=calc["off"].get("seed") == calc["on"].get("seed"))
    res["status"] = verdict(res)
    (out / f"identity_{set_code}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not keep_homes:
        for h in homes.values():
            shutil.rmtree(h, ignore_errors=True)
    print(f"G2 {set_code}: {res['status']} | Sealed draws arrays equal {draws['n_equal']}/{draws['n_keys']}, logits max diff "
          f"{logits['max_abs_diff_draw0']:.2e} (draw 0) / {logits['max_abs_diff_draws']:.2e} (draws) over {logits['n']} deck x format | "
          f"calc {res['seconds_off']:.0f}s off, {res['seconds_on']:.0f}s on", flush=True)
    return res


# =====================================================================================================================
# the all-sets report
# =====================================================================================================================

def report(d: Path) -> str:
    res = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(d).glob("identity_*.json"))]
    if not res:
        raise SystemExit(f"no identity_*.json in {d}")
    L = ["# Draft mode G2 (Sealed unchanged) and G3 (calculation time)", "",
         "Per set: two full calculations from the same corpus copy, same seed, the app's defaults; the switch off / on (draft_boot 16). G2: every Sealed "
         "draws array identical and `SetModel.logit` within 1e-9 (draw 0 and all draws) on 6 registered decks of Sealed events; any difference blocks.", "",
         "| set | Sealed draws arrays equal | differing arrays | logit max diff (draw 0) | logit max diff (draws) | decks x formats | calc s off | calc s on | on / off | status |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in res:
        dr, lg = r["draws"], r["logits"]
        L.append(f"| {r['set']} | {dr['n_equal']}/{dr['n_keys']} | {', '.join(x['key'] for x in dr['differ']) or '-'} | {lg['max_abs_diff_draw0']:.1e} | "
                 f"{lg['max_abs_diff_draws']:.1e} | {lg['n']} | {r['seconds_off']:.0f} | {r['seconds_on']:.0f} | {r['seconds_on'] / max(r['seconds_off'], 1e-9):.2f} | "
                 f"**{r['status']}** |")
    allpass = all(r["status"] == "PASS" for r in res)
    L += ["", f"**G2: {'PASS' if allpass else 'BLOCK'}** ({sum(r['status'] == 'PASS' for r in res)}/{len(res)} sets identical)",
          f"G3 (reported, not blocking): total calculation time off {sum(r['seconds_off'] for r in res) / 60:.0f} min, on "
          f"{sum(r['seconds_on'] for r in res) / 60:.0f} min over {len(res)} sets."]
    md = "\n".join(L) + "\n"
    (Path(d) / "draft_sealed_identity.md").write_text(md, encoding="utf-8")
    (Path(d) / "draft_sealed_identity.json").write_text(json.dumps(dict(g2="PASS" if allpass else "BLOCK", sets=res), indent=1), encoding="utf-8")
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", nargs="?")
    ap.add_argument("src", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--keep-homes", action="store_true")
    ap.add_argument("--report", default=None, metavar="DIR", help="aggregate DIR/identity_*.json into DIR/draft_sealed_identity.md / .json")
    ap.add_argument("--calc", nargs=4, default=None, metavar=("SET", "HOME", "on|off", "RESULT"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.calc:
        return _calc_worker(a.calc[0], a.calc[1], a.calc[2] == "on", a.calc[3], a.jobs)
    if a.report:
        print(report(Path(a.report)))
        return 0
    if not (a.set and a.src and a.out):
        ap.error("SET SRC OUT are required (or --report DIR)")
    res = run_set(a.set, Path(a.src).resolve(), Path(a.out).resolve(), a.jobs, a.keep_homes)
    return 0 if res["status"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
