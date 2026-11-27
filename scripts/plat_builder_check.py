"""Does a builder whose model learns only from Platinum+ drafters build different decks? (descriptive; nothing ships)

    python scripts/plat_builder_check.py OUT [--sets TMT ...] [--jobs N] [--stage prepare|calc|build|all]
                                         [--source SCRATCH_HOME] [--pools-from docs/builder_scoreboard/search_check.json]

Rank exists only in Draft games (PremierDraft, PickTwoDraft); Sealed games and TradDraft carry none. The "plat" model is therefore the
normal model except that its draft stage (the first stage of the card-value chain) sees only the Draft games whose rank is platinum,
diamond or mythic; Sealed games and TradDraft are unchanged (rows with fmt != "draft" are kept whole).

Stages (each resumable; finished work is skipped):

  prepare  per set: OUT/base/SET-home and OUT/plat/SET-home = copies (never links) of the source home's root *.json plus sets/SET.
           In plat, sets/SET/corpus/games.parquet is rewritten (atomically) keeping fmt != "draft" OR lower(rank) in (platinum, diamond,
           mythic); manifest.json and the shards stay untouched, so the corpus "up to date" quick path keeps the filtered file.
           The old model folder is deleted in both homes. Draft game counts before / after, per event and tier, go to OUT/prepare/SET.json.
  calc     per set and home, the same call the CLI `calc` makes (`api.calculate_set`) with identical options in both homes:
           quick=True, pickprior="off" (the CLI cannot pass it), fetch_scryfall=False, n_boot=16, everything else default.
           Run in a child process of this script (`--calc-child`) with SEALED_LAB_HOME set to the home. Logs: OUT/logs.
           Asserts that the plat corpus was not re-ingested (row count of games.parquet, manifest unchanged, log line).
  build    per held-out pool of search_check.json (pool read from that home's corpus/pools.parquet): `builder.build(model, pool, fmt,
           n_decks=1, families=True, n_jobs=1)` in worker processes with SEALED_LAB_HOME set per task; then cross scoring (each model
           scores the other's deck). Writes OUT/plat_builder.json and OUT/plat_builder.md.

Never writes outside OUT; the source home and the 17lands raw data are only read. The validation / phi caches copied with the set folder
are content-keyed (games digest) and stay valid in base; plat gets its own validation entries.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALL_SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
DEFAULT_SOURCE = Path(os.environ.get("SEALED_LAB_SCRATCH", tempfile.gettempdir())) / "pphome"   # a byte copy of the app home
DEFAULT_POOLS = ROOT / "docs" / "builder_scoreboard" / "search_check.json"
PLAT = ("platinum", "diamond", "mythic")
VARIANTS = ("base", "plat")
PP = 0.2475 * 100.0              # logit difference -> win-rate points at a 55% deck


# ------------------------------------------------------------------------------------------------ helpers
def home_of(out: Path, variant: str, sc: str) -> Path:
    return out / variant / f"{sc}-home"


def games_path(home: Path, sc: str) -> Path:
    return home / "sets" / sc / "corpus" / "games.parquet"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_links(root: Path) -> list[str]:
    """Symlinks, junctions and multiply-linked files under `root` (must be none: copies, never links)."""
    bad = []
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = Path(dp) / n
            try:
                if p.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(p)):
                    bad.append(str(p))
                elif p.is_file() and os.stat(p).st_nlink > 1:
                    bad.append(str(p))
            except OSError:
                bad.append(str(p))
    return bad


def sha(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_out_path(out: Path) -> None:
    la = os.environ.get("LOCALAPPDATA")
    for bad in (Path(la) / "sealed_lab" if la else None, Path(os.environ["SEALED_LAB_RAW_DIR"]) if os.environ.get("SEALED_LAB_RAW_DIR") else None):
        if bad is not None:
            try:
                out.resolve().relative_to(bad.resolve())
            except ValueError:
                continue
            raise SystemExit(f"refusing to work under {bad}")


# ------------------------------------------------------------------------------------------------ prepare
def _counts(df) -> list[dict]:
    import polars as pl

    d = df.filter(pl.col("fmt") == "draft")
    g = d.group_by("event", "rank").len().sort("event", "rank")
    return [dict(event=e, rank=r, games=int(n)) for e, r, n in g.iter_rows()]


def fix_signature_child(sc: str) -> int:
    """Child (SEALED_LAB_HOME = a work home): store in manifest.json the signature the quick path will compute for this home.

    The scratch home's stored signature includes the stat (size:mtime_ns) of the Scryfall cache file, which a byte copy does not
    keep, so the corpus would be rebuilt from the shards (rewriting games.parquet unfiltered). Only the "signature" field changes;
    shards, tables and every other manifest field stay as they were. Done identically in base and plat.
    """
    from sealed_lab.ingest import corpus, registry

    files = registry.files(sc)
    games = [corpus._refresh(f) for f in files if f.kind == "game"]
    others = [f for f in files if f.kind in ("cards", "scryfall")]
    new = corpus._signature(sc, [f.fingerprint for f in games], [f.fingerprint for f in others], False)
    from sealed_lab import config

    man = config.corpus_dir(sc) / "manifest.json"
    d = json.loads(man.read_text(encoding="utf-8"))
    old = d.get("signature")
    if old != new:
        d["signature"] = new
        write_json(man, d)
    print(json.dumps(dict(old=old, new=new)))
    return 0


def fix_signature(sc: str, home: Path) -> dict:
    env = dict(os.environ, SEALED_LAB_HOME=str(home), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--fix-signature-child", sc], env=env, capture_output=True,
                       text=True, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"{sc}: signature fix failed: {r.stderr[-800:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def prepare_set(sc: str, out: Path, source: Path) -> dict:
    import polars as pl

    marker = out / "prepare" / f"{sc}.json"
    homes = {v: home_of(out, v, sc) for v in VARIANTS}
    if marker.exists() and all(h.exists() for h in homes.values()):
        print(f"prepare {sc}: done earlier, skipped", flush=True)
        return read_json(marker)
    t0 = time.time()
    sig_fix = {}
    src_set = source / "sets" / sc
    if not src_set.is_dir():
        raise SystemExit(f"{sc}: no {src_set}")
    for v, home in homes.items():
        if home.exists():
            shutil.rmtree(home)
        part = home.with_name(home.name + ".partial")
        if part.exists():
            shutil.rmtree(part)
        part.mkdir(parents=True)
        for f in source.glob("*.json"):
            shutil.copy2(f, part / f.name)
        shutil.copytree(src_set, part / "sets" / sc, symlinks=False)
        bad = find_links(part)
        if bad:
            raise SystemExit(f"{sc} {v}: links found in the copy: {bad[:5]}")
        shutil.rmtree(part / "sets" / sc / "model", ignore_errors=True)       # fresh model from the calc
        os.replace(part, home)
        sig_fix[v] = fix_signature(sc, home)
    gp = games_path(homes["plat"], sc)
    games = pl.read_parquet(gp)
    before = _counts(games)
    keep = (pl.col("fmt") != "draft") | pl.col("rank").str.to_lowercase().is_in(list(PLAT)).fill_null(False)
    filt = games.filter(keep)
    after = _counts(filt)
    tmp = gp.with_name("games.parquet.tmp")
    filt.write_parquet(tmp)
    os.replace(tmp, gp)
    nb = sum(r["games"] for r in before)
    na = sum(r["games"] for r in after)
    info = dict(set=sc, draft_games_before=nb, draft_games_after=na, draft_share_kept=na / nb if nb else None,
                rows_before=int(games.height), rows_after=int(filt.height),
                other_rows_unchanged=int(games.filter(pl.col("fmt") != "draft").height) == int(filt.filter(pl.col("fmt") != "draft").height),
                by_event_rank_before=before, by_event_rank_after=after,
                manifest_signature_fix=sig_fix, source=str(source), seconds=time.time() - t0)
    write_json(marker, info)
    print(f"prepare {sc}: draft games {nb} -> {na} ({info['draft_share_kept']:.1%}), {info['seconds']:.0f}s", flush=True)
    return info


# ------------------------------------------------------------------------------------------------ calc
def calc_child(sc: str) -> int:
    """Child process: the `cli calc` call with identical options (quick, pickprior off, no Scryfall, boot 16)."""
    from sealed_lab import api

    last = [None]

    def prog(frac: float, msg: str) -> None:
        if msg != last[0]:
            last[0] = msg
            print(f"[{time.strftime('%H:%M:%S')}] {frac:5.1%} {msg}", flush=True)

    api.calculate_set(sc, progress=prog, n_boot=16, fetch_scryfall=False, quick=True, pickprior="off")
    return 0


def calc_set(sc: str, out: Path, jobs: int | None) -> None:
    import polars as pl

    for v in VARIANTS:
        home = home_of(out, v, sc)
        marker = out / "calc" / f"{sc}_{v}.json"
        if marker.exists() and (home / "sets" / sc / "model" / "meta.json").exists():
            print(f"calc {sc} {v}: done earlier, skipped", flush=True)
            continue
        if not home.exists():
            raise SystemExit(f"{sc} {v}: run the prepare stage first")
        shutil.rmtree(home / "sets" / sc / "model", ignore_errors=True)
        gp = games_path(home, sc)
        man = home / "sets" / sc / "corpus" / "manifest.json"
        rows_before, man_before = pl.scan_parquet(gp).select(pl.len()).collect().item(), sha(man)
        log = out / "logs" / f"calc_{sc}_{v}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, SEALED_LAB_HOME=str(home), PYTHONIOENCODING="utf-8")
        t0 = time.time()
        print(f"calc {sc} {v}: start (log {log.name})", flush=True)
        with open(log, "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--calc-child", sc], env=env, stdout=lf,
                                 stderr=subprocess.STDOUT, cwd=str(ROOT))
        secs = time.time() - t0
        if rc != 0:
            raise SystemExit(f"calc {sc} {v}: failed (exit {rc}); see {log}")
        text = log.read_text(encoding="utf-8", errors="replace")
        rows_after = pl.scan_parquet(gp).select(pl.len()).collect().item()
        up_to_date = "Corpus is up to date" in text
        proof = dict(rows_before=rows_before, rows_after=rows_after, manifest_unchanged=sha(man) == man_before,
                     log_says_corpus_up_to_date=up_to_date)
        if v == "plat" and not (rows_before == rows_after and proof["manifest_unchanged"] and up_to_date):
            raise SystemExit(f"calc {sc} plat: the corpus was re-ingested ({proof}); see {log}")
        meta = read_json(home / "sets" / sc / "model" / "meta.json")
        rec = dict(set=sc, variant=v, wall_seconds=secs, meta_seconds=meta.get("seconds"), n_games=meta.get("n_games"),
                   quick=meta.get("quick"), deployed=meta.get("deployed", {}).get("components"),
                   deployed_formats={k: x.get("components") for k, x in meta.get("deployed", {}).get("formats", {}).items()},
                   pickprior=meta.get("pickprior"), draft_stage=meta.get("stages", {}).get("draft"),
                   degraded=meta.get("degraded"), corpus_proof=proof)
        write_json(marker, rec)
        print(f"calc {sc} {v}: {secs / 60:.1f} min, deployed {rec['deployed']}, corpus rows {rows_before}->{rows_after}, "
              f"up-to-date line: {up_to_date}", flush=True)


# ------------------------------------------------------------------------------------------------ build workers
def _build_task(task: dict) -> dict:
    os.environ["SEALED_LAB_HOME"] = task["home"]
    from sealed_lab.deck import builder

    model = builder.load_model(task["set"])
    fmt = task["fmt"]
    t0 = time.time()
    res = builder.build(model, task["pool"], fmt, n_decks=1, families=True, n_jobs=1)
    secs = time.time() - t0
    top = res.decks[0]
    deck = {c.card_id: c.count for c in top.cards if c.card_id is not None}
    nl, n = model.deck_lands(deck)
    return dict(set=task["set"], fmt=fmt, draft_id=task["draft_id"], variant=task["variant"], seconds=secs,
                deck={str(k): int(v) for k, v in sorted(deck.items())}, colors=top.colors, splash=top.splash,
                n_lands=int(nl), n_cards=int(n), logit=float(model.logit(deck, fmt)))


def _score_task(task: dict) -> dict:
    """One model (task["home"]) scores decks built by the other model."""
    os.environ["SEALED_LAB_HOME"] = task["home"]
    from sealed_lab.deck import builder

    model = builder.load_model(task["set"])
    rows = []
    for key, fmt, deck in task["items"]:
        d = {int(k): int(v) for k, v in deck.items()}
        rows.append((key, float(model.logit(d, fmt))))
    return dict(set=task["set"], variant=task["variant"], scores=rows)


def _beta_task(task: dict) -> dict:
    import polars as pl

    d = Path(task["home"]) / "sets" / task["set"] / "model"
    cards = pl.read_parquet(d / "cards.parquet").sort("card_id")
    z = np.load(d / "draws.npz", allow_pickle=False)
    out = dict(set=task["set"], variant=task["variant"], card_id=cards["card_id"].to_list(), name=cards["name"].to_list(),
               is_basic=cards["is_basic"].to_list(), is_land=cards["is_land"].to_list())
    for f in ("bo1", "bo3"):
        k = f"{f}__beta"
        out[f"beta_{f}"] = z[k][0].astype(float).tolist() if k in z.files else None
        out[f"se_{f}"] = cards[f"se_{f}"].to_list() if f"se_{f}" in cards.columns else None
    return out


def load_pools(out: Path, sets: list[str], pools_from: Path) -> list[dict]:
    """Held-out pools of the scoreboard file, read from each home's corpus pools.parquet (identical in both homes)."""
    import polars as pl

    entries = read_json(pools_from)["pools"]
    tasks = []
    for sc in sets:
        es = [e for e in entries if e["set"] == sc]
        want: dict[str, list[dict]] = {}
        for e in es:
            want.setdefault(e["draft_id"], []).append(e)
        per_variant = {}
        for v in VARIANTS:
            pools = pl.read_parquet(home_of(out, v, sc) / "sets" / sc / "corpus" / "pools.parquet")
            sub = pools.filter(pl.col("draft_id").is_in(list(want)))
            m = {}
            for did, p in sub.partition_by("draft_id", as_dict=True, include_key=True).items():
                did = did[0] if isinstance(did, tuple) else did
                m[did] = {int(c): int(n) for c, n in zip(p["card_id"].to_list(), p["n"].to_list())}
            per_variant[v] = m
        if per_variant["base"] != per_variant["plat"]:
            raise SystemExit(f"{sc}: pools differ between the homes")
        n_missing = len(es) - sum(len(want[d]) for d in per_variant["base"])
        if n_missing:
            print(f"WARNING {sc}: {n_missing} pools of the scoreboard file not found in the corpus", flush=True)
        for did, pool in per_variant["base"].items():
            for e in want[did]:
                tasks.append(dict(set=sc, fmt=e["fmt"], draft_id=did, pool=pool))
    tasks.sort(key=lambda t: (t["set"], t["fmt"], str(t["draft_id"])))
    return tasks


def pkey(t: dict) -> str:
    return f"{t['set']}|{t['fmt']}|{t['draft_id']}"


def build_stage(out: Path, sets: list[str], jobs: int, pools_from: Path) -> dict:
    pools = load_pools(out, sets, pools_from)
    ctx = multiprocessing.get_context("spawn")
    timing = {}
    for sc in sets:
        for v in VARIANTS:
            if not (home_of(out, v, sc) / "sets" / sc / "model" / "meta.json").exists():
                raise SystemExit(f"{sc} {v}: no model, run the calc stage first")
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        for sc in sets:
            ps = [p for p in pools if p["set"] == sc]
            bpath = out / "build" / f"{sc}_decks.json"
            if bpath.exists():
                print(f"build {sc}: decks done earlier, skipped", flush=True)
            else:
                t0 = time.time()
                ts = [dict(p, variant=v, home=str(home_of(out, v, sc))) for p in ps for v in VARIANTS]
                rows = []
                for k, r in enumerate(ex.map(_build_task, ts, chunksize=1)):
                    rows.append(r)
                    if (k + 1) % 20 == 0:
                        print(f"  build {sc}: {k + 1}/{len(ts)}, {time.time() - t0:.0f}s", flush=True)
                write_json(bpath, dict(set=sc, wall_seconds=time.time() - t0, jobs=jobs, rows=rows))
                print(f"build {sc}: {len(ts)} builds in {(time.time() - t0) / 60:.1f} min", flush=True)
            res = read_json(bpath)
            timing[sc] = dict(build_wall=res["wall_seconds"])
            spath = out / "build" / f"{sc}_cross.json"
            if spath.exists():
                continue
            t0 = time.time()
            by = {(r["variant"], pkey(r)): r for r in res["rows"]}
            ts = []
            for v in VARIANTS:                                       # model v scores the other variant's decks
                other = "plat" if v == "base" else "base"
                items = [(pkey(p), p["fmt"], by[(other, pkey(p))]["deck"]) for p in ps]
                ts.append(dict(set=sc, variant=v, home=str(home_of(out, v, sc)), items=items))
            cross = {r["variant"]: dict(r["scores"]) for r in ex.map(_score_task, ts)}
            write_json(spath, cross)
            print(f"cross-score {sc}: {time.time() - t0:.0f}s", flush=True)
    betas = {}
    with cf.ProcessPoolExecutor(max_workers=min(jobs, 4), mp_context=ctx) as ex:
        ts = [dict(set=sc, variant=v, home=str(home_of(out, v, sc))) for sc in sets for v in VARIANTS]
        for r in ex.map(_beta_task, ts):
            betas[(r["set"], r["variant"])] = r
    return assemble(out, sets, pools, betas, timing)


# ------------------------------------------------------------------------------------------------ compare
def _rank(a: np.ndarray) -> np.ndarray:
    r = np.empty(len(a))
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    return r


def card_compare(b: dict, p: dict) -> dict:
    res = {}
    nb = np.array([not x for x in b["is_basic"]])
    for f in ("bo1", "bo3"):
        if b[f"beta_{f}"] is None or p[f"beta_{f}"] is None:
            continue
        x, y = np.array(b[f"beta_{f}"]), np.array(p[f"beta_{f}"])
        m = nb & np.isfinite(x) & np.isfinite(y) & ((x != 0) | (y != 0))
        d = y - x
        order = [i for i in np.argsort(-np.abs(np.where(m, d, 0.0)))[:10] if m[i]]
        movers = [dict(name=b["name"][i], card_id=b["card_id"][i], beta_base=float(x[i]), beta_plat=float(y[i]), diff=float(d[i]),
                       diff_pp=float(d[i] * PP), se_base=(b[f"se_{f}"][i] if b[f"se_{f}"] else None)) for i in order]
        res[f] = dict(n_cards=int(m.sum()), pearson=float(np.corrcoef(x[m], y[m])[0, 1]),
                      spearman=float(np.corrcoef(_rank(x[m]), _rank(y[m]))[0, 1]),
                      mean_beta_base=float(x[m].mean()), mean_beta_plat=float(y[m].mean()),
                      sd_beta_base=float(x[m].std()), sd_beta_plat=float(y[m].std()),
                      rms_diff_pp=float(np.sqrt(np.mean(d[m] ** 2)) * PP), movers=movers)
    return res


def _summ(rows: list[dict]) -> dict:
    if not rows:
        return {}
    a = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    cc = a("cards_changed")
    return dict(pools=len(rows), share_identical=float(np.mean([r["identical"] for r in rows])),
                share_same_colors=float(np.mean([r["same_main_colors"] for r in rows])),
                mean_cards_changed=float(cc.mean()), median_cards_changed=float(np.median(cc)),
                mean_land_diff=float(a("land_diff").mean()), mean_abs_land_diff=float(np.abs(a("land_diff")).mean()),
                mean_base_model_gap_pp=float(a("base_model_gap_pp").mean()), mean_plat_model_gap_pp=float(a("plat_model_gap_pp").mean()),
                median_base_model_gap_pp=float(np.median(a("base_model_gap_pp"))),
                median_plat_model_gap_pp=float(np.median(a("plat_model_gap_pp"))))


def assemble(out: Path, sets: list[str], pools: list[dict], betas: dict, timing: dict) -> dict:
    rows = []
    for sc in sets:
        res = read_json(out / "build" / f"{sc}_decks.json")["rows"]
        cross = read_json(out / "build" / f"{sc}_cross.json")
        by = {(r["variant"], pkey(r)): r for r in res}
        nonbasic = {i: not bs for i, bs in zip(betas[(sc, "base")]["card_id"], betas[(sc, "base")]["is_basic"])}
        for p in (q for q in pools if q["set"] == sc):
            k = pkey(p)
            b, q = by[("base", k)], by[("plat", k)]
            bd = {int(c): n for c, n in b["deck"].items()}
            qd = {int(c): n for c, n in q["deck"].items()}
            ids = set(bd) | set(qd)
            added = sum(max(qd.get(c, 0) - bd.get(c, 0), 0) for c in ids if nonbasic.get(c, True))
            cut = sum(max(bd.get(c, 0) - qd.get(c, 0), 0) for c in ids if nonbasic.get(c, True))
            lb, lq = b["logit"], q["logit"]
            lb_of_q, lq_of_b = cross["base"][k], cross["plat"][k]
            rows.append(dict(
                set=sc, fmt=p["fmt"], draft_id=p["draft_id"], identical=bd == qd,
                same_main_colors=sorted(b["colors"]) == sorted(q["colors"]),
                colors_base=b["colors"], splash_base=b["splash"], colors_plat=q["colors"], splash_plat=q["splash"],
                same_splash=sorted(b["splash"]) == sorted(q["splash"]),
                nonbasic_added=int(added), nonbasic_cut=int(cut), cards_changed=int(max(added, cut)),
                lands_base=b["n_lands"], lands_plat=q["n_lands"], land_diff=q["n_lands"] - b["n_lands"],
                logit_base_deck_base_model=lb, logit_plat_deck_plat_model=lq,
                logit_plat_deck_base_model=lb_of_q, logit_base_deck_plat_model=lq_of_b,
                base_model_gap_pp=(lb_of_q - lb) * PP, plat_model_gap_pp=(lq_of_b - lq) * PP,
                seconds_base=b["seconds"], seconds_plat=q["seconds"],
                deck_base=b["deck"], deck_plat=q["deck"]))
    summary = dict(overall=_summ(rows), by_set_format={}, by_set={}, by_format={})
    for sc in sets:
        summary["by_set"][sc] = _summ([r for r in rows if r["set"] == sc])
        for f in ("bo1", "bo3"):
            summary["by_set_format"][f"{sc} {f}"] = _summ([r for r in rows if r["set"] == sc and r["fmt"] == f])
    for f in ("bo1", "bo3"):
        summary["by_format"][f] = _summ([r for r in rows if r["fmt"] == f])
    cards = {sc: card_compare(betas[(sc, "base")], betas[(sc, "plat")]) for sc in sets}
    prep = {sc: read_json(out / "prepare" / f"{sc}.json") for sc in sets if (out / "prepare" / f"{sc}.json").exists()}
    calc = {sc: {v: read_json(out / "calc" / f"{sc}_{v}.json") for v in VARIANTS if (out / "calc" / f"{sc}_{v}.json").exists()} for sc in sets}
    doc = dict(question="Does a Platinum+-only draft stage change the decks the builder builds?", pp_per_logit=PP, sets=sets,
               summary=summary, card_values=cards, prepare=prep, calc=calc, timing=timing, pools=rows)
    write_json(out / "plat_builder.json", doc)
    (out / "plat_builder.md").write_text(render_md(doc), encoding="utf-8")
    for key, s in summary["by_set_format"].items():
        if s:
            print(f"{key}: n={s['pools']} identical {s['share_identical']:.0%} same colors {s['share_same_colors']:.0%} "
                  f"cards changed mean {s['mean_cards_changed']:.1f} (median {s['median_cards_changed']:.0f}) "
                  f"land diff {s['mean_land_diff']:+.2f} gap base {s['mean_base_model_gap_pp']:+.2f}pp plat {s['mean_plat_model_gap_pp']:+.2f}pp",
                  flush=True)
    return doc


def render_md(doc: dict) -> str:
    L = ["# Platinum+ draft stage: do the built decks differ?", "",
         "Descriptive only; nothing ships. Base = the normal model; plat = the same chain, but its draft stage sees only Draft games "
         "of platinum / diamond / mythic players (Sealed and TradDraft games unchanged). Both calculated with quick=True, pickprior off, "
         "no Scryfall, 16 bootstrap draws. Held-out pools of docs/builder_scoreboard/search_check.json, top family deck of "
         "`builder.build(n_decks=1, families=True)` in each model.", "",
         "Gaps in win-rate points at a 55% deck (logit difference x 24.75). Base-model gap = base logit(plat deck) - base logit(base deck); "
         "plat-model gap = plat logit(base deck) - plat logit(plat deck). Each model prefers its own deck, so both are expected to be "
         "negative (a positive value means the other model's deck scores higher under this model, i.e. the search missed it).", "",
         "## Draft games kept", "", "| set | draft games before | after | kept |", "|---|---|---|---|"]
    for sc, p in doc["prepare"].items():
        L.append(f"| {sc} | {p['draft_games_before']:,} | {p['draft_games_after']:,} | {p['draft_share_kept']:.1%} |")
    hdr = ("| cell | pools | identical | same main colors | cards changed (mean / median) | land diff (mean / mean abs) "
           "| base-model gap pp | plat-model gap pp |")
    sep = "|---|---|---|---|---|---|---|---|"

    def line(name, s):
        return (f"| {name} | {s['pools']} | {s['share_identical']:.0%} | {s['share_same_colors']:.0%} | "
                f"{s['mean_cards_changed']:.1f} / {s['median_cards_changed']:.0f} | {s['mean_land_diff']:+.2f} / {s['mean_abs_land_diff']:.2f} | "
                f"{s['mean_base_model_gap_pp']:+.2f} | {s['mean_plat_model_gap_pp']:+.2f} |")
    S = doc["summary"]
    L += ["", "## Decks, overall", "", hdr, sep, line("all pools", S["overall"])]
    for f, s in S["by_format"].items():
        if s:
            L.append(line(f, s))
    L += ["", "## Decks, per set and format", "", hdr, sep]
    for k, s in S["by_set_format"].items():
        if s:
            L.append(line(k, s))
    L += ["", "## Card values, final models (draw 0, nonbasic cards)", "",
          "| set | fmt | cards | Pearson | Spearman | RMS diff pp | SD base | SD plat |", "|---|---|---|---|---|---|---|---|"]
    for sc, c in doc["card_values"].items():
        for f, x in c.items():
            L.append(f"| {sc} | {f} | {x['n_cards']} | {x['pearson']:.3f} | {x['spearman']:.3f} | {x['rms_diff_pp']:.2f} | "
                     f"{x['sd_beta_base']:.3f} | {x['sd_beta_plat']:.3f} |")
    for sc, c in doc["card_values"].items():
        for f, x in c.items():
            L += ["", f"### {sc} {f}: ten biggest movers (plat - base, logit; pp at a 55% deck)", "",
                  "| card | base | plat | diff | diff pp |", "|---|---|---|---|---|"]
            for m in x["movers"]:
                L.append(f"| {m['name']} | {m['beta_base']:+.3f} | {m['beta_plat']:+.3f} | {m['diff']:+.3f} | {m['diff_pp']:+.2f} |")
    L += ["", "## Calculation", "", "| set | model | minutes | deployed components |", "|---|---|---|---|"]
    for sc, c in doc["calc"].items():
        for v, r in c.items():
            L.append(f"| {sc} | {v} | {r['wall_seconds'] / 60:.1f} | {', '.join(r.get('deployed') or []) or '-'} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--stage", default="all", choices=["prepare", "calc", "build", "all"])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="scratch app home to copy from (read only)")
    ap.add_argument("--pools-from", default=str(DEFAULT_POOLS))
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    check_out_path(out)
    out.mkdir(parents=True, exist_ok=True)
    sets = [s.upper() for s in (args.sets or ALL_SETS)]
    stages = ("prepare", "calc", "build") if args.stage == "all" else (args.stage,)
    times = {}
    for st in stages:
        t0 = time.time()
        if st == "prepare":
            for sc in sets:
                prepare_set(sc, out, Path(args.source))
        elif st == "calc":
            for sc in sets:
                calc_set(sc, out, None)
        else:
            build_stage(out, sets, args.jobs, Path(args.pools_from))
        times[st] = time.time() - t0
        print(f"stage {st}: {times[st] / 60:.1f} min", flush=True)
    prev = read_json(out / "stage_times.json") if (out / "stage_times.json").exists() else {}
    prev.update({k: round(v, 1) for k, v in times.items()})
    write_json(out / "stage_times.json", prev)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--calc-child":
        sys.exit(calc_child(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--fix-signature-child":
        sys.exit(fix_signature_child(sys.argv[2]))
    sys.exit(main())
