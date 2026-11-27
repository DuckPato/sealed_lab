"""Does a rank-corrected Draft stage change the decks the builder builds? (descriptive; nothing ships)

    python scripts/rank_builder_check.py [--selfcheck] [--stage prepare|calc|build|compare|all] [--jobs N] [--sets FRA ...]

"rank" model = the normal model (same quick calculation as plat_builder_check: quick=True, pickprior off, no Scryfall, 16 bootstrap
draws) except that, in the FINAL fit, the Draft stage carries the 6 rank indicators of plat_draft_check (arm R: silver, gold, platinum,
diamond, mythic, unranked; bronze = reference; prior SD 1.0) in its covariate block. Bo1 / Bo3 stages get no rank columns, so the card
betas the deck model uses keep their shapes.

Wiring (inside the calc child process only; nothing under sealed_lab/ is edited):
  * pipeline.GAME_COLS gets "rank" so that load_inputs keeps the column;
  * design.build_stage is wrapped: when called from pipeline's `final_fit` for the "draft" stage, the stage goes through
    plat_draft_check.make_rank_stage (indicators appended to C and to enc.sigma / enc.names). Every other caller (the hold-out
    validations that decide the components) gets the plain stage, so the deployed components stay those of the base model;
  * cardvalue.run_chain is wrapped (only to record the draft stage's rank coefficients and SEs when called from `final_fit`).
  `final_fit` builds the stages once, then runs run_chain (final fit) and cardvalue.bootstrap(stages_, fits_, ...) on those same stage
  objects, so all 16 draws see the rank-corrected draft stage (the bootstrap resamples the bo1 / bo3 stages and perturbs their prior,
  the draft-stage estimate, by its own SE; the draft stage itself is fitted once).

Stages: prepare (copy base homes), calc, build (rank decks, determinism check of the base decks), compare (writes the json / md).
Never writes outside SCR/rank_run; refuses to run when SEALED_LAB_HOME is the real app home. The base homes are only read.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import plat_builder_check as pbc  # noqa: E402

SCR = Path(os.environ.get("SEALED_LAB_SCRATCH", tempfile.gettempdir()))   # holds plat_run/ (plat_builder_check) and rank_run/
OUT = SCR / "rank_run"
BASE_ROOT = SCR / "plat_run" / "base"
PLAT_JSON = SCR / "plat_run" / "plat_builder.json"
POOLS_FROM = ROOT / "docs" / "builder_scoreboard" / "search_check.json"
SETS = ["FRA", "HOB", "MSH", "SOS", "TMT"]
PP = pbc.PP
RANK_COLS = ["silver", "gold", "platinum", "diamond", "mythic", "unranked"]
DETERMINISM_POOLS = [("TMT", "bo1"), ("FRA", "bo3"), ("MSH", "bo1")]


# ------------------------------------------------------------------------------------------------ safety
def real_home() -> Path | None:
    la = os.environ.get("LOCALAPPDATA")
    return Path(la) / "sealed_lab" if la else None


def refuse_real_home() -> None:
    h = os.environ.get("SEALED_LAB_HOME")
    rh = real_home()
    if not h:
        raise SystemExit("SEALED_LAB_HOME is not set: refusing to run")
    if rh is not None:
        try:
            Path(h).resolve().relative_to(rh.resolve())
        except ValueError:
            return
        raise SystemExit(f"SEALED_LAB_HOME resolves to the real app home {rh}: refusing to run")


def snapshot(root: Path | None) -> dict:
    snap = {}
    if root is None or not root.exists():
        return snap
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = os.path.join(dp, n)
            try:
                snap[p] = os.stat(p).st_mtime_ns
            except OSError:
                snap[p] = -1
    return snap


def raw_root() -> Path:
    return Path(os.environ.get("SEALED_LAB_RAW_DIR", str(Path.home() / "17LandsData")))   # raw 17Lands files, read only


def snap_before() -> None:
    f = OUT / "apphome_before.json"
    if not f.exists():
        pbc.write_json(f, dict(app=snapshot(real_home()), raw=snapshot(raw_root()), time=time.time()))


def snap_check() -> dict:
    f = OUT / "apphome_before.json"
    if not f.exists():
        return dict(error="no snapshot before")
    b = pbc.read_json(f)
    res = {}
    for key, root in (("app", real_home()), ("raw", raw_root())):
        now = snapshot(root)
        old = b[key]
        changed = [p for p in set(old) | set(now) if old.get(p) != now.get(p)]
        res[key] = dict(root=str(root), files_before=len(old), files_after=len(now), changed=len(changed), examples=changed[:5])
    return res


def rank_home(sc: str) -> Path:
    return OUT / "rank" / f"{sc}-home"


def base_home(sc: str) -> Path:
    return BASE_ROOT / f"{sc}-home"


# ------------------------------------------------------------------------------------------------ prepare
def prepare_set(sc: str, dest: Path, tag: str = "rank") -> None:
    if (dest / "sets" / sc / "corpus" / "manifest.json").exists() and (OUT / "prepare" / f"{tag}_{sc}.json").exists():
        print(f"prepare {tag} {sc}: done earlier, skipped", flush=True)
        return
    src = base_home(sc)
    if not src.is_dir():
        raise SystemExit(f"no base home {src}")
    if dest.exists():
        shutil.rmtree(dest)
    part = dest.with_name(dest.name + ".partial")
    if part.exists():
        shutil.rmtree(part)
    t0 = time.time()
    shutil.copytree(src, part, symlinks=False)
    bad = pbc.find_links(part)
    if bad:
        raise SystemExit(f"{sc}: links found in the copy: {bad[:5]}")
    shutil.rmtree(part / "sets" / sc / "model", ignore_errors=True)       # a fresh model from the calc
    os.replace(part, dest)
    sig = pbc.fix_signature(sc, dest)                                    # only the stored signature (Scryfall stat) may change
    pbc.write_json(OUT / "prepare" / f"{tag}_{sc}.json", dict(set=sc, source=str(src), dest=str(dest), signature_fix=sig,
                                                              seconds=time.time() - t0))
    print(f"prepare {tag} {sc}: copied in {time.time() - t0:.0f}s, signature {'changed' if sig['old'] != sig['new'] else 'unchanged'}",
          flush=True)


# ------------------------------------------------------------------------------------------------ calc child (patched)
def _called_from_final_fit() -> bool:
    f = sys._getframe(2)
    for _ in range(3):
        if f is None:
            return False
        if f.f_code.co_name == "final_fit":
            return True
        f = f.f_back
    return False


def calc_child(sc: str, zero: bool, record: Path) -> int:
    refuse_real_home()
    import plat_draft_check as pdc
    from sealed_lab import api
    from sealed_lab.model import cardvalue, design, pipeline

    pipeline.GAME_COLS = list(pipeline.GAME_COLS) + ["rank"]
    orig_bs, orig_rc = design.build_stage, cardvalue.run_chain
    rec: dict = dict(set=sc, zero=zero, stage_calls=[], chain_calls=[])

    def build_stage(name, g, space, enc=None):
        st = orig_bs(name, g, space, enc)
        if name == "draft" and _called_from_final_fit():
            st, k0 = pdc.make_rank_stage(st, zero=zero)
            rec["stage_calls"].append(dict(k0=int(k0), k=int(st.prob.k), n_games=int(st.n_games), names=list(st.enc.names[-6:])))
            print(f"[patch] draft stage in final_fit: k {k0} -> {st.prob.k}, games {st.n_games}, zero={zero}", flush=True)
        return st

    def run_chain(stages, *a, **kw):
        fits = orig_rc(stages, *a, **kw)
        if "draft" in stages and _called_from_final_fit() and any(n.startswith("rank:") for n in stages["draft"].enc.names):
            st, fd = stages["draft"], fits["draft"]
            idx = [i for i, n in enumerate(st.enc.names) if n.startswith("rank:")]
            rec["chain_calls"].append(dict(
                k=int(fd.k), sigma=float(fd.sigma), converged=bool(fd.converged),
                rank_coef={st.enc.names[i][5:]: dict(logit=float(fd.coef[i]), se_logit=float(fd.se[i]), pp=float(fd.coef[i] * PP),
                                                     se_pp=float(fd.se[i] * PP)) for i in idx},
                other_stages={n: dict(k=int(f.k)) for n, f in fits.items()}))
            pbc.write_json(record, rec)
        return fits

    design.build_stage, cardvalue.run_chain = build_stage, run_chain
    last = [None]

    def prog(frac: float, msg: str) -> None:
        if msg != last[0]:
            last[0] = msg
            print(f"[{time.strftime('%H:%M:%S')}] {frac:5.1%} {msg}", flush=True)

    api.calculate_set(sc, progress=prog, n_boot=16, fetch_scryfall=False, quick=True, pickprior="off")
    pbc.write_json(record, rec)
    if not rec["stage_calls"] or not rec["chain_calls"]:
        print("[patch] ERROR: the patch never fired in final_fit", flush=True)
        return 3
    return 0


def calc_set(sc: str, home: Path, tag: str, zero: bool = False) -> dict:
    import polars as pl

    marker = OUT / "calc" / f"{tag}_{sc}.json"
    if marker.exists() and (home / "sets" / sc / "model" / "meta.json").exists():
        print(f"calc {tag} {sc}: done earlier, skipped", flush=True)
        return pbc.read_json(marker)
    shutil.rmtree(home / "sets" / sc / "model", ignore_errors=True)
    gp = pbc.games_path(home, sc)
    man = home / "sets" / sc / "corpus" / "manifest.json"
    rows_before, man_before = pl.scan_parquet(gp).select(pl.len()).collect().item(), pbc.sha(man)
    log = OUT / "logs" / f"calc_{tag}_{sc}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    record = OUT / "calc" / f"{tag}_{sc}_rankcoef.json"
    env = dict(os.environ, SEALED_LAB_HOME=str(home), PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
    cmd = [sys.executable, str(Path(__file__).resolve()), "--calc-child", sc, str(record)] + (["--zero"] if zero else [])
    t0 = time.time()
    print(f"calc {tag} {sc}: start (log {log.name})", flush=True)
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
    secs = time.time() - t0
    if rc != 0:
        raise SystemExit(f"calc {tag} {sc}: failed (exit {rc}); see {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    rows_after = pl.scan_parquet(gp).select(pl.len()).collect().item()
    proof = dict(rows_before=rows_before, rows_after=rows_after, manifest_unchanged=pbc.sha(man) == man_before,
                 log_says_corpus_up_to_date="Corpus is up to date" in text)
    if not (rows_before == rows_after and proof["manifest_unchanged"] and proof["log_says_corpus_up_to_date"]):
        raise SystemExit(f"calc {tag} {sc}: the corpus was re-ingested ({proof}); see {log}")
    mdir = home / "sets" / sc / "model"
    meta = pbc.read_json(mdir / "meta.json")
    z, zb = np.load(mdir / "draws.npz"), np.load(base_home(sc) / "sets" / sc / "model" / "draws.npz")
    shapes = {k: list(z[k].shape) for k in z.files}
    shapes_base = {k: list(zb[k].shape) for k in zb.files}
    rec = pbc.read_json(record) if record.exists() else {}
    out = dict(set=sc, tag=tag, zero=zero, wall_seconds=secs, meta_seconds=meta.get("seconds"), n_games=meta.get("n_games"),
               quick=meta.get("quick"), deployed=meta.get("deployed", {}).get("components"),
               deployed_formats={k: x.get("components") for k, x in meta.get("deployed", {}).get("formats", {}).items()},
               pickprior=meta.get("pickprior"), degraded=meta.get("degraded"), corpus_proof=proof,
               draws_shapes=shapes, draws_shapes_equal_base=shapes == shapes_base,
               quick_log_lines=[ln for ln in text.splitlines() if "quick" in ln.lower()][:3],
               patch=dict(stage_calls=rec.get("stage_calls"), chain_calls=rec.get("chain_calls")))
    pbc.write_json(marker, out)
    print(f"calc {tag} {sc}: {secs / 60:.1f} min, deployed {out['deployed']}, rows {rows_before}->{rows_after}, shapes equal base: "
          f"{out['draws_shapes_equal_base']}", flush=True)
    return out


def draws_diff(home_a: Path, home_b: Path, sc: str) -> dict:
    a = np.load(home_a / "sets" / sc / "model" / "draws.npz")
    b = np.load(home_b / "sets" / sc / "model" / "draws.npz")
    res = {}
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            res[k] = "missing in one"
        elif a[k].shape != b[k].shape:
            res[k] = f"shape {a[k].shape} vs {b[k].shape}"
        else:
            x, y = a[k].astype(float), b[k].astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            res[k] = dict(max_abs_all=float(np.max(np.abs(x - y)[m])) if m.any() else 0.0,
                          max_abs_draw0=float(np.max(np.abs(x[0] - y[0])[m[0]])) if m[0].any() else 0.0)
    return res


def selfcheck() -> dict:
    sc = "TMT"
    dest = OUT / "selfcheck" / f"{sc}-home"
    prepare_set(sc, dest, tag="selfcheck")
    r = calc_set(sc, dest, "selfcheck", zero=True)
    d = draws_diff(dest, base_home(sc), sc)
    betas = {k: v for k, v in d.items() if k.endswith("__beta") and isinstance(v, dict)}
    mx = max((v["max_abs_all"] for v in betas.values()), default=float("nan"))
    mx0 = max((v["max_abs_draw0"] for v in betas.values()), default=float("nan"))
    res = dict(calc=r, diffs=d, max_abs_beta_all_draws=mx, max_abs_beta_draw0=mx0,
               ok=bool(mx < 1e-3 and r["deployed"] == pbc.read_json(SCR / "plat_run" / "calc" / "TMT_base.json")["deployed"]))
    pbc.write_json(OUT / "selfcheck.json", res)
    print(f"SELFCHECK TMT zero indicators vs base: max |dbeta| draw0 {mx0:.2e}, all draws {mx:.2e}; deployed {r['deployed']}; ok={res['ok']}",
          flush=True)
    return res


# ------------------------------------------------------------------------------------------------ build
def load_pools(sets: list[str]) -> list[dict]:
    import polars as pl

    entries = pbc.read_json(POOLS_FROM)["pools"]
    tasks = []
    for sc in sets:
        es = [e for e in entries if e["set"] == sc]
        want: dict = {}
        for e in es:
            want.setdefault(e["draft_id"], []).append(e)
        per = {}
        for tag, home in (("base", base_home(sc)), ("rank", rank_home(sc))):
            pools = pl.read_parquet(home / "sets" / sc / "corpus" / "pools.parquet")
            sub = pools.filter(pl.col("draft_id").is_in(list(want)))
            m = {}
            for did, p in sub.partition_by("draft_id", as_dict=True, include_key=True).items():
                did = did[0] if isinstance(did, tuple) else did
                m[did] = {int(c): int(n) for c, n in zip(p["card_id"].to_list(), p["n"].to_list())}
            per[tag] = m
        if per["base"] != per["rank"]:
            raise SystemExit(f"{sc}: pools differ between the homes")
        for did, pool in per["base"].items():
            for e in want[did]:
                tasks.append(dict(set=sc, fmt=e["fmt"], draft_id=did, pool=pool))
    tasks.sort(key=lambda t: (t["set"], t["fmt"], str(t["draft_id"])))
    return tasks


def plat_rows() -> dict:
    d = pbc.read_json(PLAT_JSON)
    return {f"{r['set']}|{r['fmt']}|{r['draft_id']}": r for r in d["pools"]}


def build_stage(sets: list[str], jobs: int) -> None:
    pools = load_pools(sets)
    prow = plat_rows()
    for p in pools:
        if pbc.pkey(p) not in prow:
            raise SystemExit(f"pool {pbc.pkey(p)} missing from plat_builder.json")
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        # determinism check: rebuild 3 base decks with the base home, compare with plat_builder.json
        dpath = OUT / "build" / "determinism.json"
        if not dpath.exists():
            picks = []
            for sc, f in DETERMINISM_POOLS:
                if sc in sets:
                    picks.append(next(p for p in pools if p["set"] == sc and p["fmt"] == f))
            t0 = time.time()
            res = list(ex.map(pbc._build_task, [dict(p, variant="base", home=str(base_home(p["set"]))) for p in picks], chunksize=1))
            rows = []
            for p, r in zip(picks, res):
                old = {int(k): v for k, v in prow[pbc.pkey(p)]["deck_base"].items()}
                new = {int(k): v for k, v in r["deck"].items()}
                rows.append(dict(pool=pbc.pkey(p), identical=old == new, logit_old=prow[pbc.pkey(p)]["logit_base_deck_base_model"],
                                 logit_new=r["logit"]))
            pbc.write_json(dpath, dict(rows=rows, seconds=time.time() - t0))
            print("determinism (base rebuild vs plat_builder.json):", [(r["pool"], r["identical"]) for r in rows], flush=True)
        for sc in sets:
            ps = [p for p in pools if p["set"] == sc]
            bpath = OUT / "build" / f"{sc}_decks.json"
            if not bpath.exists():
                t0 = time.time()
                ts = [dict(p, variant="rank", home=str(rank_home(sc))) for p in ps]
                rows = []
                for k, r in enumerate(ex.map(pbc._build_task, ts, chunksize=1)):
                    rows.append(r)
                    if (k + 1) % 5 == 0:
                        print(f"  build {sc}: {k + 1}/{len(ts)}, {time.time() - t0:.0f}s", flush=True)
                pbc.write_json(bpath, dict(set=sc, wall_seconds=time.time() - t0, jobs=jobs, rows=rows))
                print(f"build {sc}: {len(ts)} builds in {(time.time() - t0) / 60:.1f} min", flush=True)
            spath = OUT / "build" / f"{sc}_cross.json"
            if spath.exists():
                continue
            res = pbc.read_json(bpath)["rows"]
            by = {pbc.pkey(r): r for r in res}
            t0 = time.time()
            ts = [dict(set=sc, variant="base", home=str(base_home(sc)), items=[(pbc.pkey(p), p["fmt"], by[pbc.pkey(p)]["deck"]) for p in ps]),
                  dict(set=sc, variant="rank", home=str(rank_home(sc)),
                       items=[(pbc.pkey(p), p["fmt"], prow[pbc.pkey(p)]["deck_base"]) for p in ps])]
            cross = {r["variant"]: dict(r["scores"]) for r in ex.map(pbc._score_task, ts)}
            pbc.write_json(spath, cross)
            print(f"cross-score {sc}: {time.time() - t0:.0f}s", flush=True)
        betas = {}
        ts = [dict(set=sc, variant=v, home=str(h(sc))) for sc in sets for v, h in (("base", base_home), ("rank", rank_home))]
        for r in ex.map(pbc._beta_task, ts):
            betas[(r["set"], r["variant"])] = r
    pbc.write_json(OUT / "build" / "betas.json", {f"{k[0]}|{k[1]}": v for k, v in betas.items()})


# ------------------------------------------------------------------------------------------------ compare
def summ(rows: list[dict], kb: str, kv: str) -> dict:
    if not rows:
        return {}
    a = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    cc = a("cards_changed")
    return dict(pools=len(rows), share_identical=float(np.mean([r["identical"] for r in rows])),
                share_same_colors=float(np.mean([r["same_main_colors"] for r in rows])),
                mean_cards_changed=float(cc.mean()), median_cards_changed=float(np.median(cc)),
                mean_land_diff=float(a("land_diff").mean()), mean_abs_land_diff=float(np.abs(a("land_diff")).mean()),
                mean_base_model_gap_pp=float(a(kb).mean()), mean_variant_model_gap_pp=float(a(kv).mean()),
                median_base_model_gap_pp=float(np.median(a(kb))), median_variant_model_gap_pp=float(np.median(a(kv))))


def tables(rows: list[dict], kb: str, kv: str, sets: list[str]) -> dict:
    t = dict(overall=summ(rows, kb, kv), by_format={}, by_set={}, by_set_format={})
    for f in ("bo1", "bo3"):
        t["by_format"][f] = summ([r for r in rows if r["fmt"] == f], kb, kv)
    for sc in sets:
        t["by_set"][sc] = summ([r for r in rows if r["set"] == sc], kb, kv)
        for f in ("bo1", "bo3"):
            t["by_set_format"][f"{sc} {f}"] = summ([r for r in rows if r["set"] == sc and r["fmt"] == f], kb, kv)
    return t


def compare(sets: list[str]) -> dict:
    pools = load_pools(sets)
    prow = plat_rows()
    betas = {tuple(k.split("|")): v for k, v in pbc.read_json(OUT / "build" / "betas.json").items()}
    rank_rows, plat_sel = [], []
    for sc in sets:
        res = pbc.read_json(OUT / "build" / f"{sc}_decks.json")["rows"]
        cross = pbc.read_json(OUT / "build" / f"{sc}_cross.json")
        by = {pbc.pkey(r): r for r in res}
        nonbasic = {i: not bs for i, bs in zip(betas[(sc, "base")]["card_id"], betas[(sc, "base")]["is_basic"])}
        for p in (q for q in pools if q["set"] == sc):
            k = pbc.pkey(p)
            pr, q = prow[k], by[k]
            bd = {int(c): n for c, n in pr["deck_base"].items()}
            qd = {int(c): n for c, n in q["deck"].items()}
            ids = set(bd) | set(qd)
            added = sum(max(qd.get(c, 0) - bd.get(c, 0), 0) for c in ids if nonbasic.get(c, True))
            cut = sum(max(bd.get(c, 0) - qd.get(c, 0), 0) for c in ids if nonbasic.get(c, True))
            lb = pr["logit_base_deck_base_model"]
            lr = q["logit"]
            lb_of_r = cross["base"][k]            # base model scores the rank deck
            lr_of_b = cross["rank"][k]            # rank model scores the base deck
            rank_rows.append(dict(
                set=sc, fmt=p["fmt"], draft_id=p["draft_id"], identical=bd == qd,
                same_main_colors=sorted(pr["colors_base"]) == sorted(q["colors"]), colors_base=pr["colors_base"], colors_rank=q["colors"],
                splash_base=pr["splash_base"], splash_rank=q["splash"],
                nonbasic_added=int(added), nonbasic_cut=int(cut), cards_changed=int(max(added, cut)),
                lands_base=pr["lands_base"], lands_rank=q["n_lands"], land_diff=q["n_lands"] - pr["lands_base"],
                logit_base_deck_base_model=lb, logit_rank_deck_rank_model=lr, logit_rank_deck_base_model=lb_of_r,
                logit_base_deck_rank_model=lr_of_b,
                base_model_gap_pp=(lb_of_r - lb) * PP, rank_model_gap_pp=(lr_of_b - lr) * PP,
                deck_base=pr["deck_base"], deck_rank=q["deck"]))
            plat_sel.append(pr)
    cards = {sc: pbc.card_compare(betas[(sc, "base")], betas[(sc, "rank")]) for sc in sets}
    plat_cards = pbc.read_json(PLAT_JSON)["card_values"]
    doc = dict(
        question="Does a rank-corrected Draft stage change the decks the builder builds?", pp_per_logit=PP, sets=sets,
        rank=tables(rank_rows, "base_model_gap_pp", "rank_model_gap_pp", sets),
        plat=tables(plat_sel, "base_model_gap_pp", "plat_model_gap_pp", sets),
        card_values=cards, plat_card_values={sc: plat_cards.get(sc) for sc in sets},
        selfcheck=pbc.read_json(OUT / "selfcheck.json") if (OUT / "selfcheck.json").exists() else None,
        determinism=pbc.read_json(OUT / "build" / "determinism.json"),
        calc={sc: pbc.read_json(OUT / "calc" / f"rank_{sc}.json") for sc in sets},
        base_calc={sc: pbc.read_json(SCR / "plat_run" / "calc" / f"{sc}_base.json") for sc in sets},
        apphome_check=snap_check(), pools=rank_rows)
    pbc.write_json(OUT / "rank_builder.json", doc)
    (OUT / "rank_builder.md").write_text(render_md(doc), encoding="utf-8")
    return doc


def render_md(doc: dict) -> str:
    L = ["# Rank-corrected draft stage: do the built decks differ?", "",
         "Descriptive only; nothing ships. Base = the normal model; rank = the same chain whose FINAL-fit Draft stage carries the 6 rank "
         "indicators (silver, gold, platinum, diamond, mythic, unranked; bronze reference; prior SD 1.0). Plat = Platinum+-only draft "
         "stage (earlier run, same pools, shown for context). Same quick calculation (pickprior off, no Scryfall, 16 draws). "
         "Held-out pools of docs/builder_scoreboard/search_check.json, top family deck of `builder.build(n_decks=1, families=True)`.", "",
         "Gaps in win-rate points at a 55% deck (logit x 24.75). base-model gap = base logit(variant deck) - base logit(base deck); "
         "variant-model gap = variant logit(base deck) - variant logit(variant deck). Each model prefers its own deck, so both are "
         "expected negative.", ""]
    hdr = ("| cell | variant | pools | identical | same main colors | cards changed (mean / median) | land diff (mean / mean abs) "
           "| base-model gap pp | variant-model gap pp |")
    sep = "|---|---|---|---|---|---|---|---|---|"

    def line(name, var, s):
        return (f"| {name} | {var} | {s['pools']} | {s['share_identical']:.0%} | {s['share_same_colors']:.0%} | "
                f"{s['mean_cards_changed']:.1f} / {s['median_cards_changed']:.0f} | {s['mean_land_diff']:+.2f} / {s['mean_abs_land_diff']:.2f} | "
                f"{s['mean_base_model_gap_pp']:+.2f} | {s['mean_variant_model_gap_pp']:+.2f} |")

    def section(title, getter):
        L.extend(["", f"## {title}", "", hdr, sep])
        for name, r, p in getter:
            if r:
                L.append(line(name, "rank", r))
                L.append(line(name, "plat", p))

    R, P = doc["rank"], doc["plat"]
    section("Decks, overall", [("all pools", R["overall"], P["overall"])])
    section("Decks, per format", [(f, R["by_format"][f], P["by_format"][f]) for f in ("bo1", "bo3")])
    section("Decks, per set", [(s, R["by_set"][s], P["by_set"][s]) for s in doc["sets"]])
    section("Decks, per set and format", [(k, R["by_set_format"][k], P["by_set_format"][k]) for k in R["by_set_format"]])
    L += ["", "## Card values, final models (draw 0, nonbasic cards)", "",
          "| set | fmt | cards | rank Pearson | rank Spearman | rank RMS diff pp | SD base | SD rank | plat Pearson | plat RMS diff pp |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sc, c in doc["card_values"].items():
        for f, x in c.items():
            pc = ((doc["plat_card_values"].get(sc) or {}).get(f)) or {}
            L.append(f"| {sc} | {f} | {x['n_cards']} | {x['pearson']:.3f} | {x['spearman']:.3f} | {x['rms_diff_pp']:.2f} | "
                     f"{x['sd_beta_base']:.3f} | {x['sd_beta_plat']:.3f} | {pc.get('pearson', float('nan')):.3f} | "
                     f"{pc.get('rms_diff_pp', float('nan')):.2f} |")
    for sc, c in doc["card_values"].items():
        for f, x in c.items():
            L += ["", f"### {sc} {f}: ten biggest movers (rank - base, logit; pp at a 55% deck)", "",
                  "| card | base | rank | diff | diff pp |", "|---|---|---|---|---|"]
            for m in x["movers"]:
                L.append(f"| {m['name']} | {m['beta_base']:+.3f} | {m['beta_plat']:+.3f} | {m['diff']:+.3f} | {m['diff_pp']:+.2f} |")
    L += ["", "## Draft-stage rank coefficients (final fit, pp at a 55% deck; reference bronze)", "",
          "| set | " + " | ".join(RANK_COLS) + " | draft sigma |", "|---|" + "---|" * (len(RANK_COLS) + 1)]
    for sc, c in doc["calc"].items():
        cc = (c["patch"]["chain_calls"] or [{}])[-1]
        rc = cc.get("rank_coef", {})
        L.append(f"| {sc} | " + " | ".join(f"{rc[n]['pp']:+.1f} ({rc[n]['se_pp']:.1f})" if n in rc else "-" for n in RANK_COLS)
                 + f" | {cc.get('sigma', float('nan')):.3f} |")
    L += ["", "## Calculation", "", "| set | model | minutes | deployed components | draws shapes equal base |", "|---|---|---|---|---|"]
    for sc in doc["sets"]:
        b, r = doc["base_calc"][sc], doc["calc"][sc]
        L.append(f"| {sc} | base | {b['wall_seconds'] / 60:.1f} | {', '.join(b.get('deployed') or []) or '-'} | - |")
        L.append(f"| {sc} | rank | {r['wall_seconds'] / 60:.1f} | {', '.join(r.get('deployed') or []) or '-'} | {r['draws_shapes_equal_base']} |")
    sc_ = doc.get("selfcheck")
    if sc_:
        L += ["", f"## Self-check (TMT, indicators forced to zero vs base): max |dbeta| draw 0 {sc_['max_abs_beta_draw0']:.2e}, "
                  f"all draws {sc_['max_abs_beta_all_draws']:.2e}"]
    L += ["", "## Determinism of base decks", ""] + [f"- {r['pool']}: identical={r['identical']}" for r in doc["determinism"]["rows"]]
    L += ["", f"## Real app home / raw files check: {json.dumps(doc['apphome_check'])}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--stage", default="all", choices=["prepare", "calc", "build", "compare", "all"])
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    pbc.check_out_path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    sets = [s.upper() for s in (args.sets or SETS)]
    snap_before()
    if args.selfcheck:
        selfcheck()
        print("app home check:", json.dumps(snap_check()), flush=True)
        return 0
    stages = ("prepare", "calc", "build", "compare") if args.stage == "all" else (args.stage,)
    for st in stages:
        t0 = time.time()
        if st == "prepare":
            for sc in sets:
                prepare_set(sc, rank_home(sc))
        elif st == "calc":
            for sc in sets:
                calc_set(sc, rank_home(sc), "rank")
        elif st == "build":
            build_stage(sets, args.jobs)
        else:
            compare(sets)
        print(f"stage {st}: {(time.time() - t0) / 60:.1f} min", flush=True)
    print("app home check:", json.dumps(snap_check()), flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) >= 4 and sys.argv[1] == "--calc-child":
        sys.exit(calc_child(sys.argv[2], "--zero" in sys.argv[4:], Path(sys.argv[3])))
    sys.exit(main())
