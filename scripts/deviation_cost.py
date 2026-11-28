"""Deviation-cost test: are the builder's predicted gains real?

    python scripts/deviation_cost.py SET OUTDIR [--fmt bo3 bo1 | dbo1 dbo3 | draft] [--max-events 900] [--jobs N] [--boot 300] [--legacy] [--matchups auto|on|off] [--extra-draft auto|on|off]

1. A train-only model is fitted on the events (all formats) that started before the hold-out cut (start of the last
   25% of target-format events, the same split as the report), in a private copy of the set under OUTDIR/home.
2. For every held-out build: q_h = model's q of the deck the player registered; q_b = q of the builder's best deck
   from that player's pool (deck + sideboard).
3. Game-level logistic regression on the held-out games (pilot covariates fitted jointly, as in baseline B0):
       logit P(win) = covariates + a * (logit q_b - ref) + b * (logit q_b - logit q_h)
   The model predicts a = 1 and b = -1: the further a player's deck falls below the builder's best, the more they
   lose. b ~ 0 means deviating from the builder costs nothing - its predicted gains would be an artefact
   (winner's curse / extrapolation). -b is the share of the predicted gain that shows up in real results.
   95% intervals: cluster bootstrap over held-out events.
Writes OUTDIR/deviation_cost.json (rewritten after every format) and prints a summary. Never touches the real model.

The source set is read from SEALED_LAB_HOME at start-up (point it at a scratch copy, never at the real app home). The held-out
events are chosen with a fixed seed from the events after the cut, so two runs on the same data (e.g. the default and `--legacy`
model options) use the same events; each format's result lists them (`event_ids`) so this can be verified.
`--variant NAME` builds with a builder variant of `scripts/search_check.py` VARIANTS (e.g. `wide`, docs/builder_wide/RULE.md) on
the train-only model already in OUTDIR/home (never refitted for a variant run); `--results DIR` writes rows_*.json and
deviation_cost.json there instead of OUTDIR, so `builder_scoreboard.py VARIANTDIR OUT --legacy SHIPPEDDIR` pairs the two runs.
`--opp-blind` blanks opp_colors in the training games and switches the matchup layer off (docs/opp_blind/RULE.md).
Draft formats (docs/draft_mode/RULE.md, guardrail G1): `--fmt dbo1 dbo3` (or `--fmt draft` = the draft formats the set has; also mixed with
bo1 / bo3) fits the train-only model with `draft_formats=True` (`--draft-boot`, default 4) on the events before EACH format's own hold-out
cut (Sealed events before the Sealed cut = `split_half_search.training_events`' cut; the dbo1 / dbo3 cut = `validate.time_split` of that
format's games; the other draft events before the dbo1 cut) and scores the held-out events of the draft format (`rows_dbo1.json`...).
`builder_scoreboard.py` then pools the slope b of each draft format across sets and prints the G1 line. Without a draft format in `--fmt`
nothing changes.
Second test (docs/draft_mode2/RULE.md, G1): `--exclude-first-300 --max-events 2000 --seed 1` recomputes the first G1's events (`rng =
default_rng(0)`, 300 of the held-out events, `draft_split_half.first_g1_events`), excludes them, and takes up to `--max-events` of the remaining
held-out events with `default_rng(--seed)` (`draft_split_half.new_g1_events`). Both lists are written to RESULTS/events.json before the model is
fitted or anything is built (an existing file must be identical), with zero overlap asserted. Without the flag nothing changes.
`--legacy` fits the train-only model with the pre-Session-8 options (`--sigma-cv random --stack off --no-confirm
--recency-scope pooled`) for a paired comparison.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # search_check (builder variants)


def _build_task(task: dict) -> dict:
    """Worker: builder's best deck for one pool and q of both decks under the train-only model."""
    os.environ["SEALED_LAB_HOME"] = task["home"]
    from sealed_lab.deck import builder

    model = builder.load_model(task["set"])
    # top family representative; this worker is already one of N processes, so the family search runs in-process
    import contextlib

    patch = contextlib.nullcontext()
    if task.get("variant"):
        from search_check import VARIANTS, patched_search

        patch = patched_search(VARIANTS[task["variant"]])
    with patch:
        res = builder.build(model, task["pool"], task["fmt"], n_decks=1, families=True, n_jobs=1)
    best = {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}
    ca = model.card_arrays

    def comp(deck: dict[int, int]) -> tuple[int, int]:
        """(nonland creature copies, land copies)"""
        return (sum(n for i, n in deck.items() if not ca.is_land[i] and ca.is_creature[i]),
                sum(n for i, n in deck.items() if ca.is_land[i]))

    out = dict(draft_id=task["draft_id"], logit_b=model.logit(best, task["fmt"]), builds={}, comp_b=comp(best), comp_h={})
    for bk, deck in task["decks"].items():
        out["builds"][bk] = model.logit(deck, task["fmt"])
        out["comp_h"][bk] = comp(deck)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("outdir")
    ap.add_argument("--fmt", nargs="+", default=["bo3", "bo1"], help="bo3 bo1 and / or dbo1 dbo3 (draft formats); 'draft' = both draft formats")
    ap.add_argument("--draft-boot", type=int, default=4, help="bootstrap draws of the draft formats of the train-only model (draft formats only)")
    ap.add_argument("--max-events", type=int, default=900)
    ap.add_argument("--exclude-first-300", action="store_true",
                    help="second test (docs/draft_mode2/RULE.md): exclude the 300 held-out events of the first G1 (recomputed: default_rng(0), "
                         "--max-events 300) and take up to --max-events of the rest with default_rng(--seed)")
    ap.add_argument("--events-only", action="store_true",
                    help="with --exclude-first-300: write RESULTS/events.json (the event lists, fixed before any fit or build) and stop")
    ap.add_argument("--seed", type=int, default=0, help="seed of the event draw with --exclude-first-300 (the rule: 1); ignored without it")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--top-skill", type=float, default=0.60,
                    help="17lands game win-rate bucket from which a pilot counts as a top player (split analysis)")
    ap.add_argument("--reuse-rows", action="store_true",
                    help="reuse the builder results saved in OUTDIR/rows_FMT.json by an earlier run (same events) instead of building")
    ap.add_argument("--refit", action="store_true", help="refit the train-only model even if OUTDIR has one")
    ap.add_argument("--variant", default=None, help="builder variant (scripts/search_check.py VARIANTS) on the existing train-only model")
    ap.add_argument("--results", default=None, help="folder for rows_*.json and deviation_cost.json (default OUTDIR)")
    ap.add_argument("--attributes", default="auto", choices=["auto", "on", "off"],
                    help="card-attribute prior of the train-only model (on = force it, to test the builder with it)")
    ap.add_argument("--manasim", default="auto", choices=["auto", "v2", "v1"],
                    help="mana simulator of the train-only model (v2 = force the calibrated simulator, v1 = the original)")
    ap.add_argument("--castability", default="auto", choices=["auto", "on", "off"],
                    help="castability of the train-only model (on = force it: card values depend on whether the deck's mana casts them)")
    ap.add_argument("--oppflags", default="off", choices=["auto", "on", "off"],
                    help="opponent colour flags in the game covariates of the train-only model (on = force them)")
    ap.add_argument("--opp-blind", action="store_true",
                    help="blank opp_colors in every training game (docs/opp_blind/RULE.md): the opponent pair one-hot becomes a constant "
                         "and the matchup layer is off")
    ap.add_argument("--matchups", default="auto", choices=["auto", "on", "off"],
                    help="card x opponent-colour interactions of the train-only model (on = force them, to test the builder with them)")
    ap.add_argument("--extra-draft", default="auto", choices=["auto", "on", "off"],
                    help="PickTwo / QuickDraft draft games of the train-only model (on = keep them untested, as before Session 11b)")
    ap.add_argument("--sigma-cv", default="time", choices=["time", "random"],
                    help="how the Sealed stages choose their prior strength (time = forward-chaining folds, random = pre-Session-8)")
    ap.add_argument("--stack", default="auto", choices=["auto", "off"],
                    help="auto: card values q = w M + (1 - w) P with w chosen on the folds; off: M only (pre-Session-8)")
    ap.add_argument("--no-confirm", action="store_true", help="components need no confirmation on the later half (pre-Session-8)")
    ap.add_argument("--recency-scope", default="format", choices=["format", "pooled"],
                    help="half-life chosen per format or one for all (pre-Session-8)")
    ap.add_argument("--theta-sigma", default="rule", choices=["rule", "cv"],
                    help="prior SD of the structure terms in the Sealed stages: rule = clip(4 x card sigma), cv = chosen by CV")
    ap.add_argument("--legacy", action="store_true",
                    help="shorthand for --sigma-cv random --stack off --no-confirm --recency-scope pooled")
    args = ap.parse_args(argv)
    if args.opp_blind:
        args.matchups = "off"
    if args.legacy:
        args.sigma_cv, args.stack, args.no_confirm, args.recency_scope = "random", "off", True, "pooled"
    args.fmt = [f for x in args.fmt for f in (("dbo1", "dbo3") if x == "draft" else (x,))]
    draft_fmts = [f for f in args.fmt if f in ("dbo1", "dbo3")]
    model_opts = dict(sigma_cv=args.sigma_cv, stack=args.stack, confirm=not args.no_confirm, recency_scope=args.recency_scope,
                      theta_sigma=args.theta_sigma, extra_draft=args.extra_draft)

    import polars as pl
    import scipy.sparse as sp
    from scipy.special import expit

    from sealed_lab import config
    from sealed_lab.model import design, glm, pipeline, validate

    sc = args.set.upper()
    out = Path(args.outdir).resolve()
    res_dir = Path(args.results).resolve() if args.results else out
    res_dir.mkdir(parents=True, exist_ok=True)
    if args.variant:
        from search_check import VARIANTS

        if args.variant not in VARIANTS:
            raise SystemExit(f"unknown variant {args.variant!r}")
        if args.refit or not (out / "home" / "sets" / sc / "model" / "meta.json").exists():
            raise SystemExit("a variant run needs the train-only model of a shipped-builder run in OUTDIR (and no --refit)")
    src_set = config.set_dir(sc)
    home = out / "home"
    dst_set = home / "sets" / sc
    games_all = pl.read_parquet(config.corpus_dir(sc) / "games.parquet")
    cut_of: dict = {}
    if draft_fmts:         # Sealed cut + one cut per draft format (draft_split_half.cuts_of), games with the draft formats' names
        from draft_split_half import named_games, train_only_ids

        cuts_d, train_ids_d = train_only_ids(games_all)
        if cuts_d["sealed"] is None:
            print("not enough Sealed events for a hold-out (the Sealed cut is needed for the train-only model)")
            return 1
        t_cut = cuts_d["sealed"]
        draft_fmts = [f for f in draft_fmts if cuts_d[f] is not None]
        args.fmt = [f for f in args.fmt if f not in ("dbo1", "dbo3") or f in draft_fmts]
        cut_of = {f: cuts_d[f] for f in draft_fmts}
        games_all = named_games(games_all)
        print(f"hold-out cuts: Sealed {t_cut}, " + ", ".join(f"{f} {c}" for f, c in cut_of.items()))
    else:
        target = next(f for f in args.fmt if games_all.filter(pl.col("fmt") == f).height)
        split = validate.time_split(games_all.filter(pl.col("fmt") == target))
        if split is None:
            print("not enough events for a hold-out")
            return 1
        _, t_cut = split
        print(f"hold-out cut {t_cut} (start of the last 25% of {target} events)")

    # ---- 0. second test: the event lists, fixed and saved before the model is fitted or anything is built
    chosen: dict = {}
    if args.exclude_first_300:
        from draft_split_half import FIRST_G1_MAX_EVENTS, first_g1_events, new_g1_events

        doc: dict = dict(set=sc, seed=args.seed, max_events=args.max_events, first_max_events=FIRST_G1_MAX_EVENTS, formats={})
        for fmt in args.fmt:
            ho = design.event_starts(games_all.filter(pl.col("fmt") == fmt)).filter(pl.col("start") >= cut_of.get(fmt, t_cut))["draft_id"].to_list()
            if not ho:
                continue
            first = first_g1_events(ho)
            chosen[fmt] = new_g1_events(ho, first, args.max_events, args.seed)
            doc["formats"][fmt] = dict(n_heldout=len(ho), n_first=len(first), n_new=len(chosen[fmt]), overlap=0,
                                       first=sorted(str(d) for d in first), new=sorted(str(d) for d in chosen[fmt]))
            print(f"{fmt}: {len(ho)} held-out events, first G1 used {len(first)}, new list {len(chosen[fmt])} (overlap 0)")
        ev_path = res_dir / "events.json"
        if ev_path.exists():
            if json.loads(ev_path.read_text(encoding="utf-8")) != json.loads(json.dumps(doc)):
                raise SystemExit(f"{ev_path} exists and differs from the recomputed event lists: the lists are fixed before any build")
        else:
            ev_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")

    if args.events_only:
        if not args.exclude_first_300:
            raise SystemExit("--events-only needs --exclude-first-300")
        return 0

    # ---- 1. train-only model in a private home
    if args.refit or not (dst_set / "model" / "meta.json").exists():
        if dst_set.exists():
            shutil.rmtree(dst_set)
        dst_set.mkdir(parents=True)
        for sub in ("corpus", "cache"):
            shutil.copytree(src_set / sub, dst_set / sub)
        shutil.copy(src_set / "registry.json", dst_set / "registry.json")
        os.environ["SEALED_LAB_HOME"] = str(home)
        if draft_fmts:
            train_ids = train_ids_d
        else:
            starts = design.event_starts(games_all)
            train_ids = set(starts.filter(pl.col("start") < t_cut)["draft_id"].to_list())
        orig = pipeline.load_inputs

        def load_train(set_code):
            inp = orig(set_code)
            inp.games = inp.games.filter(pl.col("draft_id").is_in(train_ids))
            inp.builds = inp.builds.filter(pl.col("draft_id").is_in(train_ids))
            if args.opp_blind:
                inp.games = inp.games.with_columns(pl.lit("").alias("opp_colors"))
            return inp

        pipeline.load_inputs = load_train
        t0 = time.time()
        pipeline.calculate(sc, progress=lambda f, m: None, n_jobs=args.jobs, n_boot=4, recency="off", attributes=args.attributes,
                           manasim=args.manasim, castability=args.castability, matchups=args.matchups, oppflags=args.oppflags,
                           **(dict(draft_formats=True, draft_boot=args.draft_boot) if draft_fmts else {}), **model_opts)
        pipeline.load_inputs = orig
        fit_seconds = time.time() - t0
        print(f"train-only model fitted in {fit_seconds:.0f}s (options {model_opts})")
    else:
        fit_seconds = None
    os.environ["SEALED_LAB_HOME"] = str(home)

    pools = pl.read_parquet(config.corpus_dir(sc) / "pools.parquet")
    dcards = pl.read_parquet(config.corpus_dir(sc) / "deck_cards.parquet")
    decks = pl.read_parquet(config.corpus_dir(sc) / "decks.parquet")
    results = {"set": sc, "variant": args.variant, "cut": str(t_cut), **({"cuts": {"sealed": str(t_cut), **{f: str(c) for f, c in cut_of.items()}}} if draft_fmts else {}), "model_opts": dict(model_opts, matchups=args.matchups, opp_blind=bool(args.opp_blind)), "fit_seconds": fit_seconds, "max_events": args.max_events,
               "formats": {}}
    for fmt in args.fmt:
        t_cut_f = cut_of.get(fmt, t_cut)             # the format's own hold-out cut (the Sealed formats share `t_cut`)
        g = games_all.filter(pl.col("fmt") == fmt)
        starts = design.event_starts(g)
        ho_ids = starts.filter(pl.col("start") >= t_cut_f)["draft_id"].to_list()
        if not ho_ids:
            continue
        rng = np.random.default_rng(0)
        if args.exclude_first_300:
            ho_ids = chosen.get(fmt, [])
        elif len(ho_ids) > args.max_events:
            ho_ids = sorted(rng.choice(ho_ids, args.max_events, replace=False).tolist())
        if not ho_ids:
            continue
        g_ho = g.filter(pl.col("draft_id").is_in(ho_ids)).join(decks.select("deck_key", "n_cards"), on="deck_key", how="left")
        g_ho = g_ho.filter(pl.col("n_cards") == 40)
        tasks = []
        pool_by = {d: dict(zip(p["card_id"].to_list(), p["n"].to_list())) for d, p in
                   pools.filter(pl.col("draft_id").is_in(ho_ids)).partition_by("draft_id", as_dict=True, include_key=True).items()}
        pool_by = {(k[0] if isinstance(k, tuple) else k): v for k, v in pool_by.items()}
        for did, gg in g_ho.partition_by("draft_id", as_dict=True).items():
            did = did[0] if isinstance(did, tuple) else did
            if did not in pool_by:
                continue
            dk = {}
            for key in gg["deck_key"].unique().to_list():
                dk[str(key)] = {int(c): int(n) for c, n in dcards.filter(pl.col("deck_key") == key).select("card_id", "n").iter_rows()}
            tasks.append(dict(home=str(home), set=sc, fmt=fmt, draft_id=did, pool={int(k): int(v) for k, v in pool_by[did].items()},
                              decks=dk, variant=args.variant))
        print(f"{fmt}: {len(tasks)} held-out events with a pool; building…")
        t0 = time.time()
        rows_path = res_dir / f"rows_{fmt}.json"
        want = sorted(str(t["draft_id"]) for t in tasks)
        rows = []
        if args.reuse_rows and rows_path.exists():
            saved = json.loads(rows_path.read_text(encoding="utf-8"))
            if sorted(str(r["draft_id"]) for r in saved) == want:
                rows = saved
                print("  reusing the saved builder results")
        if not rows:
            ctx = multiprocessing.get_context("spawn")
            with cf.ProcessPoolExecutor(max_workers=args.jobs or os.cpu_count(), mp_context=ctx) as ex:
                for r in ex.map(_build_task, tasks, chunksize=4):
                    rows.append(r)
            rows_path.write_text(json.dumps(rows), encoding="utf-8")
        build_seconds = time.time() - t0
        print(f"  built in {build_seconds:.0f}s")
        lb = {r["draft_id"]: r["logit_b"] for r in rows}
        lh = {(r["draft_id"], int(k)): v for r in rows for k, v in r["builds"].items()}
        g_ho = g_ho.filter(pl.col("draft_id").is_in(list(lb)))
        did = g_ho["draft_id"].to_list()
        dkey = g_ho["deck_key"].to_list()
        Lb = np.array([lb[d] for d in did])
        Lh = np.array([lh[(d, int(k))] for d, k in zip(did, dkey)])
        gap = Lb - Lh

        # pilot covariates encoded as in the training stage of this format (time clamped at the cut)
        g_tr = g.filter(pl.col("draft_id").is_in(set(design.event_starts(g).filter(pl.col("start") < t_cut_f)["draft_id"].to_list())))
        enc = design.CovEncoder.fit(g_tr)
        C = enc.transform(g_ho, clamp_time=True)
        ref = float(np.median(Lh))
        X = sp.hstack([C, sp.csr_matrix(np.column_stack([Lb - ref, gap]))], format="csr")
        y = g_ho["won"].cast(pl.Float64).to_numpy()
        sig = np.concatenate([enc.sigma, [10.0, 10.0]])
        prob = glm.Problem(y=y, C=X)

        def fit(w=None):
            return glm.fit(prob, np.zeros(len(sig)), sig, w=w, maxiter=400).coef[-2:]

        a, b = fit()
        groups = np.unique(np.array(did), return_inverse=True)[1]
        G = int(groups.max()) + 1
        mult = rng.multinomial(G, np.full(G, 1.0 / G), size=args.boot).astype(float)
        boots = np.array([fit(mult[i][groups]) for i in range(args.boot)])
        # descriptive: held-out win rate by gap quintile, net of covariates-only predictions
        C0 = sp.csr_matrix(C)
        b0 = glm.fit(glm.Problem(y=y, C=C0), np.zeros(len(enc.sigma)), enc.sigma, maxiter=400).coef
        p0 = expit(C0 @ b0)
        q_ev = {}
        for d, gp in zip(did, gap):
            q_ev.setdefault(d, gp)
        ev_gap = np.array([q_ev[d] for d in did])
        edges = np.quantile(list(q_ev.values()), [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(edges, ev_gap)
        quint = []
        for k in range(5):
            m = qi == k
            pred_cost = float(np.mean(expit(Lh[m]) - expit(Lb[m])))     # model: q_h - q_b (negative)
            quint.append(dict(quintile=k + 1, games=int(m.sum()), mean_gap_logit=float(gap[m].mean()),
                              predicted_wr_cost_pp=100 * pred_cost, observed_minus_covariates_pp=100 * float((y[m] - p0[m]).mean())))
        # composition of the deviation: does playing more creatures / lands than the builder cost (or gain) beyond what
        # the model's gap already predicts? 0 = the model prices the trade correctly; d = player's deck minus builder's.
        cb = {r["draft_id"]: r["comp_b"] for r in rows}
        ch = {(r["draft_id"], int(k)): v for r in rows for k, v in r["comp_h"].items()}
        d_cre = np.array([ch[(d, int(k))][0] - cb[d][0] for d, k in zip(did, dkey)], dtype=float)
        d_land = np.array([ch[(d, int(k))][1] - cb[d][1] for d, k in zip(did, dkey)], dtype=float)
        X2 = sp.hstack([C, sp.csr_matrix(np.column_stack([Lb - ref, gap, d_cre, d_land]))], format="csr")
        sig2 = np.concatenate([enc.sigma, [10.0] * 4])
        prob2 = glm.Problem(y=y, C=X2)

        def fit2(w=None):
            return glm.fit(prob2, np.zeros(len(sig2)), sig2, w=w, maxiter=400).coef[-4:]

        c2 = fit2()
        boots2 = np.array([fit2(mult[i][groups]) for i in range(args.boot)])
        comp_terms = {}
        for j, nm in enumerate(("a", "b", "per_creature_more_than_builder", "per_land_more_than_builder")):
            comp_terms[nm] = dict(coef=float(c2[j]), lo=float(np.percentile(boots2[:, j], 2.5)),
                                  hi=float(np.percentile(boots2[:, j], 97.5)))
            if j >= 2:
                comp_terms[nm].update(pp=25 * float(c2[j]), lo_pp=25 * comp_terms[nm]["lo"], hi_pp=25 * comp_terms[nm]["hi"])
        comp_terms["mean_d_creatures"] = float(d_cre.mean())
        comp_terms["mean_d_lands"] = float(d_land.mean())

        # top players vs the rest: one joint fit with every term interacted with "top" (pilot win-rate bucket >= --top-skill;
        # the skill level itself is already a covariate). For each group: b (cost of the model's gap, ideal -1) and the
        # creature / land terms beyond the gap (ideal 0). A top-player b closer to 0 than the rest, or positive creature / land
        # terms for top players only, would mean their deviations from the builder carry knowledge the model lacks.
        sk = g_ho["skill"].cast(pl.Float64).to_numpy()
        top = np.where(np.isnan(sk), 0.0, (sk >= args.top_skill).astype(float))
        base_cols = np.column_stack([Lb - ref, gap, d_cre, d_land])
        X3 = sp.hstack([C, sp.csr_matrix(np.column_stack([base_cols, base_cols * top[:, None]]))], format="csr")
        sig3 = np.concatenate([enc.sigma, [10.0] * 8])
        prob3 = glm.Problem(y=y, C=X3)

        def fit3(w=None):
            return glm.fit(prob3, np.zeros(len(sig3)), sig3, w=w, maxiter=400).coef[-8:]

        c3 = fit3()
        boots3 = np.array([fit3(mult[i][groups]) for i in range(args.boot)])
        names = ("a", "b", "per_creature", "per_land")
        split = {"threshold": args.top_skill, "top_share_games": float(top.mean())}

        def ci(v):
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

        for grp, sel in (("rest", 0.0), ("top", 1.0)):
            m = top == sel
            gdict = {"games": int(m.sum()), "events": int(len(set(np.array(did)[m].tolist()))),
                     "mean_gap_pp": 100 * float(np.mean(expit(Lb[m]) - expit(Lh[m]))) if m.any() else float("nan"),
                     "mean_d_creatures": float(d_cre[m].mean()) if m.any() else float("nan"),
                     "mean_d_lands": float(d_land[m].mean()) if m.any() else float("nan")}
            for j, nm in enumerate(names):
                est = c3[j] + sel * c3[4 + j]
                bv = boots3[:, j] + sel * boots3[:, 4 + j]
                scale = 25.0 if nm.startswith("per_") else 1.0
                gdict[nm] = dict(est=scale * float(est), ci=[scale * x for x in ci(bv)])
            split[grp] = gdict
        for j, nm in enumerate(names):
            scale = 25.0 if nm.startswith("per_") else 1.0
            split["diff_" + nm] = dict(est=scale * float(c3[4 + j]), ci=[scale * x for x in ci(boots3[:, 4 + j])])
        res = dict(n_events=G, n_games=int(len(y)), a=float(a), b=float(b), composition=comp_terms, top_split=split,
                   event_ids=sorted(str(d) for d in lb), build_seconds=build_seconds,
                   # bootstrap replicates of b (rng seeded identically, so runs on the same events share their multiplicities
                   # and the difference of two runs' b has a paired interval)
                   b_boots=[float(v) for v in boots[:, 1]], a_boots=[float(v) for v in boots[:, 0]],
                   a_ci=[float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))],
                   b_ci=[float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5))],
                   mean_gap_logit=float(np.mean(list(q_ev.values()))),
                   mean_predicted_gain_pp=100 * float(np.mean([expit(lb[d]) for d in q_ev]) - np.mean(expit(Lh))),
                   quintiles=quint)
        results["formats"][fmt] = res
        print(f"  {fmt}: {G} events, {len(y)} games | pool strength a = {a:+.2f} {res['a_ci']} (model: +1) | "
              f"gap b = {b:+.2f} {res['b_ci']} (model: -1, artefact: 0)")
        for nm, mean_key in (("per_creature_more_than_builder", "mean_d_creatures"), ("per_land_more_than_builder", "mean_d_lands")):
            t = comp_terms[nm]
            print(f"    {nm}: {t['pp']:+.2f} pp [{t['lo_pp']:+.2f}, {t['hi_pp']:+.2f}] beyond the model's gap "
                  f"(mean player-minus-builder {comp_terms[mean_key]:+.2f})")
        for grp in ("rest", "top"):
            gd = split[grp]
            print(f"    {grp:4s} ({gd['games']} games, {gd['events']} events; decks {gd['mean_gap_pp']:+.1f} pp below the builder's, "
                  f"{gd['mean_d_creatures']:+.2f} creatures, {gd['mean_d_lands']:+.2f} lands vs builder): "
                  f"b = {gd['b']['est']:+.2f} [{gd['b']['ci'][0]:+.2f}, {gd['b']['ci'][1]:+.2f}] | "
                  f"per creature {gd['per_creature']['est']:+.2f} pp [{gd['per_creature']['ci'][0]:+.2f}, {gd['per_creature']['ci'][1]:+.2f}] | "
                  f"per land {gd['per_land']['est']:+.2f} pp [{gd['per_land']['ci'][0]:+.2f}, {gd['per_land']['ci'][1]:+.2f}]")
        for nm in ("b", "per_creature", "per_land"):
            dd = split["diff_" + nm]
            print(f"    top minus rest, {nm}: {dd['est']:+.2f} [{dd['ci'][0]:+.2f}, {dd['ci'][1]:+.2f}]")
        for qd in quint:
            print(f"    gap quintile {qd['quintile']}: predicted {qd['predicted_wr_cost_pp']:+.1f} pp, observed vs covariates "
                  f"{qd['observed_minus_covariates_pp']:+.1f} pp ({qd['games']} games)")
        (res_dir / "deviation_cost.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (res_dir / "deviation_cost.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
