"""Within-event check (read-only): the same pilot with the same pool registers several different decks in one event.
Does the deck the model scores higher win more?  Linear probability model with event fixed effects:

    won = a_event + b * logit q(deck of this game) + controls (on_play, game order, first-build, bo3 game >= 2)

b / mean(p(1-p)) is the slope on the logit scale: 1 = the model's within-pool differences are realised one for one.

    python scripts/within_event_check.py OUT.json JOBS [SET,SET,...]
    python scripts/within_event_check.py --train-only DIR JOBS [SET,SET,...]

`--train-only DIR` (the standing scoreboard line, docs/within_event/RULE.md): for every DIR/SET written by
`scripts/deviation_cost.py SET DIR/SET`, the train-only model in DIR/SET/home scores the decks of the HOLD-OUT events only (events
that started at or after DIR/SET/deviation_cost.json's cut); results go to DIR/SET/within_event.json and DIR/within_event.json.

Session 12 audit (2026-11-12), results in docs/audit_1112/.  Reads the app home (SEALED_LAB_HOME, default the real one) and
never writes to it; OUT goes wherever you point it.  Several of these may run side by side as plain processes (JOBS = 1);
only one spawn pool at a time on this machine.
"""
import json
import multiprocessing
import os
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _all_sets() -> list[str]:
    from sealed_lab import config

    return sorted(p.name for p in config.sets_dir().iterdir() if p.is_dir())


def task(args):
    set_code, fmt = args[:2]
    home, cut = (args[2], args[3]) if len(args) > 2 else (None, None)
    os.environ["POLARS_MAX_THREADS"] = "2"
    if home:
        os.environ["SEALED_LAB_HOME"] = home
    import numpy as np
    import polars as pl

    from sealed_lab import config
    from sealed_lab.deck import builder

    model = builder.load_model(set_code)
    if fmt not in model.formats:
        return dict(set=set_code, fmt=fmt, error="format not in model")
    ctx = builder.Ctx(model, fmt)
    cd = config.corpus_dir(set_code)
    g = pl.read_parquet(cd / "games.parquet", columns=["fmt", "draft_id", "deck_key", "game_time", "game_number", "match_number", "on_play",
                                                       "won", "build_index"]).filter(pl.col("fmt") == fmt)
    decks = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards", "n_lands"])
    g = g.join(decks, on="deck_key").filter(pl.col("n_cards") == 40)
    if cut is not None:                       # train-only mode: hold-out events only
        from sealed_lab.model import design, validate

        g_all = pl.read_parquet(cd / "games.parquet")              # the cut exactly as deviation_cost.py computes it
        target = next(f for f in ("bo3", "bo1") if g_all.filter(pl.col("fmt") == f).height)
        t_cut = validate.time_split(g_all.filter(pl.col("fmt") == target))[1]
        if str(t_cut) != cut:
            return dict(set=set_code, fmt=fmt, error=f"cut mismatch {t_cut} vs {cut}")
        st = design.event_starts(g_all.filter(pl.col("fmt") == fmt))
        g = g.filter(pl.col("draft_id").is_in(st.filter(pl.col("start") >= t_cut)["draft_id"].to_list()))
    multi = g.group_by("draft_id").agg(pl.col("deck_key").n_unique().alias("k")).filter(pl.col("k") >= 2)
    g = g.join(multi.select("draft_id"), on="draft_id").sort("draft_id", "game_time")
    if g.height < 500:
        return dict(set=set_code, fmt=fmt, error="too few games")
    keys = g["deck_key"].unique().to_list()
    dcs = pl.read_parquet(cd / "deck_cards.parquet").filter(pl.col("deck_key").is_in(keys))
    lg = {}
    for k, v in dcs.group_by("deck_key"):
        deck = dict(zip([int(i) for i in v["card_id"].to_list()], [int(n) for n in v["n"].to_list()]))
        lg[k[0]] = ctx.logit(deck) - ctx.ref
    g = g.with_columns(pl.col("deck_key").replace_strict(lg, default=None, return_dtype=pl.Float64).alias("x"),
                       pl.int_range(pl.len()).over("draft_id").alias("order"),
                       (pl.col("deck_key") == pl.col("deck_key").first().over("draft_id")).alias("first_build"),
                       pl.len().over("draft_id").alias("n_ev")).filter(pl.col("x").is_not_null())
    g = g.with_columns((pl.col("order") / pl.col("n_ev")).alias("order"))
    ev = g["draft_id"].to_physical().to_numpy() if g["draft_id"].dtype != pl.String else g["draft_id"].cast(pl.Categorical).to_physical().to_numpy()
    y = g["won"].to_numpy().astype(float)
    cols = [g["x"].to_numpy(), g["on_play"].fill_null(False).to_numpy().astype(float), g["order"].to_numpy().astype(float),
            g["first_build"].to_numpy().astype(float)]
    if fmt == "bo3":
        cols.append((g["game_number"].fill_null(1).to_numpy() >= 2).astype(float))
    X = np.column_stack(cols)

    def demean(a):
        order = np.argsort(ev, kind="stable")
        e = ev[order]
        cuts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
        cnt = np.diff(np.r_[cuts, len(e)])
        s = np.add.reduceat(a[order], cuts, axis=0)
        m = np.repeat(s / (cnt[:, None] if a.ndim == 2 else cnt), cnt, axis=0)
        out = np.empty_like(a, dtype=float)
        out[order] = a[order] - m
        return out, order, cuts

    Xd, order, cuts = demean(X)
    yd, _, _ = demean(y)
    XtX = Xd.T @ Xd
    b = np.linalg.solve(XtX, Xd.T @ yd)
    r = yd - Xd @ b
    S = np.add.reduceat(Xd[order] * r[order, None], cuts, axis=0)
    V = np.linalg.solve(XtX, np.linalg.solve(XtX, S.T @ S).T)
    se = np.sqrt(np.diag(V))
    pq = float(y.mean() * (1 - y.mean()))
    # the same without controls, and the x contrast alone (later builds minus the first)
    b0 = float((Xd[:, 0] @ yd) / (Xd[:, 0] @ Xd[:, 0]))
    first = g.filter(pl.col("first_build")).group_by("draft_id").agg(pl.col("x").mean().alias("x1"))
    later = g.filter(~pl.col("first_build")).group_by("draft_id").agg(pl.col("x").mean().alias("x2"))
    d = first.join(later, on="draft_id")
    dx = (d["x2"] - d["x1"]).to_numpy()
    return dict(set=set_code, fmt=fmt, n_games=int(len(y)), n_events=int(len(cuts)), n_decks=len(lg),
                b_lpm=float(b[0]), se_lpm=float(se[0]), pq=pq, b_logit=float(b[0] / pq), se_logit=float(se[0] / pq),
                b_nocontrol_logit=b0 / pq, coef_first_build=float(b[3]), se_first_build=float(se[3]),
                coef_order=float(b[2]), sd_x_within=float(Xd[:, 0].std()), mean_dx_later_minus_first=float(dx.mean()),
                share_later_better=float((dx > 0).mean()), sd_dx=float(dx.std()))


if __name__ == "__main__":
    train_dir = None
    if sys.argv[1] == "--train-only":
        train_dir = Path(sys.argv[2]).resolve()
        del sys.argv[1]
    out, jobs = sys.argv[1], int(sys.argv[2])
    if train_dir is not None:
        sets = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(p.name for p in train_dir.iterdir() if (p / "deviation_cost.json").exists())
        cuts = {s: json.loads((train_dir / s / "deviation_cost.json").read_text(encoding="utf-8"))["cut"] for s in sets}
        tasks = [(s, f, str(train_dir / s / "home"), cuts[s]) for s in sets for f in ("bo3", "bo1")]
        out = str(train_dir / "within_event.json")
    else:
        sets = sys.argv[3].split(",") if len(sys.argv) > 3 else _all_sets()
        tasks = [(s, f) for s in sets for f in ("bo3", "bo1")]
    res = []
    t0 = time.time()
    if jobs <= 1:
        for t in tasks:
            res.append(task(t))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[-1].items()}, round(time.time() - t0), flush=True)
    else:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            for r in ex.map(task, tasks, chunksize=1):
                res.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, round(time.time() - t0), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    if train_dir is not None:
        for s in sets:
            (train_dir / s / "within_event.json").write_text(json.dumps([r for r in res if r["set"] == s], indent=1), encoding="utf-8")
