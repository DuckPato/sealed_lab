"""Opponent colours blanked in the fit: pre-game hold-out test and card-value report (docs/opp_blind/RULE.md).

    python scripts/opp_blind_compare.py SDIR BDIR OUT [--sets ECL,EOE,...] [--jobs N] [--boot 2000]

SDIR / BDIR each hold one folder per set written by `scripts/deviation_cost.py SET DIR/SET` (train-only model in DIR/SET/home, the
cut in DIR/SET/deviation_cost.json key "cut"). S = shipped arm, B = opponent-blind arm. Read-only on both. Writes OUT.json and
OUT.md (OUT without extension). `--jobs 0` runs the tasks in-process (no pool); the default uses a spawn ProcessPoolExecutor.

Per set x format (bo1, bo3; skipped with a reason when either model lacks the format):
    hold-out     = all games of the format's events that started at or after the cut, 40-card decks (as field_holdout.task);
                   games and decks are read from the S arm's corpus (both arms were fitted on the same corpus; card names are asserted equal)
    x_arm        = deck logit of the arm's model (builder.Ctx.logit, draw 0) minus that model's reference logit Ctx.ref
    loss_arm     = per-game log loss of the logistic with offset x_arm and ML pilot covariates (intercept, skill, skill^2, skill missing,
                   on the play), the same X for both arms (arm P of field_holdout)
    d            = loss_S - loss_B per game (positive = blanking predicts better); whole / half A (earlier events) / half B with the
                   95% event-cluster bootstrap (seed 17); cell verdict by field_holdout.verdict.
Pooled per format (DerSimonian-Laird, zero-width cells excluded with a reason): "blanking better" = pooled-better and no cell
materially worse; "blanking worse" = any pooled part with hi < 0 and gain <= -MIN_GAIN; otherwise "no detectable effect".

Reported, not decisive (per cell): SD of x under S and B over the hold-out games, slope (OLS with intercept) and correlation of
x_B on x_S; card values: the per-card value is Ctx.beta (draw 0) = the per-copy card coefficient of the fitted model, the vector
that Ctx.logit multiplies by the copy counts (sealed_lab/deck/builder.py). Over the non-basic cards that appear in the hold-out
decks (card weighted equally) the slope of beta_B on beta_S (OLS with intercept), their correlation, and RMS(beta_B - beta_S) / SD(beta_S).

Models are loaded with SetModel.load directly under each arm's SEALED_LAB_HOME (not through builder.load_model, whose cache is keyed
by set code and file mtime only), and the second load is guarded by a check that the model really came from the second home.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from field_holdout import FORMATS, MIN_GAIN, Z90_1S, Z95, boot_mean, dl_pool, logistic_offset, verdict  # noqa: E402,F401


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Slope of y on x (with intercept) and correlation; None when x or y has no spread."""
    if len(x) < 2 or x.std() == 0:
        return None, None
    slope = float(np.cov(x, y, bias=True)[0, 1] / x.var())
    corr = float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 else None
    return slope, corr


def task(args) -> list[dict]:
    set_code, s_home, b_home, cut, n_boot = args
    os.environ["POLARS_MAX_THREADS"] = "2"
    import polars as pl

    from sealed_lab import config
    from sealed_lab.deck import builder
    from sealed_lab.model import design, validate
    from sealed_lab.model.artifact import SetModel

    os.environ["SEALED_LAB_HOME"] = s_home
    cd = config.corpus_dir(set_code)
    g_all = pl.read_parquet(cd / "games.parquet")
    target = next(f for f in ("bo3", "bo1") if g_all.filter(pl.col("fmt") == f).height)
    t_cut = validate.time_split(g_all.filter(pl.col("fmt") == target))[1]
    if str(t_cut) != cut:
        return [dict(set=set_code, fmt=f, status="error", reason=f"cut mismatch {t_cut} vs {cut}") for f in FORMATS]
    decks = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards"])
    dcards = pl.read_parquet(cd / "deck_cards.parquet")

    def load(home: str):
        os.environ["SEALED_LAB_HOME"] = home
        assert str(config.app_home()) == home
        m = SetModel.load(set_code)
        return m

    mS = load(s_home)
    mB = load(b_home)
    os.environ["SEALED_LAB_HOME"] = s_home
    if mS.cards.sort("card_id")["name"].to_list() != mB.cards.sort("card_id")["name"].to_list():
        return [dict(set=set_code, fmt=f, status="error", reason="card tables differ between the arms") for f in FORMATS]
    out = []
    for fmt in FORMATS:
        rec = dict(set=set_code, fmt=fmt)
        missing = [n for n, m in (("S", mS), ("B", mB)) if fmt not in m.formats]
        if missing:
            out.append(dict(rec, status="skip", reason=f"format not in the model of arm {'/'.join(missing)}"))
            continue
        g = g_all.filter(pl.col("fmt") == fmt).join(decks, on="deck_key").filter(pl.col("n_cards") == 40)
        st = design.event_starts(g_all.filter(pl.col("fmt") == fmt))
        ho = st.filter(pl.col("start") >= t_cut)
        te = g.filter(pl.col("draft_id").is_in(ho["draft_id"].to_list()))
        if te.height < 500:
            out.append(dict(rec, status="skip", reason=f"too few hold-out games ({te.height})"))
            continue
        keys = te["deck_key"].unique().to_list()
        dec = {}
        for k, v in dcards.filter(pl.col("deck_key").is_in(keys)).group_by("deck_key"):
            dec[k[0]] = dict(zip([int(i) for i in v["card_id"].to_list()], [int(n) for n in v["n"].to_list()]))
        xs: dict[str, dict] = {}
        betas = {}
        for arm, home, m in (("S", s_home, mS), ("B", b_home, mB)):
            os.environ["SEALED_LAB_HOME"] = home
            ctx = builder.Ctx(m, fmt)
            xs[arm] = {k: ctx.logit(d) - ctx.ref for k, d in dec.items()}
            betas[arm] = (ctx.beta.copy(), ctx.is_basic.copy())
        os.environ["SEALED_LAB_HOME"] = s_home
        te = te.filter(pl.col("deck_key").is_in(list(dec)))
        dk = te["deck_key"].to_list()
        xS = np.array([xs["S"][k] for k in dk])
        xB = np.array([xs["B"][k] for k in dk])
        if s_home != b_home and np.allclose(xS, xB, atol=1e-12):
            out.append(dict(rec, status="error", reason="deck logits identical although the homes differ (model load not separated?)"))
            continue
        sk = te["skill"].to_numpy().astype(float)
        miss = np.isnan(sk)
        sk = np.where(miss, np.nanmean(sk) if (~miss).any() else 0.0, sk)
        y = te["won"].to_numpy().astype(float)
        X = np.column_stack([np.ones(len(y)), sk, sk ** 2, miss.astype(float), te["on_play"].fill_null(False).to_numpy().astype(float)])
        d = logistic_offset(X, y, xS) - logistic_offset(X, y, xB)
        ev = np.array([str(x) for x in te["draft_id"].to_list()])
        order = [str(x) for x in ho["draft_id"].to_list()]
        half_a = set(order[: len(order) // 2])
        in_a = np.array([e in half_a for e in ev])
        res = {}
        for name, m in (("whole", np.ones(len(d), bool)), ("A", in_a), ("B", ~in_a)):
            gain, lo, hi = boot_mean(d[m], ev[m], n_boot, seed=17)
            res[name] = dict(gain=gain, lo=lo, hi=hi, n_games=int(m.sum()), n_events=int(len(set(ev[m]))))
        slope_x, corr_x = _ols(xS, xB)
        used = np.zeros(len(betas["S"][0]), bool)
        for dd in dec.values():
            used[list(dd)] = True
        used &= ~betas["S"][1]
        bS, bB = betas["S"][0][used], betas["B"][0][used]
        slope_c, corr_c = _ols(bS, bB)
        rms = float(np.sqrt(np.mean((bB - bS) ** 2)))
        out.append(dict(rec, status="ok", sd_x_S=float(xS.std()), sd_x_B=float(xB.std()), slope_x=slope_x, corr_x=corr_x,
                        n_cards=int(used.sum()), card_slope=slope_c, card_corr=corr_c,
                        card_rms_over_sd=(rms / float(bS.std())) if bS.std() > 0 else None, **res))
    return out


def _fmt_opt(v, spec=".3f") -> str:
    return "n/a" if v is None else format(v, spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sdir")
    ap.add_argument("bdir")
    ap.add_argument("out")
    ap.add_argument("--sets", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    sroot, broot = Path(a.sdir).resolve(), Path(a.bdir).resolve()
    sets = a.sets.split(",") if a.sets else sorted(
        p.name for p in sroot.iterdir() if (p / "deviation_cost.json").exists() and (broot / p.name / "deviation_cost.json").exists())
    tasks, cells = [], []
    for s in sets:
        cs = json.loads((sroot / s / "deviation_cost.json").read_text(encoding="utf-8"))["cut"]
        cb = json.loads((broot / s / "deviation_cost.json").read_text(encoding="utf-8"))["cut"]
        if cs != cb:
            cells += [dict(set=s, fmt=f, status="error", reason=f"cut strings differ between arms ({cs} vs {cb})") for f in FORMATS]
            continue
        tasks.append((s, str(sroot / s / "home"), str(broot / s / "home"), cs, a.boot))

    def show(r):
        for c in r:
            print({k: v for k, v in c.items() if k in ("set", "fmt", "status", "reason")}, c.get("whole"), flush=True)

    if a.jobs == 0:
        for t in tasks:
            r = task(t)
            cells += r
            show(r)
    else:
        with cf.ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            for r in ex.map(task, tasks, chunksize=1):
                cells += r
                show(r)
    for c in cells:
        if c["status"] == "ok":
            c["verdict"] = {"better": "blanking better", "materially worse": "blanking worse", "neither": "neither"}[verdict(c)]
    pooled, excluded = {}, []
    for fmt in FORMATS:
        ok = []
        for c in cells:
            if c["fmt"] != fmt or c["status"] != "ok":
                continue
            if any(c[x]["hi"] - c[x]["lo"] <= 0 for x in ("whole", "A", "B")):
                excluded.append(dict(set=c["set"], fmt=fmt, reason="zero-width CI (no difference between the arms)"))
                continue
            ok.append(c)
        if len(ok) < 2:
            pooled[fmt] = dict(status="too few cells", k=len(ok))
            continue
        pw = {h: dl_pool(ok, h) for h in ("whole", "A", "B")}
        A, B = pw["A"], pw["B"]
        better = A["lo"] > 0 and A["gain"] >= MIN_GAIN and B["gain"] > 0 and B["gain"] - Z90_1S * B["se"] > 0
        worse = any(pw[h]["hi"] < 0 and pw[h]["gain"] <= -MIN_GAIN for h in pw)
        any_mw = any(c["verdict"] == "blanking worse" for c in cells if c["fmt"] == fmt and c["status"] == "ok")
        verd = "blanking better" if (better and not any_mw) else "blanking worse" if worse else "no detectable effect"
        pooled[fmt] = dict(status="ok", **pw, pooled_better=bool(better), pooled_worse=bool(worse), cell_materially_worse=any_mw, verdict=verd)
    res = dict(rule="docs/opp_blind/RULE.md", min_gain=MIN_GAIN, cells=cells, pooled=pooled, excluded=excluded)
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    f = lambda h: f"{h['gain'] * 1e4:+.2f} [{h['lo'] * 1e4:+.2f}, {h['hi'] * 1e4:+.2f}]"  # noqa: E731
    L = ["# Opponent colours blanked in the fit: pre-game hold-out test", "",
         "Rule: `docs/opp_blind/RULE.md`. Gains are log loss per game x 10,000 (positive = the opponent-blind arm B predicts later Sealed "
         "games better than the shipped arm S); MIN_GAIN = 1.0 in these units. Card values = per-copy card coefficients (Ctx.beta) of "
         "non-basic cards in the hold-out decks.", "",
         "| set | format | hold-out games | SD x S | SD x B | slope x (B on S) | corr x | cards | card slope | card corr | RMS(B-S)/SD(S) | whole [95% CI] | half A [95% CI] | half B [95% CI] | verdict |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["fmt"], c["set"])):
        if c["status"] == "ok":
            L.append(f"| {c['set']} | {c['fmt']} | {c['whole']['n_games']:,} | {c['sd_x_S']:.3f} | {c['sd_x_B']:.3f} | {_fmt_opt(c['slope_x'])} | "
                     f"{_fmt_opt(c['corr_x'], '.4f')} | {c['n_cards']} | {_fmt_opt(c['card_slope'])} | {_fmt_opt(c['card_corr'], '.4f')} | "
                     f"{_fmt_opt(c['card_rms_over_sd'])} | {f(c['whole'])} | {f(c['A'])} | {f(c['B'])} | {c['verdict']} |")
        else:
            L.append(f"| {c['set']} | {c['fmt']} | | | | | | | | | | {c['status']}: {c.get('reason', '')} | | | |")
    L += ["", "## Pooled (DerSimonian-Laird random effects)", "", "| format | cells | whole | half A | half B | I² (A) | pooled-better | pooled-worse | a cell materially worse | verdict |",
          "|---|---:|---|---|---|---:|---|---|---|---|"]
    for fmt, p in pooled.items():
        if p.get("status") != "ok":
            L.append(f"| {fmt} | {p.get('k', 0)} | {p['status']} | | | | | | | |")
            continue
        L.append(f"| {fmt} | {p['A']['k']} | {f(p['whole'])} | {f(p['A'])} | {f(p['B'])} | {p['A']['I2']:.0%} | {p['pooled_better']} | "
                 f"{p['pooled_worse']} | {p['cell_materially_worse']} | **{p['verdict']}** |")
    if excluded:
        L += ["", "Excluded cells: " + "; ".join(f"{e['set']} {e['fmt']} ({e['reason']})" for e in excluded)]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
