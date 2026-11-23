"""Card values against the Sealed field: hold-out prediction test (docs/sealed_field/RULE.md, candidate 2 of the Session 12 audit).

    python scripts/field_holdout.py DIR OUT [--sets ECL,EOE,...] [--jobs N] [--boot 2000]

DIR holds one folder per set written by `scripts/deviation_cost.py SET DIR/SET` (train-only model in DIR/SET/home, the cut in
DIR/SET/deviation_cost.json). Read-only on DIR. Writes OUT.json and OUT.md (OUT without extension).

Per set x format with a matchup layer in the train-only model:
    field shares   = opponents' colour strings of the format's TRAINING games (events before the cut), 40-card decks
    m_c            = sum_o share_o * centred matchup effect of card c against o (draw 0)
    x1 (offset)    = deck logit of the train-only model minus its reference logit
    x2             = sum over non-basic cards of n_c * m_c
    arm P          = logistic on the hold-out games: covariates (1, skill, skill^2, skill missing, on the play) + offset x1
    arm F          = the same with offset x1 + x2          (no coefficient on the field term is fitted)
    gain per game  = log loss P - log loss F; paired cluster bootstrap over events (whole hold-out, half A = earlier events, half B).
The covariates are fitted once per arm on the whole hold-out (the same covariates in both arms) and the halves are subsets of
those per-game losses. Decision and pooling exactly as the rule: MIN_GAIN = 1e-4, half-A significance confirmed on half B,
DerSimonian-Laird per format over the tested cells.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_GAIN = 1e-4
Z95 = 1.959964
Z90_1S = 1.281552
FORMATS = ("bo1", "bo3")


def logistic_offset(X: np.ndarray, y: np.ndarray, off: np.ndarray, iters: int = 50) -> np.ndarray:
    """Maximum-likelihood logistic coefficients with an offset (Newton); returns the per-game log loss."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = off + X @ b
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = off + X @ b
    return np.logaddexp(0.0, eta) - y * eta


def boot_mean(d: np.ndarray, ev: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean of per-game d with a 95% cluster bootstrap over events (percentile)."""
    u, inv = np.unique(ev, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        w = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u)))
        reps[i] = (w @ s) / (w @ n)
    return float(d.mean()), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def task(args) -> list[dict]:
    set_code, home, cut, n_boot = args
    os.environ["SEALED_LAB_HOME"] = home
    os.environ["POLARS_MAX_THREADS"] = "2"
    import polars as pl

    from sealed_lab import config
    from sealed_lab.deck import builder
    from sealed_lab.model import design, validate

    model = builder.load_model(set_code)
    cd = config.corpus_dir(set_code)
    g_all = pl.read_parquet(cd / "games.parquet")
    target = next(f for f in ("bo3", "bo1") if g_all.filter(pl.col("fmt") == f).height)
    t_cut = validate.time_split(g_all.filter(pl.col("fmt") == target))[1]
    if str(t_cut) != cut:
        return [dict(set=set_code, fmt=f, status="error", reason=f"cut mismatch {t_cut} vs {cut}") for f in FORMATS]
    decks = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards"])
    dcards = pl.read_parquet(cd / "deck_cards.parquet")
    out = []
    for fmt in FORMATS:
        rec = dict(set=set_code, fmt=fmt)
        if fmt not in model.formats:
            out.append(dict(rec, status="skip", reason="format not in the model"))
            continue
        if model.opp is None or fmt not in (model.opp_formats or []):
            out.append(dict(rec, status="skip", reason="no matchup layer in this format in the train-only model"))
            continue
        g = g_all.filter(pl.col("fmt") == fmt).join(decks, on="deck_key").filter(pl.col("n_cards") == 40)
        st = design.event_starts(g_all.filter(pl.col("fmt") == fmt))
        train_ids = st.filter(pl.col("start") < t_cut)["draft_id"].to_list()
        ho = st.filter(pl.col("start") >= t_cut)
        tr = g.filter(pl.col("draft_id").is_in(train_ids))
        te = g.filter(pl.col("draft_id").is_in(ho["draft_id"].to_list()))
        if tr.height < 200 or te.height < 500:
            out.append(dict(rec, status="skip", reason=f"too few games (train {tr.height}, hold-out {te.height})"))
            continue
        c = tr.group_by("opp_colors").len()
        opps, share = c["opp_colors"].to_list(), (c["len"] / c["len"].sum()).to_numpy()
        opm = model.opp
        m_field = share @ np.stack([opm.effect(o, draws=False) for o in opps])
        ctx = builder.Ctx(model, fmt)
        keys = te["deck_key"].unique().to_list()
        x1, x2 = {}, {}
        for k, v in dcards.filter(pl.col("deck_key").is_in(keys)).group_by("deck_key"):
            ids = np.array(v["card_id"].to_list(), dtype=np.int64)
            cnt = np.array(v["n"].to_list(), dtype=float)
            x1[k[0]] = ctx.logit(dict(zip(ids.tolist(), [int(n) for n in cnt]))) - ctx.ref
            nb = ~ctx.is_basic[ids]
            x2[k[0]] = float(m_field[ids[nb]] @ cnt[nb])
        te = te.filter(pl.col("deck_key").is_in(list(x1)))
        dk = te["deck_key"].to_list()
        a1 = np.array([x1[k] for k in dk])
        a2 = np.array([x2[k] for k in dk])
        sk = te["skill"].to_numpy().astype(float)
        miss = np.isnan(sk)
        sk = np.where(miss, np.nanmean(sk) if (~miss).any() else 0.0, sk)
        y = te["won"].to_numpy().astype(float)
        X = np.column_stack([np.ones(len(y)), sk, sk ** 2, miss.astype(float), te["on_play"].fill_null(False).to_numpy().astype(float)])
        d = logistic_offset(X, y, a1) - logistic_offset(X, y, a1 + a2)
        ev = np.array([str(x) for x in te["draft_id"].to_list()])
        order = [str(x) for x in ho["draft_id"].to_list()]          # sorted by start
        half_a = set(order[: len(order) // 2])
        in_a = np.array([e in half_a for e in ev])
        res = {}
        for name, m in (("whole", np.ones(len(d), bool)), ("A", in_a), ("B", ~in_a)):
            gain, lo, hi = boot_mean(d[m], ev[m], n_boot, seed=17)
            res[name] = dict(gain=gain, lo=lo, hi=hi, n_games=int(m.sum()), n_events=int(len(set(ev[m]))))
        out.append(dict(rec, status="ok", n_train_games=tr.height, n_opp_strings=len(opps), sd_x2_pp=float(a2.std() * 100 * 0.2475),
                        sd_x1=float(a1.std()), corr_x1_x2=float(np.corrcoef(a1, a2)[0, 1]) if a2.std() > 0 else None, **res))
    return out


def verdict(c: dict) -> str:
    A, B = c["A"], c["B"]
    se_b = (B["hi"] - B["lo"]) / (2 * Z95)
    if any(c[x]["hi"] < 0 and c[x]["gain"] <= -MIN_GAIN for x in ("whole", "A", "B")):
        return "materially worse"
    if A["lo"] > 0 and A["gain"] >= MIN_GAIN and B["gain"] > 0 and B["gain"] - Z90_1S * se_b > 0:
        return "better"
    return "neither"


def dl_pool(cells: list[dict], key: str) -> dict:
    est = np.array([c[key]["gain"] for c in cells])
    se = np.array([(c[key]["hi"] - c[key]["lo"]) / (2 * Z95) for c in cells])
    w = 1 / se ** 2
    fe = float((w * est).sum() / w.sum())
    q = float((w * (est - fe) ** 2).sum())
    df = len(est) - 1
    tau2 = max(0.0, (q - df) / (w.sum() - (w ** 2).sum() / w.sum())) if df > 0 else 0.0
    ws = 1 / (se ** 2 + tau2)
    re = float((ws * est).sum() / ws.sum())
    se_re = float(1 / math.sqrt(ws.sum()))
    return dict(gain=re, se=se_re, lo=re - Z95 * se_re, hi=re + Z95 * se_re, fixed=fe, Q=q, I2=max(0.0, (q - df) / q) if q > 0 else 0.0,
                tau2=tau2, k=len(est))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--sets", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    root = Path(a.dir).resolve()
    sets = a.sets.split(",") if a.sets else sorted(p.name for p in root.iterdir() if (p / "deviation_cost.json").exists())
    tasks = [(s, str(root / s / "home"), json.loads((root / s / "deviation_cost.json").read_text(encoding="utf-8"))["cut"], a.boot) for s in sets]
    cells = []
    with cf.ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
        for r in ex.map(task, tasks, chunksize=1):
            cells += r
            for c in r:
                print({k: v for k, v in c.items() if k in ("set", "fmt", "status", "reason")}, c.get("whole"), flush=True)
    for c in cells:
        if c["status"] == "ok":
            c["verdict"] = verdict(c)
    pooled, excluded = {}, []
    for fmt in FORMATS:
        ok = []
        for c in cells:
            if c["fmt"] != fmt or c["status"] != "ok":
                continue
            if any(c[x]["hi"] - c[x]["lo"] <= 0 for x in ("whole", "A", "B")):
                excluded.append(dict(set=c["set"], fmt=fmt, reason="zero-width CI (no field-term spread)"))
                continue
            ok.append(c)
        if len(ok) < 2:
            pooled[fmt] = dict(status="too few cells", k=len(ok))
            continue
        pw = {h: dl_pool(ok, h) for h in ("whole", "A", "B")}
        A, B = pw["A"], pw["B"]
        better = A["lo"] > 0 and A["gain"] >= MIN_GAIN and B["gain"] > 0 and B["gain"] - Z90_1S * B["se"] > 0
        worse = any(pw[h]["hi"] < 0 and pw[h]["gain"] <= -MIN_GAIN for h in pw)
        any_mw = any(c["verdict"] == "materially worse" for c in cells if c["fmt"] == fmt and c["status"] == "ok")
        pooled[fmt] = dict(status="ok", **pw, pooled_better=better, pooled_worse=worse, cell_materially_worse=any_mw,
                           qualifies=bool(better and not any_mw))
    res = dict(rule="docs/sealed_field/RULE.md", min_gain=MIN_GAIN, cells=cells, pooled=pooled, excluded=excluded)
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# Card values against the Sealed field: hold-out test", "", "Rule: `docs/sealed_field/RULE.md`. Gains are log loss per game x 10,000 "
         "(positive = the field term predicts later Sealed games better); MIN_GAIN = 1.0 in these units.", "",
         "| set | format | hold-out games | field term SD (pp) | whole [95% CI] | half A [95% CI] | half B [95% CI] | verdict |",
         "|---|---|---:|---:|---|---|---|---|"]
    f = lambda h: f"{h['gain'] * 1e4:+.2f} [{h['lo'] * 1e4:+.2f}, {h['hi'] * 1e4:+.2f}]"  # noqa: E731
    for c in sorted(cells, key=lambda c: (c["fmt"], c["set"])):
        if c["status"] == "ok":
            L.append(f"| {c['set']} | {c['fmt']} | {c['whole']['n_games']:,} | {c['sd_x2_pp']:.2f} | {f(c['whole'])} | {f(c['A'])} | {f(c['B'])} | {c['verdict']} |")
        else:
            L.append(f"| {c['set']} | {c['fmt']} | | | {c['status']}: {c.get('reason', '')} | | | |")
    L += ["", "## Pooled (DerSimonian-Laird random effects)", "", "| format | cells | whole | half A | half B | I² (A) | pooled-better | a cell materially worse | qualifies |",
          "|---|---:|---|---|---|---:|---|---|---|"]
    for fmt, p in pooled.items():
        if p.get("status") != "ok":
            L.append(f"| {fmt} | {p.get('k', 0)} | {p['status']} | | | | | | |")
            continue
        L.append(f"| {fmt} | {p['A']['k']} | {f(p['whole'])} | {f(p['A'])} | {f(p['B'])} | {p['A']['I2']:.0%} | {p['pooled_better']} | {p['cell_materially_worse']} | **{p['qualifies']}** |")
    if excluded:
        L += ["", "Excluded cells: " + "; ".join(f"{e['set']} {e['fmt']} ({e['reason']})" for e in excluded)]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
