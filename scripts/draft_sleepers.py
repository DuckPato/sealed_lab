"""Descriptive analysis: which cards win more (skill- and deck-adjusted) than their draft pick position suggests ("sleepers"), and which
win less ("overdrafted"). Nothing here changes the app or any model.

  value   the app's plain Draft stage (design.build_stage("draft") + cardvalue.run_chain, PremierDraft games only), card effect in logit,
          x24.75 = percentage points at a 55% deck. Also fitted on two halves of the events (crc32(draft_id) parity).
  ATA     mean 1-based pick number within the pack (the file's pick_number is 0-based) over the picks where the card was taken.
  expect  per set and rarity group (common, uncommon, rare+mythic) a decreasing isotonic curve of value on ATA (weights 1/SE^2);
          its sampling SD at a card comes from a card bootstrap. Sleeper score = (value - expected)/sqrt(SE^2 + SE_curve^2).
  robust  a card is listed only if its residual has the same sign and |residual| >= 1 pp in both halves, exposure >= 2000 deck-games
          (Draft stage) and takes >= 500.

Reads the app home and the raw files strictly read-only (SEALED_LAB_HOME must point at a scratch copy); writes only to OUT.

    python scripts/draft_sleepers.py OUT [--sets FRA HOB ...] [--jobs N] [--reuse]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import os
import time
import zlib
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from sealed_lab import config  # noqa: E402
from sealed_lab.model import cardvalue, design  # noqa: E402
import plat_draft_check as pdc  # noqa: E402  (prepare_set, check helpers)

DEFAULT_SETS = ["FRA", "HOB", "MSH", "SOS", "TMT"]
RAW = Path(os.environ.get("SEALED_LAB_RAW_DIR", str(Path.home() / "17LandsData")))   # raw 17Lands files, read only
PP = pdc.PP
MIN_EXPO, MIN_TAKES, MIN_HALF_PP = 2000.0, 500, 1.0
EARLY_DAYS = 14
MIN_TAKES_PERIOD = 150
N_BOOT = 300
GROUPS = {"common": "common", "uncommon": "uncommon", "rare": "rare+mythic", "mythic": "rare+mythic"}


# ------------------------------------------------------------------------------------------------------ raw files
def read_picks(sc: str) -> dict:
    """One pass over the PremierDraft pick file: takes per (day, card, pick_number) and, per day, the number of picks whose pack held
    each card (card names as in the pack_card_* headers)."""
    path = RAW / sc / f"draft_data_public.{sc}.PremierDraft.csv.gz"
    hdr = pl.read_csv(path, n_rows=0, infer_schema_length=0).columns
    pc = [c for c in hdr if c.startswith("pack_card_")]
    ov = {c: pl.Float32 for c in pc}
    lf = (pl.scan_csv(path, infer_schema_length=0, schema_overrides=ov, low_memory=False)
          .select(["draft_time", "pick_number", "pick"] + pc)
          .with_columns(pl.col("draft_time").str.slice(0, 10).alias("day"), pl.col("pick_number").cast(pl.Int32)))
    takes_lf = lf.group_by("day", "pick", "pick_number").len()
    sees_lf = lf.group_by("day").agg([(pl.col(c) > 0).sum().alias(c[len("pack_card_"):]) for c in pc] + [pl.len().alias("__rows")])
    takes, sees = pl.collect_all([takes_lf, sees_lf])
    return dict(takes=takes, sees=sees, pick_min=int(takes["pick_number"].min()), pick_max=int(takes["pick_number"].max()))


def read_gih(sc: str) -> pl.DataFrame:
    """17Lands-style GIH WR per card name from the PremierDraft game file (opening hand or drawn)."""
    path = RAW / sc / f"game_data_public.{sc}.PremierDraft.csv.gz"
    hdr = pl.read_csv(path, n_rows=0, infer_schema_length=0).columns
    oh = [c for c in hdr if c.startswith("opening_hand_")]
    dr = {c[len("drawn_"):]: c for c in hdr if c.startswith("drawn_")}
    names = [c[len("opening_hand_"):] for c in oh]
    ov = {c: pl.Int8 for c in oh + list(dr.values())}
    lf = pl.scan_csv(path, infer_schema_length=0, schema_overrides=ov, low_memory=False).select(["won"] + oh + list(dr.values()))
    lf = lf.with_columns((pl.col("won").str.to_lowercase() == "true").cast(pl.Int8).alias("w"))
    exprs = []
    for n in names:
        gih = ((pl.col("opening_hand_" + n) + (pl.col(dr[n]) if n in dr else 0)) > 0)
        exprs += [gih.sum().alias("n|" + n), (gih & (pl.col("w") == 1)).sum().alias("k|" + n)]
    r = lf.select(exprs).collect()
    rows = [dict(name=n, gih_n=int(r["n|" + n][0]), gih_k=int(r["k|" + n][0])) for n in names]
    return pl.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------ fits
def fit_values(st_games: pl.DataFrame, space, tmp: Path, n_jobs: int, seed: int) -> dict:
    st = design.build_stage("draft", st_games, space)
    tmp.mkdir(parents=True, exist_ok=True)
    fits = cardvalue.run_chain({"draft": st}, space.n_card, space.n_feat, str(tmp), n_jobs, seed=seed, prior_ok=set(),
                               opts=cardvalue.ChainOpts())
    f = fits["draft"]
    cardvalue.cleanup(str(tmp))
    return dict(beta=np.asarray(f.beta_total, float), se=np.asarray(f.card_se, float), exposure=np.asarray(st.exposure, float),
                n_games=int(st.n_games), n_events=int(st.n_groups), sigma=float(f.sigma), converged=bool(f.converged))


def collect_set(sc: str, out: Path, n_jobs: int) -> dict:
    t0 = time.time()
    inp, space, _formats, _ed = pdc.prepare_set(sc, n_jobs)
    g = inp.games.filter((pl.col("fmt") == "draft") & (pl.col("event") == "PremierDraft"))
    csort = inp.cards.sort("card_id")
    name_of = dict(zip(csort["card_id"].to_list(), csort["name"].to_list()))
    names = [name_of[int(i)] for i in space.nb_ids]
    half = (g["draft_id"].map_elements(lambda s: zlib.crc32(s.encode()) & 1, return_dtype=pl.Int8, skip_nulls=True)).to_numpy()
    tmp = out / "tmp" / sc
    res = {}
    for tag, sub in (("all", g), ("h0", g.filter(pl.Series(half == 0))), ("h1", g.filter(pl.Series(half == 1)))):
        tm = time.time()
        res[tag] = fit_values(sub, space, tmp / tag, n_jobs, 7)
        print(f"[{sc}] fit {tag}: games={res[tag]['n_games']} events={res[tag]['n_events']} sigma={res[tag]['sigma']:.3f} "
              f"conv={res[tag]['converged']} {time.time() - tm:.0f}s", flush=True)
    picks = read_picks(sc)
    print(f"[{sc}] picks read, pick_number {picks['pick_min']}..{picks['pick_max']} {time.time() - t0:.0f}s", flush=True)
    gih = read_gih(sc)
    print(f"[{sc}] gih read {time.time() - t0:.0f}s", flush=True)
    cards = inp.cards.select("card_id", "name", "front_name", "rarity", "colors", "types", "mana_value", "is_creature")
    mp = config.model_dir(sc) / "cards.parquet"
    sealed = pl.read_parquet(mp).select("name", "beta_bo1", "se_bo1") if mp.exists() else None
    return dict(set=sc, names=names, fits=res, picks=picks, gih=gih, cards=cards, sealed=sealed, seconds=time.time() - t0,
                n_pm_games=int(g.height))


# ------------------------------------------------------------------------------------------------------ pure analysis
def pav_decreasing(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted isotonic regression of y on x, non-increasing. Returns a function evaluating the step curve (linear between block
    centres, flat outside) at new x."""
    o = np.argsort(x, kind="stable")
    xs, ys, ws = x[o], y[o], w[o]
    # non-increasing in x == non-decreasing in -x; run PAV on reversed order
    ys_r, ws_r = ys[::-1], ws[::-1]
    vals, wts, cnt = [], [], []
    for v, wt in zip(ys_r, ws_r):
        vals.append(v); wts.append(wt); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
            wts[-2] = tw; cnt[-2] += cnt[-1]
            vals.pop(); wts.pop(); cnt.pop()
    fit_r = np.repeat(vals, cnt)
    fit = fit_r[::-1]                                  # non-increasing along ascending xs
    # block centres for interpolation
    centres, level = [], []
    i = 0
    while i < len(fit):
        j = i
        while j + 1 < len(fit) and fit[j + 1] == fit[i]:
            j += 1
        centres.append(float(np.average(xs[i:j + 1]))); level.append(float(fit[i]))
        i = j + 1
    centres, level = np.array(centres), np.array(level)

    def f(xn):
        return np.interp(np.asarray(xn, float), centres, level)
    return f


def curve_fit_boot(x, y, se, seed=0, n_boot=N_BOOT):
    """Isotonic curve (weights 1/SE^2) at the cards' own x, plus its bootstrap SD (resampling cards)."""
    w = 1.0 / np.maximum(se, 1e-4) ** 2
    f = pav_decreasing(x, y, w)
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = np.empty((n_boot, n))
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(x[ix])) < 3:
            draws[b] = f(x)
            continue
        draws[b] = pav_decreasing(x[ix], y[ix], w[ix])(x)
    return f, f(x), draws.std(axis=0)


def corr_pair(a, b):
    from scipy.stats import pearsonr, spearmanr
    if len(a) < 4:
        return dict(n=int(len(a)), pearson=None, spearman=None)
    return dict(n=int(len(a)), pearson=float(pearsonr(a, b)[0]), spearman=float(spearmanr(a, b)[0]))


def residual_table(df: pl.DataFrame, xcol="ata", seed=0) -> pl.DataFrame:
    """Add expected/exp_se/resid/z (pp) columns using the per-group isotonic curve fitted on eligible cards; ineligible cards get the
    curve value too (evaluated from the curve) but are flagged."""
    out = []
    for gi, (grp, sub) in enumerate(df.group_by("rgroup", maintain_order=True)):
        sub = sub.with_columns(pl.lit(None, dtype=pl.Float64).alias("expected"), pl.lit(None, dtype=pl.Float64).alias("exp_se"))
        el = sub.filter(pl.col("eligible"))
        if el.height < 8:
            out.append(sub.with_columns(pl.col("expected").fill_null(float("nan")), pl.col("exp_se").fill_null(float("nan"))))
            continue
        x, y, se = el[xcol].to_numpy(), el["value_pp"].to_numpy(), el["se_pp"].to_numpy()
        f, _, sd = curve_fit_boot(x, y, se, seed=seed + gi)
        # curve SD for every card of the group: bootstrap again at all cards' ata
        w = 1.0 / np.maximum(se, 1e-4) ** 2
        rng = np.random.default_rng(seed + 100 + gi)
        xa = sub[xcol].fill_null(float(np.nanmean(x))).to_numpy()
        draws = np.empty((N_BOOT, len(xa)))
        for b in range(N_BOOT):
            ix = rng.integers(0, len(x), len(x))
            draws[b] = pav_decreasing(x[ix], y[ix], w[ix])(xa) if len(np.unique(x[ix])) >= 3 else f(xa)
        out.append(sub.with_columns(pl.Series("expected", f(xa)), pl.Series("exp_se", draws.std(axis=0))))
    d = pl.concat(out)
    return d.with_columns((pl.col("value_pp") - pl.col("expected")).alias("resid"),
                          ((pl.col("value_pp") - pl.col("expected")) / (pl.col("se_pp") ** 2 + pl.col("exp_se") ** 2).sqrt()).alias("z"))


def build_table(raw: dict) -> tuple[pl.DataFrame, dict]:
    sc = raw["set"]
    names = raw["names"]
    a, h0, h1 = raw["fits"]["all"], raw["fits"]["h0"], raw["fits"]["h1"]
    v = pl.DataFrame(dict(name=names, value_pp=a["beta"] * PP, se_pp=a["se"] * PP, expo=a["exposure"],
                          v0_pp=h0["beta"] * PP, s0_pp=h0["se"] * PP, v1_pp=h1["beta"] * PP, s1_pp=h1["se"] * PP))
    cards = raw["cards"]
    # map pick names to card names (name, then front_name)
    lookup = {}
    for r in cards.iter_rows(named=True):
        lookup.setdefault(r["name"], r["name"])
    for r in cards.iter_rows(named=True):
        if r["front_name"]:
            lookup.setdefault(r["front_name"], r["name"])
    takes = raw["picks"]["takes"]
    takes = takes.with_columns(pl.col("pick").map_elements(lambda s: lookup.get(s), return_dtype=pl.Utf8).alias("cname"))
    unm = takes.filter(pl.col("cname").is_null())
    unmatched_rate = float(unm["len"].sum() / takes["len"].sum())
    takes = takes.filter(pl.col("cname").is_not_null()).with_columns((pl.col("pick_number") + 1).alias("pos"))
    days = sorted(takes["day"].unique().to_list())
    d0 = days[0]
    cut = (np.datetime64(d0) + np.timedelta64(EARLY_DAYS, "D")).astype(str)
    takes = takes.with_columns((pl.col("day") < cut).alias("early"))
    agg = takes.group_by("cname").agg(pl.col("len").sum().alias("takes"), (pl.col("pos") * pl.col("len")).sum().alias("psum"))
    agg = agg.with_columns((pl.col("psum") / pl.col("takes")).alias("ata")).select("cname", "takes", "ata")
    for tag, flt in (("early", pl.col("early")), ("late", ~pl.col("early"))):
        t = takes.filter(flt).group_by("cname").agg(pl.col("len").sum().alias(f"takes_{tag}"), (pl.col("pos") * pl.col("len")).sum().alias("ps"))
        t = t.with_columns((pl.col("ps") / pl.col(f"takes_{tag}")).alias(f"ata_{tag}")).select("cname", f"takes_{tag}", f"ata_{tag}")
        agg = agg.join(t, on="cname", how="left")
    # seen counts: map pack names to card names
    sees = raw["picks"]["sees"]
    scols = [c for c in sees.columns if c not in ("day", "__rows")]
    tot = {}
    for c in scols:
        nm = lookup.get(c)
        if nm is not None:
            tot[nm] = tot.get(nm, 0) + int(sees[c].sum())
    seen = pl.DataFrame(dict(cname=list(tot), seen=list(tot.values())))
    agg = agg.join(seen, on="cname", how="left").with_columns((pl.col("takes") / pl.col("seen")).alias("taken_when_seen"))
    df = v.join(agg, left_on="name", right_on="cname", how="left")
    df = df.join(cards.select("name", "rarity", "colors", "types", "mana_value"), on="name", how="left")
    df = df.join(raw["gih"].with_columns((100.0 * pl.col("gih_k") / pl.col("gih_n")).alias("gih_wr")).select("name", "gih_n", "gih_wr"),
                 on="name", how="left")
    if raw["sealed"] is not None:
        df = df.join(raw["sealed"].with_columns((pl.col("beta_bo1") * PP).alias("sealed_bo1_pp")).select("name", "sealed_bo1_pp"),
                     on="name", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("sealed_bo1_pp"))
    df = df.with_columns(pl.col("rarity").replace_strict(GROUPS, default=None).alias("rgroup"))
    df = df.filter(pl.col("rgroup").is_not_null() & pl.col("ata").is_not_null())
    df = df.with_columns(((pl.col("expo") >= MIN_EXPO) & (pl.col("takes") >= MIN_TAKES)).alias("eligible"))
    meta = dict(set=sc, early_cut=cut, first_day=d0, last_day=days[-1], unmatched_pick_rate=unmatched_rate,
                pick_number_base=("0-based (min %d, max %d); ATA uses pick_number+1" % (raw["picks"]["pick_min"], raw["picks"]["pick_max"])))
    return df, meta


def analyze(raw: dict) -> dict:
    sc = raw["set"]
    df, meta = build_table(raw)
    df = residual_table(df, "ata", seed=1)
    # halves' residual against the full-data curve
    df = df.with_columns((pl.col("v0_pp") - pl.col("expected")).alias("r0"), (pl.col("v1_pp") - pl.col("expected")).alias("r1"))
    df = df.with_columns(
        (pl.col("eligible") & (pl.col("resid") > 0) & (pl.col("r0") >= MIN_HALF_PP) & (pl.col("r1") >= MIN_HALF_PP)).alias("robust_sleeper"),
        (pl.col("eligible") & (pl.col("resid") < 0) & (pl.col("r0") <= -MIN_HALF_PP) & (pl.col("r1") <= -MIN_HALF_PP)).alias("robust_over"))
    el = df.filter(pl.col("eligible"))
    corr = {}
    for rg in ("common", "uncommon", "rare+mythic"):
        s = el.filter(pl.col("rgroup") == rg)
        corr[rg] = dict(eligible=corr_pair(s["value_pp"].to_numpy(), s["ata"].to_numpy()))
        s2 = df.filter(pl.col("rgroup") == rg)
        corr[rg]["all_cards"] = corr_pair(s2["value_pp"].to_numpy(), s2["ata"].to_numpy())
    for rg in ("mythic",):
        s = el.filter(pl.col("rarity") == rg)
        corr[rg] = dict(eligible=corr_pair(s["value_pp"].to_numpy(), s["ata"].to_numpy()))
    # early vs late
    elx = df.filter(pl.col("eligible") & (pl.col("takes_early") >= MIN_TAKES_PERIOD) & (pl.col("takes_late") >= MIN_TAKES_PERIOD))
    elx = elx.with_columns((pl.col("ata_late") - pl.col("ata_early")).alias("d_ata"))
    # residual at early ATA and at late ATA (curves fitted on that period's ATA vs the full-format value)
    parts = {}
    for tag in ("early", "late"):
        t = elx.with_columns(pl.col(f"ata_{tag}").alias("x"))
        t = t.select("name", "rgroup", "x", "value_pp", "se_pp", pl.lit(True).alias("eligible"))
        t = residual_table(t, "x", seed=5 if tag == "early" else 9)
        parts[tag] = t.select("name", pl.col("resid").alias(f"resid_{tag}"), pl.col("z").alias(f"z_{tag}"))
    elx = elx.join(parts["early"], on="name").join(parts["late"], on="name")
    ea = elx.filter(pl.col("z_early") >= 2.0)
    movers = dict(
        n_cards=int(elx.height),
        corr_resid_early_vs_dATA=corr_pair(elx["resid_early"].to_numpy(), elx["d_ata"].to_numpy()),
        mean_dATA_all=float(elx["d_ata"].mean()),
        early_sleepers=dict(n=int(ea.height), z_threshold=2.0, mean_resid_early=float(ea["resid_early"].mean()) if ea.height else None,
                            mean_resid_late=float(ea["resid_late"].mean()) if ea.height else None,
                            mean_dATA=float(ea["d_ata"].mean()) if ea.height else None),
        biggest_earlier=elx.sort("d_ata").head(10).select("name", "rarity", "ata_early", "ata_late", "d_ata", "value_pp", "resid_early", "resid_late").to_dicts(),
        biggest_later=elx.sort("d_ata", descending=True).head(10).select("name", "rarity", "ata_early", "ata_late", "d_ata", "value_pp", "resid_early", "resid_late").to_dicts(),
        early_sleepers_list=ea.sort("z_early", descending=True).head(10).select(
            "name", "rarity", "ata_early", "ata_late", "d_ata", "resid_early", "resid_late", "z_early").to_dicts(),
    )
    cols = ["name", "rarity", "colors", "types", "mana_value", "ata", "takes", "taken_when_seen", "value_pp", "se_pp", "expected", "exp_se", "resid", "z",
            "r0", "r1", "sealed_bo1_pp", "gih_wr", "expo"]

    def lst(f, desc):
        t = df.filter(f).sort("z", descending=desc).head(10)
        rows = []
        for r in t.select(cols).to_dicts():
            r["ci_lo"], r["ci_hi"] = r["value_pp"] - 1.96 * r["se_pp"], r["value_pp"] + 1.96 * r["se_pp"]
            rows.append(r)
        return rows
    sleepers = lst(pl.col("robust_sleeper"), True)
    over = lst(pl.col("robust_over"), False)
    nonrobust = dict(
        sleepers_top_z_not_robust=df.filter(pl.col("eligible") & (pl.col("z") > 2) & ~pl.col("robust_sleeper")).height,
        over_top_z_not_robust=df.filter(pl.col("eligible") & (pl.col("z") < -2) & ~pl.col("robust_over")).height)
    fit_info = {k: dict(n_games=v["n_games"], n_events=v["n_events"], sigma=v["sigma"], converged=v["converged"]) for k, v in raw["fits"].items()}
    return dict(set=sc, meta=meta, fit=fit_info, n_cards=int(df.height), n_eligible=int(el.height),
                n_robust_sleepers=int(df["robust_sleeper"].sum()), n_robust_over=int(df["robust_over"].sum()), resid_sd_eligible=float(el["resid"].std()),
                z_sd_eligible=float(el["z"].std()), corr_within_rarity=corr, sleepers=sleepers, overdrafted=over, early_late=movers,
                nonrobust=nonrobust, seconds=round(raw["seconds"], 1), table=df)


# ------------------------------------------------------------------------------------------------------ output
def fmt_row(r):
    return (f"| {r['name']} | {r['rarity'][0].upper()} | {r['colors'] or '-'} | {r['ata']:.2f} | {r['value_pp']:+.1f} "
            f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] | {r['expected']:+.1f} | {r['resid']:+.1f} (z {r['z']:+.1f}) | "
            f"{r['r0']:+.1f} / {r['r1']:+.1f} | {r['sealed_bo1_pp'] if r['sealed_bo1_pp'] is not None else float('nan'):+.1f} | {r['gih_wr']:.1f} |")


def write_md(res: dict, path: Path):
    L = ["# Draft sleepers and overdrafted cards (PremierDraft)", "",
         "Value = skill- and deck-adjusted Draft-stage card effect, pp at a 55% deck (logit x 24.75), fitted on PremierDraft games only. "
         "ATA = mean pick within the pack, 1-based (the files' pick_number is 0-based). Expected = decreasing isotonic curve of value on ATA, "
         "per set and rarity group (common / uncommon / rare+mythic), weights 1/SE^2. Residual = value - expected; z uses the card SE and the "
         "curve's bootstrap SD. Halves = events split by crc32(draft_id) parity; each half's residual is its own value minus the full-data "
         "expected value. Listed only if both halves agree in sign and are >= 1 pp, exposure >= 2000 deck-games, takes >= 500. "
         "Sealed bo1 = shipped model value (pp); GIH WR = 17Lands-style game-in-hand win rate (%), reference only.", ""]
    head = "| Card | R | Col | ATA | Value pp [95% CI] | Expected | Residual | Half residuals | Sealed bo1 | GIH WR |\n|---|---|---|---|---|---|---|---|---|---|"
    for sc, r in res.items():
        m = r["meta"]
        L += [f"## {sc}", "",
              f"{r['n_cards']} cards, {r['n_eligible']} eligible; robust sleepers {r['n_robust_sleepers']}, robust overdrafted {r['n_robust_over']}; "
              f"SD of residual among eligible cards {r['resid_sd_eligible']:.2f} pp (z SD {r['z_sd_eligible']:.2f}). Draft stage: "
              f"{r['fit']['all']['n_games']} games / {r['fit']['all']['n_events']} events. Picks: {m['pick_number_base']}; early = before {m['early_cut']} "
              f"(format start {m['first_day']}).", "", "Correlation of value with ATA within rarity group (eligible cards):", ""]
        for k, c in r["corr_within_rarity"].items():
            e = c["eligible"]
            L.append(f"- {k}: n={e['n']}, Pearson {e['pearson']:+.2f}, Spearman {e['spearman']:+.2f}" if e["pearson"] is not None else f"- {k}: n={e['n']}")
        L += ["", "### Top sleepers (win more than their pick position suggests)", "", head] + [fmt_row(x) for x in r["sleepers"]]
        L += ["", "### Top overdrafted (win less than their pick position suggests)", "", head] + [fmt_row(x) for x in r["overdrafted"]]
        el = r["early_late"]
        es = el["early_sleepers"]
        L += ["", "### Early vs late ATA", "",
              f"{el['n_cards']} cards with >= {MIN_TAKES_PERIOD} takes in both periods; mean ATA change late-early {el['mean_dATA_all']:+.2f}. "
              f"Correlation of the early residual with the ATA change: Pearson {el['corr_resid_early_vs_dATA']['pearson']:+.2f}, "
              f"Spearman {el['corr_resid_early_vs_dATA']['spearman']:+.2f} (negative = early sleepers got picked earlier later). "
              f"Early sleepers (early z >= 2): n={es['n']}, mean residual early {es['mean_resid_early']}, late {es['mean_resid_late']}, mean ATA change {es['mean_dATA']}.", "",
              "Biggest ATA improvements (picked earlier later in the format):", "", "| Card | R | ATA early | ATA late | Change | Value pp | Resid early | Resid late |\n|---|---|---|---|---|---|---|---|"]
        for x in el["biggest_earlier"]:
            L.append(f"| {x['name']} | {x['rarity'][0]} | {x['ata_early']:.2f} | {x['ata_late']:.2f} | {x['d_ata']:+.2f} | {x['value_pp']:+.1f} | {x['resid_early']:+.1f} | {x['resid_late']:+.1f} |")
        L += ["", "Early sleepers (z >= 2 at early ATA) and what happened:", "", "| Card | R | ATA early | ATA late | Change | Resid early | Resid late |\n|---|---|---|---|---|---|---|"]
        for x in el["early_sleepers_list"]:
            L.append(f"| {x['name']} | {x['rarity'][0]} | {x['ata_early']:.2f} | {x['ata_late']:.2f} | {x['d_ata']:+.2f} | {x['resid_early']:+.1f} | {x['resid_late']:+.1f} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="reuse the cached per-set raw results in OUT")
    a = ap.parse_args(argv)
    pdc.check_home_not_real()
    out = Path(a.out)
    pdc.check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    t0 = time.time()
    for sc in a.sets:
        cache = out / f"raw_{sc}.pkl"
        if a.reuse and cache.exists():
            raw = pickle.loads(cache.read_bytes())
        else:
            raw = collect_set(sc, out, a.jobs)
            cache.write_bytes(pickle.dumps(raw))
        r = analyze(raw)
        r["table"].write_csv(out / f"table_{sc}.csv")
        res[sc] = r
        print(f"[{sc}] corr {({k: round(v['eligible']['spearman'], 2) for k, v in r['corr_within_rarity'].items() if v['eligible']['spearman'] is not None})} "
              f"sleepers {r['n_robust_sleepers']} over {r['n_robust_over']} | {time.time() - t0:.0f}s", flush=True)
    slim = {sc: {k: v for k, v in r.items() if k != "table"} for sc, r in res.items()}
    (out / "sleepers.json").write_text(json.dumps(slim, indent=1, default=_jd, ensure_ascii=False), encoding="utf-8")
    write_md(res, out / "sleepers.md")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
