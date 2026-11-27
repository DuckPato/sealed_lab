"""Descriptive Draft value-vs-pick-position analysis ("drafted later / earlier than its value").

Ports the algorithm verified in `scripts/draft_sleepers.py`:
  1. Card values come from the set's model:
     - Exact plain Draft-stage fit on PremierDraft games (plus two independent draft-ID halves)
       when the set has PremierDraft game data in its corpus;
     - Otherwise `beta_bo1` / `se_bo1` / `exposure_bo1` from `model/cards.parquet` (including
       provisional pick-only sets, with even/odd bootstrap draws from `draws.npz` as halves
       when available).
  2. Pick position (ATA = mean 1-based pick number) and taken-when-seen (`takes / seen`) come
     from the set's registered PremierDraft pick file(s).
  3. Within each rarity group (`common`, `uncommon`, `rare+mythic`, with `mythic` also reported
     separately), a weighted decreasing isotonic regression (`pav_decreasing`) estimates the
     expected card value for a given pick position, with bootstrap curve uncertainty.
  4. A card is listed in the robust sleeper ("drafted later than its value") or overdrafted
     ("drafted earlier than its value") lists only when it meets the exposure & takes floors,
     and both independent halves agree in sign with at least `MIN_HALF_PP` (1.0 pp) gap.
  5. Early-vs-late ATA drift compares the first `EARLY_DAYS` (14) days of the format against
     later days for cards with at least `MIN_TAKES_PERIOD` (150) takes in both periods.

Purely descriptive: never changes card values, models, or Deck builder choices.
Cached next to the model at `sets/<SET>/model/draft_sleepers.json`, keyed by portable
pick-file fingerprints and the model's `built_at` stamp (no absolute paths).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import tempfile
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from sealed_lab import config
from sealed_lab.ingest import registry

log = logging.getLogger(__name__)

VERSION = 1
PP = 0.55 * 0.45 * 100.0  # 1 logit unit -> win-rate percentage points around 55% (24.75)
MIN_EXPO = 2000.0
MIN_TAKES = 500
MIN_HALF_PP = 1.0
EARLY_DAYS = 14
MIN_TAKES_PERIOD = 150
N_BOOT = 300
TOP_N = 10
DRIFT_TOP_N = 10
MIN_DRAFT_GAMES = 100

GROUPS = {
    "common": "common",
    "uncommon": "uncommon",
    "rare": "rare+mythic",
    "mythic": "rare+mythic",
}


# --------------------------------------------------------------------------- core math
def pav_decreasing(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted decreasing isotonic regression (PAV) evaluated at `x`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs, ys, ws = x[order], y[order], w[order]
    n = len(xs)
    if n == 0:
        return np.array([], dtype=float)
    # Increasing PAV on reversed order -> decreasing in x
    yr, wr = ys[::-1].copy(), ws[::-1].copy()
    blocks: list[list[float]] = []  # [start, end, w_sum, wy_sum]
    for i in range(n):
        blocks.append([i, i, float(wr[i]), float(wr[i] * yr[i])])
        while len(blocks) >= 2 and (blocks[-2][3] / blocks[-2][2]) > (blocks[-1][3] / blocks[-1][2]):
            b = blocks.pop()
            blocks[-1][1] = b[1]
            blocks[-1][2] += b[2]
            blocks[-1][3] += b[3]
    vals_r = np.empty(n, dtype=float)
    for s, e, ws_, wys in blocks:
        vals_r[int(s) : int(e) + 1] = wys / ws_
    vals = vals_r[::-1]
    out = np.empty(n, dtype=float)
    out[order] = vals
    return out


def _eval_isotonic(xs: np.ndarray, fitted: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Interpolate a monotone step function defined on sorted `xs` at `x_eval`."""
    n = len(xs)
    centres: list[float] = []
    levels: list[float] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and fitted[j + 1] == fitted[i]:
            j += 1
        centres.append(float(np.average(xs[i : j + 1])))
        levels.append(float(fitted[i]))
        i = j + 1
    return np.interp(x_eval, centres, levels)


def curve_fit_boot(
    x: np.ndarray,
    y: np.ndarray,
    se: np.ndarray,
    x_eval: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit weighted decreasing isotonic curve `(x, y)` and return `(expected, boot_sd)` at `x_eval`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    w = 1.0 / np.maximum(se, 1e-4) ** 2

    def fit_at(xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, xn: np.ndarray) -> np.ndarray:
        ord_ = np.argsort(xi, kind="mergesort")
        xs = xi[ord_]
        fv = pav_decreasing(xs, yi[ord_], wi[ord_])
        return _eval_isotonic(xs, fv, xn)

    exp = fit_at(x, y, w, x_eval)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty((n_boot, len(x_eval)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = fit_at(x[idx], y[idx], w[idx], x_eval)
    return exp, boots.std(axis=0)


def corr_pair(a: np.ndarray, b: np.ndarray) -> dict | None:
    """Return Pearson and Spearman correlations for finite pairs (minimum 4 points)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 4:
        return None
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return dict(n=int(len(a)), pearson=0.0, spearman=0.0)
    pr = float(np.corrcoef(a, b)[0, 1])
    sr = float(spearmanr(a, b).statistic)
    return dict(
        n=int(len(a)),
        pearson=pr if math.isfinite(pr) else 0.0,
        spearman=sr if math.isfinite(sr) else 0.0,
    )


def residual_table(
    df: pl.DataFrame,
    *,
    xcol: str = "ata",
    seed: int = 0,
    n_boot: int = N_BOOT,
) -> pl.DataFrame:
    """Attach isotonic `expected`, `exp_se`, `resid`, `gap_se`, `gap_ci_lo`, `gap_ci_hi`, `z` per rarity group."""
    parts: list[pl.DataFrame] = []
    for gi, (_, grp) in enumerate(df.group_by("rgroup", maintain_order=True)):
        el = grp.filter(pl.col("eligible"))
        if el.height < 8:
            exp = np.full(grp.height, np.nan, dtype=float)
            esd = np.full(grp.height, np.nan, dtype=float)
        else:
            exp, esd = curve_fit_boot(
                el[xcol].to_numpy(),
                el["value_pp"].to_numpy(),
                el["se_pp"].to_numpy(),
                grp[xcol].to_numpy(),
                n_boot=n_boot,
                seed=seed + 100 + gi,
            )
        val = grp["value_pp"].to_numpy()
        se_pp = grp["se_pp"].to_numpy()
        resid = val - exp
        gap_se = np.sqrt(se_pp ** 2 + esd ** 2)
        z = resid / np.where(gap_se > 0, gap_se, np.nan)
        parts.append(
            grp.with_columns(
                pl.Series("expected", exp),
                pl.Series("exp_se", esd),
                pl.Series("resid", resid),
                pl.Series("gap_se", gap_se),
                pl.Series("gap_ci_lo", resid - 1.96 * gap_se),
                pl.Series("gap_ci_hi", resid + 1.96 * gap_se),
                pl.Series("ci_lo", val - 1.96 * se_pp),
                pl.Series("ci_hi", val + 1.96 * se_pp),
                pl.Series("z", z),
            )
        )
    if not parts:
        return df
    return pl.concat(parts)


# --------------------------------------------------------------------------- pure analysis on tables
def _finite(x: object) -> float | None:
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _pack_card_row(r: dict) -> dict:
    out = dict(
        card_id=int(r["card_id"]),
        name=str(r["name"]),
        rarity=str(r["rarity"]),
        rgroup=str(r["rgroup"]),
        colors=str(r.get("colors") or ""),
        mana_value=_finite(r.get("mana_value")),
        ata=round(float(r["ata"]), 3),
        tws=round(float(r["tws"]), 4) if _finite(r.get("tws")) is not None else None,
        takes=int(r["takes"]),
        seen=int(r["seen"]),
        exposure=round(float(r["exposure"]), 1),
        value_pp=round(float(r["value_pp"]), 3),
        se_pp=round(float(r["se_pp"]), 3),
        ci_lo=round(float(r["ci_lo"]), 3) if _finite(r.get("ci_lo")) is not None else None,
        ci_hi=round(float(r["ci_hi"]), 3) if _finite(r.get("ci_hi")) is not None else None,
        expected=round(float(r["expected"]), 3) if _finite(r.get("expected")) is not None else None,
        exp_se=round(float(r["exp_se"]), 3) if _finite(r.get("exp_se")) is not None else None,
        resid=round(float(r["resid"]), 3) if _finite(r.get("resid")) is not None else None,
        gap_se=round(float(r["gap_se"]), 3) if _finite(r.get("gap_se")) is not None else None,
        gap_ci_lo=round(float(r["gap_ci_lo"]), 3) if _finite(r.get("gap_ci_lo")) is not None else None,
        gap_ci_hi=round(float(r["gap_ci_hi"]), 3) if _finite(r.get("gap_ci_hi")) is not None else None,
        z=round(float(r["z"]), 2) if _finite(r.get("z")) is not None else None,
    )
    r0 = _finite(r.get("r0"))
    r1 = _finite(r.get("r1"))
    out["r0"] = round(r0, 3) if r0 is not None else None
    out["r1"] = round(r1, 3) if r1 is not None else None
    return out


def _rarity_summary(sub: pl.DataFrame, *, has_halves: bool) -> dict:
    sub_f = sub.filter(pl.col("expected").is_not_nan() & pl.col("expected").is_not_null())
    half_corr = (
        corr_pair(sub_f["r0"].to_numpy(), sub_f["r1"].to_numpy())
        if (has_halves and sub_f.height >= 4)
        else None
    )
    resid_sd = (
        round(float(np.nanstd(sub_f["resid"].to_numpy())), 3)
        if sub_f.height >= 2
        else None
    )
    return dict(
        n_eligible=int(sub.height),
        corr_ata=corr_pair(sub["ata"].to_numpy(), sub["value_pp"].to_numpy()),
        corr_tws=corr_pair(sub["tws"].to_numpy(), sub["value_pp"].to_numpy()),
        resid_sd=resid_sd,
        half_resid_corr=half_corr,
        n_robust_sleepers=int(sub_f.filter(pl.col("robust") & (pl.col("resid") > 0)).height),
        n_robust_overdrafted=int(sub_f.filter(pl.col("robust") & (pl.col("resid") < 0)).height),
    )


def build_table(
    cards_df: pl.DataFrame,
    takes_df: pl.DataFrame,
    sees_df: pl.DataFrame,
    *,
    min_expo: float = MIN_EXPO,
    min_takes: int = MIN_TAKES,
    early_days: int = EARLY_DAYS,
    min_takes_period: int = MIN_TAKES_PERIOD,
) -> tuple[pl.DataFrame, str, str, str]:
    """Join card values (`cards_df`) with daily pick counts (`takes_df`, `sees_df`).

    `cards_df` must contain: `card_id`, `name`, `rarity`, `is_basic`, `value_pp`, `se_pp`, `exposure`
    (and optionally `front_name`, `colors`, `mana_value`, `v0_pp`, `v1_pp`).
    `takes_df`: `day` (YYYY-MM-DD str), `pick` (card name), `pick_number` (0-based int), `n` (int).
    `sees_df`: `day`, `name`, `seen` (int).
    """
    days = sorted(d for d in takes_df["day"].unique().to_list() if d)
    if not days:
        return pl.DataFrame(), "", "", ""
    first_day = days[0]
    last_day = days[-1]
    d0 = np.datetime64(first_day)
    early_end = str(d0 + np.timedelta64(int(early_days) - 1, "D"))

    takes = takes_df.with_columns(
        (pl.col("pick_number").cast(pl.Float64) + 1.0).alias("pos"),
        (pl.col("day") <= early_end).alias("is_early"),
    )

    tot_t = (
        takes.group_by("pick")
        .agg(
            pl.col("n").sum().alias("takes"),
            (pl.col("pos") * pl.col("n")).sum().alias("pos_sum"),
        )
        .with_columns((pl.col("pos_sum") / pl.col("takes")).alias("ata"))
        .rename({"pick": "name"})
    )
    tot_s = sees_df.group_by("name").agg(pl.col("seen").sum())
    per_t = (
        takes.group_by(["pick", "is_early"])
        .agg(
            pl.col("n").sum().alias("takes"),
            (pl.col("pos") * pl.col("n")).sum().alias("pos_sum"),
        )
        .with_columns((pl.col("pos_sum") / pl.col("takes")).alias("ata"))
        .rename({"pick": "name"})
    )
    ea = (
        per_t.filter(pl.col("is_early"))
        .select("name", pl.col("takes").alias("takes_early"), pl.col("ata").alias("ata_early"))
    )
    la = (
        per_t.filter(~pl.col("is_early"))
        .select("name", pl.col("takes").alias("takes_late"), pl.col("ata").alias("ata_late"))
    )

    picks_all = (
        tot_t.join(tot_s, on="name", how="left")
        .join(ea, on="name", how="left")
        .join(la, on="name", how="left")
    )
    by_name = {r["name"]: r for r in picks_all.iter_rows(named=True)}

    has_v0 = "v0_pp" in cards_df.columns and "v1_pp" in cards_df.columns
    rows: list[dict] = []
    for r in cards_df.iter_rows(named=True):
        if r.get("is_basic"):
            continue
        rar = str(r.get("rarity") or "").lower()
        rg = GROUPS.get(rar)
        if rg is None:
            continue
        p = by_name.get(r["name"])
        if p is None and r.get("front_name"):
            p = by_name.get(r["front_name"])
        if p is None or not p["takes"]:
            continue
        seen = int(p["seen"] or 0)
        takes_n = int(p["takes"])
        expo = float(r["exposure"])
        te = int(p["takes_early"] or 0)
        tl = int(p["takes_late"] or 0)
        v0 = _finite(r.get("v0_pp")) if has_v0 else None
        v1 = _finite(r.get("v1_pp")) if has_v0 else None
        rows.append(
            dict(
                card_id=int(r["card_id"]),
                name=str(r["name"]),
                rarity=rar,
                rgroup=rg,
                colors=str(r.get("colors") or ""),
                mana_value=_finite(r.get("mana_value")),
                value_pp=float(r["value_pp"]),
                se_pp=float(r["se_pp"]),
                exposure=expo,
                v0_pp=v0 if v0 is not None else float("nan"),
                v1_pp=v1 if v1 is not None else float("nan"),
                takes=takes_n,
                seen=seen,
                ata=float(p["ata"]),
                tws=float(takes_n) / seen if seen > 0 else float("nan"),
                takes_early=te,
                takes_late=tl,
                ata_early=float(p["ata_early"]) if p["ata_early"] is not None else float("nan"),
                ata_late=float(p["ata_late"]) if p["ata_late"] is not None else float("nan"),
                eligible=bool(expo >= float(min_expo) and takes_n >= int(min_takes)),
                eligible_early=bool(expo >= float(min_expo) and te >= int(min_takes_period)),
                eligible_late=bool(expo >= float(min_expo) and tl >= int(min_takes_period)),
            )
        )
    if not rows:
        return pl.DataFrame(), first_day, last_day, early_end
    return pl.DataFrame(rows), first_day, last_day, early_end


def analyze_table(
    df: pl.DataFrame,
    *,
    set_code: str,
    first_day: str,
    last_day: str,
    early_end: str,
    provisional: bool = False,
    value_source: str = "draft",
    halves_source: str = "draft_games",
    n_draft_games: int = 0,
    min_expo: float = MIN_EXPO,
    min_takes: int = MIN_TAKES,
    min_half_pp: float = MIN_HALF_PP,
    early_days: int = EARLY_DAYS,
    min_takes_period: int = MIN_TAKES_PERIOD,
    n_boot: int = N_BOOT,
    top_n: int = TOP_N,
    drift_top_n: int = DRIFT_TOP_N,
) -> dict:
    """Compute the complete descriptive sleeper / overdrafted / drift payload from a joined table."""
    if df.height == 0:
        return dict(
            version=VERSION,
            available=False,
            set_code=set_code.upper(),
            provisional=bool(provisional),
            value_source=value_source,
            halves_source=halves_source,
            reason=f"No matching cards were found between the model and PremierDraft picks for {set_code.upper()}.",
        )

    res = residual_table(df, xcol="ata", seed=1, n_boot=n_boot)
    has_halves = halves_source != "none" and not bool(res["v0_pp"].is_nan().all())
    if has_halves:
        res = res.with_columns(
            (pl.col("v0_pp") - pl.col("expected")).alias("r0"),
            (pl.col("v1_pp") - pl.col("expected")).alias("r1"),
        )
        robust_expr = (
            pl.col("eligible")
            & pl.col("expected").is_finite()
            & pl.col("resid").is_finite()
            & pl.col("r0").is_finite()
            & pl.col("r1").is_finite()
            & ((pl.col("r0") * pl.col("resid")) > 0)
            & ((pl.col("r1") * pl.col("resid")) > 0)
            & (pl.col("r0").abs() >= float(min_half_pp))
            & (pl.col("r1").abs() >= float(min_half_pp))
        )
    else:
        res = res.with_columns(
            pl.lit(float("nan")).alias("r0"),
            pl.lit(float("nan")).alias("r1"),
        )
        robust_expr = (
            pl.col("eligible")
            & pl.col("expected").is_finite()
            & pl.col("resid").is_finite()
            & pl.col("z").is_finite()
            & (pl.col("resid").abs() >= float(min_half_pp))
            & (pl.col("z").abs() >= 2.0)
        )
    res = res.with_columns(robust_expr.alias("robust"))

    el = res.filter(pl.col("eligible"))
    by_rarity: dict[str, dict] = {}
    for rg in ("common", "uncommon", "rare+mythic"):
        sub = el.filter(pl.col("rgroup") == rg)
        if sub.height > 0:
            by_rarity[rg] = _rarity_summary(sub, has_halves=has_halves)
    # Report mythic separately as well (using the rare+mythic curve's residuals)
    sub_m = el.filter(pl.col("rarity") == "mythic")
    if sub_m.height > 0:
        by_rarity["mythic"] = _rarity_summary(sub_m, has_halves=has_halves)

    rob_df = el.filter(pl.col("robust"))
    sleepers_robust = [
        _pack_card_row(r)
        for r in rob_df.filter(pl.col("resid") > 0).sort(["z", "resid"], descending=True).head(top_n).iter_rows(named=True)
    ]
    overdrafted_robust = [
        _pack_card_row(r)
        for r in rob_df.filter(pl.col("resid") < 0).sort(["z", "resid"], descending=False).head(top_n).iter_rows(named=True)
    ]
    el_valid = el.filter(pl.col("resid").is_finite() & pl.col("z").is_finite())
    sleepers_top_all = [
        _pack_card_row(r)
        for r in el_valid.sort(["z", "resid"], descending=True).head(top_n).iter_rows(named=True)
    ]
    overdrafted_top_all = [
        _pack_card_row(r)
        for r in el_valid.sort(["z", "resid"], descending=False).head(top_n).iter_rows(named=True)
    ]

    # Early-vs-late ATA drift (eligible cards with >= MIN_TAKES_PERIOD in both periods)
    both = df.filter(pl.col("eligible") & pl.col("eligible_early") & pl.col("eligible_late"))
    if both.height >= 4:
        el_e = residual_table(
            both.with_columns(pl.lit(True).alias("eligible")),
            xcol="ata_early",
            seed=5,
            n_boot=n_boot,
        )
        el_l = residual_table(
            both.with_columns(pl.lit(True).alias("eligible")),
            xcol="ata_late",
            seed=9,
            n_boot=n_boot,
        )
        drift = (
            el_e.select(
                "card_id",
                "name",
                "rarity",
                "rgroup",
                "colors",
                "value_pp",
                "se_pp",
                "ata_early",
                "takes_early",
                pl.col("resid").alias("resid_early"),
                pl.col("expected").alias("expected_early"),
            )
            .join(
                el_l.select(
                    "card_id",
                    "ata_late",
                    "takes_late",
                    pl.col("resid").alias("resid_late"),
                ),
                on="card_id",
                how="inner",
            )
            .with_columns(
                (pl.col("ata_late") - pl.col("ata_early")).alias("d_ata"),
                (pl.col("resid_late") - pl.col("resid_early")).alias("d_resid"),
            )
        )
        early_sleepers = drift.filter(pl.col("resid_early") >= float(min_half_pp))

        def _pack_drift(r: dict) -> dict:
            return dict(
                card_id=int(r["card_id"]),
                name=str(r["name"]),
                rarity=str(r["rarity"]),
                rgroup=str(r["rgroup"]),
                colors=str(r.get("colors") or ""),
                value_pp=round(float(r["value_pp"]), 3),
                ata_early=round(float(r["ata_early"]), 2),
                ata_late=round(float(r["ata_late"]), 2),
                d_ata=round(float(r["d_ata"]), 2),
                takes_early=int(r["takes_early"]),
                takes_late=int(r["takes_late"]),
                resid_early=round(float(r["resid_early"]), 2) if _finite(r.get("resid_early")) is not None else None,
                resid_late=round(float(r["resid_late"]), 2) if _finite(r.get("resid_late")) is not None else None,
            )

        early_late: dict = dict(
            available=True,
            first_day=first_day,
            last_day=last_day,
            early_end=early_end,
            early_days=int(early_days),
            min_takes_period=int(min_takes_period),
            n_both_periods=int(drift.height),
            corr_resid_early_vs_dATA=corr_pair(
                drift["resid_early"].to_numpy(), drift["d_ata"].to_numpy()
            ),
            n_early_sleepers=int(early_sleepers.height),
            early_sleepers_mean_dATA=(
                round(float(early_sleepers["d_ata"].mean()), 3) if early_sleepers.height else None
            ),
            early_sleepers_median_dATA=(
                round(float(early_sleepers["d_ata"].median()), 3) if early_sleepers.height else None
            ),
            early_sleepers_frac_moved_earlier=(
                round(float((early_sleepers["d_ata"] < 0).mean()), 3) if early_sleepers.height else None
            ),
            biggest_earlier=[
                _pack_drift(r)
                for r in drift.filter(pl.col("d_ata") < 0).sort("d_ata", descending=False).head(drift_top_n).iter_rows(named=True)
            ],
            biggest_later=[
                _pack_drift(r)
                for r in drift.filter(pl.col("d_ata") > 0).sort("d_ata", descending=True).head(drift_top_n).iter_rows(named=True)
            ],
            early_sleepers_list=[
                _pack_drift(r)
                for r in early_sleepers.sort("resid_early", descending=True).head(drift_top_n).iter_rows(named=True)
            ],
        )
    else:
        early_late = dict(
            available=False,
            first_day=first_day,
            last_day=last_day,
            early_end=early_end,
            early_days=int(early_days),
            min_takes_period=int(min_takes_period),
            n_both_periods=int(both.height),
            reason=(
                f"Not enough early and late pick data yet to compare drafting over time "
                f"(needs data beyond the first {early_days} days with at least {min_takes_period} takes per period)."
            ),
            biggest_earlier=[],
            biggest_later=[],
            early_sleepers_list=[],
        )

    return dict(
        version=VERSION,
        available=True,
        set_code=set_code.upper(),
        provisional=bool(provisional),
        value_source=value_source,
        halves_source=halves_source,
        n_eligible=int(el.height),
        n_picks=int(df["takes"].sum()),
        n_draft_games=int(n_draft_games),
        first_day=first_day,
        last_day=last_day,
        min_expo=float(min_expo),
        min_takes=int(min_takes),
        min_half_pp=float(min_half_pp),
        early_days=int(early_days),
        min_takes_period=int(min_takes_period),
        by_rarity=by_rarity,
        sleepers_robust=sleepers_robust,
        overdrafted_robust=overdrafted_robust,
        sleepers_top_all=sleepers_top_all,
        overdrafted_top_all=overdrafted_top_all,
        early_late=early_late,
        cached=False,
    )


# --------------------------------------------------------------------------- pick file streaming
def read_picks(paths: Path | Iterable[Path]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stream PremierDraft pick CSV(.gz) file(s) -> `(takes_df, sees_df)` by day."""
    plist = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    takes_parts: list[pl.DataFrame] = []
    sees_parts: list[pl.DataFrame] = []
    for path in plist:
        hdr = pl.read_csv(path, n_rows=0, infer_schema_length=0).columns
        pack_cols = [c for c in hdr if c.startswith("pack_card_")]
        if "pick" not in hdr or "pick_number" not in hdr or "draft_time" not in hdr:
            continue

        ov = {c: pl.Float32 for c in pack_cols}
        lf = (
            pl.scan_csv(path, infer_schema_length=0, schema_overrides=ov, low_memory=False)
            .select(["draft_time", "pick", "pick_number"] + pack_cols)
            .with_columns(
                pl.col("draft_time").str.slice(0, 10).alias("day"),
                pl.col("pick_number").cast(pl.Int32),
            )
        )
        takes_lf = lf.group_by(["day", "pick", "pick_number"]).agg(pl.len().cast(pl.Int64).alias("n"))
        if pack_cols:
            sees_lf = lf.group_by("day").agg(
                [(pl.col(c) > 0).sum().cast(pl.Int64).alias(c[len("pack_card_") :].replace('""', '"')) for c in pack_cols]
            )
            t_part, sees_wide = pl.collect_all([takes_lf, sees_lf])
            s_part = sees_wide.unpivot(index="day", variable_name="name", value_name="seen").filter(pl.col("seen") > 0)
            sees_parts.append(s_part)
        else:
            t_part = takes_lf.collect()
        takes_parts.append(t_part)

    if not takes_parts:
        empty_t = pl.DataFrame(schema={"day": pl.String, "pick": pl.String, "pick_number": pl.Int32, "n": pl.Int64})
        empty_s = pl.DataFrame(schema={"day": pl.String, "name": pl.String, "seen": pl.Int64})
        return empty_t, empty_s

    takes = (
        pl.concat(takes_parts)
        .group_by(["day", "pick", "pick_number"])
        .agg(pl.col("n").sum())
    )
    if sees_parts:
        sees = pl.concat(sees_parts).group_by(["day", "name"]).agg(pl.col("seen").sum())
    else:
        sees = pl.DataFrame(schema={"day": pl.String, "name": pl.String, "seen": pl.Int64})
    return takes, sees


# --------------------------------------------------------------------------- model / draft-stage value loaders
def _read_model_meta(set_code: str) -> dict:
    p = config.model_dir(set_code) / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_phi_readonly(inp, cache_dir: Path, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    """Read `phi_cache.npz` strictly read-only, computing any uncached decks in memory only."""
    from sealed_lab.model import pipeline, structure

    decks = inp.decks.sort("deck_key")
    keys = decks["deck_key"].to_numpy()
    hashes = decks["deck_hash"].to_numpy()
    sig = structure.cards_signature(inp.cards, 1, None)
    phi = np.full((len(keys), structure.N_TABLE), np.nan)
    cpath = pipeline.phi_cache_path(cache_dir, 1, None)
    if cpath.exists():
        with np.load(cpath, allow_pickle=False) as z:
            if str(z["sig"]) == sig and len(z["hashes"]):
                ch, cp = z["hashes"], z["phi"]
                order = np.argsort(ch)
                pos = np.minimum(np.searchsorted(ch[order], hashes), len(ch) - 1)
                hit = ch[order][pos] == hashes
                phi[hit] = cp[order][pos[hit]]
    todo = np.nonzero(np.isnan(phi[:, 0]))[0]
    if len(todo):
        offsets, cid, cnt = pipeline.deck_arrays(keys[todo], inp.deck_cards)
        phi[todo] = structure.compute_phi(offsets, cid, cnt, inp.cards, n_jobs=n_jobs, version=1, regime=None)
    return keys, phi


def _load_draft_stage_values(set_code: str, n_jobs: int) -> tuple[pl.DataFrame, int] | None:
    """Fit exact plain Draft-stage values (+ two draft-ID halves) from PremierDraft corpus games.

    Returns `(cards_df, n_draft_games)` or `None` if PremierDraft game data is unavailable.
    Never writes inside `APP_HOME` (uses a temporary directory for chain scratch files).
    """
    cdir = config.corpus_dir(set_code)
    if not (cdir / "games.parquet").exists() or not (cdir / "cards.parquet").exists():
        return None

    from sealed_lab.model import cardvalue, castability, design, pipeline

    inp = pipeline.load_inputs(set_code)
    g = inp.games.filter((pl.col("fmt") == "draft") & (pl.col("event") == "PremierDraft"))
    if g.height < MIN_DRAFT_GAMES:
        return None

    cards = castability.with_cost_groups(inp.cards, set_code)
    inp_w = dataclasses.replace(inp, cards=cards)
    rep_path = config.model_dir(set_code) / "report.json"
    ed_applied = False
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            ed_applied = bool((rep.get("extra_draft") or {}).get("applied", False))
        except (OSError, ValueError):
            ed_applied = False
    if pipeline.extra_draft_events(inp_w.games) and not ed_applied:
        inp_w = pipeline.without_extra_draft(inp_w)
    keys, phi_table = _read_phi_readonly(inp_w, config.cache_dir(set_code), n_jobs)
    space, _ = pipeline.plain_space(inp_w, keys, phi_table)

    half = np.fromiter(
        ((zlib.crc32(str(s).encode()) & 1) for s in g["draft_id"].to_list()),
        dtype=np.int8,
        count=g.height,
    )
    g_h = g.with_columns(pl.Series("_half", half))
    subs = {
        "all": g,
        "h0": g_h.filter(pl.col("_half") == 0).drop("_half"),
        "h1": g_h.filter(pl.col("_half") == 1).drop("_half"),
    }
    res: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with tempfile.TemporaryDirectory(prefix="sealed_lab_draft_insights_") as td:
        tmp = Path(td)
        for tag, sub in subs.items():
            st = design.build_stage("draft", sub, space)
            if st is None:
                return None
            tag_dir = tmp / tag
            tag_dir.mkdir(parents=True, exist_ok=True)
            out = cardvalue.run_chain(
                {"draft": st},
                space.n_card,
                space.n_feat,
                str(tag_dir),
                n_jobs=n_jobs,
                seed=7,
                prior_ok=set(),
                opts=cardvalue.ChainOpts(),
            )["draft"]
            cardvalue.cleanup(str(tag_dir))
            res[tag] = (
                np.asarray(out.beta_total, dtype=float) * PP,
                np.asarray(out.card_se, dtype=float) * PP,
                np.asarray(st.exposure, dtype=float),
            )

    val, se, expo = res["all"]
    v0, _, _ = res["h0"]
    v1, _, _ = res["h1"]
    nb_ids = [int(i) for i in space.nb_ids]
    nb_df = pl.DataFrame(
        dict(
            card_id=nb_ids,
            value_pp=val,
            se_pp=se,
            exposure=expo,
            v0_pp=v0,
            v1_pp=v1,
        )
    )
    c_sorted = cards.sort("card_id")
    cols = ["card_id", "name", "rarity", "is_basic"]
    for opt in ("front_name", "colors", "mana_value"):
        if opt in c_sorted.columns:
            cols.append(opt)
    df = c_sorted.select(cols).join(nb_df, on="card_id", how="inner")
    return df, int(g.height)


def _load_bo1_model_values(set_code: str) -> tuple[pl.DataFrame, str] | None:
    """Fallback loader using `model/cards.parquet` `beta_bo1` + optional `draws.npz` halves."""
    mdir = config.model_dir(set_code)
    cpath = mdir / "cards.parquet"
    if not cpath.exists():
        return None
    cards = pl.read_parquet(cpath).sort("card_id")
    if "beta_bo1" not in cards.columns or "se_bo1" not in cards.columns:
        return None
    val = cards["beta_bo1"].to_numpy().astype(float) * PP
    se = cards["se_bo1"].to_numpy().astype(float) * PP
    expo = (
        cards["exposure_bo1"].to_numpy().astype(float)
        if "exposure_bo1" in cards.columns
        else np.zeros(cards.height, dtype=float)
    )

    v0 = np.full(cards.height, np.nan, dtype=float)
    v1 = np.full(cards.height, np.nan, dtype=float)
    halves_source = "none"
    dpath = mdir / "draws.npz"
    if dpath.exists():
        try:
            with np.load(dpath, allow_pickle=False) as z:
                if "bo1__beta" in z:
                    arr = np.asarray(z["bo1__beta"], dtype=float)
                    if arr.ndim == 2 and arr.shape[0] >= 5 and arr.shape[1] == cards.height:
                        v0 = arr[1::2].mean(axis=0) * PP
                        v1 = arr[2::2].mean(axis=0) * PP
                        halves_source = "bootstrap_draws"
        except (OSError, ValueError):
            pass

    cols = ["card_id", "name", "rarity", "is_basic"]
    for opt in ("front_name", "colors", "mana_value"):
        if opt in cards.columns:
            cols.append(opt)
    df = cards.select(cols).with_columns(
        pl.Series("value_pp", val),
        pl.Series("se_pp", se),
        pl.Series("exposure", expo),
        pl.Series("v0_pp", v0),
        pl.Series("v1_pp", v1),
    )
    return df, halves_source


# --------------------------------------------------------------------------- caching & public entry point
def _registered_pick_files(set_code: str) -> list[tuple[str, str, Path]]:
    """Return sorted `(event, fingerprint, Path)` tuples for registered PremierDraft pick files that exist."""
    out: list[tuple[str, str, Path]] = []
    for f in registry.files(set_code.upper(), include_shared=False):
        if f.kind == "draft" and f.event == "PremierDraft":
            p = Path(f.path)
            if p.exists():
                out.append((str(f.event), str(f.fingerprint), p))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def cache_path(set_code: str, write_dir: Path | None = None) -> Path:
    base = Path(write_dir) if write_dir is not None else config.model_dir(set_code.upper())
    return base / "draft_sleepers.json"


def _read_cache(
    path: Path,
    *,
    set_code: str,
    model_built_at: str,
    pick_fingerprints: list[list[str]],
) -> dict | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("version") != VERSION:
        return None
    if str(d.get("set_code") or "").upper() != set_code.upper():
        return None
    if model_built_at and str(d.get("model_built_at") or "") != model_built_at:
        return None
    if pick_fingerprints:
        if d.get("pick_fingerprints") != pick_fingerprints:
            return None
    else:
        # Bundled / ready-made set with no local pick files: serve bundled cache only if it was available.
        if not d.get("available"):
            return None
    d["cached"] = True
    return d


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    out = dict(payload)
    out.pop("cached", None)
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def draft_sleepers(
    set_code: str,
    *,
    refresh: bool = False,
    read_only: bool = False,
    write_dir: Path | None = None,
    n_jobs: int | None = None,
    n_boot: int = N_BOOT,
) -> dict:
    """Compute or load the per-set Draft value-vs-pick-position ("sleepers / overdrafted") summary.

    - Serves cached `sets/<SET>/model/draft_sleepers.json` when the model `built_at` and
      PremierDraft pick-file fingerprints match (or for ready-made bundled sets with no local
      pick files).
    - Pass `write_dir` (e.g. `dest / "model"` in `tools/bundle_sets.py`) to write the cache to
      an alternate directory without touching `APP_HOME`.
    - Pass `read_only=True` to never write any cache file.
    """
    sc = set_code.upper()
    meta = _read_model_meta(sc)
    model_built_at = str(meta.get("built_at") or "")
    provisional = bool(meta.get("provisional"))
    pick_entries = _registered_pick_files(sc)
    pick_fps = [[ev, fp] for ev, fp, _ in pick_entries]

    cpath = cache_path(sc, write_dir=None)
    if not refresh:
        hit = _read_cache(cpath, set_code=sc, model_built_at=model_built_at, pick_fingerprints=pick_fps)
        if hit is not None:
            if write_dir is not None and not read_only and hit.get("available"):
                try:
                    _write_cache(cache_path(sc, write_dir=write_dir), hit)
                except OSError:
                    log.debug("failed to copy draft_sleepers cache to %s", write_dir, exc_info=True)
            return hit

    if not pick_entries:
        return dict(
            version=VERSION,
            available=False,
            set_code=sc,
            provisional=provisional,
            model_built_at=model_built_at,
            pick_fingerprints=[],
            reason=f"No PremierDraft pick file is available for {sc}.",
            cached=False,
        )

    jobs = int(n_jobs) if n_jobs is not None else max(1, min(8, os.cpu_count() or 1))
    cards_df: pl.DataFrame | None = None
    value_source = "draft"
    halves_source = "draft_games"
    n_draft_games = 0

    if not provisional:
        loaded = _load_draft_stage_values(sc, n_jobs=jobs)
        if loaded is not None:
            cards_df, n_draft_games = loaded

    if cards_df is None:
        bo1_loaded = _load_bo1_model_values(sc)
        if bo1_loaded is None:
            return dict(
                version=VERSION,
                available=False,
                set_code=sc,
                provisional=provisional,
                model_built_at=model_built_at,
                pick_fingerprints=pick_fps,
                reason=f"No card value model is available for {sc}.",
                cached=False,
            )
        cards_df, halves_source = bo1_loaded
        value_source = "provisional" if provisional else "bo1"

    takes_df, sees_df = read_picks([p for _, _, p in pick_entries])
    tbl, first_day, last_day, early_end = build_table(cards_df, takes_df, sees_df)
    payload = analyze_table(
        tbl,
        set_code=sc,
        first_day=first_day,
        last_day=last_day,
        early_end=early_end,
        provisional=provisional,
        value_source=value_source,
        halves_source=halves_source,
        n_draft_games=n_draft_games,
        n_boot=n_boot,
    )
    payload["model_built_at"] = model_built_at
    payload["pick_fingerprints"] = pick_fps

    if not read_only and payload.get("available"):
        target = cache_path(sc, write_dir=write_dir)
        try:
            _write_cache(target, payload)
        except OSError:
            log.debug("failed to write draft_sleepers cache at %s", target, exc_info=True)
    return payload
