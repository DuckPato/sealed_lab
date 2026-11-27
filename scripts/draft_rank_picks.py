"""Descriptive: how do Diamond+ drafters pick differently from the rest (17Lands PremierDraft pick files)?

Pass 1 (per set, in parallel, streamed in row blocks): for every pick, every card in the pack is a candidate ("seen");
the outcome is whether it was the pick. Counts are accumulated per card x rank group x skill group x stratum x half, where
stratum = calendar week x pack x pick-number band, and half = parity of the draft's first-appearance index (used to estimate
how much the naive, pick-level CIs understate the uncertainty because the same drafter sees the same card several times).
Also per-draft colour counts (P1p1-p5 picks, main-deck picks) and week-wise ATA sums.

Pass 2 (cheap, from the saved accumulators): Mantel-Haenszel odds ratios (Robins-Breslow-Greenland variance), per-set tables,
colour analysis, card-type analysis, correlation with the model-value residual, rank vs skill.

Reads raw pick files and a scratch copy of the app home (SEALED_LAB_HOME) read-only; writes only to OUT.

    python scripts/draft_rank_picks.py OUT [--sets FRA HOB MSH SOS TMT] [--jobs N] [--reuse] [--max-rows N]

Env: PICKS_DIR (folder holding <SET>/draft_data_public.<SET>.PremierDraft.csv.gz), SEALED_LAB_HOME.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

SETS = ["FRA", "HOB", "MSH", "SOS", "TMT"]
DEF_PICKS = os.environ.get("SEALED_LAB_RAW_DIR", str(Path.home() / "17LandsData"))   # raw 17Lands files, read only
RANK_GRP = {"bronze": 0, "silver": 0, "gold": 0, "platinum": 1, "diamond": 2, "mythic": 2}
RG_NAMES = ["low", "plat", "D+"]
SG_NAMES = ["<0.54", "0.54-0.60", ">=0.60", "unknown"]
W = 70                      # weeks from 2026-01-01
DAY0 = 20454                # 2026-01-01 in days since epoch
BANDS = np.array([0, 1, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6])   # pick_number (0-based) -> band
NB = 7
S = W * 3 * NB
COLORS = "WUBRG"
MIN_SEEN = 300


def home() -> Path:
    return Path(os.environ["SEALED_LAB_HOME"])


# ----------------------------------------------------------------------------------------------------------- pass 1
def card_table(sc: str):
    c = pl.read_parquet(home() / "sets" / sc / "model" / "cards.parquet")
    return c


def header(path: Path) -> list[str]:
    import gzip
    import csv
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def accumulate(sc: str, picks_dir: str, out: Path, max_rows: int | None) -> dict:
    t0 = time.time()
    path = Path(picks_dir) / sc / f"draft_data_public.{sc}.PremierDraft.csv.gz"
    hdr = header(path)
    pcols = [h for h in hdr if h.startswith("pack_card_")]
    names = [h[len("pack_card_"):] for h in pcols]
    C = len(names)
    cards = card_table(sc)
    by_name = {}
    for r in cards.iter_rows(named=True):
        by_name.setdefault(r["name"], r)
        by_name.setdefault(r["front_name"], r)
    colmask = np.zeros((C, 5), dtype=np.int8)       # card colours
    colorless = np.zeros(C, dtype=bool)
    matched = np.zeros(C, dtype=bool)
    for i, n in enumerate(names):
        r = by_name.get(n)
        if r is None:
            continue
        matched[i] = True
        for j, ch in enumerate(COLORS):
            colmask[i, j] = ch in (r["colors"] or "")
        colorless[i] = not (r["colors"] or "")
    meta = ["draft_id", "draft_time", "rank", "pack_number", "pick_number", "pick", "pick_maindeck_rate",
            "user_game_win_rate_bucket"]
    csv_path = Path(os.environ.get("CSV_CACHE", "")) / f"{sc}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path}: decompress the .csv.gz into CSV_CACHE first (polars is ~100x faster than pyarrow on this machine)")
    so = {c: pl.Float32 for c in pcols}
    so.update({"pack_number": pl.Float64, "pick_number": pl.Float64, "pick_maindeck_rate": pl.Float64, "user_game_win_rate_bucket": pl.Float64,
               "draft_id": pl.String, "draft_time": pl.String, "rank": pl.String, "pick": pl.String})
    import io

    def blocks(path, size=40 << 20):
        with open(path, "rb") as fh:
            head = fh.readline()
            rest = b""
            while True:
                buf = fh.read(size)
                if not buf:
                    if rest.strip():
                        yield pl.read_csv(io.BytesIO(head + rest), columns=meta + pcols, schema_overrides=so)
                    return
                buf = rest + buf
                cut = buf.rfind(b"\n")
                if cut < 0:
                    rest = buf
                    continue
                rest = buf[cut + 1:]
                yield pl.read_csv(io.BytesIO(head + buf[:cut + 1]), columns=meta + pcols, schema_overrides=so)
    rdr = blocks(csv_path)
    name_idx = {n: i for i, n in enumerate(names)}

    seen = np.zeros((C, 3, 4, S, 2), dtype=np.int64)
    taken = np.zeros((C, 3, 4, S, 2), dtype=np.int64)
    pos_sum = np.zeros((C, 3, W), dtype=np.float64)
    pos_cnt = np.zeros((C, 3, W), dtype=np.int64)
    # per-draft arrays (grown)
    cap = 1 << 16
    d_rank = np.full(cap, -1, np.int8)
    d_skill = np.full(cap, 3, np.int8)
    d_week = np.full(cap, -1, np.int16)
    d_deck = np.zeros((cap, 5), np.int32)
    d_ndeck = np.zeros(cap, np.int32)
    d_early = np.zeros((cap, 6), np.int32)           # P1p1-p5 picks: colour counts + colourless
    d_nearly = np.zeros(cap, np.int32)
    id2code: dict[str, int] = {}
    n_rows = n_nopick = n_norank = 0
    Npick = 14
    for b in rdr:
        if max_rows and n_rows >= max_rows:
            break
        nb = b.height
        n_rows += nb
        rg = b["rank"].replace_strict(RANK_GRP, default=-1, return_dtype=pl.Int8).to_numpy().astype(np.int8)
        sk = b["user_game_win_rate_bucket"].fill_null(np.nan).to_numpy().astype(np.float64)
        sg = np.where(np.isnan(sk), 3, np.where(sk >= 0.595, 2, np.where(sk >= 0.535, 1, 0))).astype(np.int8)
        day = b["draft_time"].str.slice(0, 10).str.to_date("%Y-%m-%d").cast(pl.Int32).to_numpy().astype(np.int64)
        wk = (day - DAY0) // 7
        assert wk.min() >= 0 and wk.max() < W, (wk.min(), wk.max())
        pk = b["pick_number"].to_numpy().astype(np.int64)
        pack = b["pack_number"].to_numpy().astype(np.int64)
        mdr = b["pick_maindeck_rate"].fill_null(0.0).to_numpy()
        pickidx = b["pick"].replace_strict(name_idx, default=-1, return_dtype=pl.Int32).to_numpy().astype(np.int64)
        # draft codes
        ids = b["draft_id"]
        u = ids.unique()
        dic = u.to_list()
        loc = np.empty(len(dic), np.int64)
        for i, d in enumerate(dic):
            c = id2code.get(d)
            if c is None:
                c = len(id2code)
                id2code[d] = c
            loc[i] = c
        code = pl.Series(ids).replace_strict(dict(zip(dic, loc.tolist())), return_dtype=pl.Int64).to_numpy()
        mxc = code.max() + 1
        if mxc > cap:
            ncap = max(cap * 2, mxc + 1)
            def grow(a, fill):
                n = np.full((ncap,) + a.shape[1:], fill, a.dtype)
                n[:cap] = a
                return n
            d_rank = grow(d_rank, -1); d_skill = grow(d_skill, 3); d_week = grow(d_week, -1)
            d_deck = grow(d_deck, 0); d_ndeck = grow(d_ndeck, 0); d_early = grow(d_early, 0); d_nearly = grow(d_nearly, 0)
            cap = ncap
        d_rank[code] = rg; d_skill[code] = sg; d_week[code] = wk
        half = (code & 1).astype(np.int64)
        okr = rg >= 0
        n_norank += int((~okr).sum())
        n_nopick += int((okr & (pickidx < 0)).sum())
        stratum = (wk * 3 + pack) * NB + BANDS[np.minimum(pk, len(BANDS) - 1)]
        gidx = rg.astype(np.int64) * 4 + sg
        # taken-when-seen
        X = b.select(pcols).to_numpy()
        X[~okr] = 0
        r, c = np.nonzero(X > 0)
        del X
        base = ((c.astype(np.int64) * 12 + gidx[r]) * S + stratum[r]) * 2 + half[r]
        tk = c == pickidx[r]
        seen += np.bincount(base, minlength=seen.size).reshape(seen.shape)
        taken += np.bincount(base[tk], minlength=seen.size).reshape(seen.shape)
        # ATA by week
        m = okr & (pickidx >= 0)
        pos = (pack * Npick + pk + 1).astype(np.float64)
        li = (pickidx[m] * 3 + rg[m].astype(np.int64)) * W + wk[m]
        pos_sum += np.bincount(li, weights=pos[m], minlength=pos_sum.size).reshape(pos_sum.shape)
        pos_cnt += np.bincount(li, minlength=pos_sum.size).reshape(pos_sum.shape)
        # per-draft colours
        pm = m
        e = pm & (pack == 0) & (pk < 5)
        if e.any():
            ci = pickidx[e]
            for j in range(5):
                np.add.at(d_early[:, j], code[e], colmask[ci, j])
            np.add.at(d_early[:, 5], code[e], colorless[ci])
            np.add.at(d_nearly, code[e], 1)
        dm = pm & (mdr >= 0.5)
        if dm.any():
            ci = pickidx[dm]
            for j in range(5):
                np.add.at(d_deck[:, j], code[dm], colmask[ci, j])
            np.add.at(d_ndeck, code[dm], 1)
        if n_rows % 2_000_000 < nb:
            print(f"[{sc}] {n_rows:,} rows {time.time()-t0:.0f}s", flush=True)
    nd = len(id2code)
    np.savez_compressed(
        out / f"acc_{sc}.npz", seen=seen, taken=taken, pos_sum=pos_sum, pos_cnt=pos_cnt,
        d_rank=d_rank[:nd], d_skill=d_skill[:nd], d_week=d_week[:nd], d_deck=d_deck[:nd], d_ndeck=d_ndeck[:nd],
        d_early=d_early[:nd], d_nearly=d_nearly[:nd], names=np.array(names), matched=matched)
    info = dict(set=sc, rows=n_rows, drafts=nd, cards=C, unmatched_cards=[n for n, m_ in zip(names, matched) if not m_],
                rows_no_rank=n_norank, picks_not_in_columns=n_nopick, secs=time.time() - t0)
    (out / f"info_{sc}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[{sc}] done {n_rows:,} rows, {nd:,} drafts, {time.time()-t0:.0f}s", flush=True)
    return info


# ----------------------------------------------------------------------------------------------------------- pass 2
def mh(a1, n1, a0, n0):
    """Mantel-Haenszel OR of group A (a1 taken of n1 seen) vs B over strata (last axis). Returns (log OR, se, n1_used, n0_used)."""
    n1 = n1.astype(np.float64); n0 = n0.astype(np.float64); a1 = a1.astype(np.float64); a0 = a0.astype(np.float64)
    N = n1 + n0
    use = (n1 > 0) & (n0 > 0)
    Nn = np.where(use, N, 1.0)
    a, b, c, d = a1, n1 - a1, a0, n0 - a0
    P = (a + d) / Nn; Q = (b + c) / Nn
    Rr = np.where(use, a * d / Nn, 0.0); Ss = np.where(use, b * c / Nn, 0.0)
    R = Rr.sum(-1); Sx = Ss.sum(-1)
    with np.errstate(all="ignore"):
        lor = np.log(R / Sx)
        var = ((P * Rr).sum(-1) / (2 * R ** 2) + ((P * Ss + Q * Rr).sum(-1)) / (2 * R * Sx) + (Q * Ss).sum(-1) / (2 * Sx ** 2))
    return lor, np.sqrt(var), (n1 * use).sum(-1), (n0 * use).sum(-1)


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 3 else float("nan")


def spearman(x, y):
    from scipy.stats import rankdata
    return pearson(rankdata(x), rankdata(y))


def boot_corr(x, y, B=2000, seed=0, fn=pearson):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 8:
        return [float("nan")] * 2
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v.append(fn(x[i], y[i]))
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def dl_mean(y, se, extra_var=0.0):
    """DerSimonian-Laird random-effects mean and CI."""
    y = np.asarray(y, float); v = np.asarray(se, float) ** 2 + extra_var
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    if len(y) < 3:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), tau2=float("nan"), n=int(len(y)))
    w = 1 / v
    fe = (w * y).sum() / w.sum()
    Q = (w * (y - fe) ** 2).sum()
    tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    w2 = 1 / (v + tau2)
    m = (w2 * y).sum() / w2.sum(); se_m = math.sqrt(1 / w2.sum())
    return dict(mean=float(m), lo=float(m - 1.96 * se_m), hi=float(m + 1.96 * se_m), tau2=float(tau2), n=int(len(y)))


def cat_flags(cards: pl.DataFrame, names: list[str]) -> dict[str, np.ndarray]:
    """Rough card-type flags aligned with the pick-file column order."""
    idx = {}
    for i, r in enumerate(cards.iter_rows(named=True)):
        idx.setdefault(r["name"], i)
        idx.setdefault(r["front_name"], i)
    rows = cards.to_dicts()
    beta = np.array([r["beta_bo1"] if r["beta_bo1"] is not None else np.nan for r in rows], float)
    nonland = np.array([not r["is_land"] for r in rows])
    thr = np.nanpercentile(beta[nonland & np.isfinite(beta)], 90) if (nonland & np.isfinite(beta)).sum() > 10 else np.nan
    rem_re = re.compile(r"destroy target|exile target|deals? (\d+|x) damage to (any target|target|each)|fights? |gets? -\d+/-\d+|-x/-x|"
                        r"deals damage equal|return target [^.]* to (its|their) owner's hand|sacrifices? (a|an) ")
    fl = {k: np.zeros(len(names), bool) for k in
          ["removal_rough", "creature", "cheap_creature_mv<=2", "mid_creature_mv3-4", "big_creature_mv>=5", "noncreature_spell",
           "cheap_spell_mv<=2", "expensive_mv>=5", "fixing_rough", "bomb_top10pct_model", "rare_or_mythic", "common", "multicolor", "colorless_nonland", "land"]}
    for k, n in enumerate(names):
        i = idx.get(n)
        if i is None:
            continue
        r = rows[i]
        txt = (r["oracle_text"] or "").lower()
        mv = r["mana_value"] or 0
        land = bool(r["is_land"])
        fl["land"][k] = land
        fl["creature"][k] = bool(r["is_creature"])
        if not land:
            fl["removal_rough"][k] = bool(rem_re.search(txt)) and "target" in txt
            fl["cheap_spell_mv<=2"][k] = mv <= 2
            fl["expensive_mv>=5"][k] = mv >= 5
            if r["is_creature"]:
                fl["cheap_creature_mv<=2"][k] = mv <= 2
                fl["mid_creature_mv3-4"][k] = 3 <= mv <= 4
                fl["big_creature_mv>=5"][k] = mv >= 5
            else:
                fl["noncreature_spell"][k] = True
            fl["bomb_top10pct_model"][k] = bool(np.isfinite(beta[i]) and np.isfinite(thr) and beta[i] >= thr)
            fl["multicolor"][k] = len(r["colors"] or "") >= 2
            fl["colorless_nonland"][k] = not (r["colors"] or "")
        else:
            prod = r["produces"] or ""
            fl["fixing_rough"][k] = (not r["is_basic"]) and len(prod) >= 2
        if not land and re.search(r"search your library for (a|up to \w+) (basic )?land|any color|basic land card", txt):
            fl["fixing_rough"][k] = True
        fl["rare_or_mythic"][k] = r["rarity"] in ("rare", "mythic")
        fl["common"][k] = r["rarity"] == "common"
    return fl


def analyse_set(sc: str, out: Path) -> dict:
    z = np.load(out / f"acc_{sc}.npz")
    seen, taken = z["seen"], z["taken"]            # (C, 3, 4, S, 2)
    names = [str(x) for x in z["names"]]
    C = len(names)
    cards = card_table(sc)
    rows_by = {}
    for r in cards.iter_rows(named=True):
        rows_by.setdefault(r["name"], r); rows_by.setdefault(r["front_name"], r)
    meta = [rows_by.get(n) for n in names]
    beta_pp = np.array([m["beta_bo1"] * 24.75 if (m and m["beta_bo1"] is not None) else np.nan for m in meta])
    low_data = np.array([bool(m["low_data_bo1"]) if m else True for m in meta])
    is_land = np.array([bool(m["is_land"]) if m else False for m in meta])
    is_basic = np.array([bool(m["is_basic"]) if m else False for m in meta])

    def grp(arr, rgs, sgs, half=None):
        a = arr[:, rgs][:, :, sgs]                 # (C, r, s, S, 2)
        a = a.sum(axis=(1, 2))
        return a.sum(-1) if half is None else a[..., half]

    ALL = [0, 1, 2, 3]; KNOWN = [0, 1, 2]
    defs = {
        "rank_Dplus_vs_low": dict(A=([2], ALL), B=([0], ALL)),
        "rank_Dplus_vs_low_skill054_060": dict(A=([2], [1]), B=([0], [1])),
        "skill_high_vs_rest": dict(A=([0, 1, 2], [2]), B=([0, 1, 2], [0, 1])),
        "skill_high_vs_mid_within_Dplus": dict(A=([2], [2]), B=([2], [1])),
        "skill_high_vs_mid_within_low": dict(A=([0], [2]), B=([0], [1])),
        "rank_plat_vs_low": dict(A=([1], ALL), B=([0], ALL)),
    }
    res = {}
    for k, d in defs.items():
        for half in (None, 0, 1):
            sA, tA = grp(seen, *d["A"], half), grp(taken, *d["A"], half)
            sB, tB = grp(seen, *d["B"], half), grp(taken, *d["B"], half)
            lor, se, n1, n0 = mh(tA, sA, tB, sB)
            res[(k, half)] = dict(lor=lor, se=se, n1=n1, n0=n0)
    # design effect from the two halves
    def inflation(k):
        a, b = res[(k, 0)], res[(k, 1)]
        zz = (a["lor"] - b["lor"]) / np.sqrt(a["se"] ** 2 + b["se"] ** 2)
        full = res[(k, None)]
        ok = np.isfinite(zz) & (full["n1"] >= MIN_SEEN) & (full["n0"] >= MIN_SEEN) & (a["n1"] >= 100) & (b["n1"] >= 100)
        return float(np.sqrt(np.var(zz[ok]))) if ok.sum() > 10 else float("nan"), int(ok.sum())
    infl = {}
    for k in defs:
        v, n = inflation(k)
        infl[k] = dict(se_inflation=v, n_cards=n)
    # overall pick rate and ATA
    s_all = seen.sum(axis=(1, 2, 4)); t_all = taken.sum(axis=(1, 2, 4))     # (C, S)
    seen_tot = s_all.sum(-1); take_tot = t_all.sum(-1)
    with np.errstate(all="ignore"):
        pick_rate = take_tot / seen_tot
        ata_all = z["pos_sum"].sum(axis=(1, 2)) / z["pos_cnt"].sum(axis=(1, 2))
    ps, pc_ = z["pos_sum"], z["pos_cnt"]       # (C,3,W)
    def ata_grp(g):
        with np.errstate(all="ignore"):
            return ps[:, g].sum(-1) / pc_[:, g].sum(-1)
    def ata_gap(gA, gB):
        # week-standardised gap, weights = taken count of group A per week (weeks with both groups)
        with np.errstate(all="ignore"):
            a = ps[:, gA] / pc_[:, gA]; b = ps[:, gB] / pc_[:, gB]
            w = np.where((pc_[:, gA] > 0) & (pc_[:, gB] > 0), pc_[:, gA], 0).astype(float)
            return np.nansum(np.where(w > 0, (a - b) * w, 0), -1) / w.sum(-1)

    full = res[("rank_Dplus_vs_low", None)]
    nD = grp(seen, [2], ALL); nL = grp(seen, [0], ALL)
    seenD = nD.sum(-1); seenL = nL.sum(-1)
    elig = (full["n1"] >= MIN_SEEN) & (full["n0"] >= MIN_SEEN) & np.isfinite(full["lor"]) & ~is_basic
    infl_main = infl["rank_Dplus_vs_low"]["se_inflation"]
    f_adj = max(1.0, infl_main) if np.isfinite(infl_main) else 1.0

    def card_rows(order):
        out_rows = []
        for i in order:
            m = meta[i] or {}
            out_rows.append(dict(
                card=names[i], rarity=m.get("rarity"), colors=m.get("colors"), mana_value=m.get("mana_value"),
                model_pp=None if not np.isfinite(beta_pp[i]) else round(float(beta_pp[i]), 2),
                OR=round(float(np.exp(full["lor"][i])), 3), CI95_naive=[round(float(np.exp(full["lor"][i] - 1.96 * full["se"][i])), 3), round(float(np.exp(full["lor"][i] + 1.96 * full["se"][i])), 3)],
                CI95_adj=[round(float(np.exp(full["lor"][i] - 1.96 * full["se"][i] * f_adj)), 3), round(float(np.exp(full["lor"][i] + 1.96 * full["se"][i] * f_adj)), 3)],
                seen_Dplus=int(seenD[i]), seen_low=int(seenL[i]),
                ATA_Dplus=round(float(ata_grp(2)[i]), 2), ATA_plat=round(float(ata_grp(1)[i]), 2), ATA_low=round(float(ata_grp(0)[i]), 2),
                ATA_gap_week_adj=round(float(ata_gap(2, 0)[i]), 2),
                P_taken_when_seen=round(float(pick_rate[i]), 3)))
        return out_rows
    idx_el = np.where(elig)[0]
    ordr = idx_el[np.argsort(full["lor"][idx_el])]
    top_more = card_rows(ordr[::-1][:15]); top_less = card_rows(ordr[:15])

    # ----- correlation with model residual
    def resid_corr(key, mask_extra=None, x_kind="rate"):
        r = res[(key, None)]
        ok = (np.isfinite(r["lor"]) & (r["n1"] >= MIN_SEEN) & (r["n0"] >= MIN_SEEN) & np.isfinite(beta_pp) & ~is_basic & ~is_land
              & np.isfinite(pick_rate) & (pick_rate > 0) & (pick_rate < 1))
        if mask_extra is not None:
            ok &= mask_extra
        if x_kind == "rate":
            x = np.log(pick_rate / (1 - pick_rate))
        else:
            x = ata_all
        ok &= np.isfinite(x)
        if ok.sum() < 15:
            return None
        X = np.vander(x[ok], 4)
        coef, *_ = np.linalg.lstsq(X, beta_pp[ok], rcond=None)
        resid = np.full(C, np.nan); resid[ok] = beta_pp[ok] - X @ coef
        y = r["lor"]
        rr = pearson(resid[ok], y[ok]); sp = spearman(resid[ok], y[ok])
        # precision-weighted version
        w = 1 / (r["se"][ok] ** 2 + np.var(y[ok]) * 0.0 + 1e-6)
        mu_y = np.average(y[ok], weights=w); mu_r = np.average(resid[ok], weights=w)
        wc = (w * (y[ok] - mu_y) * (resid[ok] - mu_r)).sum() / math.sqrt((w * (y[ok] - mu_y) ** 2).sum() * (w * (resid[ok] - mu_r) ** 2).sum())
        # signal share: var(logOR) beyond noise
        sig_sd = math.sqrt(max(0.0, np.var(y[ok]) - np.mean(r["se"][ok] ** 2)))
        n = int(ok.sum())
        fz = math.atanh(rr); sef = 1 / math.sqrt(n - 3)
        return dict(n_cards=n, pearson=rr, pearson_boot_CI=boot_corr(resid[ok], y[ok]), pearson_fisher_CI=[math.tanh(fz - 1.96 * sef), math.tanh(fz + 1.96 * sef)],
                    spearman=sp, pearson_raw_model_pp=pearson(beta_pp[ok], y[ok]), precision_weighted_pearson=float(wc), sd_logOR=float(np.std(y[ok])), mean_se_logOR=float(np.mean(r["se"][ok])),
                    signal_sd_logOR=sig_sd, resid_sd_pp=float(np.std(resid[ok])),
                    _y=y[ok].tolist(), _r=resid[ok].tolist(), _w=(1 / r["se"][ok] ** 2).tolist())
    corr = {}
    for key in ["rank_Dplus_vs_low", "rank_Dplus_vs_low_skill054_060", "skill_high_vs_rest", "skill_high_vs_mid_within_Dplus",
                "skill_high_vs_mid_within_low", "rank_plat_vs_low"]:
        corr[key] = resid_corr(key)
    corr_excl_low = resid_corr("rank_Dplus_vs_low", mask_extra=~low_data)
    corr_ata = resid_corr("rank_Dplus_vs_low", x_kind="ata")
    # split-half replication of the log-OR vector
    r0, r1 = res[("rank_Dplus_vs_low", 0)], res[("rank_Dplus_vs_low", 1)]
    okh = elig & np.isfinite(r0["lor"]) & np.isfinite(r1["lor"])
    half_corr = pearson(r0["lor"][okh], r1["lor"][okh])
    # rank vs skill logOR vectors
    rs = res[("skill_high_vs_rest", None)]
    okb = elig & np.isfinite(rs["lor"]) & (rs["n1"] >= MIN_SEEN) & (rs["n0"] >= MIN_SEEN)
    rank_skill_corr = pearson(full["lor"][okb], rs["lor"][okb])
    rm = res[("rank_Dplus_vs_low_skill054_060", None)]
    okm = elig & np.isfinite(rm["lor"]) & (rm["n1"] >= MIN_SEEN) & (rm["n0"] >= MIN_SEEN)
    rank_vs_rankmid_corr = pearson(full["lor"][okm], rm["lor"][okm])

    # ----- sizes
    d_rank, d_skill = z["d_rank"], z["d_skill"]
    sizes = dict(drafts_by_rank={RG_NAMES[g]: int((d_rank == g).sum()) for g in range(3)},
                 drafts_by_skill={SG_NAMES[g]: int((d_skill == g).sum()) for g in range(4)},
                 drafts_Dplus_by_skill={SG_NAMES[g]: int(((d_rank == 2) & (d_skill == g)).sum()) for g in range(4)},
                 drafts_low_by_skill={SG_NAMES[g]: int(((d_rank == 0) & (d_skill == g)).sum()) for g in range(4)})

    # ----- colours: P1p1-5
    d_week, d_early, d_nearly = z["d_week"], z["d_early"], z["d_nearly"]
    d_deck, d_ndeck = z["d_deck"], z["d_ndeck"]

    def std_diff(values, valid, gA, gB, nw=W):
        """values: (D, K) per-draft indicator/share; valid: (D,) bool. Returns per-K mean A, B (standardised to A's week mix), diff, se."""
        K = values.shape[1]
        wA = np.array([((d_rank == gA) & valid & (d_week == w)).sum() for w in range(nw)], float)
        wB = np.array([((d_rank == gB) & valid & (d_week == w)).sum() for w in range(nw)], float)
        use = (wA >= 5) & (wB >= 5)
        wts = np.where(use, wA, 0); wts = wts / wts.sum()
        mA = np.zeros(K); mB = np.zeros(K); vA = np.zeros(K); vB = np.zeros(K)
        for w in np.where(use)[0]:
            for g, m_, v_ in ((gA, mA, vA), (gB, mB, vB)):
                sel = (d_rank == g) & valid & (d_week == w)
                vals = values[sel]
                m_ += wts[w] * vals.mean(0)
                v_ += wts[w] ** 2 * vals.var(0, ddof=1) / sel.sum()
        return mA, mB, mA - mB, np.sqrt(vA + vB)

    col_res = {}
    ve = d_nearly >= 4          # TMT has no P1p1 rows in the file (picks 2-5 only)
    with np.errstate(all="ignore"):
        early_share = d_early.astype(float) / np.maximum(d_nearly, 1)[:, None]
    for nm, (gA, gB) in {"Dplus_vs_low": (2, 0), "plat_vs_low": (1, 0)}.items():
        mA, mB, df, se = std_diff(early_share, ve, gA, gB)
        col_res[f"P1p1-5_pick_colour_share_{nm}"] = {c: dict(A=round(float(mA[j]), 4), B=round(float(mB[j]), 4), diff=round(float(df[j]), 4),
                                                            CI95=[round(float(df[j] - 1.96 * se[j]), 4), round(float(df[j] + 1.96 * se[j]), 4)])
                                                    for j, c in enumerate(list(COLORS) + ["colourless"])}
    # deck pairs
    valid_deck = d_ndeck >= 15
    order = np.argsort(-d_deck, axis=1, kind="stable")
    top1 = order[:, 0]; top2 = order[:, 1]
    cnt2 = np.take_along_axis(d_deck, order[:, 1:2], 1)[:, 0]
    pairs = ["".join(sorted([COLORS[a], COLORS[b]], key=COLORS.index)) for a in range(5) for b in range(a + 1, 5)]
    lab = np.empty(len(d_rank), dtype=object)
    for i in range(len(d_rank)):
        a, b = top1[i], top2[i]
        lab[i] = "".join(sorted([COLORS[a], COLORS[b]], key=COLORS.index)) if cnt2[i] >= 4 else "mono-" + COLORS[a]
    labs = pairs + ["mono-" + c for c in COLORS]
    ind = np.stack([(lab == l).astype(float) for l in labs], 1)
    # colour share among deck cards (colour-weighted)
    deck_col_share = d_deck / np.maximum(d_deck.sum(1, keepdims=True), 1)
    for nm, (gA, gB) in {"Dplus_vs_low": (2, 0), "plat_vs_low": (1, 0)}.items():
        mA, mB, df, se = std_diff(ind, valid_deck, gA, gB)
        col_res[f"deck_colour_pair_share_{nm}"] = {l: dict(A=round(float(mA[j]), 4), B=round(float(mB[j]), 4), diff=round(float(df[j]), 4),
                                                          CI95=[round(float(df[j] - 1.96 * se[j]), 4), round(float(df[j] + 1.96 * se[j]), 4)])
                                                  for j, l in enumerate(labs)}
        mA, mB, df, se = std_diff(deck_col_share, valid_deck, gA, gB)
        col_res[f"deck_card_colour_share_{nm}"] = {c: dict(A=round(float(mA[j]), 4), B=round(float(mB[j]), 4), diff=round(float(df[j]), 4),
                                                           CI95=[round(float(df[j] - 1.96 * se[j]), 4), round(float(df[j] + 1.96 * se[j]), 4)])
                                                   for j, c in enumerate(COLORS)}
    col_res["n_decks_valid"] = {RG_NAMES[g]: int(((d_rank == g) & valid_deck).sum()) for g in range(3)}
    # also by skill-matched (mid band) for deck pairs
    for nm, (gA, gB) in {"Dplus_vs_low_skill054_060": (2, 0)}.items():
        keep = valid_deck & (d_skill == 1)
        mA, mB, df, se = std_diff(ind, keep, gA, gB)
        col_res[f"deck_colour_pair_share_{nm}"] = {l: dict(A=round(float(mA[j]), 4), B=round(float(mB[j]), 4), diff=round(float(df[j]), 4),
                                                          CI95=[round(float(df[j] - 1.96 * se[j]), 4), round(float(df[j] + 1.96 * se[j]), 4)])
                                                  for j, l in enumerate(labs)}

    # ----- card types
    fl = cat_flags(cards, names)
    types_res = {}
    seenR = grp(seen, [2], ALL); seenLw = grp(seen, [0], ALL)
    for gname, (rgA, sgA, rgB, sgB) in {"Dplus_vs_low": ([2], ALL, [0], ALL), "Dplus_vs_low_skill054_060": ([2], [1], [0], [1]),
                                         "skill_high_vs_rest": ([0, 1, 2], [2], [0, 1, 2], [0, 1])}.items():
        tkA = grp(taken, rgA, sgA); tkB = grp(taken, rgB, sgB)       # (C, S)
        totA = tkA.sum(0).astype(float); totB = tkB.sum(0).astype(float)
        use = (totA > 50) & (totB > 50)
        wts = np.where(use, totA, 0); wts = wts / wts.sum()
        rr = {}
        for fname, m in fl.items():
            if not m.any():
                continue
            pA = np.where(use, tkA[m].sum(0) / np.maximum(totA, 1), 0); pB = np.where(use, tkB[m].sum(0) / np.maximum(totB, 1), 0)
            vA = np.where(use, pA * (1 - pA) / np.maximum(totA, 1), 0); vB = np.where(use, pB * (1 - pB) / np.maximum(totB, 1), 0)
            sA = (wts * pA).sum(); sB = (wts * pB).sum()
            se = math.sqrt(((wts ** 2) * (vA + vB)).sum())
            rr[fname] = dict(A=round(float(sA), 4), B=round(float(sB), 4), diff=round(float(sA - sB), 4),
                             CI95_naive=[round(float(sA - sB - 1.96 * se), 4), round(float(sA - sB + 1.96 * se), 4)], n_cards=int(m.sum()))
        types_res[gname] = rr
    # per-card random-effects mean logOR by category (D+ vs low, pooled with card-level noise)
    types_card = {}
    r = res[("rank_Dplus_vs_low", None)]
    for fname, m in fl.items():
        sel = elig & m
        types_card[fname] = dl_mean(r["lor"][sel], r["se"][sel] * f_adj)
    # rarity
    rar = np.array([m["rarity"] if m else "?" for m in meta])
    for rn in ["common", "uncommon", "rare", "mythic"]:
        types_card["rarity_" + rn] = dl_mean(r["lor"][elig & (rar == rn)], r["se"][elig & (rar == rn)] * f_adj)

    # strip heavy arrays
    for k, v in corr.items():
        pass
    out_corr = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if v else None) for k, v in corr.items()}
    out_corr["rank_Dplus_vs_low_excl_low_data_cards"] = {kk: vv for kk, vv in (corr_excl_low or {}).items() if not kk.startswith("_")} or None
    out_corr["rank_Dplus_vs_low_x=ATA"] = {kk: vv for kk, vv in (corr_ata or {}).items() if not kk.startswith("_")} or None
    summary_rows = {k: dict(n_elig_cards=int(((res[(k, None)]["n1"] >= MIN_SEEN) & (res[(k, None)]["n0"] >= MIN_SEEN) & np.isfinite(res[(k, None)]["lor"])).sum()),
                            sd_logOR=float(np.nanstd(res[(k, None)]["lor"][(res[(k, None)]["n1"] >= MIN_SEEN) & (res[(k, None)]["n0"] >= MIN_SEEN)])),
                            mean_se=float(np.nanmean(res[(k, None)]["se"][(res[(k, None)]["n1"] >= MIN_SEEN) & (res[(k, None)]["n0"] >= MIN_SEEN)])),
                            se_inflation=infl[k]["se_inflation"]) for k in defs}
    for k, v in summary_rows.items():
        v["signal_sd_logOR"] = math.sqrt(max(0.0, v["sd_logOR"] ** 2 - (v["mean_se"] * (v["se_inflation"] if np.isfinite(v["se_inflation"]) else 1)) ** 2))
    return dict(set=sc, sizes=sizes, se_inflation_from_halves=infl, comparisons=summary_rows, top_taken_more_by_Dplus=top_more,
                top_taken_less_by_Dplus=top_less, n_eligible_cards=int(elig.sum()), split_half_corr_logOR=half_corr,
                corr_rank_logOR_vs_skill_logOR=rank_skill_corr, corr_logOR_all_skill_vs_midband=rank_vs_rankmid_corr,
                correlation_with_model_residual=out_corr, colours=col_res, card_types_pick_share=types_res,
                card_types_mean_logOR_D_vs_low=types_card), corr, {k: [(res[(k, None)]["lor"]), (res[(k, None)]["se"])] for k in defs}


def pooled_corr(per_set_corr: dict, key: str):
    ys, rs, ss = [], [], []
    zs, ws = [], []
    for sc, c in per_set_corr.items():
        v = c.get(key)
        if not v:
            continue
        y = np.array(v["_y"]); r = np.array(v["_r"])
        ys.append(y - y.mean()); rs.append(r - r.mean()); ss += [sc] * len(y)
        zs.append(math.atanh(v["pearson"])); ws.append(v["n_cards"] - 3)
    if not ys:
        return None
    y = np.concatenate(ys); r = np.concatenate(rs)
    rr = pearson(r, y)
    # bootstrap cards
    meta_z = np.average(zs, weights=ws); se = 1 / math.sqrt(sum(ws))
    return dict(n_cards=int(len(y)), pooled_pearson_set_demeaned=rr, boot_CI=boot_corr(r, y, B=2000), spearman=spearman(r, y),
                fisher_meta_pearson=math.tanh(meta_z), fisher_meta_CI=[math.tanh(meta_z - 1.96 * se), math.tanh(meta_z + 1.96 * se)],
                per_set={sc: c[key]["pearson"] for sc, c in per_set_corr.items() if c.get(key)})


# ----------------------------------------------------------------------------------------------------------- markdown
def fmt_cards(rows):
    h = "| card | rar | col | mv | model pp | OR (95% CI naive) | CI adj | seen D+ / low | ATA D+ | ATA plat | ATA low | ATA gap wk-adj | P(taken|seen) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        h += (f"| {r['card']} | {r['rarity']} | {r['colors'] or '-'} | {r['mana_value']:.0f} | {r['model_pp']} | {r['OR']:.2f} ({r['CI95_naive'][0]:.2f}-{r['CI95_naive'][1]:.2f}) | "
              f"{r['CI95_adj'][0]:.2f}-{r['CI95_adj'][1]:.2f} | {r['seen_Dplus']:,} / {r['seen_low']:,} | {r['ATA_Dplus']} | {r['ATA_plat']} | {r['ATA_low']} | {r['ATA_gap_week_adj']:+.2f} | {r['P_taken_when_seen']} |\n")
    return h


def write_md(res: dict, path: Path):
    L = []
    L.append("# Do Diamond+ drafters pick differently? (PremierDraft pick files)\n")
    L.append(f"Sets: {', '.join(res['sets'])}. Method and caveats at the end.\n")
    for sc in res["sets"]:
        s = res["per_set"][sc]
        L.append(f"\n## {sc}\n")
        L.append(f"Drafts by rank: {s['sizes']['drafts_by_rank']}; D+ by skill: {s['sizes']['drafts_Dplus_by_skill']}; low by skill: {s['sizes']['drafts_low_by_skill']}.\n")
        L.append(f"Eligible cards (>= {MIN_SEEN} seen in both groups): {s['n_eligible_cards']}. Split-half correlation of the card log-OR vector: {s['split_half_corr_logOR']:.2f}. "
                 f"SE inflation from halves (rank D+ vs low): {s['se_inflation_from_halves']['rank_Dplus_vs_low']['se_inflation']:.2f}.\n")
        L.append("\n### Taken more readily by D+ (OR > 1)\n"); L.append(fmt_cards(s["top_taken_more_by_Dplus"]))
        L.append("\n### Taken less readily by D+ (OR < 1)\n"); L.append(fmt_cards(s["top_taken_less_by_Dplus"]))
        c = s["colours"]
        L.append("\n### Colours\n")
        t = c["P1p1-5_pick_colour_share_Dplus_vs_low"]
        L.append("P1p1-p5 picks that carry the colour (share of those picks; D+ vs low, week-standardised to the D+ week mix):\n\n| colour | D+ | low | diff | 95% CI |\n|---|---|---|---|---|\n" +
                 "".join(f"| {k} | {v['A']:.3f} | {v['B']:.3f} | {v['diff']:+.3f} | {v['CI95'][0]:+.3f}..{v['CI95'][1]:+.3f} |\n" for k, v in t.items()))
        t = c["deck_colour_pair_share_Dplus_vs_low"]
        L.append("\nFinal main-deck colour pair (top-2 colours of main-deck picks; second colour needs >= 4 cards else mono), share of decks:\n\n| pair | D+ | low | diff | 95% CI | diff (skill 0.54-0.60 only) |\n|---|---|---|---|---|---|\n")
        t2 = c["deck_colour_pair_share_Dplus_vs_low_skill054_060"]
        for k, v in sorted(t.items(), key=lambda kv: -abs(kv[1]["diff"])):
            L.append(f"| {k} | {v['A']:.3f} | {v['B']:.3f} | {v['diff']:+.3f} | {v['CI95'][0]:+.3f}..{v['CI95'][1]:+.3f} | {t2[k]['diff']:+.3f} |\n")
        L.append(f"\nDecks used: {c['n_decks_valid']}.\n")
        L.append("\n### Card types (share of all picks that are of the type; D+ vs low, week x pack x band standardised)\n\n| type | D+ | low | diff | CI naive | diff skill-matched | diff high-skill vs rest | mean card log-OR (RE) |\n|---|---|---|---|---|---|---|---|\n")
        ty = s["card_types_pick_share"]; tc = s["card_types_mean_logOR_D_vs_low"]
        for k, v in ty["Dplus_vs_low"].items():
            sm = ty["Dplus_vs_low_skill054_060"].get(k, {}).get("diff", float("nan")); hs = ty["skill_high_vs_rest"].get(k, {}).get("diff", float("nan"))
            m = tc.get(k, {})
            L.append(f"| {k} | {v['A']:.3f} | {v['B']:.3f} | {v['diff']:+.4f} | {v['CI95_naive'][0]:+.4f}..{v['CI95_naive'][1]:+.4f} | {sm:+.4f} | {hs:+.4f} | {m.get('mean', float('nan')):+.3f} ({m.get('lo', float('nan')):+.3f}..{m.get('hi', float('nan')):+.3f}) |\n")
        L.append("\n### Direction check and rank vs skill\n\n| comparison | cards | corr(logOR, model residual) | boot 95% CI | Spearman | sd logOR | mean SE | signal sd |\n|---|---|---|---|---|---|---|---|\n")
        for k, v in s["correlation_with_model_residual"].items():
            if v:
                L.append(f"| {k} | {v['n_cards']} | {v['pearson']:+.3f} | {v['pearson_boot_CI'][0]:+.2f}..{v['pearson_boot_CI'][1]:+.2f} | {v['spearman']:+.3f} | {v['sd_logOR']:.3f} | {v['mean_se_logOR']:.3f} | {v['signal_sd_logOR']:.3f} |\n")
        L.append(f"\nCorrelation between the card log-OR vector of rank (D+ vs low) and of skill (high vs rest): {s['corr_rank_logOR_vs_skill_logOR']:.2f}; rank log-OR all skills vs skill-matched band: {s['corr_logOR_all_skill_vs_midband']:.2f}.\n")
    L.append("\n## Pooled across sets: do D+ deviations point the right way?\n\n| comparison | cards | pooled corr (set-demeaned) | boot 95% CI | Fisher-meta corr | meta 95% CI | per set |\n|---|---|---|---|---|---|---|\n")
    for k, v in res["pooled_correlation"].items():
        if v:
            L.append(f"| {k} | {v['n_cards']} | {v['pooled_pearson_set_demeaned']:+.3f} | {v['boot_CI'][0]:+.2f}..{v['boot_CI'][1]:+.2f} | {v['fisher_meta_pearson']:+.3f} | {v['fisher_meta_CI'][0]:+.2f}..{v['fisher_meta_CI'][1]:+.2f} | " +
                     ", ".join(f"{a} {b:+.2f}" for a, b in v["per_set"].items()) + " |\n")
    L.append("\n## Method\n" + res["method"] + "\n")
    path.write_text("".join(L), encoding="utf-8")


METHOD = """
- Unit: every pick row; every card with pack_card_* > 0 is "seen"; "taken" = it equals `pick` (`pick_2` is empty in all sets, so ignored).
- Groups: rank low = bronze/silver/gold, plat, D+ = diamond/mythic (drafts with no rank dropped). Skill from user_game_win_rate_bucket: <0.54, 0.54-0.60 (buckets 0.54-0.58), >=0.60, unknown.
- Strata: calendar week (from 2026-01-01) x pack x pick-number band (picks 1,2,3,4,5-6,7-9,10-14). Mantel-Haenszel OR (D+ over low) per card across strata with both groups present; Robins-Breslow-Greenland variance.
- Naive CIs treat picks as independent although a drafter sees a card repeatedly and one drafter makes ~42 picks. A split-half (parity of draft order index) comparison of the card log-ORs gives an empirical SE inflation factor (sd of the half-difference z); "CI adj" multiplies the SE by max(1, that factor).
- Minimum 300 "seen" per group per card within strata with both groups present; basic lands excluded.
- ATA = mean overall pick position (pack*14 + pick) of the card when taken, raw per group; "ATA gap wk-adj" = D+ minus low, week-standardised (weights = D+ takes per week).
- Colours: P1p1-p5 share is the per-draft share of the 5 picks (TMT: the file has no P1p1 rows, so p2-p5; drafts need >= 4 of them) carrying the colour (multicolour counts in each colour; unit = draft), standardised to the D+ week mix. Deck pair: from picks with pick_maindeck_rate >= 0.5 (drafts with >= 15 such picks), top-2 colours by count of main-deck picks carrying the colour (gold cards count in both); second colour needs >= 4 cards, else mono.
- Card types are rough regex flags on oracle_text/types/mana_value/produces (removal, fixing especially coarse); "bomb" = top 10% of the set's beta_bo1 among non-lands. Pick share = fraction of all picks that are of the type, standardised across strata to the D+ mix. Random-effects (DerSimonian-Laird) mean of card log-ORs by type uses SE inflated as above.
- Direction check: model value pp = beta_bo1 x 24.75; residual = pp minus a cubic fit on logit(overall P(taken|seen)); correlate with the card's rank log-OR (positive = D+ take more what the crowd undervalues). Bootstrap over cards for CIs. Pooled = set-demeaned vectors concatenated, plus a Fisher-z meta-average.
- Rank vs skill: same pipeline for skill high vs rest, plus rank within a fixed skill band and skill within rank.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    picks = os.environ.get("PICKS_DIR", DEF_PICKS)
    t0 = time.time()
    if not a.reuse:
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(accumulate, sc, picks, out, a.max_rows) for sc in a.sets]
            for f in futs:
                f.result()
    per_set, corrs, infos = {}, {}, {}
    for sc in a.sets:
        r, c, _ = analyse_set(sc, out)
        per_set[sc] = r; corrs[sc] = c
        infos[sc] = json.loads((out / f"info_{sc}.json").read_text())
    keys = ["rank_Dplus_vs_low", "rank_Dplus_vs_low_skill054_060", "skill_high_vs_rest", "skill_high_vs_mid_within_Dplus", "skill_high_vs_mid_within_low", "rank_plat_vs_low"]
    pooled = {k: pooled_corr(corrs, k) for k in keys}
    res = dict(sets=a.sets, per_set=per_set, pooled_correlation=pooled, data_info=infos, method=METHOD, runtime_secs=time.time() - t0)
    (out / "rank_picks.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_md(res, out / "rank_picks.md")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
