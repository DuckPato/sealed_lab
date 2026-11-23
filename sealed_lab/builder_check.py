"""Builder check: do REAL player decks shaped more like the builder's win more, among players of the same skill?

    run(set_code) -> dict      (both formats; pure computation: reads the corpus, the model's meta and the saved builder composition)

Pre-registered design (do not change after seeing results):
  * Games: report.load_games (the format's 40-card-deck games; bo1 = Sealed, bo3 = TradSealed), one row per game.
  * Model: logistic regression of the game win on the four STANDARDISED deck metrics jointly (lands, creatures, two_drops, mean_mv)
    plus the colour-pair fixed effects (report.pair_group dummies, reference = most common group, as report.pairs_section) and the
    report's pilot covariates and penalties (design.CovEncoder: skill bins, on the play, mulligans, event, time, opponent colours),
    fitted by report._fit (glm.fit) with standard errors clustered by draft (glm.cluster_se).  The metric columns are standardised
    with a weak N(0, 10) prior (as the report's dummies).  Games without main colours are dropped (as in the pairs section).
  * Deck metrics use scripts/builder_composition.py's definitions: creatures = nonland creature cards, two_drops = nonland cards of
    mana value 2, lands = land cards, mean_mv = mean mana value of nonland cards.
  * Quantity: delta_logit = sum_k beta_k d_k (beta per raw unit, d_k the builder's `diff_players`), in points at the format's average
    win rate p: p (1 - p) delta_logit 100.  se_fixed treats d_k as fixed (delta method on the clustered covariance); se_total adds
    the uncertainty of d_k: Var = d' S d + beta' diag(se_d^2) beta.  The badge uses se_total.
  * Badge: |z| >= 3 strong, >= 2 likely, else weak (report.badge_of); only likely / strong may be stated as a finding.
  * Minimum data: fewer than MIN_GAMES games -> the format is skipped (reason recorded).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from sealed_lab import config, report as R

log = logging.getLogger(__name__)

CHECK_VERSION = 1
FORMATS = R.FORMATS
METRICS = ("lands", "creatures", "two_drops", "mean_mv")
MIN_GAMES = 3000
PRIOR_SIGMA = 10.0
COMPOSITION_DIR = Path(__file__).resolve().parents[1] / "docs" / "builder_composition"


# ---------------------------------------------------------------------------------------------
# pure arithmetic (unit-tested by hand)
# ---------------------------------------------------------------------------------------------

def badge(z) -> str:
    """Strong | Likely | Weak (the app's rule, report.badge_of)."""
    return R.badge_of(z).capitalize()


def pp_scale(p: float) -> float:
    """Points of game win rate per logit at the average win rate p."""
    return 100.0 * p * (1.0 - p)


def combine(beta, cov, d, se_d, p: float) -> dict:
    """Predicted change in win rate (points) when a deck moves by the builder's differences d.
    beta: per-raw-unit coefficients, cov: their covariance, d / se_d: the differences and their standard errors."""
    beta, d, se_d = (np.asarray(x, dtype=float) for x in (beta, d, se_d))
    cov = np.asarray(cov, dtype=float)
    k = pp_scale(p)
    pp = k * float(beta @ d)
    var_fixed = float(d @ cov @ d)
    var_total = var_fixed + float(np.sum(beta ** 2 * se_d ** 2))
    se_fixed, se_total = k * np.sqrt(max(var_fixed, 0.0)), k * np.sqrt(max(var_total, 0.0))
    z = pp / se_total if se_total > 0 else None
    return dict(pp=pp, se_fixed=float(se_fixed), se_total=float(se_total), z=z, badge=badge(z))


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------

def deck_metrics(set_code: str, cards: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per deck_key: lands, creatures, two_drops, mean_mv (same definitions as scripts/builder_composition.py)."""
    cd = config.corpus_dir(set_code)
    if cards is None:
        cards = pl.read_parquet(cd / "cards.parquet")
    c = cards.select("card_id", pl.col("is_land").fill_null(False), pl.col("is_creature").fill_null(False),
                     pl.col("mana_value").fill_null(0).cast(pl.Float64).alias("mv"))
    dc = pl.read_parquet(cd / "deck_cards.parquet", columns=["deck_key", "card_id", "n"]).join(c, on="card_id", how="left").with_columns(
        pl.col("is_land").fill_null(False), pl.col("is_creature").fill_null(False), pl.col("mv").fill_null(0.0))
    nl = ~pl.col("is_land")
    return dc.group_by("deck_key").agg(
        (pl.col("n") * pl.col("is_land")).sum().cast(pl.Float64).alias("lands"),
        (pl.col("n") * (nl & pl.col("is_creature"))).sum().cast(pl.Float64).alias("creatures"),
        (pl.col("n") * (nl & (pl.col("mv") == 2))).sum().cast(pl.Float64).alias("two_drops"),
        ((pl.col("n") * nl * pl.col("mv")).sum() / pl.col("n").filter(nl).sum().clip(lower_bound=1)).alias("mean_mv"))


def load_diffs(set_code: str, path: Path | None = None) -> tuple[dict, str | None]:
    """({fmt: {metric: (diff_players, diff_players_se)}}, why-not) from builder_composition_<SET>.json."""
    p = Path(path) if path else COMPOSITION_DIR / f"builder_composition_{set_code}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {}, f"builder composition file missing or unreadable ({p.name}: {e.__class__.__name__})"
    out: dict = {}
    for fmt, blk in (d.get("formats") or {}).items():
        m = {}
        for k in METRICS:
            v = (blk or {}).get(k) or {}
            if v.get("diff_players") is not None and v.get("diff_players_se") is not None:
                m[k] = (float(v["diff_players"]), float(v["diff_players_se"]))
        out[fmt] = m
    return out, None


# ---------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------

def analyze(g: pl.DataFrame, diffs: dict) -> dict:
    """Fit on games `g` (report.GAME_COLS plus the four metric columns); `diffs` = {metric: (diff_players, se)} available for the
    builder.  Returns the format block (betas, combined)."""
    g = g.with_columns(pl.col("main_colors").map_elements(R.pair_group, return_dtype=pl.String).alias("grp")).filter(pl.col("grp").is_not_null())
    n = g.height
    out: dict = dict(n_games=n, n_decks=int(g["deck_key"].n_unique()))
    if n < MIN_GAMES:
        out["skipped"] = f"only {n} games (minimum {MIN_GAMES})"
        return out
    X = g.select(list(METRICS)).to_numpy().astype(float)
    mu, sd = X.mean(0), X.std(0)
    use = [i for i in range(len(METRICS)) if sd[i] > 1e-9]
    skipped_m = {METRICS[i]: "no variation across decks" for i in range(len(METRICS)) if i not in use}
    Z = (X[:, use] - mu[use]) / sd[use]
    # colour-pair dummies: the most common group is the reference level (report.pairs_section)
    per = g.group_by("grp").agg(pl.col("deck_key").n_unique().alias("nd")).sort("nd", descending=True)
    keys = per["grp"].to_list()
    others = keys[1:]
    ix = {k_: i for i, k_ in enumerate(others)}
    grp = g["grp"].to_numpy()
    rows_i = np.nonzero(np.isin(grp, others))[0]
    D = sp.csr_matrix((np.ones(len(rows_i)), (rows_i, np.array([ix[k_] for k_ in grp[rows_i]], dtype=np.int64))), shape=(n, len(others)))
    extra = sp.hstack([sp.csr_matrix(Z), D], format="csr")
    m = len(use)
    pairs_ix = [(a, b) for a in range(m) for b in range(a + 1, m)]
    scale = 1.0 / sd[use]                               # per-raw-unit = per-standardised-unit / sd

    def contrasts(k: int, ncol: int) -> np.ndarray:
        L = np.zeros((m + len(pairs_ix), ncol))
        for a in range(m):
            L[a, k + a] = scale[a]
        for r_, (a, b) in enumerate(pairs_ix):
            L[m + r_, k + a] = scale[a]
            L[m + r_, k + b] = scale[b]
        return L

    coef, enc, _prob, c_rob, _eta = R._fit(g, extra, np.full(extra.shape[1], PRIOR_SIGMA), contrasts)
    k0 = enc.k
    beta = np.array([coef[k0 + a] * scale[a] for a in range(m)])           # logit per raw unit
    var = c_rob[:m] ** 2
    cov = np.diag(var)
    for r_, (a, b) in enumerate(pairs_ix):
        cov[a, b] = cov[b, a] = (c_rob[m + r_] ** 2 - var[a] - var[b]) / 2.0
    p = float(g["won"].cast(pl.Float64).mean())
    kpp = pp_scale(p)
    betas: dict = {}
    for a, i in enumerate(use):
        name = METRICS[i]
        eff, se = kpp * float(beta[a]), kpp * float(np.sqrt(max(cov[a, a], 0.0)))
        z = eff / se if se > 0 else None
        dd = diffs.get(name)
        betas[name] = dict(pp_per_unit=eff, se=se, z=z, badge=badge(z), builder_diff=None if dd is None else dd[0],
                           builder_diff_se=None if dd is None else dd[1],
                           builder_direction=None if dd is None else ("more" if dd[0] > 0 else "fewer" if dd[0] < 0 else "same"),
                           sd=float(sd[i]), mean=float(mu[i]))
    out.update(avg_win_rate=p, betas=betas, skipped_metrics=skipped_m)
    sel = [a for a, i in enumerate(use) if METRICS[i] in diffs]
    out["builder_diffs"] = {METRICS[use[a]]: dict(diff_players=diffs[METRICS[use[a]]][0], diff_players_se=diffs[METRICS[use[a]]][1]) for a in sel}
    out["n_clusters"] = int(g["draft_id"].n_unique())
    if not sel:
        out["skipped"] = "no builder composition metric available for this format"
        out["combined"] = None
        return out
    d = np.array([diffs[METRICS[use[a]]][0] for a in sel])
    sd_d = np.array([diffs[METRICS[use[a]]][1] for a in sel])
    ix_ = np.array(sel)
    out["combined"] = combine(beta[ix_], cov[np.ix_(ix_, ix_)], d, sd_d, p)
    out["combined"]["metrics_used"] = [METRICS[use[a]] for a in sel]
    missing = [METRICS[i] for i in use if METRICS[i] not in diffs]
    if missing:
        out["skipped_metrics"].update({k_: "builder diff missing in the composition file" for k_ in missing})
    return out


def _model_built_at(set_code: str) -> str | None:
    try:
        return json.loads((config.model_dir(set_code) / "meta.json").read_text(encoding="utf-8")).get("built_at")
    except (OSError, ValueError):
        return None


def run(set_code: str, composition_path: Path | None = None) -> dict:
    """Both formats of a set.  Reads only."""
    set_code = set_code.upper()
    out: dict = dict(set=set_code, version=CHECK_VERSION, computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     built_at=_model_built_at(set_code), formats={})
    diffs, why = load_diffs(set_code, composition_path)
    if why:
        out["skipped"] = why
        return out
    dm = None
    for fmt in FORMATS:
        if fmt not in diffs or not diffs[fmt]:
            out["formats"][fmt] = dict(skipped=f"no builder composition data for {fmt}")
            continue
        g = R.load_games(set_code, fmt)
        if g.height < MIN_GAMES:
            out["formats"][fmt] = dict(n_games=g.height, skipped=f"only {g.height} games (minimum {MIN_GAMES})")
            continue
        if dm is None:
            dm = deck_metrics(set_code)
        g = g.join(dm, on="deck_key", how="inner")
        out["formats"][fmt] = R._py(analyze(g, diffs[fmt]))
    return out


def write(res: dict, outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"builder_check_{res['set']}.json"
    p.write_text(json.dumps(R._py(res), indent=1), encoding="utf-8")
    return p
