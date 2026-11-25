"""The "Set report": plain-language facts about one set and Sealed format, computed from the corpus and the deployed model.

    compute_report(set_code, fmt) -> dict      (JSON-serialisable; read-only on the corpus and the model)
    get_report(set_code, fmt)     -> dict      (the same, cached in the set's cache folder, keyed by the model's built-at stamp)

Sections (each comes with a note on whether the deck builder uses it):
  glance : median game length, win rate on the play vs on the draw, mulligan rate, decks / games.
  pairs  : share of decks per main-color pair and the SKILL-ADJUSTED win rate of each, plus how often decks splash and whether it pays.
  cards  : where players misjudge cards (underrated / overrated) and the best commons / uncommons per color.

Honesty rules (the audience has no statistics background, so the report must not mislead):
  * Every win-rate figure is adjusted for pilot skill with the model's own per-game covariates (`design.CovEncoder`: the player's
    skill bin, on the play, mulligans, event, time, opponent colors) in a penalised logistic regression (`glm.fit`).  Without it "good
    players pick this" would read as "this is good".
  * Color-pair and splash effects are observed differences (`glm.cluster_se`: standard errors robust to the games of one event
    sharing a pilot and a pool).  They are in points of game win rate at 50 % (logit x 25) against an average deck.
  * Card values are the deployed model's per-copy effects (`beta_<fmt>`, `se_<fmt>` of the model's card table = SetModel.card_beta /
    card_se), in the same points, relative to the average PLAYED card of the same color group (weighted by the copies decks play).
  * Badges: Strong = the ~95 % interval is clear of zero by |z| >= 3, Likely = |z| >= 2, otherwise Weak.  A card is listed as
    underrated / overrated only with |z| >= 2.
  * Minimum samples (cards): at least MIN_ELIGIBLE decks that could have played the card and MIN_EXPOSURE card-games in the format
    (the model's `exposure_<fmt>`).  Between MIN_EXPOSURE and the model's own low-data line (THIN_EXPOSURE = 2000) the value leans more on
    the model's prior (Bo3 values are largely borrowed from Bo1 because Bo3 has few games): such cards stay listed, marked "limited
    data".  Color pairs: at least MIN_PAIR_DECKS decks.
  * The figures are observed; the deck builder uses ONLY the card values (color-pair and splash effects are shown, not used: it judges
    decks card by card).

Play rate of a card = among the Sealed decks whose pool contains the card AND whose main colors include all of the card's colors (any
deck for a colorless card or a land), the share that played at least one copy.  Each deck (a build of an event) counts once.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.special import expit

from sealed_lab import config
from sealed_lab.model import design, glm

log = logging.getLogger(__name__)

REPORT_VERSION = 7                 # bump to invalidate every cached report when the computation or the text changes
FORMATS = ("bo1", "bo3")
PT = 25.0                          # logit -> points of game win rate at 50 % (1 / 4 of a logit, in %)
MIN_ELIGIBLE = 150                 # decks that could have played a card (colors) before its play rate / value is shown
MIN_PAIR_DECKS = 40                # decks in a color group before its effect is shown
MIN_EXPOSURE = 500.0               # card-games of the format before a card's value is shown at all
THIN_EXPOSURE = 2000.0             # artifact.LOW_DATA_EXPOSURE: below it the value leans on the model's prior (flagged "limited data", still listed)
Z_STRONG, Z_LIKELY = 3.0, 2.0
UNDER_PCT, OVER_PCT = 40.0, 60.0   # play-rate percentile (among shown cards) below / above which a card counts as under- / over-played
LIST_N, LIST_MAX = 3, 5            # underrated / overrated cards per color group: 3, up to 5 when the extra ones are Strong
TOP_N = 5                          # cards per color in the top lists
TRAP_MAX = 5                       # trap rares listed (most negative first)
TRAP_RARITIES = ("rare", "mythic")
MIN_SPLASH_ELIG = 50               # off-color pools needed before a card's splash rate is shown
SPLASH_RATE_MIN = 0.08             # off-color splash rate needed to highlight a card as splashed
SPLASH_LIST_MAX = 6                # splashed bombs / splash traps listed
COLORS = "WUBRG"
COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
GROUP_LABELS = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green", "gold": "Multicolor",
                "colorless": "Colorless", "land": "Lands"}

BUILDER_USES = {
    "glance": "Background facts only: the builder does not use them.",
    "pairs": "Observed only: the builder judges decks card by card, it does not use color-pair or splash win rates.",
    "cards": "Yes: the card values are what the builder uses to pick your 40.",
    "archetypes": "Observed only: the builder values each card the same in every archetype (tested; archetype-specific values did not improve predictions).",
    "shape": "Descriptions only: for reference, the builder handles curve and land count through the model.",
}

Progress = Callable[[float, str], None]


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------

def badge_of(z: float | None) -> str:
    """"strong" | "likely" | "weak" from the z-score (effect / standard error)."""
    if z is None or not np.isfinite(z):
        return "weak"
    a = abs(z)
    return "strong" if a >= Z_STRONG else "likely" if a >= Z_LIKELY else "weak"


def signed(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}"


def range_text(eff: float, se: float, unit: str = "points") -> str:
    """Plain words for a ~95 % interval: "likely between +0.4 and +1.9 points"."""
    lo, hi = eff - 1.96 * se, eff + 1.96 * se
    return f"likely between {signed(lo)} and {signed(hi)} {unit}"


def color_key(colors: str | None) -> str:
    return "".join(c for c in COLORS if c in (colors or ""))


def pair_group(main: str | None) -> str | None:
    """Group of a deck by its main colors: one of the 10 pairs ("WU"...), "mono" (one color), "tri" (three or more), None (unknown)."""
    k = color_key(main)
    if not k:
        return None
    return "mono" if len(k) == 1 else k if len(k) == 2 else "tri"


def group_label(key: str) -> str:
    if key == "mono":
        return "One main color"
    if key == "tri":
        return "Three or more main colors"
    return "-".join(COLOR_NAMES[c] for c in key)


def card_group(colors: str | None, is_land: bool) -> str:
    k = color_key(colors)
    if is_land:
        return "land"
    return "colorless" if not k else k if len(k) == 1 else "gold"


def _py(x):
    """numpy -> plain python (JSON)."""
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------

GAME_COLS = ["fmt", "event", "draft_id", "deck_key", "match_number", "game_number", "game_time", "main_colors", "splash_colors", "on_play",
             "num_mulligans", "opp_num_mulligans", "opp_colors", "num_turns", "won", "skill"]


def load_games(set_code: str, fmt: str) -> pl.DataFrame:
    """Games of the format played with a 40-card deck (the model's own population)."""
    cd = config.corpus_dir(set_code)
    d40 = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards"]).filter(pl.col("n_cards") == 40).select("deck_key")
    g = (pl.read_parquet(cd / "games.parquet", columns=GAME_COLS).filter(pl.col("fmt") == fmt)
         .join(d40, on="deck_key", how="semi"))
    return g.with_columns(pl.col("main_colors").fill_null(""), pl.col("splash_colors").fill_null(""))


def _fit(g: pl.DataFrame, extra: sp.csr_matrix | None, extra_sigma: np.ndarray | None, contrasts_fn):
    """Penalised logistic regression of `won` on the model's pilot covariates (+ `extra` columns).  Returns
    (coef, enc, prob, se_robust of each contrast row, eta)."""
    enc = design.CovEncoder.fit(g)
    C = sp.csr_matrix(enc.transform(g))
    sig = enc.sigma
    if extra is not None:
        C = sp.hstack([C, extra], format="csr")
        sig = np.concatenate([sig, extra_sigma])
    y = g["won"].cast(pl.Float64).to_numpy()
    prob = glm.Problem(y=y, C=C)
    res = glm.fit(prob, np.zeros(len(sig)), sig, maxiter=500)
    grp = np.unique(g["draft_id"].to_numpy(), return_inverse=True)[1].astype(np.int64)
    L = contrasts_fn(enc.k, C.shape[1])
    _lap, _rob, _cl, c_rob = glm.cluster_se(prob, res.coef, sig, grp, contrasts=L)
    return res.coef, enc, prob, c_rob, prob.eta(res.coef)


# ---------------------------------------------------------------------------------------------
# section 1: set at a glance
# ---------------------------------------------------------------------------------------------

def glance_section(g: pl.DataFrame, pools_n: int) -> dict:
    n_games = g.height
    decks = g.select("draft_id", "deck_key").unique()
    out: dict = dict(n_games=n_games, n_decks=decks.height, n_drafts=int(g["draft_id"].n_unique()), n_pools=pools_n,
                     median_turns=float(g["num_turns"].median()) if n_games else None,
                     mulligan_rate=float((g["num_mulligans"].fill_null(0) > 0).mean()) if n_games else None)
    # play / draw: game 1 only (in Bo3 the loser of a game chooses, which would mix skill into "on the play"); adjusted for skill anyway
    g1 = g.filter(pl.col("game_number").fill_null(1) <= 1)
    out["play_n_games"] = g1.height
    out["on_play"] = play_effect(g1)
    return out


def play_effect(g1: pl.DataFrame) -> dict | None:
    """Skill-adjusted win rate on the play vs on the draw among first games `g1` (None below 300 games)."""
    if g1.height < 300:
        return None
    op = g1["on_play"].fill_null(False).cast(pl.Float64).to_numpy()
    coef, enc, prob, c_rob, eta = _fit(g1, None, None, lambda k, ncol: np.eye(ncol)[[1]])
    c = float(coef[1])
    e0 = eta - c * op
    p1, p0 = expit(e0 + c), expit(e0)
    wr_play, wr_draw = float(p1.mean()), float(p0.mean())
    diff = 100.0 * (wr_play - wr_draw)
    se = float(100.0 * (p1 * (1 - p1) + p0 * (1 - p0)).mean() / 2.0 * c_rob[0])
    z = diff / se if se > 0 else None
    return dict(wr_play=wr_play, wr_draw=wr_draw, diff_pp=diff, se_pp=se, z=z, badge=badge_of(z), range=range_text(diff, se))


def glance_text(gl: dict, fmt: str) -> list[str]:
    """Plain sentences for the section."""
    out = []
    if gl.get("median_turns") is not None:
        out.append(f"A typical game lasts {gl['median_turns']:.0f} turns (as counted by 17lands).")
    op = gl.get("on_play")
    if op:
        d = op["diff_pp"]
        if op["badge"] == "weak":
            out.append(f"Playing first or second makes no clear difference ({signed(d)} points for the player on the play).")
        else:
            who = "first" if d > 0 else "second"
            out.append(f"Going {who} is better: about {abs(d):.1f} points of win rate "
                       f"({op['wr_play'] * 100:.1f}% on the play against {op['wr_draw'] * 100:.1f}% on the draw).")
    mr = gl.get("mulligan_rate")
    if mr is not None:
        out.append(f"About {mr * 100:.0f}% of games start with a mulligan" + (" (the best-of-three hand smoother is off, so more than in Bo1)." if fmt == "bo3" and mr > 0.15 else "."))
    return out


# ---------------------------------------------------------------------------------------------
# section 2: color pairs and splash
# ---------------------------------------------------------------------------------------------

def pairs_section(g: pl.DataFrame) -> dict:
    g = g.with_columns(pl.col("main_colors").map_elements(pair_group, return_dtype=pl.String).alias("grp"),
                       (pl.col("splash_colors").str.len_chars() > 0).alias("splash"))
    g = g.filter(pl.col("grp").is_not_null())
    n_games = g.height
    decks = g.group_by("draft_id", "deck_key").agg(pl.col("grp").first(), pl.col("splash").first(), pl.len().alias("games"))
    n_decks = decks.height
    per = decks.group_by("grp").agg(pl.len().alias("n_decks"), pl.col("splash").mean().alias("splash_share"), pl.col("games").sum().alias("n_games"))
    per = per.sort("n_decks", descending=True)
    sec: dict = dict(n_decks=n_decks, n_games=n_games, min_decks=MIN_PAIR_DECKS, rows=[], splash=None)
    if n_games < 500 or per.height < 2:
        sec["too_few"] = True
        return sec
    keys = per["grp"].to_list()
    ref = keys[0]                                        # the most common group is the reference level (its dummy is dropped)
    others = keys[1:]
    ix = {k: i for i, k in enumerate(others)}
    grp = g["grp"].to_numpy()
    rows_i = np.nonzero(np.isin(grp, others))[0]
    cols_i = np.array([ix[k] for k in grp[rows_i]], dtype=np.int64)
    D = sp.csr_matrix((np.ones(len(rows_i)), (rows_i, cols_i)), shape=(n_games, len(others)))
    spl = sp.csr_matrix(g["splash"].cast(pl.Float64).to_numpy()[:, None])
    extra = sp.hstack([D, spl], format="csr")
    w = {k: float(n) / n_games for k, n in zip(keys, per["n_games"].to_list())}          # game share of each group = the "average deck"

    def contrasts(k: int, ncol: int) -> np.ndarray:
        L = np.zeros((len(keys) + 1, ncol))
        for r_, key in enumerate(keys):
            if key != ref:
                L[r_, k + ix[key]] += 1.0
            for h in others:
                L[r_, k + ix[h]] -= w[h]
        L[len(keys), ncol - 1] = 1.0                    # the splash coefficient itself
        return L

    coef, enc, prob, c_rob, _eta = _fit(g, extra, np.full(len(others) + 1, 10.0), contrasts)
    k = enc.k
    b = {ref: 0.0, **{key: float(coef[k + ix[key]]) for key in others}}
    mean_b = sum(w[key] * b[key] for key in keys)
    rows = []
    for r_, key in enumerate(keys):
        nd = int(per["n_decks"][r_])
        eff = PT * (b[key] - mean_b)
        se = PT * float(c_rob[r_])
        z = eff / se if se > 0 else None
        rows.append(dict(key=key, label=group_label(key), colors=key if len(key) == 2 else "", share=nd / n_decks, n_decks=nd,
                         n_games=int(per["n_games"][r_]), splash_share=float(per["splash_share"][r_]),
                         effect_pp=eff, se_pp=se, z=z, badge=badge_of(z), range=range_text(eff, se), hidden=nd < MIN_PAIR_DECKS))
    sec["rows"] = rows
    seff, sse = PT * float(coef[-1]), PT * float(c_rob[len(keys)])
    sz = seff / sse if sse > 0 else None
    sec["splash"] = dict(share=float(decks["splash"].mean()), n_decks=int(decks["splash"].sum()), effect_pp=seff, se_pp=sse, z=sz,
                         badge=badge_of(sz), range=range_text(seff, sse))
    return sec


def pairs_text(sec: dict) -> list[str]:
    out = []
    rows = [r for r in sec.get("rows", []) if not r["hidden"]]
    if not rows:
        return out
    clear = [r for r in rows if r["badge"] != "weak"]
    best = max(clear, key=lambda r: r["effect_pp"], default=None)
    worst = min(clear, key=lambda r: r["effect_pp"], default=None)
    if best is not None and best["effect_pp"] > 0:
        out.append(f"{best['label']} decks do best: {signed(best['effect_pp'])} points of win rate against an average deck, "
                   f"for players of the same skill.")
    if worst is not None and worst is not best and worst["effect_pp"] < 0:
        out.append(f"{worst['label']} decks do worst: {signed(worst['effect_pp'])} points.")
    if best is None:
        out.append("No color pair clearly beats the others once pilot skill is taken into account.")
    sp_ = sec.get("splash")
    if sp_:
        pay = ("pays off" if sp_["effect_pp"] > 0 else "costs") if sp_["badge"] != "weak" else "makes no clear difference"
        tail = f": {signed(sp_['effect_pp'])} points" if sp_["badge"] != "weak" else ""
        out.append(f"{sp_['share'] * 100:.0f}% of decks splash a third color; splashing {pay}{tail}.")
    return out


# ---------------------------------------------------------------------------------------------
# section 3: cards players misjudge
# ---------------------------------------------------------------------------------------------

def card_play_stats(set_code: str, fmt: str, g: pl.DataFrame, cards: pl.DataFrame) -> pl.DataFrame:
    """Per card: n_elig (decks that had it in the pool and in main colors), n_played (>= 1 copy), copies (total copies played
    in the format), plus splash stats: n_splash_elig (decks whose main colors are missing exactly 1 color of the card) and
    n_splashed (how many of those decks played >= 1 copy anyway)."""
    cd = config.corpus_dir(set_code)
    bits = {c: 1 << i for i, c in enumerate(COLORS)}

    def mask(s: str | None) -> int:
        return sum(bits[c] for c in color_key(s))

    builds = (g.group_by("draft_id", "deck_key").agg(pl.col("main_colors").first())
              .with_columns(pl.col("main_colors").map_elements(mask, return_dtype=pl.Int64).alias("mmask")))
    cinfo = cards.select("card_id", pl.col("colors").map_elements(mask, return_dtype=pl.Int64).alias("cmask"))
    pool_builds = (pl.read_parquet(cd / "pools.parquet", columns=["draft_id", "card_id"])
                   .join(builds.select("draft_id", "deck_key", "mmask"), on="draft_id")
                   .join(cinfo, on="card_id"))
    dc = pl.read_parquet(cd / "deck_cards.parquet", columns=["deck_key", "card_id", "n"]).join(builds.select("deck_key").unique(), on="deck_key")
    pools_main = pool_builds.filter((pl.col("cmask") & pl.col("mmask")) == pl.col("cmask"))
    el = (pools_main.join(dc, on=["deck_key", "card_id"], how="left")
          .group_by("card_id").agg(pl.len().alias("n_elig"), (pl.col("n").fill_null(0) > 0).sum().alias("n_played")))
    # Off-color splash eligibility: card has >= 1 color not in main_colors, and missing bits is a single power of two (1 off-color)
    miss = pl.col("cmask") & (~pl.col("mmask"))
    pools_splash = pool_builds.filter((miss > 0) & ((miss & (miss - 1)) == 0) & (pl.col("mmask") > 0))
    sp_el = (pools_splash.join(dc, on=["deck_key", "card_id"], how="left")
             .group_by("card_id").agg(pl.len().alias("n_splash_elig"), (pl.col("n").fill_null(0) > 0).sum().alias("n_splashed")))
    cp = dc.group_by("card_id").agg(pl.col("n").sum().alias("copies"))
    return (cards.select("card_id").join(el, on="card_id", how="left").join(sp_el, on="card_id", how="left").join(cp, on="card_id", how="left")
            .with_columns(pl.col("n_elig", "n_played", "n_splash_elig", "n_splashed", "copies").fill_null(0)))


def _group_refs(s: pl.DataFrame, beta: str, copies: str) -> dict[str, float]:
    """Value of the average PLAYED card of each color group: `beta` weighted by the `copies` decks play."""
    ref: dict[str, float] = {}
    for grp, part in s.filter(pl.col(beta).is_finite() & (pl.col(copies) > 0)).group_by("grp"):
        w = part[copies].to_numpy().astype(float)
        ref[grp[0]] = float(np.dot(w, part[beta].to_numpy()) / w.sum())
    return ref


EXPLORER_COLS = ["id", "name", "group", "rarity", "play_rate", "value_pp", "z", "n_elig", "thin", "value_other", "thin_other", "dz",
                 "splash_rate", "n_splash_elig"]
COMPARE_N = 8                      # biggest clear Bo1-vs-Bo3 card differences listed


def explorer_rows(shown: pl.DataFrame, fmt: str, ref_o: dict[str, float], has_o: bool) -> tuple[dict, list[dict]]:
    """Every card with enough data as a compact row (see EXPLORER_COLS; numbers rounded), and the biggest clear Bo1-vs-Bo3 differences.
    value_other = the card's value in the other format (same points, against that format's average played card of the group); dz = z of
    (Bo3 value - Bo1 value) with the two standard errors added as if independent (conservative: they are positively linked); a difference is
    called clear only with |dz| >= 2 and no 'limited data' flag in either format."""
    rows, cmp = [], []
    for r in shown.iter_rows(named=True):
        vo = dz = None
        thin_o = 0
        if has_o and r["exposure_o"] >= MIN_EXPOSURE and np.isfinite(r["beta_o"]) and r["se_o"] > 0:
            vo = (r["beta_o"] - ref_o.get(r["grp"], 0.0)) * PT
            thin_o = int(r["exposure_o"] < THIN_EXPOSURE)
            d = (vo - r["value_pp"]) if fmt == "bo1" else (r["value_pp"] - vo)                  # Bo3 - Bo1
            dz = d / (PT * float(np.hypot(r["se"], r["se_o"])))
        nse = int(r.get("n_splash_elig") or 0)
        nsp = int(r.get("n_splashed") or 0)
        sr = round(float(nsp / nse), 3) if nse >= MIN_SPLASH_ELIG else None
        rows.append([int(r["card_id"]), r["name"], r["grp"], r["rarity"], round(float(r["play_rate"]), 3), round(float(r["value_pp"]), 2),
                     round(float(r["z"]), 1), int(r["n_elig"]), int(bool(r["thin"])), None if vo is None else round(float(vo), 2), thin_o,
                     None if dz is None else round(float(dz), 1), sr, nse if sr is not None else 0])
        if dz is not None and abs(dz) >= Z_LIKELY and not r["thin"] and not thin_o:
            v1, v3 = (r["value_pp"], vo) if fmt == "bo1" else (vo, r["value_pp"])
            cmp.append(dict(card_id=int(r["card_id"]), name=r["name"], group=r["grp"], rarity=r["rarity"], v_bo1=round(float(v1), 2),
                            v_bo3=round(float(v3), 2), diff_pp=round(float(v3 - v1), 2), dz=round(float(dz), 1), badge=badge_of(dz)))
    cmp.sort(key=lambda c: -abs(c["diff_pp"]))
    rows.sort(key=lambda x: x[1])
    return dict(cols=EXPLORER_COLS, rows=rows), cmp[:COMPARE_N]


def build_card_lists(stats: pl.DataFrame, fmt: str, keep_rows: bool = False) -> dict:
    """The card lists from a table with one row per card: card_id, name, rarity, colors, is_land, is_basic, beta, se, exposure,
    n_elig, n_played, copies (and, to compare the two formats, the same measures of the OTHER format as beta_o, se_o, exposure_o, copies_o).
    (Separate from the corpus code so that the fake backend can feed it synthetic rows.)"""
    s = stats.filter(~pl.col("is_basic")).with_columns(
        pl.struct("colors", "is_land").map_elements(lambda r: card_group(r["colors"], bool(r["is_land"])), return_dtype=pl.String).alias("grp"))
    ok = (pl.col("n_elig") >= MIN_ELIGIBLE) & (pl.col("exposure") >= MIN_EXPOSURE) & pl.col("beta").is_finite() & (pl.col("se") > 0)
    n_all = s.height
    shown = s.filter(ok)
    # average PLAYED card of each color group: beta weighted by the copies decks play (all cards of the group with a value)
    ref = _group_refs(s, "beta", "copies")
    has_o = "beta_o" in s.columns
    ref_o = _group_refs(s, "beta_o", "copies_o") if has_o else {}
    if shown.height:
        shown = shown.with_columns(pl.col("grp").replace_strict(ref, default=0.0, return_dtype=pl.Float64).alias("ref"))
        shown = shown.with_columns(((pl.col("beta") - pl.col("ref")) * PT).alias("value_pp"), (pl.col("se") * PT).alias("se_pp"),
                                   (pl.col("n_played") / pl.col("n_elig")).alias("play_rate"),
                                   (pl.col("exposure") < THIN_EXPOSURE).alias("thin"))
        shown = shown.with_columns((pl.col("value_pp") / pl.col("se_pp")).alias("z"))
    out: dict = dict(n_cards=n_all, n_shown=shown.height, n_hidden=n_all - shown.height, min_eligible=MIN_ELIGIBLE,
                     min_exposure=int(MIN_EXPOSURE), underrated=[], overrated=[], traps=[], splashed=[], splash_traps=[],
                     top={}, play_rate_median=None, explorer=dict(cols=EXPLORER_COLS, rows=[]), compare_cards=[])
    if shown.height < 10:
        return out
    rates = shown["play_rate"].to_numpy()
    lo_cut, hi_cut = float(np.percentile(rates, UNDER_PCT)), float(np.percentile(rates, OVER_PCT))
    out["play_rate_median"] = float(np.median(rates))
    out["play_rate_cuts"] = [lo_cut, hi_cut]

    def row(r: dict) -> dict:
        z = r["z"]
        nse = int(r.get("n_splash_elig") or 0)
        nsp = int(r.get("n_splashed") or 0)
        d = dict(card_id=int(r["card_id"]), name=r["name"], rarity=r["rarity"], colors=r["colors"] or "", group=r["grp"], thin=bool(r["thin"]),
                 play_rate=float(r["play_rate"]), n_elig=int(r["n_elig"]), value_pp=float(r["value_pp"]), se_pp=float(r["se_pp"]),
                 z=float(z), badge=badge_of(z), range=range_text(r["value_pp"], r["se_pp"]))
        if nse > 0:
            d["n_splash_elig"] = nse
            d["splash_rate"] = float(nsp / nse)
        return d

    rows = [row(r) for r in shown.iter_rows(named=True)]
    out["explorer"], out["compare_cards"] = explorer_rows(shown, fmt, ref_o, has_o)
    if keep_rows:
        out["_rows"] = rows                                 # (for the archetype section; removed before the report is stored)
    def typical(r: dict) -> str:
        lab = GROUP_LABELS.get(r["group"], r["group"]).lower()
        return f"the average {lab} {'one' if r['group'] == 'land' else 'card'} decks play"

    def why(r: dict, under: bool) -> str:
        pr = f"{r['play_rate'] * 100:.0f}%"
        if under:
            return f"Played in only {pr} of the decks that could play it, yet each copy adds {signed(r['value_pp'])} points of win rate more than {typical(r)}."
        return f"Played in {pr} of the decks that could play it, yet each copy is worth {signed(r['value_pp'])} points of win rate against {typical(r)}."

    under = [dict(r, reason=why(r, True)) for r in rows if r["play_rate"] <= lo_cut and r["z"] >= Z_LIKELY]
    over = [dict(r, reason=why(r, False)) for r in rows if r["play_rate"] >= hi_cut and r["z"] <= -Z_LIKELY]
    # per group (owner): W U B R G, multicolor, artifacts & colorless (lands included); LIST_N each, up to LIST_MAX when the extra
    # cards are Strong.  The flat lists keep that group order.
    for key, items, sign in (("underrated", under, -1), ("overrated", over, 1)):
        by: dict[str, list] = {}
        for gk in list(COLORS) + ["gold", "colorless"]:
            part = sorted((r for r in items if (r["group"] if r["group"] != "land" else "colorless") == gk), key=lambda r: sign * r["value_pp"])
            keep = part[:LIST_N] + [r for r in part[LIST_N:LIST_MAX] if r["badge"] == "strong"]
            if keep:
                by[gk] = keep
        out[key + "_by"] = by
        out[key] = [r for gk in by for r in by[gk]]
    out["n_underrated"], out["n_overrated"] = len(under), len(over)
    # trap rares: a rare / mythic is auto-included by most players, so a clearly negative one is the costliest Sealed mistake
    traps = sorted((r for r in over if r["rarity"] in TRAP_RARITIES and r["value_pp"] < 0 and r["badge"] in ("likely", "strong")), key=lambda r: r["value_pp"])
    out["traps"] = [dict(r, reason=f"A {r['rarity']} that {r['play_rate'] * 100:.0f}% of the decks that could play it do play, yet each copy loses "
                                   f"{abs(r['value_pp']):.1f} points of win rate against {typical(r)}.") for r in traps[:TRAP_MAX]]
    # off-color splashable bombs and splash traps (among pools missing exactly one of the card's colors)
    sp_cand = [r for r in rows if r.get("n_splash_elig", 0) >= MIN_SPLASH_ELIG and (r.get("splash_rate") or 0.0) >= SPLASH_RATE_MIN]
    sp_good = sorted((r for r in sp_cand if r["value_pp"] > 0 and r["badge"] in ("likely", "strong")), key=lambda r: -r["value_pp"])
    sp_bad = sorted((r for r in sp_cand if r["value_pp"] < 0 and r["badge"] in ("likely", "strong")), key=lambda r: r["value_pp"])
    out["splashed"] = [dict(r, reason=f"Splashed in {r['splash_rate'] * 100:.0f}% of the {r['n_splash_elig']:,} decks missing one of its colors; "
                                      f"across all the decks that play it, each copy adds {signed(r['value_pp'])} points of win rate against {typical(r)} "
                                      f"(not measured for splashed copies alone).")
                       for r in sp_good[:SPLASH_LIST_MAX]]
    out["splash_traps"] = [dict(r, reason=f"Splashed in {r['splash_rate'] * 100:.0f}% of the {r['n_splash_elig']:,} decks missing one of its colors, "
                                          f"yet across all the decks that play it each copy loses {abs(r['value_pp']):.1f} points of win rate against {typical(r)} "
                                          f"(not measured for splashed copies alone).")
                           for r in sp_bad[:4]]
    top: dict[str, dict[str, list]] = {}
    for rar in ("common", "uncommon"):
        by: dict[str, list] = {}
        for gk in list(COLORS) + ["gold", "colorless", "land"]:
            part = sorted((r for r in rows if r["rarity"] == rar and r["group"] == gk), key=lambda r: -r["value_pp"])[:TOP_N]
            if part:
                by[gk] = [dict(r, reason=f"{signed(r['value_pp'])} points of win rate per copy against {typical(r)}; "
                                         f"played in {r['play_rate'] * 100:.0f}% of the decks that could.") for r in part]
        top[rar] = by
    out["top"] = top
    out["thin_share"] = float(shown["thin"].mean())
    if out["thin_share"] >= 0.5:
        out["note"] = ("Many of these cards have little data in this format, so their values lean on the model's prior "
                       + ("(best-of-three values are largely borrowed from best-of-one)." if fmt == "bo3" else "(mostly from the draft results)."))
    return out


def cards_section(set_code: str, fmt: str, g: pl.DataFrame, model, keep_rows: bool = False, g_other: pl.DataFrame | None = None) -> dict:
    cards = model.cards.sort("card_id")
    stats = card_play_stats(set_code, fmt, g, cards)
    tab = cards.select("card_id", "name", "rarity", "colors", "is_land", "is_basic").with_columns(
        pl.Series("beta", model.card_beta(fmt).astype(float)), pl.Series("se", model.card_se(fmt).astype(float)),
        pl.Series("exposure", cards[f"exposure_{fmt}"].to_numpy().astype(float)))
    other = "bo3" if fmt == "bo1" else "bo1"
    with_o = g_other is not None and other in model.formats and g_other.height >= 300
    if with_o:
        tab = tab.with_columns(pl.Series("beta_o", model.card_beta(other).astype(float)), pl.Series("se_o", model.card_se(other).astype(float)),
                               pl.Series("exposure_o", cards[f"exposure_{other}"].to_numpy().astype(float)))
    tab = tab.join(stats, on="card_id")
    if with_o:
        tab = tab.join(card_play_stats(set_code, other, g_other, cards).select("card_id", pl.col("copies").alias("copies_o")), on="card_id")
    return build_card_lists(tab, fmt, keep_rows=keep_rows)


# ---------------------------------------------------------------------------------------------
# the whole report
# ---------------------------------------------------------------------------------------------

class CorpusMismatch(RuntimeError):
    """The corpus was rebuilt after the model (an interrupted / later calculation) and numbers the cards differently."""


def corpus_stamp(set_code: str) -> str:
    """`built_at` of the set's corpus manifest ("" when there is none)."""
    try:
        d = json.loads((config.corpus_dir(set_code.upper()) / "manifest.json").read_text(encoding="utf-8"))
        return str(d.get("built_at") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def model_stamp(model_or_meta) -> str:
    """Cache stamp of a report: the model's build time, the corpus it reads now and the report version (a corpus rebuilt after the
    model therefore never serves a report computed from the old one).  The corpus part is added only when the corpus differs from the
    one the model was built on, so stamps of consistent sets keep their old form (cached reports and written-report stamps stay valid)."""
    meta = getattr(model_or_meta, "meta", model_or_meta)
    cs = corpus_stamp(meta["set_code"]) if meta.get("set_code") else ""
    built = meta.get("built_corpus_at") or meta.get("corpus_built_at") or ""
    extra = f"|c{cs}" if cs and built and cs != built else ""
    return f"{meta.get('built_at') or ''}{extra}|v{REPORT_VERSION}"


def check_corpus_matches(set_code: str, model) -> None:
    """Refuse to compute from a corpus that numbers the cards differently from the model (the card ids of the corpus tables would
    value the wrong cards); only warn when the corpus is newer but the ids are the same."""
    meta = getattr(model, "meta", {}) or {}
    built = meta.get("built_corpus_at") or meta.get("corpus_built_at")
    cur = corpus_stamp(set_code)
    if not built or not cur or built == cur:
        return
    from sealed_lab.deck.explain import _corpus_to_model_ids

    if _corpus_to_model_ids(model, config.corpus_dir(set_code.upper())) is not None:
        raise CorpusMismatch(f"{set_code.upper()}: the data tables were rebuilt after the model and number the cards differently "
                             "(an interrupted or later calculation): calculate the set again")
    log.warning("%s: the corpus (%s) is newer than the model's (%s); card ids agree, continuing", set_code, cur, built)


def compute_report(set_code: str, fmt: str, model=None, progress: Progress | None = None, cancel=None) -> dict:
    """The full report of a set and Sealed format ("bo1" | "bo3") as a JSON-serialisable dict.  Reads the corpus and the model only."""
    set_code = set_code.upper()
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    if model is None:
        from sealed_lab.model.artifact import SetModel

        model = SetModel.load(set_code)
    if fmt not in model.formats:
        raise ValueError(f"{set_code} has no {fmt} model")
    check_corpus_matches(set_code, model)

    def prog(f: float, m: str) -> None:
        if cancel is not None and cancel.is_set():
            from sealed_lab.types import Cancelled

            raise Cancelled("report cancelled")
        if progress is not None:
            progress(f, m)

    t0 = time.time()
    prog(0.02, "Reading games")
    g = load_games(set_code, fmt)
    pools_n = int(pl.read_parquet(config.corpus_dir(set_code) / "pools.parquet", columns=["draft_id"])["draft_id"].n_unique())
    out: dict = dict(version=REPORT_VERSION, set_code=set_code, fmt=fmt, built_at=model.meta.get("built_at"), stamp=model_stamp(model),
                     computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), notes=dict(BUILDER_USES))
    if g.height < 300:
        out.update(glance=dict(n_games=g.height, n_decks=0), pairs=dict(rows=[], too_few=True), cards=dict(underrated=[], overrated=[], traps=[], top={}),
                   archetypes=dict(pairs=[]), shape=dict(rows=[], metrics=[]), pair_details=dict(rows=[]), evolution=dict(too_few=True),
                   compare=None, data_notes={},
                   too_few=True, seconds=time.time() - t0)
        return out
    prog(0.08, "At a glance")
    out["glance"] = glance_section(g, pools_n)
    out["glance"]["text"] = glance_text(out["glance"], fmt)
    prog(0.35, "Color pairs (adjusted for pilot skill)")
    out["pairs"] = pairs_section(g)
    out["pairs"]["text"] = pairs_text(out["pairs"])
    prog(0.5, "Cards players misjudge")
    other = "bo3" if fmt == "bo1" else "bo1"
    g_other = load_games(set_code, other) if other in model.formats else None
    out["cards"] = cards_section(set_code, fmt, g, model, keep_rows=True, g_other=g_other)
    card_rows = out["cards"].pop("_rows", [])
    from sealed_lab import report_arch, report_detail

    prog(0.58, "Details per color pair")
    out["pair_details"] = report_detail.pair_details(g, out["pairs"])

    prog(0.65, "Archetypes and deck shape")
    out["archetypes"], out["shape"] = report_arch.archetypes_section(
        set_code, fmt, g, model, out["pairs"], card_rows, progress=lambda f, m: prog(0.65 + 0.3 * f, m))
    prog(0.95, "How the format changed")
    out["evolution"] = report_detail.evolution_section(set_code, fmt, g, model.cards.sort("card_id"), card_rows)
    out["compare"] = None
    if g_other is not None and g_other.height >= 300:
        try:
            out["compare"] = report_detail.compare_pairs(fmt, out["pairs"], pairs_section(g_other))
            out["compare"]["cards"] = out["cards"].get("compare_cards", [])
        except Exception:
            log.info("format comparison failed", exc_info=True)
    out["data_notes"] = report_detail.data_notes(g, out["glance"], fmt)
    out["seconds"] = time.time() - t0
    prog(1.0, "Done")
    return _py(out)


# ---------------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------------

def cache_path(set_code: str, fmt: str) -> Path:
    return config.cache_dir(set_code) / f"report_{fmt}.json"


def read_cached(set_code: str, fmt: str, stamp: str) -> dict | None:
    p = cache_path(set_code, fmt)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("stamp") == stamp and d.get("set_code") == set_code.upper() and d.get("fmt") == fmt else None


def write_cache(rep: dict) -> None:
    p = cache_path(rep["set_code"], rep["fmt"])
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        log.info("report cache for %s %s could not be written", rep.get("set_code"), rep.get("fmt"), exc_info=True)


def get_report(set_code: str, fmt: str, model=None, refresh: bool = False, progress: Progress | None = None, cancel=None) -> dict:
    """The cached report when its stamp (model built-at + report version) still matches, else computed and cached."""
    set_code = set_code.upper()
    if model is None:
        from sealed_lab.model.artifact import SetModel

        model = SetModel.load(set_code)
    stamp = model_stamp(model)
    if not refresh:
        hit = read_cached(set_code, fmt, stamp)
        if hit is not None:
            hit["cached"] = True
            return hit
    rep = compute_report(set_code, fmt, model=model, progress=progress, cancel=cancel)
    write_cache(rep)
    rep["cached"] = False
    return rep


# ---------------------------------------------------------------------------------------------
# text summary (CLI)
# ---------------------------------------------------------------------------------------------

def summary_text(rep: dict) -> str:
    L = [f"Set report {rep['set_code']} {rep['fmt']} (model built {rep.get('built_at') or '?'})"]
    gl = rep.get("glance") or {}
    L.append(f"  {gl.get('n_decks', 0)} decks, {gl.get('n_games', 0)} games")
    for t in gl.get("text", []):
        L.append("  - " + t)
    L.append("  [" + rep["notes"]["glance"] + "]")
    pr = rep.get("pairs") or {}
    L.append("Color pairs (skill-adjusted, points of game win rate vs an average deck):")
    for r in sorted(pr.get("rows", []), key=lambda r: -r["effect_pp"]):
        if r["hidden"]:
            continue
        L.append(f"  {r['label']:30s} {r['share'] * 100:5.1f}% of decks  {signed(r['effect_pp'], 2):>6s}  [{r['badge']}]  splash {r['splash_share'] * 100:3.0f}%")
    for t in pr.get("text", []):
        L.append("  - " + t)
    L.append("  [" + rep["notes"]["pairs"] + "]")
    cs = rep.get("cards") or {}
    L.append(f"Cards ({cs.get('n_shown', 0)} of {cs.get('n_cards', 0)} have enough data: >= {cs.get('min_eligible')} decks and >= {cs.get('min_exposure')} card-games):")
    for title, key in (("Underrated", "underrated"), ("Overrated", "overrated")):
        L.append(f"  {title}:")
        for r in cs.get(key, [])[:8]:
            L.append(f"    {r['name']:34s} played {r['play_rate'] * 100:3.0f}%  value {signed(r['value_pp'], 2):>6s}  [{r['badge']}]")
    L.append("  [" + rep["notes"]["cards"] + "]")
    ar = rep.get("archetypes") or {}
    if ar.get("pairs"):
        L.append(f"Archetypes (color pairs; sub-archetypes only where they replicate in two halves; {ar.get('n_with_subs', 0)} pairs split):")
        for a in ar["pairs"]:
            L.append(f"  {a['code']:5s} {a['label']:18s} {a['share'] * 100:5.1f}% {signed(a['effect_pp'], 2):>6s} [{a['badge']}]  splash {a['splash_share'] * 100:3.0f}%  "
                     + ", ".join(s["name"] for s in a["signature"]))
            for s in a["subs"]:
                L.append(f"      - {s['name']:34s} {s['share_of_pair'] * 100:4.0f}% of pair  {signed(s.get('effect_pp', 0.0), 2):>6s} [{s.get('badge', '?')}]  "
                         + ", ".join(c["name"] for c in s["signature"]) + f"  (ARI {a['split']['ari']:.2f})")
        L.append("  [" + rep["notes"]["archetypes"] + "]")
    sh = rep.get("shape") or {}
    for r in sh.get("rows", [])[:1]:
        L.append(f"Deck shape, {r['label']} ({r['n_decks']} decks, {r['n_strong']} with strong results = {sh.get('strong_def')}): middle half")
        for key, label in [(m["key"], m["label"]) for m in sh["metrics"]]:
            m = r["metrics"][key]
            L.append(f"  {label:20s} {m['all']['lo']:.1f}-{m['all']['hi']:.1f}" + (f"   strong: {m['strong']['lo']:.1f}-{m['strong']['hi']:.1f}" if m["strong"] else ""))
    return "\n".join(L)
