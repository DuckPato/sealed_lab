"""Draft TARGET formats of the model: `dbo1` (PremierDraft games) and `dbo3` (TradDraft games), docs/draft_mode/RULE.md.

They are fitted by the SAME machinery as the Sealed chain (draft -> bo1 -> bo3 of `cardvalue.run_chain`) on a relabelled copy of the
game table, so that the prior-strength CV, lambda, stacking, structure terms, bootstrap and the hold-out validation are literally the
Sealed code with different games in the stages:

    stage "draft"  <- the draft events other than PremierDraft / TradDraft (the prior of dbo1; zero prior when < MIN_PRIOR_GAMES games)
    stage "bo1"    <- PremierDraft games  = the `dbo1` format
    stage "bo3"    <- TradDraft games     = the `dbo3` format (its prior is the `dbo1` stage, game-number covariate as Sealed Bo3)

The Sealed chain never sees any of it: its own draft prior stage keeps selecting ALL draft events (`fmt == "draft"` of the real table).
`relabel` therefore works on a COPY of the table; the model's `games` frame is not touched.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import polars as pl

from sealed_lab import config

Progress = Callable[[float, str], None]

# draft format -> the stage slot it occupies in the (draft -> bo1 -> bo3) machinery
SLOT: dict[str, str] = {"dbo1": "bo1", "dbo3": "bo3"}
FORMAT_OF_SLOT: dict[str, str] = {v: k for k, v in SLOT.items()}
# slot name -> name in the meta / report ("draft" slot = the other draft events: the prior of dbo1)
STAGE_NAME: dict[str, str] = {"draft": "dprior", "bo1": "dbo1", "bo3": "dbo3"}


def relabel(games: pl.DataFrame, named: bool = False) -> pl.DataFrame:
    """The DRAFT games of `games` with `fmt` rewritten (Sealed games are dropped; the input is not modified).

    `named=False` (the chain slots): PremierDraft -> "bo1", TradDraft -> "bo3", every other draft event -> "draft".
    `named=True` (the formats' own names): PremierDraft -> "dbo1", TradDraft -> "dbo3", others stay "draft"."""
    prem, trad = config.DRAFT_FORMAT_EVENT["dbo1"], config.DRAFT_FORMAT_EVENT["dbo3"]
    a, b = ("dbo1", "dbo3") if named else ("bo1", "bo3")
    ev = pl.col("event")
    return (games.filter(pl.col("fmt") == "draft")
            .with_columns(pl.when(ev == prem).then(pl.lit(a)).when(ev == trad).then(pl.lit(b)).otherwise(pl.lit("draft")).alias("fmt")))


def present(games: pl.DataFrame) -> list[str]:
    """Draft target formats (in model order) that have at least one game (a set without TradDraft games has no `dbo3`)."""
    out = []
    for f in config.DRAFT_FORMATS:
        ev = config.DRAFT_FORMAT_EVENT[f]
        if games.filter((pl.col("fmt") == "draft") & (pl.col("event") == ev)).height >= 1:
            out.append(f)
    return out


def rename_validation(v: dict, f: str) -> dict:
    """A `validate.run_validation` / `validate_deployed` result of slot `SLOT[f]`, with the stage names the draft formats use."""
    v = dict(v)
    if "fmt" in v:
        v["fmt"] = f
    st = v.get("stages")
    if isinstance(st, dict):
        v["stages"] = {STAGE_NAME.get(k, k): s for k, s in st.items()}
    if v.get("prior_stage") in STAGE_NAME:
        v["prior_stage"] = STAGE_NAME[v["prior_stage"]]
    return v


def stage_summary_named(summary: dict) -> dict:
    """`validate.stage_summary` of a chain stage with `lam_from` in the draft formats' names."""
    s = dict(summary)
    if s.get("lam_from") in STAGE_NAME:
        s["lam_from"] = STAGE_NAME[s["lam_from"]]
    return s


def fit_final(set_code: str, games_d: pl.DataFrame, formats: list[str], space, tmp: str, n_jobs: int, seed: int, n_boot: int,
              cancel: "threading.Event | object | None", opts, prog: Progress, rn, memory, min_prior_games: int) -> dict:
    """Final fit of the draft chains on all data (CV of every stage, lambda, stacking, structure terms, then the event bootstrap).

    `games_d`: `relabel(games)`; `formats`: the draft formats to fit ("dbo1" / "dbo3"); `space`: the deck space of the DEPLOYED Sealed
    model (the same mana / castability features: the draft formats share the model's feature columns).  Components other than that
    deck space (recency, attributes, cross-set prior, synergy, archetype columns, matchups, play/draw, opponent flags) are off.
    Returns dict(stages, fits, coefs) keyed by the slot names ("draft", "bo1", "bo3"); `coefs[slot]` has `n_boot + 1` rows."""
    from sealed_lab.model import cardvalue
    from sealed_lab.model.design import build_stage

    slots = [SLOT[f] for f in formats]
    n_games = {n: int(games_d.filter(pl.col("fmt") == n).height) for n in ("draft", "bo1", "bo3")}
    prior_names = {n for n in ("draft", "bo1") if n_games.get(n, 0) >= min_prior_games}
    need = set(slots)
    if "draft" in prior_names:
        need.add("draft")
    if "bo3" in slots and "bo1" in prior_names:
        need.add("bo1")
    d = os.path.join(tmp, "final")
    os.makedirs(d, exist_ok=True)
    with rn.slot(memory.estimate_opp_bytes(sum(n_games.get(n, 0) for n in need))):
        stages = {n: build_stage(n, games_d.filter(pl.col("fmt") == n), space) for n in ("draft", "bo1", "bo3") if n in need}
        prior_ok = {n for n in stages if n in prior_names}
        prog(0.0, "Fitting the draft model chain on all data")
        fits = cardvalue.run_chain(stages, space.n_card, space.n_feat, d, n_jobs, seed=seed + 500, cancel=cancel,
                                   progress=lambda fr, m: prog(0.5 * fr, m + " (draft)"), prior_ok=prior_ok, opts=opts)
        coefs = cardvalue.bootstrap(stages, fits, n_boot, d, n_jobs, slots, seed=seed + 900, cancel=cancel,
                                    progress=lambda fr, m: prog(0.5 + 0.5 * fr, m + " (draft)"), prior_ok=prior_ok)
    return dict(stages=stages, fits=fits, coefs=coefs)
