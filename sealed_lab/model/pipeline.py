"""End-to-end model calculation for one set: corpus -> phi -> chain -> bootstrap -> checks -> validation -> artifact."""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from sealed_lab import config
from sealed_lab.model import design, structure
from sealed_lab.model.design import DeckSpace, build_deck_space, with_attributes
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

GAME_COLS = ["game_key", "event", "fmt", "draft_id", "build_index", "match_number", "game_number", "deck_key",
             "draft_time", "game_time", "on_play", "num_mulligans", "opp_num_mulligans", "opp_colors", "won", "skill",
             "main_colors"]
FMT_ORDER = ["bo3", "bo1"]


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


@dataclass
class Inputs:
    set_code: str
    cards: pl.DataFrame
    games: pl.DataFrame
    decks: pl.DataFrame
    deck_cards: pl.DataFrame
    builds: pl.DataFrame
    manifest: dict
    lazy: dict


def load_inputs(set_code: str) -> Inputs:
    from sealed_lab.ingest.corpus import load_corpus, read_manifest

    c = load_corpus(set_code)
    games = c["games"].select(GAME_COLS).collect()
    return Inputs(set_code=set_code, cards=c["cards"], games=games, decks=c["decks"], deck_cards=c["deck_cards"].collect(),
                  builds=c["builds"], manifest=read_manifest(set_code) or {}, lazy=c)


def deck_arrays(deck_keys: np.ndarray, deck_cards: pl.DataFrame):
    """CSR-like (offsets, card_ids, counts) of the decks in `deck_keys` (sorted unique) from deck_cards."""
    dc = deck_cards.sort("deck_key", "card_id")
    k = dc["deck_key"].to_numpy()
    lo = np.searchsorted(k, deck_keys, side="left")
    hi = np.searchsorted(k, deck_keys, side="right")
    lens = hi - lo
    offsets = np.concatenate([[0], np.cumsum(lens)])
    take = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if len(lo) else np.zeros(0, np.int64)
    return offsets, dc["card_id"].to_numpy()[take].astype(np.int64), dc["n"].to_numpy()[take].astype(np.int64)


def phi_cache_path(cache_dir: Path | str, version: int = 1, regime=None) -> Path:
    """On-disk phi cache of one simulator configuration: the v1 file keeps its historical name."""
    from sealed_lab.model.manasim import regime_name

    return Path(cache_dir) / ("phi_cache.npz" if int(version) == 1 else f"phi_cache_v2_{regime_name(regime) or 'random'}.npz")


def phi_for_corpus(inp: Inputs, n_jobs: int | None, cancel=None, progress: Progress | None = None,
                   cache_dir: Path | None = None, version: int = 1, regime=None,
                   need: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """phi for every unique corpus deck, with an on-disk cache keyed by deck hash. Returns (sorted deck_keys, phi).

    `version` / `regime` select the mana simulator (v1, or v2 for one hand/draw regime; the cache is keyed by simulator
    version, revision and regime). `need` (bool per sorted deck) restricts the computation to the decks that matter in this
    regime: the other rows stay NaN."""
    decks = inp.decks.sort("deck_key")
    keys = decks["deck_key"].to_numpy()
    hashes = decks["deck_hash"].to_numpy()
    sig = structure.cards_signature(inp.cards, version, regime)
    phi = np.full((len(keys), structure.N_TABLE), np.nan)
    cpath = None if cache_dir is None else phi_cache_path(cache_dir, version, regime)
    if cpath is not None and cpath.exists():
        try:
            with np.load(cpath, allow_pickle=False) as z:   # closed at once: Windows cannot replace an open file
                ok = str(z["sig"]) == sig
                ch, cp = (z["hashes"], z["phi"]) if ok else (None, None)
            if ok and len(ch):
                order = np.argsort(ch)
                pos = np.searchsorted(ch[order], hashes)
                pos = np.minimum(pos, len(ch) - 1)
                hit = ch[order][pos] == hashes
                phi[hit] = cp[order][pos[hit]]
        except Exception:  # noqa: BLE001
            log.warning("phi cache unreadable; recomputing")
    want = np.ones(len(keys), dtype=bool) if need is None else np.asarray(need, dtype=bool)
    todo = np.nonzero(np.isnan(phi[:, 0]) & want)[0]
    label = "Mana simulation" if int(version) == 1 else f"Mana simulation v2 ({getattr(regime, 'name', regime)})"
    if len(todo):
        offsets, cid, cnt = deck_arrays(keys[todo], inp.deck_cards)
        phi[todo] = structure.compute_phi(offsets, cid, cnt, inp.cards, n_jobs=n_jobs,
                                          progress=(lambda f, m: progress(f, m.replace("Mana simulation", label))) if progress else None,
                                          cancel=cancel, version=version, regime=regime)
        if cpath is not None:
            try:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                tmp = cpath.with_suffix(".tmp.npz")
                good = np.isfinite(phi[:, 0])
                np.savez(tmp, sig=np.array(sig), hashes=hashes[good].astype("U40"), phi=phi[good])
                os.replace(tmp, cpath)
            except OSError:
                log.warning("could not write phi cache")
    elif progress:
        progress(1.0, f"{label} (cached)")
    return keys, phi


def regime_masks(games: pl.DataFrame, keys: np.ndarray) -> dict[str, np.ndarray]:
    """{regime: bool per sorted deck key} - the decks that were played in at least one game of that hand/draw regime."""
    ev = games.select("event", "deck_key").unique()
    reg = {e: config.event_regime(str(e)) for e in ev["event"].unique().to_list()}
    ev = ev.with_columns(pl.col("event").replace_strict(reg, default="hand").alias("regime"))
    out: dict[str, np.ndarray] = {}
    for r in config.REGIMES:
        dk = ev.filter(pl.col("regime") == r)["deck_key"].unique().to_numpy().astype(np.int64)
        if len(dk):
            m = np.zeros(len(keys), dtype=bool)
            pos = np.searchsorted(keys, dk)
            pos = np.minimum(pos, len(keys) - 1)
            m[pos[keys[pos] == dk]] = True
            out[r] = m
    return out


def v2_space(inp: Inputs, games: pl.DataFrame, keys: np.ndarray, table_v1: np.ndarray, weights: np.ndarray, n_jobs: int | None,
             cancel=None, progress: Progress | None = None, cache_dir: Path | None = None,
             regimes: dict | None = None) -> tuple[DeckSpace, dict[str, np.ndarray]]:
    """DeckSpace whose structure block holds the mana simulator v2 features of each game's own regime.

    A deck gets the v2 table of every regime it was played in (the simulation of other regimes is skipped and the row filled
    from the v1 table: it is never used). `regimes` ({name: SimRegime}) carries the set-specific keep rules. Returns
    (space, {regime: full table incl. the extra simulator columns})."""
    masks = regime_masks(games, keys)
    regimes = regimes or {}
    ev = games.group_by("deck_key", "event").len()
    regs = {e: config.event_regime(str(e)) for e in ev["event"].unique().to_list()}
    ev = ev.with_columns(pl.col("event").replace_strict(regs, default="hand").alias("regime"))
    n_reg = len(masks)
    tabs: dict[str, np.ndarray] = {}
    for i, (r, m) in enumerate(masks.items()):
        _, tab = phi_for_corpus(inp, n_jobs, cancel, (lambda f, msg, i=i: progress((i + f) / max(n_reg, 1), msg)) if progress else None,
                                cache_dir, version=2, regime=regimes.get(r, r), need=m)
        bad = ~np.isfinite(tab[:, 0])
        tab[bad] = table_v1[bad]
        tabs[r] = tab
    rw: dict[str, np.ndarray] = {}
    for r in tabs:
        w = np.zeros(len(keys))
        sub = ev.filter(pl.col("regime") == r).group_by("deck_key").agg(pl.col("len").sum())
        pos = np.searchsorted(keys, sub["deck_key"].to_numpy())
        w[pos] = sub["len"].to_numpy()
        rw[r] = np.maximum(w, 1e-9)
    space = build_deck_space(keys, None, inp.deck_cards, inp.cards, regime_phi={r: t[:, :structure.N_FEATURES] for r, t in tabs.items()},
                             regime_weights=rw)
    return space, tabs



def cast_for_corpus(inp: Inputs, regime, n_jobs: int | None, cancel=None, progress: Progress | None = None,
                    cache_dir: Path | None = None, need: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Castability of every corpus deck under one hand/draw regime, with an on-disk cache keyed by deck hash.

    Returns (offsets, p, dead) for ALL decks (sorted by deck_key): `p[offsets[i]:offsets[i+1]]` is the castability of the cards of
    deck i in `deck_arrays` order (card_id ascending; lands and unplayed decks 1.0), `dead[i]` the expected number of dead cards.
    `need` (bool per sorted deck) restricts the computation to the decks played in this regime."""
    from sealed_lab.model import castability

    decks = inp.decks.sort("deck_key")
    keys = decks["deck_key"].to_numpy()
    hashes = decks["deck_hash"].to_numpy()
    offsets, cid, cnt = deck_arrays(keys, inp.deck_cards)
    cc = castability.cast_cards(inp.cards)
    sig = castability.cast_signature(cc, regime)
    lens = np.diff(offsets)
    p = np.ones(len(cid), dtype=np.float32)
    dead = np.full(len(keys), np.nan)
    cpath = None if cache_dir is None else castability.cache_path(cache_dir, regime)
    if cpath is not None and cpath.exists():
        try:
            with np.load(cpath, allow_pickle=False) as z:
                ok = str(z["sig"]) == sig
                ch, cl, cd, cp = (z["hashes"], z["lens"], z["dead"], z["p"]) if ok else (None,) * 4
            if ok and len(ch):
                order = np.argsort(ch)
                pos = np.minimum(np.searchsorted(ch[order], hashes), len(ch) - 1)
                src = order[pos]
                hit = (ch[src] == hashes) & (cl[src] == lens)
                cstart = np.concatenate([[0], np.cumsum(cl)])
                hi = np.nonzero(hit)[0]
                if len(hi):
                    ln = lens[hi]
                    dst = np.repeat(offsets[hi], ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
                    srcpos = np.repeat(cstart[src[hi]], ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
                    p[dst] = cp[srcpos]
                    dead[hi] = cd[src[hi]]
        except Exception:  # noqa: BLE001
            log.warning("castability cache unreadable; recomputing")
    want = np.ones(len(keys), dtype=bool) if need is None else np.asarray(need, dtype=bool)
    todo = np.nonzero(np.isnan(dead) & want)[0]
    label = f"Castability ({getattr(regime, 'name', regime)})"
    if len(todo):
        o2, c2, n2 = deck_arrays(keys[todo], inp.deck_cards)
        d_new, p_new = castability.compute_cast(o2, c2, n2, cc, regime, n_jobs=n_jobs, progress=progress, cancel=cancel, label=label)
        dead[todo] = d_new
        ln = np.diff(o2)
        dst = np.repeat(offsets[todo], ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
        p[dst] = p_new
        if cpath is not None:
            try:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                tmp = cpath.with_suffix(".tmp.npz")
                good = np.nonzero(np.isfinite(dead))[0]
                ln_g = lens[good]
                idx = np.repeat(offsets[good], ln_g) + (np.arange(ln_g.sum()) - np.repeat(np.cumsum(ln_g) - ln_g, ln_g))
                np.savez(tmp, sig=np.array(sig), hashes=hashes[good].astype("U40"), lens=ln_g, dead=dead[good], p=p[idx])
                os.replace(tmp, cpath)
            except OSError:
                log.warning("could not write castability cache")
    elif progress:
        progress(1.0, f"{label} (cached)")
    dead = np.where(np.isfinite(dead), dead, 0.0)
    return offsets, p, dead


def regime_game_weights(games: pl.DataFrame, keys: np.ndarray) -> dict[str, np.ndarray]:
    """{regime: games played with each sorted deck under that regime (>= 1e-9)} - the standardisation weights of a regime space."""
    ev = games.group_by("deck_key", "event").len()
    regs = {e: config.event_regime(str(e)) for e in ev["event"].unique().to_list()}
    ev = ev.with_columns(pl.col("event").replace_strict(regs, default="hand").alias("regime"))
    rw: dict[str, np.ndarray] = {}
    for r in config.REGIMES:
        sub = ev.filter(pl.col("regime") == r).group_by("deck_key").agg(pl.col("len").sum())
        if sub.height == 0:
            continue
        w = np.zeros(len(keys))
        w[np.searchsorted(keys, sub["deck_key"].to_numpy())] = sub["len"].to_numpy()
        rw[r] = np.maximum(w, 1e-9)
    return rw


def cast_space(inp: Inputs, games: pl.DataFrame, keys: np.ndarray, phi_tabs: dict[str, np.ndarray], cast_tabs: dict[str, tuple],
               ) -> DeckSpace:
    """DeckSpace with castability: per regime the structure block of `phi_tabs[regime]` (raw phi, N_FEATURES columns) plus the
    expected-dead-cards feature, and the card block holding copies x castability.  `cast_tabs[regime]` = `cast_for_corpus` result."""
    import scipy.sparse as sp

    rw = regime_game_weights(games, keys)
    nb_col = np.full(inp.cards.height, -1, dtype=np.int64)
    nb = inp.cards.filter(~pl.col("is_basic"))["card_id"].to_numpy().astype(np.int64)
    nb_col[nb] = np.arange(len(nb))
    offsets, cid, cnt = deck_arrays(keys, inp.deck_cards)
    rows = np.repeat(np.arange(len(keys)), np.diff(offsets))
    cols = nb_col[cid]
    keep = cols >= 0
    phis: dict[str, np.ndarray] = {}
    cblocks: dict[str, sp.csr_matrix] = {}
    for r in rw:
        _o, pr, dead = cast_tabs.get(r, cast_tabs[next(iter(cast_tabs))])
        phis[r] = np.hstack([phi_tabs.get(r, phi_tabs[next(iter(phi_tabs))])[:, :structure.N_FEATURES], dead[:, None]])
        cblocks[r] = sp.csr_matrix((cnt[keep] * pr[keep].astype(float), (rows[keep], cols[keep])), shape=(len(keys), len(nb)))
    return build_deck_space(keys, None, inp.deck_cards, inp.cards, regime_phi=phis, regime_weights=rw, cast_cards=cblocks)


def castability_stats(inp: Inputs, games: pl.DataFrame, keys: np.ndarray, cast_tabs: dict[str, tuple], formats: list[str]) -> dict:
    """Report numbers: share of real decks / games with a card under castability 0.8 / 0.5, and the mean expected dead cards."""
    out: dict = {}
    for f in formats:
        r = config.format_regime(f)
        if r not in cast_tabs:
            continue
        offsets, p, dead = cast_tabs[r]
        g = games.filter(pl.col("fmt") == f).group_by("deck_key").len().sort("deck_key")
        pos = np.searchsorted(keys, g["deck_key"].to_numpy())
        w = g["len"].to_numpy().astype(float)
        lens = np.diff(offsets)
        mins = np.ones(len(keys))
        nz = lens > 0
        mins[nz] = np.minimum.reduceat(p, offsets[:-1][nz]) if len(p) else 1.0
        m = mins[pos]
        d = dead[pos]
        out[f] = dict(n_decks=int(len(pos)), n_games=int(w.sum()),
                      decks_below_080=float((m < 0.8).mean()), games_below_080=float((w * (m < 0.8)).sum() / w.sum()),
                      decks_below_050=float((m < 0.5).mean()), games_below_050=float((w * (m < 0.5)).sum() / w.sum()),
                      mean_dead=float((w * d).sum() / w.sum()), sd_dead=float(np.sqrt((w * (d - (w * d).sum() / w.sum()) ** 2).sum() / w.sum())),
                      q_dead={str(q): float(np.percentile(d, q)) for q in (10, 50, 90, 99)})
    return out

# ---------------------------------------------------------------------------------------------
# calculate
# ---------------------------------------------------------------------------------------------

def _draws(coefs: np.ndarray, k: int, space: DeckSpace, ref_row: np.ndarray,
           mull_cols: "tuple[int, int] | None" = None) -> dict[str, np.ndarray]:
    """Per-draw card effects, structure effects and reference logit.

    With card-attribute columns A.gamma is folded into the per-card beta (total effect) and the constant that centred
    the attribute sums into the reference logit, so consumers see ref + sum_c beta_c x_c + theta.z exactly.
    Trailing extra columns (card pairs, then card x archetype) come back as `extra` (B, n_pair + n_archc).  `mull_cols` (the
    covariate columns of "mull1" and "mull2p") returns their draws as `mull` (B, 2): the prediction adds the mulligan cost of
    each deck itself (the reference logit has no mulligans)."""
    B = coefs.shape[0]
    deck = coefs[:, k:]
    nc, nf = space.n_card, space.n_feat
    u = deck[:, :nc]
    ref = coefs[:, :k] @ ref_row
    if space.n_attr:
        gam = deck[:, nc + nf:nc + nf + space.n_attr]
        u = u + gam @ space.attr_load.T
        ref = ref - gam @ (space.attr_mu / space.attr_sd)
    beta = np.zeros((B, space.n_cards_total))
    beta[:, space.nb_ids] = u
    out = dict(beta=beta, theta=deck[:, nc:nc + nf], ref=ref, extra=deck[:, nc + nf + space.n_attr:])
    if mull_cols is not None:
        out["mull"] = coefs[:, list(mull_cols)].copy()
    return out


def _full(space: DeckSpace, v: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.full(space.n_cards_total, fill)
    out[space.nb_ids] = v
    return out


def _card_section(fmt: str, cards: pl.DataFrame, space: DeckSpace, beta: np.ndarray, se: np.ndarray, exposure: np.ndarray,
                  psi: np.ndarray, psi_se: np.ndarray, hc: dict) -> dict:
    """Top / bottom 20 + scatter data for the report. All arrays are over the non-basic card columns."""
    nb = space.nb_ids
    c = cards.sort("card_id")
    names = c["name"].to_numpy()[nb]
    rarity = c["rarity"].to_numpy()[nb]
    lg = np.log(0.55 / 0.45)
    pp = 100 * (1 / (1 + np.exp(-(lg + beta))) - 0.55)
    se_pp = 100 * 0.55 * 0.45 * se
    flag = hc["flag"]
    kappa = hc.get("kappa") or 0.0

    def rec(i):
        return dict(name=str(names[i]), rarity=str(rarity[i]), beta=float(beta[i]), pp=float(pp[i]), se_pp=float(se_pp[i]),
                    exposure=float(exposure[i]), psi=float(psi[i]), flag=bool(flag[i]), low_data=bool(exposure[i] < 2000),
                    expected=float(kappa * psi[i]))
    order = np.argsort(-beta)
    m = exposure > 0
    order = np.array([i for i in order if m[i]]) if m.any() else order
    top = [rec(i) for i in order[:20]]
    bottom = [rec(i) for i in order[::-1][:20]]
    sel = exposure >= 200
    return dict(top=top, bottom=bottom,
                scatter=dict(psi=psi[sel].tolist(), beta=beta[sel].tolist(), flag=flag[sel].tolist()),
                handcheck=dict(kappa=hc.get("kappa"), kappa_se=hc.get("kappa_se"), tau=hc.get("tau"), spearman=hc.get("spearman"), n=hc["n"],
                               n_flagged=int(flag.sum()), informative=bool(hc.get("informative"))),
                flagged=[rec(i) for i in np.nonzero(flag)[0][:60]])


RECENCY_MODES = ("auto", "on", "off")
COMPONENT_MODES = ("auto", "on", "off")
MATCHUPS_MODES = ("pooled", "auto", "on", "off")    # matchups: "pooled" = the per-set rule, plus the evidence pooled over the other sets
MANASIM_MODES = ("auto", "v2", "v1")        # the calibrated mana simulator: tested ("auto"), forced on ("v2") or off ("v1")
_MANASIM_AS_COMPONENT = {"auto": "auto", "v2": "on", "v1": "off", "on": "on", "off": "off"}
# Materiality floor for auto decisions: a paired difference can be "significant" yet negligible (a near-identical
# model has a near-zero noise too). Gains/losses below this held-out log loss (per game) neither apply nor veto.
MIN_GAIN = 1e-4


def _split(tested: dict) -> tuple[list[str], list[str], list[str]]:
    """(formats with a material significant gain, with a material significant loss, significant but negligible)."""
    better = [f for f, o in tested.items() if o["lo"] > 0 and o["gain"] >= MIN_GAIN]
    worse = [f for f, o in tested.items() if o["hi"] < 0 and o["gain"] <= -MIN_GAIN]
    tiny = [f for f, o in tested.items() if (o["lo"] > 0 or o["hi"] < 0) and abs(o["gain"]) < MIN_GAIN]
    return better, worse, tiny


def _verdict(tested: dict, confirm: bool) -> tuple[list[str], list[str], list[str], dict]:
    """(better, worse, tiny, confirmation info) of a component's per-format hold-out tests.

    Without `confirm` (or without the half-split numbers, `halves` of each entry) this is `_split` on the whole hold-out: the
    original rule. With it the hold-out events are split in time into half A (the earlier events) and half B (the later ones):
      - a format counts as better only if it passes the original rule on half A (95% CI above zero, gain >= MIN_GAIN) AND on half B
        the gain has the same sign with a one-sided 90% bound above zero (the 10th percentile of the bootstrap gain > 0);
      - a format counts as worse if it is materially and significantly worse on the whole hold-out or on either half (a safety rule:
        the confirmation split only ever makes a component harder to apply)."""
    better, worse, tiny = _split(tested)
    info = dict(confirm=False, whole_better=better, whole_worse=worse)
    if not confirm or not tested or not all(isinstance(o.get("halves"), dict) and "A" in o["halves"] and "B" in o["halves"]
                                            for o in tested.values()):
        return better, worse, tiny, info
    hv = {f: o["halves"] for f, o in tested.items()}
    pass_a = [f for f, h in hv.items() if h["A"]["lo"] > 0 and h["A"]["gain"] >= MIN_GAIN]
    conf = [f for f in pass_a if hv[f]["B"]["gain"] > 0 and hv[f]["B"]["lo90"] > 0]
    worse_h = [f for f, h in hv.items() if any(h[x]["hi"] < 0 and h[x]["gain"] <= -MIN_GAIN for x in "AB")]
    info.update(confirm=True, passed_A=pass_a, confirmed=conf, failed_B=[f for f in pass_a if f not in conf], worse_halves=worse_h,
                halves={f: {x: {k: h[x][k] for k in ("gain", "lo", "hi", "lo90")} for x in "AB"} for f, h in hv.items()})
    return conf, sorted(set(worse) | set(worse_h)), tiny, info


def _halves_text(info: dict, fmts: list[str]) -> str:
    h = info.get("halves") or {}
    return "; ".join(f"{f}: first half {h[f]['A']['gain']:+.5f} [{h[f]['A']['lo']:+.5f}, {h[f]['A']['hi']:+.5f}], "
                     f"second half {h[f]['B']['gain']:+.5f} (90% lower bound {h[f]['B']['lo90']:+.5f})" for f in fmts if f in h)


def _negligible(what: str, tiny: list[str]) -> str:
    return (f"{what}: the hold-out differences are below the materiality floor ({MIN_GAIN:g} log loss per game"
            + (f"; statistically detectable but negligible in {', '.join(tiny)}" if tiny else "") + "); not applied")


def decide_component(mode: str, key: str, val: dict, available: bool, unavailable: str, what: str, confirm: bool = True) -> dict:
    """Shared auto/on/off rule for a model component tested against the plain chain on the hold-out.

    auto: used only if the paired log-loss gain has a 95% CI above zero and is at least MIN_GAIN in >= 1 format, and no
    format is significantly and materially worse.  With `confirm` the decision is taken on the earlier half of the hold-out events
    and must be confirmed on the later half (same sign, one-sided 90% bound above zero), see `_verdict`; out["confirmation"]
    keeps both verdicts so a component the confirmation turned off is visible."""
    out = dict(mode=mode, applied=False, formats=[f for f, v in val.items() if v.get("status") == "ok" and v.get(key)])
    if mode == "off":
        out["reason"] = "switched off by the user"
        return out
    if not available:
        out["reason"] = unavailable
        return out
    tested = {f: v[key] for f, v in val.items() if v.get("status") == "ok" and v.get(key)}
    if mode == "on":
        out.update(applied=True, reason="forced on by the user" + ("" if tested else " (no format had enough data for the hold-out test)"))
        return out
    if not tested:
        out["reason"] = "no format had enough data for the hold-out test"
        return out
    better, worse, tiny, cinfo = _verdict(tested, confirm)
    cinfo["old_rule_applied"] = bool(cinfo["whole_better"]) and not cinfo["whole_worse"]
    out["confirmation"] = cinfo
    if better and not worse:
        out.update(applied=True, reason=f"{what} improves held-out log loss with a 95% CI above zero in " + ", ".join(better)
                   + " and is not significantly worse elsewhere"
                   + (" (decided on the earlier half of the hold-out events and confirmed on the later half)" if cinfo["confirm"] else ""))
    elif worse:
        out["reason"] = f"{what} is significantly WORSE on the hold-out in " + ", ".join(worse) + "; not applied"
    elif cinfo["confirm"] and cinfo["failed_B"]:
        out["reason"] = (f"{what} passes on the earlier half of the hold-out events in " + ", ".join(cinfo["failed_B"])
                         + " but is NOT confirmed on the later half (needs the same sign and a one-sided 90% bound above zero: "
                         + _halves_text(cinfo, cinfo["failed_B"]) + "); not applied")
    elif cinfo["confirm"] and cinfo["whole_better"]:
        out["reason"] = (f"{what} improves the whole hold-out in " + ", ".join(cinfo["whole_better"]) + " but does not pass on the "
                         "earlier half alone (the decision half); not applied, as the confirmation split needs both")
    elif tiny:
        out["reason"] = _negligible(what, tiny)
    else:
        out["reason"] = (f"the hold-out cannot distinguish {what} from the plain chain (every 95% CI includes zero); not applied, "
                         "as every model change must show a significant gain")
    return out


def decide_castability(mode: str, val: dict, available: bool, unavailable: str = "", confirm: bool = True) -> dict:
    """Shipping rule of the castability component ("MK" vs "M" on the same hold-out) - the owner's explicit rule for THIS component.

    Castability is a safety constraint against extrapolation (an unrestricted search otherwise fills the deck with strong cards
    the mana cannot cast, which real decks hardly ever contain), so it does not have to prove a gain: in auto it SHIPS UNLESS a
    format is materially worse (paired log-loss gain <= -MIN_GAIN with the 95% CI below zero) on the whole hold-out or, with
    `confirm`, on either half of its events. Real data can only partly test it (real decks do contain thin splashes).
    "on" always ships it, "off" never; the hold-out test is reported either way."""
    out = dict(mode=mode, applied=False, formats=[f for f, v in val.items() if v.get("status") == "ok" and v.get("castability")],
               rule="ships unless a format is materially worse (gain <= -%g with the CI below zero)%s"
                    % (MIN_GAIN, " on the whole hold-out or on either half of its events" if confirm else ""))
    if mode == "off":
        out["reason"] = "switched off by the user"
        return out
    if not available:
        out["reason"] = unavailable or "castability could not be computed"
        return out
    tested = {f: v["castability"] for f, v in val.items() if v.get("status") == "ok" and v.get("castability")}
    gains = ", ".join(f"{f} {o['gain']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}]" for f, o in tested.items())
    if mode == "on":
        out.update(applied=True, reason="forced on by the user" + (f" (hold-out MK - M: {gains})" if tested else ""))
        return out
    _b, worse, _t, cinfo = _verdict(tested, confirm)
    better, _w, _t2 = _split(tested)
    cinfo["old_rule_applied"] = not cinfo["whole_worse"]
    out["confirmation"] = cinfo
    if worse:
        out["reason"] = ("castability is significantly and materially WORSE on the hold-out in " + ", ".join(worse) + f" ({gains}); not applied")
    else:
        out.update(applied=True, reason=("shipped as a safety constraint against extrapolation: no format is materially worse on the hold-out"
                                         + (" or on either half of its events" if cinfo["confirm"] else "")
                                         + (f" ({gains})" if tested else " (no format had enough data for the hold-out test)")
                                         + ("; it also improves the log loss significantly in " + ", ".join(better) if better else "")))
    return out


def decide_matchups(mode: str, val: dict, formats: list[str], set_code: str, confirm: bool = True, others: dict | None = None,
                    home: Path | None = None) -> dict:
    """Whether and in which formats the matchup layer (model "MO", the opponent-colour interactions) is applied.

    "off" / "on" / "auto" are `decide_component`'s (key "opponent"); "pooled" (the default) is "auto" plus the evidence pooled across
    sets (`model/pooling.py`, the pre-registered rule of docs/pooled_retest/POOLING_RULE.md): when the per-set rule does not apply the
    layer, each format is decided by the OTHER sets' latest stored hold-out numbers (their report.json in the app home; `others` =
    {fmt: {set: cell}} overrides that read), random-effects pooled, provided at least pooling.MIN_SETS other sets have numbers in
    that format and this set's own numbers there are not materially worse (whole hold-out or either half).  The set itself never
    contributes to its own pooled verdict.  A format the pooled rule cannot decide falls back to the per-set result (not applied).

    Besides `decide_component`'s keys the result has `applied_formats` (the formats the layer's advice is shown in; the layer itself
    is one fit over the pooled games of all stages), `source` ({fmt: "per-set" | "pooled" | "forced"}), and, in "pooled" mode when the
    per-set rule did not apply, `pooled` ({fmt: pooling.summary}) and `per_set_reason`."""
    from sealed_lab.model import pooling

    d = decide_component("auto" if mode == "pooled" else mode, "opponent", val, True, "", "the card x opponent-colour interactions", confirm)
    d["mode"] = mode
    tested = list(d["formats"])
    d.update(applied_formats=[], source={})
    if mode == "off":
        return d
    if d["applied"]:
        # (the layer is one flag in these modes: advice in every tested format; none tested = forced on without a test)
        d["applied_formats"] = tested or list(formats)
        d["source"] = {f: "forced" if mode == "on" else "per-set" for f in d["applied_formats"]}
        return d
    if mode != "pooled" or not tested:
        return d
    if others is None:
        others = pooling.other_cells(set_code, "matchups", "opponent", tuple(tested), home)
    own = {f: pooling.make_cell(val[f]["opponent"], val[f]["opponent"].get("halves")) for f in tested}
    dec = pooling.pooled_decision(own, others, exclude=set_code)
    d["pooled"] = {f: pooling.summary(r) for f, r in dec.items()}
    d["per_set_reason"] = d["reason"]
    lines = "; ".join(pooling.describe(f, sm) for f, sm in d["pooled"].items())
    yes = [f for f, r in dec.items() if r["apply"]]
    if yes:
        d.update(applied=True, applied_formats=yes, source={f: "pooled" for f in yes},
                 reason=f"applied in {', '.join(yes)} from the hold-out evidence pooled across sets (the per-set test alone did not "
                        f"apply it: {d['per_set_reason']}). " + lines)
    else:
        d["reason"] = d["per_set_reason"] + "; evidence pooled across sets: " + lines
    return d


def decide_oppflags(mode: str, val: dict, formats: list[str], set_code: str, confirm: bool = True, others: dict | None = None,
                    home: Path | None = None) -> dict:
    """Whether the opponent colour flags (model "MF", key "oppflags") enter the game covariates of the deployed chain.

    `decide_component`'s auto / on / off on the "oppflags" hold-out test (paired with the chain without them, half A decides, half B
    confirms).  In "auto", where the per-set rule does not apply them, the pooled route of `decide_matchups` (model/pooling.py,
    docs/pooled_retest/POOLING_RULE.md, leave-this-set-out) can switch them ON; it never switches them off.  The flags are one design
    change shared by all formats, so they are applied if the per-set rule or the pooled rule says so in ANY tested format."""
    from sealed_lab.model import pooling

    d = decide_component(mode, "oppflags", val, True, "", "the opponent colour flags", confirm)
    d["source"] = "forced" if mode == "on" else ("per-set" if d["applied"] else None)
    if mode != "auto" or d["applied"]:
        return d
    tested = list(d["formats"])
    if not tested:
        return d
    if others is None:
        others = pooling.other_cells(set_code, "oppflags", "oppflags", tuple(tested), home)
    own = {f: pooling.make_cell(val[f]["oppflags"], val[f]["oppflags"].get("halves")) for f in tested}
    dec = pooling.pooled_decision(own, others, exclude=set_code)
    d["pooled"] = {f: pooling.summary(r) for f, r in dec.items()}
    d["per_set_reason"] = d["reason"]
    lines = "; ".join(pooling.describe(f, sm) for f, sm in d["pooled"].items())
    yes = [f for f, r in dec.items() if r["apply"]]
    if yes:
        d.update(applied=True, source="pooled",
                 reason=f"applied from the hold-out evidence pooled across sets in {', '.join(yes)} (the per-set test alone did not "
                        f"apply it: {d['per_set_reason']}). " + lines)
    else:
        d["reason"] = d["per_set_reason"] + "; evidence pooled across sets: " + lines
    return d


def decide_recency(mode: str, chosen: float | None, inner: dict, val: dict, confirm: bool = True) -> dict:
    """Whether the deployed fit uses recency weights (one half-life for the formats given). auto: only with a significant hold-out
    gain (confirmed on the later half of the events, see `_verdict`) and no significant loss."""
    from sealed_lab.model import validate

    out = dict(mode=mode, chosen=chosen, inner=inner, applied=False, half_life=None)
    if mode == "off":
        out["reason"] = "switched off by the user"
        return out
    tested = {f: v["recency"] for f, v in val.items() if v.get("status") == "ok" and v.get("recency")}
    if mode == "on":
        h = chosen if chosen is not None else validate.best_finite_half_life(inner)
        out.update(applied=True, half_life=h, reason=f"forced on by the user (half-life {h:g} days"
                   + (", the best on the inner split)" if chosen is not None else
                      ", the least bad on the inner split - no half-life beat equal weights there)"))
        return out
    if chosen is None:
        out["reason"] = "no half-life beat equal weights when fitted on older events and scored on newer ones (inner split)"
        return out
    if not tested:
        out["reason"] = "no format had enough data for the hold-out test"
        return out
    better, worse, tiny, cinfo = _verdict(tested, confirm)
    cinfo["old_rule_applied"] = bool(cinfo["whole_better"]) and not cinfo["whole_worse"]
    out["confirmation"] = cinfo
    if better and not worse:
        out.update(applied=True, half_life=chosen,
                   reason=f"half-life {chosen:g} days improves held-out log loss with a 95% CI above zero in "
                          + ", ".join(better) + " and is not significantly worse elsewhere"
                          + (" (decided on the earlier half of the hold-out events and confirmed on the later half)" if cinfo["confirm"] else ""))
    elif worse:
        out["reason"] = (f"half-life {chosen:g} days is significantly WORSE on the hold-out in " + ", ".join(worse)
                         + "; equal weights kept")
    elif cinfo["confirm"] and cinfo["failed_B"]:
        out["reason"] = (f"half-life {chosen:g} days passes on the earlier half of the hold-out events in " + ", ".join(cinfo["failed_B"])
                         + " but is NOT confirmed on the later half (" + _halves_text(cinfo, cinfo["failed_B"]) + "); equal weights kept")
    elif cinfo["confirm"] and cinfo["whole_better"]:
        out["reason"] = (f"half-life {chosen:g} days improves the whole hold-out in " + ", ".join(cinfo["whole_better"])
                         + " but does not pass on the earlier half alone; equal weights kept")
    elif tiny:
        out["reason"] = _negligible(f"half-life {chosen:g} days", tiny) + "; equal weights kept"
    else:
        out["reason"] = (f"half-life {chosen:g} days: the hold-out cannot distinguish it from equal weights (every 95% CI includes "
                         "zero); equal weights kept, as every model change must show a significant gain")
    return out


def decide_recency_formats(mode: str, chosen: dict, inner: dict, val: dict, confirm: bool = True) -> dict:
    """Per-format recency: each format has its OWN half-life (chosen on its inner split) and its own hold-out decision. The draft
    stage is never weighted, and a format that does not benefit stays equal-weighted while another one uses recency.
    Returns the usual recency dict with `chosen` and `half_life` as {format: days or None}, `applied` = any format, `formats` =
    the per-format decisions."""
    per = {f: decide_recency(mode, chosen.get(f), {f: inner[f]} if f in inner else {}, {f: val[f]} if f in val else {}, confirm)
           for f in chosen}
    for d in per.values():
        d.pop("inner", None)
    return dict(mode=mode, scope="format", chosen=dict(chosen), inner=inner, applied=any(d["applied"] for d in per.values()),
                half_life={f: (d["half_life"] if d["applied"] else None) for f, d in per.items()}, formats=per,
                confirmation={f: d["confirmation"] for f, d in per.items() if d.get("confirmation")},
                reason="; ".join(f"{f}: {d['reason']}" for f, d in per.items()))


# ---------------------------------------------------------------------------------------------
# extra draft events (PickTwo / QuickDraft ...): kept or dropped before anything else is fitted
# ---------------------------------------------------------------------------------------------

EXTRA_DRAFT_WHAT = "the extra draft events (PickTwo / QuickDraft)"


def _is_extra_draft() -> pl.Expr:
    """Row mask of the draft games of events outside the standard draft data (config.CORE_DRAFT_EVENTS)."""
    return (pl.col("fmt") == "draft") & ~pl.col("event").fill_null("").is_in(list(config.CORE_DRAFT_EVENTS))


def extra_draft_events(games: pl.DataFrame) -> dict[str, int]:
    """{event: games} of the extra draft events in the game table (empty when the set has none)."""
    g = games.filter(_is_extra_draft())
    return {str(e): int(n) for e, n in g.group_by("event").len().sort("event").iter_rows()} if g.height else {}


def without_extra_draft(inp: Inputs) -> Inputs:
    """`inp` without the extra draft events' games.  The decks nobody played in the remaining games go too: a deck that only
    existed in a dropped event must not take part in the deck space, the phi tables or the archetype clusters."""
    games = inp.games.filter(~_is_extra_draft())
    used = games.select("deck_key").unique()
    return replace(inp, games=games, decks=inp.decks.join(used, on="deck_key", how="semi"),
                   deck_cards=inp.deck_cards.join(used, on="deck_key", how="semi"))


def plain_space(inp: Inputs, keys: np.ndarray, table: np.ndarray) -> tuple[DeckSpace, np.ndarray]:
    """(deck space, games per deck) of the plain chain over the decks `keys` (phi `table`), standardised by the games of `inp`."""
    per_deck = inp.games.group_by("deck_key").len().sort("deck_key")
    wts = np.zeros(len(keys))
    wts[np.searchsorted(keys, per_deck["deck_key"].to_numpy())] = per_deck["len"].to_numpy()
    return build_deck_space(keys, table[:, :structure.N_FEATURES], inp.deck_cards, inp.cards, weights=np.maximum(wts, 1e-9)), wts


def val_cache(set_code: str, games: pl.DataFrame, cards: pl.DataFrame):
    """The validation cache of one data state: every stored hold-out result is keyed by the content of the data it saw."""
    from sealed_lab.model import vcache

    return vcache.ValCache(config.cache_dir(set_code), dict(
        set_code=set_code, games=vcache.digest(games), cards=vcache.digest(cards),
        hands=vcache.file_digest(config.corpus_dir(set_code) / "hands.parquet")))


def extra_draft_holdout(set_code: str, inp: Inputs, keys: np.ndarray, table: np.ndarray, formats: list[str], n_jobs: int, seed: int,
                        n_rep: int, opts, cancel: threading.Event | None = None, progress: Progress | None = None) -> dict:
    """Hold-out gain of the extra draft events: the plain chain "M" with them in the draft stage versus without them, scored on the
    same time hold-out of each Sealed format (the draft stage and the Bo1 stage keep only events that started before the cut, see
    `validate.chain_stages`).  Without them a set that has no other draft games has no draft stage at all.
    Returns {fmt: dict(gain, lo, hi, halves, logloss_with, logloss_without, n_holdout_events, n_holdout_games)}, gain > 0 = the extra
    events help; a format too small for a hold-out is missing.  The plain-chain results of both data states go into the validation
    cache, so the later validation of the chosen one finds its plain chain there and an unchanged data state costs nothing."""
    from sealed_lab.model import cardvalue, glm, memory, parallel, validate

    inp_wo = without_extra_draft(inp)
    sel = np.isin(keys, inp_wo.decks["deck_key"].to_numpy())
    cond = {"with": (inp, plain_space(inp, keys, table)[0]), "without": (inp_wo, plain_space(inp_wo, keys[sel], table[sel])[0])}
    caches = {c: val_cache(set_code, i_.games, inp.cards) for c, (i_, _sp) in cond.items()}
    keeps: dict[str, dict] = {c: {} for c in cond}
    tmp = cardvalue.make_tmpdir(config.cache_dir(set_code))
    vals: dict = {}
    try:
        with contextlib.ExitStack() as stack:
            if n_jobs > 1:
                stack.enter_context(memory.budget())
                stack.enter_context(glm.shared_pool(n_jobs))
                rn = stack.enter_context(parallel.Runner(cancel, max_active=8))
            else:
                rn = parallel.InlineRunner(cancel)
            runs = [(c, i, f) for c in cond for i, f in enumerate(formats)]
            mux = parallel.ProgressMux(progress, 0.0, 1.0, len(runs))
            futs = {(c, f): rn.spawn(validate.run_validation, f, cond[c][0].games, cond[c][1], inp.cards, inp.lazy["hands"],
                                     os.path.join(tmp, "extra_" + c), n_jobs, seed=seed + i, n_rep=n_rep, cancel=rn.cancel,
                                     progress=mux.slot(k), cache=caches[c], runner=rn, keep=keeps[c], opts=opts)
                    for k, (c, i, f) in enumerate(runs)}
            vals = {key: rn.result(fut) for key, fut in futs.items()}
    finally:
        cardvalue.cleanup(tmp)
    out: dict = {}
    for i, f in enumerate(formats):
        kw, ko = keeps["with"].get(f), keeps["without"].get(f)
        if vals[("with", f)].get("status") != "ok" or vals[("without", f)].get("status") != "ok" or kw is None or ko is None:
            continue
        if not (np.array_equal(kw["y"], ko["y"]) and np.array_equal(kw["groups"], ko["groups"])):
            log.warning("extra draft events: the two hold-outs of %s differ; format not tested", f)
            continue
        g_t = inp.games.filter(pl.col("fmt") == f)
        hold_ids = validate._holdout_split(g_t)[0]
        is_first = validate.half_masks(g_t.filter(pl.col("draft_id").is_in(hold_ids)))
        y, groups = np.asarray(kw["y"]), np.asarray(kw["groups"])
        met = validate.cluster_metrics(y, {"M": np.asarray(kw["pM"]), "B": np.asarray(ko["pM"])}, groups, n_rep, seed + i + 5)
        d = met["B"]["vs_M"]["logloss"]                              # without minus with: positive = the extra events help
        out[f] = dict(gain=d["diff"], lo=d["lo"], hi=d["hi"], logloss_with=met["M"]["logloss"], logloss_without=met["B"]["logloss"],
                      halves=validate.half_gains(y, np.asarray(ko["pM"]), np.asarray(kw["pM"]), groups, is_first, n_rep, seed + i + 7),
                      n_holdout_events=int(groups.max()) + 1, n_holdout_games=int(len(y)))
    return out


def decide_extra_draft(mode: str, events: dict[str, int], core_draft_games: int, tested: dict | None, confirm: bool = True) -> dict:
    """The extra-draft decision: same rule as `decide_component` for "auto" (95% CI above zero and at least MIN_GAIN in >= 1 format,
    no format materially worse, confirmed on the later half), "on" keeps the events untested, "off" drops them.  `tested` =
    `extra_draft_holdout`'s result; `events` = {event: games} of the extra events (empty = nothing to decide: not available)."""
    out = dict(mode=mode, available=bool(events), applied=False, events=dict(events), games=int(sum(events.values())),
               core_draft_games=int(core_draft_games), formats=[], gain={})
    if not events:
        out["reason"] = "no extra draft events (PickTwo / QuickDraft) in this set's data"
        return out
    if mode == "on":
        out.update(applied=True, reason="forced on by the user (the hold-out test was skipped)")
        return out
    if mode == "off":
        out["reason"] = "switched off by the user"
        return out
    val = {f: {"status": "ok", "extra_draft": g} for f, g in (tested or {}).items()}
    d = decide_component("auto", "extra_draft", val, True, "", EXTRA_DRAFT_WHAT, confirm)
    out.update(applied=d["applied"], reason=d["reason"], formats=d["formats"], gain=dict(tested or {}))
    if d.get("confirmation"):
        out["confirmation"] = d["confirmation"]
    if not core_draft_games:
        out["reason"] += " (without them this set has no draft stage at all)"
    return out


def calculate(set_code: str, progress: Progress | None = None, cancel: threading.Event | None = None,
              n_jobs: int | None = None, n_boot: int = 16, seed: int = 0, n_rep: int | None = None,
              recency: str = "auto", attributes: str = "auto", crossset: str = "auto", synergy: str = "auto",
              archcond: str = "auto", matchups: str = "pooled", min_pair_games: int | None = None,
              manasim: str = "auto", castability: str = "auto", sigma_cv: str = "time", stack: str = "auto",
              confirm: bool = True, recency_scope: str = "format", leak_free: bool = False, extra_draft: str = "auto",
              theta_sigma: str = "rule", playdraw: str = "auto", oppflags: str = "off") -> dict:
    """Run the whole model for a set (the corpus must exist; it is built incrementally if missing). Returns meta.

    `oppflags` (default "off", docs/opp_flags/RESULT.md): 17lands' opp_colors are the colours the opponent SHOWED in the game (they
    differ between games of the same bo3 match in 44% of matches; 0-1 colours shown = short games we mostly win), so the flags carry
    the game's outcome and their large hold-out gain is not evidence about the opponent's deck; do not switch on without a pre-game
    measure of the opponent's colours.
    Five multi-hot opponent colour flags (W U B R G) and the opponent's number of colours join the game covariates next to
    the opponent colour-pair one-hot (docs/opp_flags/RULE.md); "auto" tests the chain with them against the chain without
    (model "MF", per-set hold-out rule with half A/B confirmation, plus the pooled route that can only switch them on), "on" forces,
    "off" skips. When they are deployed the matchup layer is tested and fitted on top of that base (second validation pass).

    `sigma_cv`: how the Sealed stages (Bo1, Bo3) choose their prior strength sigma - "time" (default): forward-chaining folds
    over event start times (train on earlier events, score the next block) on a grid reaching two octaves lower; "random": the
    original random event folds and grid. `stack`: "auto" also chooses, per Sealed stage and on the same folds, the weight w of
    q = w*M + (1-w)*P (P = the stage's prior mean, the previous stage's values rescaled) and deploys q as the card values (builder,
    draws and report alike); "off" deploys M. `confirm`: a component (recency, attributes, cross-set prior, synergy, archetype
    columns, matchups, mana simulator v2) is applied only if it passes the usual test on the earlier half of the hold-out events AND
    is confirmed on the later half (same sign, one-sided 90% bound above zero); castability must not be materially worse on either
    half. `recency_scope`: "format" (default) chooses and tests a half-life separately per format, weighting only that format's own
    games (the draft stage and the prior Bo1 stage keep equal weights); "pooled": one half-life for all formats (every stage but the
    draft stage weighted). `leak_free`: in the time folds that choose sigma and w, the prior P of each fold is refitted from the
    prior stages' events before the scored block; False (default) reuses the prior fitted on all training data (better on the hold-out).

    `playdraw`: card x play/draw interactions on top of the chain (model MP, model/playdraw.py; only feeds the games-2-3 advice, never the
    deck score). Modes "auto" (default: applied only if it passes the usual hold-out test with confirmation, `decide_component`), "on", "off".
    TODO: no "pooled" route (evidence pooled across sets, as `matchups`) yet; scripts/pooled_retest.py can run it from the stored reports.

    `manasim`: the calibrated mana simulator (v2: queue-specific hands and draws, real keep rule, card flow; see
    model/manasim.py). "auto" fits the chain with the v2 structure features as model "MM" next to the v1 chain "M" on the
    same hold-out and deploys v2 only if the paired log-loss gain has a 95% CI above zero and reaches MIN_GAIN in >= 1
    format with no format materially worse; "v2" always uses it (the test still runs and is reported); "v1" keeps the
    original simulator and skips the test.

    `castability`: card copies weighted by the probability the deck's mana can cast the card on time (Karsten-style, see
    model/castability.py) plus an expected-dead-cards feature, tested as model "MK" against the plain chain "M". Unlike the other
    components it ships in "auto" UNLESS a format is materially worse (it is a safety constraint against extrapolation: see
    `decide_castability`); "on" forces it, "off" skips it.

    `recency`: "auto" tests whether weighting recent events more improves held-out predictions and uses it only if
    it does; "on" always uses the best half-life found; "off" skips the test (all games weigh the same).
    `attributes`: the card-attribute prior beta_c = A_c.gamma + u_c; "auto" tests it on the hold-out (model MA) and uses it
    only if it helps, "on" always uses it (the test still runs and is reported), "off" skips it.
    `crossset`: prior mean for the structure terms from the models of other sets (model MX); same three modes, and a
    no-op when no other set has a model.
    `synergy`: card-pair interaction columns min(copies a, copies b) (model MS); `archcond`: card x archetype columns
    (model MC); `matchups`: card x opponent-colour interactions on top of the chain (model MO). Same three modes each:
    "auto" deploys a component only if its paired hold-out log-loss gain has a 95% CI above zero and no format is
    significantly worse. The archetype clusters themselves (descriptive) are always built.
    `matchups` has a fourth mode, "pooled" (the default): the per-set "auto" rule, and in every format where that rule does not apply
    the layer, the OTHER sets' stored hold-out numbers decide (random-effects pooled over >= 5 other sets, the pre-registered rule of
    docs/pooled_retest/POOLING_RULE.md) unless this set's own numbers are materially worse there; see `decide_matchups`.
    `min_pair_games`: shared games a card pair needs to get a column (default synergy.MIN_PAIR_GAMES).
    `extra_draft`: draft games of the events other than PremierDraft / TradDraft (PickTwo, QuickDraft: the draft stage's "extra
    events"). "auto" decides FIRST, before any other test, whether they stay: the plain chain with versus without them on the same
    time hold-out of each Sealed format, same rule as the components (see `decide_extra_draft`); "on" keeps them untested, "off"
    drops them. A set without such games is not tested and not changed."""
    from sealed_lab import CODE_VERSION
    from sealed_lab.model import artifact, cardvalue, castability as cast_mod, handcheck, reference, validate
    from sealed_lab.model import manasim as manasim_mod
    from sealed_lab.model.design import build_stage
    from scipy.special import expit

    t_start = time.time()
    set_code = set_code.upper()
    if manasim not in _MANASIM_AS_COMPONENT:
        raise ValueError(f"manasim must be one of {MANASIM_MODES}, got {manasim!r}")
    if castability not in COMPONENT_MODES:
        raise ValueError(f"castability must be one of {COMPONENT_MODES}, got {castability!r}")
    if playdraw not in COMPONENT_MODES:
        raise ValueError(f"playdraw must be one of {COMPONENT_MODES}, got {playdraw!r}")
    if oppflags not in COMPONENT_MODES:
        raise ValueError(f"oppflags must be one of {COMPONENT_MODES}, got {oppflags!r}")
    if matchups not in MATCHUPS_MODES:
        raise ValueError(f"matchups must be one of {MATCHUPS_MODES}, got {matchups!r}")
    if extra_draft not in COMPONENT_MODES:
        raise ValueError(f"extra_draft must be one of {COMPONENT_MODES}, got {extra_draft!r}")
    if stack not in ("auto", "off"):
        raise ValueError(f"stack must be 'auto' or 'off', got {stack!r}")
    if recency_scope not in ("format", "pooled"):
        raise ValueError(f"recency_scope must be 'format' or 'pooled', got {recency_scope!r}")
    opts = cardvalue.ChainOpts(sigma_cv=sigma_cv, stack=stack != "off", leak_free=bool(leak_free),
                               theta_sigma=theta_sigma)      # (validates sigma_cv, theta_sigma)
    n_jobs = int(n_jobs or os.cpu_count() or 1)
    last = [0.35]

    def prog(f: float, m: str) -> None:
        last[0] = max(last[0], 0.35 + 0.65 * min(max(f, 0.0), 1.0))
        if progress:
            progress(last[0], m)

    from sealed_lab.ingest import registry as registry_mod

    source_files = sorted(f"{f.kind}:{f.fingerprint}" for f in registry_mod.files(set_code)
                          if f.kind in ("game", "replay", "cards", "scryfall"))
    prog(0.0, "Loading corpus")
    try:
        inp = load_inputs(set_code)
    except FileNotFoundError:
        from sealed_lab.ingest.corpus import build_corpus
        build_corpus(set_code, progress=None, cancel=cancel, n_jobs=n_jobs, fetch_scryfall=False)
        inp = load_inputs(set_code)
    _check(cancel)
    inp.cards = cast_mod.with_cost_groups(inp.cards, set_code)             # + the exact colour groups of every mana cost
    games, cards = inp.games, inp.cards
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    formats = [f for f in FMT_ORDER if n_games.get(f, 0) >= 1]
    if not formats:
        raise ValueError(f"{set_code}: no Sealed (Bo1) or TradSealed (Bo3) games in the corpus - nothing to model "
                         "(draft data alone only serves as a prior)")
    warnings: list[str] = [w for w in inp.manifest.get("warnings", []) if "model for this format will lean on priors" not in w]
    degraded: list[str] = []                 # silent fallbacks: shown in the report, in meta and in the set's status

    def degrade(msg: str) -> None:
        warnings.append(msg)
        degraded.append(msg)

    # ---- the extra draft events (PickTwo / QuickDraft ...): kept or dropped FIRST, so that everything below (the other components'
    #      tests, the deployed fit, the report) works on the draft stage that was chosen.  A set without them is not touched.
    xd_events = extra_draft_events(games)
    xd_core = n_games["draft"] - sum(xd_events.values())
    t_xd = time.time()
    if xd_events and extra_draft == "off":              # (dropped before the phi tables, which then skip their decks)
        inp = without_extra_draft(inp)
        games = inp.games
    keys, table = phi_for_corpus(inp, n_jobs, cancel, lambda f, m: prog(0.0 + 0.10 * f, m), config.cache_dir(set_code))
    tested_xd = None
    if xd_events and extra_draft == "auto":
        prog(0.10, "Testing the extra draft events")
        tested_xd = extra_draft_holdout(set_code, inp, keys, table, formats, n_jobs, seed, n_rep or validate.N_REP, opts, cancel,
                                        lambda f, m: prog(0.10 + 0.04 * f, "Extra draft events: " + m))
    xd = decide_extra_draft(extra_draft, xd_events, xd_core, tested_xd, confirm)
    if xd_events and extra_draft == "auto" and not xd["applied"]:
        inp = without_extra_draft(inp)
        games = inp.games
        keep_rows = np.isin(keys, inp.decks["deck_key"].to_numpy())
        keys, table = keys[keep_rows], table[keep_rows]
    xd["seconds"] = round(time.time() - t_xd, 1)
    if xd_events:
        log.info("extra draft events %s: %s (%s)", ", ".join(f"{e} {n:,}" for e, n in xd_events.items()),
                 "kept" if xd["applied"] else "dropped", xd["reason"])
    _check(cancel)
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}

    for f in formats:
        if n_games[f] < 1000:
            warnings.append(f"only {n_games[f]:,} {f} games: estimates for this format lean heavily on the draft prior"
                            + (" and the Bo1 data" if f == "bo3" else "") + "; treat intervals as the guide")
    if "draft" not in n_games or n_games["draft"] < validate.MIN_PRIOR_GAMES:
        warnings.append("no usable draft data: card effects are shrunk towards zero rather than towards the draft estimate")

    # ---- structure features for every corpus deck
    space, wts = plain_space(inp, keys, table)
    _check(cancel)
    # mana simulator v2: a second deck space whose structure block is computed per hand/draw regime (model "MM")
    space_m = None
    tabs_v2: dict[str, np.ndarray] = {}
    sim_regimes: dict = {}
    keep_info: dict = {}
    # the set's keep rule (castability's dealing, the v2 simulator and the mulligan term of q(D) all use it): always measured
    try:
        from sealed_lab.model import simcalib

        sim_regimes, keep_info = simcalib.keep_regimes(set_code, cancel=cancel, progress=lambda f, m: prog(0.095 + 0.005 * f, m))
    except Cancelled:
        raise
    except Exception:  # noqa: BLE001 - fall back to the shipped average keep rule
        log.exception("could not measure the set's keep rule; using the shipped average")
        degrade("the set's own mulligan (keep) rule could not be measured; the average of MSH and SOS was used for the "
                "mulligans of the simulator, castability and the deck's mulligan cost")
    if manasim != "v1":                  # "v1" never uses the v2 feature space: do not build it (castability only needs the dealing machinery)
        try:
            space_m, tabs_v2 = v2_space(inp, games, keys, table, wts, n_jobs, cancel,
                                        lambda f, m: prog(0.10 + 0.04 * f, m), config.cache_dir(set_code), regimes=sim_regimes)
        except Cancelled:
            raise
        except Exception:  # noqa: BLE001 - never lose the model over the optional simulator
            log.exception("mana simulator v2 unavailable; using v1")
            degrade("the calibrated mana simulator (v2) could not be computed; the original simulator was used")
            space_m = None
    _check(cancel)
    # castability (model "MK"): per hand/draw regime, the castability of every card of every deck played in it
    space_k = None
    cast_tabs: dict[str, tuple] = {}
    cast_stats: dict = {}
    support_ref: dict = {}
    if castability != "off":
        try:
            masks_k = regime_masks(games, keys)
            for i_, (r_, m_) in enumerate(masks_k.items()):
                cast_tabs[r_] = cast_for_corpus(inp, sim_regimes.get(r_, r_), n_jobs, cancel,
                                                lambda f, msg, i_=i_: prog(0.14 + 0.035 * (i_ + f) / max(len(masks_k), 1), msg),
                                                config.cache_dir(set_code), need=m_)
            space_k = cast_space(inp, games, keys, {r_: table for r_ in cast_tabs}, cast_tabs)
            cast_stats = castability_stats(inp, games, keys, cast_tabs, formats)
        except Cancelled:
            raise
        except Exception:  # noqa: BLE001 - never lose the model over the optional component
            log.exception("castability unavailable")
            degrade("castability could not be computed; the model has no castability layer")
            space_k = None
    try:                                  # reference for the "outside the data" flags (also without the castability model)
        support_ref = _support_reference(inp, games, keys, formats)
    except Exception:  # noqa: BLE001
        log.exception("support reference unavailable")
        degrade("the reference for the 'outside the data' checks (typical spell colours / sources of real decks) could not be computed; "
                "decks are not checked against it")
    _check(cancel)

    config.cache_dir(set_code).mkdir(parents=True, exist_ok=True)
    if recency not in RECENCY_MODES:
        raise ValueError(f"recency must be one of {RECENCY_MODES}, got {recency!r}")
    for nm_, md_ in (("attributes", attributes), ("crossset", crossset), ("synergy", synergy), ("archcond", archcond)):
        if md_ not in COMPONENT_MODES:
            raise ValueError(f"{nm_} must be one of {COMPONENT_MODES}, got {md_!r}")
    # ---- card attributes (deck-level columns) and the cross-set structure prior
    from sealed_lab.model import attributes as attr_mod
    from sealed_lab.model import crossset as xset

    space_a = None
    if attributes != "off":
        A_mat, A_names = attr_mod.attribute_matrix(cards)
        cols = attr_mod.select_columns(A_mat, space.nb_ids)
        if len(cols):
            space_a = with_attributes(space, A_mat, A_names, cols)
            log.info("card attributes: %d deck-level columns (%s)", space_a.n_attr, ", ".join(space_a.attr_names))
    xprior = None
    if crossset != "off":
        xprior = xset.build_prior(xset.other_set_effects(set_code, structure.FEATURE_NAMES), structure.FEATURE_NAMES, space.sd)
    xinfo = xset.describe(xprior, set_code) if crossset != "off" else dict(available=False)

    # ---- card pairs (synergy), archetypes (descriptive + card x archetype columns), opponent-matchup modes
    from sealed_lab.model import archetypes as arch_mod
    from sealed_lab.model import opponent as opp_mod
    from sealed_lab.model import playdraw as playdraw_mod
    from sealed_lab.model import synergy as syn_mod

    csort = cards.sort("card_id")
    names = csort["name"].to_list()
    is_land_c = csort["is_land"].to_numpy()
    is_basic_c = csort["is_basic"].to_numpy()
    groups = syn_mod.deck_groups(games, space.keys)
    pair_ab, pair_games, space_s, lift, syn_info = np.zeros((0, 2), np.int64), np.zeros(0), None, None, None
    if synergy != "off":
        pair_ab, pair_games = syn_mod.select_pairs(space, wts, is_land_c, min_games=min_pair_games or syn_mod.MIN_PAIR_GAMES)
        if len(pair_ab):
            space_s = design.with_pairs(space, pair_ab)
            lift = syn_mod.lift_table(space, groups)
            syn_info = dict(names=names, games=pair_games, lift=lift)
            log.info("synergy: %d card pairs with >= %d shared games", len(pair_ab), min_pair_games or syn_mod.MIN_PAIR_GAMES)
    arch = deck_arch = ca_cols = space_c = None
    if len(space.keys) >= 400:
        spell_c = ~is_land_c & ~is_basic_c
        Xc = arch_mod.deck_matrix(space.keys, inp.deck_cards, len(names), spell_c)
        arch = arch_mod.build(space.keys, Xc, groups, names, seed=seed)
        deck_arch = arch._labels                                           # type: ignore[attr-defined]
        arch_mod.attach_win_rates(arch, games.select("deck_key", "fmt", "draft_id", "won"), space.keys, deck_arch, seed=seed)
        log.info("archetypes: %d clusters over %d decks", arch.n, len(space.keys))
        if archcond != "off" and arch.n > 0:
            ca_cols = arch_mod.columns_for_conditional(deck_arch, design.card_copies(space), wts, space.nb_ids, arch.n,
                                                       min_games=max(100.0, 0.003 * float(wts.sum())))   # 0.3% of all games
            if len(ca_cols):
                space_c = design.with_archetype_columns(space, deck_arch, ca_cols)
    opp_modes = opp_mod.MODES if matchups != "off" else None
    tmp = cardvalue.make_tmpdir(config.cache_dir(set_code))
    from sealed_lab.model import glm, memory, parallel, vcache

    # the validation cache: every stored hold-out result is keyed by the content of the data it saw, the code, and its settings
    t_sig = time.time()
    vc = val_cache(set_code, games, cards)
    log.info("validation cache key: data hashed in %.1fs", time.time() - t_sig)
    try:
        # all independent fits below (recency split, the plain chain and every component test of each format, the deployed
        # combination, the final fit) are driven by threads that share ONE process pool
        # the memory budget: concurrent fits are admitted only while their estimated footprint fits in 60% of the RAM that is
        # available now (after the setup above), see memory.py; with one core everything is sequential anyway
        with (memory.budget() if n_jobs > 1 else contextlib.nullcontext()) as mem_budget,                 (glm.shared_pool(n_jobs) if n_jobs > 1 else contextlib.nullcontext()),                 (parallel.Runner(cancel, max_active=8) if n_jobs > 1 else parallel.InlineRunner(cancel)) as rn:
            if mem_budget is not None:
                log.info(mem_budget.summary())
            cx = rn.cancel
            nf = len(formats)

            def gated(nbytes, fn, *a, **k):
                """`fn(*a, **k)` inside a reservation of `nbytes` of the main-process memory budget."""
                with rn.slot(nbytes):
                    return fn(*a, **k)

            # ---- diagnostics that need nothing from the fits (simulator tables, realism against the replay data, replay
            #      section): computed while the validation runs instead of after the final fit
            def diag_realism():
                try:
                    from sealed_lab.model import simcheck

                    return simcheck.realism(set_code, n_jobs, cancel=cx, regimes=sim_regimes or None), False
                except Cancelled:
                    raise
                except Exception:  # noqa: BLE001 - the realism table is a diagnostic; never fail a calculation over it
                    log.exception("simulator realism check skipped")
                    return None, True

            def diag_replay():
                try:
                    from sealed_lab.model import replaystats

                    return replaystats.render_replay_section(replaystats.replay_summary(set_code, cancel=cx, n_jobs=n_jobs)), False
                except Cancelled:
                    raise
                except Exception:  # noqa: BLE001 - the replay section is optional; never fail a calculation over it
                    log.exception("replay section skipped")
                    return None, True

            fut_sim = rn.spawn(gated, 512 * memory.MB, validate.simulator_validation, games, inp.decks, table, keys, cards, inp.lazy["hands"],
                               structure.TABLE_NAMES, v2_tables=tabs_v2 or None)
            fut_realism = rn.spawn(gated, 1 * memory.GB, diag_realism)
            fut_replay = rn.spawn(gated, 1 * memory.GB, diag_replay)
            # ---- recency weighting, step 1: pick a half-life on the training part only
            inner: dict[str, dict] = {}
            chosen = None
            fut_inner: dict[str, object] = {}
            if recency != "off":
                mux_in = parallel.ProgressMux(prog, 0.10, 0.18, nf)
                for i, f in enumerate(formats):
                    fut_inner[f] = rn.spawn(gated, memory.estimate_chain_bytes(n_games.get("draft", 0) + n_games.get("bo1", 0) + n_games[f]),
                                            validate.recency_inner, f, games, space, tmp, n_jobs, seed=seed + i, cancel=cx,
                                            progress=mux_in.slot(i), cache=vc, scope=recency_scope, opts=opts)
            h_cell: dict = {}
            h_lock = threading.Lock()

            def half_life_for_test(f: str | None = None):
                """h_test of format `f` (per-format scope: as soon as ITS inner split is done) or of all formats (pooled scope: as soon
                as every format's inner split is done). The component tests that do not need it never wait."""
                with h_lock:
                    key_ = f if recency_scope == "format" else "h"
                    if key_ not in h_cell:
                        inn = {g: rn.result(fut_inner[g]) for g in ([f] if recency_scope == "format" else formats)} if recency != "off" else {}
                        ch = validate.choose_half_life(inn) if recency != "off" else None
                        h_cell[key_] = dict(inner=inn, chosen=ch, h=(ch if recency == "auto" else (None if recency == "off" else
                                            (ch if ch is not None else validate.best_finite_half_life(inn)))))
                    return h_cell[key_]["h"]

            # ---- out-of-sample validation (fitted on the earlier 75% only); step 2 of the recency test rides along
            mux_val = parallel.ProgressMux(prog, 0.18, 0.30, nf)
            keep: dict[str, dict] = {}
            fut_val = {f: rn.spawn(validate.run_validation, f, games, space, cards, inp.lazy["hands"], tmp, n_jobs, seed=seed + i,
                                   n_rep=n_rep or validate.N_REP, cancel=cx, progress=mux_val.slot(i),
                                   recency_h=(lambda f=f: half_life_for_test(f)) if recency != "off" else None, space_a=space_a,
                                   xprior=xprior, recency_scope=recency_scope, opts=opts,
                                   flags_test=oppflags != "off", space_s=space_s, space_c=space_c, opp_modes=opp_modes, syn_info=syn_info, space_m=space_m,
                                   space_k=space_k if castability != "off" else None, cache=vc, runner=rn, keep=keep,
                                   playdraw=playdraw != "off")
                       for i, f in enumerate(formats)}
            val: dict[str, dict] = {}
            for f in formats:
                val[f] = rn.result(fut_val[f])
                if val[f].get("status") != "ok":
                    warnings.append(f"{f}: {val[f].get('reason')}")
            if recency_scope == "format":
                for f in formats:
                    half_life_for_test(f)
                inner = {f: h_cell[f]["inner"][f] for f in formats if f in h_cell[f]["inner"]}
                chosen = {f: h_cell[f]["chosen"] for f in formats}
                rec = decide_recency_formats(recency, chosen, inner, val, confirm)
            else:
                half_life_for_test()
                inner, chosen = h_cell["h"]["inner"], h_cell["h"]["chosen"]
                rec = decide_recency(recency, chosen, inner, val, confirm)
            log.info("recency weighting: %s (%s)", "applied" if rec["applied"] else "not applied", rec["reason"])
            att = decide_component(attributes, "attributes", val, space_a is not None,
                                   "no card attribute is carried by enough cards" if attributes != "off" else "", "the card-attribute prior",
                                   confirm)
            xs = decide_component(crossset, "crossset", val, xprior is not None,
                                  "no other set with a model on this machine: nothing to borrow from (no-op)" if crossset != "off" else "",
                                  "the cross-set structure prior", confirm)
            syn = decide_component(synergy, "synergy", val, space_s is not None,
                                   f"no card pair shares {min_pair_games or syn_mod.MIN_PAIR_GAMES} games" if synergy != "off" else "",
                                   "the card-pair synergy layer", confirm)
            acd = decide_component(archcond, "archcond", val, space_c is not None,
                                   "too few decks to cluster, or no card is played in enough games within any archetype" if archcond != "off" else "",
                                   "archetype-conditional card values", confirm)
            of = decide_oppflags(oppflags, val, formats, set_code, confirm)
            log.info("opponent colour flags: %s (%s)", "applied" if of["applied"] else "not applied", of["reason"])
            games_f = design.with_opp_flags(games) if of["applied"] else games          # the games every later fit is built from
            if of["applied"] and opp_modes:
                # the matchup layer sits on top of the chain: with the flags deployed it is tested on THAT chain (second pass, only "MO")
                keep_f: dict[str, dict] = {}
                fut_val2 = {f: rn.spawn(validate.run_validation, f, games_f, space, cards, inp.lazy["hands"], os.path.join(tmp, "flags_base"), n_jobs, seed=seed + i,
                                        n_rep=n_rep or validate.N_REP, cancel=cx, opts=opts, opp_modes=opp_modes, cache=vc, runner=rn,
                                        keep=keep_f, only=("MO",))
                            for i, f in enumerate(formats) if val[f].get("status") == "ok"}
                for f, fu_ in fut_val2.items():
                    v2 = rn.result(fu_)
                    val[f]["opponent_plain_base"] = val[f].get("opponent")
                    val[f]["opponent"] = v2.get("opponent")
            mo = decide_matchups(matchups, val, formats, set_code, confirm)
            pdw = decide_component(playdraw, "playdraw", val, True, "", "the card x play/draw interactions", confirm)
            mm = decide_component(_MANASIM_AS_COMPONENT[manasim], "manasim", val, space_m is not None,
                                  "the calibrated simulator could not be computed" if manasim != "v1" else "",
                                  "the calibrated mana simulator (v2)", confirm)
            mm["mode"] = manasim
            mm["applied"] = bool(mm["applied"] and space_m is not None)
            ck = decide_castability(castability, val, space_k is not None,
                                    "castability could not be computed" if castability != "off" else "", confirm)
            ck["applied"] = bool(ck["applied"] and space_k is not None)
            syn.update(n_pairs=int(len(pair_ab)), min_games=int(min_pair_games or syn_mod.MIN_PAIR_GAMES), max_pairs=syn_mod.MAX_PAIRS,
                       lift_top=syn_mod.lift_summary(lift, space, names) if lift is not None else [])
            acd.update(n_columns=0 if ca_cols is None else int(len(ca_cols)), n_archetypes=0 if arch is None else arch.n)
            att.update(n_attr=0 if space_a is None else space_a.n_attr, attr_names=[] if space_a is None else list(space_a.attr_names))
            xs["prior"] = xinfo
            log.info("card-attribute prior: %s (%s)", "applied" if att["applied"] else "not applied", att["reason"])
            log.info("cross-set structure prior: %s (%s)", "applied" if xs["applied"] else "not applied", xs["reason"])
            for what, d_ in (("card-pair synergy layer", syn), ("archetype-conditional values", acd), ("matchup layer", mo), ("play/draw layer", pdw),
                             ("calibrated mana simulator (v2)", mm), ("castability", ck)):
                log.info("%s: %s (%s)", what, "applied" if d_["applied"] else "not applied", d_["reason"])
            _check(cancel)

            def final_space():
                """(base, fspace, fnames, attr_load, theta_prior, extras, n_pair, n_arch) for the components currently applied."""
                base_ = space_m if mm["applied"] else space         # the deck space the final fit and the artifact are built on
                fnames_ = list(structure.FEATURE_NAMES)
                if ck["applied"]:                                  # castability on top of whichever mana simulator is deployed
                    base_ = cast_space(inp, games, keys, tabs_v2 if mm["applied"] else {r_: table for r_ in cast_tabs}, cast_tabs)
                    fnames_ = fnames_ + [cast_mod.FEATURE_NAME]
                if att["applied"]:
                    fs_ = with_attributes(base_, A_mat, A_names, cols) if base_ is not space else space_a
                else:
                    fs_ = base_
                al_ = fs_.attr_load if att["applied"] else None
                tp_ = xprior if xs["applied"] else None
                if tp_ is not None and base_ is not space:
                    tp_ = xset.build_prior(xset.other_set_effects(set_code, fnames_), fnames_, base_.sd)
                ex_, npair_, narch_ = [], 0, 0
                if syn["applied"] and len(pair_ab):
                    fs_ = design.with_pairs(fs_, pair_ab)
                    ex_.append(dict(name="pairs", n=len(pair_ab)))
                    npair_ = len(pair_ab)
                if acd["applied"] and ca_cols is not None and len(ca_cols):
                    fs_ = design.with_archetype_columns(fs_, deck_arch, ca_cols)
                    ex_.append(dict(name="arch", n=len(ca_cols)))
                    narch_ = len(ca_cols)
                return base_, fs_, fnames_, al_, tp_, ex_, npair_, narch_

            base, fspace, fnames, attr_load, theta_prior, extras, n_pair_f, n_arch_f = final_space()

            # ---- the final fit on all data (chain, event bootstrap, matchup layer) for one configuration of components
            prior_names = {n for n in ("draft", "bo1") if n_games.get(n, 0) >= validate.MIN_PRIOR_GAMES}
            need = set(formats)
            if "draft" in prior_names:
                need.add("draft")
            if "bo3" in formats and "bo1" in prior_names:
                need.add("bo1")

            def final_fit(tag: str, fspace_, rec_applied: bool, half_life, attr_load_, theta_prior_, extras_, mo_formats: list[str],
                          cancel_: object, pd_on: bool = False, games_=None) -> dict:
                """Everything that depends on WHICH components are deployed.  Pure function of its arguments (no shared state is
                modified), so it can run speculatively while the deployed combination is still being validated."""
                d = os.path.join(tmp, f"final_{tag}")
                os.makedirs(d, exist_ok=True)
                with rn.slot(memory.estimate_opp_bytes(sum(n_games.get(n, 0) for n in need))):
                    stages_ = {n: build_stage(n, (games if games_ is None else games_).filter(pl.col("fmt") == n), fspace_) for n in ("draft", "bo1", "bo3") if n in need}
                    rec_info = {}
                    if rec_applied:
                        t_ref = games["game_time"].max()          # weights measured back from the newest game in the corpus
                        rec_info["t_ref"] = str(t_ref)
                        stages_ = validate.weight_stages(stages_, t_ref, half_life)
                        rec_info["ess"] = {n: float(st.w.sum() ** 2 / max((st.w ** 2).sum(), 1e-12)) for n, st in stages_.items()}
                    prior_ok_ = {n for n in stages_ if n in prior_names}
                    prog(0.32, "Fitting the model chain on all data")
                    fits_ = cardvalue.run_chain(stages_, space.n_card, fspace_.n_feat, d, n_jobs, seed=seed + 500, cancel=cancel_,
                                                progress=lambda fr, m: prog(0.32 + 0.20 * fr, m), prior_ok=prior_ok_,
                                                attr_load=attr_load_, theta_prior=theta_prior_, extras=extras_ or None, opts=opts)
                    coefs_ = cardvalue.bootstrap(stages_, fits_, n_boot, d, n_jobs, formats, seed=seed + 900, cancel=cancel_,
                                                 progress=lambda fr, m: prog(0.52 + 0.20 * fr, m), prior_ok=prior_ok_)
                    # ---- matchup layer: deltas on the pooled games of the final fit, event bootstrap
                    opp_model_ = opp_final_ = None
                    if mo_formats:
                        tested = [(f, val[f]["opponent"]) for f in formats
                                  if f in mo_formats and val[f].get("status") == "ok" and val[f].get("opponent")]
                        mode_f, sigma_f = (tested[0][1]["mode"], tested[0][1]["sigma"]) if tested else (opp_mod.MODES[0], 0.01)
                        prog(0.72, "Fitting card x opponent interactions (all games)")
                        opp_model_, opp_final_ = opp_mod.fit_final(stages_, {n: f_.coef for n, f_ in fits_.items()}, fspace_,
                                                                   space.n_cards_total, mode_f, sigma_f, n_boot, d, n_jobs,
                                                                   seed=seed + 1300, cancel=cancel_,
                                                                   progress=lambda fr, m: prog(0.72 + 0.02 * fr, m))
                        opp_final_ = validate.finalize_opponent_summary(opp_final_, csort["name"].to_numpy()[space.nb_ids].tolist())
                    # ---- play/draw layer: gammas on the pooled games of the final fit, event bootstrap
                    pd_model_ = pd_final_ = None
                    if pd_on:
                        tested_pd = [val[f]["playdraw"] for f in formats if val[f].get("status") == "ok" and val[f].get("playdraw")]
                        sigma_pd = tested_pd[0]["sigma"] if tested_pd else 0.01
                        prog(0.74, "Fitting card x play/draw interactions (all games)")
                        pd_model_, pd_final_ = playdraw_mod.fit_final(stages_, {n: f_.coef for n, f_ in fits_.items()}, space.n_card,
                                                                      space.nb_ids, space.n_cards_total, sigma_pd, n_boot, d, n_jobs,
                                                                      seed=seed + 1700, cancel=cancel_)
                return dict(stages=stages_, fits=fits_, coefs=coefs_, opp_model=opp_model_, opp_final=opp_final_, rec=rec_info,
                            pd_model=pd_model_, pd_final=pd_final_)

            # ---- the deployed COMBINATION on the hold-out: each component was tested alone against the plain chain above.
            #      The final fit of that very combination does not depend on this test's outcome unless it vetoes the
            #      combination (rare), so it is started now, in parallel, and discarded if the veto comes.
            decisions = dict(recency=rec, attributes=att, crossset=xs, synergy=syn, archcond=acd, matchups=mo, manasim=mm,
                             castability=ck, oppflags=of)
            applied_now = [k_ for k_, d_ in decisions.items() if d_["applied"]]
            deployed = dict(components=applied_now, formats={}, fallback=False)
            spec_flag = threading.Event()
            mo_fmts = [f for f in formats if f in mo["applied_formats"]] if mo["applied"] else []     # formats the matchup layer is applied in
            fut_final = rn.spawn(final_fit, "a", fspace, bool(rec["applied"]), rec["half_life"], attr_load, theta_prior, extras,
                                 mo_fmts, parallel.CombinedEvent(cx, spec_flag), bool(pdw["applied"]), games_f)
            if applied_now:
                mo_tested = [val[f]["opponent"] for f in mo_fmts if val[f].get("status") == "ok" and val[f].get("opponent")]
                # the deployed combination of a format holds the matchup layer only where it is applied (pooled: bo3 but maybe not bo1)
                comps_f = {f: [k_ for k_ in applied_now if k_ != "matchups" or f in mo_fmts] for f in formats}
                mux_dp = parallel.ProgressMux(prog, 0.30, 0.318, nf)
                fut_dep = {f: rn.spawn(gated, memory.estimate_opp_bytes(n_games.get('draft', 0) + n_games.get('bo1', 0) + n_games[f]),
                                       validate.validate_deployed, f, games_f, space, fspace, cards, tmp, n_jobs, comps_f[f], seed=seed + i,
                                       n_rep=n_rep or validate.N_REP, cancel=cx, progress=mux_dp.slot(i),
                                       recency_h=rec["half_life"] if rec["applied"] else None, attr_load=attr_load,
                                       theta_prior=theta_prior, extras=extras,
                                       opp_mode=(mo_tested[0]["mode"] if (f in mo_fmts and mo_tested) else None),
                                       m_ref=keep.get(f), cache=vc, opts=opts)
                           for i, f in enumerate(formats) if comps_f[f]}
                for f in formats:
                    deployed["formats"][f] = (rn.result(fut_dep[f]) if f in fut_dep else
                                              dict(status="skipped", reason="no optional component is applied in this format", components=[]))
                worse = [f for f, o in deployed["formats"].items()
                         if o.get("status") == "ok" and o["hi"] < 0 and o["gain"] <= -MIN_GAIN]
                if worse:
                    deployed["fallback"] = True
                    deployed["reason"] = ("the combination of " + ", ".join(applied_now) + " is materially WORSE than the plain chain on "
                                          "the hold-out in " + ", ".join(worse) + " although each passed alone; the plain chain is deployed")
                    warnings.append(deployed["reason"])
                    for d_ in decisions.values():
                        if d_["applied"]:
                            d_["applied"] = False
                            d_["reason"] = "dropped: " + deployed["reason"]
                    mo["applied_formats"], mo["source"] = [], {}
                    spec_flag.set()                                   # discard the speculative final fit of the vetoed combination
                    try:
                        rn.result(fut_final)
                    except Cancelled:
                        if cancel is not None and cancel.is_set():
                            raise
                    base, fspace, fnames, attr_load, theta_prior, extras, n_pair_f, n_arch_f = final_space()
                    fut_final = rn.spawn(final_fit, "b", fspace, False, None, attr_load, theta_prior, extras, [], cx, bool(pdw["applied"]), games)
                else:
                    deployed["reason"] = ("the deployed combination (" + ", ".join(applied_now) + ") is not materially worse than "
                                          "the plain chain in any format")
                log.info("deployed combination: %s", deployed["reason"])
            _check(cancel)

            ff = rn.result(fut_final)
            stages, fits, coefs = ff["stages"], ff["fits"], ff["coefs"]
            opp_model, opp_final = ff["opp_model"], ff["opp_final"]
            pd_model, pd_final = ff["pd_model"], ff["pd_final"]
            if rec["applied"]:
                rec.update(ff["rec"])
            for fit_ in fits.values():
                warnings += fit_.warnings
            sim_rows = rn.result(fut_sim)
            realism, realism_failed = rn.result(fut_realism)
            rep_html, replay_failed = rn.result(fut_replay)
            if mem_budget is not None:
                log.info(mem_budget.summary())
    finally:
        cardvalue.cleanup(tmp)

    # ---- hand check
    prog(0.73, "Opening-hand check")
    parts = []
    for n, st in stages.items():
        e = st.prob.eta(fits[n].coef)
        nm = st.games["num_mulligans"].fill_null(0).to_numpy() == 0
        parts.append((st.games["game_key"].to_numpy()[nm], e[nm], st.prob.y[nm]))
    hc_res = handcheck.compute_psi(parts, inp.lazy["hands"], space)
    psi, psi_se = hc_res["psi"], hc_res["se"]
    _check(cancel)

    # ---- artifact pieces
    draws = {}
    se_tab = {}
    hcs = {}
    cards_out = cards.sort("card_id")
    extra_cols = {}
    beta_tot: dict[str, np.ndarray] = {}
    prog(0.78, "Reference distribution")
    reference_q = {}
    for f in formats:
        st, ft = stages[f], fits[f]
        k = ft.k
        enc_ix = {nm: i_ for i_, nm in enumerate(st.enc.names)}
        draws[f] = _draws(coefs[f], k, fspace, st.ref_row, (enc_ix["mull1"], enc_ix["mull2p"]))
        ex_f = draws[f].pop("extra")
        if n_pair_f:
            draws[f]["pair"] = ex_f[:, :n_pair_f]
        if n_arch_f:
            draws[f]["arch"] = ex_f[:, n_pair_f:n_pair_f + n_arch_f]
        se_c = ft.card_se
        beta_tot[f] = ft.beta_total
        hcs[f] = handcheck.kappa_and_flags(ft.beta_total, se_c, psi, psi_se, st.exposure)
        extra_cols[f"beta_{f}"] = _full(space, ft.beta_total)
        extra_cols[f"beta_attr_{f}"] = _full(space, ft.beta_attr)
        extra_cols[f"se_{f}"] = _full(space, se_c)
        extra_cols[f"exposure_{f}"] = _full(space, st.exposure)
        extra_cols[f"low_data_{f}"] = _full(space, (st.exposure < 2000).astype(float), fill=0.0) > 0.5
        extra_cols[f"psi_flag_{f}"] = _full(space, hcs[f]["flag"].astype(float)) > 0.5
        bkeys = reference.build_deck_keys(inp.builds, f)
        rows = space.rows(bkeys)
        keep_f = sim_regimes.get(config.format_regime(f)) or config.format_regime(f)
        mull_p = reference.mull_inputs(inp.decks, bkeys, lambda nl_, nc_, r_=keep_f: manasim_mod.mull_split(nl_, nc_, r_))
        deck_coefs = np.hstack([draws[f]["beta"][:, space.nb_ids], draws[f]["theta"]])
        extra_ref = (fspace.D[:, fspace.extra_start:], ex_f) if ex_f.shape[1] else None
        reference_q[f] = reference.reference_q(base, rows, deck_coefs, draws[f]["ref"], extra=extra_ref,
                                               regime=config.format_regime(f) if base.regime_phi else None,
                                               mull=(mull_p, draws[f]["mull"]))
        se_tab[f] = ft.se_total[k + space.n_card:k + space.n_card + fspace.n_feat]
    extra_cols["psi"] = _full(space, psi)
    extra_cols["psi_se"] = _full(space, psi_se, fill=np.nan)
    cards_art = cards_out.with_columns([pl.Series(k, v) for k, v in extra_cols.items()])

    # ---- simulator validation (computed above, next to the validation; only the failure notice is raised here)
    prog(0.84, "Validating the mana simulator")
    if realism_failed:
        degrade("the simulator realism check could not be computed (the report section is missing)")
    _check(cancel)

    # ---- report
    prog(0.90, "Writing report")
    built_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    from datetime import datetime, timezone
    built_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stage_info = {n: validate.stage_summary(f, stages[n]) for n, f in fits.items()}
    theta_info = {f: {nm: dict(est=float(fits[f].theta[j]), se=float(se_tab[f][j]))
                      for j, nm in enumerate(fnames)} for f in formats}
    card_sections = {f: _card_section(f, cards, space, beta_tot[f], fits[f].card_se, stages[f].exposure, psi, psi_se, hcs[f])
                     for f in formats}
    # the attribute coefficients of the deployed fit replace the validation-fit ones in the report
    if att["applied"]:
        att["gamma_source"] = "final"
        att["gamma"] = {f: validate.gamma_table(fits[f], fspace) for f in formats}
    # synergy / archetype / matchup report data
    if syn["applied"] and n_pair_f:
        syn["final"] = {f: syn_mod.pair_summary(pair_ab, fits[f].extra[:n_pair_f], fits[f].extra_se[:n_pair_f], pair_games, names, lift,
                                                space.col_of_card) for f in formats}
    arch_rep = arch_mod.archetype_report(arch) if arch is not None else None
    acd["archetypes"] = arch_rep
    if opp_final is not None:
        mo["final"] = opp_final
    if pd_final is not None:
        pdw["final"] = pd_final
    audit = _audit(inp)
    R = dict(set_code=set_code, built_at=built_at, code_version=CODE_VERSION, n_games=n_games, n_boot=n_boot,
             seconds=time.time() - t_start, formats=formats, validation=val, warnings=warnings, stages=stage_info,
             cards=card_sections, theta=theta_info, feature_names=fnames, simulator=sim_rows, audit=audit,
             psi_games=hc_res["n_games"], oppflags=of, recency=rec, attributes=att, crossset=xs, synergy=syn, archcond=acd, matchups=mo, playdraw=pdw,
             archetypes=arch_rep, manasim=mm, sim_version=2 if mm["applied"] else 1, realism=realism, deployed=deployed,
             extra_draft=xd, castability=dict(ck, stats=cast_stats, support=support_ref, revision=cast_mod.CAST_REVISION, slack=cast_mod.SLACK,
                              categories={k_: v_ for k_, v_ in cast_mod.category_summary(cards).items()}))
    html_ = validate.render_report(R)
    # replay data (turn by turn): simulator vs real play, value of curving out, per-card cast patterns (rendered above)
    prog(0.92, "Replay data")
    if replay_failed:
        degrade("the replay-data section of the report could not be computed (it is missing)")
    elif rep_html is not None:
        html_ = html_.replace("</main></body></html>", rep_html + "</main></body></html>")
    ref_logit ={f: dict(ref=float(draws[f]["ref"][0]), draws_sd=float(np.std(draws[f]["ref"][1:])) if n_boot > 1 else 0.0,
                         reference_wr=float(expit(np.median(np.log(np.clip(reference_q[f][0], 1e-6, 1 - 1e-6) /
                                                                     (1 - np.clip(reference_q[f][0], 1e-6, 1 - 1e-6)))))))
                 for f in formats}
    headline = {}
    for f in formats:
        v = val[f]
        if v.get("status") == "ok":
            d = v["metrics"]["B1"]["vs_M"]["logloss"]
            headline[f] = dict(status="ok", gain_vs_B1=d["diff"], ci=[d["lo"], d["hi"]], model_logloss=v["metrics"]["M"]["logloss"],
                               model_auc=v["metrics"]["M"]["auc"], calibration_slope=v["calibration"]["slope"],
                               calibration={k_: v["calibration"].get(k_) for k_ in (
                                   "intercept", "intercept_ci", "slope_ci", "level_logit", "level_logit_ci", "level_pp", "level_pp_ci", "n")},
                               holdout_start=v.get("holdout_start"), base_rate_train=v.get("base_rate_train"),
                               base_rate_holdout=v.get("base_rate_holdout"))
        else:
            headline[f] = dict(status=v.get("status"), reason=v.get("reason"))
    meta = dict(set_code=set_code, formats=formats, built_at=built_iso, built_corpus_at=inp.manifest.get("built_at"),
                code_version=CODE_VERSION, n_games=n_games, n_boot=n_boot, n_draws=n_boot + 1, seed=seed,
                feature_names=fnames, phi_mean=base.mu.tolist(), phi_sd=base.sd.tolist(),
                n_cards=space.n_cards_total, stages=stage_info, ref=ref_logit, warnings=warnings, validation=headline,
                deployed=deployed, degraded=list(degraded), source_files=source_files,
                recency=dict(mode=rec["mode"], applied=rec["applied"], half_life=rec["half_life"], reason=rec["reason"],
                             scope=recency_scope),
                prior_strength=dict(sigma_cv=opts.sigma_cv, stack=bool(opts.stack), confirm=bool(confirm), leak_free=bool(opts.leak_free),
                                    theta_sigma=opts.theta_sigma,
                                    stages={n: dict(sigma=float(f_.sigma), scheme=f_.cv_scheme, w=float(f_.w), grid=f_.cv_grid, loss=f_.cv_loss,
                                                    w_grid=f_.w_grid, w_loss=f_.w_loss, theta_sd=cardvalue.theta_sd_of(f_),
                                                    theta_cv=f_.theta_cv_sd is not None, theta_grid=f_.theta_grid,
                                                    theta_loss=f_.theta_loss) for n, f_ in fits.items()}),
                attributes=dict(mode=att["mode"], applied=att["applied"], reason=att["reason"], n_attr=att["n_attr"],
                                names=att["attr_names"],
                                gain={f: {k_: val[f]["attributes"][k_] for k_ in ("gain", "lo", "hi")} for f in formats
                                      if val[f].get("status") == "ok" and val[f].get("attributes")}),
                synergy=_comp_meta(syn, val, "synergy", formats, n_pairs=int(len(pair_ab))),
                archcond=_comp_meta(acd, val, "archcond", formats, n_columns=acd["n_columns"]),
                matchups=_comp_meta(mo, val, "opponent", formats, encoding=None if opp_model is None else opp_model.mode,
                                    sigma=None if opp_model is None else opp_model.sigma,
                                    applied_formats=list(mo["applied_formats"]) if mo["applied"] else [],
                                    source=dict(mo["source"]) if mo["applied"] else {}, pooled=mo.get("pooled") or {},
                                    per_set_reason=mo.get("per_set_reason")),
                playdraw=_comp_meta(pdw, val, "playdraw", formats, sigma=None if pd_model is None else pd_model.sigma),
                manasim=_comp_meta(mm, val, "manasim", formats, version=2 if mm["applied"] else 1, revision=manasim_mod.SIM2_REVISION,
                                   regimes={f: config.format_regime(f) for f in formats},
                                   keep={r: (None if sr.keep is None else [[list(row) for row in m_] for m_ in sr.keep])
                                         for r, sr in sim_regimes.items()}, keep_info=keep_info),
                oppflags=dict(_comp_meta(of, val, "oppflags", formats), source=of.get("source"), pooled=of.get("pooled"),
                              matchups_gain_on_plain_base={f: {k_: val[f]["opponent_plain_base"][k_] for k_ in ("gain", "lo", "hi")}
                                                           for f in formats if (val[f].get("opponent_plain_base") or {}).get("gain") is not None},
                              matchups_gain_on_flags_base={f: {k_: val[f]["opponent"][k_] for k_ in ("gain", "lo", "hi")}
                                                           for f in formats if of["applied"] and (val[f].get("opponent") or {}).get("gain") is not None}),
                castability=dict(_comp_meta(ck, val, "castability", formats), n_trials=cast_mod.manasim.DEFAULT_TRIALS,
                                 revision=cast_mod.CAST_REVISION, slack=cast_mod.SLACK, support=support_ref, stats=cast_stats,
                                 rule=ck.get("rule")),
                archetypes=dict(n=0 if arch is None else arch.n),
                extra_draft=_extra_draft_meta(xd),
                crossset=dict(mode=xs["mode"], applied=xs["applied"], reason=xs["reason"], sets=xinfo.get("sets", []),
                              gain={f: {k_: val[f]["crossset"][k_] for k_ in ("gain", "lo", "hi")} for f in formats
                                    if val[f].get("status") == "ok" and val[f].get("crossset")}),
                land_support=_land_support(games, inp.decks, formats),
                seconds=round(time.time() - t_start, 1))
    pairs_art = archcols_art = None
    if n_pair_f:
        ca_, cb_ = space.col_of_card[pair_ab[:, 0]], space.col_of_card[pair_ab[:, 1]]
        pairs_art = dict(ab=pair_ab, games=pair_games, lift=lift["lift"][ca_, cb_] if lift is not None else np.full(len(pair_ab), np.nan),
                         se={f: fits[f].extra_se[:n_pair_f] for f in formats})
    if n_arch_f:
        archcols_art = dict(ca=ca_cols, se={f: fits[f].extra_se[n_pair_f:n_pair_f + n_arch_f] for f in formats})
    artifact.save_extras(config.model_dir(set_code), pairs_art, archcols_art, arch, opp_model, formats, playdraw=pd_model)
    artifact.save_artifact(set_code, meta, cards_art, draws, reference_q, html_, R)
    prog(1.0, "Model ready")
    return meta


def _comp_meta(d: dict, val: dict, key: str, formats: list[str], **extra) -> dict:
    """meta.json summary of a validated component."""
    return dict(mode=d["mode"], applied=d["applied"], reason=d["reason"],
                gain={f: {k_: val[f][key][k_] for k_ in ("gain", "lo", "hi")} for f in formats
                      if val[f].get("status") == "ok" and val[f].get(key)}, **extra)


def _extra_draft_meta(xd: dict) -> dict:
    """meta.json summary of the extra-draft decision (per format: whole hold-out gain and its confirmation halves)."""
    return dict(mode=xd["mode"], available=xd["available"], applied=xd["applied"], reason=xd["reason"], events=xd["events"],
                games=xd["games"], core_draft_games=xd["core_draft_games"], seconds=xd.get("seconds"),
                gain={f: {k_: g[k_] for k_ in ("gain", "lo", "hi")} for f, g in xd["gain"].items()},
                halves={f: {x: {k_: g["halves"][x][k_] for k_ in ("gain", "lo", "hi", "lo90")} for x in "AB"} for f, g in xd["gain"].items()})


def _land_support(games: pl.DataFrame, decks: pl.DataFrame, formats: list[str]) -> dict[str, dict[str, float]]:
    """Per format: share of games played with each land count (the builder stays where the data are)."""
    g = games.select("fmt", "deck_key").join(decks.select("deck_key", "n_lands"), on="deck_key", how="inner")
    out = {}
    for f in formats:
        c = g.filter(pl.col("fmt") == f).group_by("n_lands").len()
        tot = max(int(c["len"].sum()), 1)
        out[f] = {str(int(n)): round(int(k) / tot, 5) for n, k in c.iter_rows() if n is not None}
    return out



def _support_reference(inp: Inputs, games: pl.DataFrame, keys: np.ndarray, formats: list[str]) -> dict:
    """Per format: what real decks look like (spell colours, thinnest colour's sources) - the reference of the support flags."""
    import scipy.sparse as sp

    from sealed_lab.model import castability

    cc = castability.cast_cards(inp.cards)
    offsets, cid, cnt = deck_arrays(keys, inp.deck_cards)
    rows = np.repeat(np.arange(len(keys)), np.diff(offsets))
    M = sp.csr_matrix((cnt.astype(float), (rows, cid)), shape=(len(keys), inp.cards.height))
    ok40 = inp.decks.sort("deck_key")["n_cards"].to_numpy() == 40          # reference = proper 40-card decks
    out = {}
    for f in formats:
        g = games.filter(pl.col("fmt") == f).group_by("deck_key").len().sort("deck_key")
        w = np.zeros(len(keys))
        w[np.searchsorted(keys, g["deck_key"].to_numpy())] = g["len"].to_numpy()
        w = w * ok40
        if w.sum() > 0:
            out[f] = castability.support_reference(M, w, cc)
    return out


def _audit(inp: Inputs) -> dict:
    a = inp.manifest.get("audit", {}) or {}
    out = {}
    if a.get("n_games_by_fmt"):
        out["games per format"] = ", ".join(f"{k}: {v:,}" for k, v in a["n_games_by_fmt"].items())
    if a.get("pool_invariance_rate"):
        out["share of events with invariant deck+sideboard pool"] = ", ".join(
            f"{k}: {('n/a' if v is None else f'{v:.1%}')}" for k, v in a["pool_invariance_rate"].items())
    if a.get("deck_size_hist"):
        h = a["deck_size_hist"]
        tot = sum(h.values()) or 1
        out["deck size distribution"] = ", ".join(f"{k} cards: {v / tot:.2%}" for k, v in sorted(h.items(), key=lambda kv: -kv[1])[:5])
    if "hand_violation_rate" in a:
        out["games where opening+drawn exceed the deck copies"] = f"{a['hand_violation_rate']:.3%}"
    if a.get("rows_dropped"):
        out["game rows dropped (invalid 'won')"] = f"{a['rows_dropped']:,}"
    if a.get("unique_decks") is not None:
        out["unique decks / builds"] = f"{a.get('unique_decks'):,} / {a.get('unique_builds'):,}"
    src = inp.manifest.get("card_meta_sources")
    if src:
        out["card metadata sources"] = ", ".join(f"{k}: {v}" for k, v in src.items())
    files = [f for f in inp.manifest.get("files", []) if f.get("kind") == "game"]
    if files:
        out["game files"] = "; ".join(f"{Path(f['path']).name} ({f.get('rows') or '?'} rows)" for f in files)
    return out
