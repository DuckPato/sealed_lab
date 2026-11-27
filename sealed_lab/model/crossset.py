"""Cross-set prior for the structure terms theta (land count, curve, mana statistics).

Within one set 93% of the decks play 17 lands, so the land / curve / mana terms are weakly identified. Other sets that
already have a model share the same game rules, so their fitted structure effects are a sensible prior mean for this
set's draft stage.  Effects are stored per standard deviation of the feature *in that set*; they are converted through raw
feature units (theta_std / sd_other) and back into this set's standardisation (x sd_this).  The prior SD is the
between-set spread of the converted effects, never smaller than the structure sigma the chain would use anyway.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from sealed_lab import config

log = logging.getLogger(__name__)


def other_set_effects(set_code: str, feature_names: list[str], home: Path | None = None) -> list[dict]:
    """Raw-unit structure effects (per unit of each feature) of every OTHER set that has a model.

    One entry per set: dict(set, formats, raw) with raw aligned to `feature_names` (NaN where the other set has no such
    feature). The mean over the set's formats of the final fit (draw 0) is used."""
    root = (Path(home) if home is not None else config.app_home()) / "sets"
    out: list[dict] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        code = d.name.upper()
        if code == set_code.upper() or not (d / "model" / "meta.json").exists() or not (d / "model" / "draws.npz").exists():
            continue
        try:
            meta = json.loads((d / "model" / "meta.json").read_text(encoding="utf-8"))
            if meta.get("provisional"):          # a pick-file model carries the borrowed transfer terms, not this set's own evidence
                continue
            z = np.load(d / "model" / "draws.npz", allow_pickle=False)
            names = list(meta["feature_names"])
            sd = np.asarray(meta["phi_sd"], dtype=float)
            per_fmt = []
            for f in meta["formats"]:
                th = np.asarray(z[f"{f}__theta"][0], dtype=float)
                if th.shape[0] == len(names) == len(sd):
                    per_fmt.append(th / sd)
            if not per_fmt:
                continue
            raw_other = np.mean(per_fmt, axis=0)
        except Exception as e:  # noqa: BLE001 - a broken neighbour set must never break this set's calculation
            log.warning("cross-set prior: cannot read the model of %s (%s)", code, e)
            continue
        pos = {n: i for i, n in enumerate(names)}
        raw = np.array([raw_other[pos[n]] if n in pos else np.nan for n in feature_names])
        out.append(dict(set=code, formats=list(meta["formats"]), raw=raw))
    return out


def build_prior(others: list[dict], feature_names: list[str], sd_this: np.ndarray) -> dict | None:
    """dict(mean, spread, sets, n_sets, raw_mean) in THIS set's standardised units, or None without other sets.

    spread is the between-set standard deviation (ddof 1); it is 0 with a single other set (then only the floor of the
    chain's own structure sigma applies)."""
    if not others:
        return None
    R = np.vstack([o["raw"] for o in others])                       # (n_sets, n_feat) raw units
    with np.errstate(all="ignore"):
        raw_mean = np.nanmean(R, axis=0)
        spread = np.nanstd(R, axis=0, ddof=1) if len(others) > 1 else np.zeros(R.shape[1])
    ok = np.isfinite(raw_mean)
    sd = np.asarray(sd_this, dtype=float)
    mean = np.where(ok, raw_mean * sd, 0.0)
    spread = np.where(np.isfinite(spread), spread, 0.0) * sd
    return dict(mean=mean, spread=spread, sets=[o["set"] for o in others], n_sets=len(others), raw_mean=np.where(ok, raw_mean, 0.0),
                feature_names=list(feature_names))


def describe(prior: dict | None, set_code: str) -> dict:
    """JSON-able summary for the report / meta."""
    if prior is None:
        return dict(available=False, note=f"{set_code} is the only set with data on this machine: no cross-set prior (a no-op)")
    return dict(available=True, sets=prior["sets"], n_sets=prior["n_sets"],
                mean={n: float(m) for n, m in zip(prior["feature_names"], prior["mean"])},
                spread={n: float(s) for n, s in zip(prior["feature_names"], prior["spread"])},
                note=("one other set: its effects are the prior mean, sigma stays at the chain's own structure sigma"
                      if prior["n_sets"] == 1 else
                      f"{prior['n_sets']} other sets: prior mean = their average, sigma floored at the chain's own structure sigma"))
