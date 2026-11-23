"""Read-only: creature counts per archetype group, every calculated set, vs the builder's creature window (central 96%).

Usage: python scripts/archetype_creature_check.py OUT.json   (reads %LOCALAPPDATA%/sealed_lab/sets, writes only OUT.json)"""
import json, os, sys
from pathlib import Path
import numpy as np, polars as pl
from sealed_lab.model.archetypes import Archetypes, deck_matrix

TAIL = 0.02
HOME = Path(os.environ["LOCALAPPDATA"]) / "sealed_lab" / "sets"

def wq(x, w, q):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(w[o]) / w.sum()
    return x[o][min(np.searchsorted(cw, q), len(cw) - 1)]

out = {}
for sd in sorted(HOME.iterdir()):
    m, cd = sd / "model", sd / "corpus"
    arch = Archetypes.load(m)
    if arch is None or not (cd / "games.parquet").exists():
        continue
    cards = pl.read_parquet(m / "cards.parquet", columns=["card_id", "name", "is_creature", "is_land", "is_basic", "types"]).sort("card_id")
    n = int(cards["card_id"].max()) + 1
    is_cre = np.zeros(n, bool); spell = np.zeros(n, bool)
    cid = cards["card_id"].to_numpy()
    is_cre[cid] = cards["is_creature"].fill_null(False).to_numpy() & ~cards["is_land"].fill_null(False).to_numpy()
    spell[cid] = ~cards["is_land"].fill_null(False).to_numpy() & ~cards["is_basic"].fill_null(False).to_numpy()
    d40 = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards"]).filter(pl.col("n_cards") == 40)["deck_key"]
    dc = pl.read_parquet(cd / "deck_cards.parquet").filter(pl.col("deck_key").is_in(d40.implode()))
    games = pl.read_parquet(cd / "games.parquet", columns=["fmt", "deck_key"])
    keys = np.sort(d40.to_numpy().astype(np.int64))
    X = deck_matrix(keys, dc, arch.n_cards, spell[:arch.n_cards] if len(spell) >= arch.n_cards else np.pad(spell, (0, arch.n_cards - len(spell))))
    lab, sim = arch.assign_matrix(X)
    cm = is_cre[:arch.n_cards] if len(is_cre) >= arch.n_cards else np.pad(is_cre, (0, arch.n_cards - len(is_cre)))
    ncre = np.asarray(X @ cm.astype(float)).ravel()
    nspell = np.asarray(X.sum(1)).ravel()
    res = {}
    for fmt in ("bo1", "bo3"):
        g = games.filter(pl.col("fmt") == fmt).group_by("deck_key").len()
        if g.height == 0:
            continue
        gk = g["deck_key"].to_numpy(); gw = g["len"].to_numpy().astype(float)
        row = np.searchsorted(keys, gk); ok = (row < len(keys)) & (keys[np.minimum(row, len(keys) - 1)] == gk)
        r, w = row[ok], gw[ok]
        x = ncre[r]; lo, hi = wq(x, w, TAIL), wq(x, w, 1 - TAIL)
        rows = []
        for a, info in enumerate(arch.info):
            s = lab[r] == a
            if w[s].sum() < 200:      # >= 200 games
                continue
            xa, wa = x[s], w[s]
            rows.append(dict(arch=info["label"], games=int(wa.sum()), decks=int(s.sum()),
                             mean=float(np.average(xa, weights=wa)), p10=float(wq(xa, wa, .10)), p50=float(wq(xa, wa, .5)),
                             p90=float(wq(xa, wa, .90)), below=float(wa[xa < lo].sum() / wa.sum()), above=float(wa[xa > hi].sum() / wa.sum()),
                             spells=float(np.average(nspell[r][s], weights=wa))))
        res[fmt] = dict(lo=int(lo), hi=int(hi), mean=float(np.average(x, weights=w)), games=int(w.sum()), rows=rows)
    out[sd.name] = res
    print(sd.name, {f: (v["lo"], v["hi"], round(v["mean"], 1), len(v["rows"])) for f, v in res.items()}, flush=True)

Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
