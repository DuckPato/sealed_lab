"""The "More detail" statistics of the Analysis tab: per color pair, deck shape, format evolution, Bo1 vs Bo3.  Read-only on the corpus.

    pair_details(g, pairs_sec)                          play vs draw, game length, mulligans, splash pay-off by splash color, per color pair
    shape_details(g, decks, sh)                         skill-adjusted win rate by land count and by creature-count band
    evolution_section(set_code, fmt, g, cards, rows)    the format in early / middle / late thirds of the events: color pairs and card play rates
    compare_pairs(fmt, pairs_this, pairs_other)         color-pair effects of the two formats side by side
    data_notes(...)                                     what the figures rest on

Same honesty rules as `report`: win rates are skill-adjusted (`report._fit`), standard errors are robust to the games of one event
(`glm.cluster_se`), badges Strong |z| >= 3 / Likely |z| >= 2 / Weak, and only Likely / Strong results are ever stated as findings.

Format evolution: the events (draft_id) are sorted by the time of their first game and cut into three thirds with the same number of events.
The events are also split at random (fixed seed) into two halves, so every third has two independent half-estimates.  A color pair or a
card "got better / worse" (or was "played more / less") only when (1) the change late - early is Likely or Strong (z >= 2, the standard
errors of the two thirds added: they hold different events) AND (2) the change has the same sign in both halves of the data AND, for
card play rates, (3) it is at least EVOL_MIN_CHANGE of the decks and each end has at least EVOL_MIN_ELIG eligible decks.  Anything else is
"stable / no clear change".  The estimate of a third is the precision-weighted mean of its two half-estimates.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
import scipy.sparse as sp

from sealed_lab import config, report as R
from sealed_lab.model import design

log = logging.getLogger(__name__)

MIN_SPLASH_DECKS = 60              # decks of a pair that splash a color before that splash is judged
MIN_BAND_DECKS = 100               # decks in a land-count / creature-count band before its win rate is shown
EVOL_MIN_DRAFTS = 450              # events before the format is split into thirds
EVOL_MIN_ELIG = 100                # decks that could play a card, in the early AND the late third
EVOL_MIN_CHANGE = 0.05             # a play rate must move by at least 5 points of decks
EVOL_CARDS_N = 8                   # cards listed per direction
EVOL_SEED = 20240
THIRDS = ("early", "middle", "late")
NOTE_DETAIL = "Observed only: the builder does not use these figures."
SHAPE_DETAIL_NOTE = "For reference: the builder handles curve and land count through the model, these figures are not used by it."


def _eff(eff: float, se: float) -> dict:
    z = eff / se if se > 0 else None
    return dict(effect_pp=eff, se_pp=se, z=z, badge=R.badge_of(z), range=R.range_text(eff, se))


def _clear(b: str) -> bool:
    return b in ("likely", "strong")


def _date(t) -> str:
    return str(t)[:10]


# ---------------------------------------------------------------------------------------------
# per color pair
# ---------------------------------------------------------------------------------------------

def _splash_by_colour(gp: pl.DataFrame, key: str) -> list[dict]:
    """Win rate of the pair's decks that splash a color against its decks that do not, per splash color (skill-adjusted, one fit).
    A deck splashing two colors counts for both."""
    if gp.height < 500:
        return []
    dk = gp.group_by("draft_id", "deck_key").agg(pl.col("splash_colors").first())
    n_dk = dk.height
    cnt = {c: int(dk["splash_colors"].str.contains(c).sum()) for c in R.COLORS if c not in key}
    use = [c for c in cnt if cnt[c] >= MIN_SPLASH_DECKS]
    if not use:
        return []
    cols = np.column_stack([gp["splash_colors"].str.contains(c).cast(pl.Float64).to_numpy() for c in use])
    extra = sp.csr_matrix(cols)

    def contrasts(k: int, ncol: int) -> np.ndarray:
        L = np.zeros((len(use), ncol))
        for i in range(len(use)):
            L[i, k + i] = 1.0
        return L

    coef, enc, _prob, c_rob, _eta = R._fit(gp, extra, np.full(len(use), 10.0), contrasts)
    k = enc.k
    return [dict(color=c, label=R.COLOR_NAMES[c], n_decks=cnt[c], share=cnt[c] / n_dk, **_eff(R.PT * float(coef[k + i]), R.PT * float(c_rob[i])))
            for i, c in enumerate(use)]


def pair_text(d: dict, typical_turns: float | None, typical_mull: float | None) -> list[str]:
    out = []
    op = d.get("on_play")
    if op:
        if op["badge"] == "weak":
            out.append(f"Playing first or second makes no clear difference for these decks ({R.signed(op['diff_pp'])} points on the play).")
        else:
            who = "first" if op["diff_pp"] > 0 else "second"
            out.append(f"These decks do better going {who}: {abs(op['diff_pp']):.1f} points of win rate "
                       f"({op['wr_play'] * 100:.1f}% on the play, {op['wr_draw'] * 100:.1f}% on the draw).")
    if d.get("median_turns") is not None:
        out.append(f"A typical game lasts {d['median_turns']:.0f} turns and {d['mulligan_rate'] * 100:.0f}% of games start with a mulligan"
                   + (f" (whole set: {typical_turns:.0f} turns, {typical_mull * 100:.0f}%)." if typical_turns is not None and typical_mull is not None else "."))
    sp_ = d.get("splash") or []
    good = [s for s in sp_ if _clear(s["badge"])]
    for s in good:
        pay = "pays off" if s["effect_pp"] > 0 else "costs"
        out.append(f"Splashing {s['label'].lower()} {pay}: {R.signed(s['effect_pp'])} points of win rate ({s['badge']}).")
    if sp_ and not good:
        out.append("No splash color clearly pays off or costs.")
    return out


def pair_details(g: pl.DataFrame, pairs_sec: dict) -> dict:
    """{rows: [one per color pair with enough decks, best-winning first]}: on_play (as `report.play_effect`, first games only), median_turns,
    mulligan_rate (observed, not skill-adjusted), splash = [one per splash color with >= MIN_SPLASH_DECKS decks]."""
    prow = {r["key"]: r for r in pairs_sec.get("rows", []) if not r["hidden"] and r["key"] in design.PAIRS}
    typ_t = float(g["num_turns"].median()) if g.height else None
    typ_m = float((g["num_mulligans"].fill_null(0) > 0).mean()) if g.height else None
    gg = g.with_columns(pl.col("main_colors").map_elements(R.pair_group, return_dtype=pl.String).alias("grp"))
    rows = []
    for key, pr in sorted(prow.items(), key=lambda kv: -kv[1]["effect_pp"]):
        gp = gg.filter(pl.col("grp") == key)
        if gp.height < 300:
            continue
        d = dict(key=key, label=pr["label"], colors=key, n_decks=pr["n_decks"], n_games=gp.height,
                 median_turns=float(gp["num_turns"].median()), mulligan_rate=float((gp["num_mulligans"].fill_null(0) > 0).mean()))
        try:
            d["on_play"] = R.play_effect(gp.filter(pl.col("game_number").fill_null(1) <= 1))
        except Exception:                                                                       # one failed fit must not lose the block
            log.info("play effect of %s failed", key, exc_info=True)
            d["on_play"] = None
        try:
            d["splash"] = _splash_by_colour(gp, key)
        except Exception:
            log.info("splash of %s failed", key, exc_info=True)
            d["splash"] = []
        d["text"] = pair_text(d, typ_t, typ_m)
        rows.append(d)
    return dict(rows=rows, min_splash_decks=MIN_SPLASH_DECKS, typical_turns=typ_t, typical_mulligan=typ_m,
                note=NOTE_DETAIL + "  Play / draw uses first games only and is adjusted for skill; game length and mulligan rate are plain figures.")


# ---------------------------------------------------------------------------------------------
# deck shape: win rate by land count and creature count
# ---------------------------------------------------------------------------------------------

def shape_details(g: pl.DataFrame, decks: pl.DataFrame, sh: dict[str, np.ndarray]) -> dict:
    """Skill-adjusted win rate (against the average deck of the shown bands) by land count (16 or fewer / 17 / 18 or more) and by
    creature-count band (thirds of the decks, cut at whole creatures).  A band needs MIN_BAND_DECKS decks."""
    from sealed_lab import report_arch as A

    lands = np.rint(sh["lands"]).astype(int)
    cre = np.rint(sh["creatures"]).astype(int)
    c1, c2 = (int(round(x)) for x in np.percentile(cre, [100 / 3, 200 / 3])) if len(cre) else (0, 0)
    dims = {
        "lands": (np.where(lands <= 16, "L16", np.where(lands == 17, "L17", "L18")),
                  {"L16": "16 or fewer lands", "L17": "17 lands", "L18": "18 or more lands"}),
        "creatures": (np.where(cre <= c1, "C1", np.where(cre <= c2, "C2", "C3")),
                      {"C1": f"{c1} or fewer creatures", "C2": f"{c1 + 1}-{c2} creatures" if c2 > c1 + 1 else f"{c2} creatures",
                       "C3": f"{c2 + 1} or more creatures"}),
    }
    out: dict = dict(min_band_decks=MIN_BAND_DECKS, note=SHAPE_DETAIL_NOTE, text=[])
    n_decks = len(lands)
    for dim, (akey, labels) in dims.items():
        keys = [k for k in labels if int((akey == k).sum()) >= MIN_BAND_DECKS]
        rows = [dict(key=k, label=labels[k], n_decks=int((akey == k).sum()), share=float((akey == k).sum() / max(n_decks, 1))) for k in labels]
        eff: dict = {}
        if len(keys) >= 2:
            gk = g.join(decks.select("draft_id", "deck_key", pl.Series("akey", akey)), on=["draft_id", "deck_key"], how="inner")
            gk = gk.filter(pl.col("akey").is_in(keys))
            try:
                eff = A.fit_effects(gk, gk["akey"].to_numpy(), {k: k for k in keys})
            except Exception:
                log.info("shape detail fit failed", exc_info=True)
        for r in rows:
            e = eff.get(r["key"])
            r.update(_eff(e["eff"], e["se"]) if e else dict(effect_pp=None, se_pp=None, z=None, badge="weak", range=""))
            r["hidden"] = not e
        out[dim] = rows
        shown = [r for r in rows if not r["hidden"]]
        clear = [r for r in shown if _clear(r["badge"])]
        what = "land count" if dim == "lands" else "number of creatures"
        if len(shown) < 2:
            out["text"].append(f"Too few decks in each band to compare by {what}.")
        elif not clear:
            out["text"].append(f"Win rate does not clearly depend on the {what} once pilot skill is taken into account.")
        else:
            for r in sorted(clear, key=lambda r: -r["effect_pp"]):
                out["text"].append(f"Decks with {r['label']} ({r['share'] * 100:.0f}% of decks): {R.signed(r['effect_pp'])} points of win rate "
                                   f"against an average deck ({r['badge']}).")
    return out


# ---------------------------------------------------------------------------------------------
# format evolution
# ---------------------------------------------------------------------------------------------

def _combine(es: list[float], ses: list[float]) -> tuple[float, float]:
    """Precision-weighted mean of independent estimates and its standard error."""
    w = np.array([1.0 / s ** 2 for s in ses])
    return float(np.dot(w, es) / w.sum()), float(1.0 / np.sqrt(w.sum()))


def _verdict(change: float, se: float, halves: list[float]) -> dict:
    z = change / se if se > 0 else None
    badge = R.badge_of(z)
    same = all(h * change > 0 for h in halves)
    verdict = ("better" if change > 0 else "worse") if _clear(badge) and same else "stable"
    return dict(change=change, change_se=se, change_z=z, change_badge=badge, halves=halves, replicated=same, verdict=verdict)


def card_play_by_draft(set_code: str, g: pl.DataFrame, cards: pl.DataFrame) -> pl.DataFrame:
    """Per (card, event): decks that could play it and decks that played it (>= 1 copy); same rules as `report.card_play_stats`."""
    cd = config.corpus_dir(set_code)
    bits = {c: 1 << i for i, c in enumerate(R.COLORS)}

    def mask(s: str | None) -> int:
        return sum(bits[c] for c in R.color_key(s))

    builds = (g.group_by("draft_id", "deck_key").agg(pl.col("main_colors").first())
              .with_columns(pl.col("main_colors").map_elements(mask, return_dtype=pl.Int64).alias("mmask")))
    cinfo = cards.filter(~pl.col("is_basic")).select("card_id", pl.col("colors").map_elements(mask, return_dtype=pl.Int64).alias("cmask"))
    pools = (pl.read_parquet(cd / "pools.parquet", columns=["draft_id", "card_id"])
             .join(builds.select("draft_id", "deck_key", "mmask"), on="draft_id")
             .join(cinfo, on="card_id")
             .filter((pl.col("cmask") & pl.col("mmask")) == pl.col("cmask")))
    dc = pl.read_parquet(cd / "deck_cards.parquet", columns=["deck_key", "card_id", "n"]).join(builds.select("deck_key").unique(), on="deck_key")
    return (pools.join(dc, on=["deck_key", "card_id"], how="left")
            .group_by("card_id", "draft_id").agg(pl.len().alias("n_elig"), (pl.col("n").fill_null(0) > 0).sum().alias("n_played")))


def _cards_evolution(set_code: str, g: pl.DataFrame, cards: pl.DataFrame, cellmap: pl.DataFrame, card_rows: list[dict]) -> dict:
    per = card_play_by_draft(set_code, g, cards).join(cellmap, on="draft_id")
    cell = per.group_by("card_id", "third", "half").agg(pl.col("n_elig").sum().alias("E"), pl.col("n_played").sum().alias("P"))
    cell = cell.with_columns((pl.col("P") / pl.col("E")).alias("p"))
    res = (per.join(cell.select("card_id", "third", "half", "p"), on=["card_id", "third", "half"])
           .with_columns(((pl.col("n_played") - pl.col("p") * pl.col("n_elig")) ** 2).alias("r2"))
           .group_by("card_id", "third", "half").agg(pl.col("r2").sum().alias("S")))
    cell = cell.join(res, on=["card_id", "third", "half"])
    E = {}
    for cid, t, h, e, p_, s in cell.select("card_id", "third", "half", "E", "P", "S").iter_rows():
        E.setdefault(cid, np.zeros((3, 2, 3)))[t, h] = (e, p_, s)
    info = {int(r["card_id"]): r for r in cards.iter_rows(named=True)}
    vals = {r["card_id"]: r for r in card_rows}
    up, down = [], []
    for cid, A in E.items():
        e3, p3, s3 = A[:, :, 0].sum(1), A[:, :, 1].sum(1), A[:, :, 2].sum(1)
        if e3[0] < EVOL_MIN_ELIG or e3[2] < EVOL_MIN_ELIG or (A[0, :, 0] == 0).any() or (A[2, :, 0] == 0).any():
            continue
        pr = p3 / np.maximum(e3, 1)
        var = [s3[t] / max(e3[t], 1) ** 2 + 0.25 / max(e3[t], 1) ** 2 for t in range(3)]          # (a small floor: a play rate of 0 or 1 has no spread)
        change, se = float(pr[2] - pr[0]), float(np.sqrt(var[0] + var[2]))
        halves = [float(A[2, h, 1] / A[2, h, 0] - A[0, h, 1] / A[0, h, 0]) for h in (0, 1)]
        v = _verdict(change, se, halves)
        if v["verdict"] == "stable" or abs(change) < EVOL_MIN_CHANGE:
            continue
        c = info[cid]
        val = vals.get(cid)
        learned = None
        if val and _clear(val["badge"]):
            learned = bool((change > 0) == (val["value_pp"] > 0))
        row = dict(card_id=cid, name=c["name"], group=R.card_group(c["colors"], bool(c["is_land"])), rarity=c["rarity"],
                   pr_early=round(float(pr[0]), 3), pr_middle=round(float(pr[1]), 3), pr_late=round(float(pr[2]), 3),
                   change_pp=round(100 * change, 1), se_pp=round(100 * se, 1), z=round(float(v["change_z"]), 1), badge=v["change_badge"],
                   halves_pp=[round(100 * h, 1) for h in halves], n_early=int(e3[0]), n_late=int(e3[2]),
                   value_pp=None if not val else round(val["value_pp"], 2), value_badge=None if not val else val["badge"], learned=learned,
                   range=f"likely between {R.signed(100 * (change - 1.96 * se), 0)} and {R.signed(100 * (change + 1.96 * se), 0)} points of decks")
        (up if change > 0 else down).append(row)
    up.sort(key=lambda r: -r["change_pp"])
    down.sort(key=lambda r: r["change_pp"])
    allc = up + down
    learned_n = sum(1 for r in allc if r["learned"] is True)
    wrong_n = sum(1 for r in allc if r["learned"] is False)
    return dict(up=up[:EVOL_CARDS_N], down=down[:EVOL_CARDS_N], n_up=len(up), n_down=len(down), n_learned=learned_n, n_against=wrong_n,
                min_elig=EVOL_MIN_ELIG, min_change=EVOL_MIN_CHANGE)


def evolution_text(ev: dict) -> list[str]:
    out = []
    th = ev["thirds"]
    out.append(f"The {ev['n_drafts']:,} events run from {th[0]['from']} to {th[2]['to']}; they are cut into three thirds of about "
               f"{ev['n_drafts'] // 3:,} events each (early: to {th[0]['to']}; late: from {th[2]['from']}).")
    pairs = ev["pairs"]
    moved = [p for p in pairs if p["verdict"] != "stable"]
    for p in sorted(moved, key=lambda p: -abs(p["change"])):
        out.append(f"{p['label']} decks got {p['verdict']} over the format: {R.signed(p['effect'][0])} points early, {R.signed(p['effect'][2])} late "
                   f"({R.signed(p['change'])} points, {p['change_badge']}; same direction in both halves of the data).")
    if not moved:
        out.append("No color pair clearly got better or worse over the format: the differences between the thirds are within what chance can produce, or do not repeat in both halves of the data.")
    cs = ev.get("cards") or {}
    if cs:
        n = cs.get("n_up", 0) + cs.get("n_down", 0)
        if n:
            out.append(f"{n} cards were clearly played more or less often by the end of the format ({cs['n_up']} more, {cs['n_down']} less; "
                       f"at least {int(cs['min_change'] * 100)} points of the decks that could play them, same direction in both halves of the data).")
            tot = cs["n_learned"] + cs["n_against"]
            if tot:
                out.append(f"Of the {tot} such cards that players clearly misjudged at the start, {cs['n_learned']} moved toward their true value and {cs['n_against']} away from it.")
        else:
            out.append("No card's play rate clearly changed over the format.")
    return out


def evolution_section(set_code: str, fmt: str, g: pl.DataFrame, cards: pl.DataFrame, card_rows: list[dict] | None = None) -> dict:
    """The format split into thirds of its events by date; see the module docstring for the rule."""
    starts = g.group_by("draft_id").agg(pl.col("game_time").min().alias("t"), pl.col("game_time").max().alias("t_end")).sort(["t", "draft_id"])
    n = starts.height
    if n < EVOL_MIN_DRAFTS:
        return dict(too_few=True, n_drafts=n, min_drafts=EVOL_MIN_DRAFTS)
    third = np.arange(n) * 3 // n
    half = np.random.default_rng(EVOL_SEED).permutation(n) % 2
    cellmap = pl.DataFrame(dict(draft_id=starts["draft_id"], third=third.astype(np.int64), half=half.astype(np.int64)))
    g2 = g.join(cellmap, on="draft_id")
    thirds = []
    for t in range(3):
        st = starts.filter(pl.Series(third == t))
        thirds.append(dict(name=THIRDS[t], n_drafts=st.height, n_games=int(g2.filter(pl.col("third") == t).height), **{"from": _date(st["t"].min()), "to": _date(st["t"].max())}))
    # --- color pairs: six cells (third x half), the estimate of a third = the precision-weighted mean of its two halves
    cellrows: dict[tuple[int, int], dict] = {}
    for t in range(3):
        for h in (0, 1):
            sec = R.pairs_section(g2.filter((pl.col("third") == t) & (pl.col("half") == h)))
            cellrows[(t, h)] = {r["key"]: r for r in sec.get("rows", []) if not r["hidden"] and r["se_pp"] > 0}
    pairs = []
    for key in design.PAIRS:
        cells = [cellrows[(t, h)].get(key) for t in range(3) for h in (0, 1)]
        if any(c is None for c in cells):
            continue
        eff, se, nd = [], [], []
        for t in range(3):
            e, s = _combine([cellrows[(t, h)][key]["effect_pp"] for h in (0, 1)], [cellrows[(t, h)][key]["se_pp"] for h in (0, 1)])
            eff.append(e)
            se.append(s)
            nd.append(sum(cellrows[(t, h)][key]["n_decks"] for h in (0, 1)))
        halves = [cellrows[(2, h)][key]["effect_pp"] - cellrows[(0, h)][key]["effect_pp"] for h in (0, 1)]
        v = _verdict(eff[2] - eff[0], float(np.hypot(se[0], se[2])), halves)
        v["halves"] = [round(h, 2) for h in halves]
        pairs.append(dict(key=key, label=R.group_label(key), colors=key, effect=eff, se=se, n_decks=nd, **v, range=R.range_text(v["change"], v["change_se"])))
    pairs.sort(key=lambda p: (p["verdict"] == "stable", -abs(p["change"])))
    ev: dict = dict(n_drafts=n, thirds=thirds, pairs=pairs, min_elig=EVOL_MIN_ELIG, note=NOTE_DETAIL)
    try:
        ev["cards"] = _cards_evolution(set_code, g, cards, cellmap, card_rows or [])
    except Exception:
        log.info("card evolution failed", exc_info=True)
        ev["cards"] = {}
    ev["text"] = evolution_text(ev)
    return ev


# ---------------------------------------------------------------------------------------------
# Bo1 vs Bo3, data notes
# ---------------------------------------------------------------------------------------------

def compare_pairs(fmt: str, pairs_this: dict, pairs_other: dict) -> dict:
    """Color-pair effects of both formats side by side; the difference Bo3 - Bo1 gets a z with the two standard errors added
    (different games, so independent).  Sorted by the size of the difference."""
    a = {r["key"]: r for r in pairs_this.get("rows", []) if not r["hidden"] and r["key"] in design.PAIRS}
    b = {r["key"]: r for r in pairs_other.get("rows", []) if not r["hidden"] and r["key"] in design.PAIRS}
    rows = []
    for key in design.PAIRS:
        if key not in a or key not in b:
            continue
        r1, r3 = (a[key], b[key]) if fmt == "bo1" else (b[key], a[key])
        d, se = r3["effect_pp"] - r1["effect_pp"], float(np.hypot(r1["se_pp"], r3["se_pp"]))
        rows.append(dict(key=key, label=R.group_label(key), colors=key, bo1=r1["effect_pp"], bo1_se=r1["se_pp"], bo1_badge=r1["badge"],
                         bo3=r3["effect_pp"], bo3_se=r3["se_pp"], bo3_badge=r3["badge"], **{"diff_" + k: v for k, v in _eff(d, se).items()}))
    rows.sort(key=lambda r: -abs(r["diff_z"] or 0.0))
    clear = [r for r in rows if _clear(r["diff_badge"])]
    text = []
    for r in clear[:3]:
        better = "Best of 3" if r["diff_effect_pp"] > 0 else "Best of 1"
        text.append(f"{r['label']} decks do better in {better} than in the other format: {R.signed(r['bo1'])} points in Best of 1, "
                    f"{R.signed(r['bo3'])} in Best of 3 (difference {R.signed(r['diff_effect_pp'])}, {r['diff_badge']}).")
    if not clear:
        text.append("No color pair clearly does better in one format than in the other.")
    return dict(pairs=rows, text=text)


def data_notes(g: pl.DataFrame, gl: dict, fmt: str) -> dict:
    return dict(n_games=g.height, n_decks=gl.get("n_decks"), n_drafts=gl.get("n_drafts"),
                date_from=_date(g["game_time"].min()) if g.height else None, date_to=_date(g["game_time"].max()) if g.height else None,
                min_eligible=R.MIN_ELIGIBLE, min_exposure=int(R.MIN_EXPOSURE), thin_exposure=int(R.THIN_EXPOSURE), min_pair_decks=R.MIN_PAIR_DECKS,
                min_splash_decks=MIN_SPLASH_DECKS, min_band_decks=MIN_BAND_DECKS)
