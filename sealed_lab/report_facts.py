"""The facts file for the written report: a compact, deterministic digest of the Bo1 and Bo3 statistics of one set.

    facts_from_reports(set_code, {"bo1": report, "bo3": report}, builder=None) -> dict   (JSON-serialisable; no clock, no randomness)
    build_facts(set_code)                  -> dict   (reads the reports through `api.set_report`, i.e. the cache)
    render_text(facts)                     -> str    (what the writer reads)
    load_builder_composition(set_code)     -> dict | None   (the stored results of scripts/builder_composition.py, if any)

A claim is a fact with an id and a Likely / Strong badge (descriptive facts, such as deck-shape ranges, carry badge "descriptive").  Items
that showed "no clear difference" (Weak) are listed apart under NOCLEAR: the writer may only say there is no clear difference for them.

Ids: PLAY.bo1 / GLANCE.bo1 / SPLASH.bo1 / PAIR.bo1.BG / SPEC.bo1.WG / CARD.bo1.<slug> / SHAPE.bo1.all|<pair> / BUILDER.bo1.<metric>;
details: PAIRDET.bo1.BG.PLAY / PAIRDET.bo1.BG.SPLASH.R / SHAPEDET.bo1.lands.L17 / SHAPEDET.bo1.creatures.C3 / EVOL.bo1.PAIR.BG / EVOL.bo1.CARD.<slug>.
Each fact has `text` (terse, what the writer relies on), `nums` (every number the text may be quoted with, signed, percentages as 0-100) and
`builder_uses` ("yes" = the deck builder uses it, "observed" = shown only).  The checker (scripts/check_report.py) works from the JSON.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from sealed_lab import config, report as R

FACTS_VERSION = 2
FORMATS = R.FORMATS
MAX_UNDER, MAX_OVER, MAX_TRAPS, TOP_COMMON, TOP_UNCOMMON = 3, 3, 3, 1, 1
MAX_PAIR_PLAY, MAX_SPLASH, MAX_EVOL_PAIRS, MAX_EVOL_CARDS = 2, 2, 3, 2     # details listed per format (the largest; keeps the file short)
LIKELY_UP = ("likely", "strong")
BADGE_WORD = {"strong": "Strong", "likely": "Likely"}
FMT_WORD = {"bo1": "Best of 1", "bo3": "Best of 3"}
BUILDER_METRICS = {"creatures": "creatures", "two_drops": "two-drops", "lands": "lands", "mean_mv": "average mana value"}
# where `scripts/builder_composition.py SET OUTDIR` output may be kept (first hit wins)
BUILDER_DIRS = (Path(__file__).resolve().parents[1] / "docs" / "builder_composition",)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "card"


def _r(x: float, d: int = 3) -> float:
    return round(float(x), d)


def _fact(fid: str, kind: str, text: str, nums: list[float], badge: str, uses: str, **extra) -> dict:
    return dict(id=fid, kind=kind, text=text, nums=sorted({_r(n) for n in nums}), badge=badge, builder_uses=uses, **extra)


def _eff_nums(eff: float, se: float) -> list[float]:
    return [eff, eff - 1.96 * se, eff + 1.96 * se]


def _rng(eff: float, se: float) -> str:
    return f"{R.signed(eff - 1.96 * se)} to {R.signed(eff + 1.96 * se)}"


# ---------------------------------------------------------------------------------------------
# one format
# ---------------------------------------------------------------------------------------------

def _facts_of_format(fmt: str, rep: dict, facts: list[dict], noclear: list[dict]) -> None:
    gl, pr, cs = rep.get("glance") or {}, rep.get("pairs") or {}, rep.get("cards") or {}
    if rep.get("too_few"):
        return
    if gl.get("n_decks"):
        facts.append(_fact(f"GLANCE.{fmt}", "glance",
                           f"{gl['n_decks']} decks, {gl['n_games']} games, typical game {gl['median_turns']:.0f} turns, "
                           f"{gl['mulligan_rate'] * 100:.0f}% of games with a mulligan",
                           [gl["n_decks"], gl["n_games"], gl["median_turns"], gl["mulligan_rate"] * 100], "descriptive", "observed", fmt=fmt,
                           mulligan_pct=_r(gl["mulligan_rate"] * 100, 2), median_turns=_r(gl["median_turns"], 2)))
    op = gl.get("on_play")
    if op:
        if op["badge"] in LIKELY_UP:
            facts.append(_fact(f"PLAY.{fmt}", "play",
                               f"playing first {R.signed(op['diff_pp'])} points against playing second ({op['wr_play'] * 100:.1f}% against "
                               f"{op['wr_draw'] * 100:.1f}% win rate; range {_rng(op['diff_pp'], op['se_pp'])}), {BADGE_WORD[op['badge']]}",
                               _eff_nums(op["diff_pp"], op["se_pp"]) + [op["wr_play"] * 100, op["wr_draw"] * 100], op["badge"], "observed", fmt=fmt,
                               value=_r(op["diff_pp"], 2)))
        else:
            noclear.append(dict(id=f"PLAY.{fmt}", fmt=fmt, text="playing first or second"))
    for r in sorted((r for r in pr.get("rows", []) if not r["hidden"]), key=lambda r: (-r["effect_pp"], r["key"])):
        fid = f"PAIR.{fmt}.{r['key']}"
        if r["badge"] in LIKELY_UP:
            facts.append(_fact(fid, "pair",
                               f"{r['label']} ({r['share'] * 100:.1f}% of decks) {R.signed(r['effect_pp'])} points against an average deck "
                               f"(range {_rng(r['effect_pp'], r['se_pp'])}), {BADGE_WORD[r['badge']]}",
                               _eff_nums(r["effect_pp"], r["se_pp"]) + [r["share"] * 100, r["splash_share"] * 100], r["badge"], "observed", fmt=fmt,
                               value=_r(r["effect_pp"], 2)))
        else:
            noclear.append(dict(id=fid, fmt=fmt, text=f"{r['label']} as a color group"))
    sp_ = pr.get("splash")
    if sp_:
        if sp_["badge"] in LIKELY_UP:
            word = "pays off" if sp_["effect_pp"] > 0 else "costs"
            facts.append(_fact(f"SPLASH.{fmt}", "splash",
                               f"{sp_['share'] * 100:.0f}% of decks splash; splashing {word} {R.signed(sp_['effect_pp'])} points "
                               f"(range {_rng(sp_['effect_pp'], sp_['se_pp'])}), {BADGE_WORD[sp_['badge']]}",
                               _eff_nums(sp_["effect_pp"], sp_["se_pp"]) + [sp_["share"] * 100], sp_["badge"], "observed", fmt=fmt,
                               value=_r(sp_["effect_pp"], 2)))
        else:
            noclear.append(dict(id=f"SPLASH.{fmt}", fmt=fmt, text="splashing a third color"))
    ar = (rep.get("archetypes") or {}).get("pairs", [])
    for a in ar:
        sp = a.get("spectrum") or {}
        if not sp.get("shown"):
            continue
        fid = f"SPEC.{fmt}.{a['key']}"
        win, ea, eb = sp["win"], sp["a"], sp["b"]
        if win.get("finding") and win["badge"] in LIKELY_UP:
            better, worse = (eb, ea) if win["effect_pp"] > 0 else (ea, eb)
            sg = 1.0 if win["effect_pp"] > 0 else -1.0                                     # the range seen from the winning end
            lo, hi = sorted((sg * (win["effect_pp"] - 1.96 * win["se_pp"]), sg * (win["effect_pp"] + 1.96 * win["se_pp"])))
            facts.append(_fact(fid, "spectrum",
                               f"{a['code']}: the {better['label']} end beats the {worse['label']} end by {abs(win['effect_pp']):.1f} points "
                               f"(range {R.signed(lo)} to {R.signed(hi)}), {BADGE_WORD[win['badge']]}, same direction in both halves",
                               [abs(win["effect_pp"]), lo, hi], win["badge"], "observed", fmt=fmt,
                               value=_r(abs(win["effect_pp"]), 2),
                               ends=dict(a=ea["label"], b=eb["label"], winner=better["label"],
                                         winner_avg_mv=_r(better["avg_mv"]), other_avg_mv=_r(worse["avg_mv"]))))
        else:
            noclear.append(dict(id=fid, fmt=fmt, text=f"{a['key']} decks: the {ea['label']} end against the {eb['label']} end"))
    # cards: traps, overrated, underrated, then the best commons / uncommons
    seen: set[int] = set()
    used: dict[str, int] = {}

    def card(r: dict, kind: str) -> None:
        if r["card_id"] in seen:
            return
        seen.add(r["card_id"])
        base = slug(r["name"])
        used[base] = used.get(base, 0) + 1
        fid = f"CARD.{fmt}.{base}" + ("" if used[base] == 1 else f"_{used[base]}")
        pr_ = r["play_rate"] * 100
        what = {"trap": f"trap {r['rarity']}", "over": f"overrated {r['rarity']}", "under": f"underrated {r['rarity']}", "top": f"top {r['rarity']}"}[kind]
        grp = r["group"] if r["group"] in R.COLORS else {"gold": "multicolor", "colorless": "colorless", "land": "land"}.get(r["group"], r["group"])
        facts.append(_fact(fid, f"card_{kind}",
                           f"{r['name']} ({grp}) {what}, in {pr_:.0f}% of eligible decks, {R.signed(r['value_pp'])} points per copy "
                           f"(range {_rng(r['value_pp'], r['se_pp'])}), {BADGE_WORD[r['badge']]}" + (", limited data" if r.get("thin") else ""),
                           _eff_nums(r["value_pp"], r["se_pp"]) + [pr_], r["badge"], "yes", fmt=fmt, name=r["name"], limited_data=bool(r.get("thin")), value=_r(r["value_pp"], 2)))

    for r in (cs.get("traps") or [])[:MAX_TRAPS]:
        card(r, "trap")
    for kind, key, cap, sign in (("over", "overrated", MAX_OVER, 1), ("under", "underrated", MAX_UNDER, -1)):
        n_kind = 0
        for r in sorted((r for r in cs.get(key, []) if r["badge"] in LIKELY_UP), key=lambda r: (sign * r["value_pp"], r["name"])):
            if n_kind >= cap:
                break
            if r["card_id"] not in seen:
                card(r, kind)
                n_kind += 1
    for rar, n in (("common", TOP_COMMON), ("uncommon", TOP_UNCOMMON)):
        rows = [r for grp in (cs.get("top") or {}).get(rar, {}).values() for r in grp if r["badge"] in LIKELY_UP and r["value_pp"] > 0]
        for r in sorted(rows, key=lambda r: (-r["value_pp"], r["name"]))[:n]:
            card(r, "top")
    _detail_facts(fmt, rep, facts, noclear)
    # deck shape: all decks (the per-pair rows stay in the Analysis tab)
    sh = rep.get("shape") or {}
    rows = {r["key"]: r for r in sh.get("rows", [])}
    pick = ["all"]
    for key in pick:
        row = rows.get(key)
        if not row:
            continue
        nums: list[float] = []
        parts, parts_s = [], []
        for m in sh.get("metrics", []):
            d = row["metrics"][m["key"]]
            dg = 1 if m["key"] == "avg_mv" else 0
            lab = m["label"].lower().replace("average mana value", "avg mana value")
            for dd, lst in ((d.get("all"), parts), (d.get("strong"), parts_s)):
                if dd:
                    lo, hi = round(dd["lo"], dg), round(dd["hi"], dg)
                    nums += [lo, hi, dd["med"]]
                    lst.append(f"{lab} {lo:.{dg}f}" + ("" if lo == hi else f"-{hi:.{dg}f}"))
        tail = (f"; decks with strong results ({sh.get('strong_def')}): " + ", ".join(parts_s)) if parts_s and parts_s != parts else \
               "; the same among decks with strong results" if parts_s else ""
        facts.append(_fact(f"SHAPE.{fmt}.{key}", "shape", f"{row['label']} ({row['n_decks']} decks), middle half: " + ", ".join(parts) + tail,
                           nums + [row["n_decks"], row["n_strong"]], "descriptive", "observed", fmt=fmt))


def _detail_facts(fmt: str, rep: dict, facts: list[dict], noclear: list[dict]) -> None:
    """PAIRDET.*, SHAPEDET.*, EVOL.*: the "More detail" results of the Analysis tab, with the same Likely+ rule."""
    rows_pd = (rep.get("pair_details") or {}).get("rows", [])
    big_play = {d["key"] for d in sorted((d for d in rows_pd if d.get("on_play") and d["on_play"]["badge"] in LIKELY_UP),
                                         key=lambda d: (-abs(d["on_play"]["diff_pp"]), d["key"]))[:MAX_PAIR_PLAY]}      # (keeps the file short)
    big_spl = {(d["key"], s_["color"]) for d, s_ in sorted(((d, s_) for d in rows_pd for s_ in d.get("splash") or [] if s_["badge"] in LIKELY_UP),
                                                         key=lambda t: (-abs(t[1]["effect_pp"]), t[0]["key"]))[:MAX_SPLASH]}
    for d in rows_pd:
        key, lab = d["key"], d["label"]
        op = d.get("on_play")
        if op:
            if op["badge"] in LIKELY_UP and key not in big_play:
                continue
            if op["badge"] in LIKELY_UP:
                facts.append(_fact(f"PAIRDET.{fmt}.{key}.PLAY", "pairdet_play",
                                   f"{lab} decks: playing first {R.signed(op['diff_pp'])} points against playing second ({op['wr_play'] * 100:.1f}% against "
                                   f"{op['wr_draw'] * 100:.1f}% win rate; range {_rng(op['diff_pp'], op['se_pp'])}), {BADGE_WORD[op['badge']]}",
                                   _eff_nums(op["diff_pp"], op["se_pp"]) + [op["wr_play"] * 100, op["wr_draw"] * 100], op["badge"], "observed", fmt=fmt))
        for s_ in d.get("splash") or []:
            if s_["badge"] in LIKELY_UP and (key, s_["color"]) in big_spl:
                word = "pays off" if s_["effect_pp"] > 0 else "costs"
                facts.append(_fact(f"PAIRDET.{fmt}.{key}.SPLASH.{s_['color']}", "pairdet_splash",
                                   f"{lab} decks: splashing {s_['label'].lower()} {word} {R.signed(s_['effect_pp'])} points ({s_['n_decks']} decks; "
                                   f"range {_rng(s_['effect_pp'], s_['se_pp'])}), {BADGE_WORD[s_['badge']]}",
                                   _eff_nums(s_["effect_pp"], s_["se_pp"]) + [s_["n_decks"]], s_["badge"], "observed", fmt=fmt))
    sd = (rep.get("shape") or {}).get("details") or {}
    for dim, word in (("lands", "land count"), ("creatures", "creature count")):
        rows = [r for r in sd.get(dim, []) if not r.get("hidden")]
        clear = [r for r in rows if r["badge"] in LIKELY_UP]
        for r in clear:
            facts.append(_fact(f"SHAPEDET.{fmt}.{dim}.{r['key']}", "shapedet",
                               f"decks with {r['label']} ({r['share'] * 100:.0f}% of decks) {R.signed(r['effect_pp'])} points against an average deck "
                               f"(range {_rng(r['effect_pp'], r['se_pp'])}), {BADGE_WORD[r['badge']]}",
                               _eff_nums(r["effect_pp"], r["se_pp"]) + [r["share"] * 100], r["badge"], "observed", fmt=fmt,
                               value=_r(r["effect_pp"], 2)))
        if len(rows) >= 2 and not clear:
            noclear.append(dict(id=f"SHAPEDET.{fmt}.{dim}", fmt=fmt, text=f"win rate by {word}"))
    ev = rep.get("evolution") or {}
    if ev.get("too_few") or not ev.get("pairs"):
        return
    th = ev["thirds"]
    movers = {p["key"] for p in sorted((p for p in ev["pairs"] if p["verdict"] != "stable"), key=lambda p: (-abs(p["change"]), p["key"]))[:MAX_EVOL_PAIRS]}
    for p in ev["pairs"]:
        fid = f"EVOL.{fmt}.PAIR.{p['key']}"
        if p["verdict"] != "stable" and p["key"] not in movers:
            continue
        if p["verdict"] != "stable":
            facts.append(_fact(fid, "evol_pair",
                               f"{p['label']} decks got {p['verdict']} over the format: {R.signed(p['effect'][0])} points early ({th[0]['from']} to {th[0]['to']}), "
                               f"{R.signed(p['effect'][2])} late ({th[2]['from']} to {th[2]['to']}); change {R.signed(p['change'])} points "
                               f"(range {_rng(p['change'], p['change_se'])}), {BADGE_WORD[p['change_badge']]}, same direction in both halves of the data",
                               _eff_nums(p["change"], p["change_se"]) + [p["effect"][0], p["effect"][2]], p["change_badge"], "observed", fmt=fmt))
        else:
            noclear.append(dict(id=fid, fmt=fmt, text=f"{p['label']} decks over the format"))
    cs = ev.get("cards") or {}
    used: dict[str, int] = {}
    for kind, rows in (("up", cs.get("up") or []), ("down", cs.get("down") or [])):
        for r in rows[:MAX_EVOL_CARDS]:
            base = slug(r["name"])
            used[base] = used.get(base, 0) + 1
            fid = f"EVOL.{fmt}.CARD.{base}" + ("" if used[base] == 1 else f"_{used[base]}")
            tail = "" if r.get("learned") is None else (", toward its measured value" if r["learned"] else ", away from its measured value")
            facts.append(_fact(fid, "evol_card",
                               f"{r['name']}: played in {r['pr_early'] * 100:.0f}% of the decks that could play it early in the format, "
                               f"{r['pr_late'] * 100:.0f}% late ({'more' if kind == 'up' else 'less'} often; change {R.signed(r['change_pp'], 0)} points of decks, "
                               f"{BADGE_WORD[r['badge']]}, same direction in both halves{tail})",
                               [r["pr_early"] * 100, r["pr_late"] * 100, r["change_pp"], r["change_pp"] - 1.96 * r["se_pp"], r["change_pp"] + 1.96 * r["se_pp"]],
                               r["badge"], "observed", fmt=fmt, name=r["name"]))


# ---------------------------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------------------------

def load_builder_composition(set_code: str, path: str | Path | None = None) -> dict | None:
    """The stored output of scripts/builder_composition.py (builder_composition_<SET>.json): in `path`, else in docs/builder_composition/,
    else in the set's cache folder.  None when nothing is stored."""
    name = f"builder_composition_{set_code.upper()}.json"
    cands = [Path(path)] if path else [d / name for d in BUILDER_DIRS] + [config.cache_dir(set_code) / name]
    for p in cands:
        try:
            if p.is_dir():
                p = p / name
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "formats" in d:
            try:                         # scripts/builder_check.py: do real decks shaped like the builder's win more?
                d["check"] = json.loads((p.parent / f"builder_check_{set_code.upper()}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            return d
    return None


MIN_PLAUSIBLE_CREATURES = 5   # players' average below this means missing card types, not a real count


def _builder_facts(builder: dict, facts: list[dict], noclear: list[dict]) -> None:
    for fmt in FORMATS:
        res = (builder.get("formats") or {}).get(fmt)
        if not res:
            continue
        for k, label in BUILDER_METRICS.items():
            m = res.get(k)
            if not m:
                continue
            if k == "creatures" and m["players_mean"] < MIN_PLAUSIBLE_CREATURES:
                continue   # the set's card data lacks types (e.g. OM1): the count is not real
            d, se = m["diff_players"], m["diff_players_se"]
            fid = f"BUILDER.{fmt}.{k}"
            if se > 0 and abs(d) >= 2 * se:
                more = ("higher" if d > 0 else "lower") if k == "mean_mv" else ("more" if d > 0 else "fewer")
                p = 2 if k == "mean_mv" else 1
                facts.append(_fact(fid, "builder",
                                   f"builder plays {m['builder_mean']:.{p}f} {label} on average, players with the same pools {m['players_mean']:.{p}f} "
                                   f"({abs(d):.{p}f} {more}; {res['n_pools']} pools)",
                                   [m["builder_mean"], m["players_mean"], abs(d), res["n_pools"]], "likely", "yes", fmt=fmt,
                                   value=_r(d, 3)))
            else:
                noclear.append(dict(id=fid, fmt=fmt, text=f"builder against players on {label}"))
        _builder_check_fact(((builder.get("check") or {}).get("formats") or {}).get(fmt), fmt, facts, noclear)


def _builder_check_fact(chk: dict | None, fmt: str, facts: list[dict], noclear: list[dict]) -> None:
    """BUILDERCHECK.<fmt>: real player decks moved by the builder's average shape differences (lands, creatures, two-drops, mana
    value), win rate among players of the same skill, color pair held fixed (scripts/builder_check.py).  Observed association only."""
    c = (chk or {}).get("combined") or {}
    if chk is None or chk.get("skipped") or c.get("pp") is None:
        return
    fid = f"BUILDERCHECK.{fmt}"
    badge = str(c.get("badge", "")).lower()
    if badge in ("likely", "strong"):
        pp, se = float(c["pp"]), float(c["se_total"])
        facts.append(_fact(fid, "builder",
                           f"real player decks shaped like the builder's (its average differences in lands, creatures, two-drops and mana value) "
                           f"{R.signed(pp)} points for players of the same skill in the same color pair (range {_rng(pp, se)}; "
                           f"{chk.get('n_games', 0):,} games), {badge.capitalize()}; observed association, not a test of the builder",
                           _eff_nums(pp, se), badge, "observed", fmt=fmt, value=_r(pp, 2)))
    else:
        noclear.append(dict(id=fid, fmt=fmt, text="real player decks shaped like the builder's: no clear difference in results"))


# ---------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------

def facts_from_reports(set_code: str, reports: dict[str, dict], builder: dict | None = None) -> dict:
    """The facts of a set from report dicts (as `report.compute_report` / `api.set_report` return them), keyed by format."""
    facts: list[dict] = []
    noclear: list[dict] = []
    for fmt in FORMATS:
        if fmt in reports:
            _facts_of_format(fmt, reports[fmt], facts, noclear)
    if builder:
        _builder_facts(builder, facts, noclear)
    ids = [f["id"] for f in facts]
    assert len(ids) == len(set(ids)), "duplicate fact ids"
    return dict(version=FACTS_VERSION, report_version=R.REPORT_VERSION, set=set_code.upper(), formats=[f for f in FORMATS if f in reports],
                stamps={f: reports[f].get("stamp") for f in FORMATS if f in reports},
                builder_available=bool(builder),
                facts={f["id"]: f for f in facts}, noclear={n["id"]: n for n in noclear})


def build_facts(set_code: str, refresh: bool = False, progress=None) -> dict:
    from sealed_lab import api

    if set_code.upper() == "ALL":                       # the cross-set lessons (sealed_lab/cross_set.py)
        from sealed_lab import cross_set

        return cross_set.build_cross_facts(refresh=refresh, progress=progress)
    st = api.set_status(set_code)
    reports = {f: api.set_report(set_code, f, refresh=refresh, progress=progress) for f in FORMATS if f in st.formats}
    return facts_from_reports(set_code, reports, load_builder_composition(set_code))


def render_text(facts: dict) -> str:
    """Plain text for the writer: claims by format, then the builder, then the NOCLEAR ids."""
    if facts.get("set") == "ALL":
        from sealed_lab import cross_set

        return cross_set.render_text(facts)
    L = [f"FACTS {facts['set']} (statistics v{facts['report_version']}). Line = [id] fact. Only these ids may back a claim. Win rates: points of game win "
         "rate, adjusted for pilot skill; 'range' = the ~95% interval; B at the end = the deck builder uses it, otherwise observed only."]
    by = list(facts["facts"].values())
    for fmt in facts["formats"]:
        L += ["", f"== {FMT_WORD[fmt]} =="]
        L += [f"[{f['id']}] {f['text']}{' B' if f['builder_uses'] == 'yes' else ''}" for f in by if f["fmt"] == fmt and f["kind"] != "builder"]
    bl = [f for f in by if f["kind"] == "builder"]
    L.append("")
    if bl:
        L.append("== Deck builder against real players on the same pools ==")
        L += [f"[{f['id']}] {f['text']} B" for f in bl]
    else:
        L.append("== Deck builder: no stored comparison with players for this set: make no claim about what the builder does ==")
    L += ["", "NOCLEAR (no clear difference; the only allowed statement: there is no clear difference):"]
    for fmt in facts["formats"]:
        ids = [n["id"] for n in facts["noclear"].values() if n["fmt"] == fmt]
        if ids:
            L.append("  " + " ".join(ids))
    return "\n".join(L) + "\n"
