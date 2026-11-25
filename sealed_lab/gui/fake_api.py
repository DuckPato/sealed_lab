"""Fake backend implementing the ``sealed_lab.api`` facade with plausible data (development and tests).

Enable in the app with ``SEALED_LAB_FAKE_API=1``. ``SEALED_LAB_FAKE_DELAY`` scales sleeps (seconds per step,
default 0.04; ``calculate_set`` takes ~40 steps).
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import tempfile
import zlib
import threading
import time
from pathlib import Path
from typing import Callable

from sealed_lab.types import (
    BuildResult,
    Cancelled,
    CardLine,
    DeckEval,
    FileInfo,
    ScoreResult,
    SetBusy,
    SetStatus,
)

IS_FAKE = True

_COLORS = "WUBRG"
_BASICS = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
_lock = threading.Lock()


def _delay() -> float:
    try:
        return float(os.environ.get("SEALED_LAB_FAKE_DELAY", "0.04"))
    except ValueError:
        return 0.04


# --------------------------------------------------------------------------- fake card universe

def _make_cards() -> dict[int, dict]:
    rng = random.Random(7)
    creature_words = ["Knight", "Drake", "Ghoul", "Goblin", "Elf", "Sphinx", "Golem", "Warden", "Hydra", "Raider",
                      "Cleric", "Wraith"]
    spell_words = ["Bolt", "Ward", "Surge", "Rite", "Gambit", "Verdict", "Snare", "Bloom", "Eruption", "Charm"]
    adj = {"W": "Dawn", "U": "Tide", "B": "Grave", "R": "Ember", "G": "Thorn", "": "Iron"}
    cards: dict[int, dict] = {}
    cid = 0
    for col in list(_COLORS) + [""]:
        for i in range(10 if col else 6):
            is_creature = i < 6 or not col
            word = (creature_words if is_creature else spell_words)[(i + _COLORS.find(col) * 2) % 10]
            name = f"{adj[col]} {word}" + (" of Aldra" if i >= 8 else "")
            cards[cid] = {"name": name, "mana_value": float(rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 6])),
                          "is_creature": is_creature, "is_land": False, "colors": col}
            cid += 1
    # a few special names exercising punctuation/accents
    cards[cid] = {"name": "Fire // Ice", "mana_value": 2.0, "is_creature": False, "is_land": False, "colors": "RU"}
    cid += 1
    cards[cid] = {"name": "Jötun's Rage, the Unbowed", "mana_value": 5.0, "is_creature": True, "is_land": False,
                  "colors": "R"}
    cid += 1
    cards[cid] = {"name": "Scoured Crossroads", "mana_value": 0.0, "is_creature": False, "is_land": True,
                  "colors": ""}
    cid += 1
    for col in _COLORS:
        cards[cid] = {"name": _BASICS[col], "mana_value": 0.0, "is_creature": False, "is_land": True, "colors": col}
        cid += 1
    return cards


_CARDS = _make_cards()
_BY_NAME = {d["name"].casefold(): i for i, d in _CARDS.items()}
_DUAL_ID = _BY_NAME["scoured crossroads"]
_BASIC_ID = {col: _BY_NAME[_BASICS[col].casefold()] for col in _COLORS}


def card_info(set_code: str) -> dict[int, dict]:
    """Grouping info: {card_id: {name, mana_value, is_creature, is_land, colors, rarity, types, cost, is_basic, produces}}."""
    out = {}
    for i, d in _CARDS.items():
        e = dict(d)
        basic = d["is_land"] and d["name"] in _BASICS.values()
        mv = int(d["mana_value"])
        cols = d["colors"]
        e.update(rarity=("rare" if i % 9 == 0 else "uncommon" if i % 4 == 0 else "common") if not d["is_land"] else "common",
                 types=("Basic Land" if basic else "Land") if d["is_land"] else ("Creature - Fake" if d["is_creature"] else "Instant"),
                 cost="" if d["is_land"] else (str(max(mv - len(cols), 0)) if mv > len(cols) or not cols else "") + cols,
                 is_basic=basic, produces=(cols if basic else ("WU" if d["is_land"] else "")))
        out[i] = e
    return out


def _beta(cid: int) -> float:
    return random.Random(f"beta{cid}").gauss(0.02, 0.05)


def card_values(set_code: str, fmt: str = "bo3", deck: dict[int, int] | None = None, cids: list[int] | None = None) -> dict[int, dict]:
    _guard(set_code)
    out = {i: dict(beta=_beta(i), low_data=i % 7 == 0) for i, d in _CARDS.items()
           if d["name"] not in _BASICS.values() and (cids is None or i in cids)}
    if deck:
        for i, v in out.items():
            v.update(value_pp=100.0 * v["beta"] / 4.0, castable=None, contrib=v["beta"])
    return out


def set_report(set_code: str, fmt: str = "bo3", refresh: bool = False, progress=None, cancel=None) -> dict:
    """A plausible synthetic Set report (the real one is `sealed_lab.report.compute_report`; the card lists use its own builder)."""
    import polars as pl

    from sealed_lab import report as R

    _guard(set_code)
    time.sleep(_delay())
    if progress:
        progress(0.5, "Color pairs (adjusted for pilot skill)")
    rng = random.Random(f"report{fmt}")
    n_decks, n_games = (15_300, 61_000) if fmt == "bo1" else (2_900, 7_500)
    keys = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG", "tri", "mono"]
    shares = [rng.uniform(0.04, 0.09) for _ in keys[:10]] + [0.33, 0.01]
    tot = sum(shares)
    rows = []
    for k, sh in zip(keys, shares):
        nd = int(n_decks * sh / tot)
        eff, se = rng.gauss(0.0, 1.2), rng.uniform(0.5, 1.0) * (1.0 if fmt == "bo1" else 2.2) * (7.0 if k == "mono" else 1.0)
        z = eff / se
        rows.append(dict(key=k, label=R.group_label(k), colors=k if len(k) == 2 else "", share=nd / n_decks, n_decks=nd, n_games=nd * 4,
                         splash_share=rng.uniform(0.2, 0.7), effect_pp=eff, se_pp=se, z=z, badge=R.badge_of(z), range=R.range_text(eff, se),
                         hidden=nd < R.MIN_PAIR_DECKS))
    spl = dict(share=0.42, n_decks=int(n_decks * 0.42), effect_pp=-1.0, se_pp=0.4, z=-2.5, badge="likely", range=R.range_text(-1.0, 0.4))
    pairs = dict(n_decks=n_decks, n_games=n_games, min_decks=R.MIN_PAIR_DECKS, rows=rows, splash=spl)
    pairs["text"] = R.pairs_text(pairs)
    rar = {"common": 0.0, "uncommon": 0.0, "rare": 0.0}
    tab = []
    for i, d in _CARDS.items():
        if d["name"] in _BASICS.values():
            continue
        r = random.Random(f"rep{i}")
        info = card_info(set_code)[i]
        pr = min(max(r.gauss(0.6 if info["rarity"] != "common" else 0.5, 0.25), 0.02), 0.99)
        nse = 320 + 5 * i if d["colors"] else 0
        nsp = int(nse * min(0.35, max(0.01, 0.04 + (0.14 if info["rarity"] == "rare" else 0.02)))) if nse else 0
        tab.append(dict(card_id=i, name=d["name"], rarity=info["rarity"], colors=d["colors"], is_land=d["is_land"], is_basic=False,
                        beta=_beta(i) * 1.6 - 0.02 * pr, se=0.012, exposure=300.0 if i % 11 == 0 else 4000.0, n_elig=900 + 10 * i,
                        n_played=int(pr * (900 + 10 * i)), copies=int(pr * (900 + 10 * i) * 1.1),
                        n_splash_elig=nse, n_splashed=nsp,
                        beta_o=_beta(i) * 1.6 - 0.02 * pr + (0.05 if i % 17 == 0 else 0.0), se_o=0.012, exposure_o=300.0 if i % 13 == 0 else 3000.0,
                        copies_o=int(pr * (900 + 10 * i) * 1.1)))
    for row in [t for t in tab if t["rarity"] == "rare"][:2]:                  # a trap rare for the GUI: played almost always, clearly negative
        row.update(beta=-0.09, se=0.012, exposure=4000.0, n_elig=1500, n_played=1450, copies=1500)
    cards = R.build_card_lists(pl.DataFrame(tab), fmt, keep_rows=True)
    card_rows = cards.pop("_rows", [])
    glance = dict(n_games=n_games, n_decks=n_decks, n_drafts=n_decks // 3, n_pools=n_decks // 3, median_turns=10.0, mulligan_rate=0.10,
                  play_n_games=n_games, on_play=dict(wr_play=0.585, wr_draw=0.549, diff_pp=3.6, se_pp=0.5, z=7.2, badge="strong",
                                                     range=R.range_text(3.6, 0.5)))
    glance["text"] = R.glance_text(glance, fmt)
    stamp = f"fake|{fmt}"
    arch, shape = _fake_archetypes(set_code, fmt, rows, card_rows, rng)
    extra = _fake_details(fmt, rows, cards, card_rows, shape, rng, n_games, n_decks)
    return dict(extra, version=R.REPORT_VERSION, set_code=set_code.upper(), fmt=fmt, built_at="2026-10-20 18:42", stamp=stamp, computed_at="", notes=dict(R.BUILDER_USES),
                glance=glance, pairs=pairs, cards=cards, archetypes=arch, shape=shape, seconds=0.1, cached=False)


def _fake_details(fmt: str, pair_rows: list[dict], cards: dict, card_rows: list[dict], shape: dict, rng: random.Random, n_games: int, n_decks: int) -> dict:
    """Synthetic "More detail" sections (pair details, shape details, evolution, Bo1-vs-Bo3, data notes) in the real report's format."""
    from sealed_lab import report as R

    def eff(e: float, se: float) -> dict:
        z = e / se
        return dict(effect_pp=e, se_pp=se, z=z, badge=R.badge_of(z), range=R.range_text(e, se))

    pairs = [r for r in pair_rows if len(r["key"]) == 2 and not r["hidden"]]
    rows = []
    for r in sorted(pairs, key=lambda r: -r["effect_pp"]):
        d3 = rng.uniform(-1, 6)
        op = dict(wr_play=0.55 + d3 / 200, wr_draw=0.55 - d3 / 200, diff_pp=d3, se_pp=1.2, z=d3 / 1.2, badge=R.badge_of(d3 / 1.2), range=R.range_text(d3, 1.2))
        spl = [dict(color=c, label=R.COLOR_NAMES[c], n_decks=80 + 20 * j, share=0.05, **eff(rng.uniform(-3, 3), 1.4))
               for j, c in enumerate(c for c in R.COLORS if c not in r["key"][:2])][:3]
        d = dict(key=r["key"], label=r["label"], colors=r["key"], n_decks=r["n_decks"], n_games=r["n_games"], median_turns=9.0 + rng.random(),
                 mulligan_rate=0.09 + rng.random() * 0.03, on_play=op, splash=spl)
        rows.append(d)
    pair_details = dict(rows=rows, min_splash_decks=60, typical_turns=9.5, typical_mulligan=0.1,
                        note="Observed only: the builder does not use these figures.  Play / draw uses first games only and is adjusted for skill; game length and mulligan rate are plain figures.")
    lands = [dict(key="L16", label="16 or fewer lands", n_decks=900, share=0.1, hidden=False, **eff(0.4, 0.6)),
             dict(key="L17", label="17 lands", n_decks=int(n_decks * 0.85), share=0.85, hidden=False, **eff(-0.05, 0.1)),
             dict(key="L18", label="18 or more lands", n_decks=60, share=0.01, hidden=True, effect_pp=None, se_pp=None, z=None, badge="weak", range="")]
    cre = [dict(key="C1", label="13 or fewer creatures", n_decks=4000, share=0.4, hidden=False, **eff(0.3, 0.3)),
           dict(key="C2", label="14-15 creatures", n_decks=4000, share=0.4, hidden=False, **eff(0.2, 0.3)),
           dict(key="C3", label="16 or more creatures", n_decks=2000, share=0.2, hidden=False, **eff(-1.0, 0.4))]
    shape["details"] = dict(min_band_decks=100, note="For reference: the builder handles curve and land count through the model, these figures are not used by it.",
                            lands=lands, creatures=cre,
                            text=["Win rate does not clearly depend on the land count once pilot skill is taken into account.",
                                  "Decks with 16 or more creatures (20% of decks): -1.0 points of win rate against an average deck (likely)."])
    th = [dict(name="early", n_drafts=900, n_games=n_games // 3, **{"from": "2026-06-23", "to": "2026-06-25"}),
          dict(name="middle", n_drafts=900, n_games=n_games // 3, **{"from": "2026-06-25", "to": "2026-07-01"}),
          dict(name="late", n_drafts=900, n_games=n_games // 3, **{"from": "2026-07-01", "to": "2026-07-21"})]
    evp = []
    for i, r in enumerate(sorted(pairs, key=lambda r: r["key"])):
        e0, ch = rng.uniform(-2, 2), (3.4 if i == 0 else -3.4 if i == 1 else rng.uniform(-1, 1))
        v = "better" if i == 0 else "worse" if i == 1 else "stable"
        evp.append(dict(key=r["key"], label=r["label"], colors=r["key"], effect=[e0, e0 + ch / 2, e0 + ch], se=[1.0, 1.0, 1.0], n_decks=[300, 300, 300],
                        change=ch, change_se=1.4, change_z=ch / 1.4, change_badge=R.badge_of(ch / 1.4), halves=[ch * 0.8, ch * 1.2], replicated=True,
                        verdict=v, range=R.range_text(ch, 1.4)))
    evp.sort(key=lambda p: (p["verdict"] == "stable", -abs(p["change"])))
    named = [c for c in card_rows if c["badge"] != "weak"] or card_rows

    def evc(c: dict, up: bool) -> dict:
        ch = 22.0 if up else -18.0
        return dict(card_id=c["card_id"], name=c["name"], group=c["group"], rarity=c["rarity"], pr_early=0.4 if up else 0.7, pr_middle=0.5, pr_late=0.4 + ch / 100 if up else 0.52,
                    change_pp=ch, se_pp=4.0, z=ch / 4.0, badge="strong", halves_pp=[ch * 0.9, ch * 1.1], n_early=400, n_late=400, value_pp=c["value_pp"],
                    value_badge=c["badge"], learned=(up == (c["value_pp"] > 0)) if c["badge"] != "weak" else None, range="likely between +12 and +32 points of decks")

    cs = dict(up=[evc(c, True) for c in named[:3]], down=[evc(c, False) for c in named[3:6]], n_up=3, n_down=3, n_learned=3, n_against=2, min_elig=100, min_change=0.05)
    evolution = dict(n_drafts=2700, thirds=th, pairs=evp, min_elig=100, cards=cs, note="Observed only: the builder does not use these figures.")
    from sealed_lab import report_detail as D

    evolution["text"] = D.evolution_text(evolution)
    cmp_rows = []
    for r in pairs:
        b1, b3 = r["effect_pp"], r["effect_pp"] + rng.uniform(-2, 2)
        cmp_rows.append(dict(key=r["key"], label=r["label"], colors=r["key"], bo1=b1, bo1_se=0.8, bo1_badge=r["badge"], bo3=b3, bo3_se=1.5, bo3_badge="weak",
                             **{"diff_" + k: v for k, v in eff(b3 - b1, 1.7).items()}))
    cmp_rows.sort(key=lambda r: -abs(r["diff_z"]))
    compare = dict(pairs=cmp_rows, text=["No color pair clearly does better in one format than in the other."], cards=cards.get("compare_cards", []))
    notes = dict(n_games=n_games, n_decks=n_decks, n_drafts=n_decks // 3, date_from="2026-06-23", date_to="2026-07-21", min_eligible=150, min_exposure=500,
                 thin_exposure=2000, min_pair_decks=40, min_splash_decks=60, min_band_decks=100)
    for d in rows:
        d["text"] = D.pair_text(d, 9.5, 0.1)
    return dict(pair_details=pair_details, evolution=evolution, compare=compare, data_notes=notes)


FAKE_WRITTEN_OUTDATED = False          # tests flip this to see the "outdated" banner
FAKE_WRITTEN_NONE = False              # tests flip this to see a set without a written report
_FAKE_WRITTEN_FACTS = dict(
    set="TST", stamps={"bo1": "fake|bo1"},
    facts={"PLAY.bo1": dict(id="PLAY.bo1", fmt="bo1", text="playing first wins 3.6 points more often (range +2.6 to +4.6)", nums=[3.6, 2.6, 4.6], badge="strong"),
           "PAIR.bo1.WU": dict(id="PAIR.bo1.WU", fmt="bo1", text="White-Blue: +2.2 points (range +0.9 to +3.5)", nums=[2.2, 0.9, 3.5], badge="likely")},
    noclear={"PAIR.bo1.rest": dict(id="PAIR.bo1.rest", fmt="bo1", text="the other pairs: no clear difference")})
for _id, _fmt, _text, _nums, _badge in (        # facts of the other sections (a click on their sentences jumps there)
        ("GLANCE.bo3", "bo3", "typical game 9 turns", [9], "descriptive"),
        ("CARD.bo3.tide_ghoul", "bo3", "Tide Ghoul (blue) overrated common, -0.8 points per copy (range -1.3 to -0.3)", [0.8], "strong"),
        ("CARD.bo1.dawn_knight", "bo1", "Dawn Knight (white) top common, +0.6 points per copy (range +0.2 to +1.0)", [0.6], "likely"),
        ("SPEC.bo1.UB", "bo1", "UB: the lower-curve end beats the higher-curve end by 1.4 points (range +0.5 to +2.3)", [1.4], "likely"),
        ("SHAPEDET.bo3.lands.L17", "bo3", "decks with 17 lands +0.9 points against an average deck (range +0.3 to +1.5)", [17, 0.9], "likely"),
        ("EVOL.bo3.PAIR.WU", "bo3", "White-Blue decks got better over the format by 1.9 points (range +0.8 to +3.0)", [1.9], "likely"),
        ("BUILDER.bo1.lands", "bo1", "builder plays 16.4 lands on average, players with the same pools 16.9 (0.5 fewer; 400 pools)", [16.4, 16.9, 0.5, 400], "likely")):
    _FAKE_WRITTEN_FACTS["facts"][_id] = dict(id=_id, fmt=_fmt, text=_text, nums=_nums, badge=_badge)
FAKE_WRITTEN_TEXT = """## Headlines

- In Best of 1, playing first wins about 3.6 points more often than playing second.<!--F:PLAY.bo1-->
- White-Blue is the best pair, at +2.2 points against an average deck.<!--F:PAIR.bo1.WU-->
- For the other pairs there is no clear difference from an average deck.<!--F:PAIR.bo1.rest-->

## The cards

- In Best of 3, Tide Ghoul is overrated: players love it, but it loses 0.8 points per copy.<!--F:CARD.bo3.tide_ghoul-->
- In Best of 1, Dawn Knight is a top common, at +0.6 points per copy.<!--F:CARD.bo1.dawn_knight-->

## The decks

- Blue-Black decks range from a lower-curve end to a higher-curve end, and the lower-curve end beats the other by 1.4 points.<!--F:SPEC.bo1.UB-->
- In Best of 3, decks with 17 lands did 0.9 points better than an average deck.<!--F:SHAPEDET.bo3.lands.L17-->
- White-Blue decks got better over the format, by 1.9 points in Best of 3.<!--F:EVOL.bo3.PAIR.WU-->
- The builder plays 0.5 fewer lands than players with the same pools in Best of 1.<!--F:BUILDER.bo1.lands-->
- The typical game lasts 9 turns in Best of 3.<!--F:GLANCE.bo3-->

## How to build

Count your lands and keep the curve low; *this paragraph* has no numbers, so it needs no tag.
"""


def synthetic_facts(code: str, claims=(), noclear=(), formats=("bo1", "bo3"), mulligan=(10.0, 21.0), turns=(8.0, 9.0)) -> dict:
    """Per-set facts (the structure of `report_facts.facts_from_reports`, only the fields `sealed_lab.cross_set` reads) for tests and the fake backend.
    claims: (id, badge, value[, extra dict]) - kind "card_trap" for ids starting CARD; noclear: ids.  A GLANCE fact per format is added."""
    facts: dict = {}
    for fmt, mu, tu in zip(formats, mulligan, turns):
        fid = f"GLANCE.{fmt}"
        facts[fid] = dict(id=fid, fmt=fmt, kind="glance", text="glance", nums=[mu, tu], badge="descriptive", builder_uses="observed", mulligan_pct=mu, median_turns=tu)
    for c in claims:
        fid, badge, value, extra = (tuple(c) + ({},))[:4]
        kind = "card_trap" if fid.startswith("CARD") else "claim"
        facts[fid] = dict(id=fid, fmt=fid.split(".")[1], kind=kind, text=f"{fid} {value:+}", nums=[value], badge=badge, builder_uses="observed", value=value, **extra)
    return dict(version=2, report_version=7, set=code, formats=list(formats), stamps={f: f"{code}|{f}" for f in formats}, builder_available=True,
                facts=facts, noclear={i: dict(id=i, fmt=i.split(".")[1], text=i) for i in noclear})


def fake_cross_per_set() -> dict[str, dict]:
    """Three small sets with a pattern (playing first, Bo1), a mixed result (splash, Bo1), a rare one (Bo3 play) and a few more."""
    spec = lambda w, o: dict(ends=dict(a="x", b="y", winner="x", winner_avg_mv=w, other_avg_mv=o))
    return {
        "AAA": synthetic_facts("AAA", [("PLAY.bo1", "strong", 5.0), ("PLAY.bo3", "likely", 3.0), ("SPLASH.bo1", "strong", -1.5), ("PAIR.bo1.WU", "likely", 2.0),
                                        ("PAIR.bo1.UB", "likely", -2.5), ("SPEC.bo1.WU", "likely", 2.7, spec(2.8, 3.1)),
                                        ("CARD.bo1.big_dragon", "strong", -1.0, dict(name="Big Dragon")), ("BUILDER.bo1.lands", "likely", -0.6)],
                                ["SPLASH.bo3", "PAIR.bo3.WU", "PAIR.bo3.UB", "BUILDER.bo3.lands", "BUILDERCHECK.bo1", "SHAPEDET.bo1.lands"]),
        "BBB": synthetic_facts("BBB", [("PLAY.bo1", "likely", 4.0), ("SPLASH.bo1", "likely", 1.0), ("PAIR.bo1.WU", "strong", 2.5), ("PAIR.bo1.UB", "likely", -3.0),
                                        ("SPEC.bo1.WU", "likely", 3.5, spec(3.4, 3.0)), ("CARD.bo1.odd_elf", "likely", -0.8, dict(name="Odd Elf")),
                                        ("BUILDER.bo1.lands", "likely", -0.4), ("SHAPEDET.bo1.lands.L16", "likely", 1.0)],
                                ["PLAY.bo3", "SPLASH.bo3", "PAIR.bo3.WU", "PAIR.bo3.UB", "BUILDER.bo3.lands", "BUILDERCHECK.bo1"]),
        "CCC": synthetic_facts("CCC", [("PLAY.bo1", "strong", 6.0), ("PAIR.bo1.WU", "likely", 1.8), ("PAIR.bo1.UB", "strong", -1.5), ("CARD.bo3.old_wall", "likely", -0.7, dict(name="Old Wall")),
                                        ("BUILDER.bo1.lands", "likely", -0.9), ("PAIR.bo3.UB", "weak", -1.0), ("SPLASH.bo1", "weak", 0.2)],
                                ["PLAY.bo3", "SPLASH.bo3", "PAIR.bo3.WU", "BUILDER.bo3.lands", "BUILDERCHECK.bo1"], mulligan=(12.0, 23.0), turns=(9.0, 9.0)),
    }


FAKE_CROSS_TEXT = """## Headlines

- In Best of 1, playing first did better in a clear way in 3 of 3 sets, a pattern.<!--F:XSET.bo1.PLAY-->
- White-Blue decks were clearly above an average deck in 3 of 3 Best of 1 sets, a pattern.<!--F:XSET.bo1.PAIR.WU-->
- Splashing gave mixed results in Best of 1.<!--F:XSET.bo1.SPLASH-->
- In Best of 3 there is no clear pattern for playing first.<!--F:XSET.bo3.PLAY-->
"""


def written_report(set_code: str | None = None) -> dict:
    """A short sample written report for TST and for "ALL" (the real ones are stored under <app home>/written_reports)."""
    out: dict = dict(text=None, meta=None, outdated=False, available=[] if FAKE_WRITTEN_NONE else ["ALL", "TST"])
    if set_code and set_code.upper() == "TST" and not FAKE_WRITTEN_NONE:
        out.update(text=FAKE_WRITTEN_TEXT, meta=dict(set="TST", stamp=dict(_FAKE_WRITTEN_FACTS["stamps"]), written_at="2026-10-21T09:00:00+00:00",
                                                    facts=_FAKE_WRITTEN_FACTS), outdated=FAKE_WRITTEN_OUTDATED)
    if set_code and set_code.upper() == "ALL" and not FAKE_WRITTEN_NONE:
        from sealed_lab import cross_set as X

        facts = X.cross_from_facts(fake_cross_per_set())["facts"]
        out.update(text=FAKE_CROSS_TEXT, meta=dict(set="ALL", stamp=dict(facts["stamps"]), written_at="2026-10-21T09:00:00+00:00", facts=facts),
                   outdated=FAKE_WRITTEN_OUTDATED)
    return out


def cross_set(sets=None, refresh: bool = False, progress=None) -> dict:
    """The cross-set lessons of the three synthetic sets (the real one is `sealed_lab.cross_set.build_cross`)."""
    from sealed_lab import cross_set as X

    time.sleep(_delay())
    if progress:
        progress(0.5, "Facts of the sets")
    return X.cross_from_facts(fake_cross_per_set())


def cheat_sheet(set_code: str, fmt: str = "bo3") -> str:
    """The one-page cheat sheet of the fake set (same assembly as the real api)."""
    from sealed_lab import cheatsheet

    _guard(set_code)
    return cheatsheet.cheat_sheet_html(set_code.upper(), fmt, set_report(set_code, fmt), card_info(set_code), card_values(set_code, fmt))


def export_set_html(set_code: str) -> str:
    """The export page of the fake set (written report + both synthetic analyses)."""
    from sealed_lab import export_html

    _guard(set_code)
    written = written_report(set_code)
    reports = {f: set_report(set_code, f) for f in ("bo1", "bo3")}
    return export_html.export_set_html(set_code.upper(), written, reports, (written.get("meta") or {}).get("facts"))


def _fake_archetypes(set_code: str, fmt: str, pair_rows: list[dict], card_rows: list[dict], rng: random.Random) -> tuple[dict, dict]:
    """Synthetic archetype and deck-shape sections: three pairs (the first with a splash) get two sub-archetypes, the others none."""
    from sealed_lab import report as R
    from sealed_lab import report_arch as A

    info = card_info(set_code)
    names = {c: [(i, d["name"]) for i, d in info.items() if not d["is_land"] and set(d["colors"]) <= set(c)] for c in R.design.PAIRS}
    pairs = []
    n_decks = sum(r["n_decks"] for r in pair_rows)
    mu = {"WU": dict(best=dict(vs="BR", vs_label="Black-Red", effect_pp=4.1, se_pp=1.4, z=2.9, badge="likely", range=R.range_text(4.1, 1.4), n_games=310)),
          "UB": dict(worst=dict(vs="WG", vs_label="White-Green", effect_pp=-5.0, se_pp=1.8, z=-2.8, badge="likely", range=R.range_text(-5.0, 1.8), n_games=260))}         if fmt == "bo3" else {}
    for r in sorted((r for r in pair_rows if len(r["key"]) == 2 and not r["hidden"]), key=lambda r: -r["n_decks"]):
        p = r["key"]
        pool = names.get(p, []) or [(0, "Card")]
        pick = lambda k: [dict(card_id=i, name=n, rate=rng.uniform(0.25, 0.7), ref_rate=rng.uniform(0.05, 0.2)) for i, n in rng.sample(pool, min(k, len(pool)))]
        vals = [c for c in card_rows if set(c["colors"]) <= set(p) and c["value_pp"] > 0][:5]
        best = [dict(card_id=c["card_id"], name=c["name"], rarity=c["rarity"], value_pp=c["value_pp"], z=c["z"], badge=c["badge"], range=c["range"],
                     thin=False, play_rate=rng.uniform(0.15, 0.6), reason=f"In 40% of these decks; {R.signed(c['value_pp'])} points of win rate per copy.") for c in vals]
        splash_share = r["splash_share"]
        shape = dict(lands=17.0, creatures=float(rng.choice([14, 15, 16])), two_drops=float(rng.choice([3, 4, 5])), avg_mv=rng.uniform(2.7, 3.2),
                     splash_cards=float(rng.choice([0, 1, 2])), splash_share=splash_share)
        splash_cols = {"WU": ["R"], "UB": ["R", "G"], "BG": ["U"]}.get(p, [])
        pair = dict(key=p, label=r["label"], code=A.code_of(p, splash_cols), splashy=bool(splash_cols), splash_cols=splash_cols,
                    splash_col_shares={c: 0.3 for c in splash_cols}, splash_share=splash_share,
                    splash_top="R" if "R" not in p else "G", splash_top_label="Red" if "R" not in p else "Green", splash_top_share=0.4, n_decks=r["n_decks"],
                    share=r["share"], effect_pp=r["effect_pp"], se_pp=r["se_pp"], z=r["z"], badge=r["badge"], range=r["range"], signature=pick(3), shape=shape,
                    best_cards=best, matchups=mu.get(p, {}), subs=[])
        split_wanted = p in ("WU", "UB", "BG")
        tried = [dict(k=2, ari=0.71 if split_wanted else 0.12, aris=[0.7, 0.71, 0.74], min_share=0.4, min_decks=int(0.4 * r["n_decks"]), silhouette=0.31 if split_wanted else 0.11,
                      silhouette_needed=0.14, replicates=split_wanted, distinct=split_wanted or None, accepted=split_wanted,
                      why="" if split_wanted else "clusters do not match across halves")]
        pair["spectrum"] = _fake_spectrum(p, pick, rng)
        pair["split"] = dict(k=2 if split_wanted else 1, ari=0.71 if split_wanted else None, silhouette=0.31 if split_wanted else None, tried=tried, why="")
        if split_wanted:
            for j, (style, words, d_eff) in enumerate((("aggro", ["lower curve", "more creatures"], 1.6), ("control", ["higher curve", "more removal", "more card draw"], -1.2))):
                share_p = 0.58 if j == 0 else 0.42
                e = r["effect_pp"] + d_eff
                se = r["se_pp"] * 1.4
                sub = dict(key=f"{p}#{j + 1}", name=f"{p} {style}", style=style, words=words, n_decks=int(r["n_decks"] * share_p), share_of_pair=share_p,
                           share=r["share"] * share_p, signature=pick(3), shape=dict(shape, creatures=shape["creatures"] + (1 if j == 0 else -2),
                           avg_mv=shape["avg_mv"] + (-0.25 if j == 0 else 0.3), two_drops=shape["two_drops"] + (1 if j == 0 else -1)),
                           best_cards=best[: 4 - j], style_diff=dict(dmv=-0.25 if j == 0 else 0.3, dcre=0.04, drem=0.0, ddraw=0.0, dfly=0.0, dcheap=0.0))
                sub.update(A.effect_fields(e, se))
                sub.update(A.effect_fields(d_eff, se * 0.8, "vs_pair_"))
                sub["text"] = A.sub_text(sub, pair)
                pair["subs"].append(sub)
        pair["text"] = A.pair_text(pair)
        pairs.append(pair)
    arch = dict(note=A.BUILDER_NOTE, splash_x=A.SPLASH_COLOR_X, axis_min_corr=A.AXIS_MIN_CORR, pairs=pairs, n_decks=n_decks,
                n_with_subs=sum(1 for a in pairs if a["subs"]), n_with_spectrum=sum(1 for a in pairs if a["spectrum"].get("shown")),
                params=dict(ari_min=A.ARI_MIN, min_cluster_share=A.MIN_CLUSTER_SHARE, min_cluster_decks=A.MIN_CLUSTER_DECKS, k_max=A.K_MAX, n_splits=A.N_SPLITS))
    metric_base = dict(lands=(16, 17, 17), creatures=(13, 15, 16), two_drops=(3, 4, 5), avg_mv=(2.7, 2.95, 3.2), splash_cards=(0, 1, 3))

    def rng_of(vals, shift=0.0):
        return dict(lo=vals[0] + shift, med=vals[1] + shift, hi=vals[2] + shift)

    rows = []
    for key, label in [("all", "All decks")] + [(a["key"], a["label"]) for a in pairs]:
        nd = n_decks if key == "all" else next(a["n_decks"] for a in pairs if a["key"] == key)
        strong = nd // 5
        metrics = {m: dict(all=rng_of(v), strong=rng_of(v, 0.0 if m in ("lands",) else (-0.1 if m == "avg_mv" else 0.0)) if strong >= A.STRONG_MIN_DECKS else None)
                   for m, v in metric_base.items()}
        rows.append(dict(key=key, label=label, n_decks=nd, n_strong=strong, metrics=metrics))
    sdef = "won 5 or more games" if fmt == "bo1" else "won 3 or more matches"
    shape = dict(rows=rows, metrics=[dict(key=k, label=l) for k, l in A.SHAPE_METRICS], strong_def=sdef, strong_min_decks=A.STRONG_MIN_DECKS,
                 percentiles=list(A.SHAPE_PCT), note=A.SHAPE_NOTE,
                 text=["Most decks (the middle half) play 16-17 lands, 13-16 creatures and 3-5 two-drops.",
                       f"\"Strong results\" means the deck {sdef}. Skill is not adjusted here, so this shows what such decks look like, not what causes wins."])
    return arch, shape


def _fake_spectrum(p: str, pick, rng: random.Random) -> dict:
    """Synthetic style spectrum: UB (and WR) with a win-rate finding, WU / BG with "no clear difference", RG with an axis that does not replicate."""
    from sealed_lab import report as R
    from sealed_lab import report_arch as A

    if p == "RG":
        return dict(shown=False, why="the main style axis does not replicate", corr=0.41, corrs=[0.4, 0.41, 0.5], min_corr=A.AXIS_MIN_CORR)
    if p not in ("UB", "WU", "BG", "WR"):
        return dict(shown=False, why="too few decks")
    finding = p in ("UB", "WR")
    words = {"UB": (["lower curve", "more removal"], ["more fliers", "more card draw"], "removal / low curve", "fliers / card draw"),
             "WU": (["lower curve", "more 2-drops"], ["higher curve", "more fliers"], "low curve / 2-drops", "high curve / fliers"),
             "BG": (["more removal", "lower curve"], ["more creatures", "higher curve"], "removal / low curve", "creatures / high curve"),
             "WR": (["more 2-drops", "lower curve"], ["more removal", "higher curve"], "2-drops / low curve", "removal / high curve")}[p]
    def end(w, label, effect, n=300):
        return dict(label=label, words=w, description=A._join_words(w), n_decks=n, signature=pick(3), avg_mv=3.0,
                    **A.effect_fields(effect, 1.1))
    a, b = end(words[0], words[2], 1.8 if finding else 0.3), end(words[1], words[3], -1.6 if finding else 0.1)
    d, se = (3.4, 1.2) if finding else (0.4, 1.3)
    win = dict(**A.effect_fields(-d if p == "UB" else d, se), finding=finding, replicated=finding, halves=[-d * 0.8, -d * 1.2] if p == "UB" else [d * 0.8, d * 1.2])
    if not finding:
        win["halves"] = [0.9, -0.3]
    return dict(shown=True, corr=0.93, corrs=[0.9, 0.93, 0.95], min_corr=A.AXIS_MIN_CORR, a=a, b=b, win=win, differences={},
                text=A.spectrum_texts(p, a, b, win))


def card_image(set_code: str, name: str, fetch: bool = True) -> str | None:
    return None  # the GUI draws placeholders


_INTERCEPT = -0.25          # toy baseline logit (the average toy deck then wins about 55%)
_anchors: list[tuple[dict[int, int], float]] = []   # generated decks: (cards, logit offset) so that re-evaluating an
                                                    # unedited generated deck gives its generated win rate


def _anchor_offset(deck: dict[int, int]) -> float:
    """Logit offset of the generated deck that `deck` is closest to (>= 30 shared cards), else 0."""
    best, off = 0, 0.0
    for cnt, o in _anchors:
        shared = sum(min(n, cnt.get(i, 0)) for i, n in deck.items())
        if shared > best:
            best, off = shared, o
    return off if best >= 30 else 0.0


def _register_anchor(counts: dict[int, int], wr: float) -> None:
    import math

    wr = min(max(wr, 1e-4), 1 - 1e-4)
    base = _INTERCEPT + sum(_beta(i) * n for i, n in counts.items() if not _CARDS.get(i, {}).get("is_land"))
    lands = sum(n for i, n in counts.items() if _CARDS.get(i, {}).get("is_land"))
    _anchors.append((dict(counts), math.log(wr / (1 - wr)) - (base - 0.08 * abs(lands - 17))))


def _sources(deck: dict[int, int]) -> dict[str, int]:
    """Cards of the deck that make each color (basics and the dual land)."""
    src = {c: 0 for c in _COLORS}
    for i, n in deck.items():
        d = _CARDS.get(i)
        if d and d["is_land"]:
            for c in (d["colors"] if d["name"] in _BASICS.values() else "WU"):
                src[c] += n
    return src


def _castable(deck: dict[int, int], cid: int) -> float | None:
    """Toy P(castable on time): 1 - exp(-0.32 * sources) per color of the card, multiplied; colorless cards 0.98."""
    import math

    d = _CARDS.get(cid)
    if d is None or d["is_land"]:
        return None
    src = _sources(deck)
    p = 0.98
    for c in d["colors"]:
        p *= 1 - math.exp(-0.32 * src.get(c, 0))
    return round(p, 3)


def _support(deck: dict[int, int], fmt: str) -> list[str]:
    """Toy 'outside what real decks look like' reasons: odd land counts, colors without sources, three main colors."""
    lands = sum(n for i, n in deck.items() if _CARDS.get(i, {}).get("is_land"))
    out: list[str] = []
    if lands < 15 or lands > 18:
        share = "0.8%" if lands < 15 else "1.5%"
        out.append(f"{lands} lands: only {share} of real {fmt.upper()} decks play {'fewer than 15' if lands < 15 else 'more than 18'}")
    src = _sources(deck)
    cols: dict[str, int] = {}
    for i, n in deck.items():
        d = _CARDS.get(i)
        if d and not d["is_land"]:
            for c in d["colors"]:
                cols[c] = cols.get(c, 0) + n
    for c, n in sorted(cols.items()):
        if n >= 3 and src[c] < 2:
            out.append(f"{n} {_COLOR_WORD[c]} cards but only {src[c]} {_COLOR_WORD[c]} source(s): real decks with that many have at least 2")
    if sum(1 for n in cols.values() if n >= 8) >= 3:
        out.append("three colors with eight or more cards each: only about 3% of real decks do this")
    return out


_COLOR_WORD = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}


def evaluate_deck(set_code: str, deck: dict[int, int], fmt: str = "bo3") -> DeckEval:
    """Deterministic toy evaluation: logit = _INTERCEPT + sum of card betas - 0.08 * |lands - 17|."""
    import math

    _guard(set_code)
    if not deck:
        raise ValueError("the deck is empty")
    lands = sum(n for i, n in deck.items() if _CARDS.get(i, {}).get("is_land"))
    L = _toy_logit(deck)
    wr = 1 / (1 + math.exp(-L))
    cards = [_line(i, _CARDS[i]["name"], n, None if _CARDS[i]["name"] in _BASICS.values()
                   else round(100 * (wr - 1 / (1 + math.exp(-(L - _beta(i))))), 2), i % 7 == 0, castable=_castable(deck, i))
             for i, n in deck.items() if i in _CARDS]
    n = sum(deck.values())
    se = 0.021
    sc = 100 * min(0.98, max(0.02, (wr - 0.44) / 0.2))
    return DeckEval(cards=cards, colors="WU", splash="", game_wr=wr, game_wr_lo=wr - 1.28 * se, game_wr_hi=wr + 1.28 * se,
                    match_wr=wr * wr * (3 - 2 * wr) if fmt == "bo3" else None,
                    score=round(sc, 1), score_lo=max(0.0, round(sc - 9, 1)), score_hi=min(100.0, round(sc + 8, 1)),
                    exp_wins=2.0, p_max_wins=0.1,
                    breakdown={"intercept": _INTERCEPT, "cards": L - _INTERCEPT - (-0.08 * abs(lands - 17)),
                               "structure": -0.08 * abs(lands - 17), "mana": 0.0},
                    mana={"p_screw": 0.08, "p_flood": 0.07, "p_color_screw": 0.05, "mana_spent": 0.72, "mull_rate": 0.09},
                    warnings=[] if n == 40 else [f"deck has {n} cards (the model is fitted on 40-card decks)"],
                    support=_support(deck, fmt))


# --------------------------------------------------------------------------- registry state

def _fi(path: str, kind: str, set_code: str | None, event: str | None, fmt: str | None, size_mb: float,
        note: str = "") -> FileInfo:
    fp = hashlib.sha1(f"{path}|{size_mb}".encode()).hexdigest()[:16]
    return FileInfo(path=path, kind=kind, set_code=set_code, event=event, fmt=fmt, size=int(size_mb * 1e6),
                    fingerprint=fp, note=note)


_files: dict[str, list[FileInfo]] = {}
_models: dict[str, dict] = {}
_folders: dict[str, str] = {}              # per-set folders (set_set_folder)
_missing: dict[str, list[str]] = {}        # registered files "gone from disk" (tests: mark_missing)
_scan_seen: dict[str, tuple[int, int]] = {}    # folder-scan cache: path -> (size, mtime_ns)
_calc: dict[str, int] = {}                 # sets being calculated right now (readers raise SetBusy)
untracked_remote: list[dict] = []          # sets on "17lands' list" that are not tracked (tests fill it)
scan_calls: list[dict] = []                # arguments of every scan_data_folders call (tests)
calc_fail: dict[str, str] = {}             # tests: set -> error message raised by calculate_set


def reset() -> None:
    """Restore the initial fake state (tests)."""
    with _lock:
        _anchors.clear()
        _files.clear()
        _models.clear()
        _folders.clear()
        _missing.clear()
        _scan_seen.clear()
        _calc.clear()
        untracked_remote.clear()
        scan_calls.clear()
        calc_fail.clear()
        _files["TST"] = [
            _fi("/data/game_data_public.TST.PremierDraft.csv.gz", "game", "TST", "PremierDraft", "draft", 412.3),
            _fi("/data/game_data_public.TST.TradSealed.csv.gz", "game", "TST", "TradSealed", "bo3", 18.4),
            _fi("/data/game_data_public.TST.Sealed.csv.gz", "game", "TST", "Sealed", "bo1", 46.9),
            _fi("/data/draft_data_public.TST.PremierDraft.csv.gz", "draft", "TST", "PremierDraft", "draft", 98.0,
                "registered, not used"),
        ]
        _files["NEO"] = [
            _fi("/data/game_data_public.NEO.PremierDraft.csv.gz", "game", "NEO", "PremierDraft", "draft", 205.0),
        ]
        _models["TST"] = {"formats": ["bo3", "bo1"], "n_games": {"draft": 601_204, "bo1": 49_977, "bo3": 20_310},
                          "built_at": "2026-10-20 18:42", "stale": False, "pending": [],
                          "warnings": ["bo3: only 20,310 games - card values lean on the Bo1/draft prior"]}


reset()


def _status(code: str) -> SetStatus:
    m = _models.get(code)
    files = list(_files.get(code, []))
    stale = bool(m and m["stale"]) or (m is None and any(f.kind == "game" for f in files))
    pending = list(m.get("pending") or []) if m else (["data never used yet: no model has been calculated for this set"] if stale else [])
    if stale and not pending:
        pending = ["the data changed since the model was built"]
    return SetStatus(
        set_code=code, files=files, has_corpus=m is not None, has_model=m is not None,
        stale=stale, formats=list(m["formats"]) if m else [],
        n_games=dict(m["n_games"]) if m else {}, built_at=m["built_at"] if m else None,
        warnings=list(m["warnings"]) if m else [], pending=pending if stale else [], missing=list(_missing.get(code, [])),
        folder=_folders.get(code, ""))


def mark_missing(set_code: str, path: str) -> None:
    """Tests: a registered file of the set vanished from disk."""
    with _lock:
        _missing.setdefault(set_code, []).append(path)


def _guard(set_code: str) -> None:
    if _calc.get(set_code.upper()):
        raise SetBusy(f"{set_code.upper()} is being recalculated: its model is available again when the calculation finishes")


def calculating_sets() -> list[str]:
    return sorted(c for c, n in _calc.items() if n)


def untrack_set(set_code: str) -> None:
    sc = set_code.upper()
    _guard(sc)
    with _lock:
        _files.pop(sc, None)
        _models.pop(sc, None)
        _missing.pop(sc, None)


def list_sets() -> list[SetStatus]:
    time.sleep(_delay())
    with _lock:
        return [_status(c) for c in sorted(_files)]


_NAME_RE = re.compile(r"(game|draft|replay)_data_public\.([A-Za-z0-9]+)\.([A-Za-z]+)\.csv(\.gz)?$")


def _detect(path: Path) -> FileInfo:
    size = path.stat().st_size if path.exists() else 0
    fp = hashlib.sha1(f"{path}|{size}".encode()).hexdigest()[:16]
    m = _NAME_RE.search(path.name)
    if not m:
        return FileInfo(str(path), "unknown", None, None, None, size, fp, "could not detect file type")
    kind, code, event = m.group(1), m.group(2).upper(), m.group(3)
    fmt = "bo3" if event == "TradSealed" else "bo1" if event == "Sealed" else "draft"
    note = "registered, not used" if kind != "game" else ""
    return FileInfo(str(path), kind, code, event, fmt, size, fp, note)


def add_files(paths: list[str]) -> list[FileInfo]:
    time.sleep(_delay() * 3)
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(x for x in p.rglob("*") if x.is_file())
        else:
            found.append(p)
    out = []
    with _lock:
        for p in found:
            fi = _detect(p)
            out.append(fi)
            if fi.set_code:
                lst = _files.setdefault(fi.set_code, [])
                lst[:] = [x for x in lst if not (x.kind == fi.kind and x.event == fi.event)] + [fi]
                if fi.set_code in _models:
                    _note_new_file(fi.set_code, fi)
    return out


def _note_new_file(code: str, fi: FileInfo) -> None:
    m = _models[code]
    m["stale"] = True
    names = m.setdefault("new_files", [])
    if Path(fi.path).name not in names:
        names.append(Path(fi.path).name)
    m["pending"] = [f"{len(names)} new or changed file{'s' if len(names) != 1 else ''}: " + ", ".join(names[:2])
                    + (" …" if len(names) > 2 else "")]


def remove_file(set_code: str, fingerprint: str) -> None:
    time.sleep(_delay())
    with _lock:
        _files[set_code] = [f for f in _files.get(set_code, []) if f.fingerprint != fingerprint]
        if set_code in _models:
            _models[set_code]["stale"] = True
            _models[set_code]["pending"] = ["1 file the model was computed from was replaced or removed"]
        if not _files[set_code]:
            del _files[set_code]
            _models.pop(set_code, None)


def set_status(set_code: str) -> SetStatus:
    with _lock:
        return _status(set_code)


_STAGES = [
    (0.00, "Scanning source files"), (0.12, "Ingesting game data (parquet cache)"),
    (0.35, "Fetching Scryfall card data"), (0.42, "Fitting draft prior (5-fold CV)"),
    (0.58, "Fitting Bo1 stage"), (0.70, "Fitting Bo3 stage"), (0.80, "Bootstrap replicates"),
    (0.90, "Validation report"), (0.97, "Writing model artifact"),
]


def calculate_set(set_code: str, progress: Callable[[float, str], None] | None = None, cancel=None,
                  n_jobs: int | None = None, n_boot: int = 16, fetch_scryfall: bool = True,
                  recency: str = "auto", attributes: str = "auto", crossset: str = "auto", synergy: str = "auto",
                  archcond: str = "auto", matchups: str = "pooled", manasim: str = "auto", castability: str = "auto",
                  extra_draft: str = "auto", quick: bool = False) -> SetStatus:
    steps = 40
    with _lock:
        _calc[set_code.upper()] = _calc.get(set_code.upper(), 0) + 1
    try:
        if calc_fail.get(set_code):
            raise RuntimeError(calc_fail[set_code])
        for i in range(steps + 1):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            frac = i / steps
            msg = [m for f, m in _STAGES if f <= frac][-1]
            if not fetch_scryfall and msg.startswith("Fetching"):
                msg = "Skipping Scryfall (offline)"
            if progress:
                progress(frac, msg)
            time.sleep(_delay())
        with _lock:
            _models[set_code] = {
                "formats": ["bo3", "bo1"], "n_games": {"draft": 601_204, "bo1": 49_977, "bo3": 20_310},
                "built_at": time.strftime("%Y-%m-%d %H:%M"), "stale": False, "pending": [],
                "warnings": [f"{n_boot} bootstrap replicates: intervals are rough"] if n_boot < 20 else [],
            }
            return _status(set_code)
    finally:
        with _lock:
            _calc[set_code.upper()] -= 1


def calculate_set_isolated(set_code: str, progress: Callable[[float, str], None] | None = None, cancel=None,
                           retries: int = 1, **options) -> SetStatus:
    """The real one runs `calculate_set` in a child process; the fake runs it in place (same signature)."""
    return calculate_set(set_code, progress=progress, cancel=cancel, **options)


def calibration_note(set_code: str, fmt: str) -> dict | None:
    return None                      # (no hold-out in the fake: the real one returns None for such a format too)


def prefetch_images(set_code: str, names: list[str]) -> int:
    return 0                         # (the fake draws placeholders)


def set_release_dates(codes) -> dict[str, str]:
    return {}                        # (the real one is offline-safe: unknown sets have no date)


def report_path(set_code: str) -> Path | None:
    if set_code not in _models:
        return None
    p = Path(tempfile.gettempdir()) / f"sealed_lab_fake_report_{set_code}.html"
    p.write_text(f"<html><body><h1>Fake report for {set_code}</h1><p>Generated by the fake backend.</p></body></html>",
                 encoding="utf-8")
    return p


# --------------------------------------------------------------------------- parsing helpers

_LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)(?:\s+\([A-Za-z0-9]+\)\s+\S+)?\s*$")


def _parse(text: str) -> tuple[dict[str, int], dict[str, int]]:
    main: dict[str, int] = {}
    side: dict[str, int] = {}
    cur = main
    saw_header = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if main and not saw_header:
                cur = side
            continue
        low = line.lower()
        if low in ("deck", "main", "maindeck"):
            cur, saw_header = main, True
            continue
        if low == "sideboard":
            cur, saw_header = side, True
            continue
        m = _LINE_RE.match(line)
        if m:
            cur[m.group(2)] = cur.get(m.group(2), 0) + int(m.group(1))
    return main, side


# --------------------------------------------------------------------------- fake evaluation

def _line(cid: int | None, name: str, count: int, value: float | None, low: bool, note: str = "",
          castable: float | None = None) -> CardLine:
    return CardLine(card_id=cid, name=name, count=count, value_pp=value, low_data=low, note=note, castable=castable)


def _fake_deck(seed: int, colors: str, splash: str, base_wr: float, names: dict[str, int] | None = None,
               fmt: str = "bo3") -> DeckEval:
    rng = random.Random(seed)
    pool_ids = [i for i, d in _CARDS.items()
                if not d["is_land"] and (d["colors"] == "" or all(c in colors + splash for c in d["colors"]))]
    rng.shuffle(pool_ids)
    cards: list[CardLine] = []
    if names:  # a user deck: use as given
        for nm, n in names.items():
            cid = _BY_NAME.get(nm.casefold())
            if cid is None:
                cards.append(_line(None, nm, n, None, False, "unmatched"))
            elif _CARDS[cid]["is_land"] and nm in _BASICS.values():
                cards.append(_line(cid, _CARDS[cid]["name"], n, None, False))
            else:
                r = random.Random(f"{seed}{cid}")
                cards.append(_line(cid, _CARDS[cid]["name"], n, round(r.gauss(0.5, 1.1), 2), r.random() < 0.12))
    else:
        spells = pool_ids[:23]
        for cid in spells:
            r = random.Random(f"{seed}{cid}")
            sp = bool(splash) and _CARDS[cid]["colors"] == splash
            cards.append(_line(cid, _CARDS[cid]["name"], 1, round(r.gauss(0.6, 1.2), 2), r.random() < 0.15,
                               "splash" if sp else ""))
        main_cols = list(colors) or ["W"]
        n_dual = 1 if len(main_cols) > 1 else 0
        n_splash = 3 if splash else 0
        basics = 17 - n_dual - n_splash
        splits = [basics // len(main_cols) + (1 if i < basics % len(main_cols) else 0) for i in range(len(main_cols))]
        for col, n in zip(main_cols, splits):
            cards.append(_line(_BASIC_ID[col], _BASICS[col], n, None, False))
        for col in splash:
            cards.append(_line(_BASIC_ID[col], _BASICS[col], n_splash, None, False))
        if n_dual:
            cards.append(_line(_DUAL_ID, "Scoured Crossroads", 1, round(rng.gauss(0.1, 0.2), 2), False))
    counts = {c.card_id: c.count for c in cards if c.card_id is not None}
    for c in cards:
        if c.card_id is not None:
            c.castable = _castable(counts, c.card_id)
    if names is None:
        _register_anchor(counts, base_wr)
    wr = base_wr
    se = 0.021
    mw = wr * wr * (3 - 2 * wr) if fmt == "bo3" else None
    return DeckEval(
        cards=cards, colors=colors, splash=splash, game_wr=wr, game_wr_lo=wr - 1.28 * se, game_wr_hi=wr + 1.28 * se,
        match_wr=mw, score=round(100 * min(0.98, max(0.02, (wr - 0.44) / 0.2)), 1),
        score_lo=max(0.0, round(100 * (wr - 0.44) / 0.2 - 9, 1)), score_hi=min(100.0, round(100 * (wr - 0.44) / 0.2 + 8, 1)),
        exp_wins=round(2.0 + (wr - 0.5) * 9, 2), p_max_wins=round(0.12 + (wr - 0.5) * 1.6, 3),
        breakdown={"intercept": 0.08, "cards": round(0.55 + (wr - 0.57) * 6, 3), "structure": 0.06,
                   "mana": round(-0.07 + rng.random() * 0.06, 3)},
        mana={"p_screw": 0.081 + rng.random() * 0.02, "p_flood": 0.074 + rng.random() * 0.02,
              "p_color_screw": 0.046 + rng.random() * 0.03, "mana_spent": 0.71 + rng.random() * 0.05,
              "mull_rate": 0.087 + rng.random() * 0.01},
        warnings=["3 cards have little data (low_data); their values lean on the prior"] if names is None else [],
        support=_support(counts, fmt) if names is not None else [],
    )


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _own_deck(text: str) -> bool:
    """True for an Arena deck export with a Deck AND a Sideboard section (the Deck is the user's own build)."""
    from sealed_lab.deck.arena import parse_arena_full

    p = parse_arena_full(text)
    return bool(p.had_headers and p.main and p.side)


def build_pool(set_code: str, text: str, fmt: str = "bo3", n_decks: int = 3, min_diff: int = 10,
               progress=None, cancel=None, locked=None, excluded=None, name_overrides=None) -> BuildResult:
    from sealed_lab.deck.names import apply_overrides

    _guard(set_code)
    if not text.strip():
        raise ValueError("The pool is empty.")
    main, side = _parse(text)
    main, side = apply_overrides(main, name_overrides), apply_overrides(side, name_overrides)
    unmatched = [n for n in {**main, **side} if n.casefold() not in _BY_NAME and n not in _BASICS.values()]
    for i in range(10):
        _check(cancel)
        if progress:
            progress(i / 10, ["Matching card names", "Enumerating configurations", "Solving MILPs",
                              "Local search"][min(3, i // 3)])
        time.sleep(_delay())
    if locked or excluded:
        pool_ids: dict[int, int] = {}
        for nm, n in {**main, **side}.items():
            cid = _BY_NAME.get(nm.casefold())
            if cid is not None and nm not in _BASICS.values():
                pool_ids[cid] = pool_ids.get(cid, 0) + n
        return build_from_pool(set_code, pool_ids, fmt, n_decks=n_decks, progress=progress, cancel=cancel,
                               locked=locked, excluded=excluded)
    plans = [("WU", "R", 0.573), ("WB", "", 0.568), ("UG", "", 0.554)][:max(1, n_decks)]
    decks = [_fake_deck(zlib.crc32(f"{set_code}{fmt}{i}".encode()) & 0xffff, c, s, wr, fmt=fmt) for i, (c, s, wr) in enumerate(plans)]
    if progress:
        progress(1.0, "Done")
    pool: dict[int, int] = {}
    for nm, n in {**main, **side}.items():
        cid = _BY_NAME.get(nm.casefold())
        if cid is not None and nm not in _BASICS.values():
            pool[cid] = pool.get(cid, 0) + n
    p_best = [0.56, 0.31, 0.08][:len(decks)]
    fams = [_fake_family(d, i, p_best[i], decks[0].game_wr, pool) for i, d in enumerate(decks)]
    user = None
    user_unmatched: list[str] = []
    if _own_deck(text):
        user = _fake_deck(11, "WU", "", 0.548, names=main, fmt=fmt)
        user_unmatched = [n_ for n_ in main if n_.casefold() not in _BY_NAME and n_ not in _BASICS.values()]
    meta = dict(n_draws=16, n_candidates=187, n_families=9, n_shown=len(decks), n_hidden=9 - len(decks), cut=7.0, show_p_best=0.05,
                show_gap_pp=1.0, max_families=n_decks, jobs=12, seconds=11.2, step=1 / 16)
    return BuildResult(set_code=set_code, fmt=fmt, decks=decks, p_best=p_best,
                       diffs=[0, 12, 15][:len(decks)], pool_unmatched=sorted(unmatched),
                       warnings=["Pool has only 2 cards in green: UG build is weak"], pool=pool, families=fams, family_meta=meta,
                       user_deck=user, user_unmatched=user_unmatched)


def _fake_family(deck: DeckEval, rank: int, p_best: float, top_wr: float, pool: dict[int, int]) -> dict:
    """Family details consistent with a fake deck: first 15 spells are core, the next 4 flex (with alternatives from the pool)."""
    spells = [c for c in deck.cards if c.card_id is not None and not _CARDS[c.card_id]["is_land"]]
    core = [dict(card_id=c.card_id, name=c.name, count=c.count) for c in spells[:15]]
    in_deck = {c.card_id for c in deck.cards}
    outside = [i for i in sorted(pool) if i not in in_deck and not _CARDS[i]["is_land"]
               and all(ch in deck.colors + deck.splash for ch in _CARDS[i]["colors"])]
    flex = []
    for k, c in enumerate(spells[15:19]):
        alts = [dict(card_id=i, name=_CARDS[i]["name"], freq=round(0.3 - 0.1 * j, 2)) for j, i in enumerate(outside[2 * k:2 * k + 2])]
        flex.append(dict(card_id=c.card_id, name=c.name, count=c.count, core_count=0, freq=round(0.8 - 0.15 * k, 2), alts=alts))
    gap = 100 * (top_wr - deck.game_wr)
    return dict(rank=rank, colors=deck.colors, splash=deck.splash, archetype="" if rank else "Fake Fliers", p_best=p_best, wr=deck.game_wr,
                gap_pp=gap, gap_lo_pp=gap - 1.4 if rank else 0.0, gap_hi_pp=gap + 1.9 if rank else 0.0, p_beats_top=0.0 if not rank else 0.19,
                n_members=40 - 9 * rank, why="best representative under the full-data fit" if rank == 0 else f"best deck in {p_best:.0%} of the model refits",
                diff_to_top=[0, 12, 15][rank], core=core, flex=flex,
                extra=[dict(card_id=i, name=_CARDS[i]["name"], freq=0.25) for i in outside[:2]], n_decks=16)


def score_deck(set_code: str, text: str, fmt: str = "bo3", name_overrides=None, search: bool = True) -> ScoreResult:
    from sealed_lab.deck.names import apply_overrides

    _guard(set_code)
    time.sleep(_delay() * 3)
    if not text.strip():
        raise ValueError("The deck is empty.")
    main, side = _parse(text)
    main, side = apply_overrides(main, name_overrides), apply_overrides(side, name_overrides)
    n = sum(main.values())
    deck = _fake_deck(11, "WU", "", 0.548, names=main, fmt=fmt)
    unmatched = [n_ for n_ in main if n_.casefold() not in _BY_NAME]
    warns = [f"Deck has {n} cards (expected 40)"] if n != 40 else []
    eff = alt = None
    if side and search:
        eff = -1.84
        alt = _fake_deck(5, "WU", "R", 0.566, fmt=fmt)
    pool: dict[int, int] = {}
    for nm, k in {**main, **side}.items():
        cid = _BY_NAME.get(nm.casefold())
        if cid is not None and nm not in _BASICS.values():
            pool[cid] = pool.get(cid, 0) + k
    return ScoreResult(set_code=set_code, fmt=fmt, deck=deck, efficiency_pp=eff, best_alternative=alt,
                       unmatched=unmatched, warnings=warns, pool=pool)


def export_arena(deck: DeckEval, sideboard: dict[str, int] | None = None, set_code: str | None = None) -> str:
    lines = ["Deck"] + [f"{c.count} {c.name}" for c in deck.cards]
    if sideboard:
        lines += ["", "Sideboard"] + [f"{n} {name}" for name, n in sideboard.items()]
    return "\n".join(lines) + "\n"


def sample_pool_text() -> str:
    """A pasteable pool for demos/tests."""
    rows = [f"1 {d['name']}" for i, d in _CARDS.items() if not d["is_land"]][:60]
    return "Deck\n" + "\n".join(rows) + "\n1 Unknown Card Name\n"


def sample_deck_text(with_side: bool = False) -> str:
    rows = [f"1 {_CARDS[i]['name']}" for i in range(0, 60) if _CARDS[i]["colors"] in ("W", "U", "")][:23]
    txt = "Deck\n" + "\n".join(rows) + "\n9 Plains\n8 Island\n"
    if with_side:
        txt += "\nSideboard\n1 Ember Bolt\n1 Grave Ward\n"
    return txt


# --------------------------------------------------------------------------- locks, explanation, swaps, Arena log

def _ids_of(spec) -> dict[int, int]:
    if not spec:
        return {}
    if isinstance(spec, (str, int)):
        spec = [spec]
    items = spec.items() if isinstance(spec, dict) else ((c, 1) for c in spec)
    out: dict[int, int] = {}
    for c, n in items:
        cid = c if isinstance(c, int) else _BY_NAME.get(str(c).casefold())
        if cid is not None:
            out[int(cid)] = out.get(int(cid), 0) + int(n)
    return out


def build_from_pool(set_code: str, pool: dict[int, int], fmt: str = "bo3", n_decks: int = 1, min_diff: int = 10,
                    progress=None, cancel=None, locked=None, excluded=None) -> BuildResult:
    """Fake builder honouring locks / exclusions: locked cards first, then the best remaining spells by toy beta."""
    _guard(set_code)
    lk, ex = _ids_of(locked), set(_ids_of(excluded))
    clash = [i for i in lk if i in ex]
    if clash:
        raise ValueError(f"{_CARDS[clash[0]]['name']!r} is both locked and excluded")
    for i in range(6):
        _check(cancel)
        if progress:
            progress(i / 6, "Solving MILPs" if i < 4 else "Local search")
        time.sleep(_delay())
    spells = {i: min(n, 4) for i, n in pool.items() if i in _CARDS and not _CARDS[i]["is_land"] and i not in ex}
    deck: dict[int, int] = {i: min(n, spells.get(i, 0)) for i, n in lk.items() if i in spells}
    for i in sorted(spells, key=lambda c: -_beta(c)):
        while deck.get(i, 0) < spells[i] and sum(deck.values()) < 23:
            deck[i] = deck.get(i, 0) + 1
    deck[_BASIC_ID["W"]] = 9
    deck[_BASIC_ID["U"]] = 40 - sum(deck.values())
    ev = evaluate_deck(set_code, deck, fmt)
    ev.colors = "WU"
    if progress:
        progress(1.0, "Done")
    warns = [f"{sum(lk.values())} locked card copies in every deck"] if lk else []
    return BuildResult(set_code=set_code, fmt=fmt, decks=[ev], p_best=[1.0], diffs=[0], pool_unmatched=[], warnings=warns,
                       pool={int(k): int(v) for k, v in pool.items() if k in _CARDS and _CARDS[k]["name"] not in _BASICS.values()})


def _toy_logit(deck: dict[int, int]) -> float:
    lands = sum(n for i, n in deck.items() if _CARDS.get(i, {}).get("is_land"))
    return (_INTERCEPT + sum(_beta(i) * n for i, n in deck.items() if not _CARDS.get(i, {}).get("is_land"))
            - 0.08 * abs(lands - 17) + _anchor_offset(deck))


def explain_deck(set_code: str, deck: dict[int, int], fmt: str = "bo3"):
    """Toy waterfall consistent with `evaluate_deck`'s toy logit; steps add up exactly."""
    _guard(set_code)
    import math

    from sealed_lab.deck.explain import DeckExplanation, WaterfallStep

    b_avg, n_avg = 0.02, 23
    start = _INTERCEPT + n_avg * b_avg + _anchor_offset(deck)
    rows = [(i, _CARDS[i]["name"], n, n * (_beta(i) - b_avg)) for i, n in deck.items()
            if i in _CARDS and not _CARDS[i]["is_land"]]
    n_sp = sum(r[2] for r in rows)
    lands = sum(n for i, n in deck.items() if _CARDS.get(i, {}).get("is_land"))
    rows.sort(key=lambda r: -abs(r[3]))
    top = sorted(rows[:8], key=lambda r: -r[3])
    rest = rows[8:]
    raw = [(f"{r[1]}" + (f" ×{r[2]}" if r[2] > 1 else ""), "card", r[3], r[0], []) for r in top]
    if rest:
        raw.append((f"{len(rest)} other cards", "cards", sum(r[3] for r in rest), None, [(r[1], r[3]) for r in rest[:12]]))
    raw.append(("Nonbasic lands", "lands", 0.0, None, []))
    d_slot = (n_sp - n_avg) * b_avg
    d_land = -0.08 * abs(lands - 17)
    raw.append(("Land count", "structure", d_land + d_slot,
                None, [("lands", d_land), ("spell slots (land swapped for an average spell)", d_slot)]))
    for lab in ("Creatures / spells", "Curve", "Colors & splash", "Mana simulation"):
        raw.append((lab, "structure", 0.0, None, []))

    def sig(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    steps = [WaterfallStep("Average build", "start", 0.0, 0.0, sig(start))]
    L = start
    for lab, kind, d, cid, det in raw:
        steps.append(WaterfallStep(lab, kind, d, 100 * (sig(L + d) - sig(L)), sig(L + d), cid, det))
        L += d
    steps.append(WaterfallStep("This deck", "end", 0.0, 0.0, sig(L)))
    return DeckExplanation(fmt=fmt, start_wr=sig(start), end_wr=sig(L), steps=steps, card_steps=rows, n_builds=1234)


def swap_analysis(set_code: str, deck: dict[int, int], pool: dict[int, int], fmt: str = "bo3", locked=None, excluded=None,
                  cancel=None, on_partial=None):
    """Toy swap analysis: gain of replacing a deck spell by a sideboard spell from the toy logit."""
    _guard(set_code)
    import math

    from sealed_lab.deck.explain import Swap, SwapAnalysis

    ex = set(excluded or ())
    lk = dict(locked or {})
    side = {i: n - deck.get(i, 0) for i, n in pool.items() if i in _CARDS and n - deck.get(i, 0) > 0
            and not _CARDS[i]["is_land"] and i not in ex}
    outs = [i for i in deck if i in _CARDS and not _CARDS[i]["is_land"]]

    def sig(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    L0 = _toy_logit(deck)
    res = SwapAnalysis(base_wr=sig(L0))
    pairs: dict[tuple[int, int], Swap] = {}
    for o in outs:
        _check(cancel)
        for i in side:
            if i != o:
                d = dict(deck)
                d[o] -= 1
                if d[o] <= 0:
                    del d[o]
                d[i] = d.get(i, 0) + 1
                g = 100 * (sig(_toy_logit(d)) - sig(L0))
                p = min(1.0, max(0.0, 0.5 + g * 1.2))
                pairs[(o, i)] = Swap(o, i, g, round(p * 16) / 16)
    for (o, i), s in pairs.items():
        if o not in res.deck_best or s.gain_pp > res.deck_best[o].gain_pp:
            res.deck_best[o] = s
        if i not in res.side_best or s.gain_pp > res.side_best[i].gain_pp:
            res.side_best[i] = s
    res.var_pp = {o: -s.gain_pp for o, s in res.deck_best.items()}
    uo: set[int] = set()
    ui: set[int] = set()
    for s in sorted(pairs.values(), key=lambda s: -s.gain_pp):
        if s.gain_pp <= 1e-4 or len(res.suggestions) >= 3:
            break
        if s.out_id in uo or s.in_id in ui or s.out_id in lk:
            continue
        uo.add(s.out_id)
        ui.add(s.in_id)
        res.suggestions.append(s)
    res.n_eval = len(pairs)
    return res


matchups_mode = "ok"       # tests switch it: "ok" | "not_supported"
synergy_mode = "ok"        # "ok" | "off"


def model_components(set_code: str) -> dict:
    return {"synergy": dict(mode="auto", applied=synergy_mode == "ok", reason="fake", gain={}),
            "archcond": dict(mode="auto", applied=False, reason="fake", gain={}),
            "matchups": dict(mode="pooled", applied=matchups_mode == "ok", reason="fake", gain={}, applied_formats=["bo3"], source={"bo3": "pooled"}),
            "manasim": dict(mode="auto", applied=False, reason="fake", gain={}, version=1),
            "extra_draft": dict(mode="auto", available=False, applied=False, reason="fake", gain={})}


def synergies(set_code: str, deck: dict[int, int], pool: dict[int, int] | None = None, fmt: str = "bo3"):
    """Toy synergy view: some deterministic pairs of deck spells are 'reliable'; sideboard cards complete some of them."""
    _guard(set_code)
    from sealed_lab.deck.insight import SynergyView, SynRow

    if synergy_mode != "ok":
        return SynergyView(status="off", message="Pair synergies are not used: the hold-out could not show a gain (fake).")
    out = SynergyView(status="ok", n_reliable=7)
    spells = sorted(i for i in deck if i in _CARDS and not _CARDS[i]["is_land"])
    side = sorted(i for i, n in (pool or {}).items() if i in _CARDS and not _CARDS[i]["is_land"] and n - deck.get(i, 0) > 0)

    def row(a: int, b: int, scale: float) -> SynRow:
        h = (a * 31 + b * 17) % 7
        c = (0.35 + 0.1 * h) * scale
        return SynRow(a=a, b=b, pp=c, coef_pp=c, lo_pp=c - 0.25, hi_pp=c + 0.25, p_pos=0.93 if c > 0 else 0.07, games=1200.0 + 50 * h)
    for x in range(len(spells)):
        for y in range(x + 1, len(spells)):
            if (spells[x] * 31 + spells[y] * 17) % 11 == 0:
                r = row(spells[x], spells[y], 1.0 if (spells[x] + spells[y]) % 5 else -1.0)
                out.present.append(r)
                out.by_card.setdefault(r.a, []).append(r)
                out.by_card.setdefault(r.b, []).append(r)
    for s_ in side:
        for p_ in spells:
            if (s_ * 31 + p_ * 17) % 13 == 0:
                r = row(p_, s_, 1.0)
                out.candidates.append(r)
                out.by_card.setdefault(s_, []).append(r)
    return out


def archetype_of(set_code: str, deck: dict[int, int], fmt: str = "bo3"):
    _guard(set_code)
    from sealed_lab.deck.insight import ArchetypeMatch

    cols: dict[str, int] = {}
    for i, n in deck.items():
        d = _CARDS.get(i)
        if d and not d["is_land"] and len(d["colors"]) == 1:
            cols[d["colors"]] = cols.get(d["colors"], 0) + n
    top = [c for c, _ in sorted(cols.items(), key=lambda kv: -kv[1])[:2]]
    grp = "".join(c for c in _COLORS if c in top) or "W"
    sig = [i for i, d in _CARDS.items() if not d["is_land"] and d["colors"] in tuple(top)][:3]
    absent = [i for i, d in _CARDS.items() if not d["is_land"] and d["colors"] in tuple(top) and i not in deck][:3]
    miss = [dict(card_id=i, name=_CARDS[i]["name"], rate=0.4 - 0.05 * k, ref=0.15) for k, i in enumerate(absent)]
    return ArchetypeMatch(status="ok", index=0, label=f"{grp}: {_CARDS[sig[0]]['name'] if sig else 'Iron'} / {_CARDS[sig[1]]['name'] if len(sig) > 1 else 'Surge'}",
                          group=grp, similarity=0.83, n_decks=4210,
                          rates={fmt: dict(games=1840, wins=1040, wr=0.565, shrunk=0.561, lo=0.541, hi=0.581)},
                          rate=dict(games=1840, wins=1040, wr=0.565, shrunk=0.561, lo=0.541, hi=0.581), rate_fmt=fmt,
                          missing=miss, runner_up=("UB: Tide Drake / Grave Ward", 0.61))


def deck_archetype(set_code: str, fmt: str, deck: dict[int, int]) -> dict:
    """Plausible synthetic archetype line: the deck's two most common colors; the style third follows the deck's average mana value."""
    from sealed_lab import archetype_link as L
    from sealed_lab import report as R

    _guard(set_code)
    cols: dict[str, int] = {}
    mv_sum = n_sp = 0
    for i, n in deck.items():
        d = _CARDS.get(i)
        if d and not d["is_land"]:
            n_sp += n
            mv_sum += n * float(d.get("mana_value", 3) or 3)
            for c in d["colors"]:
                cols[c] = cols.get(c, 0) + n
    top = [c for c, _ in sorted(cols.items(), key=lambda kv: -kv[1])[:2]]
    key = R.pair_group("".join(c for c in _COLORS if c in top))
    if key is None or len(key) != 2:
        return dict(available=False, reason="The deck has no clear pair of main colors.")
    rep = set_report(set_code, fmt)
    a = next((x for x in rep["archetypes"]["pairs"] if x["key"] == key), None)
    if a is None:
        return dict(available=False, reason="Too few decks in this pair.")
    avg = mv_sum / max(n_sp, 1)
    third = 0 if avg < 2.9 else 2 if avg > 3.3 else 1
    return L.compose(a, key, third if a["spectrum"].get("shown") else None, avg)


def sideboard_plan(set_code: str, deck: dict[int, int], pool: dict[int, int], fmt: str = "bo3", locked=None, excluded=None,
                   cancel=None):
    """Toy plan: the two best-valued sideboard spells replace the two worst-valued main-deck spells against UB and WG."""
    _guard(set_code)
    from sealed_lab.deck.matchups import PlanSwap, SideboardPlan

    if matchups_mode != "ok":
        return SideboardPlan(status="not_supported", fmt=fmt, message="The data do not support opponent-specific plans for this "
                             "set: card x opponent-color interactions did not improve held-out predictions. Nothing is marked.")
    if fmt != "bo3":
        return SideboardPlan(status="off", fmt=fmt, message="Sideboarding only matters in Bo3.")
    _check(cancel)
    outs = sorted((i for i in deck if i in _CARDS and not _CARDS[i]["is_land"]), key=_beta)
    ins = sorted((i for i, n in pool.items() if i in _CARDS and not _CARDS[i]["is_land"] and n - deck.get(i, 0) > 0),
                 key=lambda c: -_beta(c))
    plan = SideboardPlan(status="ok", fmt=fmt, opponents=["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"], encoding="pair",
                         base_wr={o: 0.56 for o in ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]})
    for k, opp in enumerate(("UB", "WG")):
        picked = []
        for j in range(2):
            if j < len(outs) and j < len(ins):
                ps = PlanSwap(opp=opp, out_id=outs[j + k], in_id=ins[j + k] if j + k < len(ins) else ins[j], gain_pp=0.9 - 0.2 * j,
                              p_better=0.94, gain_avg_pp=-0.3, delta_in_pp=1.1 - 0.2 * j, delta_out_pp=-0.2, p_delta_in=0.95,
                              reliable=True)
                picked.append(ps)
                plan.chips.setdefault(ps.in_id, []).append(ps)
        if picked:
            plan.swaps[opp] = picked
    return plan


arena_log_mode = "ok"      # tests switch it: "ok" | "not_found" | "detailed_off" | "no_pool" | "unknown_cards"


def import_arena_log(set_code: str | None = None, log_dir: str | None = None, course_key: tuple | None = None):
    from sealed_lab.deck import arenalog

    time.sleep(_delay())
    if arena_log_mode == "not_found":
        raise arenalog.LogNotFound("Arena log not found in C:/fake/MTGA.")
    if arena_log_mode == "detailed_off":
        raise arenalog.DetailedLogsOff("Arena is not writing detailed logs. Enable Options > Account > 'Detailed Logs "
                                       "(Plugin Support)' in Arena.")
    if arena_log_mode == "no_pool":
        raise arenalog.NoSealedPool("No sealed pool found in the Arena log.")
    if arena_log_mode == "unknown_cards":
        raise arenalog.UnknownCards("Could not translate the 84 card ids to names.")
    return arenalog.ArenaPool(event="Sealed_TST_20261024", timestamp="2026-10-24T11:03:43", text=sample_pool_text(),
                              n_cards=60, set_code=(set_code or "TST").upper(), source="Player.log",
                              unknown_ids=[], warnings=[])


# ---------------------------------------------------------------------------------------------
# replay analysis (fake data with the real schema of model/replaystats.replay_summary)
# ---------------------------------------------------------------------------------------------

def _fake_cmp(real: float, sim: float, n: int, half: float) -> dict:
    return dict(n=n, real=real, lo=real - half, hi=real + half, sim=sim, diff=real - sim, diff_lo=real - sim - half,
                diff_hi=real - sim + half, raw=real, play_share=0.5)


def replay_summary(set_code: str, progress: Callable[[float, str], None] | None = None, cancel=None,
                   n_jobs: int | None = None, refresh: bool = False) -> dict:
    """Plausible fake `replay_summary` (same keys as the real one; the numbers are invented)."""
    time.sleep(_delay())
    if progress:
        progress(1.0, "Replay analysis (fake)")
    events = [("TradSealed", "bo3", "Bo3 TradSealed", 7000), ("Sealed", "bo1", "Bo1 Sealed", 50000)]
    rows, calib, keep, bias = [], {}, {}, {}
    for ev, fmt, _lab, n in events:
        for L, sim_s in ((16, 0.0885), (17, 0.0695), (None, 0.075)):
            real_s = sim_s * (0.6 if fmt == "bo3" else 0.2)
            rows.append(dict(
                event=ev, fmt=fmt, n_lands=L, n_games=n // 4,
                mull=_fake_cmp(0.19 if fmt == "bo3" else 0.097, 0.12, n // 4, 0.01),
                screw=dict(_fake_cmp(real_s, sim_s, n // 4, 0.006), rule=sim_s * 0.85),
                screw_hand=_fake_cmp(real_s, real_s + 0.027, n // 5, 0.006),
                screw_hand_nd=_fake_cmp(real_s, real_s + 0.02, n // 5, 0.006),
                mana4=_fake_cmp(6.5, 7.8, 1200, 0.15), mana4_long=_fake_cmp(6.45, 7.8, 1000, 0.16),
                mana7=_fake_cmp(0.64, 0.655, n // 5, 0.006),
                flood16=dict(_fake_cmp(0.11, 0.138, n // 10, 0.01), unconditional=0.134, sim_model=0.142, mean_lands=6.7,
                             expected_lands=6.8),
                first7=_fake_cmp(2.97, 2.975, n // 4, 0.03),
                first7_tail=dict(_fake_cmp(0.12 if fmt == "bo3" else 0.001, 0.12, n // 4, 0.01), exact=0.1215)))
        calib[ev] = dict(
            n_games=n, n_decks=n // 3,
            screw=dict(n=n, slope=dict(est=1.3, se=0.2, lo=0.9, hi=1.7), intercept=dict(est=0.0, se=0.1, lo=-0.2, hi=0.2),
                       n_clusters=500, sd_sim=0.03),
            sim_sd_within_lands=0.0,
            mana7_slope=dict(n=n // 5, slope=dict(est=0.7, se=0.03, lo=0.64, hi=0.76),
                             intercept=dict(est=3.0, se=0.5, lo=2.0, hi=4.0), sd_sim=1.1))
        keep[ev] = [dict(on_play=p, lands=k, n=500, p_mull=pm, ci=[pm - 0.03, pm + 0.03])
                    for p in (True, False) for k, pm in enumerate([1.0, 0.9, 0.17, 0.06, 0.06, 0.2, 0.75])]
        bias[ev] = [dict(lands_in_hand=k, n=3000, real=0.45 - 0.035 * (k - 2), lo=0.43, hi=0.47, sim=0.455 - 0.0303 * (k - 2),
                         diff=0.0, diff_lo=-0.02, diff_hi=0.02, raw=0.45, play_share=0.5) for k in (2, 3, 4)]
    tempo, preds = {}, {}

    def eff(c, lo, hi):
        return dict(coef=dict(est=c / 25, se=0.01, lo=lo / 25, hi=hi / 25), ame_pp=[c, lo, hi])

    for ev, _f, _l, n in events:
        tempo[ev] = dict(
            n_games=n, win_rate=0.6, mean_mana4=6.5, share_full_curve=0.4,
            by_mana4=[dict(bin=b, n=n // 7, raw=0.45 + 0.04 * i, adj=0.46 + 0.039 * i, adj_lo=0.45 + 0.039 * i, adj_hi=0.47 + 0.039 * i)
                      for i, b in enumerate(["<=3", "4", "5", "6", "7", "8", "9+"])],
            by_hits=[dict(hits=h, n=n // 4, raw=0.38 + 0.09 * h, adj=0.39 + 0.09 * h, adj_lo=0.38 + 0.09 * h, adj_hi=0.40 + 0.09 * h)
                     for h in range(4)],
            effects=dict(mana4_per_point=eff(3.0, 2.6, 3.4), full_curve=eff(8.5, 7.0, 10.0), per_curve_turn=eff(6.8, 5.7, 8.0),
                         mana4_given_lands=eff(2.6, 2.2, 3.0),
                         lands4_ge4_vs_le2=dict(ame_pp=[35.0, 30.0, 40.0], share_le2=0.02),
                         within_build=dict(mana4_per_point=dict(pp=3.4, lo=3.0, hi=3.8), full_curve=dict(pp=9.3, lo=7.5, hi=11.0),
                                           per_curve_turn=dict(pp=8.0, lo=7.0, hi=9.0), n_games=n, n_builds=n // 3)))
        preds[ev] = dict(
            n_games=n, p_full=0.4, mean_mana4=6.5,
            rows=[dict(feature=f, mean=m, full_curve_pp=[a, a - 0.5, a + 0.5], mana4=[b, b - 0.03, b + 0.03])
                  for f, m, a, b in (("n_lands", 16.9, 3.0, 0.1), ("curve_le1", 2.3, 2.5, 0.1), ("curve_2", 6.8, 4.8, 0.1),
                                     ("curve_3", 6.7, 1.7, 0.05), ("n_creatures", 14.7, 3.0, 0.16))],
            by_lands=[dict(n_lands=16, n=n // 5, full_curve=0.43, lo=0.41, hi=0.45, mana4=6.7),
                      dict(n_lands=17, n=n // 2, full_curve=0.40, lo=0.39, hi=0.41, mana4=6.6)])
    names = [(i, d["name"], d["mana_value"]) for i, d in list(_CARDS.items()) if not d.get("is_land")][:10]

    def lst(key_val):
        return [dict(card_id=i, name=nm, mana_value=mv, value=key_val(j), n=300 - 10 * j, open_cast_rate=0.9 - 0.04 * j,
                     wr_cast_shrunk=0.62, wr_deck_nocast_shrunk=0.55, cast_rate=0.8)
                for j, (i, nm, mv) in enumerate(names[:8])]
    hl = dict(min_seen=40, n_cards=len(names), slow_open=lst(lambda j: 7.5 - 0.3 * j), fast_open=lst(lambda j: 1.2 + 0.1 * j),
              low_cast=lst(lambda j: 0.05 + 0.02 * j), cast_gap_high=lst(lambda j: 0.15 - 0.01 * j), cast_gap_low=lst(lambda j: -0.05),
              gih_top=lst(lambda j: 0.70 - 0.01 * j), gih_bottom=lst(lambda j: 0.50 + 0.01 * j))
    return dict(
        available=True, set_code=set_code.upper(), version=1, signature="fake", built_at="2026-10-24T12:00:00",
        events=[dict(event=ev, fmt=f, label=lab, n_games=n, n_turn3=n - 50, n_turn4=n - 200, n_turn7=n // 2, win_rate=0.6,
                     on_play_share=0.5) for ev, f, lab, n in events],
        link=dict(n_replay_games=57000, n_linked=57000, n_with_phi=57000),
        idmap=dict(n_ids=1386, n_mapped=1358, mass_mapped=0.99998),
        simulator=dict(rows=rows, calibration=calib, keep_rules=keep, draw_bias=bias, n_trials=2000, land_bins=[15, 16, 17, 18]),
        tempo=tempo, curve_predictors=preds,
        cards={"bo3": dict(events=["TradSealed"], highlights=hl, n_cards=len(names)),
               "bo1": dict(events=["Sealed"], highlights=hl, n_cards=len(names))},
        notes=["Fake data for the GUI test mode."], timings=dict(ingest=1.0, stats=10.0, total=11.0))


def replay_card_stats(set_code: str, fmt: str = "bo3", n_jobs: int | None = None, refresh: bool = False,
                      progress: Callable[[float, str], None] | None = None, cancel=None) -> dict[int, dict]:
    """Plausible fake per-card replay statistics ({card_id: {...}}, same fields as the real `replay_card_stats`)."""
    time.sleep(_delay())
    out = {}
    for i, d in _CARDS.items():
        if d.get("is_land"):
            continue
        rng = random.Random(zlib.crc32(f"{d['name']}{fmt}".encode()))
        mv = float(d["mana_value"])
        seen = 400 + rng.randint(0, 800)
        castr = max(0.3, 0.97 - 0.06 * mv + rng.uniform(-0.05, 0.05))
        wr = 0.56 + rng.uniform(-0.06, 0.08)
        out[i] = dict(
            name=d["name"], rarity=d.get("rarity", "common"), mana_value=mv, n_deck=seen * 2, wr_deck=wr - 0.01, n_seen=seen,
            wr_seen=wr, wr_seen_shrunk=wr, n_unseen=seen, wr_unseen=wr - 0.04, n_cast=int(seen * castr), cast_rate=castr,
            wr_cast=wr + 0.04, wr_cast_shrunk=wr + 0.04, wr_deck_nocast=wr - 0.03, n_deck_nocast=seen,
            wr_deck_nocast_shrunk=wr - 0.03, wr_seen_nocast=wr - 0.02, n_seen_nocast=int(seen * (1 - castr)), cast_gap=0.07,
            mean_cast_turn=max(1.0, mv + 0.8 + rng.uniform(-0.2, 0.4)), mean_seen_turn=3.0, n_open=seen // 4,
            open_cast_rate=castr, open_cast_turn=mv + 1.2, n_early_long=seen // 3, early_cast_rate=castr, base_wr=0.58)
    return out


# --------------------------------------------------------------------------- new 17lands data check / download, set detection

datasets_items: list = []          # DatasetItem list returned by check_new_datasets (tests and demos fill it)
datasets_error: str = ""           # non-empty: the check reports being offline
datasets_dismissed: list = []      # urls passed to dismiss_datasets
fake_data_dir: str = str(Path(tempfile.gettempdir()) / "sealed_lab_fake_17lands")
datasets_fail: set = set()         # file names whose download fails
FAKE_NEW_SET = ""                  # tests set a set code (e.g. "NEWX"): the check also returns a brand-new set (Premier Draft + Sealed + draft data)


def datasets_data_dir() -> str:
    return fake_data_dir


def make_dataset_item(set_code: str = "TST", kind: str = "game", event: str = "TradSealed", status: str = "new",
                      size: int = 5_000_000, **kw):
    """A DatasetItem for the fake check (url/name derived from the arguments)."""
    from sealed_lab.ingest.datasets import DatasetItem

    name = f"{kind}_data_public.{set_code}.{event}.csv.gz"
    d = dict(url=f"https://data.17landscapes.com/analysis_data/{kind}_data/{name}", name=name, kind=kind,
             set_code=set_code, event=event, status=status, size=size, last_modified="2026-10-20T04:12:10+00:00",
             etag=f"etag-{name}", optional=kind == "draft", set_in_app=set_code in _files, set_has_model=set_code in _models,
             reason="not in your data folder" if status == "new" else "size differs")
    d.update(kw)
    return DatasetItem(**d)


def check_new_datasets(data_dir: str | None = None, force: bool = False):
    from sealed_lab.ingest.datasets import DatasetCheck

    time.sleep(_delay())
    if datasets_error:
        return DatasetCheck(data_dir=str(data_dir or fake_data_dir), error=datasets_error)
    with _lock:
        its = [i for i in datasets_items if force or i.url not in datasets_dismissed]
        if FAKE_NEW_SET:
            its += [make_dataset_item(FAKE_NEW_SET, "game", "PremierDraft", "new", 60_000_000),
                    make_dataset_item(FAKE_NEW_SET, "game", "Sealed", "new", 25_000_000),
                    make_dataset_item(FAKE_NEW_SET, "draft", "PremierDraft", "new", 90_000_000)]
    return DatasetCheck(items=its, data_dir=str(data_dir or fake_data_dir), n_remote=300, n_head=len(its),
                        checked_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def download_datasets(items, data_dir: str | None = None, progress: Callable[[float, str], None] | None = None, cancel=None):
    """Writes tiny stand-in files (no network), registers them like the real one and drops them from the pending list."""
    from sealed_lab.ingest.datasets import DatasetItem, DownloadReport

    its = [DatasetItem(**i) if isinstance(i, dict) else i for i in items]
    rep = DownloadReport()
    root = Path(data_dir or fake_data_dir)
    todo = [i for i in its if i.needs_download]
    for k, it in enumerate(todo):
        if cancel is not None and cancel.is_set():
            rep.cancelled = True
            break
        for step in range(5):
            if cancel is not None and cancel.is_set():
                rep.cancelled = True
                break
            if progress:
                progress((k + step / 5) / max(len(todo), 1), f"Downloading {it.name} ({k + 1}/{len(todo)})")
            time.sleep(_delay())
        else:
            if it.name in datasets_fail:
                rep.failed.append((it.name, "OSError: simulated failure"))
                continue
            dest = root / it.set_code / it.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"stub")
            rep.downloaded.append(str(dest))
            rep.nbytes += 4
            continue
        break
    rep.to_register = list(rep.downloaded) + [i.local_path for i in its if i.status == "unregistered" and i.local_path]
    if rep.to_register:
        rep.registered = add_files(rep.to_register)
        rep.stale_sets = sorted({f.set_code for f in rep.registered if f.kind == "game" and f.set_code
                                 and _models.get(f.set_code, {}).get("stale")})
    done = {i.url for i in its if i.name in {Path(p).name for p in rep.downloaded}} | \
           {i.url for i in its if i.status == "unregistered"}
    with _lock:
        datasets_items[:] = [i for i in datasets_items if i.url not in done]
    if progress:
        progress(1.0, "Download finished")
    return rep


def dismiss_datasets(items) -> None:
    with _lock:
        datasets_dismissed.extend(i.url if hasattr(i, "url") else i["url"] for i in items)


def detect_set(text: str) -> dict:
    """Same logic as the real one, over the fake card universe for every set that has a model."""
    from sealed_lab.deck import setdetect

    time.sleep(_delay())
    idx = _name_index()
    with _lock:
        sets = sorted(_models)
    return setdetect.detect(text, {s: idx for s in sets}).to_dict()


# --------------------------------------------------------------------------- guessed names, tracked folder, set overview

_index_lock = threading.Lock()
_fake_index = None


def _name_index():
    """The fake card universe as a NameIndex, built once (polars frames are not built from several threads at the same time)."""
    global _fake_index
    with _index_lock:
        if _fake_index is None:
            import polars as pl

            from sealed_lab.deck.names import NameIndex

            ids = sorted(_CARDS)
            _fake_index = NameIndex(pl.DataFrame({"card_id": ids, "name": [_CARDS[i]["name"] for i in ids],
                                                  "is_basic": [_CARDS[i]["name"] in _BASICS.values() for i in ids]}))
        return _fake_index


def check_pool(set_code: str, text: str) -> dict:
    """Same logic as the real one (deck.names.analyse_pool) over the fake card universe."""
    from sealed_lab.deck.names import analyse_pool

    time.sleep(_delay())
    return analyse_pool(text, _name_index())


def set_folders() -> dict[str, str]:
    return dict(_folders)


def set_set_folder(set_code: str, path: str | None) -> None:
    with _lock:
        if path:
            _folders[set_code.upper()] = str(path)
        else:
            _folders.pop(set_code.upper(), None)
        _scan_seen.clear()


SCAN_EVENTS = ("Sealed", "TradSealed", "PremierDraft", "TradDraft")      # the real scan registers these events ...
SCAN_GAME_EVENTS = SCAN_EVENTS + ("PickTwoDraft", "PickTwoTradDraft", "QuickDraft")      # ... and the game files of these


def _scanned(m) -> bool:
    return m.group(3) in SCAN_EVENTS or (m.group(1).lower() == "game" and m.group(3) in SCAN_GAME_EVENTS)


def _dataset_files(folder: Path, all_events: bool = False) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and (m := _NAME_RE.search(p.name)) and (all_events or _scanned(m)))


def scan_data_folders(data_dir: str | None = None, track: list[str] | None = None, skip_sets: list[str] | None = None,
                      progress=None, cancel=None):
    """Registers the dataset files found in the data folder's sub-folders of the known sets (and `track`), like the real scan."""
    from sealed_lab.ingest.folders import ScanReport

    root = Path(data_dir or fake_data_dir)
    scan_calls.append(dict(data_dir=str(root), track=list(track or []), skip_sets=list(skip_sets or [])))
    time.sleep(_delay())
    rep = ScanReport(data_dir=str(root))
    skip = {x.upper() for x in (skip_sets or [])}
    with _lock:
        known = set(_files) | {t.upper() for t in (track or [])}
    changed: set[str] = set()
    cands: dict[str, Path] = {}
    for sc in sorted(known):
        if sc in skip:
            rep.deferred.append(sc)
            continue
        folder = Path(_folders[sc]) if sc in _folders else root / sc
        for p in _dataset_files(folder):
            cands.setdefault(str(p), p)
    if root.is_dir():
        for p in sorted(root.glob("*")):
            if p.is_file() and (m := _NAME_RE.search(p.name)) and _scanned(m):
                cands.setdefault(str(p), p)
    for key, p in cands.items():
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        st = p.stat()
        sig = (st.st_size, st.st_mtime_ns)
        if _scan_seen.get(key) == sig:
            rep.skipped += 1
            continue
        fi = _detect(p)
        code = (fi.set_code or "").upper()
        if fi.kind == "unknown" or code not in known or code in skip:
            continue
        _scan_seen[key] = sig
        rep.scanned += 1
        with _lock:
            lst = _files.setdefault(code, [])
            if any(x.path == fi.path and x.fingerprint == fi.fingerprint for x in lst):
                continue
            lst[:] = [x for x in lst if not (x.kind == fi.kind and x.event == fi.event)] + [fi]
            if code in _models:
                _note_new_file(code, fi)
        rep.added.append(fi)
        changed.add(code)
    with _lock:
        for sc, lst in _files.items():       # registered files that were in the scanned folders and are gone
            for f in lst:
                pth = Path(f.path)
                try:
                    inside = root in pth.parents
                except ValueError:
                    inside = False
                if inside and not pth.exists() and f.path not in _missing.get(sc, []):
                    _missing.setdefault(sc, []).append(f.path)
        rep.missing = {sc: list(v) for sc, v in _missing.items() if v}
    rep.changed_sets = sorted(changed)
    return rep


def untracked_sets(data_dir: str | None = None) -> list[dict]:
    root = Path(data_dir or fake_data_dir)
    out: dict[str, dict] = {}
    if root.is_dir():
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for p in _dataset_files(sub):
                m = _NAME_RE.search(p.name)
                code = m.group(2).upper()
                if code not in _files:
                    e = out.setdefault(code, dict(set_code=code, folder=str(sub), n_local=0, n_remote=0, events=[]))
                    e["n_local"] += 1
    for r in untracked_remote:
        if r["set_code"] not in _files:
            e = out.setdefault(r["set_code"], dict(set_code=r["set_code"], folder="", n_local=0, n_remote=0, events=[]))
            e["n_remote"] += len(r.get("events", [])) or 1
            e["events"] = list(r.get("events", []))
    return [out[k] for k in sorted(out)]


def dataset_files_for_set(set_code: str, data_dir: str | None = None):
    sc = set_code.upper()
    for r in untracked_remote:
        if r["set_code"] == sc:
            return [make_dataset_item(sc, "game", ev, "new") for ev in r.get("events", ["TradSealed"])]
    return []


def set_overview(set_code: str) -> dict:
    """Plausible fake overview (same keys as the real `api.set_overview`)."""
    sc = set_code.upper()
    m = _models.get(sc)
    files = [dict(name=Path(f.path).name, path=f.path, kind=f.kind, event=f.event or "", fmt=f.fmt or "", size=f.size, note=f.note,
                  date="2026-10-20", missing=f.path in _missing.get(sc, [])) for f in sorted(_files.get(sc, []), key=lambda f: ({"game": 0, "replay": 1, "draft": 2}.get(f.kind, 9), f.event or ""))]
    out = dict(set_code=sc, n_cards=len(_CARDS), games=dict(m["n_games"]) if m else {}, events={"draft": 83_410, "bo1": 8_020, "bo3": 3_390}
               if m else {}, date_range={"draft": ["2026-09-01", "2026-10-18"], "bo1": ["2026-09-01", "2026-10-18"],
                                         "bo3": ["2026-09-03", "2026-10-17"]} if m else {}, replay_games=57_000 if m else None,
               files=files, model=None)
    if not m:
        return out
    names = [d["name"] for d in _CARDS.values() if not d["is_land"]]
    val = {}
    for fmt, g0, lo0, hi0 in (("bo3", 0.0048, 0.0010, 0.0086), ("bo1", 0.0027, 0.0017, 0.0038)):
        if fmt in m["formats"]:
            val[fmt] = dict(status="ok", reason=None, gain_vs_B0=g0, lo_vs_B0=lo0, hi_vs_B0=hi0, gain_vs_B1=g0 * 0.7, ci_vs_B1=[lo0 * 0.4, hi0 * 0.9],
                            logloss=0.597, auc=0.69, calibration_slope=1.08 if fmt == "bo3" else 0.99, level_pp=1.9, level_ci=[0.9, 2.9],
                            n_holdout_games=1782, holdout_start="2026-10-05 12:00:00")
    comps = [dict(key=k, label=lab, applied=ap, mode="auto", reason=why) for k, lab, ap, why in (
        ("extra_draft", "Extra draft events (PickTwo / QuickDraft)", False, "not better than the standard draft events alone on the hold-out"),
        ("recency", "Recency weighting", False, "no half-life beat equal weights"),
        ("attributes", "Card-attribute prior", False, "worse on the hold-out in bo1"),
        ("crossset", "Cross-set prior", False, "below the materiality floor"),
        ("synergy", "Card-pair synergies", False, "the hold-out cannot tell"),
        ("archcond", "Archetype-conditional values", False, "the hold-out cannot tell"),
        ("matchups", "Opponent matchups", matchups_mode == "ok", "improves held-out log loss with a 95% CI above zero"),
        ("manasim", "Mana simulator v2", False, "no held-out gain"),
        ("castability", "Castability", True, "shipped as a safety constraint"))]
    for c_ in comps:
        if c_["key"] == "matchups":
            c_.update(mode="pooled", applied_formats=["bo3"] if c_["applied"] else [], source={"bo3": "pooled"} if c_["applied"] else {})
    out["model"] = dict(
        built_at=m["built_at"], seconds=468.6, formats=list(m["formats"]), n_boot=16, code_version="0.1.0", components=comps, validation=val,
        deployed=dict(components=["matchups", "castability"], fallback=False, reason="the deployed combination is not materially worse than the plain chain",
                      formats={f: dict(status="ok", gain=0.0034, lo=0.0004, hi=0.0070) for f in m["formats"]}),
        castability=dict(applied=True, mode="auto", reason="shipped as a safety constraint against extrapolation", rule="ships unless a format is materially worse"),
        lam={"draft": dict(lam=None, lam_se=None, source=None, sigma=0.082), "bo1": dict(lam=0.796, lam_se=0.047, source="draft", sigma=0.008),
             "bo3": dict(lam=None, lam_se=None, source="bo1", sigma=0.008)},
        top_cards={f: [dict(name=n, pp=0.95 - 0.2 * k, beta=0.04 - 0.008 * k, exposure=2400 + 100 * k) for k, n in enumerate(names[:5])]
                   for f in m["formats"]}, degraded=[], has_report=True)
    return out


def folder_set_codes(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in _dataset_files(Path(path), all_events=True):
        m = _NAME_RE.search(p.name)
        out[m.group(2).upper()] = out.get(m.group(2).upper(), 0) + 1
    return out


# --------------------------------------------------------------------------- deck search (fake)

_DS_EVENTS = [
    # event, label, fmt, bo3, max_wins, runs
    ("PremierDraft", "Premier Draft (Bo1)", "draft", False, 7, 40),
    ("TradDraft", "Traditional Draft (Bo3)", "draft", True, 3, 14),
    ("Sealed", "Sealed (Bo1)", "bo1", False, 7, 18),
    ("TradSealed", "Traditional Sealed (Bo3)", "bo3", True, 3, 8),
]


def deck_search_cards(set_code: str) -> list[str]:
    _guard(set_code)
    time.sleep(_delay())
    return sorted(d["name"] for d in _CARDS.values() if d["name"] not in _BASICS.values())


def deck_search_events(set_code: str) -> list[dict]:
    _guard(set_code)
    time.sleep(_delay())
    return [dict(event=e, label=lab, fmt=f, bo3=b, max_wins=mw, n_runs=n, n_games=n * (5 if b else 9))
            for e, lab, f, b, mw, n in _DS_EVENTS]


def _ds_line(cid: int, n: int) -> dict:
    d = _CARDS[cid]
    basic = d["is_land"] and d["name"] in _BASICS.values()
    return dict(card_id=cid, name=d["name"], n=n, is_land=d["is_land"], is_basic=basic, is_creature=d["is_creature"],
                mana_value=d["mana_value"], colors=d["colors"], types="Land" if d["is_land"] else "Creature" if d["is_creature"] else "Instant",
                rarity="common")


def _ds_runs(set_code: str, cid: int | None) -> list[dict]:
    spells = [i for i, d in _CARDS.items() if not d["is_land"] and i != cid]
    out = []
    for e, lab, fmt, bo3, max_wins, n_runs in _DS_EVENTS:
        for k in range(n_runs):
            rng = random.Random(f"{set_code}/{cid}/{e}/{k}")
            wins = rng.randint(0, max_wins)
            losses = rng.randint(0, 3 if max_wins == 7 else 2) if wins < max_wins else rng.randint(0, 2)
            games = max(wins + losses, 1) if not bo3 else max(wins + losses, 1) * 2
            gw = games if rng.random() < 0.7 else max(1, games // 2)
            base_cards = [(cid, 1)] if cid is not None else []
            main = base_cards + [(i, 1) for i in rng.sample(spells, 23 - len(base_cards))]
            main += [(_BASIC_ID[c], 4 + (j % 2)) for j, c in enumerate("UR")] + [(_DUAL_ID, 1)]
            lines = sorted((_ds_line(i, n) for i, n in main), key=lambda x: (x["is_land"], x["mana_value"], x["name"]))
            sb = [_ds_line(i, 1) for i in rng.sample(spells, 6)] if e in ("Sealed", "TradSealed") else None
            nb = 2 if gw < games else 1
            col = rng.choice(["UR", "UR+G", "BG", "WU"])
            skill = round(rng.uniform(0.48, 0.65), 3)
            out.append(dict(
                draft_id=f"{e}-{k:03d}", event=e, event_label=lab, fmt=fmt, bo3=bo3, wins=wins, losses=losses, games=games,
                match_wins=wins if bo3 else 0, match_losses=losses if bo3 else 0, incomplete_matches=0,
                record=f"{wins}\u2013{losses}", win_rate=wins / max(wins + losses, 1), max_wins=max_wins, trophy=wins >= max_wins,
                score=wins * 10 - losses, date=f"2026-{1 + rng.randint(0, 8):02d}-{1 + rng.randint(0, 27):02d} {rng.randint(8, 22):02d}:{rng.randint(0, 59):02d}",
                rank=rng.choice(["Gold", "Platinum", "Diamond", "Mythic", ""]), skill=skill, colors=col,
                n_builds=nb, build_index=0, deck_key=f"{e}-{k}", copies=1 if cid is not None else 0,
                games_with_card=gw if cid is not None else games, wins_with_card=min(wins, gw),
                build_games=gw, build_wins=min(wins, gw), n_cards=sum(n for _i, n in main),
                n_lands=sum(x["n"] for x in lines if x["is_land"]),
                main=lines, sideboard=sb,
                sideboard_note="" if sb is not None else "Draft sideboards are not stored in the corpus (only the main deck).",
                **_fake_model_score(f"{e}-{k}", fmt)))
    return out


def _fake_model_score(key: str, fmt: str) -> dict:
    """A stable made-up model score for a fake Sealed run (draft runs have none), independent of the run generator's random stream."""
    if fmt not in ("bo1", "bo3"):
        return dict(model_q=None, model_score=None)
    import random as _r

    g = _r.Random(key)
    q = round(g.uniform(0.45, 0.62), 4)
    return dict(model_q=q, model_score=round((q - 0.45) / 0.17 * 100, 1))


_RANK_ORDER = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4, "diamond": 5, "mythic": 6}


def _ds_resolve(card) -> int:
    if isinstance(card, int):
        if card in _CARDS:
            return card
        raise ValueError(f"Unknown card: {card}")
    cid = _BY_NAME.get(str(card).strip().casefold())
    if cid is None or (_CARDS[cid]["is_land"] and _CARDS[cid]["name"] in _BASICS.values()):
        raise ValueError(f"Unknown card: {card}")
    return cid


def _ds_filter_days(rows: list[dict], days: int | None) -> tuple[list[dict], str | None]:
    if not days or int(days) <= 0 or not rows:
        return rows, None
    from datetime import datetime, timedelta

    dates = [datetime.strptime(r["date"][:10], "%Y-%m-%d") for r in rows if r.get("date")]
    if not dates:
        return rows, None
    cut = max(dates) - timedelta(days=int(days))
    cut_s = cut.strftime("%Y-%m-%d")
    return [r for r in rows if r.get("date", "")[:10] >= cut_s], cut_s


_FAKE_SCORED: set[str] = set()


def deck_model_scores(set_code: str, progress=None, cancel=None) -> int:
    _guard(set_code)
    if progress is not None:
        progress(0.5, "Scoring real decks with the model")
    _FAKE_SCORED.add(set_code.upper())
    return len(_ds_runs(set_code.upper(), None))


def has_deck_model_scores(set_code: str) -> bool:
    return set_code.upper() in _FAKE_SCORED


def search_decks(set_code: str, card=None, events=None, sort: str = "result", min_games: int = 3, min_card_share: float = 0.5,
                 limit: int = 200, offset: int = 0, *, include_cards=None, exclude_cards=None, in_pool_only=None,
                 colors=None, color_match: str = "exact", splash=None, min_skill=None, min_rank=None, ranks=None,
                 trophy_only: bool = False, min_lands=None, max_lands=None, days: int | None = None,
                 with_analytics: bool = True) -> dict:
    _guard(set_code)
    time.sleep(_delay())
    cid = _ds_resolve(card) if (card is not None and (not isinstance(card, str) or card.strip())) else None
    inc_ids = []
    for item in (include_cards or []):
        c_ref, mn = (item[0], int(item[1])) if isinstance(item, (tuple, list)) else (item, 1)
        ic = _ds_resolve(c_ref)
        inc_ids.append((ic, mn))
        if cid is None:
            cid = ic
    excl_ids = {_ds_resolve(c) for c in (exclude_cards or [])}
    pool_ids = {_ds_resolve(c) for c in (in_pool_only or [])}
    runs = _ds_runs(set_code.upper(), cid)

    def _ok_card(r: dict) -> bool:
        m = {c["card_id"]: c["n"] for c in r["main"]}
        if any(m.get(ic, 0) < mn for ic, mn in inc_ids):
            return False
        if any(m.get(ec, 0) > 0 for ec in excl_ids | pool_ids):
            return False
        if pool_ids:
            sb = {c["card_id"]: c["n"] for c in (r.get("sideboard") or [])}
            if not all(sb.get(pc, 0) > 0 for pc in pool_ids):
                return False
        return True

    runs = [r for r in runs if _ok_card(r)]
    allowed = [r for r in runs if r["games"] >= min_games and (cid is None or r["games_with_card"] / r["games"] >= min_card_share)]
    if events is not None:
        allowed = [r for r in allowed if r["event"] in events]
    if colors:
        cl = colors.strip().lower()
        def _mc(r): return str(r["colors"]).split("+")[0]
        if cl == "mono":
            allowed = [r for r in allowed if len(_mc(r)) == 1]
        elif cl == "2c":
            allowed = [r for r in allowed if len(_mc(r)) == 2]
        elif cl in ("3c+", "3+"):
            allowed = [r for r in allowed if len(_mc(r)) >= 3]
        else:
            want = "".join(c for c in "WUBRG" if c in colors.upper())
            if color_match == "include":
                allowed = [r for r in allowed if all(c in _mc(r) for c in want)]
            else:
                allowed = [r for r in allowed if "".join(c for c in "WUBRG" if c in _mc(r)) == want]
    if splash:
        sp = splash.strip().lower()
        if sp == "none":
            allowed = [r for r in allowed if "+" not in r["colors"]]
        elif sp == "any":
            allowed = [r for r in allowed if "+" in r["colors"]]
        elif len(sp) == 1 and sp.upper() in "WUBRG":
            allowed = [r for r in allowed if "+" in r["colors"] and sp.upper() in r["colors"].split("+", 1)[1]]
    if min_skill is not None and min_skill > 0:
        allowed = [r for r in allowed if (r.get("skill") or 0.0) >= float(min_skill)]
    if min_rank:
        tier = _RANK_ORDER.get(min_rank.strip().lower(), 0)
        allowed = [r for r in allowed if _RANK_ORDER.get(str(r.get("rank") or "").lower(), 0) >= tier]
    if ranks:
        want_r = {str(x).lower() for x in ranks}
        allowed = [r for r in allowed if str(r.get("rank") or "").lower() in want_r]
    if trophy_only:
        allowed = [r for r in allowed if r["trophy"]]
    if min_lands:
        allowed = [r for r in allowed if r["n_lands"] >= int(min_lands)]
    if max_lands:
        allowed = [r for r in allowed if r["n_lands"] <= int(max_lands)]
    allowed, cut_s = _ds_filter_days(allowed, days)
    if sort == "model":
        deck_model_scores(set_code)
    if set_code.upper() not in _FAKE_SCORED:
        allowed = [dict(r, model_q=None, model_score=None) for r in allowed]

    key = {"result": lambda r: (r["trophy"], r["win_rate"], r["wins"]), "win_rate": lambda r: (r["win_rate"], r["games"]),
           "newest": lambda r: r["date"], "oldest": lambda r: r["date"],
           "model": lambda r: (r.get("model_score") is not None, r.get("model_score") or 0.0)}[sort]
    allowed.sort(key=key, reverse=sort != "oldest")
    count = lambda rs: {e: sum(1 for r in rs if r["event"] == e) for e, *_ in _DS_EVENTS}
    page = allowed[offset:offset + limit] if limit else allowed[offset:]

    analytics = None
    if with_analytics:
        tot_g = sum(r["games"] for r in allowed)
        tot_w = sum(r["wins"] for r in allowed)
        avg_wr = tot_w / tot_g if tot_g else None
        by_c: dict[str, list[dict]] = {}
        for r in allowed:
            mc = r["colors"].split("+")[0]
            by_c.setdefault(mc, []).append(r)
        by_color = []
        for mc, rs in sorted(by_c.items(), key=lambda kv: -len(kv[1])):
            gg = sum(x["games"] for x in rs)
            ww = sum(x["wins"] for x in rs)
            wr = ww / gg if gg else None
            by_color.append(dict(colors=mc, runs=len(rs), share=len(rs) / max(len(allowed), 1), games=gg,
                                 raw_wr=wr, avg_skill=0.56))
        comp_counts: dict[int, int] = {}
        for r in allowed:
            for c in r["main"]:
                if not c["is_land"] and c["card_id"] != cid:
                    comp_counts[c["card_id"]] = comp_counts.get(c["card_id"], 0) + 1
        companions = [
            dict(card_id=c, name=_CARDS[c]["name"], pct_in_matches=n / max(len(allowed), 1), base_pct=0.25,
                 lift=(n / max(len(allowed), 1)) / 0.25, wr=avg_wr, n_runs=n)
            for c, n in sorted(comp_counts.items(), key=lambda kv: -kv[1])[:6]
        ] if allowed else []
        play_profile = dict(n_games=tot_g, oh_games=max(1, tot_g // 4), oh_wr=(avg_wr or 0.5) + 0.03, oh_delta=0.035,
                            drawn_games=max(1, tot_g // 3), drawn_wr=(avg_wr or 0.5) + 0.015, drawn_delta=0.02,
                            unseen_games=max(1, tot_g // 3), unseen_wr=(avg_wr or 0.5) - 0.005,
                            replay=dict(n_seen=max(10, tot_g // 2), n_cast=max(8, tot_g // 3), cast_rate=0.78,
                                        mean_cast_turn=3.4, open_cast_rate=0.84, open_cast_turn=2.9,
                                        n_cand1=max(25, tot_g // 3), cand1_keep_rate=0.92, base_keep_rate=0.90, keep_delta=0.02,
                                        n_cand2=4, cand2_keep_rate=None,
                                        turn_hist=[dict(turn="2", count=10, share=0.25), dict(turn="3", count=18, share=0.45),
                                                   dict(turn="4", count=12, share=0.30)])) if (cid is not None and allowed) else None
        analytics = dict(
            n_runs=len(allowed), n_games=tot_g, avg_wr=avg_wr,
            trophy_rate=sum(1 for r in allowed if r["trophy"]) / len(allowed) if allowed else None,
            by_color=by_color,
            main_vs_splash=(dict(main=dict(runs=max(1, len(allowed) - 2), games=max(1, tot_g - 6), wr=avg_wr),
                                 splash=dict(runs=min(2, len(allowed)), games=min(6, tot_g), wr=(avg_wr or 0.5) - 0.02))
                            if cid is not None and allowed else None),
            by_copies=[dict(copies="1", runs=len(allowed), games=tot_g, wr=avg_wr)] if (cid is not None and allowed) else [],
            companions=companions,
            inclusion_gap=dict(all_rate=0.72, all_n=25, top_rate=0.86, top_n=12) if (cid is not None and allowed) else None,
            play_profile=play_profile,
        )

    return dict(set_code=set_code.upper(), card=_CARDS[cid]["name"] if cid is not None else None, card_id=cid,
                total=len(allowed), total_all=len(runs), shown=len(page), offset=offset,
                by_event=count(allowed), by_event_all=count(runs), sort=sort, min_games=min_games,
                min_card_share=min_card_share, days=days, date_cutoff=cut_s, analytics=analytics, results=page)


def search_similar_pools(set_code: str, pool_counts: dict, events=None, min_skill=None, trophy_only: bool = False,
                         limit: int = 50, min_games: int = 3, days: int | None = None) -> dict:
    _guard(set_code)
    time.sleep(_delay())
    if not pool_counts:
        return dict(set_code=set_code.upper(), total=0, shown=0, n_pool_cards=0, days=days, date_cutoff=None, color_breakdown=[], results=[])
    runs = [r for r in _ds_runs(set_code.upper(), None) if r["event"] in ("Sealed", "TradSealed") and r["games"] >= min_games]
    if events:
        runs = [r for r in runs if r["event"] in events]
    if min_skill:
        runs = [r for r in runs if (r.get("skill") or 0.0) >= float(min_skill)]
    if trophy_only:
        runs = [r for r in runs if r["trophy"]]
    runs, cut_s = _ds_filter_days(runs, days)
    for i, r in enumerate(runs):
        r["similarity"] = max(0.25, round(0.82 - 0.015 * i, 3))
        r["shared_count"] = max(5, 18 - i // 2)
        r["shared_cards"] = [c["name"] for c in r["main"][:5] if not c["is_land"]]
    cb = [dict(colors="UR", runs=max(1, len(runs) // 2), share=0.5, games=40, avg_wr=0.58),
          dict(colors="WU", runs=max(1, len(runs) // 3), share=0.33, games=26, avg_wr=0.54)] if runs else []
    page = runs[:limit]
    return dict(set_code=set_code.upper(), total=len(runs), shown=len(page),
                n_pool_cards=sum(int(v) for v in pool_counts.values()), days=days, date_cutoff=cut_s,
                color_breakdown=cb, results=page)


def search_rebuilds(set_code: str, card=None, events=None, min_skill=None, rescue_only: bool = False,
                    color_pivot_only: bool = False, limit: int = 200, offset: int = 0, days: int | None = None) -> dict:
    _guard(set_code)
    time.sleep(_delay())
    cid = _ds_resolve(card) if (card is not None and (not isinstance(card, str) or card.strip())) else None
    runs = [r for r in _ds_runs(set_code.upper(), cid) if r["n_builds"] > 1]
    if events:
        runs = [r for r in runs if r["event"] in events]
    if min_skill:
        runs = [r for r in runs if (r.get("skill") or 0.0) >= float(min_skill)]
    runs, cut_s = _ds_filter_days(runs, days)
    out = []
    for i, r in enumerate(runs):
        pivot = (i % 2 == 0)
        rescue = (i % 3 == 0)
        if rescue_only and not rescue:
            continue
        if color_pivot_only and not pivot:
            continue
        main = r["main"]
        added = [c for c in main if not c["is_land"]][:3]
        cut = [c for c in main if not c["is_land"]][3:6]
        builds = [
            dict(build_index=0, deck_key=f"{r['deck_key']}-0", games=2, wins=0 if rescue else 1, losses=2 if rescue else 1,
                 colors="BG" if pivot else r["colors"], main=main, added=[], cut=[], cards_changed=0),
            dict(build_index=1, deck_key=f"{r['deck_key']}-1", games=max(2, r["games"] - 2), wins=max(1, r["wins"] - (0 if rescue else 1)),
                 losses=max(0, r["losses"] - (2 if rescue else 1)), colors=r["colors"], main=main, added=added, cut=cut, cards_changed=len(added)),
        ]
        rr = dict(r, colors=(f"BG → {r['colors']}" if pivot else r["colors"]), b0_record="0–2" if rescue else "1–1",
                  after_record=f"{builds[1]['wins']}–{builds[1]['losses']}", color_pivot=pivot, is_rescue=rescue,
                  wr_gain=0.45 if rescue else 0.15, cards_changed=len(added), builds=builds)
        out.append(rr)
    page = out[offset:offset + limit] if limit else out[offset:]
    analytics = dict(n_multi=len(runs), pct_color_pivot=0.5, n_rescues=max(1, len(runs) // 3) if runs else 0,
                     avg_before_wr=0.35 if runs else None, avg_after_wr=0.61 if runs else None)
    return dict(set_code=set_code.upper(), card=_CARDS[cid]["name"] if cid is not None else None, card_id=cid,
                total=len(out), shown=len(page), offset=offset, days=days, date_cutoff=cut_s,
                analytics=analytics, results=page)


def search_sideboarding(set_code: str, card=None, events=None, my_colors=None, opp_colors=None,
                        on_play=None, min_skill=None, limit: int = 100, days: int | None = None) -> dict:
    _guard(set_code)
    time.sleep(_delay())
    cid = _ds_resolve(card) if (card is not None and (not isinstance(card, str) or card.strip())) else None
    runs = [r for r in _ds_runs(set_code.upper(), cid) if r["bo3"]]
    if events:
        runs = [r for r in runs if r["event"] in events]
    if min_skill:
        runs = [r for r in runs if (r.get("skill") or 0.0) >= float(min_skill)]
    runs, cut_s = _ds_filter_days(runs, days)
    matches = []
    for i, r in enumerate(runs[:limit]):
        oc = ["WU", "BR", "BG", "RG", "UR"][i % 5]
        if opp_colors and oc != opp_colors.upper():
            continue
        if my_colors and r["colors"].split("+")[0] != my_colors.upper():
            continue
        g2p = (i % 2 == 0)
        if on_play is not None and g2p != bool(on_play):
            continue
        spells = [c for c in r["main"] if not c["is_land"]]
        in_c = spells[:2]
        out_c = spells[2:4]
        matches.append(dict(
            draft_id=r["draft_id"], event=r["event"], event_label=r["event_label"], fmt=r["fmt"], bo3=True,
            match_number=1, record="W (2–1)" if r["wins"] >= r["losses"] else "L (1–2)",
            match_won=r["wins"] >= r["losses"], g1_won=(i % 2 == 1), g2_on_play=g2p,
            colors=r["colors"], opp_colors=oc, date=r["date"], rank=r["rank"], skill=r["skill"],
            in_cards=in_c, out_cards=out_c, n_swapped=len(in_c), n_cards=r["n_cards"], n_lands=r["n_lands"],
            main=r["main"], sideboard=r["sideboard"], sideboard_note=r["sideboard_note"],
        ))
    spells_all = [i for i, d in _CARDS.items() if not d["is_land"]]
    top_in = [dict(card_id=c, name=_CARDS[c]["name"], count=12 - k, rate=0.35 - 0.03 * k, post_wr=0.58) for k, c in enumerate(spells_all[:6])] if matches else []
    top_out = [dict(card_id=c, name=_CARDS[c]["name"], count=10 - k, rate=0.30 - 0.02 * k, post_wr=0.52) for k, c in enumerate(spells_all[6:12])] if matches else []
    cs = dict(n_in=8, n_out=4, in_rate=0.36, out_rate=0.18, top_in_rate=0.45, top_out_rate=0.15,
              in_post_wr=0.61, out_post_wr=0.50, in_on_play=3, in_on_draw=5) if (cid is not None and matches) else None
    by_opp = [dict(opp_colors="BR", n_in=4, n_out=1, net=3), dict(opp_colors="WU", n_in=2, n_out=2, net=0)] if (cid is not None and matches) else []
    return dict(
        set_code=set_code.upper(), card=_CARDS[cid]["name"] if cid is not None else None, card_id=cid,
        total_postboard=len(runs) * 2, boarded_matches=len(runs), boarded_rate=0.5 if runs else 0.0,
        days=days, date_cutoff=cut_s,
        card_summary=cs, top_in=top_in if cid is None else [], top_out=top_out if cid is None else [],
        by_opp_colors=by_opp,
        common_cut_for=[dict(card_id=spells_all[3], name=_CARDS[spells_all[3]]["name"], count=4, share=0.5)] if (cid is not None and matches) else [],
        common_in_for=[dict(card_id=spells_all[4], name=_CARDS[spells_all[4]]["name"], count=3, share=0.75)] if (cid is not None and matches) else [],
        matches=matches,
    )


def deck_search_text(set_code: str, result: dict) -> str:
    lines = ["Deck"] + [f"{c['n']} {c['name']}" for c in result.get("main") or []]
    if result.get("sideboard"):
        lines += ["", "Sideboard"] + [f"{c['n']} {c['name']}" for c in result["sideboard"]]
    return "\n".join(lines) + "\n"


def sample_trainer_pool(set_code: str, fmt: str = "bo3", min_skill: float = 0.62,
                        seed: int | None = None, exclude_ids: list[str] | None = None) -> dict:
    """Synthetic 84-card Sealed pool and top-pilot 40-card build for GUI tests and fake backend."""
    _guard(set_code)
    time.sleep(_delay())
    sc = set_code.upper()
    ex = set(exclude_ids or ())
    idx_num = next((k for k in range(1, 50) if f"trainer-{k:03d}" not in ex), 1)
    rng = random.Random(seed if seed is not None else (1000 + idx_num))
    non_basics = [cid for cid, d in _CARDS.items() if d["name"] not in _BASICS.values()]
    pool_counts: dict[int, int] = {}
    for cid in rng.choices(non_basics, k=84):
        pool_counts[cid] = pool_counts.get(cid, 0) + 1
    # Pilot plays top 23 WU/colorless cards from the pool + 9 Plains + 8 Island
    wu_cards = [cid for cid in pool_counts if set(_CARDS[cid]["colors"]) <= {"W", "U"} and not _CARDS[cid]["is_land"]]
    other_spells = [cid for cid in pool_counts if cid not in wu_cards and not _CARDS[cid]["is_land"]]
    wu_cards.sort(key=lambda c: -_beta(c))
    pilot_deck: dict[int, int] = {}
    rem = 23
    for cid in wu_cards + other_spells:
        if rem <= 0:
            break
        take = min(pool_counts[cid], rem)
        pilot_deck[cid] = take
        rem -= take
    plains_id = next(cid for cid, d in _CARDS.items() if d["name"] == "Plains")
    island_id = next(cid for cid, d in _CARDS.items() if d["name"] == "Island")
    pilot_deck[plains_id] = 9
    pilot_deck[island_id] = 8
    pool_lines = [f"{n} {_CARDS[c]['name']}" for c, n in sorted(pool_counts.items(), key=lambda kv: _CARDS[kv[0]]["name"])]
    rec = "3–0" if fmt == "bo3" else "7–1"
    pilot = dict(
        draft_id=f"trainer-{idx_num:03d}",
        event="TradSealed" if fmt == "bo3" else "Sealed",
        event_label="Traditional Sealed (Bo3)" if fmt == "bo3" else "Sealed (Bo1)",
        deck=pilot_deck,
        colors="WU",
        main_colors="WU",
        splash="",
        record=rec,
        wins=6 if fmt == "bo3" else 7,
        losses=1,
        games=7 if fmt == "bo3" else 8,
        trophy=True,
        skill=0.65,
        rank="Mythic",
        date="2026-07-18 19:30",
    )
    return dict(
        set_code=sc,
        fmt=fmt,
        draft_id=pilot["draft_id"],
        n_pool_cards=sum(pool_counts.values()),
        pool=pool_counts,
        pool_text="\n".join(pool_lines) + "\n",
        pilot=pilot,
    )


def grade_trainer_build(set_code: str, fmt: str, pool: dict[int, int], user_deck: dict[int, int],
                        pilot: dict, max_families: int = 5, progress=None, cancel=None) -> dict:
    import sys
    from sealed_lab import trainer

    _guard(set_code)
    time.sleep(_delay())
    return trainer.grade_trainer_build(sys.modules[__name__], set_code.upper(), fmt=fmt, pool=pool, user_deck=user_deck,
                                       pilot=pilot, max_families=max_families, progress=progress, cancel=cancel)
