"""Synthetic 17lands-format data generator (the test bed; no network needed).

`generate(out_dir, set_code="TST", seed=0, scale=1.0)` writes, in the exact 17lands CSV layout:

* ``game_data_public.{SET}.{PremierDraft,TradDraft,Sealed,TradSealed}.csv.gz``
* ``draft_data_public.{SET}.PremierDraft.csv.gz`` (100 rows, detection tests only)
* ``cards.csv`` (17lands card list) and ``scryfall_{SET}.json`` (Scryfall-like objects)
* ``truth.json`` (planted truth, evaluable with :func:`truth_strength`)
* ``heldout_pools.json`` (fresh Sealed pools + the synthetic player's deck, for builder tests)

Planted truth (logit of winning a game)::

    logit = intercept + drift*day + 5*(skill - opp_skill) + 0.08*on_play
            - 0.30*min(mull,2) + 0.25*min(opp_mull,2)
            - 0.5 * strength(opp deck)  +  strength(deck)

    strength(deck) = sum_c beta_c*copies_c  - 0.08*(n_lands-17)^2  - gamma*screw_index(deck) - center

``screw_index`` is the mean (over spells) probability that, among the first 12 cards seen, a spell's
coloured requirements cannot be met (hypergeometric, sources = basics + duals, mana rock counts 0.5).
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import hypergeom

COLORS = "WUBRG"
BASIC_NAMES = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
EVENTS = ["PremierDraft", "TradDraft", "Sealed", "TradSealed"]
BASE_GAMES = {"PremierDraft": 60000, "TradDraft": 8000, "Sealed": 10000, "TradSealed": 5000, "PickTwoDraft": 20000, "QuickDraft": 15000}
EVENT_CFG = {
    "PremierDraft": dict(pool="draft", bo3=False, p_swap=0.010),
    "TradDraft": dict(pool="draft", bo3=True, p_swap=0.015),
    "Sealed": dict(pool="sealed", bo3=False, p_swap=0.020),
    "TradSealed": dict(pool="sealed", bo3=True, p_swap=0.030),
    # extra draft events (not in EVENTS: asked for with generate(events=...)); same world as the standard draft
    "PickTwoDraft": dict(pool="draft", bo3=False, p_swap=0.010),
    "QuickDraft": dict(pool="draft", bo3=False, p_swap=0.010),
}

# planted constants
INTERCEPT = 0.10
ON_PLAY = 0.08
SKILL_SLOPE = 5.0
MULL = -0.30
OPP_MULL = 0.25
OPP_STRENGTH = -0.5
DRIFT_PER_DAY = 0.0015
LAND_COEF = -0.08
LAND_PEAK = 17
GAMMA = 6.0
SCREW_DRAWS = 12
WINDOW_DAYS = 30.0
T0 = np.datetime64("2024-03-01T12:00:00")

RARITIES = ["common", "uncommon", "rare", "mythic", "basic"]

_ADJ = {
    "W": ["Gilded", "Radiant", "Sunlit", "Dawnbound", "Stalwart", "Hallowed", "Argent", "Steadfast", "Luminous", "Pale", "Valiant", "Ivory"],
    "U": ["Tidal", "Drifting", "Mirrored", "Sapphire", "Veiled", "Brine-Soaked", "Frostbound", "Cerulean", "Misty", "Lucid", "Sunken", "Arcane"],
    "B": ["Grim", "Ashen", "Crypt-Born", "Hollow", "Withered", "Gloaming", "Rotting", "Umbral", "Bleak", "Tomb-Cursed", "Sallow", "Dire"],
    "R": ["Blazing", "Furious", "Cinder", "Raging", "Scorched", "Molten", "Reckless", "Ember", "Volcanic", "Wild-Eyed", "Searing", "Crimson"],
    "G": ["Verdant", "Thorned", "Mossy", "Ancient", "Feral", "Primal", "Wildwood", "Gnarled", "Towering", "Rooted", "Bramble", "Lush"],
}
_CREATURE = {
    "W": ["Paladin", "Sentinel", "Lancer", "Cleric", "Herald", "Griffin", "Warden", "Seraph", "Pikeman", "Priest", "Vanguard", "Angel"],
    "U": ["Merfolk", "Drake", "Sage", "Kraken", "Illusionist", "Siren", "Scholar", "Leviathan", "Wisp", "Tide Mage", "Sphinx", "Seer"],
    "B": ["Ghoul", "Vampire", "Wraith", "Necromancer", "Skeleton", "Cultist", "Zombie", "Horror", "Assassin", "Lich", "Shade", "Revenant"],
    "R": ["Goblin", "Dragon", "Berserker", "Pyromancer", "Ogre", "Raider", "Imp", "Hellion", "Shaman", "Salamander", "Drummer", "Giant"],
    "G": ["Elf", "Wurm", "Treant", "Druid", "Beast", "Spider", "Ranger", "Boar", "Hydra", "Stag", "Mystic", "Behemoth"],
}
_SPELL = {
    "W": ["Smite", "Verdict", "Ward", "Blessing", "Rally", "Edict", "Sanctuary", "Banishment"],
    "U": ["Counterspell Wave", "Deluge", "Memory Leak", "Riptide", "Illusion", "Whirlpool", "Forecast", "Undertow"],
    "B": ["Drain", "Curse", "Plague", "Dark Pact", "Execution", "Blight", "Reaping", "Weakness"],
    "R": ["Bolt", "Eruption", "Fury", "Blast", "Barrage", "Inferno", "Skirmish", "Scorch"],
    "G": ["Growth", "Stampede", "Overgrowth", "Hunt", "Ambush", "Rampage", "Bloom", "Snare"],
}
_ART_NAMES = ["Iron Totem", "Brass Automaton", "Clockwork Hound", "Tin Soldier", "Obsidian Idol", "Bone Lantern"]


# ---------------------------------------------------------------------------------------------
# card universe
# ---------------------------------------------------------------------------------------------

def _cost_str(generic: int, pips: list[int], hybrid: tuple[str, str] | None = None) -> str:
    s = f"{{{generic}}}" if generic > 0 else ""
    if hybrid:
        s += "".join(f"{{{hybrid[0]}/{hybrid[1]}}}" for _ in range(pips[COLORS.index(hybrid[0])]))
        return s
    for i, c in enumerate(COLORS):
        s += f"{{{c}}}" * pips[i]
    return s


def _make_card(name, rarity, kind, pips, generic, subtype="", oracle="", produces="", hybrid=None, faces=None):
    p = list(pips)
    mv = generic + sum(p)
    if hybrid:
        mv = generic + p[COLORS.index(hybrid[0])]
    colors = "".join(c for i, c in enumerate(COLORS) if p[i] > 0)
    if kind == "creature":
        type_line = "Creature — " + (subtype or "Human")
    elif kind == "land":
        type_line = "Land"
    elif kind == "basic":
        type_line = "Basic Land — " + subtype
    else:
        type_line = kind.capitalize()
    return dict(
        name=name, rarity=rarity, kind=kind, pips=p, generic=generic, mv=mv, colors=colors,
        type_line=type_line, oracle=oracle, produces=produces, hybrid=bool(hybrid),
        mana_cost=_cost_str(generic, p, hybrid) if kind not in ("land", "basic") else "",
        is_land=kind in ("land", "basic"), is_basic=kind == "basic", is_creature=kind == "creature", faces=faces,
        hybrid_pair=hybrid,
    )


def _pips_for(rng, color_idx: list[int], mv: int) -> tuple[list[int], int]:
    p = [0] * 5
    if len(color_idx) == 1:
        n = 2 if (mv >= 3 and rng.random() < 0.28) else 1
        n = min(n, mv)
        p[color_idx[0]] = n
    else:
        for ci in color_idx:
            p[ci] = 1
    return p, mv - sum(p)


def _build_cards(seed: int) -> list[dict]:
    """The card universe. Depends only on the seed (not on scale)."""
    rng = np.random.default_rng(10_000 + seed)
    cards: list[dict] = []
    used: set[str] = set()

    def take(pool, ci, kind_suffix=""):
        for _ in range(1000):
            a = _ADJ[COLORS[ci]][int(rng.integers(len(_ADJ[COLORS[ci]])))]
            n = pool[COLORS[ci]][int(rng.integers(len(pool[COLORS[ci]])))]
            nm = f"{a} {n}"
            if nm not in used:
                used.add(nm)
                return nm, n
        raise RuntimeError("name pool exhausted")

    specials = {  # (colour, rarity) -> preset name, kind
        ("W", "rare"): ("Aldric, Dawn's Herald", "creature"),
        ("U", "uncommon"): ("Jötun Tidecaller", "creature"),
        ("B", "common"): ("Nox's Bargain", "sorcery"),
        ("R", "uncommon"): ("Zürn the Smelter", "creature"),
        ("G", "mythic"): ("Ysolde, Voice of the Grove", "creature"),
    }
    used.update(v[0] for v in specials.values())
    counts = {"common": 14, "uncommon": 6, "rare": 4, "mythic": 1}
    cmc_w = {
        "common": [0.04, 0.26, 0.28, 0.20, 0.12, 0.07, 0.03],
        "uncommon": [0.02, 0.18, 0.28, 0.24, 0.16, 0.08, 0.04],
        "rare": [0.0, 0.10, 0.25, 0.25, 0.20, 0.12, 0.08],
        "mythic": [0.0, 0.05, 0.15, 0.25, 0.25, 0.18, 0.12],
    }
    for ci, col in enumerate(COLORS):
        for rar, n in counts.items():
            preset_done = False
            for j in range(n):
                preset = specials.get((col, rar)) if not preset_done else None
                if preset:
                    preset_done = True
                p_creature = {"common": 0.62, "uncommon": 0.55, "rare": 0.72, "mythic": 0.9}[rar]
                kind = preset[1] if preset else ("creature" if rng.random() < p_creature else str(rng.choice(["instant", "sorcery", "sorcery", "enchantment"])))
                mv = int(rng.choice(np.arange(1, 8), p=np.array(cmc_w[rar]) / sum(cmc_w[rar])))
                if kind != "creature":
                    mv = min(mv, 5)
                pips, gen = _pips_for(rng, [ci], mv)
                if preset:
                    nm, sub = preset[0], "Human"
                elif kind == "creature":
                    nm, sub = take(_CREATURE, ci)
                    sub = sub.split()[-1]
                else:
                    nm, sub = take(_SPELL, ci)
                oracle = "Flying" if (kind == "creature" and rng.random() < 0.3) else ("Destroy target creature." if kind in ("instant", "sorcery") and rng.random() < 0.5 else "")
                cards.append(_make_card(nm, rar, kind, pips, gen, subtype=sub, oracle=oracle))
    # gold cards: 10 pairs + 2 extra
    pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    for gi, (a, b) in enumerate(pairs + pairs[:2]):
        rar = "uncommon" if gi < 10 else "rare"
        mv = int(rng.choice([2, 3, 3, 4, 4, 5]))
        pips, gen = _pips_for(rng, [a, b], mv)
        nm = f"{_ADJ[COLORS[a]][gi % 12]} {_CREATURE[COLORS[b]][(gi * 5 + 1) % 12]}"
        while nm in used:
            nm += " II"
        used.add(nm)
        cards.append(_make_card(nm, rar, "creature", pips, gen, subtype=nm.split()[-1], oracle="Trample"))
    # split card (U/R) and DFC (G)
    pips = [0, 1, 0, 1, 0]
    split = _make_card("Searing Rift // Frozen Hold", "uncommon", "instant", pips, 0, oracle="Deal 3 damage // Tap target creature")
    split["hybrid"] = True  # either half is castable: either colour is enough
    split["mana_cost"] = "{1}{R} // {2}{U}"
    split["faces"] = [
        dict(name="Searing Rift", mana_cost="{1}{R}", type_line="Instant", oracle_text="Deal 3 damage.", colors=["R"]),
        dict(name="Frozen Hold", mana_cost="{2}{U}", type_line="Instant", oracle_text="Tap target creature.", colors=["U"]),
    ]
    split["layout"] = "split"
    split["mv"] = 2
    split["generic"] = 1
    cards.append(split)
    dfc = _make_card("Moonlit Scout // Howling Pack", "rare", "creature", [0, 0, 0, 0, 1], 2, subtype="Human Scout", oracle="Transforms")
    dfc["faces"] = [
        dict(name="Moonlit Scout", mana_cost="{2}{G}", type_line="Creature — Human Scout", oracle_text="Transform at night.", colors=["G"]),
        dict(name="Howling Pack", mana_cost="", type_line="Creature — Wolf", oracle_text="Trample", colors=["G"]),
    ]
    dfc["layout"] = "transform"
    cards.append(dfc)
    # hybrid card
    hy = _make_card("Ember-Lit Lancer", "uncommon", "creature", [1, 0, 0, 1, 0], 1, subtype="Knight", oracle="First strike", hybrid=("R", "W"))
    hy["pips"] = [2, 0, 0, 2, 0]  # {R/W}{R/W}: either colour pays each pip
    hy["mv"] = 3
    hy["mana_cost"] = "{1}{R/W}{R/W}"
    hy["colors"] = "WR"
    hy["generic"] = 1
    cards.append(hy)
    # colourless artifacts, rock
    for i, nm in enumerate(_ART_NAMES):
        rar = "common" if i < 3 else ("uncommon" if i < 5 else "rare")
        mv = [2, 3, 5, 4, 6, 4][i]
        cards.append(_make_card(nm, rar, "artifact" if i % 2 else "creature", [0] * 5, mv,
                                subtype="Construct", oracle="Colorless card"))
        cards[-1]["type_line"] = "Artifact Creature — Construct" if i % 2 == 0 else "Artifact"
        cards[-1]["is_creature"] = i % 2 == 0
    rock = _make_card("Prism Compass", "common", "artifact", [0] * 5, 2, oracle="{T}: Add one mana of any color.")
    rock["produces"] = "WUBRG"
    cards.append(rock)
    # dual lands
    for a, b in [(0, 1), (1, 2), (3, 4), (0, 3)]:
        nm = f"{['Hollow', 'Sunken', 'Rugged', 'Gleaming'][len(cards) % 4]} {['Crossing', 'Vale', 'Pass', 'Haven'][(a + b) % 4]}"
        while nm in used:
            nm += " II"
        used.add(nm)
        cards.append(_make_card(nm, "common", "land", [0] * 5, 0, oracle=f"({{T}}: Add {{{COLORS[a]}}} or {{{COLORS[b]}}}.)",
                                produces="".join(sorted((COLORS[a], COLORS[b]), key=COLORS.index))))
    # basics
    for col, nm in zip(COLORS, BASIC_NAMES):
        cards.append(_make_card(nm, "basic", "basic", [0] * 5, 0, subtype=nm, oracle=f"({{T}}: Add {{{col}}}.)", produces=col))
    # unique sorted names (stable column order)
    names = [c["name"] for c in cards]
    assert len(set(names)) == len(names), "duplicate synthetic card names"
    cards.sort(key=lambda c: c["name"])
    # planted betas
    rb = {"common": 0.0, "uncommon": 0.04, "rare": 0.10, "mythic": 0.16, "basic": 0.0}
    for c in cards:
        if c["is_basic"]:
            c["beta"] = 0.0
            continue
        b = rb[c["rarity"]] + float(rng.normal(0, 0.12))
        if c["kind"] in ("instant", "sorcery") and sum(c["pips"]) and c["rarity"] in ("common", "uncommon"):
            b += 0.05
        if c["mv"] <= 1 and not c["is_land"]:
            b -= 0.05
        if c["is_land"]:
            b = 0.04 + 0.02 * float(rng.normal())
        if c["name"] == "Prism Compass":
            b = 0.05
        c["beta"] = float(np.clip(b, -0.30, 0.40))
    return cards


@dataclass
class World:
    set_code: str
    seed: int
    cards: list[dict]
    truth: dict

    def __post_init__(self):
        self.names = [c["name"] for c in self.cards]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.arr = _arrays_from_truth(self.truth)
        a = self.arr
        self.C = len(self.names)
        self.is_land = a["is_land"]
        self.is_basic = a["is_basic"]
        self.beta = a["beta"]
        self.pips = a["pips"]
        self.cmask = self.pips > 0
        self.rarity = np.array([RARITIES.index(c["rarity"]) for c in self.cards])
        self.is_creature = np.array([c["is_creature"] for c in self.cards])
        self.mv = np.array([c["mv"] for c in self.cards], dtype=float)
        self.basic_idx = [self.index[n] for n in BASIC_NAMES]
        self.commons = [i for i, c in enumerate(self.cards) if c["rarity"] == "common"]
        self.uncommons = [i for i, c in enumerate(self.cards) if c["rarity"] == "uncommon"]
        self.rares = [i for i, c in enumerate(self.cards) if c["rarity"] == "rare"]
        self.mythics = [i for i, c in enumerate(self.cards) if c["rarity"] == "mythic"]
        w = np.ones(len(self.commons))
        for j, i in enumerate(self.commons):
            if self.cards[i]["kind"] == "land":
                w[j] = 0.4
            elif self.cards[i]["colors"] == "" and self.cards[i]["kind"] != "land":
                w[j] = 0.6
        self.common_p = w / w.sum()
        self.rarity_bonus = np.array([0.0, 0.05, 0.15, 0.25, 0.0])[self.rarity]


def _arrays_from_truth(truth: dict) -> dict:
    names = list(truth["cards"].keys())  # sorted
    n = len(names)
    pips = np.zeros((n, 5), dtype=np.int64)
    produces = np.zeros((n, 5))
    beta = np.zeros(n)
    is_land = np.zeros(n, dtype=bool)
    is_basic = np.zeros(n, dtype=bool)
    hyb = np.zeros(n, dtype=bool)
    for i, nm in enumerate(names):
        c = truth["cards"][nm]
        pips[i] = c["pips"]
        beta[i] = truth["beta"][nm]
        is_land[i] = c["is_land"]
        is_basic[i] = c["is_basic"]
        hyb[i] = c.get("hybrid", False)
        w = 1.0 if c["is_land"] else truth["colour_screw"]["rock_weight"]
        for ch in c["produces"]:
            produces[i, COLORS.index(ch)] = w
    return dict(names=names, pips=pips, produces=produces, beta=beta, is_land=is_land, is_basic=is_basic, hyb=hyb,
                tables=_screw_tables(truth["colour_screw"]["draws"]))


_TABLES: dict[int, np.ndarray] = {}


def _screw_tables(draws: int) -> np.ndarray:
    """T[k, s] = P(fewer than k sources among `draws` cards | s sources in 40), k = 1..3 (row k-1)."""
    if draws not in _TABLES:
        t = np.zeros((3, 41))
        for k in (1, 2, 3):
            for s in range(41):
                t[k - 1, s] = hypergeom.cdf(k - 1, 40, s, draws)
        _TABLES[draws] = t
    return _TABLES[draws]


def screw_index_matrix(arr: dict, D: np.ndarray) -> np.ndarray:
    """Mean over spell copies of P(coloured requirements unmet in the first 12 cards seen). D: (n, C)."""
    D = D.astype(float)
    n = D.shape[0]
    S = D @ arr["produces"]  # (n,5) sources
    tab = arr["tables"]
    pips = arr["pips"]
    spell = ~arr["is_land"]
    xs = np.arange(41)
    fail_c = np.ones((n, len(pips)))  # survival product per card, shape (n, C)
    # per colour failure probability for each card
    f_all = np.zeros((5, n, len(pips)))
    for c in range(5):
        fk = np.stack([np.interp(np.clip(S[:, c], 0, 40), xs, tab[k]) for k in range(3)], axis=1)  # (n,3)
        k_card = np.clip(pips[:, c], 0, 3)
        f = np.where(k_card[None, :] > 0, fk[:, np.maximum(k_card - 1, 0)], 0.0)
        f_all[c] = f
    has = pips > 0  # (C,5)
    surv = np.ones((n, len(pips)))
    allfail = np.ones((n, len(pips)))
    for c in range(5):
        m = has[:, c][None, :]
        surv *= np.where(m, 1.0 - f_all[c], 1.0)
        allfail *= np.where(m, f_all[c], 1.0)
    fail = np.where(arr["hyb"][None, :], allfail, 1.0 - surv)
    fail = np.where(has.any(axis=1)[None, :], fail, 0.0)
    nsp = (D * spell[None, :]).sum(axis=1)
    return (D * spell[None, :] * fail).sum(axis=1) / np.maximum(nsp, 1)


def strength_matrix(truth: dict, arr: dict, D: np.ndarray) -> np.ndarray:
    """Centred deck-strength logit for each row of D (n, C) of copy counts (basics included)."""
    D = D.astype(float)
    s = D @ arr["beta"]
    nl = D @ arr["is_land"].astype(float)
    s = s + truth["land"]["coef"] * (nl - truth["land"]["peak"]) ** 2
    s = s - truth["colour_screw"]["gamma"] * screw_index_matrix(arr, D)
    return s - truth["center"]


def truth_strength(truth: dict | str | Path, deck_by_name: dict[str, int]) -> float:
    """Evaluate the planted deck-strength logit from a truth dict / truth.json path."""
    if not isinstance(truth, dict):
        truth = json.loads(Path(truth).read_text(encoding="utf-8"))
    arr = truth.get("_arr") or _arrays_from_truth(truth)
    names = arr["names"]
    idx = {n: i for i, n in enumerate(names)}
    D = np.zeros((1, len(names)))
    for k, v in deck_by_name.items():
        D[0, idx[k]] += v
    return float(strength_matrix(truth, arr, D)[0])


_DEFAULT_WORLDS: dict[tuple[str, int], World] = {}


def true_strength(deck_by_name: dict[str, int], truth: dict | str | Path | None = None) -> float:
    """Planted deck strength (logit, centred). Without `truth`, uses the default world (TST, seed 0)."""
    if truth is None:
        key = ("TST", 0)
        if key not in _DEFAULT_WORLDS:
            _DEFAULT_WORLDS[key] = build_world("TST", 0)
        truth = _DEFAULT_WORLDS[key].truth
    return truth_strength(truth, deck_by_name)


def build_world(set_code: str = "TST", seed: int = 0) -> World:
    cards = _build_cards(seed)
    truth = dict(
        set_code=set_code, seed=seed,
        intercept=INTERCEPT, on_play=ON_PLAY, skill_slope=SKILL_SLOPE, mull=MULL, opp_mull=OPP_MULL,
        opp_strength=OPP_STRENGTH, drift_per_day=DRIFT_PER_DAY,
        land={"coef": LAND_COEF, "peak": LAND_PEAK},
        colour_screw={"gamma": GAMMA, "draws": SCREW_DRAWS, "rock_weight": 0.5},
        center=0.0,
        beta={c["name"]: c["beta"] for c in cards},
        cards={c["name"]: dict(pips=c["pips"], is_land=c["is_land"], is_basic=c["is_basic"], produces=c["produces"],
                               rarity=c["rarity"], mv=c["mv"], hybrid=c["hybrid"], colors=c["colors"])
               for c in cards},
    )
    world = World(set_code, seed, cards, truth)
    # centre on the typical Sealed deck of an average player
    rng = np.random.default_rng(seed * 7919 + 13)
    pools = np.stack([_sealed_pool(world, rng) for _ in range(300)])
    decks = np.stack([_heuristic_deck(world, p, 0.5, rng) for p in pools])
    raw = strength_matrix(truth, world.arr, decks)  # center currently 0
    truth["center"] = float(raw.mean())
    world.truth = truth
    return world


# ---------------------------------------------------------------------------------------------
# pools and the synthetic player
# ---------------------------------------------------------------------------------------------

def _booster(world: World, rng) -> np.ndarray:
    cm = rng.choice(len(world.commons), size=10, replace=False, p=world.common_p)
    un = rng.choice(len(world.uncommons), size=3, replace=False)
    rm = world.rares if rng.random() > 1 / 7.4 else world.mythics
    ids = [world.commons[i] for i in cm] + [world.uncommons[i] for i in un] + [rm[int(rng.integers(len(rm)))]]
    return np.array(ids)


def _sealed_pool(world: World, rng) -> np.ndarray:
    ids = np.concatenate([_booster(world, rng) for _ in range(6)])
    return np.bincount(ids, minlength=world.C).astype(np.int16)


def _perceived(world: World, ids: np.ndarray, skill_norm: float, rng) -> np.ndarray:
    noise = 0.30 - 0.20 * skill_norm
    return 0.4 * world.beta[ids] + world.rarity_bonus[ids] + rng.normal(0, noise, len(ids))


def _draft_pool(world: World, skill_norm: float, rng) -> np.ndarray:
    """3 rounds: pick the best 14 of 28 seen cards; after round 1 the drafter settles into a colour pair."""
    pool = np.zeros(world.C, dtype=np.int16)
    pair_mask = None
    for r in range(3):
        ids = np.concatenate([_booster(world, rng), _booster(world, rng)])
        v = _perceived(world, ids, skill_norm, rng)
        if pair_mask is not None:
            fits = ~((world.cmask[ids] & ~pair_mask[None, :]).any(axis=1))
            v = v + 0.8 * fits
        take = ids[np.argsort(-v)[:14]]
        np.add.at(pool, take, 1)
        if r == 0:
            weight = np.zeros(5)
            vt = _perceived(world, take, skill_norm, rng) + 0.5
            for cid, val in zip(take, vt):
                weight += world.cmask[cid] * max(val, 0.05)
            top = np.argsort(-weight)[:2]
            pair_mask = np.zeros(5, dtype=bool)
            pair_mask[top] = True
    return pool


_PAIRS = [(a, b) for a in range(5) for b in range(a + 1, 5)]


def _heuristic_deck(world: World, pool: np.ndarray, skill_norm: float, rng) -> np.ndarray:
    """A noisy human-like deck builder: best colour pair by perceived value, occasional splash, 16-18 lands."""
    C = world.C
    ids = np.repeat(np.arange(C), pool)
    land_mask = world.is_land[ids]
    spell_ids = ids[~land_mask]
    land_ids = ids[land_mask]
    v = _perceived(world, spell_ids, skill_norm, rng)
    need = world.cmask[spell_ids]  # (n,5)
    best, best_sc = None, -1e9
    for a, b in _PAIRS:
        pm = np.zeros(5, dtype=bool)
        pm[[a, b]] = True
        ok = ~((need & ~pm[None, :]).any(axis=1))
        vals = np.sort(np.where(ok, v, -1.0))[::-1][:23]
        sc = vals.sum() + rng.normal(0, 0.25)
        if sc > best_sc:
            best, best_sc = (a, b, ok), sc
    a, b, ok = best
    pm = np.zeros(5, dtype=bool)
    pm[[a, b]] = True
    # lands count
    nlands = int(rng.choice([16, 17, 18], p=[0.25 - 0.1 * skill_norm, 0.5 + 0.2 * skill_norm, 0.25 - 0.1 * skill_norm]))
    # nonbasic lands matching the pair
    chosen_lands: list[int] = []
    splash = -1
    splash_cards: list[int] = []
    # splash?
    if rng.random() < 0.22:
        for s in rng.permutation([x for x in range(5) if x not in (a, b)]):
            cand = np.where((need[:, s]) & (world.pips[spell_ids][:, s] == 1) &
                            ~((need & ~(pm | (np.arange(5) == s))[None, :]).any(axis=1)))[0]
            if len(cand):
                top = cand[np.argsort(-v[cand])[:int(rng.integers(1, 3))]]
                if v[top[0]] > np.sort(np.where(ok, v, -1.0))[::-1][22] + 0.05:
                    splash, splash_cards = int(s), list(top)
                    break
    allowed = pm.copy()
    if splash >= 0:
        allowed[splash] = True
    for li in np.unique(land_ids):
        prod = [COLORS.index(ch) for ch in world.cards[li]["produces"]]
        n_in = sum(allowed[x] for x in prod)
        both = all(allowed[x] for x in prod)
        if (both or (splash >= 0 and n_in >= 1 and splash in prod and any(pm[x] for x in prod))) and rng.random() < 0.9:
            chosen_lands.extend([int(li)] * int(pool[li]))
    chosen_lands = chosen_lands[:5]
    nspell = 40 - nlands
    ok_idx = np.where(ok)[0]
    order = ok_idx[np.argsort(-v[ok_idx])]
    picked = list(splash_cards)
    for i in order:
        if len(picked) >= nspell:
            break
        if i not in picked:
            picked.append(int(i))
    if len(picked) < nspell:  # not enough playables: top up with anything
        rest = [int(i) for i in np.argsort(-v) if int(i) not in picked]
        picked += rest[: nspell - len(picked)]
    deck = np.zeros(C, dtype=np.int16)
    for i in picked:
        deck[spell_ids[i]] += 1
    for li in chosen_lands:
        deck[li] += 1
    nb = 40 - int(deck.sum())
    # basics
    pipw = np.zeros(5)
    for i in picked:
        pipw += world.pips[spell_ids[i]]
    sp_b = 0
    if splash >= 0:
        sp_b = 2 if not any(world.cards[li]["produces"].find(COLORS[splash]) >= 0 for li in chosen_lands) else 1
        sp_b = min(sp_b, nb)
    nm = nb - sp_b
    wa, wb = pipw[a] + 0.5, pipw[b] + 0.5
    na = int(round(nm * wa / (wa + wb) + rng.normal(0, 0.6)))
    na = int(np.clip(na, min(3, nm), max(nm - 3, 0)))
    deck[world.basic_idx[a]] += na
    deck[world.basic_idx[b]] += nm - na
    if splash >= 0:
        deck[world.basic_idx[splash]] += sp_b
    return deck


def _perturb(world: World, deck: np.ndarray, pool: np.ndarray, rng) -> np.ndarray:
    """Between-game sideboarding: swap one spell for a castable sideboard spell (or shift a basic)."""
    new = deck.copy()
    src = np.where((new > 0) & ~world.is_land)[0]
    srcs = np.repeat(src, new[src])
    basics = np.array([new[i] for i in world.basic_idx])
    colours_ok = basics >= 3
    side = (pool - np.minimum(pool, new)).astype(int)
    side[world.is_land] = 0
    cand = np.where((side > 0) & ~(world.cmask & ~colours_ok[None, :]).any(axis=1))[0]
    if len(srcs) and len(cand):
        out = int(rng.choice(srcs))
        inn = int(rng.choice(cand))
        new[out] -= 1
        new[inn] += 1
    else:
        order = np.argsort(-basics)
        if basics[order[0]] > 4 and basics[order[1]] > 0:
            new[world.basic_idx[order[0]]] -= 1
            new[world.basic_idx[order[1]]] += 1
    return new


# ---------------------------------------------------------------------------------------------
# hands
# ---------------------------------------------------------------------------------------------

def _simulate_hands(world: World, slots: np.ndarray, nd: np.ndarray, skill_norm: np.ndarray, rng):
    """slots: (n,40) card ids of each game's deck. Returns mulligans (n,), opening (n,C) int8, drawn (n,C) int8."""
    n = len(slots)
    C = world.C
    perm = np.argsort(rng.random((n, 40)), axis=1)
    L = np.take_along_axis(slots, perm, axis=1)
    isl = world.is_land[L]
    lands7 = isl[:, :7].sum(axis=1)
    bad = (lands7 < 2) | (lands7 > 5)
    mull_p = np.where(bad, 0.9, 0.0)
    wants = rng.random(n) < mull_p
    mulls = np.zeros(n, dtype=np.int8)
    hand_removed: dict[int, list[int]] = {}
    todo = np.where(wants)[0]
    level = 0
    while len(todo):
        level += 1
        mulls[todo] = level
        p2 = np.argsort(rng.random((len(todo), 40)), axis=1)
        L2 = np.take_along_axis(slots[todo], p2, axis=1)
        L[todo] = L2
        l7 = world.is_land[L2[:, :7]].sum(axis=1)
        again = ((l7 < 2) | (l7 > 5)) & (level < 3) & (rng.random(len(todo)) < 0.9)
        todo = todo[again]
    open_counts = np.zeros((n, C), dtype=np.int8)
    rows = np.repeat(np.arange(n), 7)
    np.add.at(open_counts, (rows, L[:, :7].reshape(-1)), 1)
    # bottom m cards for mulliganed hands
    for i in np.where(mulls > 0)[0]:
        hand = list(L[i, :7])
        for _ in range(int(mulls[i])):
            nl = sum(1 for c in hand if world.is_land[c])
            if nl > 3:
                cand = [c for c in hand if world.is_land[c]]
                pick = cand[-1]
            else:
                cand = [c for c in hand if not world.is_land[c]] or hand
                pick = max(cand, key=lambda c: world.mv[c])
            hand.remove(pick)
            open_counts[i, pick] -= 1
    maxd = int(nd.max()) if n else 0
    cols = L[:, 7:7 + maxd]
    drawn = np.zeros((n, C), dtype=np.int8)
    mask = np.arange(maxd)[None, :] < nd[:, None]
    rr = np.repeat(np.arange(n)[:, None], maxd, axis=1)
    np.add.at(drawn, (rr[mask], cols[mask]), 1)
    return mulls, open_counts, drawn


# ---------------------------------------------------------------------------------------------
# event simulation
# ---------------------------------------------------------------------------------------------

_RANKS = ["bronze", "silver", "gold", "platinum", "diamond", "mythic"]


def _rank_of(theta: np.ndarray, rng) -> np.ndarray:
    z = (theta - 0.40) / 0.30 * 5.0 + rng.normal(0, 0.8, len(theta))
    return np.array(_RANKS)[np.clip(np.round(z), 0, 5).astype(int)]


def _fmt_time(t: np.ndarray) -> np.ndarray:
    return np.char.replace(t.astype("datetime64[s]").astype(str), "T", " ")


def _main_str(deck: np.ndarray, world: World) -> tuple[str, str]:
    b = np.array([deck[i] for i in world.basic_idx])
    # colours by sources + pips: main = two colours with most pips among spells
    pipw = (deck[:, None] * np.where(world.is_land[:, None], 0, world.pips)).sum(axis=0)
    order = np.argsort(-pipw, kind="stable")
    main = sorted(order[:2].tolist())
    splash = ""
    for c in range(5):
        if c not in main and pipw[c] > 0:
            splash += COLORS[c]
    return "".join(COLORS[c] for c in main), splash


def _simulate_batch(world: World, event: str, E: int, rng, ev_offset: int):
    cfg = EVENT_CFG[event]
    bo3 = cfg["bo3"]
    K = 15 if bo3 else 9
    C = world.C
    truth = world.truth
    theta = np.clip(rng.normal(0.55, 0.05, E), 0.40, 0.70)
    skill_norm = (theta - 0.40) / 0.30
    pools = np.zeros((E, C), dtype=np.int16)
    base = np.zeros((E, C), dtype=np.int16)
    for e in range(E):
        pools[e] = _sealed_pool(world, rng) if cfg["pool"] == "sealed" else _draft_pool(world, skill_norm[e], rng)
        base[e] = _heuristic_deck(world, pools[e], skill_norm[e], rng)
    # deck versions
    flag = rng.random((E, K)) < cfg["p_swap"]
    flag[:, 0] = False
    cum = np.cumsum(flag, axis=1)
    nver = cum[:, -1] + 1
    voff = np.concatenate([[0], np.cumsum(nver)[:-1]])
    Dv = np.zeros((int(nver.sum()), C), dtype=np.int16)
    ev_of_ver = np.repeat(np.arange(E), nver)
    for e in range(E):
        Dv[voff[e]] = base[e]
        for j in range(1, nver[e]):
            Dv[voff[e] + j] = _perturb(world, Dv[voff[e] + j - 1], pools[e], rng)
    ver = voff[:, None] + cum  # (E,K) global version per slot
    ts_v = strength_matrix(truth, world.arr, Dv)  # (nv,)
    ts_base = ts_v[voff]
    # opponents
    if bo3:
        opp_e = (np.arange(E)[:, None] + rng.integers(1, max(E, 2), (E, 5))) % E
        opp_e = np.repeat(opp_e, 3, axis=1)
    else:
        opp_e = (np.arange(E)[:, None] + rng.integers(1, max(E, 2), (E, K))) % E
    on_play = rng.random((E, K)) < 0.5
    user_turns = np.clip(np.round(rng.normal(5.5, 1.8, (E, K))), 3, 11).astype(int)
    nd = user_turns - on_play.astype(int)
    # hands
    flat_ver = ver.reshape(-1)
    # deck slot arrays per version (40 cards)
    nv = len(Dv)
    slots_v = np.zeros((nv, 40), dtype=np.int16)
    for v in range(nv):
        slots_v[v] = np.repeat(np.arange(C), Dv[v])[:40]
    sk_flat = np.repeat(skill_norm, K)
    mulls, open_c, drawn_c = _simulate_hands(world, slots_v[flat_ver], nd.reshape(-1), sk_flat, rng)
    opp_mull = rng.choice([0, 1, 2], size=E * K, p=[0.88, 0.10, 0.02])
    mulls_ek = mulls.reshape(E, K)
    opp_mull = opp_mull.reshape(E, K)
    # times
    start_day = rng.uniform(0, WINDOW_DAYS, E)
    draft_secs = (start_day * 86400).astype(np.int64)
    game_secs = draft_secs[:, None] + 3600 + np.arange(K)[None, :] * 1500 + rng.integers(0, 600, (E, K))
    day_idx = np.floor(game_secs / 86400.0)
    theta_opp = theta[opp_e]
    logit = (truth["intercept"] + truth["drift_per_day"] * day_idx
             + truth["skill_slope"] * (theta[:, None] - theta_opp)
             + truth["on_play"] * on_play
             + truth["mull"] * np.minimum(mulls_ek, 2) + truth["opp_mull"] * np.minimum(opp_mull, 2)
             + ts_v[ver] + truth["opp_strength"] * ts_base[opp_e])
    p = 1.0 / (1.0 + np.exp(-logit))
    won_pot = rng.random((E, K)) < p
    # walk the runs
    ev_idx, slot_idx, match_no, game_no, build_no = [], [], [], [], []
    ev_counts = np.zeros(E, dtype=int)
    for e in range(E):
        seen: dict[int, int] = {}
        mn = 0

        def rec(k, mno, gno):
            vv = int(ver[e, k])
            if vv not in seen:
                seen[vv] = len(seen)
            ev_idx.append(e); slot_idx.append(k); match_no.append(mno); game_no.append(gno); build_no.append(seen[vv])
            ev_counts[e] += 1

        if not bo3:
            w = l = 0
            k = 0
            while w < 7 and l < 3:
                rec(k, k + 1, 1)
                if won_pot[e, k]:
                    w += 1
                else:
                    l += 1
                k += 1
        else:
            mw = ml = 0
            m = 0
            while mw < 4 and ml < 2:
                gw = gl = 0
                g = 0
                while gw < 2 and gl < 2:
                    k = 3 * m + g
                    rec(k, m + 1, g + 1)
                    if won_pot[e, k]:
                        gw += 1
                    else:
                        gl += 1
                    g += 1
                if gw == 2:
                    mw += 1
                else:
                    ml += 1
                m += 1
    ev_idx = np.array(ev_idx)
    slot_idx = np.array(slot_idx)
    flat = ev_idx * K + slot_idx
    vs = ver[ev_idx, slot_idx]
    # per-event metadata
    main_s, splash_s, opp_main = [], [], []
    ev_main = []
    for e in range(E):
        ev_main.append(_main_str(base[e], world))
    out = dict(
        E=E, ev_idx=ev_idx, ev_counts=ev_counts, match_no=np.array(match_no), game_no=np.array(game_no),
        build_no=np.array(build_no), theta=theta, theta_opp=theta_opp[ev_idx, slot_idx],
        on_play=on_play[ev_idx, slot_idx], mulls=mulls_ek[ev_idx, slot_idx], opp_mull=opp_mull[ev_idx, slot_idx],
        won=won_pot[ev_idx, slot_idx], draft_secs=draft_secs[ev_idx], game_secs=game_secs[ev_idx, slot_idx],
        num_turns=(2 * user_turns[ev_idx, slot_idx] + rng.integers(0, 2, len(ev_idx))),
        open=open_c[flat], drawn=drawn_c[flat], deck=Dv[vs].astype(np.int8),
        side=np.clip(pools[ev_idx] - np.minimum(pools[ev_idx], Dv[vs]), 0, 127).astype(np.int8),
        ev_main=[ev_main[e] for e in ev_idx], opp_colors=[ev_main[opp_e[e, k]][0] for e, k in zip(ev_idx, slot_idx)],
    )
    out["side"][:, world.is_basic] = 0
    return out


def _concat_batches(batches: list[dict], ev_shift: list[int]) -> dict:
    out = {}
    for k in batches[0]:
        if k == "E":
            continue
        v0 = batches[0][k]
        if isinstance(v0, list):
            out[k] = sum((b[k] for b in batches), [])
        elif k == "ev_idx":
            out[k] = np.concatenate([b[k] + s for b, s in zip(batches, ev_shift)])
        elif k in ("ev_counts", "theta"):
            out[k] = np.concatenate([b[k] for b in batches])
        else:
            out[k] = np.concatenate([b[k] for b in batches])
    return out


def _write_game_csv(path: Path, world: World, event: str, d: dict, rng, n_rows: int, pad_cards: int = 0):
    names = world.names + [f"Filler Card {i:04d}" for i in range(pad_cards)]
    n = n_rows
    ev_idx = d["ev_idx"][:n]
    uniq_events = np.unique(ev_idx)
    hex_map = {int(e): rng.bytes(16).hex() for e in uniq_events}
    draft_id = np.array([hex_map[int(e)] for e in ev_idx])
    theta = d["theta"][ev_idx]
    reported = np.round(np.clip(theta + rng.normal(0, 0.01, n), 0.40, 0.70), 2)
    miss = rng.random(n) < 0.02
    skill_s = pl.Series("user_game_win_rate_bucket", np.where(miss, np.nan, reported)).fill_nan(None)
    nbucket = rng.choice([5, 10, 50, 100, 500, 1000], size=n)
    opp_rank = _rank_of(d["theta_opp"][:n], rng)
    rank = _rank_of(theta, rng)
    main = [m[0] for m in d["ev_main"][:n]]
    splash = [m[1] for m in d["ev_main"][:n]]
    meta = pl.DataFrame({
        "expansion": [world.set_code] * n,
        "event_type": [event] * n,
        "draft_id": draft_id,
        "draft_time": _fmt_time(T0 + d["draft_secs"][:n].astype("timedelta64[s]")),
        "game_time": _fmt_time(T0 + d["game_secs"][:n].astype("timedelta64[s]")),
        "build_index": d["build_no"][:n].astype(np.int64),
        "match_number": d["match_no"][:n].astype(np.int64),
        "game_number": d["game_no"][:n].astype(np.int64),
        "rank": rank,
        "opp_rank": opp_rank,
        "main_colors": main,
        "splash_colors": splash,
        "on_play": np.where(d["on_play"][:n], "True", "False"),
        "num_mulligans": d["mulls"][:n].astype(np.int64),
        "opp_num_mulligans": d["opp_mull"][:n].astype(np.int64),
        "opp_colors": d["opp_colors"][:n],
        "num_turns": d["num_turns"][:n].astype(np.int64),
        "won": np.where(d["won"][:n], "True", "False"),
        "user_n_games_bucket": nbucket.astype(np.int64),
    }).with_columns(skill_s)
    # draft_id must be grouped contiguously by event (already ordered)
    pad = np.zeros((n, pad_cards), dtype=np.int8)
    zero = np.zeros((n, world.C + pad_cards), dtype=np.int8)
    blocks = [("opening_hand_", np.hstack([d["open"][:n], pad])), ("drawn_", np.hstack([d["drawn"][:n], pad])),
              ("tutored_", zero), ("deck_", np.hstack([d["deck"][:n], pad])), ("sideboard_", np.hstack([d["side"][:n], pad]))]
    chunk = 8000
    with gzip.open(path, "wb", compresslevel=3) as f:
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            parts = [meta.slice(s, e - s)]
            for pre, mat in blocks:
                parts.append(pl.from_numpy(np.ascontiguousarray(mat[s:e]), schema=[pre + nm for nm in names], orient="row"))
            df = pl.concat(parts, how="horizontal")
            f.write(df.write_csv(include_header=(s == 0)).encode("utf-8"))
    return draft_id


def generate_game_file(world: World, event: str, n_games: int, path: Path, rng: np.random.Generator,
                       pad_cards: int = 0) -> int:
    """Simulate `n_games` (rounded up to whole events) of `event` and write the 17lands game CSV. Returns rows."""
    avg = 8.8 if EVENT_CFG[event]["bo3"] else 6.6
    batches, shifts, total, off = [], [], 0, 0
    while total < n_games:
        E = int((n_games - total) / avg * 1.15) + 8
        b = _simulate_batch(world, event, E, rng, off)
        batches.append(b)
        shifts.append(off)
        off += E
        total += len(b["ev_idx"])
    d = _concat_batches(batches, shifts)
    # cut at an event boundary at or after n_games
    ev = d["ev_idx"]
    cut = len(ev)
    if len(ev) > n_games:
        cut = n_games
        while cut < len(ev) and ev[cut] == ev[cut - 1]:
            cut += 1
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(ev):
            d[k] = v[:cut]
        elif isinstance(v, list) and len(v) == len(ev):
            d[k] = v[:cut]
    _write_game_csv(path, world, event, d, rng, cut, pad_cards)
    return cut


# ---------------------------------------------------------------------------------------------
# auxiliary files
# ---------------------------------------------------------------------------------------------

def _write_cards_csv(path: Path, world: World):
    rows = []
    for i, c in enumerate(world.cards):
        ident = c["produces"] if c["is_land"] else c["colors"]
        rows.append(dict(
            id=100000 + i, expansion=world.set_code, name=c["name"], rarity=c["rarity"], color_identity=ident,
            mana_value=5.0 if c.get("layout") == "split" else float(c["mv"]), types=c["type_line"].replace("—", "-"), is_booster=not c["is_basic"],
        ))
    pl.DataFrame(rows).write_csv(path)


def _scryfall_objects(world: World) -> list[dict]:
    out = []
    for i, c in enumerate(world.cards):
        o = dict(
            object="card", id=f"00000000-0000-0000-0000-{i:012d}", name=c["name"], set=world.set_code.lower(),
            rarity="common" if c["is_basic"] else c["rarity"], cmc=float(c["mv"]),
        )
        if c["faces"]:
            o["layout"] = c.get("layout", "transform")
            o["type_line"] = " // ".join(f["type_line"] for f in c["faces"])
            o["card_faces"] = c["faces"]
            if o["layout"] == "split":
                o["mana_cost"] = c["mana_cost"]
                o["cmc"] = 5.0
                o["colors"] = sorted({x for f in c["faces"] for x in f["colors"]}, key=COLORS.index)
                o["color_identity"] = o["colors"]
            else:
                o["color_identity"] = list(c["colors"])
        else:
            o["layout"] = "normal"
            o["mana_cost"] = c["mana_cost"]
            o["type_line"] = c["type_line"]
            o["oracle_text"] = c["oracle"]
            o["colors"] = list(c["colors"])
            o["color_identity"] = list(dict.fromkeys(list(c["colors"]) + (list(c["produces"]) if c["is_land"] else [])))
            o["color_identity"] = [x for x in COLORS if x in o["color_identity"]]
        if c["produces"]:
            o["produced_mana"] = list(c["produces"])
        out.append(o)
    return out


def _write_draft_file(path: Path, world: World, event: str, rng, n_rows: int = 100):
    C = world.C
    nb = [i for i in range(C) if not world.is_basic[i]]
    names = world.names
    rows = []
    did = 0
    while len(rows) < n_rows:
        pool = np.zeros(C, dtype=np.int64)
        d_id = rng.bytes(16).hex()
        for pk in range(3):
            pack = list(rng.choice(nb, size=14, replace=False))
            for pn in range(14):
                if len(rows) >= n_rows:
                    break
                pick = int(rng.choice(pack))
                cnt = np.zeros(C, dtype=np.int64)
                for c in pack:
                    cnt[c] += 1
                r = dict(expansion=world.set_code, event_type=event, draft_id=d_id,
                         draft_time="2024-03-05 10:00:00", rank="gold", event_match_wins=int(rng.integers(0, 8)),
                         event_match_losses=int(rng.integers(0, 4)), pack_number=pk, pick_number=pn,
                         pick=names[pick], pick_maindeck_rate=round(float(rng.random()), 3),
                         pick_sideboard_in_rate=round(float(rng.random()), 3))
                for i in range(C):
                    r["pack_card_" + names[i]] = int(cnt[i])
                for i in range(C):
                    r["pool_" + names[i]] = int(pool[i])
                rows.append(r)
                pack.remove(pick)
                pool[pick] += 1
        did += 1
    df = pl.DataFrame(rows)
    with gzip.open(path, "wb", compresslevel=3) as f:
        f.write(df.write_csv().encode("utf-8"))


def _heldout(world: World, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed * 104729 + 7)
    out = []
    for _ in range(n):
        pool = _sealed_pool(world, rng)
        theta = float(np.clip(rng.normal(0.55, 0.05), 0.40, 0.70))
        deck = _heuristic_deck(world, pool, (theta - 0.40) / 0.30, rng)
        out.append(dict(
            theta=theta,
            pool={world.names[i]: int(pool[i]) for i in np.nonzero(pool)[0]},
            player_deck={world.names[i]: int(deck[i]) for i in np.nonzero(deck)[0]},
        ))
    return out


def generate(out_dir: str | Path, set_code: str = "TST", seed: int = 0, scale: float = 1.0, n_heldout: int = 40,
             events: tuple[str, ...] = tuple(EVENTS)) -> dict:
    """Write the full synthetic data set into `out_dir`. Returns a dict of paths and counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = build_world(set_code, seed)
    rng = np.random.default_rng(seed)
    files: dict[str, str] = {}
    n_games: dict[str, int] = {}
    for ev in events:
        target = max(150, int(BASE_GAMES[ev] * scale))
        p = out / f"game_data_public.{set_code}.{ev}.csv.gz"
        n_games[ev] = generate_game_file(world, ev, target, p, rng)
        files[f"game_{ev}"] = str(p)
    p = out / f"draft_data_public.{set_code}.PremierDraft.csv.gz"
    _write_draft_file(p, world, "PremierDraft", rng)
    files["draft_PremierDraft"] = str(p)
    p = out / "cards.csv"
    _write_cards_csv(p, world)
    files["cards"] = str(p)
    p = out / f"scryfall_{set_code}.json"
    p.write_text(json.dumps(_scryfall_objects(world), ensure_ascii=False), encoding="utf-8")
    files["scryfall"] = str(p)
    p = out / "truth.json"
    p.write_text(json.dumps(world.truth, ensure_ascii=False, indent=1), encoding="utf-8")
    files["truth"] = str(p)
    p = out / "heldout_pools.json"
    p.write_text(json.dumps(_heldout(world, n_heldout, seed), ensure_ascii=False), encoding="utf-8")
    files["heldout"] = str(p)
    return dict(out_dir=str(out), set_code=set_code, files=files, n_games=n_games, seed=seed, scale=scale)


def load_heldout(out_dir: str | Path) -> list[dict]:
    return json.loads((Path(out_dir) / "heldout_pools.json").read_text(encoding="utf-8"))
