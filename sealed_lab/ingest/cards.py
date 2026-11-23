"""Build `cards.parquet`: Scryfall -> 17lands cards.csv -> inferred fallbacks."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

from sealed_lab.config import BASICS, COLORS
from sealed_lab.ingest import scryfall as sf
from sealed_lab.ingest.detect import open_binary

log = logging.getLogger(__name__)

SYMBOL_RE = re.compile(r"\{([^}]+)\}")

CARD_SCHEMA = {
    "card_id": pl.Int32, "name": pl.String, "front_name": pl.String, "rarity": pl.String, "colors": pl.String,
    "color_identity": pl.String, "mana_value": pl.Float64,
    "pip_W": pl.Int8, "pip_U": pl.Int8, "pip_B": pl.Int8, "pip_R": pl.Int8, "pip_G": pl.Int8,
    "hybrid": pl.Boolean, "generic": pl.Int8, "types": pl.String, "is_land": pl.Boolean, "is_basic": pl.Boolean,
    "is_creature": pl.Boolean, "produces": pl.String, "oracle_text": pl.String, "meta_source": pl.String,
    "power": pl.Float64, "toughness": pl.Float64, "pt_variable": pl.Boolean, "keywords": pl.String,
}
# bump when columns are added to the card table or name matching changes: it is part of the corpus signature, so old corpora
# are rebuilt. 3: Scryfall printed names (OM1) and lost-letter names ("Bespoke B?")
CARDS_VERSION = 3
_NUM_RE = re.compile(r"^\s*([+-]?\d+)")


def order_colors(s) -> str:
    """Subset of WUBRG in WUBRG order."""
    return "".join(c for c in COLORS if c in s)


def parse_mana_cost(cost: str | None) -> dict:
    """Parse a Scryfall mana cost.

    Returns dict(pips=[W,U,B,R,G], generic, hybrid, colors, mv).
    Hybrid / Phyrexian / twobrid symbols add one pip to EACH colour they mention (half-pip rounded up),
    set hybrid=True and contribute 1 (2 for {2/W}) to mv. {C}, {S}, {X} are treated as generic (X = 0).
    """
    pips = [0] * 5
    generic = 0
    mv = 0.0
    hybrid = False
    for tok in SYMBOL_RE.findall(cost or ""):
        t = tok.upper()
        if t.isdigit():
            generic += int(t)
            mv += int(t)
        elif t in ("X", "Y", "Z"):
            continue
        elif t in COLORS:
            pips[COLORS.index(t)] += 1
            mv += 1
        elif t in ("C", "S"):
            generic += 1
            mv += 1
        elif "/" in t:
            parts = t.split("/")
            cols = [p for p in parts if p in COLORS]
            hybrid = True
            for c in cols:
                pips[COLORS.index(c)] += 1
            mv += 2 if parts[0].isdigit() else 1
        elif t == "HW":  # half mana (un-sets)
            continue
    colors = "".join(c for i, c in enumerate(COLORS) if pips[i] > 0)
    return dict(pips=pips, generic=generic, hybrid=hybrid, colors=colors, mv=mv)


def _is_split(o: dict) -> bool:
    return o.get("layout") in ("split", "fuse", "aftermath") or (
        bool(o.get("card_faces")) and " // " in (o.get("mana_cost") or ""))


def _face_view(o: dict) -> dict:
    """Normalised view of a Scryfall object (multi-face: front face; split: either half castable)."""
    faces = o.get("card_faces") or []
    if faces and _is_split(o):
        parsed = [parse_mana_cost(f.get("mana_cost")) for f in faces]
        pips = [max(p["pips"][i] for p in parsed) for i in range(5)]
        colors = order_colors("".join(p["colors"] for p in parsed))
        differ = len({p["colors"] for p in parsed}) > 1
        pc = dict(pips=pips, generic=min(p["generic"] for p in parsed), hybrid=differ or any(p["hybrid"] for p in parsed),
                  colors=colors, mv=min(p["mv"] for p in parsed))
        text = "\n//\n".join(f.get("oracle_text", "") for f in faces)
        return dict(parsed=pc, type_line=faces[0].get("type_line") or o.get("type_line", ""), oracle=text,
                    cmc=o.get("cmc"), faces=faces)
    if faces and not o.get("mana_cost"):
        f0 = faces[0]
        pc = parse_mana_cost(f0.get("mana_cost"))
        text = "\n//\n".join(f.get("oracle_text", "") for f in faces)
        return dict(parsed=pc, type_line=f0.get("type_line") or o.get("type_line", ""), oracle=text,
                    cmc=o.get("cmc"), faces=faces)
    pc = parse_mana_cost(o.get("mana_cost"))
    tl = o.get("type_line", "")
    if faces:
        tl = faces[0].get("type_line") or tl
        text = "\n//\n".join(f.get("oracle_text", "") for f in faces)
    else:
        text = o.get("oracle_text", "")
    return dict(parsed=pc, type_line=tl, oracle=text, cmc=o.get("cmc"), faces=faces)


def _produced(o: dict) -> str:
    pm = list(o.get("produced_mana") or [])
    for f in o.get("card_faces") or []:
        pm += list(f.get("produced_mana") or [])
    return order_colors(set(pm))


def parse_pt(power, toughness) -> tuple[float | None, float | None, bool]:
    """(power, toughness, variable) from Scryfall strings: '3' -> 3, '*' / '1+*' / 'X' -> variable (numeric part, else 0).
    (None, None, False) when the card has no power / toughness."""
    if power is None and toughness is None:
        return None, None, False
    var = False
    out = []
    for v in (power, toughness):
        if v is None:
            out.append(0.0)
            continue
        m = _NUM_RE.match(str(v))
        if not str(v).strip().lstrip("+-").isdigit():
            var = True
        out.append(float(m.group(1)) if m else 0.0)
    return out[0], out[1], var


def _norm_types(t: str) -> str:
    return re.sub(r"\s*[—–]\s*", " - ", t or "").strip()


def _rarity(raw: str | None, types: str, name: str) -> str:
    if name in BASICS or types.startswith("Basic Land"):
        return "basic"
    r = (raw or "").lower()
    if r in ("common", "uncommon", "rare", "mythic", "basic"):
        return r
    if r in ("special", "bonus"):
        return "special"
    return "unknown" if not r else "special"


def _row_from_scryfall(name: str, o: dict) -> dict:
    v = _face_view(o)
    pc = v["parsed"]
    types = _norm_types(v["type_line"])
    colors = pc["colors"]
    if not colors and o.get("colors") and "Land" not in types:
        colors = order_colors(o["colors"])
    mv = float(v["cmc"]) if v["cmc"] is not None else float(pc["mv"])
    generic = pc["generic"]
    produces = _produced(o)
    is_basic = name in BASICS or types.startswith("Basic Land")
    if is_basic and name in BASICS:
        produces = BASICS[name]
    f0 = (v["faces"] or [{}])[0]
    pw, tg = o.get("power", f0.get("power")), o.get("toughness", f0.get("toughness"))
    power, tough, pvar = parse_pt(pw, tg)
    kw = o.get("keywords")
    return dict(
        name=name, rarity=_rarity(o.get("rarity"), types, name), colors=colors,
        color_identity=order_colors(o.get("color_identity") or []),
        mana_value=mv, pips=pc["pips"], hybrid=pc["hybrid"], generic=generic, types=types,
        oracle_text=v["oracle"] or "", produces=produces, meta_source="scryfall",
        power=power, toughness=tough, pt_variable=pvar,
        keywords=None if kw is None else ",".join(sorted({str(k).lower() for k in kw})),
    )


def read_cards_csv(paths: list[str | Path], set_code: str | None = None) -> dict[str, dict]:
    """Rows of 17lands cards.csv file(s) keyed by name. Later paths override earlier ones.

    Rows are kept when `expansion` matches the set (or the file has no expansion column).
    """
    out: dict[str, dict] = {}
    for p in paths:
        try:
            with open_binary(p) as raw:
                df = pl.read_csv(raw.read(), infer_schema=False)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot read cards file %s: %s", p, e)
            continue
        if "name" not in df.columns:
            continue
        for r in df.iter_rows(named=True):
            exp = (r.get("expansion") or "").strip().upper()
            if set_code and exp and exp != set_code.upper():
                continue
            out[r["name"]] = r
    return out


def _row_from_17lands(name: str, r: dict) -> dict:
    ident = order_colors((r.get("color_identity") or "").upper())
    types = _norm_types(r.get("types") or "")
    try:
        mv = float(r.get("mana_value") or 0)
    except ValueError:
        mv = 0.0
    is_land = "Land" in types.split(" - ")[0].split()
    pips = [0] * 5
    colors = ""
    generic = 0
    produces = ""
    if is_land:
        produces = ident
    else:
        colors = ident
        for c in ident:
            pips[COLORS.index(c)] = 1
        generic = max(int(round(mv)) - len(ident), 0)
    if name in BASICS:
        produces = BASICS[name]
    return dict(
        name=name, rarity=_rarity(r.get("rarity"), types, name), colors=colors, color_identity=ident, mana_value=mv,
        pips=pips, hybrid=False, generic=generic, types=types, oracle_text="", produces=produces,
        meta_source="17lands",
    )


def _row_inferred(name: str, stats: dict[str, int] | None) -> dict:
    """Last resort: colours from the most common main_colors of decks containing the card."""
    colors = ""
    if stats:
        total = sum(stats.values())
        if total >= 20:
            share = {c: sum(n for mc, n in stats.items() if c in mc) / total for c in COLORS}
            colors = "".join(c for c in COLORS if share[c] >= 0.75)
    basic = name in BASICS
    pips = [0] * 5
    if not basic:
        for c in colors:
            pips[COLORS.index(c)] = 1
    return dict(
        name=name, rarity="basic" if basic else "unknown", colors="" if basic else colors,
        color_identity="" if basic else colors, mana_value=0.0 if basic else float(max(3, len(colors) + 2)),
        pips=pips, hybrid=False, generic=0 if basic else max(3 - len(colors), 0),
        types=f"Basic Land - {name}" if basic else "", oracle_text="", produces=BASICS.get(name, ""),
        meta_source="inferred",
    )


def build_cards_table(
    names: list[str],
    scryfall_cards: list[dict] | None = None,
    cards_csv: dict[str, dict] | None = None,
    color_stats: dict[str, dict[str, int]] | None = None,
) -> pl.DataFrame:
    """One row per 17lands card name (sorted, dense card_id) with the best available metadata."""
    idx = sf.build_index(scryfall_cards or [])
    csv_f = {sf.fold(k): v for k, v in (cards_csv or {}).items()}
    csv_front = {sf.fold(sf.front(k)): v for k, v in (cards_csv or {}).items()}
    rows = []
    for name in sorted(set(names)):
        o = sf.lookup(idx, name) if scryfall_cards else None
        if o is not None:
            d = _row_from_scryfall(name, o)
        else:
            r = (cards_csv or {}).get(name) or csv_f.get(sf.fold(name)) or csv_front.get(sf.fold(sf.front(name)))
            if r is not None:
                d = _row_from_17lands(name, r)
            elif name in BASICS:
                d = _row_inferred(name, None)
            else:
                d = _row_inferred(name, (color_stats or {}).get(name))
        types = d["types"]
        head = types.split(" - ")[0]
        d["is_basic"] = name in BASICS or types.startswith("Basic Land")
        d["is_land"] = d["is_basic"] or ("Land" in head.split())
        d["is_creature"] = "Creature" in head.split()
        if d["is_basic"] and name in BASICS:
            d["produces"] = BASICS[name]
        # a name 17lands wrote with '?' for a lost letter ("Bespoke B?"): show and export Scryfall's spelling ("Bespoke Bō")
        d["front_name"] = sf.front(o["name"]) if (o is not None and "?" in name and "?" not in o.get("name", "")) else sf.front(name)
        d.setdefault("power", None)
        d.setdefault("toughness", None)
        d.setdefault("pt_variable", False)
        d.setdefault("keywords", None)
        rows.append(d)
    if not rows:
        return pl.DataFrame(schema=CARD_SCHEMA)
    data = {
        "card_id": list(range(len(rows))),
        "name": [r["name"] for r in rows],
        "front_name": [r["front_name"] for r in rows],
        "rarity": [r["rarity"] for r in rows],
        "colors": [r["colors"] for r in rows],
        "color_identity": [r["color_identity"] for r in rows],
        "mana_value": [r["mana_value"] for r in rows],
    }
    for i, c in enumerate(COLORS):
        data[f"pip_{c}"] = [r["pips"][i] for r in rows]
    data.update({
        "hybrid": [r["hybrid"] for r in rows],
        "generic": [r["generic"] for r in rows],
        "types": [r["types"] for r in rows],
        "is_land": [r["is_land"] for r in rows],
        "is_basic": [r["is_basic"] for r in rows],
        "is_creature": [r["is_creature"] for r in rows],
        "produces": [r["produces"] for r in rows],
        "oracle_text": [r["oracle_text"] for r in rows],
        "meta_source": [r["meta_source"] for r in rows],
        "power": [r["power"] for r in rows],
        "toughness": [r["toughness"] for r in rows],
        "pt_variable": [r["pt_variable"] for r in rows],
        "keywords": [r["keywords"] for r in rows],
    })
    return pl.DataFrame(data, schema=CARD_SCHEMA)
