"""One printable page ("cheat sheet") for a set and Sealed format: the best colour pairs, the best commons and uncommons of each colour, the
deck shape of the decks that did well, the play-or-draw note and the cards to pick up or to avoid.

Qt-free and pure: `cheat_sheet_html(...)` only formats what it is given (the set report, `card_info`, `card_values`), so it is the same for the
real and the fake backend.  The page has no external resources (no fonts, scripts or images) except the two links of the 17Lands credit,
and print CSS that keeps it to one portrait page (A4 or Letter).
"""
from __future__ import annotations

import html
from datetime import datetime

from .attribution import credit_html

POINTS_PER_LOGIT = 25.0         # a per-copy log-odds effect (card_values "beta") in points of game win rate near 50 %: 1/4 of a logit, in %
                                # (the same constant as sealed_lab.report.PT, which the Report tab's card values use; a test keeps them equal)
N_COMMONS, N_UNCOMMONS = 5, 3   # per colour
N_PAIRS_MAX = 5                 # most colour pairs listed (the Strong / Likely ones)
N_PAIRS_FALLBACK = 3            # listed when no pair is Strong or Likely
N_LISTED = 6                    # underrated cards to pick up / trap rares to avoid
COLORS = "WUBRG"
COLOR_NAME = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green", "other": "Gold & colourless"}
GROUPS = ["W", "U", "B", "R", "G", "other"]
BADGE_WORD = {"strong": "Strong", "likely": "Likely", "weak": "Weak"}
LOGIT_NOTE = ("Values are the model's raw points of win rate per copy (one point = one percentage point of the chance to win a game), "
              "not measured against the average card: use them to compare cards of the same colour here, not with the two lists below, "
              "which are points against the average card of their colour.  Cards with little data are left out.")


def points(beta: float) -> float:
    """Points of win rate of one copy of a card with per-copy log-odds effect `beta`."""
    return float(beta) * POINTS_PER_LOGIT


def card_group(colors: str | None) -> str:
    """"W".."G" for a one-colour card, "other" for multicolour and colourless cards."""
    k = "".join(c for c in COLORS if c in (colors or ""))
    return k if len(k) == 1 else "other"


def signed(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def top_cards(card_info: dict, card_values: dict) -> dict[str, dict[str, list[dict]]]:
    """{group: {"common": [...], "uncommon": [...]}} with at most N_COMMONS / N_UNCOMMONS cards per colour, best per-copy value first.
    Rows: {card_id, name, pts, colors, rarity}.  Lands, basics and cards with little data (or no value) are left out."""
    out: dict[str, dict[str, list[dict]]] = {g: {"common": [], "uncommon": []} for g in GROUPS}
    rows: list[dict] = []
    for cid, d in card_info.items():
        v = card_values.get(cid)
        if v is None or d.get("is_land") or d.get("is_basic") or v.get("low_data") or d.get("rarity") not in ("common", "uncommon"):
            continue
        rows.append(dict(card_id=cid, name=d["name"], pts=points(v["beta"]), colors=d.get("colors") or "", rarity=d["rarity"]))
    rows.sort(key=lambda r: (-r["pts"], r["name"]))
    limit = {"common": N_COMMONS, "uncommon": N_UNCOMMONS}
    for r in rows:
        lst = out[card_group(r["colors"])][r["rarity"]]
        if len(lst) < limit[r["rarity"]]:
            lst.append(r)
    return out


def best_pairs(report: dict) -> tuple[list[dict], bool]:
    """The colour pairs to name: the Strong / Likely ones with a positive effect (best first, at most N_PAIRS_MAX), else the top
    N_PAIRS_FALLBACK by effect.  Returns (rows, any_clear)."""
    rows = [r for r in (report.get("pairs") or {}).get("rows", []) if not r.get("hidden") and len(r.get("key", "")) == 2]
    rows.sort(key=lambda r: -r["effect_pp"])
    clear = [r for r in rows if r["badge"] in ("strong", "likely") and r["effect_pp"] > 0]
    if clear:
        return clear[:N_PAIRS_MAX], True
    return rows[:N_PAIRS_FALLBACK], False


def _pips(colors: str) -> str:
    return "".join(f'<span class="pip {c}">{c}</span>' for c in colors if c in COLORS)


def _rng(r: dict | None, digits: int = 0) -> str:
    if not r:
        return "–"
    lo, hi = round(r["lo"], digits), round(r["hi"], digits)
    f = (lambda x: f"{x:.{digits}f}")
    return f(lo) if lo == hi else f"{f(lo)}–{f(hi)}"


def shape_lines(report: dict) -> tuple[list[tuple[str, str]], str]:
    """([(label, "16-17 (typical 17)")], basis sentence): the middle half of the decks with good results (all decks when too few of those)."""
    sh = report.get("shape") or {}
    row = next((r for r in sh.get("rows", []) if r.get("key") == "all"), None)
    if row is None:
        return [], ""
    lines: list[tuple[str, str]] = []
    used_strong = False
    for key, label, digits in (("lands", "Lands", 0), ("creatures", "Creatures", 0), ("two_drops", "Two-drops", 0), ("avg_mv", "Average mana value", 1)):
        m = (row.get("metrics") or {}).get(key) or {}
        r = m.get("strong") or m.get("all")
        used_strong = used_strong or bool(m.get("strong"))
        if r:
            lines.append((label, f"{_rng(r, digits)} (typical {r['med']:.{digits}f})"))
    if used_strong:
        basis = f"The middle half of the {row['n_strong']:,} decks that {sh.get('strong_def', 'did well')}."
    else:
        basis = f"The middle half of all {row['n_decks']:,} decks."
    return lines, basis


def play_draw_note(report: dict) -> str:
    op = (report.get("glance") or {}).get("on_play")
    if not op:
        return "No play-or-draw figure for this format."
    d = op["diff_pp"]
    figures = f"{op['wr_play'] * 100:.1f}% of games won on the play against {op['wr_draw'] * 100:.1f}% on the draw ({signed(d)} points)"
    if op.get("badge") == "weak" or abs(d) < 0.5:
        return f"No clear edge either way: {figures}.  Choose by the deck."
    if d > 0:
        return f"Choose to play first: {figures}."
    return f"Choose to draw: {figures}."


def _pair_rows_html(rows: list[dict]) -> str:
    out = []
    for r in rows:
        out.append(f'<tr><td class="pc">{_pips(r.get("colors") or r["key"])}</td><td class="nm">{esc(r["label"])}</td>'
                   f'<td class="num">{signed(r["effect_pp"])} pts</td><td class="bd {esc(r["badge"])}">{esc(BADGE_WORD.get(r["badge"], "Weak"))}</td>'
                   f'<td class="num mu">{r["share"] * 100:.0f}% of decks</td></tr>')
    return "".join(out)


def _card_list_html(title: str, cards: list[dict]) -> str:
    if not cards:
        return f'<div class="sub">{esc(title)}</div><div class="none">none with enough data</div>'
    li = "".join(f'<li><span class="cn">{esc(c["name"])}</span><span class="v">{signed(c["pts"])}</span></li>' for c in cards)
    return f'<div class="sub">{esc(title)}</div><ol>{li}</ol>'


def _picks_html(cards: list[dict], none_text: str) -> str:
    if not cards:
        return f'<div class="none">{esc(none_text)}</div>'
    li = "".join(f'<li><span class="cn">{esc(c["name"])}</span><span class="rar">{esc(c.get("rarity", ""))}</span>'
                 f'<span class="v">{signed(c["value_pp"])}</span></li>' for c in cards)
    return f"<ol>{li}</ol>"


CSS = """
@page { size: portrait; margin: 9mm; }
* { box-sizing: border-box; }
html { color-scheme: light; }
body { margin: 0; background: #fff; color: #1d1d1f; font: 8.4pt/1.25 "Segoe UI", "Helvetica Neue", Arial, sans-serif;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.page { max-width: 192mm; margin: 0 auto; padding: 6mm 0; }
h1 { font-size: 16pt; margin: 0 0 1mm; }
h2 { font-size: 9.6pt; margin: 2.6mm 0 1mm; padding-bottom: .6mm; border-bottom: .3mm solid #888; }
.lead, .none, .note, .foot { color: #555; }
.lead { margin: 0 0 1mm; }
.note { font-size: 7.4pt; margin: 1mm 0 0; }
table { border-collapse: collapse; width: 100%; }
td { padding: .5mm 1.6mm .5mm 0; vertical-align: middle; }
td.pc { width: 14mm; white-space: nowrap; } td.nm { font-weight: 600; } td.num { text-align: right; white-space: nowrap; }
td.mu { color: #555; } td.bd { width: 14mm; font-weight: 600; } td.bd.strong { color: #176a3d; } td.bd.likely { color: #2a5ea8; } td.bd.weak { color: #777; }
.pip { display: inline-block; width: 4.2mm; height: 4.2mm; border-radius: 50%; margin-right: .6mm; text-align: center; font-size: 6.4pt; line-height: 4.2mm;
       font-weight: 700; border: .2mm solid rgba(0,0,0,.45); color: #111; }
.pip.W { background: #f6efc4; } .pip.U { background: #a9d4f2; } .pip.B { background: #b9b0a8; } .pip.R { background: #f2a68d; } .pip.G { background: #a6d6a0; }
.cols { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2mm; }
.col { border: .25mm solid #bbb; border-top-width: 1.2mm; border-radius: 1.2mm; padding: 1mm 1.4mm 1.2mm; min-width: 0; break-inside: avoid; }
.col.W { border-top-color: #e5d98a; } .col.U { border-top-color: #6fb2e0; } .col.B { border-top-color: #6d625b; }
.col.R { border-top-color: #e2764f; } .col.G { border-top-color: #5fb056; } .col.other { border-top-color: #b79b4a; }
.col h3 { font-size: 8.6pt; margin: 0 0 .6mm; }
.sub { font-size: 7pt; text-transform: uppercase; letter-spacing: .03em; color: #666; margin: .8mm 0 .2mm; }
ol { list-style: none; margin: 0; padding: 0; }
li { display: flex; gap: 1.4mm; align-items: baseline; padding: .05mm 0; border-bottom: .15mm dotted #ccc; }
li .cn { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
li .v { font-weight: 600; font-variant-numeric: tabular-nums; } li .rar { color: #777; font-size: 7pt; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; align-items: start; }
.box { border: .25mm solid #bbb; border-radius: 1.2mm; padding: 1mm 2mm 1.4mm; break-inside: avoid; }
.box h3 { font-size: 9pt; margin: 0 0 .8mm; }
dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: .3mm 3mm; } dt { color: #444; } dd { margin: 0; font-weight: 600; }
.foot { margin-top: 2.6mm; padding-top: 1.4mm; border-top: .3mm solid #888; font-size: 7pt; }
.foot a { color: #1f5fb4; }
@media screen { body { background: #eceff3; } .page { background: #fff; margin: 8mm auto; padding: 9mm 10mm; box-shadow: 0 1px 6px rgba(0,0,0,.25); } }
@media print { .page { padding: 0; max-width: none; } }
"""


def cheat_sheet_html(set_code: str, fmt: str, report: dict, card_info: dict, card_values: dict, generated_at: str | None = None) -> str:
    """The page.  `report` = `api.set_report(set, fmt)`; `card_info` = `api.card_info(set)`; `card_values` = `api.card_values(set, fmt)`;
    `generated_at`: the date text of the footer (default: today, ISO); pass it for a reproducible page."""
    code = set_code.upper()
    fmt_name = "Best of 1" if fmt == "bo1" else "Best of 3"
    when = generated_at if generated_at is not None else datetime.now().strftime("%Y-%m-%d")
    gl = report.get("glance") or {}
    basis = (f"Based on {gl['n_decks']:,} real {fmt_name} Sealed decks ({gl['n_games']:,} games) on 17Lands; win rates compare players of the same skill."
             if gl.get("n_decks") else f"{fmt_name} Sealed.")
    pairs, clear = best_pairs(report)
    top = top_cards(card_info, card_values)
    shape, shape_basis = shape_lines(report)
    cs = report.get("cards") or {}
    under = sorted(cs.get("underrated") or [], key=lambda r: -r["value_pp"])[:N_LISTED]
    traps = sorted(cs.get("traps") or [], key=lambda r: r["value_pp"])[:N_LISTED]

    p: list[str] = []
    p.append(f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
             f"<title>{esc(code)} Sealed cheat sheet — {esc(fmt_name)}</title><style>{CSS}</style></head><body><div class=\"page\">")
    p.append(f"<h1>{esc(code)} Sealed cheat sheet — {esc(fmt_name)}</h1><p class=\"lead\">{esc(basis)}</p>")
    p.append("<h2>Best colour pairs</h2>")
    if pairs:
        p.append(f"<table>{_pair_rows_html(pairs)}</table>")
        p.append('<p class="note">Win rate against an average deck, in points.  '
                 + ("Only the pairs with a clear (Strong or Likely) edge are listed." if clear
                    else "No pair stands out clearly in this format: these are the three best, but the differences could be chance.") + "</p>")
    else:
        p.append('<p class="none">Not enough decks for a colour-pair result.</p>')
    p.append("<h2>Best commons and uncommons by colour</h2><div class=\"cols\">")
    for g in GROUPS:
        p.append(f'<div class="col {g}"><h3>{esc(COLOR_NAME[g])}</h3>{_card_list_html("Commons", top[g]["common"])}'
                 f'{_card_list_html("Uncommons", top[g]["uncommon"])}</div>')
    p.append(f'</div><p class="note">{esc(LOGIT_NOTE)}</p>')
    p.append('<h2>Deck shape and play or draw</h2><div class="two"><div class="box"><h3>Target deck shape</h3>')
    if shape:
        p.append("<dl>" + "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in shape) + "</dl>")
        p.append(f'<p class="note">{esc(shape_basis)}</p>')
    else:
        p.append('<p class="none">No deck-shape figures for this format.</p>')
    p.append(f'</div><div class="box"><h3>Play or draw?</h3><p style="margin:0">{esc(play_draw_note(report))}</p></div></div>')
    p.append('<h2>Cards to pick up and cards to avoid</h2><div class="two"><div class="box"><h3>Underrated: pick them up</h3>')
    p.append(_picks_html(under, "Nothing clear enough to list.") + '<p class="note">Players leave them out, but they win (points against the average card of their colour).</p></div>')
    p.append('<div class="box"><h3>Trap rares: avoid</h3>')
    p.append(_picks_html(traps, "No rare or mythic is clearly a trap.") + '<p class="note">Played in most decks that can, yet they lose (points against the average card of their colour).</p></div></div>')
    p.append(f'<div class="foot">{credit_html()}  Generated {esc(when)} by sealed_lab.</div>')
    p.append("</div></body></html>\n")
    return "".join(p)
