"""Export of a set's written report and Analysis (both formats) as ONE self-contained HTML file.

`export_set_html(set_code, written, reports, facts)` is a pure function (no Qt, no file IO, no network): it only formats what the
tabs show, in the same order and the same plain wording.  Inline CSS, readable without JavaScript, printable (Print -> Save as PDF).
A tiny inline script (optional) makes the card table sortable and opens the "More detail" blocks when printing.
"""
from __future__ import annotations

import re
from datetime import datetime
from html import escape
from typing import Any

from sealed_lab.attribution import credit_html
from sealed_lab.report_check import SENTENCE, tags_of

BADGE_TEXT = {"strong": "Strong", "likely": "Likely", "weak": "Weak", "descriptive": "Descriptive"}
FMT_NAME = {"bo1": "Best of 1", "bo3": "Best of 3"}
GROUP_ORDER = ["W", "U", "B", "R", "G", "gold", "colorless", "land"]
GROUP_TITLE = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green", "gold": "Multicolor", "colorless": "Artifacts & colorless",
               "land": "Lands"}
RARITY_TITLE = {"common": "Common", "uncommon": "Uncommon", "rare": "Rare", "mythic": "Mythic"}
Z_LIKELY, Z_STRONG = 2.0, 3.0
NO_REPORT = "No written report yet for this set: the Analysis section has the numbers."
OUTDATED = "Outdated: written for older data; the Analysis section is current."
DISCLAIMER = ("Everything here is observed in 17lands data; the builder does not use archetypes.  Win rates are compared between players of "
              "the same skill and given in points (one point = one percentage point of the chance to win a game).")
BADGE_LEGEND = ("Strong: the result is clear; it would be very surprising if it were just luck.  Likely: probably real, but give it some room.  "
                "Weak: could easily be chance: do not act on it.")

CSS = """
:root{--fg:#1d2330;--mut:#5d6677;--line:#d5d9e2;--bg:#fff;--soft:#f5f6f9;--acc:#2f5fb3;--good:#1b7a3a;--bad:#b3261e;--warn:#9a6a00}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Segoe UI",system-ui,-apple-system,Arial,sans-serif}
main{max-width:900px;margin:0 auto;padding:18px 16px 48px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:21px;margin:34px 0 8px;padding-bottom:4px;border-bottom:2px solid var(--line)}
h3{font-size:17px;margin:26px 0 6px}h4{font-size:15px;margin:16px 0 4px}h5{font-size:14px;margin:10px 0 2px}
p{margin:6px 0}ul{margin:6px 0;padding-left:22px}li{margin:2px 0}
.mut{color:var(--mut)}.small{font-size:13px}.good{color:var(--good)}.bad{color:var(--bad)}.num{text-align:right;white-space:nowrap}
.meta{color:var(--mut);font-size:13px}.note{color:var(--mut);font-size:13px;margin:6px 0}
.banner{border:1px solid var(--warn);border-radius:6px;padding:6px 10px;margin:8px 0;color:var(--warn)}
.legend{background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:8px 12px;margin:12px 0;font-size:13px}
.badge{display:inline-block;border:1px solid var(--line);border-radius:9px;padding:0 7px;font-size:12px;line-height:18px;white-space:nowrap}
.badge.strong{background:#cfdcf5;border-color:var(--acc)}.badge.likely{background:#e7eefb;border-color:#8fa9d8}.badge.weak{color:var(--mut)}
.chip{display:inline-block;border:1px solid var(--line);border-radius:9px;padding:0 8px;font-size:12px;margin:0 4px 4px 0}
.chip.acc{background:#e7eefb;border-color:#8fa9d8}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}
th,td{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;font-size:13px;color:var(--mut)}th.num,td.num{text-align:right}
tr{break-inside:avoid}
.bar{display:inline-block;position:relative;width:150px;height:12px;vertical-align:middle;border-left:1px solid var(--mut)}
.bar i{position:absolute;top:1px;height:10px;border-radius:2px}.bar i.p{background:var(--good)}.bar i.n{background:var(--bad)}.bar i.w{opacity:.4}
.card{display:flex;gap:8px;justify-content:space-between;align-items:baseline}
.groups{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:4px 18px}
.group ul{list-style:none;padding:0}.group li{margin:4px 0}
details{border:1px solid var(--line);border-radius:6px;margin:8px 0;padding:4px 10px}
details>summary{cursor:pointer;font-weight:600}details[open]>summary{margin-bottom:6px}
.sub{border:1px solid var(--line);border-radius:6px;padding:6px 10px;margin:8px 0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.spec{height:10px;border-radius:5px;background:linear-gradient(90deg,#c3c8d4,var(--acc));margin:4px 0}
[title]{text-decoration:underline dotted #9aa3b5;text-underline-offset:3px}
.bar[title],.badge[title],th[title],.chip[title]{text-decoration:none}
@media print{body{font-size:12px}main{max-width:none;padding:0}h2,h3,h4{break-after:avoid}.legend,.sub,details{break-inside:avoid}
table{break-inside:auto}thead{display:table-header-group}a{color:inherit}}
@media (max-width:600px){.two{grid-template-columns:1fr}.bar{width:90px}}
"""

SCRIPT = """
(function(){
 window.addEventListener('beforeprint',function(){document.querySelectorAll('details').forEach(function(d){d.open=true;});});
 document.querySelectorAll('table.sortable').forEach(function(t){
  t.querySelectorAll('th').forEach(function(th,i){th.style.cursor='pointer';th.addEventListener('click',function(){
   var tb=t.tBodies[0],rows=Array.prototype.slice.call(tb.rows),dir=th.dataset.dir=th.dataset.dir==='1'?'-1':'1';
   rows.sort(function(a,b){var x=a.cells[i].dataset.k,y=b.cells[i].dataset.k,nx=parseFloat(x),ny=parseFloat(y);
    var r=(isNaN(nx)||isNaN(ny))?String(x).localeCompare(String(y)):nx-ny;return r*dir;});
   rows.forEach(function(r){tb.appendChild(r);});});});});
})();
"""


# ------------------------------------------------------------------------------------------------ small helpers
def e(x: Any) -> str:
    return escape("" if x is None else str(x), quote=True)


def pts(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def rng_text(r: dict | None, digits: int = 0) -> str:
    if not r:
        return "–"
    lo, hi = round(r["lo"], digits), round(r["hi"], digits)
    f = (lambda x: f"{x:.{digits}f}")
    return f(lo) if lo == hi else f"{f(lo)}–{f(hi)}"


def badge_of(z: float | None) -> str:
    if z is None:
        return "weak"
    a = abs(z)
    return "strong" if a >= Z_STRONG else "likely" if a >= Z_LIKELY else "weak"


def badge(level: str | None, tip: str = "") -> str:
    lv = level if level in ("strong", "likely") else "weak"
    t = f' title="{e(tip)}"' if tip else ""
    return f'<span class="badge {lv}"{t}>{e(BADGE_TEXT.get(lv, "Weak"))}</span>'


def val(x: float, digits: int = 1, suffix: str = "", tip: str = "") -> str:
    t = f' title="{e(tip)}"' if tip else ""
    return f'<span class="{"good" if x >= 0 else "bad"}"{t}><b>{e(pts(x, digits))}{e(suffix)}</b></span>'


def chip(text: str, accent: bool = False, tip: str = "") -> str:
    t = f' title="{e(tip)}"' if tip else ""
    return f'<span class="chip{" acc" if accent else ""}"{t}>{e(text)}</span>'


def muted(text: str) -> str:
    return f'<p class="mut small">{e(text)}</p>'


def note(text: str | None) -> str:
    return f'<p class="note">{e(text)}</p>' if text else ""


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{e(t)}</li>" for t in items) + "</ul>" if items else ""


def table(heads: list[tuple[str, str]], rows: list[list[str]], right: tuple[int, ...] = (), cls: str = "") -> str:
    h = "".join(f'<th{" class=num" if i in right else ""}{f" title={chr(34)}{e(t)}{chr(34)}" if t else ""}>{e(x)}</th>' for i, (x, t) in enumerate(heads))
    body = "".join("<tr>" + "".join(f'<td{" class=num" if i in right else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<table{f" class={chr(34)}{cls}{chr(34)}" if cls else ""}><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def details(summary: str, body: str, open_: bool = False) -> str:
    return f'<details{" open" if open_ else ""}><summary>{summary}</summary>{body}</details>'


def effect_bar(effect: float, scale: float, level: str, tip: str = "") -> str:
    """A bar from the middle (zero) to the effect: green up, red down, pale when Weak; scale = the value at either edge."""
    scale = max(scale, 0.5)
    w = min(50.0, abs(effect) / scale * 50.0)
    left = 50.0 if effect >= 0 else 50.0 - w
    cls = ("p" if effect >= 0 else "n") + (" w" if level == "weak" else "")
    return f'<span class="bar" title="{e(tip)}"><i class="{cls}" style="left:{left:.1f}%;width:{max(w, 1.5):.1f}%"></i></span>'


# ------------------------------------------------------------------------------------------------ written report
def fact_tip(fid: str, facts: dict) -> str:
    f = ((facts or {}).get("facts") or {}).get(fid)
    if f is not None:
        b = BADGE_TEXT.get(f.get("badge", ""), "")
        return f"{f.get('text', fid)}" + (f"  [{b}]" if b else "")
    n = ((facts or {}).get("noclear") or {}).get(fid)
    if n is not None:
        return f"{n.get('text', fid)}  [no clear difference]"
    return fid


def _inline(text: str) -> str:
    t = escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)


def _line_html(line: str, facts: dict) -> str:
    out = []
    for m in SENTENCE.finditer(line):
        s = m.group(0).strip()
        if not s:
            continue
        ids = tags_of(s)
        body = _inline(re.sub(r"<!--.*?-->", "", s).strip())
        if not body:
            continue
        if ids:
            tip = e("\n".join(fact_tip(i, facts) for i in ids))
            body = f'<span title="{tip}">{body}</span>'
        out.append(body)
    return " ".join(out)


def markdown_to_html(text: str, facts: dict) -> str:
    """The small Markdown subset of the written report (headings, bullets, paragraphs, bold / italics); fact tags dropped, each tagged
    sentence a span whose title is the tooltip.  Same logic as the Report tab."""
    parts: list[str] = []
    bl: list[str] = []
    para: list[str] = []

    def flush() -> None:
        nonlocal bl, para
        if bl:
            parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bl) + "</ul>")
        if para:
            parts.append("<p>" + " ".join(para) + "</p>")
        bl, para = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or (line.startswith("<!--") and line.endswith("-->") and not re.sub(r"<!--.*?-->", "", line).strip()):
            flush()
            continue
        h = re.match(r"(#{1,6})\s+(.*)", line)
        if h:
            flush()
            lvl = min(len(h.group(1)) + 2, 5)
            parts.append(f"<h{lvl}>{_inline(h.group(2))}</h{lvl}>")
            continue
        b = re.match(r"(?:[-*+]|\d+[.)])\s+(.*)", line)
        if b:
            if para:
                flush()
            bl.append(_line_html(b.group(1), facts))
            continue
        if bl:
            flush()
        para.append(_line_html(line, facts))
    flush()
    return "".join(parts)


def _written_section(written: dict | None, facts: dict | None) -> str:
    out = ['<section id="written"><h2>The written report</h2>']
    text = (written or {}).get("text")
    if not text:
        out.append(f"<p>{e(NO_REPORT)}</p>")
    else:
        fx = facts if facts is not None else ((written.get("meta") or {}).get("facts") or {})
        if written.get("outdated"):
            out.append(f'<p class="banner">{e(OUTDATED)}</p>')
        out.append(markdown_to_html(text, fx))
        out.append('<p class="note">Hover a sentence to see the figures behind it.</p>')
        wa = (written.get("meta") or {}).get("written_at")
        if wa:
            out.append(f'<p class="meta">Written {e(wa)}.</p>')
    out.append("</section>")
    return "".join(out)


# ------------------------------------------------------------------------------------------------ analysis sections
def _intro(rep: dict) -> str:
    gl = rep["glance"]
    return muted(f"Based on {gl['n_decks']:,} real {FMT_NAME.get(rep['fmt'], rep['fmt'])} Sealed decks ({gl['n_games']:,} games) on 17lands.  "
                 "Win rates are always compared between players of the same skill, and given in points (one point = one percentage point of the chance "
                 "to win a game).  Each estimate carries a badge: Strong, Likely or Weak.")


def _glance(rep: dict) -> str:
    gl = rep["glance"]
    cards = []
    if gl.get("median_turns") is not None:
        cards.append(("Typical game length", f"{gl['median_turns']:.0f} turns", "median, as counted by 17lands"))
    op = gl.get("on_play")
    if op:
        cards.append(("Playing first", pts(op["diff_pp"]) + " pts", f"{op['wr_play'] * 100:.1f}% on the play, {op['wr_draw'] * 100:.1f}% on the draw"))
    if gl.get("mulligan_rate") is not None:
        cards.append(("Games with a mulligan", f"{gl['mulligan_rate'] * 100:.0f}%", "of games start with a mulligan"))
    cards.append(("Decks studied", f"{gl['n_decks']:,}", f"{gl['n_games']:,} games"))
    tbl = table([("", ""), ("", ""), ("", "")], [[f"<b>{e(a)}</b>", e(b), f'<span class="mut">{e(c)}</span>'] for a, b, c in cards])
    tbl = tbl.replace("<thead><tr><th></th><th></th><th></th></tr></thead>", "")
    return f"<h3>The set at a glance</h3>{tbl}{bullets(gl.get('text', []))}{note(rep['notes'].get('glance'))}"


def _pairs(rep: dict) -> str:
    pr = rep["pairs"]
    out = ["<h3>Color pairs: which decks win?</h3>", bullets(pr.get("text", []))]
    rows = sorted([r for r in pr.get("rows", []) if not r["hidden"]], key=lambda r: -r["effect_pp"])
    if rows:
        tight = [r for r in rows if r["se_pp"] < 3]
        scale = max(1.5, max(max(abs(r["effect_pp"]), abs(r["effect_pp"]) + 1.96 * r["se_pp"]) for r in tight) * 1.05) if tight else 3.0
        body = []
        for r in rows:
            body.append([f'<b title="{e(f"{r["n_decks"]:,} decks, {r["n_games"]:,} games")}">{e(r["label"])}</b>', f"{r['share'] * 100:.1f}%",
                         effect_bar(r["effect_pp"], scale, r["badge"], f"{pts(r['effect_pp'])} points; {r['range']}"), val(r["effect_pp"]),
                         badge(r["badge"], f"{r['label']}: {r['range']}."), f"{r['splash_share'] * 100:.0f}%"])
        out.append(table([("Colors", ""), ("Share of decks", "How many of the decks are built in these colors"),
                          ("Win rate against an average deck (points)", "One point = one percentage point of the chance to win a game"),
                          ("", ""), ("How sure?", BADGE_LEGEND), ("Splash", "How often these decks also splash a third color")], body, right=(1, 3, 5)))
    hidden = [r for r in pr.get("rows", []) if r["hidden"]]
    if hidden:
        out.append(note(f"Not shown (fewer than {pr.get('min_decks')} decks): " + ", ".join(r["label"] for r in hidden) + "."))
    out.append(note(rep["notes"].get("pairs")))
    pd = rep.get("pair_details") or {}
    if pd.get("rows"):
        body = []
        for d in pd["rows"]:
            op = d.get("on_play")
            good = [s for s in d.get("splash", []) if s["badge"] != "weak"]
            txt = ", ".join(f"{s['label'].lower()} {pts(s['effect_pp'])} ({BADGE_TEXT[s['badge']]})" for s in good) if good else \
                ("no clear effect" if d.get("splash") else "too few splash decks")
            tip = "; ".join(f"{s['label']}: {pts(s['effect_pp'])} points, {s['range']}, {s['n_decks']} decks ({BADGE_TEXT[s['badge']]})"
                            for s in d.get("splash", [])) or "No splash color with enough decks."
            body.append([f"<b>{e(d['label'])}</b>", val(op["diff_pp"]) if op else "–", badge(op["badge"], f"{d['label']}: {op['range']}.") if op else "",
                         f"{d['median_turns']:.0f} turns", f"{d['mulligan_rate'] * 100:.0f}%", f'<span title="{e(tip)}">{e(txt)}</span>'])
        out.append(details("More detail: each color pair", muted(pd.get("note", "")) + table(
            [("Colors", ""), ("Going first (points)", "Win rate on the play minus on the draw, first games, players of the same skill"),
             ("How sure?", ""), ("Typical game", "Median number of turns, as counted by 17lands"),
             ("Mulligans", "Share of games that start with a mulligan (a plain figure, not adjusted for skill)"),
             ("Splash pay-off (points)", "Win rate of the decks that splash this color against the same pair without a splash. Only Likely or Strong results are shown.")],
            body, right=(1, 3, 4))))
    return "".join(out)


def _card_li(r: dict, long_reason: bool) -> str:
    sub = r["reason"] + (" (Limited data in this format.)" if r.get("thin") else "") if long_reason else \
        f"Played in {r['play_rate'] * 100:.0f}% of decks" + (" · limited data" if r.get("thin") else "")
    tip = f"{pts(r['value_pp'])} points per copy against its color's average played card; {r['range']}."
    return (f'<li><div class="card"><b>{e(r["name"])}</b><span>{val(r["value_pp"], tip=tip)} {badge(r["badge"], f"{r["name"]}: {r["range"]}.")}</span></div>'
            f'<div class="mut small" title="{e(r["reason"])}">{e(sub)}</div></li>')


def _group_grid(by: dict, reason_first: bool) -> str:
    out = ['<div class="groups">']
    for gk in [g for g in GROUP_ORDER + ["all"] if g in by]:
        out.append(f'<div class="group"><h5>{e(GROUP_TITLE.get(gk, ""))}</h5><ul>' + "".join(_card_li(r, reason_first) for r in by[gk]) + "</ul></div>")
    out.append("</div>")
    return "".join(out)


def _explorer(rep: dict) -> str:
    cs = rep["cards"]
    ex = cs["explorer"]
    c = {n: i for i, n in enumerate(ex["cols"])}
    fmt = rep["fmt"]
    heads = [("Card", "Card name"), ("Color", "Color group the value is compared within"), ("Rarity", ""),
             ("Played in", "Share of the decks that could play it (pool and colors) that played at least one copy"),
             ("Value", "Points of win rate per copy against the average card of its color group that decks play (this format)"),
             ("How sure?", "Strong, Likely or Weak"), ("Decks", "Decks that could have played it"), ("Value Bo1", "The card's value in Best of 1, same points"),
             ("Value Bo3", "The card's value in Best of 3, same points (largely borrowed from Best of 1 when there are few games)"),
             ("Bo1 vs Bo3", "Flagged only when the two formats clearly differ (Likely or Strong, with enough data in both)")]
    rows = []
    for r in ex["rows"]:
        v, vo, z, dz = r[c["value_pp"]], r[c["value_other"]], r[c["z"]], r[c["dz"]]
        thin, thin_o = r[c["thin"]], r[c["thin_other"]]
        v1, v3 = (v, vo) if fmt == "bo1" else (vo, v)
        diff = None if (v1 is None or v3 is None) else v3 - v1
        clear = dz is not None and abs(dz) >= Z_LIKELY and not thin and not thin_o
        bd = badge_of(z)
        rank = {"strong": 2, "likely": 1, "weak": 0}[bd] * (1 if (z or 0) >= 0 else -1)
        rar = r[c["rarity"]]
        cells = [(r[c["name"]].lower(), f"<b>{e(r[c['name']])}</b>"), (GROUP_TITLE.get(r[c["group"]], r[c["group"]]), e(GROUP_TITLE.get(r[c["group"]], r[c["group"]]))),
                 (list(RARITY_TITLE).index(rar) if rar in RARITY_TITLE else 9, e(RARITY_TITLE.get(rar, rar))),
                 (r[c["play_rate"]], f"{r[c['play_rate']] * 100:.0f}%"), (v, val(v, 1, " *" if thin else "")), (rank, badge(bd)),
                 (r[c["n_elig"]], f"{r[c['n_elig']]:,}"), (-999 if v1 is None else v1, "–" if v1 is None else val(v1)),
                 (-999 if v3 is None else v3, "–" if v3 is None else val(v3)),
                 (abs(diff) if clear else -1, "–" if diff is None else (f"Bo3 {e(pts(diff))} ({BADGE_TEXT[badge_of(dz)]})" if clear else "no clear difference"))]
        rows.append("<tr>" + "".join(f'<td data-k="{e(k)}"{" class=num" if i in (3, 4, 6, 7, 8) else ""}>{h}</td>' for i, (k, h) in enumerate(cells)) + "</tr>")
    th = "".join(f'<th{" class=num" if i in (3, 4, 6, 7, 8) else ""} title="{e(t)}">{e(h)}</th>' for i, (h, t) in enumerate(heads))
    return details("More detail: all cards", muted(
        f"The {len(ex['rows'])} cards with enough data ({cs.get('min_eligible')} or more decks that could play them and {cs.get('min_exposure')} card-games).  "
        "An asterisk marks limited data.  With JavaScript on, click a column heading to sort; the Bo1 vs Bo3 column names a difference only when the two "
        "formats clearly differ.") + f'<table class="sortable"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _cards(rep: dict) -> str:
    cs = rep["cards"]
    out = ["<h3>Cards: where players misjudge</h3>",
           f"<p>Out of {cs.get('n_cards', 0)} cards, {cs.get('n_shown', 0)} have enough data to judge: at least {cs.get('min_eligible')} decks that could "
           f"have played them and {cs.get('min_exposure')} games with them in the deck.</p>", note(cs.get("note"))]
    for title, key, blurb in (
            ("Underrated: players leave them out, but they win", "underrated",
             "Played less often than most cards, yet clearly better than the average card you would play in that color."),
            ("Overrated: players love them, but they lose", "overrated",
             "Played in most decks that can, yet clearly worse than the average card you would play in that color.")):
        out.append(f"<h4>{e(title)}</h4>{muted(blurb)}")
        by = cs.get(key + "_by") or ({"all": cs[key]} if cs.get(key) else {})
        out.append(_group_grid(by, True) if by else muted("Nothing clear enough to list."))
        extra = cs.get("n_" + key, 0) - sum(len(v) for v in by.values())
        if extra > 0:
            out.append(note(f"{extra} more not shown (each color lists its 3 clearest, up to 5 when the extra ones are Strong)."))
    traps = cs.get("traps") or []
    if traps:
        out.append("<h4>Trap rares: auto-included, but they lose</h4>")
        out.append(muted("Rares and mythics that most players put in the deck without a second thought, yet clearly worse than the average card you would play in that color."))
        out.append('<div class="groups"><div class="group"><ul>' + "".join(_card_li(r, True) for r in traps) + "</ul></div></div>")
    for rar, title in (("common", "Best commons by color"), ("uncommon", "Best uncommons by color")):
        out.append(f"<h4>{e(title)}</h4>{_group_grid((cs.get('top') or {}).get(rar, {}), False)}")
    out.append(note((rep["notes"].get("cards") or "") + "  Values are points of win rate per copy, compared with the average card of the same color that decks play."))
    if (cs.get("explorer") or {}).get("rows"):
        out.append(_explorer(rep))
    return "".join(out)


def _signature(cards: list[dict], label: str, ref: str = "the other decks", prefix: str = "Signature cards: ") -> str:
    if not cards:
        return ""
    items = ", ".join(f'<b title="{e(f"In {c["rate"] * 100:.0f}% of {label} decks, against {c["ref_rate"] * 100:.0f}% of {ref}.")}">{e(c["name"])}</b>' for c in cards)
    return f'<p>{f"<span class=mut>{e(prefix)}</span>" if prefix else ""}{items}</p>'


def _best_cards(cards: list[dict]) -> str:
    if not cards:
        return ""
    lis = "".join(f'<li title="{e(r["reason"])}"><b>{e(r["name"])}</b> {val(r["value_pp"])} {badge(r["badge"], f"{r["name"]}: {r["range"]}.")}</li>' for r in cards)
    return muted("Best cards in these decks (points of win rate per copy, against the average card of their color):") + f'<ul style="list-style:none;padding-left:0">{lis}</ul>'


def _chips(shape: dict, words: list[str] | None = None) -> str:
    tip = "Typical (median) deck of this group"
    return "<p>" + "".join(chip(w, True, "How this style differs from the other decks of its color pair") for w in words or []) + \
        chip(f"{shape['lands']:.0f} lands", tip=tip) + chip(f"{shape['creatures']:.0f} creatures", tip=tip) + \
        chip(f"{shape['two_drops']:.0f} two-drops", tip=tip + "; two-drops are creatures that cost 2") + \
        chip(f"average mana value {shape['avg_mv']:.1f}", tip=tip + "; mana value of the non-land cards") + \
        chip(f"{shape['splash_share'] * 100:.0f}% splash", tip="Share of these decks that splash a third color") + "</p>"


def _spectrum(a: dict) -> str:
    sp = a.get("spectrum") or {}
    if not sp.get("shown"):
        return ""
    ea, eb, win = sp["a"], sp["b"], sp["win"]
    ends = []
    for en, other in ((ea, eb), (eb, ea)):
        tip = f"{en['description'].capitalize()}.  The {en['n_decks']:,} decks at this end (a third of the {a['key']} decks).  " \
              f"{pts(en['effect_pp'])} points of win rate against an average deck ({en['range']}); {BADGE_TEXT.get(en['badge'], 'Weak').lower()}."
        ends.append(f'<div><b title="{e(tip)}">{e(en["label"])}</b>{_signature(en["signature"], en["label"], ref=f"the {other["label"]} decks", prefix="")}</div>')
    halves = " and ".join(pts(h) if h is not None else "n/a" for h in win["halves"])
    wtip = (f"Top third minus bottom third of the spectrum: {pts(win['effect_pp'])} points of win rate ({win['range']}), skill-adjusted.  "
            f"Fitted separately in the two halves of the data: {halves}.  Shown as a finding only if it is Likely or Strong and has the same direction in both halves.")
    return (f'<div class="sub"><p>{e(sp["text"][0])}</p><div class="two">{ends[0]}{ends[1].replace("<div>", "<div style=text-align:right>", 1)}</div>'
            f'<div class="spec" title="{e(f"From {ea["label"]} (left) to {eb["label"]} (right).")}"></div>'
            f'<p>{e(sp["text"][1])} {badge(win["badge"] if win["finding"] else "weak", wtip)}</p></div>')


def _arch_body(a: dict, label: str, words: list[str] | None = None) -> str:
    return _chips(a["shape"], words) + _signature(a.get("signature", []), label) + _best_cards(a.get("best_cards", []))


def _pair_block(a: dict) -> str:
    subs = a["subs"]
    tip = (f"{a['n_decks']:,} decks.  {a['splash_share'] * 100:.0f}% splash a third color"
           + (f", most often {a['splash_top_label'].lower()}" if a.get("splash_top_label") else "")
           + (f".  Written {a['code']}: " + ", ".join(f"{a['splash_col_shares'][c] * 100:.0f}% splash {c.lower()}" for c in a["splash_cols"]) if a.get("splash_cols") else "") + ".")
    summ = (f'<span title="{e(tip)}">{e(a["code"])}  {e(a["label"])}</span> '
            + (f'<span class="mut small">{len(subs)} styles</span> ' if subs else "")
            + f'<span class="mut small">{a["share"] * 100:.1f}% of decks</span> {val(a["effect_pp"], 1, " pts", f"{pts(a["effect_pp"])} points of win rate against an average deck; {a["range"]}.")} '
            + badge(a["badge"], f"{a['label']}: {a['range']}."))
    body = [bullets(a.get("text", [])), _spectrum(a), _arch_body(a, a["key"])]
    for sub in subs:
        head = f'<b title="{e("Found again in two independent halves of the events (agreement " + format(a["split"]["ari"], ".2f") + ", 1.0 = identical groups). Only styles that replicate this way are shown.")}">{e(sub["name"])}</b> ' \
               f'<span class="mut small">{sub["share_of_pair"] * 100:.0f}% of {e(a["key"])} decks</span>'
        if sub.get("effect_pp") is not None:
            head += f' {val(sub["effect_pp"], 1, " pts", f"{pts(sub["effect_pp"])} points of win rate against an average deck; {sub["range"]}.")} {badge(sub["badge"], f"{sub["name"]}: {sub["range"]}.")}'
        body.append(f'<div class="sub">{head}<p>{e(sub.get("text", ""))}</p>{_arch_body(sub, sub["name"], sub.get("words"))}</div>')
    return details(summ, "".join(body))


def _archetypes(rep: dict) -> str:
    ar = rep["archetypes"]
    out = ["<h3>Archetypes: how each color pair is played</h3>",
           "<p>Each block is a main-color pair, with its share of decks and its win rate against an average deck (players of the same skill). "
           "A plus sign after the colors (UB+r) names a color that at least a quarter of its decks splash. Inside a block, the style spectrum "
           "shows how the decks of that pair range between two styles (only when the same range turns up in two independent halves of the data).  "
           "Click a block to open it.</p>"]
    if ar.get("n_with_subs"):
        out.append(f"<p>{ar['n_with_subs']} of the pairs split into clearly different styles (shown inside the block). A style is only shown if the same "
                   "groups turn up again in two independent halves of the data.</p>")
    else:
        out.append("<p>No color pair splits into clearly different styles in this format: the decks of a pair vary smoothly (for example, more removal "
                   "and a lower curve against more creatures and fliers) rather than falling into separate groups.</p>")
    out.extend(_pair_block(a) for a in ar["pairs"])
    out.append(note(ar.get("note") or rep["notes"].get("archetypes")))
    return "".join(out)


def _effect_table(rows: list[dict], first: str) -> str:
    shown = [r for r in rows if not r.get("hidden")]
    tight = [r for r in shown if r["se_pp"] < 3]
    scale = max(1.5, max(max(abs(r["effect_pp"]), abs(r["effect_pp"]) + 1.96 * r["se_pp"]) for r in tight) * 1.05) if tight else 3.0
    body = [[e(r["label"]), f"{r['share'] * 100:.0f}%", effect_bar(r["effect_pp"], scale, r["badge"], f"{pts(r['effect_pp'])} points; {r['range']}; {r['n_decks']:,} decks"),
             val(r["effect_pp"]), badge(r["badge"], f"{r['label']}: {r['range']}.")] for r in shown]
    return table([(first, ""), ("Share of decks", ""), ("Win rate against an average deck (points)", ""), ("", ""), ("How sure?", BADGE_LEGEND)], body, right=(1, 3))


def _shape(rep: dict) -> str:
    sh = rep["shape"]
    metrics = sh["metrics"]
    heads = [("Colors", ""), ("Decks", "")] + [(m["label"], "") for m in metrics]
    body = []
    for row in sh["rows"]:
        a_cells, s_cells = [], []
        for m in metrics:
            d = row["metrics"][m["key"]]
            dg = 1 if m["key"] == "avg_mv" else 0
            a_cells.append(f'<span title="{e(f"{row["label"]}: the middle half of decks play {rng_text(d["all"], dg)} (typical: {d["all"]["med"]:.{dg}f}).")}">{e(rng_text(d["all"], dg))}</span>' if d["all"] else "–")
            s_cells.append(f'<span class="mut small" title="{e(f"Decks with strong results ({sh["strong_def"]}): the middle half play {rng_text(d["strong"], dg)} (typical: {d["strong"]["med"]:.{dg}f}).")}">{e(rng_text(d["strong"], dg))}</span>'
                           if d["strong"] else '<span class="mut small">too few</span>')
        body.append([f"<b>{e(row['label'])}</b>", f'<span title="{e(f"{row["n_decks"]:,} decks, {row["n_strong"]:,} of them with strong results ({sh["strong_def"]})")}">{row["n_decks"]:,}</span>'
                     f'<br><span class="mut small">strong results: {row["n_strong"]:,}</span>'] + [f"{a}<br>{s}" for a, s in zip(a_cells, s_cells)])
    out = ["<h3>Winning deck shape: what decks look like</h3>", bullets(sh.get("text", [])), table(heads, body, right=tuple(range(2, 2 + len(metrics)))),
           note("First line of each group: the middle half of all its decks.  Second line, smaller: the middle half of the decks with strong results."),
           note(sh.get("note") or rep["notes"].get("shape"))]
    det = sh.get("details")
    if det:
        inner = [bullets(det.get("text", []))]
        for dim, title in (("lands", "By land count"), ("creatures", "By number of creatures")):
            if det.get(dim):
                inner.append(f"<h5>{e(title)}</h5>{_effect_table(det[dim], 'Deck type')}")
        inner.append(muted(det.get("note", "")))
        out.append(details("More detail: win rate by land count and creatures", "".join(inner)))
    return "".join(out)


def _evolution(rep: dict) -> str:
    ev = rep["evolution"]
    out = ["<h3>How the format changed</h3>"]
    if ev.get("too_few"):
        out.append(f"<p>Not enough events ({ev.get('n_drafts', 0):,}; at least {ev.get('min_drafts', 0):,} are needed) to compare the start and the end of the format.</p>")
        return "".join(out)
    out.append("<p>The events are cut by date into an early, a middle and a late third.  A color pair \"got better\" or \"got worse\" only when the change "
               "between the early and the late third is Likely or Strong and points the same way in two independent halves of the events; otherwise it is called stable.</p>")
    out.append(bullets(ev.get("text", [])))
    pairs = ev.get("pairs", [])
    if pairs:
        th = ev["thirds"]
        body = []
        for p in pairs:
            word = {"better": "got better", "worse": "got worse"}.get(p["verdict"], "stable")
            chg = pts(p["change"]) if p["verdict"] == "stable" else val(p["change"])
            body.append([f"<b>{e(p['label'])}</b>"] + [e(pts(p["effect"][c])) for c in range(3)] +
                        [chg if p["verdict"] != "stable" else e(chg), badge(p["change_badge"], f"{p['label']}: change {p['range']}"),
                         f'<span class="{"mut" if p["verdict"] == "stable" else ""}" title="{e(f"Change in the two halves of the data: {pts(p["halves"][0])} and {pts(p["halves"][1])} points.")}">{e(word)}</span>'])
        out.append(table([("Colors", ""), ("Early", f"{th[0]['from']} to {th[0]['to']}"), ("Middle", f"{th[1]['from']} to {th[1]['to']}"),
                          ("Late", f"{th[2]['from']} to {th[2]['to']}"), ("Change (points)", "Late minus early"), ("How sure?", BADGE_LEGEND),
                          ("Over the format", "Better or worse only when Likely or Strong and the same in both halves of the data")], body, right=(1, 2, 3, 4)))
        out.append(note("Each figure is the pair's win rate against an average deck of that third (players of the same skill); the average deck itself shifts as the format changes."))
    cs = ev.get("cards") or {}
    if cs.get("up") or cs.get("down"):
        out.append("<h4>Cards whose play rate changed clearly</h4><div class=\"two\">")
        for key, title in (("up", "Played more by the end"), ("down", "Played less by the end")):
            lis = []
            for r in cs.get(key, []):
                word = {True: "moved toward its measured value", False: "moved away from its measured value", None: ""}[r.get("learned")]
                chg = f"{r['change_pp']:+.0f} pts".replace("-", "−")
                lis.append(f'<li><div class="card"><b>{e(r["name"])}</b><span><span class="{"good" if r["change_pp"] > 0 else "bad"}"><b>{e(chg)}</b></span> '
                           f'{badge(r["badge"], f"{r["name"]}: {r["range"]}.")}</span></div>'
                           f'<div class="mut small">{r["pr_early"] * 100:.0f}% → {r["pr_late"] * 100:.0f}% of decks{" · " + e(word) if word else ""}</div></li>')
            out.append(f'<div class="group"><h5>{e(title)}</h5>' + ("<ul>" + "".join(lis) + "</ul>" if lis else muted("None clear enough.")) + "</div>")
        out.append("</div>")
        out.append(note(f"Play rate = share of the decks that could play the card.  Listed: moved by at least {int(cs['min_change'] * 100)} points, Likely or Strong, "
                        f"the same direction in both halves of the data, at least {cs['min_elig']} eligible decks at each end.  \"Toward its measured value\": a card "
                        "players first misjudged (Likely or Strong) whose play rate moved toward its true value."))
    out.append(note(ev.get("note", "")))
    return "".join(out)


def _compare(rep: dict) -> str:
    cmp = rep["compare"]
    inner = [bullets(cmp.get("text", []))]
    rows = sorted(cmp.get("pairs", []), key=lambda r: -r["bo1"])
    if rows:
        body = [[f"<b>{e(r['label'])}</b>", e(pts(r["bo1"])), e(pts(r["bo3"])), val(r["diff_effect_pp"]) if r["diff_badge"] != "weak" else e(pts(r["diff_effect_pp"])),
                 badge(r["diff_badge"], f"{r['label']}: difference {r['diff_range']}")] for r in rows]
        inner.append(table([("Colors", ""), ("Best of 1", "Win rate against an average deck, points"), ("Best of 3", "Win rate against an average deck, points"),
                            ("Difference", "Best of 3 minus Best of 1"), ("How sure?", BADGE_LEGEND)], body, right=(1, 2, 3)))
    cards = cmp.get("cards") or []
    inner.append("<h5>Cards that clearly differ between the formats</h5>")
    if not cards:
        inner.append(muted("No card clearly differs (cards with limited data in either format are left out)."))
    else:
        lis = []
        for r in cards:
            cap = (f"Per copy: {pts(r['v_bo1'])} points in Best of 1, {pts(r['v_bo3'])} in Best of 3 "
                   f"(against the average {GROUP_TITLE.get(r['group'], r['group']).lower()} card decks play).")
            lis.append(f'<li title="{e(cap)}"><div class="card"><b>{e(r["name"])}</b><span><span class="mut small">Bo1 {e(pts(r["v_bo1"]))}   Bo3 {e(pts(r["v_bo3"]))}</span> '
                       f'{val(r["diff_pp"])} {badge(r["badge"], f"{r["name"]}: Best of 3 minus Best of 1")}</span></div></li>')
        inner.append('<ul style="list-style:none;padding-left:0">' + "".join(lis) + "</ul>")
    inner.append(muted("A difference is called clear only when it is Likely or Strong; Best of 3 card values lean on Best of 1 where games are few, so a real gap can hide."))
    return "<h3>Best of 1 against Best of 3</h3>" + details("More detail: both formats side by side", "".join(inner))


def _data_notes(rep: dict) -> str:
    n = rep["data_notes"]
    return ("<h3>About these numbers</h3>"
            f"<p>This analysis rests on {n['n_games']:,} {FMT_NAME.get(rep['fmt'], rep['fmt'])} games of {n['n_decks']:,} Sealed decks ({n['n_drafts']:,} events, "
            f"{e(n['date_from'])} to {e(n['date_to'])}) on 17lands.  A card is judged only with at least {n['min_eligible']} decks that could play it and "
            f"{n['min_exposure']} games with it in the deck (below {n['thin_exposure']:,} games it is marked limited data); a color pair needs {n['min_pair_decks']} "
            f"decks, a splash color {n['min_splash_decks']} and a deck-shape band {n['min_band_decks']}.  Strong means it would be very surprising if it were luck, "
            "Likely means probably real, Weak means no clear difference: only Strong and Likely results are stated as findings.</p>")


def _format_section(fmt: str, rep: dict | None) -> str:
    name = FMT_NAME.get(fmt, fmt)
    head = f'<section id="{e(fmt)}"><h2>Analysis: {e(name)}</h2>'
    if not rep:
        return head + f"<p>Not computed: the {e(name)} analysis of this set has not been calculated yet.</p></section>"
    if rep.get("too_few"):
        return head + "<p>Not enough games in this format yet for a report.</p></section>"
    parts = [_intro(rep), _glance(rep), _pairs(rep), _cards(rep)]
    if rep.get("archetypes", {}).get("pairs"):
        parts.append(_archetypes(rep))
    if rep.get("shape", {}).get("rows"):
        parts.append(_shape(rep))
    if rep.get("evolution"):
        parts.append(_evolution(rep))
    if rep.get("compare"):
        parts.append(_compare(rep))
    if rep.get("data_notes"):
        parts.append(_data_notes(rep))
    return head + "".join(parts) + "</section>"


# ------------------------------------------------------------------------------------------------ the page
def export_set_html(set_code: str, written: dict | None, reports: dict[str, dict | None], facts: dict | None = None,
                    generated_at: datetime | None = None) -> str:
    """The whole export as one HTML document.  `written` = api.written_report(set) ({text, meta, outdated}); `reports` = {fmt: stored report or None};
    `facts` = the facts JSON of the written report (default: the one stored in its meta)."""
    code = set_code.upper()
    fmts = [f for f in ("bo1", "bo3") if f in reports]
    live = [reports[f] for f in fmts if reports[f] and not reports[f].get("too_few")]
    dn = [r.get("data_notes") or {} for r in live]
    froms = sorted(d["date_from"] for d in dn if d.get("date_from"))
    tos = sorted(d["date_to"] for d in dn if d.get("date_to"))
    built = sorted({str(r["built_at"]) for r in live if r.get("built_at")})
    gen = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    meta = [f"Formats: {', '.join(FMT_NAME.get(f, f) for f in fmts) or 'none'}"]
    meta.append(f"Data: {froms[0]} to {tos[-1]}" if froms and tos else "Data range: not available")
    meta.append(f"Model built: {built[-1]}" if built else "Model build date: not available")
    meta.append(f"Generated: {gen}")
    toc = ['<a href="#written">Written report</a>'] + [f'<a href="#{f}">{e(FMT_NAME.get(f, f))}</a>' for f in fmts]
    body = [f"<header><h1>{e(code)} Sealed report</h1>", f'<p class="meta">{" · ".join(e(m) for m in meta)}</p>', f'<p class="meta credit">{credit_html()}</p>',
            f'<p class="meta">{e(DISCLAIMER)}</p>', f'<p class="meta">{" · ".join(toc)}</p>', f'<div class="legend"><b>How sure?</b>  {e(BADGE_LEGEND)}  Hover over a badge or a figure for the range the true value probably lies in.</div></header>',
            _written_section(written, facts)]
    body.extend(_format_section(f, reports.get(f)) for f in fmts)
    return ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{e(code)} Sealed report</title><style>{CSS}</style></head><body><main>{''.join(body)}</main><script>{SCRIPT}</script></body></html>\n")
