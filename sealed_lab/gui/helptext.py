"""Plain-language tooltips for every number the GUI shows (one place, so the builder, the Details tab and the Deck score tab
say the same thing).  Each text answers: what is it, how is it computed, how do I read it.

Tooltips are rich text in a fixed-width table (Qt only wraps rich-text tooltips reliably that way)."""
from __future__ import annotations

from html import escape

TIP_WIDTH = 400


def tip(*paras: str, width: int = TIP_WIDTH) -> str:
    """Rich-text tooltip: first paragraph bold (title), the others normal; `<br>` allowed in the paragraphs."""
    body = "".join((f"<p style='margin:0 0 5px 0'><b>{p}</b></p>" if i == 0 else f"<p style='margin:0 0 5px 0'>{p}</p>")
                   for i, p in enumerate(paras) if p)
    return f"<html><table width='{width}'><tr><td>{body}</td></tr></table></html>"


def _pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def fmt_name(fmt: str) -> str:
    return {"bo1": "Bo1 Sealed", "bo3": "Bo3 Traditional Sealed"}.get(fmt, fmt)


# --------------------------------------------------------------------------- headline numbers

def level_text(level: dict | None) -> str:
    """Paragraph about the absolute level error of the model's win rates (`api.calibration_note`), "" when unknown."""
    if not level or level.get("level_pp") is None:
        return ""
    pp = float(level["level_pp"])
    lo, hi = level.get("lo"), level.get("hi")
    ci = f" (95% interval {lo:+.1f} to {hi:+.1f})" if lo is not None and hi is not None else ""
    since = f" (events from {str(level['holdout_start'])[:10]})" if level.get("holdout_start") else ""
    if level.get("significant"):
        return ("<b>Absolute level:</b> on the newest events" + since + f" the absolute level was {'under' if pp > 0 else 'over'}-"
                f"predicted by {abs(pp):.1f} pp{ci}: players on average won {abs(pp):.1f} pp {'more' if pp > 0 else 'less'} than the "
                "model, which was fitted on older events.  Read the game win rate, match win rate and expected wins as "
                f"{abs(pp):.1f} pp {'too low' if pp > 0 else 'too high'} on average; the 80% interval does not include this.  "
                "Rankings, scores and differences between decks are not affected.")
    return (f"<b>Absolute level:</b> on the newest events{since} the model's level was within {abs(pp):.1f} pp{ci} of what players "
            "achieved (no significant drift).")


def game_wr_tip(fmt: str = "bo3", ref_wr: float | None = None, wr: float | None = None, has_support_issue: bool = False,
                level: dict | None = None) -> str:
    ref = ""
    if ref_wr is not None:
        diff = "" if wr is None else f" This deck is {100 * (wr - ref_wr):+.1f} pp relative to it."
        ref = (f"<b>Reference:</b> the average real {fmt_name(fmt)} deck scores {_pct(ref_wr)} on this scale.{diff}  "
               "Compare with <i>that</i>, not with 50%: the scale is not centred on 50%.")
    return tip(
        "Game win rate",
        f"<b>What:</b> the chance of winning a single game with this 40-card deck, in {fmt_name(fmt)}, if it is played by an "
        "<b>average 17lands pilot</b> of this format (skill-neutral), 50/50 on the play or the draw, against the typical mix "
        "of opponents.  It is the deck's strength, not your personal win rate: a stronger or weaker player wins more or less.",
        "<b>How:</b> a statistical model fitted on real 17lands games.  Win chance (log-odds) = a baseline + the value of each "
        "card + deck structure (land count, creatures, curve, colours) + a mana simulation (screw / flood / colour screw).",
        "<b>Interval (80%):</b> how much the estimate would move if the model were refitted on resampled events (16 bootstrap "
        "refits by default).  It measures model uncertainty about the deck, not luck in a single game or match.",
        ref,
        level_text(level),
        "<b>Careful:</b> the estimate is only trustworthy for decks that look like real ones."
        + (" This deck does not (see the warning above)." if has_support_issue else ""))


def match_wr_tip(fmt: str = "bo3") -> str:
    if fmt != "bo3":
        return tip("Match win rate", "Bo1 has no match: one game decides, so the game win rate is the number to look at.")
    return tip(
        "Match win rate (best of three)",
        "<b>What:</b> the chance of winning a Bo3 match, from the game win rate p: <b>p² (3 − 2p)</b> (win 2 of 3 games, "
        "assuming every game has the same chance p).",
        "<b>Read:</b> it exaggerates differences: +1 pp of game win rate is about +1.4 pp of match win rate near 60%.  "
        "Sideboarding and play/draw order are ignored, so treat it as a conversion of the game number, not an independent estimate.")


def score_tip(fmt: str = "bo3") -> str:
    return tip(
        "Score (0–100)",
        f"<b>What:</b> where this deck ranks among the real 17lands {fmt_name(fmt)} decks: the percentage of real decks whose "
        "estimated game win rate is lower.  <b>50 = the median real deck</b>, 80 = better than 80% of real decks.",
        "<b>How:</b> the model's game win rate of this deck is compared with the model's win rates of all real decks of the "
        "format (skill-neutral on both sides, so pilot skill does not move the score).",
        "<b>Interval (80%):</b> the score you would get if the model were refitted on resampled events.",
        "<b>Read:</b> it is a ranking, not a win chance, and it only compares estimates of the same model.  Scores near "
        "the extremes (below 5, above 95) compare the deck with very few real decks, so they are the least certain.")


def delta_tip(support: list[str] | None = None, base_label: str = "the generated build") -> str:
    parts = [
        "Change vs the generated build",
        f"<b>What:</b> game win rate of the deck as it is now minus the game win rate of {base_label} (the deck this tab "
        "started from), in percentage points (pp).  Green = your edits helped, red = they hurt.",
        "<b>Read:</b> both numbers come from the same model, so the cards the two decks share cancel out and the difference "
        "is more reliable than either win rate alone, but it still carries the uncertainty of the cards you swapped.  "
        "Differences below about 0.5 pp are within the noise of the model."]
    if support:
        parts.append("<b style='color:#c62828'>Caution:</b> this deck is outside what real decks look like, so the estimate "
                     "(and this difference) is <b>less reliable</b>: " + "; ".join(escape(s) for s in support) + ".")
    return tip(*parts)


def support_tip() -> str:
    return tip(
        "Why is the estimate less reliable?",
        "The model learned from real 17lands decks.  When a deck is unlike them (for example an unusual land count, "
        "many colours, or a colour with almost no sources) the model has to extrapolate, and its win rate can be "
        "too optimistic or too pessimistic.  Prefer decks that stay inside the usual range, or treat the number as a rough guide.")


def exp_wins_tip(fmt: str = "bo3") -> str:
    wins, losses = (7, 3) if fmt == "bo1" else (4, 2)
    return tip(
        "Expected wins per run",
        f"<b>What:</b> average number of match wins in one {fmt_name(fmt)} run (it ends at {wins} wins or {losses} losses).",
        "<b>How:</b> exact calculation from the match win rate (Bo3) or game win rate (Bo1), assuming every game or match "
        "has the same win chance.  Real runs also depend on opponents getting stronger as you win.")


def p_max_tip(fmt: str = "bo3") -> str:
    wins = 7 if fmt == "bo1" else 4
    return tip("P(max wins)", f"<b>What:</b> chance that a run with this deck reaches the maximum of {wins} wins.",
               "<b>How:</b> same calculation as the expected wins.  It is very sensitive to the win rate: a small change "
               "of the game win rate moves it a lot.")


def pbest_tip() -> str:
    return tip("P(best)",
               "<b>What:</b> the share of the model's bootstrap refits in which this deck family's best deck beats every "
               "other candidate deck.",
               "<b>Read:</b> it is not a probability of winning.  It steps in multiples of 1/(number of refits), is coarse, "
               "and depends on how many candidate decks the search found.  A low P(best) with a small gap means the "
               "model cannot tell the decks apart: pick on other grounds.")


# --------------------------------------------------------------------------- counts, curve, colours

COUNTS_TIP = tip(
    "Deck counts",
    "Cards in the deck / 40, lands, creatures and other (non-creature) spells.  The model was fitted on 40-card decks.  "
    "Real decks play 16–17 lands most of the time: other counts push the deck outside the data (see the warning).")

CURVE_TIP = tip(
    "Mana curve",
    "Number of spells (lands excluded) by mana value: ≤1, 2, … 7+.  Blue = creatures, grey = other spells.  "
    "Hover a bar for the numbers.",
    "A smooth curve with several 2- and 3-drops is what real winning decks have: the data show that casting a spell on "
    "turns 2, 3 and 4 is worth a lot.  The model already prices the curve in the win rate.")

PIPS_TIP = tip(
    "Pips",
    "Coloured mana symbols in the costs of the deck's <b>spells</b>, one per symbol and copy (a card costing 1WW counts 2 "
    "white pips).  Lands do not count.  More pips of a colour = you need more sources of that colour, and early.")

SOURCES_TIP = tip(
    "Sources",
    "Cards in the deck that can produce this colour of mana: every land that taps for it (a dual land counts once for "
    "<i>each</i> of its colours), plus mana-producing creatures and artifacts.  A card is counted once per copy.",
    "<b>Read:</b> as a rule of thumb a main colour wants about 8–9 sources in a 40-card deck and a second colour about "
    "6–7; a splash is cast with 3–4.  The mana simulation and the 'castable' percentages on the cards say what "
    "your mana really allows.")

BASICS_TIP = tip(
    "Basics",
    "Number of basic lands of this colour in the deck.  Add or remove them with the − / + buttons above the deck, "
    "double-click a basic in the 'Basic lands' strip below the deck, or drag it into the deck.")


def color_row_tip(color_name: str, pips: int, sources: int, basics: int) -> str:
    return tip(f"{color_name}",
               f"{pips} {color_name.lower()} pips in the spells · {sources} sources · {basics} basic lands of this colour.",
               "See the column titles for what each number means.")


def curve_bar_tip(label: str, creatures: int, other: int) -> str:
    return f"Mana value {label}: {creatures + other} spell(s) — {creatures} creature(s), {other} other"


SAMPLE_HAND_TIP = tip(
    "Sample hand",
    "Deals a random 7-card opening hand from the deck exactly as it is in the builder now (basic lands included), "
    "with London mulligans and draws, so you can feel how it plays.  Bo1 uses Arena's hand smoother as the mana simulation "
    "models it; Bo3 is a plain shuffle.")
SAMPLE_HAND_BO1_TIP = tip(
    "Bo1 (smoothed)",
    "The number of lands in the 7 cards follows the same distribution as the mana simulation's Bo1 Sealed model "
    "(hands with very few or very many lands are rarer than in a plain shuffle); which cards they are is random.")
SAMPLE_HAND_BO3_TIP = tip("Bo3 (plain shuffle)", "A plain random 7 cards, as in Traditional queues.")
SAMPLE_HAND_MULL_TIP = tip(
    "London mulligan",
    "Draws 7 new cards, then you click as many cards as you have mulliganed (1 for a mulligan to 6) to put on the bottom "
    "of the library, and confirm.")
SAMPLE_HAND_DRAW_TIP = tip("Draw", "Next turn: the turn counter goes up and the top card of the library is drawn "
                           "(no draw on turn 1 on the play).  Up to turn 10.")
SAMPLE_HAND_SUMMARY_TIP = tip(
    "Castable now",
    "Rough check from the cards in your hand: a spell counts if you hold at least its mana value in lands and, for each "
    "colour in its cost, at least as many lands that make that colour as it has coloured symbols.  Mana rocks, hybrid "
    "costs and X are ignored.")
SAMPLE_HAND_NOTE = ("Draws are plain random (the simulator's mild land-balancing of draws is not applied).  "
                    "One random hand says little: deal many.")


# --------------------------------------------------------------------------- mana simulation

def mana_regime_note(fmt: str, smoothing_modelled: bool) -> str:
    if fmt == "bo1":
        if smoothing_modelled:
            return ("<b>Bo1 note:</b> Arena's Bo1 queues smooth the opening hand (hands with very few or very many lands "
                    "are rare); this simulation includes that.")
        return ("<b>Bo1 note:</b> Arena's Bo1 queues smooth the opening hand (hands with 0–1 or 6–7 lands are far rarer "
                "than in a plain shuffle).  This simulation does <b>not</b> model that, so in Bo1 it overstates screw, "
                "flood and mulligans.  Compare decks with each other; do not read the levels as forecasts.")
    return ("<b>Bo3 note:</b> Traditional queues use plain shuffles, which is what the simulation does.  Its levels are "
            "approximate (it tends to overstate screw by a few pp); differences between decks are what count.")


def mana_tip(key: str, fmt: str = "bo3", smoothing_modelled: bool = False) -> str:
    regime = mana_regime_note(fmt, smoothing_modelled)
    common = ("<b>How:</b> Monte-Carlo simulation of thousands of shuffles of this exact deck (lands, colours, costs) with a "
              "keep / mulligan rule fitted to real games.  Shown in % of games.")
    t = {
        "p_screw": ("Mana screw",
                    "<b>What:</b> share of games with fewer than 3 lands in play at the end of turn 3 (you missed land drops).",
                    "<b>Read:</b> lower is better.  It is driven mostly by the land count: one more land lowers screw and raises flood."),
        "p_flood": ("Flood",
                    "<b>What:</b> share of games in which 9 or more of the first 16 cards you see are lands.",
                    "<b>Read:</b> lower is better.  Flooding wastes draws; more lands lower screw but raise flood."),
        "p_color_screw": ("Colour screw",
                          "<b>What:</b> among the spells you could afford by mana value (turns 2–6), the share whose "
                          "colours you could not produce.",
                          "<b>Read:</b> lower is better.  High for two-colour decks with uneven sources and for splashes; "
                          "fix it with more sources of the missing colour."),
        "mana_spent": ("Mana spent",
                       "<b>What:</b> mana actually spent on turns 1–7 as a share of the maximum possible (28 = 1+2+…+7).",
                       "<b>Read:</b> higher means the deck uses its mana more fully (good curve and enough lands).  "
                       "It is not good or bad alone: a very low number means wasted turns."),
        "mull_rate": ("Mulligan rate",
                      "<b>What:</b> share of games in which the opening seven is mulliganed (too few or too many lands "
                      "under the keep rule).",
                      "<b>Read:</b> lower is better.  Every mulligan costs a card.  In real games recent sets show roughly 10% in Bo1 "
                      "and about 20% in Bo3 (Bo3 players keep fewer marginal hands)."),
    }[key]
    return tip(t[0], t[1], common, t[2], regime)


# --------------------------------------------------------------------------- win-rate breakdown

BREAKDOWN_LABELS = {"intercept": "Baseline", "cards": "Cards", "structure": "Structure", "mana": "Mana",
                    "synergy": "Synergies", "archetype": "Archetype fit", "mulligans": "Mulligans", "uncastable": "Dead cards"}

_BREAKDOWN_TIPS = {
    "intercept": ("Baseline",
                  "The win rate of a deck with no card, structure or mana effect at all for an average pilot: where the "
                  "model starts.  It is not a contribution of your deck; the other bars add to it."),
    "cards": ("Cards",
              "The combined effect of the deck's non-basic cards (the sum of their values).  It is the biggest lever and the "
              "most data-hungry part: cards with few games lean on the prior."),
    "structure": ("Structure",
                  "Land count, creature / spell mix, mana curve and colour structure (number of colours, splash)."),
    "mana": ("Mana",
             "Effect of the simulated mana problems (screw, flood, colour screw, mana spent, mulligans) on the win rate."),
    "mulligans": ("Mulligans",
                  "The win-rate cost of the mulligans THIS deck makes: how often its land count leads to a mulligan (one or "
                  "two, from the mana simulation with the keep rule players really use) times what a mulligan costs in the "
                  "data.  The baseline is the win rate of a deck that never mulligans, so this bar is always negative."),
    "uncastable": ("Dead cards",
                   "Effect of cards the deck's mana often cannot cast on time (the model's dead-cards term).  Each card's "
                   "own value is already scaled by its chance to be castable."),
    "synergy": ("Synergies", "Effect of card pairs that win more (or less) together than their separate values predict."),
    "archetype": ("Archetype fit", "Effect of cards being better or worse in this deck's archetype than on average."),
}


def breakdown_tip(key: str, logodds: float, pp: float, total_note: str = "") -> str:
    title, what = _BREAKDOWN_TIPS.get(key, (key, ""))
    return tip(
        f"{title}: {pp:+.1f} pp" if key != "intercept" else f"{title}",
        what,
        f"<b>In numbers:</b> {logodds:+.3f} log-odds, shown as percentage points (pp) of game win rate.  The bars are applied "
        "one after the other, starting from 50%: the Baseline bar gets from 50% to the baseline win rate, each further bar "
        "adds its pp, and the last total is this deck's win rate.  (Log-odds add up exactly; pp depend on the order, "
        "because the same log-odds move the win rate less near 0% or 100%.)",
        total_note)


BREAKDOWN_TITLE_TIP = tip(
    "Win-rate breakdown",
    "Where the game win rate comes from, in percentage points (pp).  The model works in log-odds, which add up exactly; "
    "they are converted here to pp so that the bars read as win rate.  Hover a bar for its log-odds.")


# --------------------------------------------------------------------------- explanation, swaps, badges, families

WATERFALL_TIP = tip(
    "Why this deck",
    "From the <b>average real build</b> of the format to this deck: the biggest card effects, then nonbasic lands, "
    "synergies, land count, creature / spell mix, curve, colours and the mana simulation.  Each bar is a step in game "
    "win rate (pp) from the win rate reached so far; the steps add up exactly to the gap between the first and last row.",
    "<b>Card steps</b> compare the card with an <i>average spell slot</i> (n × (card value − average card value)), "
    "so they differ from the value badges, which compare the card with being absent.  "
    "Hover a row for the log-odds and the parts.")


def waterfall_row_tip(label: str, kind: str, d_pp: float, d_logit: float, wr_after: float, detail: list[tuple[str, float]]) -> str:
    if kind == "start":
        return tip(label, f"Game win rate of the average real build of this format: {100 * wr_after:.1f}% (model estimate, "
                          "average pilot).  The steps below start here.")
    if kind == "end":
        return tip(label, f"Game win rate of this deck: {100 * wr_after:.1f}%.  The steps above add up to the gap from the "
                          "average build.")
    what = {"card": "Effect of this card: its value compared with an average spell slot, times the copies.",
            "cards": "Together, the remaining cards (the biggest are listed below).",
            "lands": "Value of the nonbasic lands compared with the average real build.",
            "synergy": "Effect of card pairs / archetype fit compared with the average real build.",
            "structure": "Effect of deck structure compared with the average real build."}.get(kind, "")
    txt = [f"{label}: {d_pp:+.2f} pp", what,
           f"<b>In numbers:</b> {d_logit:+.3f} log-odds; the win rate after this step is {100 * wr_after:.1f}%."]
    if detail:
        txt.append("<b>Parts (log-odds):</b><br>" + "<br>".join(f"{escape(n)}: {v:+.3f}" for n, v in detail))
    return tip(*txt)


def _p_better_paras(n_refits: int | None = None) -> tuple[str, str]:
    n = n_refits or 16
    return (f"<b>P(better):</b> the share of the model's {n} bootstrap refits (the model refitted on resampled events) in "
            "which this swap raises the deck's win rate.",
            f"<b>Read:</b> it is not a probability of being right.  With {n} refits it moves in steps of {100 / n:.1f}% and "
            "saturates at 100%; 94% (15/16) is stronger evidence than 60%.  A swap with a tiny gain can still show 100% "
            "if all refits agree it is slightly positive: look at the pp gain as well.")


def p_better_tip(n_refits: int | None = None) -> str:
    return tip("P(better)", *_p_better_paras(n_refits))


def swap_tip(gain_pp: float, p: float | None, n_refits: int | None = None) -> str:
    return tip("Suggested swap",
               f"<b>Gain:</b> {gain_pp:+.2f} pp of game win rate after taking out the first card and putting in the second "
               "(everything re-evaluated: curve, colours and the mana simulation).",
               "Only swaps from your sideboard, one card for one card, are considered.  The <b>Apply</b> button makes the "
               "swap (undo with Undo).",
               *(_p_better_paras(n_refits) if p is not None else ()))


def value_badge_tip(pp: float, low: bool, in_deck: bool, mode: str = "avg", castable: float | None = None) -> str:
    """Tooltip text for the value badge of a card: `mode` 'avg' (vs average) or 'var' (above replacement).

    `castable` (the card's castability in this deck, when the model has the castability layer) makes the text describe the
    number exactly: value x castable + the dead-cards term x (1 - castable)."""
    if mode == "var":
        if in_deck:
            what = (f"<b>Value above replacement {pp:+.2f} pp:</b> game win rate of the deck minus the win rate of the deck "
                    "with the best replacement from your sideboard in this card's place.  Positive = nothing in the "
                    "sideboard does better; negative = a sideboard card is better here.")
        else:
            what = (f"<b>Best slot {pp:+.2f} pp:</b> the gain in game win rate from putting this card into the slot where it "
                    "helps most (replacing the card that hurts least).  Positive = worth a swap.")
    else:
        if castable is not None:
            how = (f"  <b>How it is computed:</b> the model counts a copy as its card value × the chance this deck's mana can cast "
                   f"it on time ({castable * 100:.0f}% here) plus the model's penalty for a card that sits dead in hand × (1 − "
                   f"{castable * 100:.0f}%); that own contribution of one copy is converted to win-rate points at this deck's win "
                   "rate.  The other cards' castability and the rest of the deck are held fixed, so a swap (which changes both) "
                   "can differ: use 'above replacement' for swaps.")
        else:
            how = "  <b>How it is computed:</b> the card's value for one copy, converted to win-rate points at this deck's win rate."
        if in_deck:
            verb = "helps" if pp >= 0 else "hurts"
            what = (f"<b>Value {pp:+.2f} pp:</b> this card {verb} the deck by {abs(pp):.2f} pp of game win rate: taking one "
                    f"copy out would change the win rate by {-pp:+.2f} pp, everything else equal.  Positive = keep it." + how)
        else:
            what = (f"<b>Value {pp:+.2f} pp:</b> change of the game win rate if one copy were added to the deck, everything "
                    "else equal (it ignores what you would take out; see 'above replacement' for swaps)." + how)
    low_txt = ("  <b style='color:#a35f00'>Yellow badge = little data:</b> few games with this card in this format, so the value "
               "leans on the Bo1 / draft prior and can be off by a lot.") if low else ""
    return what + low_txt


def castable_tip(p: float) -> str:
    return (f"<b>Castable on time: {p * 100:.0f}%.</b> Frank Karsten's measure, from the mana simulation: the chance that the "
            "deck has the <i>coloured</i> sources this card needs on its turn (mana value) and two turns later (averaged), "
            "assuming you have hit your land drops. Sources include dual lands, fixers (rocks, treasures, any-colour "
            "spells) and landcyclers.  <b>How to read it:</b> on-colour cards in real decks are usually 90-95%; a "
            "splash card is typically 60-80%.  A warning mark appears below 65%: the card will often be stuck in "
            "hand.  <b>Where it enters:</b> the win-rate model counts every copy as castable × card value, plus a penalty for each "
            "dead card (1 − castable), so the card's Value (pp) is computed exactly that way (for a sideboard card: as if added "
            "to this deck, with its castability in this deck).")


FAMILY_HELP = {
    "head": tip("Deck family",
                "A family is a group of near-identical decks (sharing about 16 or more of their ~23 spells).  The model's "
                "card values are estimates, and each bootstrap refit would pick a slightly different best deck.  Decks "
                "picked by the refits are grouped into families; the deck shown is the family's best under the full-data "
                "fit.  A family is listed if it is #1, or its P(best) is at least 5%, or its gap to #1 is at most 1 pp, "
                "or the gap interval reaches 0 (at most 5 families)."),
    "pbest": pbest_tip(),
    "gap": tip("Gap vs #1",
               "<b>What:</b> win rate of this family's best deck minus that of #1, under the full-data fit, with an 80% "
               "interval over the bootstrap refits and the share of refits in which this family beats #1.",
               "<b>Read:</b> if the interval includes 0 the model cannot say which is better.  Red = this family is behind "
               "#1 in essentially all refits."),
    "core": tip("Core cards",
                "Cards that appear in at least 90% of this family's best decks over the refits: the stable heart of the deck."),
    "flex": tip("Flex slots",
                "Slots whose card changes from refit to refit: the model sees little difference between the cards "
                "listed.  'in 65%' = share of the family's best decks that contain the card; 'instead' = what else got "
                "picked, with its share.  Choose by feel, or by what the opponent plays.  These cards carry the 'flex' "
                "mark in the deck."),
    "why": tip("Why shown",
               "The reason this family is listed (best under the full-data fit, P(best) ≥ 5%, gap ≤ 1 pp, or its gap "
               "interval reaches 0)."),
    "extra": tip("Also chosen in some refits",
                 "Other cards that showed up occasionally in this family's best decks."),
}

ARCH_TIP = tip(
    "Archetype",
    "<b>What:</b> the cluster of real 17lands decks (same colour pair, similar cards) that this deck most resembles.  "
    "'Similarity' is the cosine similarity of the card composition (1 = same as the cluster's typical deck).",
    "<b>Win rate:</b> what decks of that cluster actually won in this format, shrunk towards the format average, with a "
    "95% interval over events.  It describes decks that look like yours <i>and their pilots</i>; it is not a causal effect "
    "of the archetype and not this deck's estimate.",
    "<b>Signature cards</b> you do not play are common in that cluster's decks.")

SYNERGY_TIP = tip(
    "Synergies",
    "Pairs of cards that win more (or less) together than their single-card values predict.  Only pairs whose sign is "
    "supported by the data (P ≥ 90% of the refits) and that are worth ≥ 0.25 pp are listed; the effect is the change of "
    "this deck's game win rate.  'P 93%' = share of refits with the same sign; 'games' = games with both cards.")

PLAN_TIP = tip(
    "Sideboard plan (Bo3)",
    "Against which opponent colours would you swap cards after sideboarding?  A card is marked only if it is reliably "
    "better against that opponent than on average (P ≥ 90%, ≥ 0.5 pp) and the swap raises the win rate against that "
    "opponent by ≥ 0.5 pp in ≥ 90% of the bootstrap refits.  Several opponents are tested, so a few false positives "
    "are expected: treat it as a hint.")

COMPARE_TIP = tip(
    "Compare with",
    "Difference between this deck and another build of the same pool: win rate (pp), match win rate, score and the "
    "cards that differ.  Cards marked + (green) are only in your deck, − only in the other.")


def chip_tip(opp_lines: list[str]) -> str:
    return tip("Sideboard card vs specific opponents",
               "Bring this card in against these opponent colours: it is reliably better against them than on average, "
               "and swapping it in raises the win rate against them (P(better) = share of refits where it does).",
               "<br>".join(escape(x) for x in opp_lines))


BASIC_STRIP_TIP = tip(
    "Basic lands (unlimited)",
    "You have as many basic lands as you like.  Double-click a basic, or drag it into the deck, to add one; "
    "double-click one in the deck, drag it here or use the − button to remove one.  The number under each is how many "
    "are in the deck.  Everything is re-evaluated at once; Undo works as for other moves.")


# --------------------------------------------------------------------------- Pool tab: format, guessed names

ARENA_WATCH_HELP = tip(
    "Watch Arena for new sealed pools",
    "While this is on (it needs 'Detailed Logs (Plugin Support)' in Arena: Options > Account), the app checks Arena's Player.log "
    "every couple of seconds and notices when a new Sealed pool is written. If the pool box is empty the pool is imported at once; "
    "otherwise a bar offers it and nothing is overwritten until you click Import.",
    "A pool opened up to two hours before the app started is offered too; older events are not. Nothing is shown when the log is missing.")

FORMAT_BO1_LABEL = "Bo1 — Arena Sealed: single games, no sideboarding"
FORMAT_BO3_LABEL = "Bo3 — Traditional Sealed: best-of-three, sideboard between games"

FORMAT_HELP = tip(
    "Which format are you playing?",
    "<b>Bo1 (Arena Sealed):</b> one game per match, no sideboarding; Arena's hand smoother is on and your opponent is picked by "
    "record.  <b>Bo3 (Traditional Sealed):</b> best of three with a sideboard between games and a plain shuffle.",
    "The model fits card values <b>separately for each format</b>: Bo1 decks face a faster, more aggressive field and get no "
    "sideboard, so cheap proactive cards are worth more there; Bo3 rewards consistency and sideboard plans.  The best deck "
    "can therefore differ between the two.",
    "The format preselected for a set is the one with <b>more data</b> for that set (its estimates are tighter); the other is "
    "available when the set has a model for it.")

GUESS_HELP = tip(
    "Card names that were guessed",
    "These lines of your list are not exact card names of the set.  For each, the closest card name was used so the build "
    "can go ahead (the match score is the similarity of the two names, 100% = identical).  Check the guess: pick another "
    "of the five closest names, or 'remove this card'.  Changing a guess makes the current result out of date: build again.")


def set_status_tip(state: str, detail: str = "") -> str:
    """Tooltip of a status in the set list (state: ready | pending | queued | running | failed | untracked)."""
    texts = {
        "ready": "The model exists and was computed from exactly the files that are tracked now.",
        "pending": "New or changed data since the model was built (or the data were never used).  Press Calculate to refit.",
        "queued": "Waiting in the calculation queue: sets are calculated one after another.",
        "running": "Being calculated right now.",
        "failed": "The last calculation of this set failed; click the row for the error.  The full traceback is in logs/errors.log.",
        "untracked": "The app does not track this set yet: pick a folder with its 17lands files, or download them.",
    }
    return tip(state.capitalize() if state != "pending" else "Recalculation pending", texts.get(state, ""), detail)
