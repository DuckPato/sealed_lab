"""Sealed Pool Trainer: sample a real top-pilot Sealed pool for blind deckbuilding, then grade against the model & pilot."""
from __future__ import annotations

import random
import threading
from typing import Callable

import polars as pl

from sealed_lab import config, deck_search
from sealed_lab.types import BuildResult, DeckEval

Progress = Callable[[float, str], None]

DEFAULT_MIN_SKILL = 0.62
MIN_POOL_CARDS = 70


def sample_trainer_pool(set_code: str, fmt: str = "bo3", min_skill: float = DEFAULT_MIN_SKILL,
                        seed: int | None = None, exclude_ids: list[str] | None = None) -> dict:
    """Sample a real, pool-invariant Sealed pool piloted by a strong (`skill >= min_skill`) 17Lands player.

    Falls back gracefully (top skill quartile, or any Sealed pool with `games >= 3`) on tiny / synthetic corpora so tests
    and small datasets always return a valid pool.
    Returns JSON-friendly dict with `pool` ({card_id: count}, non-basics), `pool_text` (Arena export), and `pilot` metadata
    (whose `deck` and `record` stay hidden in the UI until Reveal & Grade).
    """
    sc = set_code.upper()
    idx = deck_search._load(sc)
    if not idx.has_pools:
        raise ValueError(f"{sc} has no Sealed pools in its corpus")

    want_ev = "TradSealed" if fmt == "bo3" else "Sealed"
    fallback_evs = ["TradSealed", "Sealed"] if fmt == "bo3" else ["Sealed", "TradSealed"]

    # Restrict to pool-invariant Sealed runs when pool_audit.parquet is present
    cdir = config.corpus_dir(sc)
    inv_ids: set[str] | None = None
    audit_path = cdir / "pool_audit.parquet"
    if audit_path.exists():
        try:
            adf = pl.read_parquet(audit_path)
            if "invariant" in adf.columns and "draft_id" in adf.columns:
                good = adf.filter(pl.col("invariant") == True)["draft_id"].to_list()  # noqa: E712
                if good:
                    inv_ids = set(good)
        except Exception:  # noqa: BLE001
            inv_ids = None

    runs = idx.runs.filter(pl.col("event").is_in(fallback_evs) & (pl.col("games") >= 3))
    if inv_ids is not None:
        r_inv = runs.filter(pl.col("draft_id").is_in(list(inv_ids)))
        if r_inv.height > 0:
            runs = r_inv
    if exclude_ids:
        r_ex = runs.filter(~pl.col("draft_id").is_in(list(exclude_ids)))
        if r_ex.height > 0:
            runs = r_ex
    if runs.height == 0:
        raise ValueError(f"no Sealed pools with completed games found in {sc}")

    # Prefer requested format if available
    r_fmt = runs.filter(pl.col("event") == want_ev)
    if r_fmt.height > 0:
        runs = r_fmt

    # Prefer strong pilots (skill >= min_skill), falling back to top quartile or all runs on small corpora
    skilled = runs.filter(pl.col("skill").is_not_null() & (pl.col("skill") >= float(min_skill)) & (pl.col("games") >= 4))
    if skilled.height == 0:
        with_sk = runs.filter(pl.col("skill").is_not_null())
        if with_sk.height >= 4:
            q75 = float(with_sk["skill"].quantile(0.75) or 0.0)
            skilled = with_sk.filter(pl.col("skill") >= q75)
        else:
            skilled = runs

    rng = random.Random(seed)
    cands = skilled.sort("draft_id").to_dicts()
    rng.shuffle(cands)

    basics = {cid for cid, m in idx.card_meta.items() if m["is_basic"]}
    chosen = None
    pool_counts: dict[int, int] = {}
    for row in cands[:25]:
        did = row["draft_id"]
        pdf = pl.scan_parquet(cdir / "pools.parquet").filter(pl.col("draft_id") == did).select("card_id", "n").collect()
        pc = {int(c): int(n) for c, n in pdf.iter_rows() if int(c) not in basics and int(n) > 0}
        if sum(pc.values()) >= MIN_POOL_CARDS or len(cands) <= 5:
            chosen = row
            pool_counts = pc
            break
    if chosen is None:
        chosen = cands[0]
        pdf = pl.scan_parquet(cdir / "pools.parquet").filter(pl.col("draft_id") == chosen["draft_id"]).select("card_id", "n").collect()
        pool_counts = {int(c): int(n) for c, n in pdf.iter_rows() if int(c) not in basics and int(n) > 0}

    did = chosen["draft_id"]
    ev = chosen["event"]
    # Pilot's primary deck: for Bo3, registered Game-1 deck of Match 1 (or longest-played registered build); for Bo1, longest-played build
    if deck_search.is_bo3_event(ev):
        reg = deck_search._bo3_registered_builds(idx).filter((pl.col("event") == ev) & (pl.col("draft_id") == did))
        b_df = reg if reg.height > 0 else idx.builds.filter((pl.col("event") == ev) & (pl.col("draft_id") == did))
    else:
        b_df = idx.builds.filter((pl.col("event") == ev) & (pl.col("draft_id") == did))
    b_df = b_df.sort(["games", "wins", "build_index"], descending=[True, True, False])
    dk = int(b_df["deck_key"][0])
    b_mc = str(b_df["main_colors"][0] or chosen.get("run_main_colors") or "")
    b_sp = str(b_df["splash_colors"][0] or chosen.get("run_splash_colors") or "")

    dc = pl.scan_parquet(cdir / "deck_cards.parquet").filter(pl.col("deck_key") == dk).select("card_id", "n").collect()
    pilot_deck = {int(c): int(n) for c, n in dc.iter_rows() if int(n) > 0}
    # Ensure every non-basic card in pilot_deck is covered by pool_counts
    for c, n in pilot_deck.items():
        if c not in basics:
            pool_counts[c] = max(pool_counts.get(c, 0), n)

    pool_lines = [f"{n} {idx.card_meta[c]['name']}" for c, n in sorted(pool_counts.items(), key=lambda kv: idx.card_meta[kv[0]]["name"]) if c in idx.card_meta]
    date = chosen.get("date")
    colors_str = b_mc + (f"+{b_sp}" if b_sp else "")
    pilot = dict(
        draft_id=did,
        event=ev,
        event_label=deck_search.event_label(ev),
        deck=pilot_deck,
        colors=colors_str,
        main_colors=b_mc,
        splash=b_sp,
        record=deck_search._record(chosen),
        wins=int(chosen["wins"]),
        losses=int(chosen["losses"]),
        games=int(chosen["games"]),
        trophy=bool(chosen["trophy"]),
        skill=None if chosen.get("skill") is None else float(chosen["skill"]),
        rank=str(chosen.get("rank") or ""),
        date=date.isoformat(sep=" ", timespec="minutes") if hasattr(date, "isoformat") and date is not None else str(date or ""),
    )
    return dict(
        set_code=sc,
        fmt=fmt,
        draft_id=did,
        n_pool_cards=int(sum(pool_counts.values())),
        pool=pool_counts,
        pool_text="\n".join(pool_lines) + "\n",
        pilot=pilot,
    )


def compare_decks(a: dict[int, int], b: dict[int, int], info: dict[int, dict]) -> dict:
    """Card-by-card spell & land comparison between two decks `{card_id: copies}`."""
    shared_spells = 0
    total_a_spells = 0
    total_b_spells = 0
    lands_a = 0
    lands_b = 0
    only_a: list[tuple[str, int]] = []
    only_b: list[tuple[str, int]] = []
    for cid in sorted(set(a) | set(b)):
        na, nb = int(a.get(cid, 0)), int(b.get(cid, 0))
        d = info.get(int(cid), {})
        if d.get("is_land") or d.get("is_basic"):
            lands_a += na
            lands_b += nb
            continue
        total_a_spells += na
        total_b_spells += nb
        shared_spells += min(na, nb)
        nm = str(d.get("name") or f"Card {cid}").split(" // ")[0]
        if na > nb:
            only_a.append((nm, na - nb))
        elif nb > na:
            only_b.append((nm, nb - na))
    only_a.sort(key=lambda kv: (-kv[1], kv[0]))
    only_b.sort(key=lambda kv: (-kv[1], kv[0]))
    return dict(
        shared_spells=shared_spells,
        user_spells=total_a_spells,
        ref_spells=total_b_spells,
        user_lands=lands_a,
        ref_lands=lands_b,
        only_user=[f"{n}x {nm}" if n > 1 else nm for nm, n in only_a],
        only_ref=[f"{n}x {nm}" if n > 1 else nm for nm, n in only_b],
    )


def _deck_counts(ev: DeckEval) -> dict[int, int]:
    out: dict[int, int] = {}
    for c in ev.cards:
        if c.card_id is not None and c.count > 0:
            out[int(c.card_id)] = out.get(int(c.card_id), 0) + int(c.count)
    return out


def grade_summary_line(user_eval: DeckEval, best_eval: DeckEval, pilot_eval: DeckEval,
                       pilot_meta: dict, vs_best: dict, vs_pilot: dict) -> dict:
    """Plain-language summary metrics for the Trainer Grade banner."""
    u_vs_m = 100.0 * (user_eval.game_wr - best_eval.game_wr)
    p_vs_m = 100.0 * (pilot_eval.game_wr - best_eval.game_wr)
    u_vs_p = 100.0 * (user_eval.game_wr - pilot_eval.game_wr)
    if u_vs_m >= -0.25:
        verdict = "Spot-on build — essentially tied with the model's #1 family"
    elif u_vs_m >= -0.75:
        verdict = "Strong build — within less than a percentage point of #1"
    elif u_vs_m >= -1.50:
        verdict = "Solid build — a few card or mana tweaks leave win rate on the table"
    else:
        verdict = "Different direction — another color pair or core shell scores noticeably higher"
    return dict(
        verdict=verdict,
        user_wr=float(user_eval.game_wr),
        model_wr=float(best_eval.game_wr),
        pilot_wr=float(pilot_eval.game_wr),
        user_vs_model_pp=float(u_vs_m),
        pilot_vs_model_pp=float(p_vs_m),
        user_vs_pilot_pp=float(u_vs_p),
        vs_model=vs_best,
        vs_pilot=vs_pilot,
        pilot_record=pilot_meta.get("record", ""),
        pilot_skill=pilot_meta.get("skill"),
        pilot_trophy=bool(pilot_meta.get("trophy")),
    )


def grade_trainer_build(api_mod, set_code: str, fmt: str, pool: dict[int, int], user_deck: dict[int, int],
                        pilot: dict, max_families: int = 5, progress: Progress | None = None,
                        cancel: threading.Event | None = None) -> dict:
    """Evaluate the user's blind build, the 17Lands pilot's build, and the model's best deck families on `pool`."""
    sc = set_code.upper()
    u_clean = {int(k): int(v) for k, v in user_deck.items() if int(v) > 0}
    if not u_clean:
        raise ValueError("Add cards to your deck before grading (aim for a 40-card deck with basic lands).")
    p_clean = {int(k): int(v) for k, v in (pilot.get("deck") or {}).items() if int(v) > 0}
    pool_clean = {int(k): int(v) for k, v in pool.items() if int(v) > 0}

    info = api_mod.card_info(sc)
    user_eval = api_mod.evaluate_deck(sc, u_clean, fmt)
    pilot_eval = api_mod.evaluate_deck(sc, p_clean, fmt)
    build_res: BuildResult = api_mod.build_from_pool(sc, pool_clean, fmt=fmt, n_decks=max_families, min_diff=10,
                                                     progress=progress, cancel=cancel)
    best_eval = build_res.decks[0] if build_res.decks else user_eval
    best_counts = _deck_counts(best_eval)

    vs_best = compare_decks(u_clean, best_counts, info)
    vs_pilot = compare_decks(u_clean, p_clean, info)
    summary = grade_summary_line(user_eval, best_eval, pilot_eval, pilot, vs_best, vs_pilot)
    return dict(
        set_code=sc,
        fmt=fmt,
        pool=pool_clean,
        user_eval=user_eval,
        pilot_eval=pilot_eval,
        pilot=pilot,
        build_result=build_res,
        summary=summary,
    )
