"""Sealed Pool Trainer: sample a real top-pilot Sealed pool for blind deckbuilding, then grade against the model & pilot
across the model's bootstrap draws."""
from __future__ import annotations

import json
import logging
import math
import random
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from sealed_lab import config, deck_search
from sealed_lab.types import BuildResult, Cancelled, DeckEval

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]

DEFAULT_MIN_SKILL = 0.62
MIN_POOL_CARDS = 70
TRAINER_MODES = ("standard", "tricky")
TRICKY_DRAW_SHARE = 7 / 8        # #1 (chosen on even draws only) ahead of the pilot in >= 7 of 8 held-out odd draws
CLOSE_DRAW_SHARE = 10 / 16       # <= 10 of 16 draws: within model draw noise of #1
SLIGHT_DRAW_SHARE = 13 / 16      # 11..13 of 16 draws: slight edge to #1; >= 14/16: clear edge
_TRICKY_CACHE_VERSION = 2
_TRICKY_SCAN_CAP = 30            # max candidate pools scanned per deal before stopping


def tricky_draw_halves(n_draws_total: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split bootstrap draws `1..B` (`n_draws_total = B + 1`) into fixed even (selection) and odd (evaluation) halves.
    For the standard 16-replicate model (`n_draws_total == 17`), returns `(2,4,6,8,10,12,14,16)` and `(1,3,5,7,9,11,13,15)`."""
    even = tuple(i for i in range(1, int(n_draws_total)) if i % 2 == 0)
    odd = tuple(i for i in range(1, int(n_draws_total)) if i % 2 == 1)
    return even, odd


def _canon_colors(s: str | None) -> str:
    if not s:
        return ""
    up = str(s).upper()
    return "".join(c for c in "WUBRG" if c in up)


def _tricky_cache_path(set_code: str, fmt: str) -> Path:
    return config.cache_dir(set_code.upper()) / f"trainer_tricky_{fmt}.json"


def _load_tricky_cache(set_code: str, fmt: str) -> tuple[tuple | None, dict[str, dict], set[str]]:
    """Return `(stamp, passed_by_did, rejected_dids)` for `set_code`/`fmt`."""
    stamp = deck_search._model_stamp(set_code.upper())
    if stamp is None:
        return None, {}, set()
    p = _tricky_cache_path(set_code, fmt)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if int(raw.get("version") or 0) != _TRICKY_CACHE_VERSION or tuple(raw.get("stamp") or ()) != stamp:
            return stamp, {}, set()
        passed = {str(k): dict(v) for k, v in (raw.get("passed") or {}).items()}
        rejected = {str(x) for x in (raw.get("rejected") or [])}
        return stamp, passed, rejected
    except (OSError, ValueError):
        return stamp, {}, set()


def _save_tricky_cache(set_code: str, fmt: str, stamp: tuple | None, passed: dict[str, dict], rejected: set[str]) -> None:
    if stamp is None:
        return
    p = _tricky_cache_path(set_code, fmt)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        payload = dict(version=_TRICKY_CACHE_VERSION, stamp=list(stamp), passed=passed, rejected=sorted(rejected))
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        log.debug("could not save trainer tricky cache for %s %s", set_code, fmt, exc_info=True)


def _extract_pilot_pool(idx, cdir: Path, row: dict, basics: set[int]) -> tuple[dict[int, int], dict[int, int], str, str]:
    """Return `(pool_counts, pilot_deck, main_colors, splash_colors)` for one run row."""
    did = row["draft_id"]
    ev = row["event"]
    pdf = pl.scan_parquet(cdir / "pools.parquet").filter(pl.col("draft_id") == did).select("card_id", "n").collect()
    pool_counts = {int(c): int(n) for c, n in pdf.iter_rows() if int(c) not in basics and int(n) > 0}
    if deck_search.is_bo3_event(ev):
        reg = deck_search._bo3_registered_builds(idx).filter((pl.col("event") == ev) & (pl.col("draft_id") == did))
        b_df = reg if reg.height > 0 else idx.builds.filter((pl.col("event") == ev) & (pl.col("draft_id") == did))
    else:
        b_df = idx.builds.filter((pl.col("event") == ev) & (pl.col("draft_id") == did))
    b_df = b_df.sort(["games", "wins", "build_index"], descending=[True, True, False])
    dk = int(b_df["deck_key"][0])
    b_mc = str(b_df["main_colors"][0] or row.get("run_main_colors") or "")
    b_sp = str(b_df["splash_colors"][0] or row.get("run_splash_colors") or "")
    dc = pl.scan_parquet(cdir / "deck_cards.parquet").filter(pl.col("deck_key") == dk).select("card_id", "n").collect()
    pilot_deck = {int(c): int(n) for c, n in dc.iter_rows() if int(n) > 0}
    for c, n in pilot_deck.items():
        if c not in basics:
            pool_counts[c] = max(pool_counts.get(c, 0), n)
    return pool_counts, pilot_deck, b_mc, b_sp


def draw_comparison(a_draws: np.ndarray | list[float], b_draws: np.ndarray | list[float]) -> dict:
    """Compare two decks across the model's bootstrap draws (`1..B`, excluding draw 0)."""
    a = np.asarray(a_draws, dtype=float)
    b = np.asarray(b_draws, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) == 0:
        return dict(n_draws=0, a_beats_b=0, b_beats_a=0, ties=0)
    if len(a) > 1:
        a, b = a[1:], b[1:]
    diff = a - b
    eps = 1e-12
    return dict(
        n_draws=int(len(diff)),
        a_beats_b=int((diff > eps).sum()),
        b_beats_a=int((diff < -eps).sum()),
        ties=int((np.abs(diff) <= eps).sum()),
    )


def draw_subset_comparison(a_draws: np.ndarray | list[float], b_draws: np.ndarray | list[float],
                           indices: tuple[int, ...] | list[int]) -> dict:
    """Compare two decks strictly on the specified draw `indices` (e.g. the 8 held-out odd bootstrap draws)."""
    a = np.asarray(a_draws, dtype=float)
    b = np.asarray(b_draws, dtype=float)
    idxs = [int(i) for i in indices if 0 <= int(i) < min(len(a), len(b))]
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or not idxs:
        return dict(n_draws=0, a_beats_b=0, b_beats_a=0, ties=0)
    diff = a[idxs] - b[idxs]
    eps = 1e-12
    return dict(
        n_draws=int(len(diff)),
        a_beats_b=int((diff > eps).sum()),
        b_beats_a=int((diff < -eps).sum()),
        ties=int((np.abs(diff) <= eps).sum()),
    )


def sample_trainer_pool(set_code: str, fmt: str = "bo3", min_skill: float = DEFAULT_MIN_SKILL,
                        seed: int | None = None, exclude_ids: list[str] | None = None, *,
                        mode: str = "standard", progress: Progress | None = None,
                        cancel: threading.Event | None = None) -> dict:
    """Sample a real, pool-invariant Sealed pool piloted by a strong (`skill >= min_skill`) 17Lands player.

    - `mode="standard"`: random strong-pilot Sealed pool (falls back gracefully on tiny/synthetic corpora).
    - `mode="tricky"`: strong-pilot pool where the model's #1 deck — chosen using ONLY the even bootstrap draws
      `(2, 4, 6, 8, 10, 12, 14, 16)` — has *different* main colors from the pilot's deck and beats the pilot's deck on at least
      7 of the 8 held-out odd draws `(1, 3, 5, 7, 9, 11, 13, 15)` (`>= ceil(7/8 * n_odd)`).  Checked pools are cached under
      `cache_dir(set_code)`, so a later deal can reuse one.
    """
    if mode not in TRAINER_MODES:
        raise ValueError(f"mode must be one of {TRAINER_MODES}")
    sc = set_code.upper()
    idx = deck_search._load(sc)
    if not idx.has_pools:
        raise ValueError(f"{sc} has no Sealed pools in its corpus")

    want_ev = "TradSealed" if fmt == "bo3" else "Sealed"
    fallback_evs = ["TradSealed", "Sealed"] if fmt == "bo3" else ["Sealed", "TradSealed"]

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
    ex_set = set(exclude_ids or ())
    if ex_set:
        r_ex = runs.filter(~pl.col("draft_id").is_in(list(ex_set)))
        if r_ex.height > 0:
            runs = r_ex
    if runs.height == 0:
        raise ValueError(f"no Sealed pools with completed games found in {sc}")

    r_fmt = runs.filter(pl.col("event") == want_ev)
    if r_fmt.height > 0:
        runs = r_fmt

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
    basics = {cid for cid, m in idx.card_meta.items() if m["is_basic"]}
    tricky_info: dict | None = None

    if mode == "tricky":
        from sealed_lab.deck import builder

        model = builder.load_model(sc)
        if fmt not in model.formats:
            raise ValueError(f"{sc} has no {fmt} model for Tricky pools")
        stamp, passed_cache, rejected_cache = _load_tricky_cache(sc, fmt)
        by_did = {r["draft_id"]: r for r in cands}

        # 1. Serve an unseen cached tricky pool immediately if one exists
        cached_hits = [did for did in passed_cache if did in by_did and did not in ex_set]
        chosen = None
        pool_counts: dict[int, int] = {}
        pilot_deck: dict[int, int] = {}
        b_mc, b_sp = "", ""
        if cached_hits:
            rng.shuffle(cached_hits)
            did = cached_hits[0]
            chosen = by_did[did]
            pool_counts, pilot_deck, b_mc, b_sp = _extract_pilot_pool(idx, cdir, chosen, basics)
            tricky_info = dict(passed_cache[did])
        else:
            # 2. Prioritise lower cached model-score pilot decks (more likely to have a different-color best build)
            ms = deck_search.cached_model_scores(sc)
            score_by_did: dict[str, float] = {}
            if ms is not None:
                b_rep = (idx.builds.filter(pl.col("event").is_in(fallback_evs))
                         .sort(["event", "draft_id", "games", "build_index"], descending=[False, False, True, False])
                         .group_by(["event", "draft_id"], maintain_order=True).agg(pl.col("deck_key").first()))
                j_ms = (skilled.select("event", "draft_id", "fmt")
                        .join(b_rep, on=["event", "draft_id"], how="inner")
                        .join(ms, on=["deck_key", "fmt"], how="inner"))
                for did_, ms_ in j_ms.select("draft_id", "model_score").iter_rows():
                    score_by_did[str(did_)] = float(ms_)
            untested = [r for r in cands if r["draft_id"] not in rejected_cache and r["draft_id"] not in passed_cache]
            if not untested:
                untested = list(cands)
            rng.shuffle(untested)
            if score_by_did:
                # Bucket-sort with jitter so lower-scored pilot builds are tried first without always repeating the same order
                untested.sort(key=lambda r: score_by_did.get(r["draft_id"], 55.0) + rng.uniform(-8.0, 8.0))

            even_draws, odd_draws = tricky_draw_halves(model.n_draws(fmt))
            if not even_draws or not odd_draws:
                raise ValueError(f"{sc} {fmt.upper()} has too few bootstrap draws for Tricky pools")
            need_eval = math.ceil(TRICKY_DRAW_SHARE * len(odd_draws))
            dirty = False
            checked = 0
            try:
                for row in untested:
                    if checked >= _TRICKY_SCAN_CAP:
                        break
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    pc, pdeck, mc, sp = _extract_pilot_pool(idx, cdir, row, basics)
                    if sum(pc.values()) < MIN_POOL_CARDS and len(untested) > 5:
                        continue
                    checked += 1
                    if progress is not None:
                        progress(min(0.95, checked / _TRICKY_SCAN_CAP),
                                 f"Looking for a tricky pool (model beats pilot in {need_eval}+ of {len(odd_draws)} held-out draws): "
                                 f"pool {checked} of up to {_TRICKY_SCAN_CAP}")
                    res = builder.build_ids(sc, pc, fmt, n_decks=1, min_diff=10, cancel=cancel, families=True,
                                            draw_subset=even_draws)
                    if not res.decks:
                        rejected_cache.add(row["draft_id"])
                        dirty = True
                        continue
                    best_eval = res.decks[0]
                    best_counts = _deck_counts(best_eval)
                    if _canon_colors(best_eval.colors) == _canon_colors(mc):
                        rejected_cache.add(row["draft_id"])
                        dirty = True
                        continue
                    cmp = draw_subset_comparison(model.logit(best_counts, fmt, draws=True),
                                                 model.logit(pdeck, fmt, draws=True), odd_draws)
                    n_dr = cmp["n_draws"]
                    need = math.ceil(TRICKY_DRAW_SHARE * n_dr) if n_dr > 0 else 1
                    if n_dr > 0 and cmp["a_beats_b"] >= need:
                        info_hit = dict(model_colors=best_eval.colors, pilot_colors=mc,
                                        model_beats_pilot=int(cmp["a_beats_b"]), n_draws=int(n_dr), held_out=True)
                        passed_cache[row["draft_id"]] = info_hit
                        dirty = True
                        chosen = row
                        pool_counts, pilot_deck, b_mc, b_sp = pc, pdeck, mc, sp
                        tricky_info = info_hit
                        break
                    rejected_cache.add(row["draft_id"])
                    dirty = True
            finally:
                if dirty:
                    _save_tricky_cache(sc, fmt, stamp, passed_cache, rejected_cache)
            if chosen is None:
                raise ValueError(f"No tricky pool found in {sc} {fmt.upper()}: none of the pools checked had a different-color "
                                 f"build beating the pilot in {need_eval}+ of {len(odd_draws)} held-out draws. Try Strong pilot mode.")
    else:
        rng.shuffle(cands)
        chosen = None
        pool_counts = {}
        pilot_deck = {}
        b_mc, b_sp = "", ""
        for row in cands[:25]:
            pc, pdeck, mc, sp = _extract_pilot_pool(idx, cdir, row, basics)
            if sum(pc.values()) >= MIN_POOL_CARDS or len(cands) <= 5:
                chosen = row
                pool_counts, pilot_deck, b_mc, b_sp = pc, pdeck, mc, sp
                break
        if chosen is None:
            chosen = cands[0]
            pool_counts, pilot_deck, b_mc, b_sp = _extract_pilot_pool(idx, cdir, chosen, basics)

    did = chosen["draft_id"]
    ev = chosen["event"]
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
        mode=mode,
        tricky_info=tricky_info,
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
                       pilot_meta: dict, vs_best: dict, vs_pilot: dict, *,
                       model_vs_user_draws: dict | None = None,
                       user_vs_pilot_draws: dict | None = None,
                       model_vs_pilot_draws: dict | None = None) -> dict:
    """Plain-language summary metrics for the Trainer Grade banner, graded across bootstrap draws rather than noisy point cut-offs."""
    u_vs_m = 100.0 * (user_eval.game_wr - best_eval.game_wr)
    p_vs_m = 100.0 * (pilot_eval.game_wr - best_eval.game_wr)
    u_vs_p = 100.0 * (user_eval.game_wr - pilot_eval.game_wr)

    m_u = model_vs_user_draws or {}
    u_p = user_vs_pilot_draws or {}
    m_p = model_vs_pilot_draws or {}
    n_draws = int(m_u.get("n_draws") or 0)
    m_beats_u = int(m_u.get("a_beats_b") or 0)
    u_beats_m = int(m_u.get("b_beats_a") or 0)
    u_beats_p = int(u_p.get("a_beats_b") or 0)
    p_beats_u = int(u_p.get("b_beats_a") or 0)
    m_beats_p = int(m_p.get("a_beats_b") or 0)

    if n_draws > 0:
        share = m_beats_u / n_draws
        within_noise = share <= CLOSE_DRAW_SHARE
        if m_beats_u == 0 and u_beats_m == 0:
            verdict = "Same rating as Model #1 across all model draws"
        elif share <= CLOSE_DRAW_SHARE:
            verdict = f"Close to Model #1 (#1 ahead in {m_beats_u} of {n_draws} draws)"
        elif share <= SLIGHT_DRAW_SHARE:
            verdict = f"Slight edge to Model #1 (#1 beats your deck in {m_beats_u} of {n_draws} draws)"
        else:
            verdict = f"Clear edge to Model #1 (#1 beats your deck in {m_beats_u} of {n_draws} draws)"
    else:
        within_noise = abs(u_vs_m) <= 0.5
        verdict = "Graded against Model #1 and the 17Lands pilot"

    return dict(
        verdict=verdict,
        within_noise=within_noise,
        n_draws=n_draws,
        model_beats_user=m_beats_u,
        user_beats_model=u_beats_m,
        user_beats_pilot=u_beats_p,
        pilot_beats_user=p_beats_u,
        model_beats_pilot=m_beats_p,
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


def _deck_logit_draws(api_mod, set_code: str, deck: dict[int, int], fmt: str) -> np.ndarray | None:
    fn = getattr(api_mod, "deck_logit_draws", None)
    if fn is not None:
        try:
            arr = fn(set_code, deck, fmt)
            return None if arr is None else np.asarray(arr, dtype=float)
        except Exception:  # noqa: BLE001
            log.debug("deck_logit_draws failed for %s %s", set_code, fmt, exc_info=True)
            return None
    try:
        from sealed_lab.deck import builder

        model = builder.load_model(set_code)
        return np.asarray(model.logit(deck, fmt, draws=True), dtype=float)
    except Exception:  # noqa: BLE001
        return None


def grade_trainer_build(api_mod, set_code: str, fmt: str, pool: dict[int, int], user_deck: dict[int, int],
                        pilot: dict, max_families: int = 5, progress: Progress | None = None,
                        cancel: threading.Event | None = None) -> dict:
    """Evaluate the user's blind build, the 17Lands pilot's build, and the model's best deck families on `pool`,
    comparing them across all bootstrap draws."""
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

    u_dr = _deck_logit_draws(api_mod, sc, u_clean, fmt)
    b_dr = _deck_logit_draws(api_mod, sc, best_counts, fmt)
    p_dr = _deck_logit_draws(api_mod, sc, p_clean, fmt)
    m_vs_u = draw_comparison(b_dr, u_dr) if (b_dr is not None and u_dr is not None) else None
    u_vs_p = draw_comparison(u_dr, p_dr) if (u_dr is not None and p_dr is not None) else None
    m_vs_p = draw_comparison(b_dr, p_dr) if (b_dr is not None and p_dr is not None) else None

    vs_best = compare_decks(u_clean, best_counts, info)
    vs_pilot = compare_decks(u_clean, p_clean, info)
    summary = grade_summary_line(user_eval, best_eval, pilot_eval, pilot, vs_best, vs_pilot,
                                 model_vs_user_draws=m_vs_u, user_vs_pilot_draws=u_vs_p, model_vs_pilot_draws=m_vs_p)
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
