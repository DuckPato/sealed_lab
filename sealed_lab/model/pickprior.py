"""Cold-start pick-order prior (LUCE5) for early-format sets (< 40,000 draft games; docs/cold_start/RESULT.md).

When a newly released set has fewer than `COLD_START_DRAFT_GAMES` (40,000) draft games and registered 17Lands
`draft_data_public.<SET>.{PremierDraft,TradDraft}` pick files, conditional-logit pick strengths on picks 1-5 (`LUCE5`)
are mapped onto the zero-mean draft card effects via exposure-weighted rarity-group intercepts + a common slope and used
as the draft stage's card prior mean. Mature sets (>= 40,000 draft games) are untouched in `auto` mode.
"""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from sealed_lab.ingest import detect, registry
from sealed_lab.model import cardvalue, design, validate

log = logging.getLogger(__name__)

COLD_START_DRAFT_GAMES = 40_000
PICK_EVENTS = ("PremierDraft", "TradDraft")
EARLY_PICKS = 5
MIN_TAKEN = 10
MIN_GROUP = 6
LUCE_RIDGE = 0.1
MAX_PICK_DRAFTS = 3_500        # pick drafts read when the game data shares no draft ids (as tested: docs/cold_start/RULE.md)
MIN_SIGNAL_CARDS = 15          # covered cards needed before the pick signal joins the rarity intercepts (as tested)


def find_pick_files(set_code: str) -> list[Path]:
    """Registered 17Lands PremierDraft / TradDraft pick files (`kind == 'draft'`) that exist on disk."""
    found: dict[str, Path] = {}
    for f in registry.files(set_code.upper(), include_shared=False):
        if f.kind == "draft" and (f.event in PICK_EVENTS or not f.event) and Path(f.path).exists():
            key = f.event or Path(f.path).name
            found[key] = Path(f.path)
    ordered = [found[k] for k in PICK_EVENTS if k in found]
    ordered += [p for k, p in sorted(found.items()) if k not in PICK_EVENTS]
    return ordered


def _pack_csr(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.asarray(mat)
    rows, cols = np.nonzero(mat)
    cnt = mat[rows, cols].astype(np.int8)
    offsets = np.zeros(mat.shape[0] + 1, dtype=np.int64)
    if mat.shape[0]:
        np.cumsum(np.bincount(rows, minlength=mat.shape[0]), out=offsets[1:])
    return offsets, cols.astype(np.int16), cnt


def read_early_picks(paths: list[Path], allowed_draft_ids: set[str] | None = None,
                     max_time_iso: str | None = None) -> tuple[dict, list[str]] | None:
    """Read picks 1..EARLY_PICKS from `paths`, filtering to `allowed_draft_ids` (or <= `max_time_iso` up to MAX_PICK_DRAFTS)."""
    names_out: list[str] = []
    name_idx: dict[str, int] = {}
    kept_drafts: set[str] = set()
    taken_list: list[int] = []
    offsets_list: list[int] = [0]
    cards_list: list[np.ndarray] = []
    counts_list: list[np.ndarray] = []
    total_entries = 0

    for path in paths:
        try:
            import pyarrow as pa
            import pyarrow.csv as pacsv

            hdr_rows = detect.read_head_rows(path, 1)
            if not hdr_rows:
                continue
            hdr = hdr_rows[0]
            pack_cols = [c for c in hdr if c.startswith("pack_card_")]
            if not pack_cols:
                continue
            for c in pack_cols:
                nm = c[len("pack_card_"):]
                if nm not in name_idx:
                    name_idx[nm] = len(names_out)
                    names_out.append(nm)
            gidx = np.array([name_idx[c[len("pack_card_"):]] for c in pack_cols], dtype=np.int16)
            meta = [c for c in ("draft_id", "draft_time", "pick_number", "pick") if c in hdr]
            if "pick_number" not in meta or "pick" not in meta:
                continue
            col_types = {"draft_id": pa.string(), "draft_time": pa.string(), "pick_number": pa.int16(), "pick": pa.string(),
                         **{c: pa.int8() for c in pack_cols}}
            with detect.open_binary(path) as stream:
                reader = pacsv.open_csv(stream, read_options=pacsv.ReadOptions(block_size=16 << 20),
                                        convert_options=pacsv.ConvertOptions(include_columns=meta + pack_cols, column_types=col_types))
                while True:
                    try:
                        batch = reader.read_next_batch()
                    except StopIteration:
                        break
                    if batch.num_rows == 0:
                        continue
                    pnum = batch.column("pick_number").to_numpy(zero_copy_only=False)
                    did = batch.column("draft_id").to_pylist() if "draft_id" in meta else [""] * batch.num_rows
                    dt = batch.column("draft_time").to_pylist() if "draft_time" in meta else [None] * batch.num_rows
                    sel = []
                    for r_i in range(batch.num_rows):
                        pn = pnum[r_i]
                        if pn is None or int(pn) + 1 > EARLY_PICKS:
                            continue
                        d = did[r_i] or ""
                        t_s = dt[r_i]
                        if allowed_draft_ids is not None and d in allowed_draft_ids:
                            kept_drafts.add(d)
                            sel.append(r_i)
                        elif d and d in kept_drafts:
                            sel.append(r_i)
                        elif len(kept_drafts) < MAX_PICK_DRAFTS and (max_time_iso is None or (t_s is not None and str(t_s) <= max_time_iso)):
                            if d:
                                kept_drafts.add(d)
                            sel.append(r_i)
                    if not sel:
                        continue
                    sub = batch.take(pa.array(sel, type=pa.int32()))
                    mat = np.column_stack([sub.column(c).to_numpy(zero_copy_only=False) for c in pack_cols]).astype(np.int8)
                    offs, cols, cnt = _pack_csr(mat)
                    picks = sub.column("pick").to_pylist()
                    taken_list.extend(name_idx.get(p, -1) if p is not None else -1 for p in picks)
                    offsets_list.extend((offs[1:] + total_entries).tolist())
                    cards_list.append(gidx[cols.astype(np.int64)])
                    counts_list.append(cnt)
                    total_entries += len(cols)
        except Exception:  # noqa: BLE001 - fallback to stdlib csv reader
            with detect.open_binary(path) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                rdr = csv.reader(text)
                hdr = next(rdr, None)
                if not hdr:
                    continue
                pack_pos = [(j, c[len("pack_card_"):]) for j, c in enumerate(hdr) if c.startswith("pack_card_")]
                if not pack_pos:
                    continue
                for _, nm in pack_pos:
                    if nm not in name_idx:
                        name_idx[nm] = len(names_out)
                        names_out.append(nm)
                ix_did = hdr.index("draft_id") if "draft_id" in hdr else -1
                ix_dt = hdr.index("draft_time") if "draft_time" in hdr else -1
                ix_pn = hdr.index("pick_number") if "pick_number" in hdr else -1
                ix_pk = hdr.index("pick") if "pick" in hdr else -1
                if ix_pn < 0 or ix_pk < 0:
                    continue
                for row in rdr:
                    try:
                        pn = int(row[ix_pn])
                    except (ValueError, IndexError):
                        continue
                    if pn + 1 > EARLY_PICKS:
                        continue
                    d = row[ix_did] if ix_did >= 0 else ""
                    t_s = row[ix_dt] if ix_dt >= 0 else None
                    if allowed_draft_ids is not None and d in allowed_draft_ids:
                        kept_drafts.add(d)
                    elif d and d in kept_drafts:
                        pass
                    elif len(kept_drafts) < MAX_PICK_DRAFTS and (max_time_iso is None or (t_s is not None and t_s <= max_time_iso)):
                        if d:
                            kept_drafts.add(d)
                    else:
                        continue
                    c_row, n_row = [], []
                    for col_j, nm in pack_pos:
                        v = int(row[col_j] or 0)
                        if v > 0:
                            c_row.append(name_idx[nm])
                            n_row.append(v)
                    taken_list.append(name_idx.get(row[ix_pk], -1))
                    total_entries += len(c_row)
                    offsets_list.append(total_entries)
                    if c_row:
                        cards_list.append(np.array(c_row, dtype=np.int16))
                        counts_list.append(np.array(n_row, dtype=np.int8))

    if not taken_list or not cards_list:
        return None
    return dict(
        taken=np.array(taken_list, dtype=np.int64),
        offsets=np.array(offsets_list, dtype=np.int64),
        cards=np.concatenate(cards_list),
        counts=np.concatenate(counts_list),
        n_drafts=len(kept_drafts),
        n_picks5=len(taken_list),
    ), names_out


def luce_fit(taken: np.ndarray, offsets: np.ndarray, cards: np.ndarray, counts: np.ndarray,
             n_items: int, ridge: float = LUCE_RIDGE, chunk: int = 2_000_000, maxiter: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Luce / conditional-logit pick strengths over `n_items`."""
    n_pick = len(taken)
    keep = taken >= 0
    lens = np.diff(offsets)
    appear = np.bincount(cards.astype(np.int64), minlength=n_items).astype(float)
    takes = np.bincount(taken[keep], minlength=n_items).astype(float)
    step = max(1, chunk // max(int(lens.mean()) if n_pick else 1, 1))
    bounds = list(range(0, n_pick, step)) + [n_pick]

    def fg(theta):
        ll = 0.0
        g = np.zeros(n_items)
        for a, b in zip(bounds[:-1], bounds[1:]):
            o0, o1 = offsets[a], offsets[b]
            cc = cards[o0:o1].astype(np.int64)
            w = counts[o0:o1].astype(float) * np.exp(theta[cc])
            lo = offsets[a:b] - o0
            S = np.add.reduceat(w, lo) if o1 > o0 else np.zeros(b - a)
            S = np.where(lens[a:b] > 0, S, 1.0)
            k = keep[a:b]
            ll += float(theta[taken[a:b][k]].sum() - np.log(S[k]).sum())
            inv = np.repeat(np.where(k, 1.0 / S, 0.0), lens[a:b])
            g -= np.bincount(cc, weights=w * inv, minlength=n_items)
        g += takes
        ll -= 0.5 * ridge * float(theta @ theta)
        g -= ridge * theta
        return -ll, -g

    res = minimize(fg, np.zeros(n_items), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=maxiter * 2))
    theta = res.x.copy()
    ok = appear > 0
    if ok.any():
        theta[ok] -= theta[ok].mean()
    theta[~ok] = np.nan
    return theta, takes


def map_signal_to_betas(cards: pl.DataFrame, space: design.DeckSpace, exposure: np.ndarray,
                        beta_zero: np.ndarray, theta_names: np.ndarray, takes_names: np.ndarray,
                        pick_names: list[str], min_taken: int = MIN_TAKEN) -> tuple[np.ndarray, dict]:
    """Exposure-weighted WLS mapping of LUCE5 strengths onto `beta_zero` with per-rarity intercepts + one common slope."""
    sig_names = np.where(takes_names >= min_taken, theta_names, np.nan)
    csort = cards.sort("card_id")
    cnames = csort["name"].to_list()
    cid_of = {n: i for i, n in enumerate(cnames)}
    for n, i in list(cid_of.items()):
        if " // " in n:
            cid_of.setdefault(n.split(" // ")[0].strip(), i)

    sig_full = np.full(csort.height, np.nan)
    for idx_n, nm in enumerate(pick_names):
        c_i = cid_of.get(nm, -1)
        if c_i >= 0:
            sig_full[c_i] = sig_names[idx_n]

    rar_list = []
    for r, il, ib in zip(csort["rarity"].to_list(), csort["is_land"].to_list(), csort["is_basic"].to_list()):
        rar_list.append("basic" if ib else ("land" if il else {"common": "C", "uncommon": "U", "rare": "R", "mythic": "M"}.get(str(r), "rest")))
    rarity = np.array(rar_list, dtype=object)
    nonbasic = rarity != "basic"

    expo_full = np.zeros(csort.height)
    expo_full[space.nb_ids] = exposure
    beta_full = np.zeros(csort.height)
    beta_full[space.nb_ids] = beta_zero

    elig = nonbasic & (expo_full > 0)
    ok = np.isfinite(sig_full) & elig
    if ok.sum() >= 2 and np.std(sig_full[ok]) > 0:
        z = np.where(ok, (sig_full - sig_full[ok].mean()) / sig_full[ok].std(), 0.0)
    else:
        z = np.zeros_like(sig_full)
        ok = np.zeros(len(sig_full), dtype=bool)
    cov = ok & elig

    groups = rarity.copy()
    for g in set(groups.tolist()):
        if int(((groups == g) & cov).sum()) < MIN_GROUP:
            groups[groups == g] = "rest"
    gl = sorted(set(groups[nonbasic].tolist()), key=str) or ["rest"]
    G = np.stack([(groups == g).astype(float) for g in gl], axis=1)
    use_sig = int(cov.sum()) >= MIN_SIGNAL_CARDS
    X = np.hstack([G, z[:, None]]) if use_sig else G
    m = (cov if use_sig else elig) & (expo_full > 0) & np.isfinite(beta_full)
    if int(m.sum()) < 2:
        return np.zeros(space.n_card), dict(covered=0, r2=0.0, slope=0.0)
    sw = np.sqrt(expo_full[m])
    A = X[m] * sw[:, None]
    y = beta_full[m] * sw
    coef = np.linalg.solve(A.T @ A + 1e-8 * np.eye(X.shape[1]), A.T @ y)
    pred = np.where(nonbasic, X @ coef, 0.0)
    wm = expo_full[m]
    ybar = float((wm * beta_full[m]).sum() / wm.sum())
    sst = float((wm * (beta_full[m] - ybar) ** 2).sum())
    sse = float((wm * (beta_full[m] - pred[m]) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return pred[space.nb_ids], dict(
        covered=int(cov.sum()),
        eligible=int(elig.sum()),
        r2=round(float(r2), 4),
        slope=round(float(coef[-1]), 5) if use_sig else 0.0,
    )


def luce5_prior_for_stage(set_code: str, cards: pl.DataFrame, space: design.DeckSpace, st_draft: design.StageData,
                          tmpdir: str, n_jobs: int, opts: cardvalue.ChainOpts,
                          pick_paths: list[Path] | None = None, min_taken: int = MIN_TAKEN,
                          attr_load: np.ndarray | None = None, extras: list | None = None) -> tuple[np.ndarray | None, dict]:
    """Compute the LUCE5 card prior vector (length `space.n_card`) for `st_draft`, or (None, info) if unavailable.

    `attr_load` / `extras`: the attribute columns and trailing blocks (card pairs, card x archetype) of the deployed design, exactly as
    `cardvalue.run_chain` gets them: the zero-mean calibration fit runs on the same columns as the fit the prior goes into, and the
    prior is the mean of the card-specific part `u` (the slot `run_chain` puts it in)."""
    paths = find_pick_files(set_code) if pick_paths is None else list(pick_paths)
    if not paths:
        return None, dict(available=False, reason="no pick files")
    draft_ids = set(st_draft.games["draft_id"].to_list())
    gt = st_draft.games["game_time"].drop_nulls()
    max_iso = str(gt.max())[:19] if gt.len() else None
    loaded = read_early_picks(paths, allowed_draft_ids=draft_ids, max_time_iso=max_iso)
    if loaded is None:
        return None, dict(available=False, reason="no matching early picks in registered pick files")
    parr, pnames = loaded
    theta_n, takes_n = luce_fit(parr["taken"], parr["offsets"], parr["cards"], parr["counts"], len(pnames))

    # Zero-mean draft fit to calibrate the rarity + LUCE5 mapping scale
    zdir = Path(tmpdir) / "pickprior_zero"
    zdir.mkdir(parents=True, exist_ok=True)
    n_attr = 0 if attr_load is None else int(attr_load.shape[1])
    n_extra = int(sum(e["n"] for e in (extras or [])))
    f0 = cardvalue._stage_fit(st_draft, np.zeros(space.n_card + space.n_feat + n_attr + n_extra), space.n_card, space.n_feat,
                              str(zdir), n_jobs, 0, None, None, attr_load=attr_load, extras=extras or None, opts=opts)
    prior_cards, map_info = map_signal_to_betas(cards, space, st_draft.exposure, f0.u, theta_n, takes_n, pnames, min_taken=min_taken)
    if map_info["covered"] == 0:
        return None, dict(available=False, reason="too few cards met minimum pick count")
    info = dict(
        available=True,
        n_picks5=int(parr["n_picks5"]),
        n_pick_drafts=int(parr["n_drafts"]),
        sigma_zero=round(float(f0.sigma), 5),
        **map_info,
    )
    return prior_cards, info


def decide_pickprior(mode: str, n_draft_games: int, pick_paths: list[Path], fit_info: dict | None = None,
                     quick: bool = False) -> dict:
    """Decision record for meta.json / report.json."""
    out = dict(mode=mode, available=bool(pick_paths), applied=False, n_draft_games=int(n_draft_games),
               threshold=COLD_START_DRAFT_GAMES)
    if mode == "off":
        out["reason"] = pipeline_off_reason(quick)
        return out
    if n_draft_games < validate.MIN_PRIOR_GAMES:
        out["reason"] = "no usable draft stage"
        return out
    if mode == "auto" and n_draft_games >= COLD_START_DRAFT_GAMES:
        out["reason"] = f"draft stage has {n_draft_games:,} games (>= {COLD_START_DRAFT_GAMES:,}): cold-start pick prior not needed"
        return out
    if not pick_paths:
        out["reason"] = (f"early-format draft ({n_draft_games:,} < {COLD_START_DRAFT_GAMES:,} games), "
                         "but no 17Lands pick files (draft_data_public) are registered")
        return out
    if fit_info is None or not fit_info.get("available"):
        out["reason"] = (fit_info or {}).get("reason") or "pick prior could not be computed from the registered pick files"
        return out
    out.update(
        applied=True,
        source="forced" if mode == "on" else "cold-start",
        n_picks5=fit_info["n_picks5"],
        n_pick_drafts=fit_info["n_pick_drafts"],
        covered=fit_info["covered"],
        r2=fit_info["r2"],
        slope=fit_info["slope"],
        reason=(f"LUCE5 early-pick prior applied ({fit_info['covered']} cards covered across {fit_info['n_picks5']:,} early picks, "
                f"R² = {fit_info['r2']:.2f}; "
                + ("forced on by the user)" if mode == "on" else
                   f"pre-registered cold-start rule for < {COLD_START_DRAFT_GAMES:,} draft games)")),
    )
    return out


def pipeline_off_reason(quick: bool = False) -> str:
    from sealed_lab.model.pipeline import off_reason

    return off_reason(quick)
