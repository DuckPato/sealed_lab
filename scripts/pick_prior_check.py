"""Does how early drafters take a card (17lands pick data) work as a PRIOR MEAN for the draft-stage card values?  (diagnostic only: no change to the model)

    python scripts/pick_prior_check.py picks    OUT [--sets ...]            pick files -> OUT/picks/SET.npz (compact: per pick the cards in the pack)
    python scripts/pick_prior_check.py fit      OUT [--sets ...] [--jobs N] both directions (A->B, B->A) per set -> OUT/results/SET.pkl
    python scripts/pick_prior_check.py report   OUT [--sets ...]            pick_prior.json + pick_prior.md
    python scripts/pick_prior_check.py time-one SET OUT [--jobs N]          time the set-up + one CV (nothing is saved)

OUT is a scratch folder; SEALED_LAB_HOME is set to a scratch home (draft_drift's OUT/homes/SET unless --homes), never the app home.

Design.  Per set, the draft-stage events (game data, every draft event type in the corpus) are split by event into halves A and B
(`split_half_search.split_events`: stratified by event type and start time).  Everything is computed on the training half T and scored on
the other half H; both directions (A->B, B->A) are run and averaged.
  Pick signal (training-half picks only): for every card, from the Premier / Trad pick files (`draft_data_public.SET.{PremierDraft,TradDraft}`),
    ATA      minus the mean pick number (1..14/15, within the pack) at which the card was taken (cards taken >= MIN_TAKEN times)
    LUCE5    Luce / conditional-logit strength theta_c: P(taken card | cards in the pack) = c_t e^theta_t / sum_j c_j e^theta_j, picks 1-5 of
             every pack only (fewer colour-commitment effects), ridge-penalised maximum likelihood (L-BFGS)
    LUCEALL  the same on all picks
  Prior mean: m_c = a_group(c) + b * z_c  (z = the signal standardised over the covered cards, 0 for cards without signal), fitted on the
    training half by exposure-weighted least squares of the training half's zero-mean draft-stage beta on the signal: per-rarity intercept
    a_g (C, U, R, M, non-basic land; groups with < MIN_GROUP covered cards merge into 'rest') and ONE common slope b (a per-rarity slope
    version is reported in-sample only).  Control "RAR": the same with the intercepts only (no pick signal): what rarity alone gives.
  Draft stage refit on the training half with prior mean m (structure features keep a zero prior mean), card sigma re-chosen by the
    pipeline's own 5-fold CV grouped by draft_id on the training half (`cardvalue.select_sigma`, random scheme, the draft grid), separately for
    the zero-mean baseline and for each prior, so every condition is tuned.
  Scoring: log loss of the held-out half's draft games under the training half's model (covariate encoder fitted on the training half,
    time clamped), per-game paired difference baseline - prior (positive = the prior helps), cluster bootstrap over held-out events.  Weighted
    MSE against the held-out half's own zero-mean betas (fitted at the sigma the baseline CV chose on that half): the held-out noise is common
    to every condition, so MSE differences are noise-corrected; also the shift of the fitted betas toward the prior, by exposure tertile.

Approximations (repeated in the report): (1) one common castability deck space per set (built on all draft events) serves both halves, as
in draft_drift; (2) the mapping regression and the pick signal use training-half data only, but the CV that re-chooses sigma reuses the
mapping fitted on the whole training half (a handful of parameters; the held-out score is clean); (3) pick drafts whose id is in no game
event are split into A / B by the same time-stratified rule on their own ids; (4) PickTwo pick files are not used (their game events are in
the draft stage but carry no pick signal); (5) log-loss gains are nats per game, as in the app's validation; (6) the held-out half's
baseline sigma for the MSE target comes from the other direction's baseline CV.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "SOS", "TDM", "TLA", "TMT"]
from sealed_lab.ingest.datasets import default_data_dir as _default_data_dir  # noqa: E402
DATA_ROOT = Path(_default_data_dir())          # env SEALED_LAB_DATA_DIR, else the app's data folder default
DEFAULT_HOMES = Path(os.environ.get("SEALED_LAB_DRIFT_DIR") or Path(tempfile.gettempdir()) / "sealed_lab_drift") / "homes"
PICK_EVENTS = ("PremierDraft", "TradDraft")
MIN_TAKEN = 10            # a card needs this many takes in the training half to carry an ATA / Luce signal
MIN_GROUP = 6             # rarity groups with fewer covered cards merge into 'rest' in the mapping regression
EARLY_PICKS = 5           # LUCE5 uses picks 1..5 of every pack
LUCE_RIDGE = 0.1          # ridge on theta (prior SD ~3): only keeps rarely seen cards finite
VARIANTS = ("base", "rar", "ata", "luce5", "luceall")        # base = zero prior mean; rar = rarity intercepts only (control)
PRIORS = VARIANTS[1:]
N_BOOT = 2000
RARITY_ORDER = ("C", "U", "R", "M", "land")
WEAK_SIGMA = 1.0          # card prior SD of the "weak" held-out target fit (almost no shrinkage: an unbiased noisy target for the MSE)
GRID_FINE = np.geomspace(0.01, 0.40, 15)      # sensitivity: the draft grid of the pipeline (8 points) with twice the resolution


# =====================================================================================================================
# pure parts (tested without any data)
# =====================================================================================================================

def pack_csr(mat: np.ndarray):
    """Dense (n_picks, n_cols) pack-content counts -> (offsets (n+1), cards, counts) of the non-zero entries, row by row."""
    mat = np.asarray(mat)
    rows, cols = np.nonzero(mat)
    cnt = mat[rows, cols].astype(np.int8)
    offsets = np.zeros(mat.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=mat.shape[0]), out=offsets[1:])
    return offsets, cols.astype(np.int16), cnt


def take_stats(taken, pick_pos, n_items: int):
    """Per item: number of takes and the mean pick position at which it was taken (NaN without takes)."""
    taken, pick_pos = np.asarray(taken), np.asarray(pick_pos, float)
    m = taken >= 0
    n = np.bincount(taken[m], minlength=n_items).astype(float)
    s = np.bincount(taken[m], weights=pick_pos[m], minlength=n_items)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
    return n, mean


def ata_signal(taken, pick_pos, n_items: int, min_taken: int = MIN_TAKEN):
    """ATA signal = minus the mean pick position (high = taken early = good); NaN for items taken fewer than `min_taken` times."""
    n, mean = take_stats(taken, pick_pos, n_items)
    return np.where(n >= min_taken, -mean, np.nan), n


def luce_fit(taken, offsets, cards, counts, n_items: int, ridge: float = LUCE_RIDGE, chunk: int = 2_000_000, maxiter: int = 300):
    """Luce / conditional-logit strengths.  Pick t chose item taken[t] among the items cards[offsets[t]:offsets[t+1]] (with multiplicities
    counts): P = c_taken e^theta_taken / sum_j c_j e^theta_j  (the taken item must be in the pack).  Maximises the log-likelihood minus
    (ridge/2) sum theta^2 by L-BFGS; the likelihood is invariant to a common shift of theta, so theta is centred (mean over the items
    that appear in some pack) afterwards.  Returns (theta (n_items; NaN for items in no pack), n_takes, n_appear)."""
    from scipy.optimize import minimize

    taken = np.asarray(taken, np.int64)
    offsets = np.asarray(offsets, np.int64)
    cards = np.asarray(cards)
    counts = np.asarray(counts, float)
    n_pick = len(taken)
    keep = taken >= 0
    lens = np.diff(offsets)
    appear = np.bincount(cards.astype(np.int64), weights=None, minlength=n_items).astype(float)
    takes = np.bincount(taken[keep], minlength=n_items).astype(float)
    bounds = list(range(0, n_pick, max(1, chunk // max(int(lens.mean()) if n_pick else 1, 1))))
    bounds.append(n_pick)

    def fg(theta):
        ll = 0.0
        g = np.zeros(n_items)
        for a, b in zip(bounds[:-1], bounds[1:]):
            o0, o1 = offsets[a], offsets[b]
            cc = cards[o0:o1].astype(np.int64)
            w = counts[o0:o1] * np.exp(theta[cc])
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
    return theta, takes, appear


def standardise(x, mask=None):
    """z-score over the finite (and masked) entries; entries without a signal become 0 (= the covered mean)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x) if mask is None else (np.isfinite(x) & np.asarray(mask, bool))
    if ok.sum() < 2 or x[ok].std() == 0:
        return np.zeros_like(x), ok
    z = np.where(ok, (x - x[ok].mean()) / x[ok].std(), 0.0)
    return z, ok


def merge_small_groups(labels, covered, min_group: int = MIN_GROUP, rest: str = "rest"):
    """Group labels with fewer than `min_group` covered entries are renamed `rest`."""
    labels = np.asarray(labels, dtype=object).copy()
    covered = np.asarray(covered, bool)
    for g in set(labels.tolist()):
        if int(((labels == g) & covered).sum()) < min_group:
            labels[labels == g] = rest
    return labels


def fit_map(z, beta, weight, groups, fit_mask, per_group_slope: bool = False, use_signal: bool = True, ridge: float = 1e-8):
    """Weighted least squares of beta on [group intercepts, z (one common slope, or one slope per group)] over the entries of `fit_mask`.
    use_signal False: intercepts only (control).  Returns dict(pred over ALL entries, params, r2 (weighted, in-sample), n)."""
    z, beta, weight = np.asarray(z, float), np.asarray(beta, float), np.asarray(weight, float)
    groups = np.asarray(groups, dtype=object)
    gl = sorted(set(groups.tolist()), key=str)
    G = np.stack([(groups == g).astype(float) for g in gl], axis=1)
    cols = [G]
    names = [f"a[{g}]" for g in gl]
    if use_signal:
        if per_group_slope:
            cols.append(G * z[:, None])
            names += [f"b[{g}]" for g in gl]
        else:
            cols.append(z[:, None])
            names.append("b")
    X = np.hstack(cols)
    m = np.asarray(fit_mask, bool) & (weight > 0) & np.isfinite(beta)
    sw = np.sqrt(weight[m])
    A = X[m] * sw[:, None]
    y = beta[m] * sw
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(X.shape[1]), A.T @ y)
    pred = X @ coef
    wm = weight[m]
    ybar = float((wm * beta[m]).sum() / wm.sum())
    sst = float((wm * (beta[m] - ybar) ** 2).sum())
    sse = float((wm * (beta[m] - pred[m]) ** 2).sum())
    return dict(pred=pred, params=dict(zip(names, [float(c) for c in coef])), r2=1.0 - sse / sst if sst > 0 else float("nan"), n=int(m.sum()))


def weighted_corr_spearman(x, y):
    """Spearman correlation over the finite pairs (NaN if fewer than 3)."""
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return float("nan")
    return float(spearmanr(x[m], y[m])[0])


def exposure_tertiles(exposure, eligible, labels=("T1", "T2", "T3")):
    """T1 (low) .. T3 (high) by exposure rank among the eligible entries (equal sizes +- 1); the others 'none'."""
    exposure = np.asarray(exposure, float)
    out = np.full(len(exposure), "none", dtype=object)
    idx = np.nonzero(np.asarray(eligible, bool))[0]
    if len(idx):
        order = idx[np.lexsort((idx, exposure[idx]))]
        for k, chunk in enumerate(np.array_split(order, len(labels))):
            out[chunk] = labels[k]
    return out


def split_by_ids(ids, starts, types, seed: int):
    """Wrapper of split_half_search.split_events (A, B lists of ids)."""
    from split_half_search import split_events

    return split_events(list(ids), list(starts), list(types), seed=seed)


def assign_halves(pick_ids, pick_starts, pick_types, game_a, game_b, seed: int):
    """Half of every pick-file draft id: the game split's half where the id is a game event, else a time-stratified split of the unmatched
    ids.  Returns (dict id -> 'A'|'B', match_rate = share of pick drafts that are game events)."""
    ga, gb = set(game_a), set(game_b)
    half: dict = {}
    unmatched = []
    for i, d in enumerate(pick_ids):
        if d in ga:
            half[d] = "A"
        elif d in gb:
            half[d] = "B"
        else:
            unmatched.append(i)
    if unmatched:
        ua, ub = split_by_ids([pick_ids[i] for i in unmatched], [pick_starts[i] for i in unmatched], [pick_types[i] for i in unmatched], seed)
        half.update({d: "A" for d in ua})
        half.update({d: "B" for d in ub})
    n = max(len(pick_ids), 1)
    return half, 1.0 - len(unmatched) / n


def paired_boot(diff, clusters, n_boot: int = N_BOOT, seed: int = 0):
    """Mean of per-game differences with a 95% percentile interval from resampling whole clusters (events).  Returns (mean, lo, hi)."""
    from split_half_search import cluster_boot

    return cluster_boot(np.asarray(diff, float), np.ones(len(diff)), clusters, n_boot=n_boot, seed=seed)


def weighted_mse(pred, target, weight, mask=None):
    pred, target, weight = np.asarray(pred, float), np.asarray(target, float), np.asarray(weight, float)
    m = np.isfinite(pred) & np.isfinite(target) & (weight > 0)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if not m.any():
        return float("nan")
    return float((weight[m] * (pred[m] - target[m]) ** 2).sum() / weight[m].sum())


def card_boot_diff(sq_base, sq_prior, weight, n_boot: int = N_BOOT, seed: int = 0):
    """Weighted mean of (sq_base - sq_prior) over cards with a 95% bootstrap interval resampling cards (positive = the prior is closer)."""
    sq_base, sq_prior, weight = (np.asarray(x, float) for x in (sq_base, sq_prior, weight))
    m = np.isfinite(sq_base) & np.isfinite(sq_prior) & (weight > 0)
    d, w = (sq_base - sq_prior)[m], weight[m]
    if len(d) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    r = (w[idx] * d[idx]).sum(1) / w[idx].sum(1)
    return float((w * d).sum() / w.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def mean_ci(values):
    """Mean of per-set values with a t-free summary (min, max) -- for the direction average."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    return float(v.mean()) if len(v) else float("nan")


# =====================================================================================================================
# pick files -> compact arrays
# =====================================================================================================================

def pick_file(set_code: str, event: str) -> Path | None:
    for name in (f"draft_data_public.{set_code}.{event}.csv.gz", f"draft_data_public.{set_code}.{event}.csv"):
        p = DATA_ROOT / set_code / name
        if p.exists():
            return p
    return None


def _header(path: Path) -> list[str]:
    import csv
    import gzip

    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def read_pick_file(path: Path, event: str, names_out: list, name_idx: dict, block_mb: int = 24):
    """Stream one pick file with pyarrow (only the meta columns and pack_card_* are parsed).  Appends to `names_out` / `name_idx` (card
    names of pack_card_ columns, global over the files of a set).  Returns dict of arrays: draft code (local), pack, pick, taken (global
    name index; -1 = not a pack card), offsets, cards (global name index), counts, plus drafts (list of (id, first draft_time str))."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    hdr = _header(path)
    pack_cols = [c for c in hdr if c.startswith("pack_card_")]
    meta = ["draft_id", "draft_time", "pack_number", "pick_number", "pick"]
    for c in pack_cols:
        nm = c[len("pack_card_"):]
        if nm not in name_idx:
            name_idx[nm] = len(names_out)
            names_out.append(nm)
    gidx = np.array([name_idx[c[len("pack_card_"):]] for c in pack_cols], dtype=np.int64)
    col_types = {"draft_id": pa.string(), "draft_time": pa.string(), "pack_number": pa.int16(), "pick_number": pa.int16(), "pick": pa.string(),
                 **{c: pa.int8() for c in pack_cols}}
    reader = pacsv.open_csv(str(path), read_options=pacsv.ReadOptions(block_size=block_mb << 20),
                            convert_options=pacsv.ConvertOptions(include_columns=meta + pack_cols, column_types=col_types))
    draft_code: dict = {}
    draft_time: dict = {}
    out = dict(draft=[], pack=[], pick=[], taken=[], offsets=[], cards=[], counts=[])
    total_cards = 0
    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        n = batch.num_rows
        if n == 0:
            continue
        did = batch.column("draft_id").to_pylist()
        dt = batch.column("draft_time").to_pylist()
        codes = np.empty(n, np.int32)
        for i, d in enumerate(did):
            c = draft_code.get(d)
            if c is None:
                c = draft_code[d] = len(draft_code)
                draft_time[d] = dt[i]
            codes[i] = c
        mat = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False) for c in pack_cols]).astype(np.int8)
        offs, cols, cnt = pack_csr(mat)
        picks = batch.column("pick").to_pylist()
        taken = np.array([name_idx.get(p, -1) if p is not None else -1 for p in picks], dtype=np.int32)
        out["draft"].append(codes)
        out["pack"].append(batch.column("pack_number").to_numpy(zero_copy_only=False).astype(np.int8))
        out["pick"].append(batch.column("pick_number").to_numpy(zero_copy_only=False).astype(np.int8))
        out["taken"].append(taken)
        out["offsets"].append(offs[1:] + total_cards)
        out["cards"].append(gidx[cols.astype(np.int64)].astype(np.int16))
        out["counts"].append(cnt)
        total_cards += len(cols)
    res = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
    res["offsets"] = np.concatenate([[0], res["offsets"]]).astype(np.int64)
    ids = [None] * len(draft_code)
    for d, c in draft_code.items():
        ids[c] = d
    res["draft_ids"] = ids
    res["draft_times"] = [draft_time[d] for d in ids]
    res["event"] = event
    return res


def build_picks(set_code: str, out: Path) -> dict:
    """Preprocess the set's Premier / Trad pick files into OUT/picks/SET.npz + SET.json (card names, per-draft ids / times / event type)."""
    t0 = time.time()
    names: list = []
    name_idx: dict = {}
    parts = []
    for ev in PICK_EVENTS:
        p = pick_file(set_code, ev)
        if p is None:
            continue
        t1 = time.time()
        r = read_pick_file(p, ev, names, name_idx)
        parts.append(r)
        print(f"  {set_code}/{ev}: {len(r['taken'])} picks, {len(r['draft_ids'])} drafts, {len(r['cards'])} pack entries, {time.time() - t1:.0f}s", flush=True)
    if not parts:
        raise FileNotFoundError(f"no pick files for {set_code}")
    draft_ids, draft_times, draft_event = [], [], []
    code_off = 0
    cols = dict(draft=[], pack=[], pick=[], taken=[], counts=[], cards=[], offsets=[])
    ent_off = 0
    for r in parts:
        cols["draft"].append(r["draft"] + code_off)
        for k in ("pack", "pick", "taken", "counts", "cards"):
            cols[k].append(r[k])
        cols["offsets"].append(r["offsets"][1:] + ent_off)
        ent_off += len(r["cards"])
        code_off += len(r["draft_ids"])
        draft_ids += r["draft_ids"]
        draft_times += r["draft_times"]
        draft_event += [r["event"]] * len(r["draft_ids"])
    arr = {k: np.concatenate(v) for k, v in cols.items()}
    arr["offsets"] = np.concatenate([[0], arr["offsets"]]).astype(np.int64)
    (out / "picks").mkdir(parents=True, exist_ok=True)
    np.savez(out / "picks" / f"{set_code}.npz", **arr)
    (out / "picks" / f"{set_code}.json").write_text(json.dumps(dict(names=names, draft_ids=draft_ids, draft_times=draft_times, draft_event=draft_event)),
                                                    encoding="utf-8")
    print(f"picks {set_code}: {len(arr['taken'])} picks, {len(draft_ids)} drafts, {len(names)} card names in {time.time() - t0:.0f}s", flush=True)
    return dict(n_picks=int(len(arr["taken"])), n_drafts=len(draft_ids))


def load_picks(out: Path, set_code: str):
    z = np.load(out / "picks" / f"{set_code}.npz")
    meta = json.loads((out / "picks" / f"{set_code}.json").read_text(encoding="utf-8"))
    return {k: z[k] for k in z.files}, meta


def subset_picks(offsets, cards, counts, sel):
    """Rows `sel` (indices of picks) of the CSR pack arrays -> new (offsets, cards, counts)."""
    sel = np.asarray(sel, np.int64)
    lens = np.diff(offsets)[sel]
    new = np.zeros(len(sel) + 1, np.int64)
    np.cumsum(lens, out=new[1:])
    total = int(new[-1])
    src = np.repeat(offsets[sel] - new[:-1], lens) + np.arange(total)
    return new, cards[src], counts[src]


def pick_signals(a: dict, n_names: int, mask) -> dict:
    """ATA / LUCE5 / LUCEALL signals over the card NAMES from the picks selected by `mask` (a boolean over picks).
    Returns dict(ata, luce5, luceall (arrays over names, NaN = no signal), n_taken, n_picks, n_picks5)."""
    sel = np.nonzero(mask)[0]
    taken = a["taken"][sel].astype(np.int64)
    pos = a["pick"][sel].astype(float) + 1.0                       # pick_number is 0-based in the files
    ata, n_taken = ata_signal(taken, pos, n_names)
    early = pos <= EARLY_PICKS
    out = dict(ata=ata, n_taken=n_taken, n_picks=int(len(sel)), n_picks5=int(early.sum()))
    for key, m in (("luce5", early), ("luceall", np.ones(len(sel), bool))):
        s2 = sel[m]
        off, cds, cnt = subset_picks(a["offsets"], a["cards"], a["counts"], s2)
        theta, takes, appear = luce_fit(a["taken"][s2].astype(np.int64), off, cds, cnt, n_names)
        out[key] = np.where(takes >= MIN_TAKEN, theta, np.nan)
        out[key + "_takes"] = takes
    return out


# =====================================================================================================================
# the experiment
# =====================================================================================================================

def scratch_guard(path) -> None:
    """Refuse to run with the app home as SEALED_LAB_HOME."""
    app = Path(os.environ.get("LOCALAPPDATA", "")) / "sealed_lab"
    p = Path(path).resolve()
    if p == app.resolve() or app.resolve() in p.parents:
        raise RuntimeError(f"refusing to use the app home {app} as a scratch home")


class SetContext:
    """Everything shared by both directions of one set."""

    def __init__(self, sc: str, out: Path, drift_out: Path):
        import polars as pl

        import draft_drift as dd
        from split_half_search import set_seed, split_events

        scratch_guard(dd.home_of(drift_out, sc))
        self.sc, self.out = sc, Path(out)
        self.setup = dd.make_setup(drift_out, sc)
        scratch_guard(os.environ["SEALED_LAB_HOME"])
        ev = dd.events_table(self.setup)
        self.events = ev
        ids, starts, types = ev["draft_id"].to_list(), ev["start"].to_list(), ev["event"].to_list()
        self.half_ids = dict(zip(("A", "B"), split_events(ids, starts, types, seed=set_seed(0, sc))))
        g = self.setup.inp.games
        self.games = g.filter(pl.col("fmt") == "draft") if "fmt" in g.columns else g
        self.fspace = self.setup.fspace
        self.n_card, self.n_feat, self.nb_ids = self.setup.n_card, self.setup.n_feat, self.setup.nb_ids
        self.n_total = self.setup.n_cards_total
        self.setup.stages = {}
        cards = pl.read_parquet(Path(os.environ["SEALED_LAB_HOME"]) / "sets" / sc / "corpus" / "cards.parquet").sort("card_id")
        self.card_names = cards["name"].to_list()
        rar = []
        for r, il, ib in zip(cards["rarity"].to_list(), cards["is_land"].to_list(), cards["is_basic"].to_list()):
            rar.append("basic" if ib else ("land" if il else {"common": "C", "uncommon": "U", "rare": "R", "mythic": "M"}.get(str(r), "rest")))
        self.rarity = np.array(rar, dtype=object)
        self.is_basic = np.array([r == "basic" for r in rar])
        self.picks, self.pmeta = load_picks(self.out, sc)
        self.pick_names = self.pmeta["names"]
        cid = {n: i for i, n in enumerate(self.card_names)}
        self.name_to_cid = np.array([cid.get(n, -1) for n in self.pick_names], dtype=np.int64)
        # halves of the pick drafts
        pdr = self.pmeta["draft_ids"]
        half, rate = assign_halves(pdr, self.pmeta["draft_times"], self.pmeta["draft_event"], self.half_ids["A"], self.half_ids["B"],
                                   seed=set_seed(1, sc))
        self.pick_half = np.array([0 if half[d] == "A" else 1 for d in pdr], dtype=np.int8)
        self.match_rate = rate
        in_game = set(ids)
        self.n_pick_drafts = len(pdr)
        self.n_game_events = len(ids)
        self.game_events_with_picks = int(len(in_game & set(pdr)))

    def to_cid(self, v_names, fill=np.nan):
        """Array over pick-file names -> array over corpus card ids."""
        out = np.full(self.n_total, fill)
        ok = self.name_to_cid >= 0
        out[self.name_to_cid[ok]] = np.asarray(v_names, float)[ok]
        return out


def per_game_ll(prob, coef) -> np.ndarray:
    e = prob.eta(coef)
    return np.logaddexp(0.0, e) - prob.y * e


def mapping_groups(labels, covered, nonbasic):
    """Rarity groups for the mapping regression: C/U/R/M/land (merged to 'rest' when < MIN_GROUP covered cards)."""
    lab = np.asarray(labels, dtype=object).copy()
    lab[~nonbasic] = "basic"
    return merge_small_groups(lab, covered & nonbasic)


def thin_ids(ids, frac: float, seed: int):
    """Random subset (share `frac`, at least 2 ids) of `ids`, order kept; frac >= 1 returns all."""
    ids = list(ids)
    if frac >= 1.0:
        return ids
    rng = np.random.default_rng(seed)
    k = max(2, int(round(frac * len(ids))))
    keep = np.sort(rng.choice(len(ids), size=k, replace=False))
    return [ids[i] for i in keep]


def run_direction(ctx: SetContext, train: str, jobs: int, tmp: Path, log=print, frac: float = 1.0) -> dict:
    """Train on half `train` (a random share `frac` of its events, and of its pick drafts, when frac < 1), score on the other half, whole.
    Returns a dict of per-variant results (see the module docstring)."""
    import polars as pl

    from sealed_lab.model import cardvalue
    from sealed_lab.model.design import build_stage, make_problem

    held = "B" if train == "A" else "A"
    t0 = time.time()
    from split_half_search import set_seed

    train_ids = thin_ids(ctx.half_ids[train], frac, set_seed(5, ctx.sc))
    gT = ctx.games.filter(pl.col("draft_id").cast(pl.Utf8).is_in([str(i) for i in train_ids]))
    gH = ctx.games.filter(pl.col("draft_id").cast(pl.Utf8).is_in([str(i) for i in ctx.half_ids[held]]))
    stT = build_stage("draft", gT, ctx.fspace)
    probH, _, _ = make_problem(gH, stT.enc, ctx.fspace, clamp_time=True)
    hid = gH["draft_id"].cast(pl.Utf8).to_numpy()
    uid, hgrp = np.unique(hid, return_inverse=True)
    n_ev = np.bincount(hgrp)
    spec = cardvalue.make_spec(stT, ctx.n_card, ctx.n_feat, "draft")
    tdir = tmp / f"{ctx.sc}_{train}" / f"f{frac:g}_g{len(cardvalue.GRIDS_RANDOM['draft'])}"
    tdir.mkdir(parents=True, exist_ok=True)
    zeros_deck = np.zeros(ctx.n_card + ctx.n_feat)
    expo = ctx.setup.to_full(stT.exposure)
    nonbasic = ~ctx.is_basic
    coef_w, _, _ = cardvalue.fit_stage(stT, zeros_deck, spec, WEAK_SIGMA, with_se=False)
    beta_weak = ctx.setup.to_full(coef_w[stT.prob.k:stT.prob.k + ctx.n_card])
    res: dict = dict(set=ctx.sc, train=train, held=held, frac=float(frac), beta_weak=beta_weak, n_train_ids=len(train_ids), n_games_train=int(stT.n_games), n_games_held=int(probH.n), n_ev_train=int(stT.n_groups),
                     n_ev_held=len(uid), held_ids=uid, held_n=n_ev, exposure=expo, seconds={}, variants={})

    def cv_and_fit(name, prior_cards):
        ta = time.time()
        prior = zeros_deck.copy()
        if prior_cards is not None:
            prior[:ctx.n_card] = prior_cards[ctx.nb_ids]
        sigma, info = cardvalue.select_sigma(stT, prior, ctx.n_card, ctx.n_feat, str(tdir), jobs, seed=0, opts=cardvalue.LEGACY_OPTS, has_prior=False)
        coef, _, fr = cardvalue.fit_stage(stT, prior, spec, sigma, with_se=False)
        k = stT.prob.k
        beta = ctx.setup.to_full(coef[k:k + ctx.n_card])
        ll = per_game_ll(probH, coef)
        ll_ev = np.bincount(hgrp, weights=ll, minlength=len(uid))
        res["variants"][name] = dict(sigma=float(sigma), cv_grid=info["grid"], cv_loss=info["loss"], beta=beta, ll_event=ll_ev, ll_mean=float(ll.sum() / probH.n),
                                     converged=bool(fr.converged), prior=None if prior_cards is None else np.asarray(prior_cards, float))
        res["seconds"][name] = time.time() - ta
        log(f"  {ctx.sc} train {train}: {name:8s} sigma {sigma:.4f}  held LL {ll.sum() / probH.n:.6f}  ({time.time() - ta:.0f}s)", flush=True)
        return beta

    beta_base = cv_and_fit("base", None)
    # pick signals from the training half's picks
    tp = time.time()
    pmask = ctx.pick_half == (0 if train == "A" else 1)
    if frac < 1.0:                                   # thin the pick drafts like the games: matched ones follow the sampled events
        sampled = set(train_ids)
        game_ids = set(ctx.events["draft_id"].to_list())
        rng = np.random.default_rng(set_seed(6, ctx.sc))
        draw = rng.random(len(pmask))
        pmask = pmask & np.array([(d in sampled) if d in game_ids else bool(u < frac) for d, u in zip(ctx.pmeta["draft_ids"], draw)])
    pm = pmask[ctx.picks["draft"]]
    sig = pick_signals(ctx.picks, len(ctx.pick_names), pm)
    res["seconds"]["signals"] = time.time() - tp
    signals = {"ata": ctx.to_cid(sig["ata"]), "luce5": ctx.to_cid(sig["luce5"]), "luceall": ctx.to_cid(sig["luceall"])}
    res["pick_stats"] = dict(n_picks=sig["n_picks"], n_picks5=sig["n_picks5"], n_unmapped_names=int((ctx.name_to_cid < 0).sum()),
                             unmapped_takes=float(sig["n_taken"][ctx.name_to_cid < 0].sum()))
    elig = nonbasic & (expo > 0)
    res["signals"] = {k: v.copy() for k, v in signals.items()}
    res["labels"] = ctx.rarity
    # coverage and correlations with the training half's zero-mean betas
    cor: dict = {}
    for k, s in signals.items():
        d: dict = {"covered": int((np.isfinite(s) & elig).sum()), "eligible": int(elig.sum())}
        d["all"] = weighted_corr_spearman(np.where(elig, s, np.nan), np.where(elig, beta_base, np.nan))
        for r in RARITY_ORDER:
            m = elig & (ctx.rarity == r)
            d[r] = weighted_corr_spearman(np.where(m, s, np.nan), np.where(m, beta_base, np.nan))
            d["n_" + r] = int((m & np.isfinite(s)).sum())
        cor[k] = d
    res["corr"] = cor
    # mappings
    maps: dict = {}
    for name in PRIORS:
        if name == "rar":
            z, cov = np.zeros(ctx.n_total), elig
            groups = mapping_groups(ctx.rarity, cov, nonbasic)
            fm = fit_map(z, beta_base, expo, groups, cov, use_signal=False)
        else:
            z, ok = standardise(np.where(elig, signals[name], np.nan))
            cov = ok & elig
            groups = mapping_groups(ctx.rarity, cov, nonbasic)
            fm = fit_map(z, beta_base, expo, groups, cov)
            fm2 = fit_map(z, beta_base, expo, groups, cov, per_group_slope=True)
            fm["r2_per_group_slope"] = fm2["r2"]
            fm["params_per_group_slope"] = fm2["params"]
        m_c = np.where(nonbasic, fm["pred"], 0.0)
        maps[name] = dict(params=fm["params"], r2=fm["r2"], n=fm["n"], r2_per_group_slope=fm.get("r2_per_group_slope"),
                          params_per_group_slope=fm.get("params_per_group_slope"), covered=int(cov.sum()))
        cv_and_fit(name, m_c)
        res["variants"][name]["shift_rms_prior"] = float(np.sqrt(np.average(m_c[elig] ** 2, weights=expo[elig])))
    res["maps"] = maps
    res["seconds"]["total"] = time.time() - t0
    return res


def run_set(sc: str, out: Path, drift_out: Path, jobs: int, directions=("A", "B"), tag: str = "", frac: float = 1.0) -> dict:
    t0 = time.time()
    ctx = SetContext(sc, out, drift_out)
    print(f"{sc}: set-up {time.time() - t0:.0f}s; {ctx.n_game_events} draft events (A {len(ctx.half_ids['A'])}, B {len(ctx.half_ids['B'])}); "
          f"pick drafts {ctx.n_pick_drafts}, game events with picks {ctx.game_events_with_picks}, match rate (pick drafts in game events) {ctx.match_rate:.3f}", flush=True)
    tmp = out / "tmp"
    (out / f"results{tag}").mkdir(parents=True, exist_ok=True)
    runs = {}
    for d in directions:
        runs[d] = run_direction(ctx, d, jobs, tmp, frac=frac)
    meta = dict(set=sc, n_game_events=ctx.n_game_events, n_events=(len(ctx.half_ids["A"]), len(ctx.half_ids["B"])), n_pick_drafts=ctx.n_pick_drafts,
                game_events_with_picks=ctx.game_events_with_picks, match_rate=ctx.match_rate, seconds=time.time() - t0,
                card_names=ctx.card_names, rarity=ctx.rarity, is_basic=ctx.is_basic)
    with open(out / f"results{tag}" / f"{sc}.pkl", "wb") as fh:
        pickle.dump(dict(meta=meta, runs=runs), fh)
    print(f"{sc} done in {time.time() - t0:.0f}s", flush=True)
    return meta


# =====================================================================================================================
# report
# =====================================================================================================================

MIN_EXPO = 100.0           # copy-games a card needs in BOTH halves to enter the beta-scale statistics
TERT = ("T1", "T2", "T3")
SCALE = 1e4                # log-loss gains are shown in 1e-4 nats per game


def boot_blocks(num_ev, den_ev, n_boot: int, seed: int):
    """Resample events: returns (sum_num_draws, sum_den_draws, point_num, point_den) from per-event aggregates."""
    num_ev, den_ev = np.asarray(num_ev, float), np.asarray(den_ev, float)
    G = len(num_ev)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    return num_ev[idx].sum(1), den_ev[idx].sum(1), float(num_ev.sum()), float(den_ev.sum())


def summarize(draws, point):
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    return dict(mean=float(point), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)),
                share_pos=float((draws > 0).mean()))


def load_results(out: Path, sets, tag: str = ""):
    res = {}
    for sc in sets:
        p = Path(out) / f"results{tag}" / f"{sc}.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                res[sc] = pickle.load(fh)
    return res


def gains_table(res: dict, n_boot: int = N_BOOT) -> dict:
    """Held-out log-loss gain (base - prior, nats per game; positive = the prior helps) per set and pooled: direction average, each
    direction, with cluster bootstrap over held-out events."""
    out: dict = {"sets": {}, "pooled": {}, "mean_of_sets": {}}
    blocks: dict = {}          # (set, dir, variant) -> (num draws, den draws, num, den)
    for si, (sc, r) in enumerate(res.items()):
        out["sets"][sc] = {}
        for v in PRIORS:
            dd = {}
            for di, (d, run) in enumerate(r["runs"].items()):
                diff = run["variants"]["base"]["ll_event"] - run["variants"][v]["ll_event"]
                b = boot_blocks(diff, run["held_n"], n_boot, seed=1000 * si + 10 * di + PRIORS.index(v) * 0 + 7)
                blocks[(sc, d, v)] = b
                dd[d] = summarize(b[0] / b[1], b[2] / b[3])
            dirs = list(r["runs"])
            avg_draws = np.mean([blocks[(sc, d, v)][0] / blocks[(sc, d, v)][1] for d in dirs], axis=0)
            avg_point = float(np.mean([blocks[(sc, d, v)][2] / blocks[(sc, d, v)][3] for d in dirs]))
            out["sets"][sc][v] = dict(avg=summarize(avg_draws, avg_point), dirs=dd)
    for v in PRIORS:
        keys = [(sc, d) for sc, r in res.items() for d in r["runs"]]
        num = sum(blocks[(sc, d, v)][0] for sc, d in keys)
        den = sum(blocks[(sc, d, v)][1] for sc, d in keys)
        pn = sum(blocks[(sc, d, v)][2] for sc, d in keys)
        pd_ = sum(blocks[(sc, d, v)][3] for sc, d in keys)
        out["pooled"][v] = summarize(num / den, pn / pd_)
        per_set_draws = [np.mean([blocks[(sc, d, v)][0] / blocks[(sc, d, v)][1] for d in r["runs"]], axis=0) for sc, r in res.items()]
        per_set_pts = [out["sets"][sc][v]["avg"]["mean"] for sc in res]
        out["mean_of_sets"][v] = summarize(np.mean(per_set_draws, axis=0), np.mean(per_set_pts))
        out["mean_of_sets"][v]["n_sets_positive"] = int(sum(p > 0 for p in per_set_pts))
    return out


def beta_rows(res: dict) -> dict:
    """Per (set, direction): the card-level arrays used for the MSE / shift statistics.  The held-out half's own zero-mean betas (the other
    direction's baseline fit, at the sigma its CV chose) are the target."""
    rows = []
    for sc, r in res.items():
        runs = r["runs"]
        if len(runs) < 2:
            continue
        for d, run in runs.items():
            other = runs["B" if d == "A" else "A"]
            tgt, eH, eT = other["variants"]["base"]["beta"], other["exposure"], run["exposure"]
            tgt_weak = other.get("beta_weak")
            elig = (~r["meta"]["is_basic"]) & (eT >= MIN_EXPO) & (eH >= MIN_EXPO)
            tert = exposure_tertiles(eT + eH, elig, TERT)
            tertT = exposure_tertiles(eT, elig, TERT)
            rows.append(dict(set=sc, dir=d, elig=elig, tert=tert, tertT=tertT, rar=r["meta"]["rarity"], eH=eH, eT=eT, target=tgt, target_weak=tgt_weak,
                             beta={v: run["variants"][v]["beta"] for v in VARIANTS},
                             prior={v: run["variants"][v]["prior"] for v in PRIORS}))
    return dict(rows=rows)


def mse_tables(rows: list, n_boot: int = N_BOOT, target: str = "target") -> dict:
    """Weighted MSE against the held-out half's betas by rarity and exposure tertile, per prior variant, pooled over sets and both
    directions with each (set, direction) weighted equally (its weights sum to 1).  Card bootstrap for the difference."""
    groups: dict = {"ALL": lambda r: r["elig"]}
    for g in RARITY_ORDER:
        groups[g] = (lambda g: lambda r: r["elig"] & (r["rar"] == g))(g)
    for t in TERT:
        groups[t] = (lambda t: lambda r: r["tert"] == t)(t)
    out: dict = {}
    for v in PRIORS:
        out[v] = {}
        for gname, sel in groups.items():
            sqb, sqv, w = [], [], []
            for r in rows:
                m = sel(r)
                if not m.any():
                    continue
                ww = r["eH"][m] / r["eH"][m].sum()
                sqb.append((r["beta"]["base"][m] - r[target][m]) ** 2)
                sqv.append((r["beta"][v][m] - r[target][m]) ** 2)
                w.append(ww)
            if not w:
                continue
            sqb, sqv, w = np.concatenate(sqb), np.concatenate(sqv), np.concatenate(w)
            mb, mv = float((w * sqb).sum() / w.sum()), float((w * sqv).sum() / w.sum())
            diff, lo, hi = card_boot_diff(sqb, sqv, w, n_boot=n_boot, seed=3)
            out[v][gname] = dict(n=int(len(w)), mse_base=mb, mse_prior=mv, diff=diff, lo=lo, hi=hi)
    return out


def shift_tables(rows: list) -> dict:
    """How much the prior moves the fitted betas: weighted (training exposure) RMS of beta_prior - beta_base, of the prior mean m, and
    the mean shift, by training-exposure tertile (pooled, each (set, direction) weighted equally)."""
    out: dict = {}
    for v in PRIORS:
        out[v] = {}
        for t in TERT + ("ALL",):
            sh, mm, w = [], [], []
            for r in rows:
                m = r["elig"] if t == "ALL" else (r["tertT"] == t)
                if not m.any():
                    continue
                sh.append(r["beta"][v][m] - r["beta"]["base"][m])
                mm.append(r["prior"][v][m])
                w.append(r["eT"][m] / r["eT"][m].sum())
            if not w:
                continue
            sh, mm, w = np.concatenate(sh), np.concatenate(mm), np.concatenate(w)
            ws = w / w.sum()
            out[v][t] = dict(rms_shift=float(np.sqrt((ws * sh ** 2).sum())), mean_shift=float((ws * sh).sum()), rms_prior=float(np.sqrt((ws * mm ** 2).sum())),
                             slope_shift_on_prior=float((ws * sh * mm).sum() / max((ws * mm ** 2).sum(), 1e-12)))
    return out


def corr_tables(res: dict) -> dict:
    """Coverage and Spearman correlation of each signal with the training half's zero-mean beta: mean over sets (each averaged over the two
    directions); per-set values kept."""
    out: dict = {"per_set": {}, "mean": {}}
    keys = ["all"] + list(RARITY_ORDER)
    for sc, r in res.items():
        out["per_set"][sc] = {}
        for s in ("ata", "luce5", "luceall"):
            vals = {k: mean_ci([run["corr"][s][k] for run in r["runs"].values()]) for k in keys}
            vals["covered"] = mean_ci([run["corr"][s]["covered"] for run in r["runs"].values()])
            vals["eligible"] = mean_ci([run["corr"][s]["eligible"] for run in r["runs"].values()])
            out["per_set"][sc][s] = vals
    for s in ("ata", "luce5", "luceall"):
        out["mean"][s] = {k: mean_ci([out["per_set"][sc][s][k] for sc in res]) for k in keys}
        out["mean"][s]["coverage"] = float(np.nanmean([out["per_set"][sc][s]["covered"] / max(out["per_set"][sc][s]["eligible"], 1) for sc in res]))
    return out


def map_tables(res: dict) -> dict:
    out: dict = {}
    for v in PRIORS:
        slopes, r2, r2g = [], [], []
        for r in res.values():
            for run in r["runs"].values():
                m = run["maps"][v]
                if "b" in m["params"]:
                    slopes.append(m["params"]["b"])
                r2.append(m["r2"])
                if m.get("r2_per_group_slope") is not None:
                    r2g.append(m["r2_per_group_slope"])
        out[v] = dict(slope_mean=mean_ci(slopes), r2_mean=mean_ci(r2), r2_per_group_slope_mean=mean_ci(r2g))
    return out


def sigma_table(res: dict) -> dict:
    return {sc: {d: {v: run["variants"][v]["sigma"] for v in VARIANTS} for d, run in r["runs"].items()} for sc, r in res.items()}


def cv_table(res: dict) -> dict:
    """Best CV loss (per game, at the chosen sigma) of each prior minus the baseline's, averaged over directions (negative = better)."""
    out = {}
    for sc, r in res.items():
        out[sc] = {v: mean_ci([min(run["variants"][v]["cv_loss"]) - min(run["variants"]["base"]["cv_loss"]) for run in r["runs"].values()]) for v in PRIORS}
    return out


def build_report(out: Path, sets, tag: str = "") -> dict:
    res = load_results(out, sets, tag)
    if not res:
        raise SystemExit("no results")
    gains = gains_table(res)
    br = beta_rows(res)
    full = all(run.get("frac", 1.0) >= 1.0 for r in res.values() for run in r["runs"].values())
    has_weak = full and all(run.get("beta_weak") is not None for r in res.values() for run in r["runs"].values())
    rep = dict(sets=list(res), tag=tag, gains=gains, corr=corr_tables(res), maps=map_tables(res), sigma=sigma_table(res), cv=cv_table(res),
               mse=mse_tables(br["rows"]) if full else {}, mse_weak=mse_tables(br["rows"], target="target_weak") if has_weak else {},
               shift=shift_tables(br["rows"]),
               meta={sc: dict(n_game_events=r["meta"]["n_game_events"], n_events=r["meta"]["n_events"], n_pick_drafts=r["meta"]["n_pick_drafts"],
                              game_events_with_picks=r["meta"]["game_events_with_picks"], match_rate=r["meta"]["match_rate"], seconds=r["meta"]["seconds"],
                              games={d: (run["n_games_train"], run["n_games_held"]) for d, run in r["runs"].items()},
                              pick_stats={d: run["pick_stats"] for d, run in r["runs"].items()}) for sc, r in res.items()})
    stem = "pick_prior" + (f"_{tag.lstrip('_')}" if tag else "")
    (Path(out) / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    (Path(out) / f"{stem}.md").write_text(render_md(rep), encoding="utf-8")
    print(f"report: {Path(out) / (stem + '.md')}", flush=True)
    return rep


def _g(d, nd=2):
    return f"{SCALE * d['mean']:+.{nd}f} [{SCALE * d['lo']:+.{nd}f}, {SCALE * d['hi']:+.{nd}f}]"


def _f(x, nd=3):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def render_md(rep: dict) -> str:
    L: list = []
    sets = rep["sets"]
    L.append("# Pick-data prior for the draft-stage card values\n")
    L.append("Gains are held-out log loss of the zero-mean baseline minus the prior (units 1e-4 nats per game; positive = the pick prior helps), average of the "
             "two directions (A->B, B->A), 95% cluster bootstrap over held-out events. LUCE5 = Luce strength from picks 1-5 of each pack, LUCEALL = all picks, "
             "ATA = minus the average pick number, RAR = rarity intercepts only (control, no pick data).\n")
    L.append("## Held-out log-loss gain per set\n")
    L.append("| set | " + " | ".join(PRIORS) + " |")
    L.append("|---|" + "---|" * len(PRIORS))
    for sc in sets:
        L.append(f"| {sc} | " + " | ".join(_g(rep["gains"]["sets"][sc][v]["avg"]) for v in PRIORS) + " |")
    L.append("| pooled (all held-out games) | " + " | ".join(_g(rep["gains"]["pooled"][v]) for v in PRIORS) + " |")
    L.append("| mean of sets | " + " | ".join(_g(rep["gains"]["mean_of_sets"][v]) + f" ({rep['gains']['mean_of_sets'][v]['n_sets_positive']}/{len(sets)} sets > 0)" for v in PRIORS) + " |")
    L.append("\nPer direction (A->B / B->A):\n")
    L.append("| set | " + " | ".join(PRIORS) + " |")
    L.append("|---|" + "---|" * len(PRIORS))
    for sc in sets:
        L.append(f"| {sc} | " + " | ".join(" / ".join(f"{SCALE * rep['gains']['sets'][sc][v]['dirs'][d]['mean']:+.2f}" for d in rep["gains"]["sets"][sc][v]["dirs"]) for v in PRIORS) + " |")
    L.append("\n## Signals: coverage and Spearman correlation with the training half's zero-mean draft beta (mean over sets and directions)\n")
    L.append("| signal | coverage | all | C | U | R | M | land |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in ("ata", "luce5", "luceall"):
        m = rep["corr"]["mean"][s]
        L.append(f"| {s} | {100 * m['coverage']:.0f}% | " + " | ".join(_f(m[k], 2) for k in ["all"] + list(RARITY_ORDER)) + " |")
    L.append("\nPer set, correlation with all non-basic cards (ata / luce5 / luceall) and number of cards with signal / eligible:\n")
    L.append("| set | ata | luce5 | luceall | covered / eligible (luce5) |")
    L.append("|---|---|---|---|---|")
    for sc in sets:
        p = rep["corr"]["per_set"][sc]
        L.append(f"| {sc} | {_f(p['ata']['all'], 2)} | {_f(p['luce5']['all'], 2)} | {_f(p['luceall']['all'], 2)} | {p['luce5']['covered']:.0f} / {p['luce5']['eligible']:.0f} |")
    L.append("\n## Mapping to the beta scale (training half; exposure-weighted LS, per-rarity intercepts + one common slope on the standardised signal)\n")
    L.append("| prior | mean slope b (logit per SD of signal) | in-sample weighted R2 | R2 with per-rarity slopes |")
    L.append("|---|---|---|---|")
    for v in PRIORS:
        m = rep["maps"][v]
        L.append(f"| {v} | {_f(m['slope_mean'], 4)} | {_f(m['r2_mean'], 2)} | {_f(m['r2_per_group_slope_mean'], 2)} |")
    for key, title, note in (("mse", "target = the held-out half's zero-mean betas at the sigma its CV chose (SHRUNK toward 0 like the baseline: favours a zero-mean prior)",
                              "The target shares the baseline's shrinkage toward zero, so a prior mean away from zero is penalised by construction; read it with the weak-target table."),
                             ("mse_weak", f"target = the held-out half's betas at sigma {WEAK_SIGMA:g} (almost unshrunk: unbiased but noisy)",
                              "Held-out noise is common to every condition, so differences are noise-corrected; absolute MSE contains that noise.")):
        if not rep.get(key):
            continue
        L.append(f"\n## Weighted MSE of the fitted betas, {title} (logit^2 x 1e-4; diff = base - prior, positive = closer; card bootstrap)\n")
        L.append(note + "\n")
        for v in PRIORS:
            L.append(f"**{v}**\n")
            L.append("| group | n cards | MSE base | MSE prior | diff [95% CI] |")
            L.append("|---|---|---|---|---|")
            for gname, d in rep[key][v].items():
                L.append(f"| {gname} | {d['n']} | {1e4 * d['mse_base']:.2f} | {1e4 * d['mse_prior']:.2f} | {1e4 * d['diff']:+.2f} [{1e4 * d['lo']:+.2f}, {1e4 * d['hi']:+.2f}] |")
            L.append("")
    L.append("## Shift of the fitted betas toward the prior (by training-exposure tertile, T1 = low)\n")
    L.append("| prior | tertile | RMS of prior mean m | RMS(beta_prior - beta_base) | mean shift | regression slope of shift on m |")
    L.append("|---|---|---|---|---|---|")
    for v in PRIORS:
        for t in TERT + ("ALL",):
            d = rep["shift"][v].get(t)
            if d:
                L.append(f"| {v} | {t} | {d['rms_prior']:.4f} | {d['rms_shift']:.4f} | {d['mean_shift']:+.4f} | {d['slope_shift_on_prior']:.2f} |")
    L.append("\n## Sigma chosen by the grouped CV on the training half (A / B)\n")
    L.append("| set | " + " | ".join(VARIANTS) + " | CV loss prior - base (1e-4, mean of dirs): " + " / ".join(PRIORS) + " |")
    L.append("|---|" + "---|" * len(VARIANTS) + "---|")
    for sc in sets:
        sg = rep["sigma"][sc]
        L.append(f"| {sc} | " + " | ".join(" / ".join(f"{sg[d][v]:.3f}" for d in sg) for v in VARIANTS) + " | " + " / ".join(f"{SCALE * rep['cv'][sc][v]:+.1f}" for v in PRIORS) + " |")
    L.append("\n## Data and runtime\n")
    L.append("| set | draft events (A/B) | games train/held (A->B) | pick drafts | pick drafts in game events | game events with picks | seconds |")
    L.append("|---|---|---|---|---|---|---|")
    for sc in sets:
        m = rep["meta"][sc]
        g = next(iter(m["games"].values()))
        L.append(f"| {sc} | {m['n_events'][0]}/{m['n_events'][1]} | {g[0]}/{g[1]} | {m['n_pick_drafts']} | {100 * m['match_rate']:.1f}% | {m['game_events_with_picks']}/{m['n_game_events']} | {m['seconds']:.0f} |")
    L.append("\n## Approximations\n")
    for line in (__doc__ or "").split("Approximations (repeated in the report):")[-1].strip().splitlines():
        L.append(line)
    return "\n".join(L) + "\n"


def add_weak(sc: str, out: Path, drift_out: Path, tag: str = "") -> None:
    """Patch results/SET.pkl with the weak-prior (sigma WEAK_SIGMA) zero-mean fit of every direction's training half (results made before
    that target existed)."""
    import polars as pl

    from sealed_lab.model import cardvalue
    from sealed_lab.model.design import build_stage

    path = Path(out) / f"results{tag}" / f"{sc}.pkl"
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    ctx = SetContext(sc, out, drift_out)
    for d, run in obj["runs"].items():
        if run.get("beta_weak") is not None:
            continue
        g = ctx.games.filter(pl.col("draft_id").cast(pl.Utf8).is_in([str(i) for i in ctx.half_ids[d]]))
        st = build_stage("draft", g, ctx.fspace)
        spec = cardvalue.make_spec(st, ctx.n_card, ctx.n_feat, "draft")
        coef, _, _ = cardvalue.fit_stage(st, np.zeros(ctx.n_card + ctx.n_feat), spec, WEAK_SIGMA, with_se=False)
        run["beta_weak"] = ctx.setup.to_full(coef[st.prob.k:st.prob.k + ctx.n_card])
        run.setdefault("frac", 1.0)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"weak target added: {sc}", flush=True)


# =====================================================================================================================
# CLI
# =====================================================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("picks", "fit", "report", "run", "weak"):
        p = sub.add_parser(name)
        p.add_argument("--tag", default="", help="results folder suffix (results<tag>/) and report name suffix, e.g. _fine")
        p.add_argument("--grid", choices=("pipeline", "fine"), default="pipeline", help="draft sigma grid of the CV (fine = 15 points)")
        p.add_argument("--frac", type=float, default=1.0, help="train on this share of the training half's events (and pick drafts); 1 = all")
        p.add_argument("out", type=Path)
        p.add_argument("--sets", nargs="+", default=SETS)
        p.add_argument("--jobs", type=int, default=5)
        p.add_argument("--drift", type=Path, default=DEFAULT_HOMES.parent, help="draft_drift OUT folder holding homes/SET (scratch homes with the full corpora)")
    p = sub.add_parser("time-one")
    p.add_argument("set")
    p.add_argument("out", type=Path)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--drift", type=Path, default=DEFAULT_HOMES.parent)
    a = ap.parse_args(argv)
    if getattr(a, "grid", "pipeline") == "fine":
        from sealed_lab.model import cardvalue

        cardvalue.GRIDS_RANDOM["draft"] = GRID_FINE            # process-local: the app tree is not edited
    if a.cmd == "picks":
        for sc in a.sets:
            build_picks(sc, a.out)
    elif a.cmd in ("fit", "run"):
        for sc in a.sets:
            if a.cmd == "run" and not (a.out / "picks" / f"{sc}.npz").exists():
                build_picks(sc, a.out)
            run_set(sc, a.out, a.drift, a.jobs, tag=a.tag, frac=a.frac)
        if a.cmd == "run":
            build_report(a.out, a.sets, a.tag)
    elif a.cmd == "report":
        build_report(a.out, a.sets, a.tag)
    elif a.cmd == "weak":
        for sc in a.sets:
            add_weak(sc, a.out, a.drift, a.tag)
    elif a.cmd == "time-one":
        t0 = time.time()
        ctx = SetContext(a.set, a.out, a.drift)
        t1 = time.time()
        r = run_direction(ctx, "A", a.jobs, a.out / "tmp")
        print(f"{a.set}: set-up {t1 - t0:.0f}s, direction A {time.time() - t1:.0f}s ({r['n_games_train']} train games)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
