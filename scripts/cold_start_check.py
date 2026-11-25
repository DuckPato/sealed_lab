"""Pre-registered early-format (< 40k draft games) cold-start prior test on Sealed hold-outs (docs/cold_start/RULE.md).

Reads the app home and registered 17Lands pick files strictly read-only; writes scratch artefacts + JSON summary to OUT.

    python scripts/cold_start_check.py OUT [--sets ...] [--jobs N]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from sealed_lab import config  # noqa: E402
from sealed_lab.ingest import datasets, registry  # noqa: E402
from sealed_lab.model import attributes as attr_mod, cardvalue, castability, design, glm, pipeline, pooling, structure, validate  # noqa: E402
import pick_prior_check as ppc  # noqa: E402

ALL_SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
DRAFT_CAP = 25_000
BO1_CAP = 4_000
BO3_CAP = 1_500
PICK_DRAFT_CAP = 3_500
SE_FLOOR = 1e-5


def check_out_not_home(out: Path) -> None:
    home = config.app_home().resolve()
    out_r = out.resolve()
    if out_r == home or home in out_r.parents:
        raise RuntimeError(f"refusing to write inside app home {home}")


def read_phi_readonly(inp: pipeline.Inputs, cache_dir: Path, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    decks = inp.decks.sort("deck_key")
    keys = decks["deck_key"].to_numpy()
    hashes = decks["deck_hash"].to_numpy()
    sig = structure.cards_signature(inp.cards, 1, None)
    phi = np.full((len(keys), structure.N_TABLE), np.nan)
    cpath = pipeline.phi_cache_path(cache_dir, 1, None)
    if cpath.exists():
        with np.load(cpath, allow_pickle=False) as z:
            if str(z["sig"]) == sig and len(z["hashes"]):
                ch, cp = z["hashes"], z["phi"]
                order = np.argsort(ch)
                pos = np.minimum(np.searchsorted(ch[order], hashes), len(ch) - 1)
                hit = ch[order][pos] == hashes
                phi[hit] = cp[order][pos[hit]]
    todo = np.nonzero(np.isnan(phi[:, 0]))[0]
    if len(todo):
        offsets, cid, cnt = pipeline.deck_arrays(keys[todo], inp.deck_cards)
        phi[todo] = structure.compute_phi(offsets, cid, cnt, inp.cards, n_jobs=n_jobs, version=1, regime=None)
    return keys, phi


def truncate_chronological(g: pl.DataFrame, max_games: int, t_cut=None) -> tuple[pl.DataFrame, object]:
    """Keep earliest events (starting strictly before `t_cut` if given) up to `max_games` total games."""
    ev = design.event_starts(g)
    if t_cut is not None:
        ev = ev.filter(pl.col("start") < t_cut)
    if ev.height == 0:
        return g.head(0), None
    cum = np.cumsum(ev["n"].to_numpy())
    n_keep = int(np.searchsorted(cum, max_games, side="right")) + 1
    n_keep = max(min(n_keep, ev.height), min(ev.height, 15))
    kept = ev.head(n_keep)
    ids = kept["draft_id"].to_list()
    t_end = kept["start"].max()
    return g.filter(pl.col("draft_id").is_in(ids)), t_end


def find_pick_files(sc: str) -> list[Path]:
    """Locate registered or data-folder PremierDraft / TradDraft pick CSVs for `sc`."""
    found: dict[str, Path] = {}
    for f in registry.files(sc, include_shared=False):
        if f.kind == "draft" and f.event in ("PremierDraft", "TradDraft") and Path(f.path).exists():
            found[f.event] = Path(f.path)
    droot = Path(datasets.default_data_dir())
    for ev in ("PremierDraft", "TradDraft"):
        if ev in found:
            continue
        for sub in (droot / sc, droot):
            for nm in (f"draft_data_public.{sc}.{ev}.csv.gz", f"draft_data_public.{sc}.{ev}.csv"):
                p = sub / nm
                if p.exists():
                    found[ev] = p
                    break
    return [found[k] for k in ("PremierDraft", "TradDraft") if k in found]


def read_cold_picks(paths: list[Path], cold_draft_ids: set[str], t_end_iso: str | None) -> tuple[dict, list[str]] | None:
    """Stream pick files keeping ONLY drafts in `cold_draft_ids` (or earliest drafts <= `t_end_iso` up to PICK_DRAFT_CAP)."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    names_out: list[str] = []
    name_idx: dict[str, int] = {}
    kept_drafts: set[str] = set()
    out = dict(pick=[], taken=[], offsets=[], cards=[], counts=[])
    total_cards = 0

    for path in paths:
        op = gzip.open if str(path).endswith(".gz") else open
        with op(path, "rt", encoding="utf-8", newline="") as fh:
            hdr = next(csv.reader(fh))
        pack_cols = [c for c in hdr if c.startswith("pack_card_")]
        if not pack_cols:
            continue
        meta = ["draft_id", "draft_time", "pick_number", "pick"]
        for c in pack_cols:
            nm = c[len("pack_card_"):]
            if nm not in name_idx:
                name_idx[nm] = len(names_out)
                names_out.append(nm)
        gidx = np.array([name_idx[c[len("pack_card_"):]] for c in pack_cols], dtype=np.int64)
        col_types = {"draft_id": pa.string(), "draft_time": pa.string(), "pick_number": pa.int16(), "pick": pa.string(),
                     **{c: pa.int8() for c in pack_cols}}
        reader = pacsv.open_csv(str(path), read_options=pacsv.ReadOptions(block_size=24 << 20),
                                convert_options=pacsv.ConvertOptions(include_columns=meta + pack_cols, column_types=col_types))
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            if batch.num_rows == 0:
                continue
            did = batch.column("draft_id").to_pylist()
            dt = batch.column("draft_time").to_pylist()
            sel_rows = []
            for r_i, (d, t_s) in enumerate(zip(did, dt)):
                if d in cold_draft_ids:
                    kept_drafts.add(d)
                    sel_rows.append(r_i)
                elif d in kept_drafts:
                    sel_rows.append(r_i)
                elif len(kept_drafts) < PICK_DRAFT_CAP and (t_end_iso is None or (t_s is not None and t_s <= t_end_iso)):
                    kept_drafts.add(d)
                    sel_rows.append(r_i)
            if not sel_rows:
                continue
            sub = batch.take(pa.array(sel_rows, type=pa.int32()))
            mat = np.column_stack([sub.column(c).to_numpy(zero_copy_only=False) for c in pack_cols]).astype(np.int8)
            offs, cols, cnt = ppc.pack_csr(mat)
            picks = sub.column("pick").to_pylist()
            taken = np.array([name_idx.get(p, -1) if p is not None else -1 for p in picks], dtype=np.int32)
            out["pick"].append(sub.column("pick_number").to_numpy(zero_copy_only=False).astype(np.int8))
            out["taken"].append(taken)
            out["offsets"].append(offs[1:] + total_cards)
            out["cards"].append(gidx[cols.astype(np.int64)].astype(np.int16))
            out["counts"].append(cnt)
            total_cards += len(cols)

    if not out["taken"]:
        return None
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["offsets"] = np.concatenate([[0], res["offsets"]]).astype(np.int64)
    res["n_drafts"] = len(kept_drafts)
    return res, names_out


def build_pick_prior_mean(cards: pl.DataFrame, space: design.DeckSpace, st_draft: design.StageData,
                          beta_draft_zero: np.ndarray, pick_arr: dict, pick_names: list[str], signal_kind: str = "luce5") -> np.ndarray:
    """Compute card prior mean (length n_card) on early draft from LUCE5 or ATA mapped onto `beta_draft_zero`."""
    n_names = len(pick_names)
    pos = pick_arr["pick"].astype(float) + 1.0
    taken = pick_arr["taken"].astype(np.int64)
    if signal_kind == "ata":
        sig_names, _ = ppc.ata_signal(taken, pos, n_names)
    else:
        early = np.nonzero(pos <= ppc.EARLY_PICKS)[0]
        off, cds, cnt = ppc.subset_picks(pick_arr["offsets"], pick_arr["cards"], pick_arr["counts"], early)
        theta, takes, _ = ppc.luce_fit(taken[early], off, cds, cnt, n_names)
        sig_names = np.where(takes >= ppc.MIN_TAKEN, theta, np.nan)

    csort = cards.sort("card_id")
    cnames = csort["name"].to_list()
    cid_of = {n: i for i, n in enumerate(cnames)}
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
    expo_full[space.nb_ids] = st_draft.exposure
    beta_full = np.zeros(csort.height)
    beta_full[space.nb_ids] = beta_draft_zero

    elig = nonbasic & (expo_full > 0)
    z, ok = ppc.standardise(np.where(elig, sig_full, np.nan))
    cov = ok & elig
    groups = ppc.mapping_groups(rarity, cov, nonbasic)
    fm = ppc.fit_map(z, beta_full, expo_full, groups, cov, use_signal=bool(cov.sum() >= 15))
    m_full = np.where(nonbasic, fm["pred"], 0.0)
    return m_full[space.nb_ids]


def run_chain_with_draft_prior(stages: dict[str, design.StageData], draft_card_prior: np.ndarray,
                               n_card: int, n_feat: int, tmpdir: str, n_jobs: int, seed: int,
                               prior_ok: set[str], opts: cardvalue.ChainOpts) -> dict[str, cardvalue.StageFit]:
    """Like cardvalue.run_chain, except the first ('draft') stage shrinks towards `draft_card_prior` instead of 0."""
    order = [n for n in ("draft", "bo1", "bo3") if n in stages]
    fits: dict[str, cardvalue.StageFit] = {}
    prev: cardvalue.StageFit | None = None
    p_deck = n_card + n_feat
    for i, name in enumerate(order):
        st = stages[name]
        lam = lam_se = lam_from = None
        if prev is None:
            prior = np.zeros(p_deck)
            if name == "draft":
                prior[:n_card] = draft_card_prior
        elif prev.name == "draft":
            lam_raw, lam_se = cardvalue.fit_lambda(st, prev.deck)
            lam = float(np.clip(lam_raw, *cardvalue.LAMBDA_CLIP))
            lam_from = prev.name
            prior = lam * prev.deck
        else:
            prior = prev.deck.copy()
            lam_from = prev.name
        s_dir = os.path.join(tmpdir, f"st_{name}")
        os.makedirs(s_dir, exist_ok=True)
        fits[name] = cardvalue._stage_fit(st, prior, n_card, n_feat, s_dir, n_jobs, seed + i, None, None,
                                          lam, lam_se, lam_from, opts=opts)
        if name in prior_ok:
            prev = fits[name]
    return fits


def paired_gain_metrics(y: np.ndarray, p_base: np.ndarray, p_new: np.ndarray, groups: np.ndarray, is_first: np.ndarray,
                        n_rep: int = 500, seed: int = 0) -> dict:
    met = validate.cluster_metrics(y, {"M": p_base, "NEW": p_new}, groups, n_rep, seed + 5)
    d = met["NEW"]["vs_M"]["logloss"]
    halves = validate.half_gains(y, p_base, p_new, groups, is_first, n_rep, seed + 7)
    return dict(gain=float(-d["diff"]), lo=float(-d["hi"]), hi=float(-d["lo"]),
                logloss_base=float(met["M"]["logloss"]), logloss_new=float(met["NEW"]["logloss"]),
                halves=halves)


def pool_with_floor(cells: dict[str, dict], se_floor: float) -> dict:
    out = {}
    for p in pooling.PARTS:
        ys = [c[p]["gain"] for c in cells.values()]
        ses = [max(pooling.se_from_ci(c[p]["lo"], c[p]["hi"]), se_floor) for c in cells.values()]
        out[p] = pooling.dersimonian_laird(ys, ses)
    return dict(pooled=out, verdict=pooling.verdict(out))


def run_set(sc: str, out: Path, n_jobs: int) -> dict:
    t0 = time.time()
    inp = pipeline.load_inputs(sc)
    inp.cards = castability.with_cost_groups(inp.cards, sc)
    rep_path = config.model_dir(sc) / "report.json"
    ed_applied = False
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            ed_applied = bool((rep.get("extra_draft") or {}).get("applied", False))
        except Exception:
            ed_applied = False
    if pipeline.extra_draft_events(inp.games) and not ed_applied:
        inp = pipeline.without_extra_draft(inp)

    keys, table = read_phi_readonly(inp, config.cache_dir(sc), n_jobs)
    space, _ = pipeline.plain_space(inp, keys, table)
    A_mat, A_names = attr_mod.attribute_matrix(inp.cards)
    a_cols = attr_mod.select_columns(A_mat, space.nb_ids)
    space_a = design.with_attributes(space, A_mat, A_names, a_cols) if len(a_cols) else space

    # Load mature model's card betas (read-only) for card-value correlation check if present
    mature_betas: dict[str, np.ndarray] = {}
    draws_path = config.model_dir(sc) / "draws.npz"
    if draws_path.exists():
        with np.load(draws_path, allow_pickle=False) as z:
            for f in ("bo1", "bo3"):
                if f"{f}__beta" in z:
                    mature_betas[f] = z[f"{f}__beta"][0][space.nb_ids]

    games = inp.games
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    formats = [f for f in pipeline.FMT_ORDER if n_games.get(f, 0) >= 1]
    opts = cardvalue.ChainOpts()
    pfiles = find_pick_files(sc)

    fmt_res: dict[str, dict] = {}
    pick_cache: dict = {}

    for i, fmt in enumerate(formats):
        seed_f = i
        gT = games.filter(pl.col("fmt") == fmt)
        split = validate._holdout_split(gT)
        if split is None:
            fmt_res[fmt] = dict(status="skipped", reason=f"too few {fmt} events/games")
            continue
        hold_ids, t_cut, _nev, _nho = split
        g_tr_full = gT.filter(~pl.col("draft_id").is_in(hold_ids))
        g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))

        # Cold-start truncated training slices strictly before t_cut
        cap_t = BO3_CAP if fmt == "bo3" else BO1_CAP
        g_tr_cold, _ = truncate_chronological(g_tr_full, cap_t)
        if g_tr_cold.height < 200:
            fmt_res[fmt] = dict(status="skipped", reason=f"too few cold {fmt} games ({g_tr_cold.height})")
            continue

        g_draft_cold, t_draft_end = truncate_chronological(games.filter(pl.col("fmt") == "draft"), DRAFT_CAP, t_cut=t_cut)
        g_bo1_cold, _ = truncate_chronological(games.filter(pl.col("fmt") == "bo1"), BO1_CAP, t_cut=t_cut) if fmt == "bo3" else (pl.DataFrame(), None)

        def make_cold_stages(sp: design.DeckSpace) -> dict[str, design.StageData]:
            sts: dict[str, design.StageData] = {}
            if g_draft_cold.height >= validate.MIN_PRIOR_GAMES:
                sts["draft"] = design.build_stage("draft", g_draft_cold, sp)
            if fmt == "bo3" and g_bo1_cold.height >= validate.MIN_PRIOR_GAMES:
                sts["bo1"] = design.build_stage("bo1", g_bo1_cold, sp)
            sts[fmt] = design.build_stage(fmt, g_tr_cold, sp)
            return sts

        tmp_f = out / "tmp" / f"{sc}_{fmt}"
        (tmp_f / "m").mkdir(parents=True, exist_ok=True)
        (tmp_f / "ma").mkdir(parents=True, exist_ok=True)

        # 1. Baseline M_cold
        st_m = make_cold_stages(space)
        prior_ok = set(st_m) - {fmt}
        fits_m = cardvalue.run_chain(st_m, space.n_card, space.n_feat, str(tmp_f / "m"), n_jobs, seed=seed_f + 100,
                                     prior_ok=prior_ok, opts=opts)
        prob_ho_m, _, _ = design.make_problem(g_ho, st_m[fmt].enc, space, clamp_time=True)
        p_m = expit(prob_ho_m.eta(fits_m[fmt].coef))
        y = prob_ho_m.y
        groups = np.unique(g_ho["draft_id"].to_numpy(), return_inverse=True)[1]
        is_first = validate.half_masks(g_ho)

        # 2. Card-attribute prior MA_cold
        st_ma = make_cold_stages(space_a)
        fits_ma = cardvalue.run_chain(st_ma, space.n_card, space_a.n_feat, str(tmp_f / "ma"), n_jobs, seed=seed_f + 100,
                                      prior_ok=set(st_ma) - {fmt}, attr_load=space_a.attr_load, opts=opts)
        prob_ho_ma, _, _ = design.make_problem(g_ho, st_ma[fmt].enc, space_a, clamp_time=True)
        p_ma = expit(prob_ho_ma.eta(fits_ma[fmt].coef))
        met_ma = paired_gain_metrics(y, p_m, p_ma, groups, is_first, seed=seed_f)

        # Mature beta correlations
        if fmt in mature_betas:
            mb = mature_betas[fmt]
            met_ma["corr_M_cold"] = float(np.corrcoef(fits_m[fmt].beta_total, mb)[0, 1])
            met_ma["corr_MA_cold"] = float(np.corrcoef(fits_ma[fmt].beta_total, mb)[0, 1])

        # 3. Pick-order priors (LUCE5 and ATA) when draft stage + pick files exist
        met_luce5 = None
        met_ata = None
        if "draft" in st_m and pfiles:
            ckey = (int(g_draft_cold.height), str(t_draft_end))
            if ckey not in pick_cache:
                cold_ids = set(g_draft_cold["draft_id"].to_list())
                t_iso = str(t_draft_end)[:19] if t_draft_end is not None else None
                pick_cache[ckey] = read_cold_picks(pfiles, cold_ids, t_iso)
            loaded = pick_cache[ckey]
            if loaded is not None:
                parr, pnames = loaded
                beta_d0 = fits_m["draft"].u
                for kind_s, tag_s in (("luce5", "luce5"), ("ata", "ata")):
                    pm_cards = build_pick_prior_mean(inp.cards, space, st_m["draft"], beta_d0, parr, pnames, kind_s)
                    (tmp_f / tag_s).mkdir(parents=True, exist_ok=True)
                    fits_pk = run_chain_with_draft_prior(st_m, pm_cards, space.n_card, space.n_feat,
                                                         str(tmp_f / tag_s), n_jobs, seed_f + 100, prior_ok, opts)
                    p_pk = expit(prob_ho_m.eta(fits_pk[fmt].coef))
                    m_pk = paired_gain_metrics(y, p_m, p_pk, groups, is_first, seed=seed_f)
                    m_pk["n_pick_drafts"] = int(parr["n_drafts"])
                    m_pk["n_picks"] = int(len(parr["taken"]))
                    if fmt in mature_betas:
                        m_pk["corr_pick"] = float(np.corrcoef(fits_pk[fmt].beta_total, mature_betas[fmt])[0, 1])
                    if kind_s == "luce5":
                        met_luce5 = m_pk
                    else:
                        met_ata = m_pk

        fmt_res[fmt] = dict(
            status="ok",
            n_draft_cold=int(g_draft_cold.height),
            n_target_cold=int(g_tr_cold.height),
            n_holdout_games=int(len(y)),
            n_holdout_events=int(groups.max()) + 1,
            MA_cold=met_ma,
            LUCE5_cold=met_luce5,
            ATA_cold=met_ata,
        )
        l5_txt = f"{met_luce5['gain']*1e4:+6.2f} [{met_luce5['lo']*1e4:+6.2f},{met_luce5['hi']*1e4:+6.2f}]" if met_luce5 else "n/a"
        ata_txt = f"{met_ata['gain']*1e4:+6.2f}" if met_ata else "n/a"
        print(f"[{sc} {fmt}] d_cold={g_draft_cold.height:5d} t_cold={g_tr_cold.height:4d} ho={len(y):5d} | "
              f"MA={met_ma['gain']*1e4:+6.2f} [{met_ma['lo']*1e4:+6.2f},{met_ma['hi']*1e4:+6.2f}] | "
              f"LUCE5={l5_txt} | ATA={ata_txt}", flush=True)

    return dict(set=sc, seconds=round(time.time() - t0, 1), has_picks=bool(pfiles), formats=fmt_res)


def summarize_arm(per_set: dict[str, dict], arm_key: str) -> dict:
    out_fmt = {}
    for fmt in ("bo1", "bo3"):
        cells = {}
        excluded = []
        for sc, r in per_set.items():
            fr = r["formats"].get(fmt)
            if not fr or fr.get("status") != "ok":
                excluded.append(dict(set=sc, reason=(fr or {}).get("reason", "absent")))
                continue
            m = fr.get(arm_key)
            if m is None:
                excluded.append(dict(set=sc, reason="no pick files"))
                continue
            cell = pooling.make_cell(m, m["halves"])
            if cell is None:
                excluded.append(dict(set=sc, reason="zero-width or non-finite CI"))
            else:
                cells[sc] = cell
        if not cells:
            out_fmt[fmt] = dict(n_sets=0, excluded=excluded)
            continue
        raw_pool = pooling.pool_parts(list(cells.values()))
        raw_v = pooling.verdict(raw_pool)
        floored = pool_with_floor(cells, SE_FLOOR)
        out_fmt[fmt] = dict(
            n_sets=len(cells), sets=sorted(cells), excluded=excluded,
            raw=dict(pooled=raw_pool, verdict=raw_v),
            se_floored=floored,
        )
    qual_raw = pooling.qualifies({f: v["raw"]["verdict"] for f, v in out_fmt.items() if v.get("n_sets")})
    qual_fl = pooling.qualifies({f: v["se_floored"]["verdict"] for f, v in out_fmt.items() if v.get("n_sets")})
    return dict(formats=out_fmt, passes_rule=bool(qual_raw and qual_fl))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?", default=Path(tempfile.gettempdir()) / "sealed_lab_cold_start")
    ap.add_argument("--sets", nargs="+", default=ALL_SETS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args(argv)
    out = a.out.resolve()
    check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)

    per_set: dict[str, dict] = {}
    t_all = time.time()
    with glm.shared_pool(a.jobs):
        for sc in a.sets:
            per_set[sc] = run_set(sc.upper(), out, a.jobs)
            cardvalue.cleanup(str(out / "tmp"))

    summary = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_seconds=round(time.time() - t_all, 1),
        draft_cap=DRAFT_CAP, bo1_cap=BO1_CAP, bo3_cap=BO3_CAP, se_floor=SE_FLOOR,
        per_set=per_set,
        pooled={arm: summarize_arm(per_set, arm) for arm in ("MA_cold", "LUCE5_cold", "ATA_cold")},
    )
    out_json = out / "cold_start_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json} in {summary['total_seconds']:.1f}s")
    for arm, sm in summary["pooled"].items():
        print(f"=== {arm} (PASS={sm['passes_rule']}) ===")
        for fmt, fv in sm["formats"].items():
            if fv.get("n_sets"):
                rw = fv["raw"]["pooled"]["whole"]
                ra = fv["raw"]["pooled"]["A"]
                rb = fv["raw"]["pooled"]["B"]
                rv = fv["raw"]["verdict"]
                flv = fv["se_floored"]["verdict"]
                print(f"  {fmt} ({fv['n_sets']} sets): whole={rw['mean']*1e4:+6.2f} [{rw['lo']*1e4:+6.2f},{rw['hi']*1e4:+6.2f}] | "
                      f"A={ra['mean']*1e4:+6.2f} [{ra['lo']*1e4:+6.2f},{ra['hi']*1e4:+6.2f}] | "
                      f"B={rb['mean']*1e4:+6.2f} (lo90={rv['B_lo90']*1e4:+6.2f}) | raw={rv['label']} | floored={flv['label']}")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
