"""sealed_lab/provisional.py: the provisional model from 17Lands pick files (docs/pick_model/RULE.md): deck rebuild, record expansion,
card list from the pick file, the artifact (SetModel.load, scoring, draws), the transfer file, the routing of api.calculate_set,
the stale / replaced logic of api.set_status and the transfer tool."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sealed_lab import api, config, provisional as prov
from sealed_lab.ingest import registry
from sealed_lab.model import structure
from sealed_lab.types import FileInfo

ROOT = Path(__file__).resolve().parents[1]
SET = "TST"
N_RUNS = 120


# ---------------------------------------------------------------------------------------------- synthetic pick data
def _world():
    from sealed_lab.synth.generate import _scryfall_objects, build_world

    w = build_world(SET, 0)
    return w, _scryfall_objects(w)


def write_pick_file(path: Path, n_runs: int = N_RUNS, seed: int = 3, late_run: bool = False, header_names: int = 6) -> list[str]:
    """A PremierDraft pick file of `n_runs` drafts: each drafts 45 cards of two colors (+ colorless) and plays 23 of them
    (pick_maindeck_rate 1), the rest 0; wins / losses written as floats ("3.0") like some 17Lands files. Returns the names it uses."""
    w, objs = _world()
    rng = np.random.default_rng(seed)
    by_color: dict[str, list[str]] = {c: [] for c in "WUBRGC"}
    for o in objs:
        if o["name"] in ("Plains", "Island", "Swamp", "Mountain", "Forest") or "Land" in o["type_line"]:
            continue
        cols = o.get("colors") or []
        if len(cols) <= 1:
            by_color["C" if not cols else cols[0]].append(o["name"])
    pairs = ["WU", "WB", "UB", "BR", "RG", "WG", "UR"]
    rows = []
    t0 = datetime(2026, 11, 10, 12, 0)
    used: set[str] = set()
    for i in range(n_runs):
        pair = pairs[i % len(pairs)]
        pool = [n for c in pair for n in by_color[c]]
        pool_c = by_color["C"]
        pick_names = list(rng.choice(pool, size=min(len(pool), 33), replace=False)) + list(rng.choice(pool_c, size=min(len(pool_c), 12), replace=False))
        main = set(rng.choice(len(pick_names), size=23, replace=False).tolist())
        wins = float(rng.integers(0, 8))
        losses = float(rng.integers(0, 4)) if wins < 7 else float(rng.integers(0, 3))
        when = t0 + timedelta(hours=i * 2) if not (late_run and i == n_runs - 1) else t0 + timedelta(days=30)
        for k, nm in enumerate(pick_names):
            used.add(nm)
            rows.append(dict(expansion=SET, event_type="PremierDraft", draft_id=f"d{i:04d}", draft_time=when.strftime("%Y-%m-%d %H:%M:%S"),
                             rank="gold", event_match_wins=wins, event_match_losses=losses, user_game_win_rate_bucket=0.5 + 0.05 * (i % 3),
                             pack_number=k // 15, pick_number=k % 15, pick=nm, pick_maindeck_rate=1.0 if k in main else 0.0))
    df = pl.DataFrame(rows)
    for nm in sorted(used)[:header_names]:
        df = df.with_columns(pl.lit(0).alias("pack_card_" + nm), pl.lit(0).alias("pool_" + nm))
    with gzip.open(path, "wb") as f:
        f.write(df.write_csv().encode("utf-8"))
    return sorted(used)


@pytest.fixture(scope="module")
def pick_dir(tmp_path_factory):
    from sealed_lab.synth.generate import _scryfall_objects, _write_cards_csv, build_world

    d = tmp_path_factory.mktemp("pickdata")
    names = write_pick_file(d / f"draft_data_public.{SET}.PremierDraft.csv.gz")
    w = build_world(SET, 0)
    (d / f"scryfall_{SET}.json").write_text(json.dumps(_scryfall_objects(w), ensure_ascii=False), encoding="utf-8")
    _write_cards_csv(d / "cards.csv", w)
    return d, names


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("SEALED_LAB_HOME", str(h))
    return h


@pytest.fixture()
def registered(home, pick_dir):
    d, names = pick_dir
    api.add_files([str(d)])
    return names


def _file_info(path: Path, kind: str, event: str, fmt: str) -> FileInfo:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 64)
    return FileInfo(path=str(path), kind=kind, set_code=SET, event=event, fmt=fmt, size=64, fingerprint=f"{kind}{event}"[:20].ljust(20, "0"), note="")


# ---------------------------------------------------------------------------------------------- the shared tested functions
def test_script_and_app_share_the_functions():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("pick_model_check", ROOT / "scripts" / "pick_model_check.py")
    pm = importlib.util.module_from_spec(spec)
    sys.modules["pick_model_check"] = pm
    spec.loader.exec_module(pm)
    for name in ("read_pick_file", "make_ctx", "color_symbols", "split_basics", "rebuild_deck", "expand_records", "runs_to_games",
                 "phi_for_decks", "draft_chain"):
        assert getattr(pm, name) is getattr(prov, name), name
    assert pm.WINDOW_DAYS == 12 and pm.MAIN_RATE == prov.MAIN_RATE


def test_expand_records_wins_then_losses():
    idx, won = prov.expand_records(np.array([2, 0, 1]), np.array([1, 2, 0]))
    assert idx.tolist() == [0, 0, 0, 1, 1, 2]
    assert won.tolist() == [True, True, False, False, False, True]


def test_build_runs_uses_every_run_without_a_window(registered, tmp_path):
    late = tmp_path / "late"
    late.mkdir()
    write_pick_file(late / f"draft_data_public.{SET}.PremierDraft.csv.gz", n_runs=40, late_run=True)
    cards, _w = prov.build_cards(SET, prov.pick_card_names([late / f"draft_data_public.{SET}.PremierDraft.csv.gz"]), fetch_scryfall=False)
    ctx = prov.make_ctx(cards)
    picks = prov.read_pick_file(late / f"draft_data_public.{SET}.PremierDraft.csv.gz")
    all_runs = prov.build_runs(picks, ctx)                       # the app: no window
    assert all_runs.info["window_days"] is None and all_runs.info["runs_in_window"] == 40
    win = prov.build_runs(picks, ctx, days=12)                   # the test's 12-day window drops the run of day 30
    assert win.info["runs_in_window"] == 39
    # a run with 23 played cards: 40 cards after the basics
    for deck in all_runs.decks:
        assert sum(deck.values()) == 40


# ---------------------------------------------------------------------------------------------- the card list
def test_card_list_from_the_pick_file(registered, pick_dir):
    d, names = pick_dir
    path = d / f"draft_data_public.{SET}.PremierDraft.csv.gz"
    got = prov.pick_card_names([path], prov.read_pick_file(path))
    assert set(names) <= set(got)                                # every picked name (header columns and pick values)
    assert {"Plains", "Island", "Swamp", "Mountain", "Forest"} <= set(got)
    cards, warns = prov.build_cards(SET, got, fetch_scryfall=False)
    assert cards.height == len(got) and cards["card_id"].to_list() == list(range(cards.height))
    assert (cards["meta_source"] == "scryfall").sum() >= len(names)       # the registered Scryfall file gave their metadata
    assert cards.filter(pl.col("is_basic")).height == 5 and not warns


def test_double_faced_names_are_listed_once(tmp_path):
    p = tmp_path / f"draft_data_public.{SET}.PremierDraft.csv.gz"
    df = pl.DataFrame(dict(draft_id=["a"], pick=["Front Face"], **{"pack_card_Front Face // Back Face": [1], "pool_Front Face // Back Face": [0]}))
    with gzip.open(p, "wb") as f:
        f.write(df.write_csv().encode("utf-8"))
    got = prov.pick_card_names([p], df)
    assert got.count("Front Face // Back Face") == 1 and "Front Face" not in got


def test_no_card_metadata_is_a_plain_error(home, pick_dir):
    d, _ = pick_dir
    registry.register(__import__("sealed_lab.ingest.detect", fromlist=["detect_file"]).detect_file(d / f"draft_data_public.{SET}.PremierDraft.csv.gz"))
    path = d / f"draft_data_public.{SET}.PremierDraft.csv.gz"
    with pytest.raises(prov.ProvisionalError, match="card data"):
        prov.build_cards(SET, prov.pick_card_names([path], prov.read_pick_file(path)), fetch_scryfall=False)


# ---------------------------------------------------------------------------------------------- the transfer file
def test_shipped_transfer_file_is_valid_and_has_no_paths():
    t = prov.load_transfer()
    assert t["feature_names"] == structure.FEATURE_NAMES and t["n_feat"] == structure.N_FEATURES
    assert len(t["sets"]) == 5 and 0.5 < t["lam_bar"] < 1.2
    for f in ("bo1", "bo3"):
        assert len(t["formats"][f]["theta"]) == structure.N_FEATURES and t["formats"][f]["cov"]["mull2p"] < t["formats"][f]["cov"]["mull1"] < 0
    text = (ROOT / "sealed_lab" / "data" / "provisional_transfer.json").read_text(encoding="utf-8")
    assert "Users" not in text and "AppData" not in text and ":\\" not in text


def test_transfer_mismatch_is_refused_with_a_clear_message(tmp_path):
    t = json.loads((ROOT / "sealed_lab" / "data" / "provisional_transfer.json").read_text(encoding="utf-8"))
    bad = dict(t, feature_names=t["feature_names"][:-1], n_feat=t["n_feat"] - 1)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(prov.TransferMismatch, match="other deck features"):
        prov.load_transfer(p)
    p.write_text(json.dumps(dict(t, feature_names=list(reversed(t["feature_names"])))), encoding="utf-8")
    with pytest.raises(prov.TransferMismatch):
        prov.load_transfer(p)
    with pytest.raises(prov.TransferMismatch, match="could not be read"):
        prov.load_transfer(tmp_path / "missing.json")


def test_calculate_refuses_a_mismatching_transfer_before_any_work(registered, tmp_path, monkeypatch):
    t = json.loads((ROOT / "sealed_lab" / "data" / "provisional_transfer.json").read_text(encoding="utf-8"))
    p = tmp_path / "t.json"
    p.write_text(json.dumps(dict(t, n_feat=3)), encoding="utf-8")
    monkeypatch.setattr(prov, "TRANSFER_PATH", p)
    with pytest.raises(prov.TransferMismatch):
        prov.calculate(SET, n_jobs=1, n_boot=1, fetch_scryfall=False)
    assert not (config.model_dir(SET) / "meta.json").exists()


# ---------------------------------------------------------------------------------------------- the calculation
@pytest.fixture(scope="module")
def calculated(tmp_path_factory, pick_dir):
    """One provisional calculation of the synthetic pick set (own home), shared by the tests below."""
    mp = pytest.MonkeyPatch()
    h = tmp_path_factory.mktemp("prov_home")
    mp.setenv("SEALED_LAB_HOME", str(h))
    d, names = pick_dir
    api.add_files([str(d)])
    seen: list[tuple[float, str]] = []
    meta = prov.calculate(SET, progress=lambda f, m: seen.append((f, m)), n_jobs=1, n_boot=4, fetch_scryfall=False)
    yield dict(home=h, meta=meta, names=names, progress=seen)
    mp.undo()


def test_artifact_loads_and_scores_a_deck(calculated):
    from sealed_lab.model.artifact import SetModel

    meta = calculated["meta"]
    m = SetModel.load(SET)
    assert m.formats == ["bo3", "bo1"] and m.meta["provisional"]["source"] == "17Lands pick files"
    assert m.n_draws("bo1") == 5 and m.n_draws("bo3") == 5                       # draw 0 + n_boot refits
    assert m.feature_names == structure.FEATURE_NAMES and m.meta["n_draws"] == 5
    # a deck: 17 basics split over the colors of 23 real cards of the synthetic set
    cards = m.cards.sort("card_id")
    nb = cards.filter(~pl.col("is_basic") & ~pl.col("is_land"))
    deck = {int(c): 1 for c in nb["card_id"].to_list()[:23]}
    for nm, k in (("Plains", 9), ("Island", 8)):
        deck[int(cards.filter(pl.col("name") == nm)["card_id"][0])] = k
    for fmt in ("bo1", "bo3"):
        ev = m.evaluate(deck, fmt)
        assert 0.2 < ev.game_wr < 0.9 and 0.0 <= ev.score <= 100.0 and ev.game_wr_lo <= ev.game_wr <= ev.game_wr_hi
        assert "mulligans" in ev.breakdown and not ev.warnings[:0]
    assert meta["provisional"]["n_runs"] > 0 and meta["provisional"]["n_games"] > 0


def test_draws_card_values_and_transferred_terms(calculated):
    from sealed_lab.model.artifact import SetModel

    t = prov.load_transfer()
    m = SetModel.load(SET)
    for fmt in ("bo1", "bo3"):
        dr = m._draws[fmt]
        assert dr["beta"].shape[0] == 5 and dr["theta"].shape == (5, structure.N_FEATURES) and dr["ref"].shape == (5,)
        assert np.allclose(dr["theta"], np.array(t["formats"][fmt]["theta"], dtype=np.float32), atol=1e-6)     # the same transferred terms in every draw
        assert np.allclose(dr["ref"], t["formats"][fmt]["ref"]) and np.allclose(dr["mull"][0], [t["formats"][fmt]["mull1"], t["formats"][fmt]["mull2p"]], atol=1e-5)
        assert not np.allclose(dr["beta"][1], dr["beta"][0]) and not np.allclose(dr["beta"][1], dr["beta"][2])  # draws 1.. are refits on resampled events
    assert np.allclose(m._draws["bo3"]["beta"], m._draws["bo1"]["beta"])                                         # Bo3 = Bo1 card values
    nbm = ~m._is_basic
    assert np.all(m._draws["bo1"]["beta"][:, ~nbm] == 0)                                                         # basics carry no value
    ref = m._ref["bo1"]
    assert ref.shape[0] == 5 and np.all(np.diff(ref, axis=1) >= 0)                                               # sorted real-build scores per draw


def test_meta_says_provisional_and_what_it_used(calculated):
    meta = json.loads((config.model_dir(SET) / "meta.json").read_text(encoding="utf-8"))
    p = meta["provisional"]
    assert p["events"] == ["PremierDraft"] and p["result"] == "docs/pick_model/RESULT.md" and p["transfer"]["sets"] == prov.load_transfer()["sets"]
    assert set(p["dropped"]) == set(prov.DROP_REASONS) and p["runs_in_file"] == N_RUNS
    assert p["n_runs"] + sum(p["dropped"].values()) == N_RUNS
    assert meta["source_files"] and all(s.startswith(("draft:", "cards:", "scryfall:")) for s in meta["source_files"])
    assert meta["validation"] == {} and meta["n_games"] == {"draft": p["n_games"]}
    assert api.calibration_note(SET, "bo1") is None


def test_draft_formats_option_is_ignored_by_a_provisional_calculation_and_never_marks_it_stale(calculated):
    api.set_option(SET, "draft_formats", True)
    try:
        assert not api.set_status(SET).stale                                       # draft models need game data: nothing to wait for
        st = api.calculate_set(SET, draft_formats=True, n_jobs=1, n_boot=4, fetch_scryfall=False)
        assert st.provisional and st.formats == ["bo3", "bo1"] and not st.stale
    finally:
        api.set_option(SET, "draft_formats", False)


def test_set_status_marks_it_provisional_and_not_stale(calculated):
    st = api.set_status(SET)
    assert st.provisional and st.has_model and not st.has_corpus and not st.stale and st.pending == [] and st.formats == ["bo3", "bo1"]
    ov = api.set_overview(SET)
    assert ov["model"]["provisional"]["n_runs"] > 0


def test_provisional_set_refuses_what_needs_game_data(calculated):
    for fn, args in ((api.search_rebuilds, (SET,)), (api.set_report, (SET, "bo1")), (api.search_sideboarding, (SET,)), (api.mulligan_summary, (SET, "bo1"))):
        with pytest.raises(ValueError, match="Not available yet"):
            fn(*args)
    assert api.card_info(SET) and api.check_pool(SET, "2 Ancient Beast\n1 Ancient Bloom")["n_copies"] == 3        # the Deck builder's calls work


def test_deck_builder_works_on_the_provisional_model(calculated):
    """The Deck builder (families, evaluation, explanation without a corpus) runs unchanged on a provisional model."""
    info = api.card_info(SET)
    pool: dict[int, int] = {}
    for cid, c in info.items():                               # a two-color pool of 50 cards
        if not c["is_land"] and not c["is_basic"] and set(c["colors"]) <= set("WU") and len(pool) < 50:
            pool[cid] = 1
    res = api.build_from_pool(SET, pool, "bo1", n_decks=1)
    assert res.decks and sum(c.count for c in res.decks[0].cards) == 40
    ev = api.evaluate_deck(SET, {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}, "bo1")
    assert 0.2 < ev.game_wr < 0.9
    with pytest.raises(Exception):                            # (optional: the waterfall needs the corpus's real builds; the Deck builder catches this)
        api.explain_deck(SET, {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}, "bo1")


def test_a_game_file_makes_the_set_stale_and_a_normal_calculation_replaces_it(calculated, monkeypatch):
    """The provisional model is replaced once a game file is added: stale at once, and the next Calculate is the normal one (stubbed here:
    its own tests run the real thing) and leaves nothing provisional behind."""
    import sealed_lab.ingest.corpus as corpus_mod
    from sealed_lab.model import artifact, pipeline

    cache = config.cache_dir(SET) / "phi_provisional.npz"
    assert cache.exists()
    registry.register(_file_info(calculated["home"] / "game_data_public.TST.Sealed.csv.gz", "game", "Sealed", "bo1"))
    st = api.set_status(SET)
    assert st.provisional and st.stale and any("game data arrived" in x for x in st.pending)
    calls = []
    monkeypatch.setattr(corpus_mod, "build_corpus", lambda *a, **k: calls.append("corpus") or {})

    def fake_normal(set_code, **kw):
        calls.append("normal")
        d = config.model_dir(set_code)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        meta = {k: v for k, v in meta.items() if k != "provisional"}      # (save_artifact writes a whole new meta: the flag cannot survive)
        meta["source_files"] = api.model_input_signature(registry.files(set_code))
        artifact._atomic_write(d / "meta.json", json.dumps(meta).encode("utf-8"))
        return meta

    monkeypatch.setattr(pipeline, "calculate", fake_normal)
    monkeypatch.setattr(api, "_precompute_analysis", lambda sc: None)
    monkeypatch.setattr(prov, "calculate", lambda *a, **k: calls.append("provisional"))
    api.calculate_set(SET, fetch_scryfall=False)
    assert calls == ["corpus", "normal"]                                  # the provisional path was NOT taken
    assert not api.set_status(SET).provisional and not prov.is_provisional(SET)
    assert not cache.exists()


# ---------------------------------------------------------------------------------------------- routing of api.calculate_set
def test_routing_pick_only_goes_to_the_provisional_path(registered, monkeypatch):
    import sealed_lab.ingest.corpus as corpus_mod

    called = []
    monkeypatch.setattr(prov, "calculate", lambda sc, **kw: called.append((sc, sorted(kw))))
    monkeypatch.setattr(corpus_mod, "build_corpus", lambda *a, **k: pytest.fail("the normal path must not run for a pick-only set"))
    assert prov.applies(SET)
    api.calculate_set(SET.lower(), n_boot=2, fetch_scryfall=False, quick=True)
    assert called and called[0][0] == SET
    assert {"progress", "cancel", "n_jobs", "n_boot", "fetch_scryfall"} <= set(called[0][1])


def test_routing_game_file_keeps_the_normal_path(registered, home, monkeypatch):
    import sealed_lab.ingest.corpus as corpus_mod

    registry.register(_file_info(home / "game_data_public.TST.Sealed.csv.gz", "game", "Sealed", "bo1"))
    assert not prov.applies(SET)
    seen = []

    class Stop(Exception):
        pass

    def fake_build(sc, **kw):
        seen.append(sc)
        raise Stop()

    monkeypatch.setattr(corpus_mod, "build_corpus", fake_build)
    monkeypatch.setattr(prov, "calculate", lambda *a, **k: pytest.fail("a set with a game file never gets a provisional model"))
    with pytest.raises(Stop):
        api.calculate_set(SET, fetch_scryfall=False)
    assert seen == [SET]


def test_no_pick_file_is_not_provisional(home):
    registry.register(_file_info(home / "draft_data_public.TST.TradDraft.csv.gz", "draft", "TradDraft", "draft"))
    assert not prov.applies(SET) and prov.pick_files(SET) == []


def test_calculate_without_files_says_so(home):
    with pytest.raises(prov.ProvisionalError, match="no PremierDraft pick file"):
        prov.calculate(SET, n_jobs=1, n_boot=1)


# ---------------------------------------------------------------------------------------------- the transfer tool
def _tool():
    spec = importlib.util.spec_from_file_location("make_provisional_transfer", ROOT / "tools" / "make_provisional_transfer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_provisional_transfer"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_combine_transfers_means_by_name():
    def one(coef_int, ref_int, theta0, lam):
        names = ["intercept", "on_play", "mull1", "mull2p", "skill_lin", "event:Sealed"]
        coef = np.array([coef_int, 0.2, -0.5, -1.0, 0.3, 9.0] + [0.0] * 3 + [theta0] + [0.0] * (structure.N_FEATURES - 1))
        ref = np.array([1.0, 0.5, 0.0, 0.0, ref_int, 0.4])
        d = dict(status="ok", names=names, coef=coef, k=6, n_card=3, n_feat=structure.N_FEATURES, ref_row=ref, lam_bo1=lam)
        return dict(bo1=d, bo3=dict(d, lam_bo1=None))
    t = prov.combine_transfers({"AAA": one(0.2, 1.0, 0.1, 0.8), "BBB": one(0.4, -1.0, 0.3, 0.6)})
    f = t["formats"]["bo1"]
    assert t["sets"] == ["AAA", "BBB"] and np.isclose(t["lam_bar"], 0.7) and t["n_feat"] == structure.N_FEATURES
    assert np.isclose(f["cov"]["intercept"], 0.3) and "event:Sealed" not in f["cov"]                  # event dummies are not transferred
    assert np.isclose(f["theta"][0], 0.2) and len(f["theta"]) == structure.N_FEATURES
    ref_a = 0.2 * 1.0 + 0.2 * 0.5 - 0.5 * 0 - 1.0 * 0 + 0.3 * 1.0                                      # per set: coef . ref_row on the transferred names
    ref_b = 0.4 * 1.0 + 0.2 * 0.5 + 0.3 * -1.0
    assert np.isclose(f["ref"], (ref_a + ref_b) / 2) and np.isclose(f["mull1"], -0.5)


def test_transfer_tool_on_a_tiny_synthetic_home_is_read_only(calc, tmp_path):
    """Run the tool on the synthetic set's home (one set): it writes only its output file, and the home is byte-for-byte unchanged."""
    tool = _tool()
    home = Path(calc["home"])
    before = tool.snapshot(home)
    out = tmp_path / "transfer.json"
    t = tool.make_transfer(home, ["TST"], 5, 2, out)
    assert tool.snapshot(home) == before
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["sets"] == ["TST"] and loaded["feature_names"] == structure.FEATURE_NAMES and loaded["n_feat"] == structure.N_FEATURES
    assert loaded["sim_version"] >= 1 and loaded["date"] and set(loaded["formats"]) == {"bo1", "bo3"} and loaded["lam_bar"] > 0
    assert t["formats"]["bo1"]["cov"]["intercept"] == loaded["formats"]["bo1"]["cov"]["intercept"]
    prov.load_transfer(out)                                                                              # and the app accepts it
