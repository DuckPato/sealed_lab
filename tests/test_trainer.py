"""Tests for Phase 4C cold-start LUCE5 pick prior (`sealed_lab.model.pickprior`) and Phase 5 Sealed Pool Trainer (`sealed_lab.trainer`)."""
from __future__ import annotations

import math
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pytest

from sealed_lab import api, config, trainer
from sealed_lab.gui import fake_api
from sealed_lab.model import cardvalue, design, pickprior


# --------------------------------------------------------------------------- Phase 4C: pickprior unit tests

def test_decide_pickprior_gates_strictly_at_40k_draft_games(tmp_path):
    dummy = tmp_path / "draft_data_public.TST.PremierDraft.csv"
    dummy.write_text("draft_id,pick_number,pick\n", encoding="utf-8")

    # Mature set (>= 40,000 draft games) -> NOT applied in auto mode
    d_mat = pickprior.decide_pickprior("auto", 40_000, [dummy], fit_info=dict(available=True, n_picks5=100, n_pick_drafts=10, covered=20, r2=0.4, slope=0.05))
    assert d_mat["applied"] is False and ">= 40,000" in d_mat["reason"]

    # Early set (< 40,000 draft games) without pick files -> NOT applied, clear explanation
    d_nopick = pickprior.decide_pickprior("auto", 12_000, [])
    assert d_nopick["applied"] is False and "no 17Lands pick files" in d_nopick["reason"]

    # Early set (< 40,000 draft games) with pick files -> applied
    d_cold = pickprior.decide_pickprior("auto", 12_000, [dummy], fit_info=dict(available=True, n_picks5=5000, n_pick_drafts=350, covered=80, r2=0.42, slope=0.06))
    assert d_cold["applied"] is True and d_cold["source"] == "cold-start"

    # Forced off -> never applied
    d_off = pickprior.decide_pickprior("off", 12_000, [dummy])
    assert d_off["applied"] is False


def test_luce5_and_mapping_recover_synthetic_pick_order(tmp_path):
    """LUCE5 conditional logit + rarity/slope WLS mapping recovers planted card strengths from synthetic pick CSV."""
    rng = np.random.default_rng(42)
    names = [f"Spell_{i}" for i in range(18)]   # >= MIN_SIGNAL_CARDS, so the pick signal joins the intercepts
    true_theta = np.linspace(-1.2, 1.2, len(names))
    pfile = tmp_path / "draft_data_public.TST.PremierDraft.csv"
    hdr = ["draft_id", "draft_time", "pack_number", "pick_number", "pick"] + [f"pack_card_{n}" for n in names]
    lines = [",".join(hdr)]
    for d in range(80):
        for pk in range(5):
            shown = rng.choice(len(names), size=6, replace=False)
            probs = np.exp(true_theta[shown])
            probs /= probs.sum()
            chosen = int(rng.choice(shown, p=probs))
            counts = [1 if j in shown else 0 for j in range(len(names))]
            lines.append(",".join([f"d{d}", "2026-07-01 12:00:00", "0", str(pk), names[chosen]] + [str(v) for v in counts]))
    pfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loaded = pickprior.read_early_picks([pfile])
    assert loaded is not None
    parr, pnames = loaded
    assert parr["n_drafts"] == 80 and parr["n_picks5"] == 400
    est_theta, takes = pickprior.luce_fit(parr["taken"], parr["offsets"], parr["cards"], parr["counts"], len(pnames))
    idx = [pnames.index(n) for n in names]
    assert np.corrcoef(true_theta, est_theta[idx])[0, 1] > 0.85

    # Mapping onto noisy betas preserves positive slope and high R2
    cards = pl.DataFrame({
        "card_id": list(range(len(names) + 1)),
        "name": names + ["Plains"],
        "rarity": ["common"] * 9 + ["uncommon"] * 9 + ["common"],
        "is_land": [False] * len(names) + [True],
        "is_basic": [False] * len(names) + [True],
    })
    dc = pl.DataFrame({"deck_key": [1], "card_id": [0], "n": [1]})
    space = design.build_deck_space(np.array([1]), np.zeros((1, 4)), dc, cards)
    beta_noisy = 0.08 * true_theta + rng.normal(0, 0.01, size=len(names))
    expo = np.full(len(names), 500.0)
    mapped, minfo = pickprior.map_signal_to_betas(cards, space, expo, beta_noisy, est_theta, takes, pnames, min_taken=5)
    assert minfo["covered"] == len(names) and minfo["slope"] > 0.03 and minfo["r2"] > 0.6
    assert np.corrcoef(mapped, true_theta)[0, 1] > 0.85


# --------------------------------------------------------------------------- Phase 5: Sealed Pool Trainer tests

def test_trainer_backend_on_fake_api():
    fake_api.reset()
    s1 = fake_api.sample_trainer_pool("TST", fmt="bo3")
    assert s1["set_code"] == "TST" and s1["n_pool_cards"] == 84 and s1["mode"] == "standard"
    assert s1["pilot"]["trophy"] is True and s1["pilot"]["skill"] >= 0.62
    s2 = fake_api.sample_trainer_pool("TST", fmt="bo3", exclude_ids=[s1["draft_id"]])
    assert s2["draft_id"] != s1["draft_id"]

    # Tricky mode returns an honest >= 7/8 held-out draw disagreement pool
    st = fake_api.sample_trainer_pool("TST", fmt="bo3", mode="tricky")
    assert st["mode"] == "tricky" and st["tricky_info"] is not None
    assert st["tricky_info"]["model_beats_pilot"] >= 7 and st["tricky_info"]["n_draws"] == 8
    assert st["tricky_info"]["model_colors"] != st["tricky_info"]["pilot_colors"]

    # Grade a user build across all 16 bootstrap draws
    user_deck = dict(s1["pilot"]["deck"])
    graded = fake_api.grade_trainer_build("TST", "bo3", s1["pool"], user_deck, s1["pilot"], max_families=3)
    sm = graded["summary"]
    assert 0.0 < sm["user_wr"] < 1.0 and 0.0 < sm["model_wr"] < 1.0
    assert sm["n_draws"] == 16 and 0 <= sm["model_beats_user"] <= 16
    assert sm["user_beats_pilot"] == 0 and sm["pilot_beats_user"] == 0   # identical to pilot across all draws
    assert sm["vs_pilot"]["shared_spells"] == sm["vs_pilot"]["ref_spells"]
    assert graded["build_result"].decks


def test_trainer_backend_on_real_corpus(calc):
    """`api.sample_trainer_pool` and `api.grade_trainer_build` work end-to-end on the calculated synthetic TST corpus."""
    sample = api.sample_trainer_pool("TST", fmt="bo1", seed=7)
    assert sample["set_code"] == "TST" and sample["pool"] and sample["pilot"]["deck"]
    graded = api.grade_trainer_build("TST", "bo1", sample["pool"], sample["pilot"]["deck"], sample["pilot"], max_families=2)
    sm = graded["summary"]
    assert sm["verdict"] and sm["n_draws"] >= 4
    assert graded["user_eval"].game_wr == pytest.approx(graded["pilot_eval"].game_wr)
    assert sm["user_beats_pilot"] == 0 and sm["pilot_beats_user"] == 0


def test_tricky_mode_runs_the_real_builder(calc, monkeypatch):
    """Tricky mode builds each candidate pool with the real deck builder (it once called a function that does not exist, which
    only the fake API hid).  The synthetic corpus may hold no pool where the model picks other colors: then the mode says so."""
    from sealed_lab.deck import builder

    real, calls = builder.build_ids, []

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(builder, "build_ids", counting)
    monkeypatch.setattr(trainer, "MIN_POOL_CARDS", 1)       # the synthetic pools are small: let every one reach the builder
    for p in config.cache_dir("TST").glob("trainer_tricky_*.json"):
        p.unlink()
    try:
        s = api.sample_trainer_pool("TST", fmt="bo1", seed=3, mode="tricky")
    except ValueError as e:
        assert "No tricky pool found" in str(e) and calls
        return
    assert calls
    ti = s["tricky_info"]
    assert s["mode"] == "tricky" and ti["n_draws"] >= 1
    assert ti["model_beats_pilot"] >= math.ceil(trainer.TRICKY_DRAW_SHARE * ti["n_draws"])
    canon = lambda c: "".join(x for x in "WUBRG" if x in (c or "").upper())  # noqa: E731
    assert canon(ti["model_colors"]) != canon(ti["pilot_colors"])


def test_draw_subset_none_identical_and_worker_slicing(calc):
    """1) `draw_subset=None` produces identical decks for `families=True` and `families=False`.
    2) With a subset, `_worker_state` and parallel `build_families` slice `_draws`/`_ref` to that subset with draw 0 = mean."""
    from sealed_lab.deck import builder, families

    model = builder.load_model("TST")
    sample = api.sample_trainer_pool("TST", fmt="bo1", seed=7)
    pool = sample["pool"]

    def sig(res):
        return [({c.card_id: c.count for c in d.cards}, d.colors, round(d.game_wr, 8), round(d.score, 6)) for d in res.decks]

    # 1. draw_subset=None is identical to default for both families=True and families=False
    r_fam_def = builder.build(model, pool, "bo1", n_decks=2, families=True, n_jobs=1)
    r_fam_none = builder.build(model, pool, "bo1", n_decks=2, families=True, n_jobs=1, draw_subset=None)
    assert sig(r_fam_def) == sig(r_fam_none)

    r_cls_def = builder.build(model, pool, "bo1", n_decks=2, families=False)
    r_cls_none = builder.build(model, pool, "bo1", n_decks=2, families=False, draw_subset=None)
    assert sig(r_cls_def) == sig(r_cls_none)

    # 2. Worker slicing restricts _draws / _ref to `subset` and sets draw 0 to the mean of `subset`
    even, _odd = trainer.tricky_draw_halves(model.n_draws("bo1"))
    assert len(even) >= 1
    stamp = builder.model_stamp("TST")
    job = dict(id="test-subset-worker", set_code="TST", stamp=stamp, fmt="bo1", pool=pool,
               locked={}, excluded=[], start_decks=[], draw_subset=even)
    st_w = families._worker_state(job)
    wm = st_w.model
    assert wm.n_draws("bo1") == 1 + len(even)
    for k, orig_arr in model._draws["bo1"].items():
        got = wm._draws["bo1"][k]
        expected_sub = np.asarray(orig_arr)[list(even)]
        assert got.shape == (1 + len(even),) + orig_arr.shape[1:]
        assert np.allclose(got[0], expected_sub.mean(axis=0))
        assert np.allclose(got[1:], expected_sub)
    orig_ref = np.asarray(model._ref["bo1"])
    got_ref = np.asarray(wm._ref["bo1"])
    assert got_ref.shape == (1 + len(even), orig_ref.shape[1])
    assert np.allclose(got_ref[0], np.sort(orig_ref[list(even)].mean(axis=0)))
    assert np.allclose(got_ref[1:], orig_ref[list(even)])

    # Parallel build_families with draw_subset runs cleanly on workers and reports only len(even) draws
    r_sub = families.build_families(model, pool, "bo1", max_families=1, n_jobs=2, draw_subset=even)
    assert r_sub.decks and r_sub.family_meta["n_draws"] == len(even)


def _make_tiny_16draw_env(tmp_path, monkeypatch, odd_wu_wins: int) -> tuple[dict[int, int], dict[int, int]]:
    """Build a tiny synthetic 16-bootstrap-draw (17 total) SetModel + 1-run Sealed corpus where:
    - Full-data draw 0 strongly favors BG (tests that full draw 0 is NEVER used to choose #1).
    - Even draws (2,4,6,8,10,12,14,16) strongly favor WU (so #1 chosen on half 1 is WU).
    - Odd draws (1,3,5,7,9,11,13,15) favor WU on `odd_wu_wins` of the 8 draws and BG on the remaining `8 - odd_wu_wins`.
    - Pilot played BG."""
    import json
    from sealed_lab import deck_search
    from sealed_lab.deck import builder, families
    from sealed_lab.model import artifact, structure

    home = tmp_path / f"home_{odd_wu_wins}"
    monkeypatch.setenv("SEALED_LAB_HOME", str(home))
    deck_search._CACHE.clear()
    builder._MODELS.clear()
    families.shutdown_pool(kill=True)

    sc = "T16"
    cdir = config.corpus_dir(sc)
    mdir = config.model_dir(sc)
    cdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    # 14 W spells, 14 U spells, 14 B spells, 14 G spells, + 5 basics = 61 cards
    rows = []
    cid = 0
    for col, nm_prefix in (("W", "White"), ("U", "Blue"), ("B", "Black"), ("G", "Green")):
        for j in range(14):
            mv = [2, 2, 2, 3, 3, 3, 4, 4, 5, 2, 3, 3, 4, 5][j]
            is_cre = j < 10
            pips = {c: (1 if c == col else 0) for c in "WUBRG"}
            rows.append(dict(
                card_id=cid, name=f"{nm_prefix}_{j}", front_name=f"{nm_prefix}_{j}",
                mana_value=float(mv), is_creature=is_cre, is_land=False, is_basic=False,
                colors=col, rarity="common", types="Creature" if is_cre else "Instant",
                produces="", generic=mv - 1, hybrid=False, **{"pip_" + c: pips[c] for c in "WUBRG"},
                exposure_bo1=5000.0, low_data_bo1=False, se_bo1=0.01,
            ))
            cid += 1
    basic_ids = {}
    for col, bname in (("W", "Plains"), ("U", "Island"), ("B", "Swamp"), ("R", "Mountain"), ("G", "Forest")):
        basic_ids[col] = cid
        rows.append(dict(
            card_id=cid, name=bname, front_name=bname,
            mana_value=0.0, is_creature=False, is_land=True, is_basic=True,
            colors="", rarity="common", types="Basic Land", produces=col, generic=0, hybrid=False,
            pip_W=0, pip_U=0, pip_B=0, pip_R=0, pip_G=0,
            exposure_bo1=5000.0, low_data_bo1=False, se_bo1=0.01,
        ))
        cid += 1
    cards = pl.DataFrame(rows)
    n_cards = cards.height

    # 17 draws: draw 0 + 16 bootstrap draws (indices 1..16)
    wu_ids = list(range(0, 28))
    bg_ids = list(range(28, 56))
    beta = np.zeros((17, n_cards), dtype=np.float32)
    # Full draw 0 favors BG (must NOT leak into half-1 selection!)
    beta[0, wu_ids] = -0.35
    beta[0, bg_ids] = +0.35
    even_idx = [2, 4, 6, 8, 10, 12, 14, 16]
    odd_idx = [1, 3, 5, 7, 9, 11, 13, 15]
    # Even draws (half 1) strongly favor WU
    for d in even_idx:
        beta[d, wu_ids] = +0.35
        beta[d, bg_ids] = -0.35
    # Odd draws (half 2): first `odd_wu_wins` favor WU, the rest favor BG
    for k, d in enumerate(odd_idx):
        if k < odd_wu_wins:
            beta[d, wu_ids] = +0.35
            beta[d, bg_ids] = -0.35
        else:
            beta[d, wu_ids] = -0.35
            beta[d, bg_ids] = +0.35

    n_feat = len(structure.FEATURE_NAMES)
    theta = np.zeros((17, n_feat), dtype=np.float32)
    ref = np.zeros(17, dtype=np.float64)
    ref_q = np.tile(np.linspace(0.3, 0.7, 50, dtype=np.float32), (17, 1))
    meta = dict(
        set_code=sc, formats=["bo1"], built_at="2026-11-26T00:00:00Z",
        feature_names=list(structure.FEATURE_NAMES),
        phi_mean=[0.0] * n_feat, phi_sd=[1.0] * n_feat,
        manasim={"version": 1}, castability={"applied": False},
        land_support={"bo1": {"16": 0.3, "17": 0.5, "18": 0.2}},
        creature_support={"bo1": {"lo": 10, "hi": 20}},
    )
    artifact.save_artifact(sc, meta, cards, {"bo1": {"beta": beta, "theta": theta, "ref": ref}},
                           {"bo1": ref_q}, "<html></html>", {})

    # Corpus: 1 Sealed Bo1 pool with all 56 nonland spells; pilot plays BG (23 BG spells + 9 Swamp + 8 Forest)
    cards.write_parquet(cdir / "cards.parquet")
    pool_counts = {c: 1 for c in range(56)}
    pilot_deck = {c: 1 for c in bg_ids[:23]}
    pilot_deck[basic_ids["B"]] = 9
    pilot_deck[basic_ids["G"]] = 8
    pl.DataFrame([dict(draft_id="d_tricky", card_id=c, n=n) for c, n in pool_counts.items()]).write_parquet(cdir / "pools.parquet")
    pl.DataFrame([dict(deck_key=1, card_id=c, n=n) for c, n in pilot_deck.items()]).write_parquet(cdir / "deck_cards.parquet")
    pl.DataFrame([dict(deck_key=1, n_lands=17, n_cards=40, main_colors="BG", splash_colors="")]).write_parquet(cdir / "decks.parquet")
    pl.DataFrame([dict(event="Sealed", draft_id="d_tricky", build_index=0, deck_key=1, games=5, wins=4)]).write_parquet(cdir / "builds.parquet")
    pl.DataFrame([
        dict(draft_id="d_tricky", event="Sealed", fmt="bo1", deck_key=1, build_index=0, match_number=1,
             game_number=g, won=(g <= 4), skill=0.65, rank="Mythic", main_colors="BG", splash_colors="",
             game_time=None, draft_time=None)
        for g in range(1, 6)
    ]).write_parquet(cdir / "games.parquet")
    (cdir / "manifest.json").write_text(json.dumps({"built_at": "2026-11-26T00:00:00Z", "tables": {"deck_cards": len(pilot_deck)}}), encoding="utf-8")
    return pool_counts, pilot_deck


def test_tricky_mode_honest_split_half_7_of_8_threshold(tmp_path, monkeypatch):
    """On a 16-draw model, Tricky mode chooses #1 strictly on even draws (ignoring draw 0 and odd draws),
    counts wins against the pilot strictly on the 8 odd draws, keeps >= 7/8 and rejects 6/8 and 0/8."""
    from sealed_lab.deck import builder

    monkeypatch.setattr(trainer, "MIN_POOL_CARDS", 1)

    # Case 1: 7 of 8 odd wins -> KEPT, model chose WU (even draws) even though draw 0 favored BG
    _make_tiny_16draw_env(tmp_path, monkeypatch, odd_wu_wins=7)
    seen_subsets = []
    real_build_ids = builder.build_ids

    def spy_build_ids(*a, **k):
        seen_subsets.append(k.get("draw_subset"))
        return real_build_ids(*a, **k)

    monkeypatch.setattr(builder, "build_ids", spy_build_ids)
    s7 = trainer.sample_trainer_pool("T16", fmt="bo1", seed=1, mode="tricky")
    assert seen_subsets == [(2, 4, 6, 8, 10, 12, 14, 16)]
    ti = s7["tricky_info"]
    assert ti["model_colors"] == "WU" and ti["pilot_colors"] == "BG"
    assert ti["model_beats_pilot"] == 7 and ti["n_draws"] == 8 and ti["held_out"] is True

    # Case 2: 6 of 8 odd wins (still 8/8 on even draws!) -> REJECTED by the 7-of-8 threshold
    _make_tiny_16draw_env(tmp_path, monkeypatch, odd_wu_wins=6)
    with pytest.raises(ValueError, match="No tricky pool found"):
        trainer.sample_trainer_pool("T16", fmt="bo1", seed=1, mode="tricky")

    # Case 3: 0 of 8 odd wins (odd draws completely favor BG, even draws favor WU) -> still chooses WU on even draws and rejects
    _make_tiny_16draw_env(tmp_path, monkeypatch, odd_wu_wins=0)
    chosen_colors = []

    def spy_build_ids_colors(*a, **k):
        res = real_build_ids(*a, **k)
        if res.decks:
            chosen_colors.append(res.decks[0].colors)
        return res

    monkeypatch.setattr(builder, "build_ids", spy_build_ids_colors)
    with pytest.raises(ValueError, match="No tricky pool found"):
        trainer.sample_trainer_pool("T16", fmt="bo1", seed=1, mode="tricky")
    assert chosen_colors == ["WU"]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from sealed_lab.gui.widgets import apply_theme

    app = QApplication.instance() or QApplication([])
    apply_theme(app, "light")
    return app


def test_gui_pool_trainer_blind_then_reveal_and_grade(qapp, tmp_path, monkeypatch):
    """PoolTab's Pool trainer deals a blind pool (values & metrics hidden), updates counts live, and reveals 3-way comparison."""
    from PySide6.QtWidgets import QApplication
    from sealed_lab.gui.app import MainWindow
    from sealed_lab.gui.widgets import new_settings

    monkeypatch.setenv("SEALED_LAB_SETTINGS", str(tmp_path / "settings.ini"))
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.002")

    old_dd = fake_api.fake_data_dir
    d = tmp_path / "17l"
    d.mkdir()
    fake_api.fake_data_dir = str(d)
    fake_api.reset()
    new_settings().clear()
    win = MainWindow(fake_api, auto_refresh=True)
    win.show()
    try:
        # Wait for initial set list
        import time

        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            QApplication.processEvents()
            if win.pool_tab.statuses and not win.sets_tab.busy:
                break
            time.sleep(0.01)

        pt = win.pool_tab
        win.tabs.setCurrentWidget(pt)
        assert pt.trainer_btn.isEnabled()
        pt.trainer_btn.click()

        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            QApplication.processEvents()
            if pt.trainer_state is not None and not pt.task.running:
                break
            time.sleep(0.01)

        # Verify blind mode: values disabled, model metrics hidden, Reveal button disabled while deck is empty
        tb = pt.trainer_builder
        assert tb is not None and tb.blind is True
        assert not tb.values_cb.isEnabled()
        assert tb._opts()["show_values"] is False
        assert tb._values("side") == {}
        assert not tb.m_game.isVisibleTo(tb)
        assert pt.trainer_banner.isVisibleTo(pt)
        assert not pt.trainer_reveal_btn.isEnabled()

        # Add 23 spells from the pool + 17 basics -> 40/40 live counter updates and Reveal enables
        rem = 23
        for cid, n in list(tb.pool.items()):
            if rem <= 0:
                break
            if not tb.info.get(cid, {}).get("is_land"):
                for _ in range(min(n, rem)):
                    tb.move(cid, "side")
                    rem -= 1
        for _ in range(9):
            tb.add_basic("W", +1)
        for _ in range(8):
            tb.add_basic("U", +1)
        QApplication.processEvents()
        assert sum(tb.deck.values()) == 40
        assert "40 / 40 cards" in pt.trainer_title.text()
        assert pt.trainer_reveal_btn.isEnabled()

        # Click Reveal & grade -> unlocks Your deck, 17Lands pilot tab, and Model #1+ family tabs
        pt.trainer_reveal_btn.click()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            QApplication.processEvents()
            if pt.trainer_state.get("revealed") and not pt.task.running:
                break
            time.sleep(0.01)

        assert pt.tabs.count() >= 3
        assert pt.tabs.tabText(0).startswith("Your deck")
        assert "17Lands pilot" in pt.tabs.tabText(1)
        assert pt.tabs.tabText(2).startswith("#1")
        assert "draws" in pt.trainer_title.text() and "Session: 1 pool" in pt.trainer_title.text()
        assert pt.trainer_detail.isVisibleTo(pt) and "Overlap:" in pt.trainer_detail.text()
        assert pt.user_builder is not None and pt.user_builder.blind is False

        # Seen draft_id was persisted in settings
        seen_saved = pt._load_trainer_seen("TST")
        assert pt.trainer_state["sample"]["draft_id"] in seen_saved

        # 1-click "Find similar pools" switches to Deck search in Similar pools mode (even on first visit before meta loads)
        assert pt.trainer_similar_btn.isVisibleTo(pt) and pt.trainer_similar_btn.isEnabled()
        pt.trainer_similar_btn.click()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            QApplication.processEvents()
            if (win.tabs.currentWidget() is win.deck_search_tab and win.deck_search_tab.mode == "similar"
                    and win.deck_search_tab.results and not win.deck_search_tab.task.running):
                break
            time.sleep(0.01)
        assert win.tabs.currentWidget() is win.deck_search_tab
        assert win.deck_search_tab.mode == "similar" and win.deck_search_tab.results

        # Switch back, deal a Tricky (>=14/16 draws) pool, and verify exit cleans up
        win.tabs.setCurrentWidget(pt)
        pt.start_trainer("TST", mode="tricky")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            QApplication.processEvents()
            if pt.trainer_state is not None and pt.trainer_state["sample"].get("mode") == "tricky" and not pt.task.running:
                break
            time.sleep(0.01)
        assert pt.trainer_state["sample"]["mode"] == "tricky"
        assert "Tricky pool" in pt.trainer_title.text()

        # Exit trainer cleans up cleanly
        pt.trainer_exit_btn.click()
        QApplication.processEvents()
        assert pt.trainer_state is None and not pt.trainer_banner.isVisibleTo(pt)
    finally:
        fake_api.fake_data_dir = old_dd
        pt.wait(3000)
        win.sets_tab.wait(3000)
        win.close()
        QApplication.processEvents()


def _write_pick_file(path, names, n_drafts=500, seed=3):
    """A 17Lands-shaped draft_data_public file: picks 1-14 of pack 1 of `n_drafts` drafts, the pick drawn from a fixed card strength."""
    import csv
    import gzip

    rng = np.random.default_rng(seed)
    strength = rng.normal(size=len(names))
    cols = ["expansion", "event_type", "draft_id", "draft_time", "pack_number", "pick_number", "pick"] + [f"pack_card_{n}" for n in names]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for d in range(n_drafts):
            pack = list(rng.choice(len(names), size=14, replace=False))
            for pick in range(14):
                p = np.exp(strength[pack])
                j = int(rng.choice(len(pack), p=p / p.sum()))
                counts = np.zeros(len(names), dtype=int)
                counts[pack] = 1
                w.writerow(["TST", "PremierDraft", f"pd{d:05d}", "2000-01-01 00:00:00", 0, pick, names[pack[j]]] + counts.tolist())
                pack.pop(j)


def test_pick_prior_runs_with_attribute_and_pair_columns(tmp_path, monkeypatch):
    """Regression: the pick prior's calibration fit has to carry the attribute columns and the trailing blocks (card pairs) of the
    deployed design; it crashed with a shape error when the attribute prior or synergies were on."""
    from sealed_lab.ingest.corpus import build_corpus
    from sealed_lab.model import pipeline
    from sealed_lab.synth.generate import generate

    home, synth = tmp_path / "home", tmp_path / "synth"
    monkeypatch.setenv("SEALED_LAB_HOME", str(home))
    generate(synth, set_code="TST", seed=2, scale=0.15)
    cards = pl.read_csv(synth / "cards.csv")
    names = [n for n, t in zip(cards["name"].to_list(), cards["types"].to_list()) if "Basic" not in str(t)]
    _write_pick_file(synth / "draft_data_public.TST.PremierDraft.csv.gz", names)
    api.add_files([str(synth)])
    build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    meta = pipeline.calculate("TST", n_jobs=2, n_boot=2, recency="off", attributes="on", synergy="on", pickprior="on")
    pp = meta["pickprior"]
    assert pp["applied"], pp["reason"]
    assert pp["covered"] >= pickprior.MIN_SIGNAL_CARDS and pp["slope"] != 0.0
    assert meta["attributes"]["applied"]
