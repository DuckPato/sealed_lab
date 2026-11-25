"""Tests for Phase 4C cold-start LUCE5 pick prior (`sealed_lab.model.pickprior`) and Phase 5 Sealed Pool Trainer (`sealed_lab.trainer`)."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pytest

from sealed_lab import api, trainer
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
    assert s1["set_code"] == "TST" and s1["n_pool_cards"] == 84
    assert s1["pilot"]["trophy"] is True and s1["pilot"]["skill"] >= 0.62
    s2 = fake_api.sample_trainer_pool("TST", fmt="bo3", exclude_ids=[s1["draft_id"]])
    assert s2["draft_id"] != s1["draft_id"]

    # Grade a user build made by swapping 2 spells from the pilot's deck
    user_deck = dict(s1["pilot"]["deck"])
    graded = fake_api.grade_trainer_build("TST", "bo3", s1["pool"], user_deck, s1["pilot"], max_families=3)
    sm = graded["summary"]
    assert 0.0 < sm["user_wr"] < 1.0 and 0.0 < sm["model_wr"] < 1.0
    assert sm["vs_pilot"]["shared_spells"] == sm["vs_pilot"]["ref_spells"]
    assert graded["build_result"].decks


def test_trainer_backend_on_real_corpus(calc):
    """`api.sample_trainer_pool` and `api.grade_trainer_build` work end-to-end on the calculated synthetic TST corpus."""
    sample = api.sample_trainer_pool("TST", fmt="bo1", seed=7)
    assert sample["set_code"] == "TST" and sample["pool"] and sample["pilot"]["deck"]
    graded = api.grade_trainer_build("TST", "bo1", sample["pool"], sample["pilot"]["deck"], sample["pilot"], max_families=2)
    assert graded["summary"]["verdict"]
    assert graded["user_eval"].game_wr == pytest.approx(graded["pilot_eval"].game_wr)


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
        assert pt.trainer_detail.isVisibleTo(pt) and "Overlap:" in pt.trainer_detail.text()
        assert pt.user_builder is not None and pt.user_builder.blind is False

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
