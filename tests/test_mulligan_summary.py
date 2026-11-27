"""Tests for the purely descriptive opening-hand keep / mulligan summary (Task 3) and its Report tab section."""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import polars as pl
import pytest

from sealed_lab import api, config, mulligan_summary as ms
from sealed_lab.ingest import replay
from sealed_lab.gui import fake_api


def _write_replay_shard(home, sc: str, event: str, fp: str, games_rows: list[dict], draws_rows: list[dict]) -> None:
    sdir = home / "sets" / sc / "cache" / "replay" / fp
    sdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(games_rows).write_parquet(sdir / "games.parquet")
    pl.DataFrame(draws_rows).write_parquet(sdir / "draws.parquet")
    (sdir / "meta.json").write_text(
        json.dumps({"version": replay.REPLAY_VERSION, "ok": True, "event": event, "fingerprint": fp, "rows": len(games_rows)}),
        encoding="utf-8",
    )


def test_compute_from_tables_cutoff_dedup_wilson_and_exclusion():
    """Exact candidate_hand_1 (kind==3) counting: dedup copies per hand, exclude lands/basics, 200-hand cutoff, Wilson CI."""
    cards = pl.DataFrame({
        "card_id": [0, 1, 2, 3, 4],
        "name": ["HighKeep", "HighMull", "Below200", "DualLand", "Plains"],
        "colors": ["W", "U", "B", "", ""],
        "mana_value": [2.0, 6.0, 3.0, 0.0, 0.0],
        "rarity": ["common", "uncommon", "common", "common", "common"],
        "is_land": [False, False, False, True, True],
        "is_basic": [False, False, False, False, True],
    })
    idmap = {1000: 0, 1001: 1, 1002: 2, 1003: 3, 1004: 4}

    # 300 games:
    # - gids 0..199 (200 games): HighKeep in candidate_hand_1 (kept in 190, mulliganed in 10 -> 95.0% kept, 5.0% mull)
    #   Also put 2 duplicate rows of HighKeep in gid 0 to verify dedup -> still 200 hands, NOT 201!
    # - gids 0..249 (250 games): HighMull in candidate_hand_1 (kept in 175, mulliganed in 75 -> 70.0% kept, 30.0% mull)
    # - gids 0..198 (199 games): Below200 in candidate_hand_1 -> < 200 hands -> MUST NOT appear!
    # - gids 0..299 (300 games): DualLand and Plains in candidate_hand_1 -> lands/basics -> MUST NOT appear!
    # - Also add kind=0 (kept opening_hand) & kind=1 (drawn) & kind=4 (candidate_hand_2) for Below200 on gid 199..250 ->
    #   MUST BE IGNORED because only kind==3 (candidate_hand_1) counts!
    games_rows = []
    draws_rows = []
    for gid in range(300):
        # Set num_mulligans so:
        # gids 0..9: mull=1 (10 games)
        # gids 10..199: mull=0 (190 games)
        # gids 200..264: mull=1 (65 games)
        # gids 265..299: mull=0 (35 games)
        mull = 1 if (gid < 10 or 200 <= gid < 265) else 0
        games_rows.append(dict(gid=gid, num_mulligans=mull))
        draws_rows.append(dict(gid=gid, kind=3, id=1003))   # DualLand
        draws_rows.append(dict(gid=gid, kind=3, id=1004))   # Plains
        if gid < 200:
            draws_rows.append(dict(gid=gid, kind=3, id=1000))  # HighKeep
        if gid == 0:
            draws_rows.append(dict(gid=gid, kind=3, id=1000))  # duplicate copy in same opener
        if gid < 250:
            draws_rows.append(dict(gid=gid, kind=3, id=1001))  # HighMull (10 + 50 = 60 mull, 190 kept -> 76% kept, 24% mull)
        if gid < 199:
            draws_rows.append(dict(gid=gid, kind=3, id=1002))  # Below200 (199 hands < 200)
        else:
            draws_rows.append(dict(gid=gid, kind=0, id=1002))  # kind=0 must not count
            draws_rows.append(dict(gid=gid, kind=1, id=1002))  # kind=1 must not count
            draws_rows.append(dict(gid=gid, kind=4, id=1002))  # kind=4 must not count

    out = ms.compute_from_tables(cards, pl.DataFrame(games_rows), pl.DataFrame(draws_rows), idmap)
    assert out["n_hands"] == 300
    assert out["n_kept"] == 225 and out["n_mull"] == 75
    assert out["min_hands"] == 200
    assert out["n_cards_qualifying"] == 2

    kept_names = [r["name"] for r in out["most_kept"]]
    mull_names = [r["name"] for r in out["most_mulliganed"]]
    assert kept_names == ["HighKeep", "HighMull"]
    assert mull_names == ["HighMull", "HighKeep"]
    assert "Below200" not in kept_names and "DualLand" not in kept_names and "Plains" not in kept_names

    hk = out["most_kept"][0]
    assert hk["n"] == 200 and hk["n_kept"] == 190 and hk["n_mull"] == 10
    assert hk["keep_rate"] == pytest.approx(0.95)
    lo, hi = ms.wilson_interval(190, 200)
    assert hk["keep_lo"] == pytest.approx(lo) and hk["keep_hi"] == pytest.approx(hi)
    assert 0.90 < lo < 0.95 < hi < 0.98

    hm = out["most_mulliganed"][0]
    assert hm["n"] == 250 and hm["n_mull"] == 60 and hm["n_kept"] == 190
    assert hm["mull_rate"] == pytest.approx(0.24)
    mlo, mhi = ms.wilson_interval(60, 250)
    assert hm["mull_lo"] == pytest.approx(mlo) and hm["mull_hi"] == pytest.approx(mhi)


def test_mulligan_summary_per_format_cache_and_missing_replays(tmp_path, monkeypatch):
    """Per-format replay isolation (Bo1 Sealed vs Bo3 TradSealed), cache round-trip, and clean unavailable result for ECL/FRA."""
    home = tmp_path / "home"
    monkeypatch.setenv("SEALED_LAB_HOME", str(home))

    # ECL and FRA have no replay shards -> available=False with clear message, no crash
    for sc in ("ECL", "FRA"):
        res = ms.mulligan_summary(sc, "bo1")
        assert res["available"] is False
        assert "No replay data" in res["reason"] and sc in res["reason"]
        assert res["most_kept"] == [] and res["most_mulliganed"] == []

    # Set SYN has Bo1 (Sealed) replay shard only, NOT Bo3 (TradSealed)
    sc = "SYN"
    cdir = config.corpus_dir(sc)
    cdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "card_id": [0], "name": ["TwoDrop"], "colors": ["W"], "mana_value": [2.0],
        "rarity": ["common"], "is_land": [False], "is_basic": [False],
    }).write_parquet(cdir / "cards.parquet")
    (cdir / "manifest.json").write_text(json.dumps({"built_at": "2026-11-27T00:00:00Z"}), encoding="utf-8")
    rdir = replay.replay_dir(sc)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "idmap.json").write_text(json.dumps({"map": {"500": 0}}), encoding="utf-8")

    g_rows = [dict(gid=i, num_mulligans=(1 if i < 20 else 0)) for i in range(210)]
    d_rows = [dict(gid=i, kind=3, id=500) for i in range(210)]
    _write_replay_shard(home, sc, "Sealed", "fp_bo1", g_rows, d_rows)

    res_bo1 = api.mulligan_summary(sc, "bo1")
    assert res_bo1["available"] is True and res_bo1["n_hands"] == 210
    assert res_bo1["most_kept"][0]["name"] == "TwoDrop"
    assert res_bo1["most_kept"][0]["n"] == 210
    assert ms.cache_path(sc, "bo1").exists()

    # Bo3 has no TradSealed shard -> explicitly unavailable for Bo3
    res_bo3 = api.mulligan_summary(sc, "bo3")
    assert res_bo3["available"] is False and "Best of 3" in res_bo3["reason"]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from sealed_lab.gui.widgets import apply_theme

    app = QApplication.instance() or QApplication([])
    apply_theme(app, "light")
    return app


def test_gui_report_tab_mulligan_section_and_no_replay_sets(qapp, tmp_path, monkeypatch):
    """ReportTab renders the descriptive opening-hand keep/mulligan tables right after Glance, and shows a clear message
    (not an empty table) for sets without replay data like ECL and FRA."""
    from PySide6.QtWidgets import QApplication, QLabel
    from sealed_lab.gui.report_tab import ReportTab
    from sealed_lab.types import SetStatus

    monkeypatch.setenv("SEALED_LAB_SETTINGS", str(tmp_path / "settings.ini"))
    monkeypatch.setenv("SEALED_LAB_FAKE_DELAY", "0.002")
    from sealed_lab.gui import workers
    monkeypatch.setattr(workers, "_closing", False)   # a main window closed by an earlier GUI test stops all new background jobs
    fake_api.reset()

    tab = ReportTab(fake_api)
    errors: list[str] = []
    tab.errorOccurred.connect(lambda short, tb: errors.append(tb or short))
    try:
        statuses = [
            SetStatus(set_code=sc, files=[], has_corpus=True, has_model=True, stale=False,
                      n_games={"bo1": 1000, "bo3": 2000}, formats=["bo1", "bo3"], built_at="2026-11-27")
            for sc in ("TST", "ECL", "FRA")
        ]
        tab.refresh_sets(statuses)
        tab.show()
        tab.select_set("TST")              # explicit: settings left by other GUI tests may remember another set or format
        tab.fmt_bo3.click()
        tab.load()                          # (a hidden-then-shown tab loads on its show event, which another test's window can swallow)

        def wait_loaded(sc: str, fmt: str):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 30.0:
                QApplication.processEvents()
                if tab._loaded_key == (sc, fmt) and not tab.task.running:
                    return
                time.sleep(0.01)
            raise AssertionError(f"timed out loading {sc} {fmt}; loaded={tab._loaded_key}; set={tab.set_code()} fmt={tab.fmt()} running={tab.task.running} msg={tab.message.text()!r} visible={tab.isVisible()} errors={errors[-1:]}")

        # 1. Set with replay data (TST Bo3): section appears after the color pairs and cards, with rates, Wilson intervals, n >= 200
        wait_loaded("TST", "bo3")
        sec_keys = list(tab._secs.keys())
        assert "glance" in sec_keys and "mulligans" in sec_keys
        assert sec_keys.index("mulligans") > sec_keys.index("cards")

        msec = tab._secs["mulligans"]
        texts = [w.text() for w in msec.findChildren(QLabel)]
        joined = " ".join(texts)
        assert "Most often kept" in joined and "Most often mulliganed" in joined
        assert "95% interval" in joined and "at least 200 opening hands" in joined
        # Purely descriptive: no forbidden labels or win-rate claims
        low = joined.lower()
        for bad in ("mulligan-saver", "trap", "win rate"):
            assert bad not in low

        # 2. Sets without replay data (ECL and FRA): say so clearly, do NOT show empty tables
        for no_rep in ("ECL", "FRA"):
            ix = tab.set_combo.findData(no_rep)
            assert ix >= 0
            tab.set_combo.setCurrentIndex(ix)
            tab._apply_set()
            wait_loaded(no_rep, "bo3")
            msec = tab._secs["mulligans"]
            joined = " ".join(w.text() for w in msec.findChildren(QLabel))
            assert "No replay data" in joined and no_rep in joined
            assert "Most often kept" not in joined and "Most often mulliganed" not in joined
    finally:
        tab.wait(3000)
        tab.close()
        QApplication.processEvents()
