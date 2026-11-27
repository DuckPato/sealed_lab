"""Unit, caching, bundling, and GUI tests for `sealed_lab.draft_insights` (Draft value vs pick position)."""
from __future__ import annotations

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
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sealed_lab import draft_insights as D  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _synthetic_fixture():
    """Build a small deterministic fixture with commons, uncommons, rares, and mythics + early/late picks.

    Cards include:
      - `Common Sleeper`: drafted late (ATA ~10), high value (+2.2 pp), stable in both halves -> robust sleeper
      - `Common Trap`: drafted early (ATA ~2), weak value (-1.6 pp), stable in both halves -> robust overdrafted
      - `Unstable Sleeper`: drafted late, overall high (+2.0 pp) but halves disagree (+3.6 vs +0.3) -> NOT robust
      - `Tiny Gap Card`: both halves only +0.5 pp -> NOT robust
      - `Low Expo Sleeper`: exposure < 2000 -> NOT eligible / NOT robust
      - `Drifter Earlier`: early ATA ~9.0, late ATA ~4.0 -> appears in biggest_earlier
      - `Drifter Later`: early ATA ~2.5, late ATA ~7.5 -> appears in biggest_later
      - Rares + Mythics sharing the `rare+mythic` curve, with `mythic` also reported separately.
    """
    cards_rows = []
    takes_rows = []
    sees_rows = []
    cid = 0

    def add_card(
        name: str,
        rarity: str,
        value_pp: float,
        ata_early: float,
        ata_late: float,
        *,
        se_pp: float = 0.25,
        exposure: float = 6000.0,
        takes_early: int = 400,
        takes_late: int = 400,
        v0_pp: float | None = None,
        v1_pp: float | None = None,
        colors: str = "U",
    ):
        nonlocal cid
        cards_rows.append(
            dict(
                card_id=cid,
                name=name,
                front_name=name,
                rarity=rarity,
                is_basic=False,
                colors=colors,
                mana_value=3.0,
                value_pp=float(value_pp),
                se_pp=float(se_pp),
                exposure=float(exposure),
                v0_pp=float(value_pp if v0_pp is None else v0_pp),
                v1_pp=float(value_pp if v1_pp is None else v1_pp),
            )
        )
        cid += 1
        # Split into early (day 2026-06-01) and late (day 2026-06-25) takes using integer pick_numbers matching mean ATA
        for day, n_t, ata in (("2026-06-01", takes_early, ata_early), ("2026-06-25", takes_late, ata_late)):
            if n_t <= 0:
                continue
            p_lo = max(0, int(np.floor(ata - 1.0)))
            frac = max(0.0, min(1.0, (ata - 1.0) - p_lo))
            n_hi = int(round(n_t * frac))
            n_lo = n_t - n_hi
            if n_lo > 0:
                takes_rows.append(dict(day=day, pick=name, pick_number=p_lo, n=n_lo))
            if n_hi > 0:
                takes_rows.append(dict(day=day, pick=name, pick_number=p_lo + 1, n=n_hi))
            sees_rows.append(dict(day=day, name=name, seen=int(n_t * max(1.5, ata * 0.6))))

    # 12 baseline commons along a clean decreasing value(ATA) curve: val = 2.0 - 0.35 * ata
    for i in range(12):
        ata = 2.0 + i * 0.85
        val = 2.0 - 0.35 * ata
        add_card(f"Base Common {i}", "common", val, ata, ata)

    # Special test commons
    add_card("Common Sleeper", "common", 2.1, 10.2, 10.2, v0_pp=2.0, v1_pp=2.2)
    add_card("Common Trap", "common", -1.9, 2.1, 2.1, v0_pp=-1.8, v1_pp=-2.0)
    add_card("Unstable Sleeper", "common", 1.9, 10.0, 10.0, v0_pp=3.6, v1_pp=-1.3)  # halves disagree around expected ~ -1.5
    add_card("Tiny Gap Card", "common", -0.8, 10.0, 10.0, v0_pp=-0.9, v1_pp=-0.7)  # expected ~ -1.5 -> gap ~ +0.7 (< 1.0)
    add_card("Low Expo Sleeper", "common", 2.5, 10.5, 10.5, exposure=900.0)
    add_card("Drifter Earlier", "common", 1.3, 9.2, 4.1, v0_pp=1.2, v1_pp=1.4)
    add_card("Drifter Later", "common", -1.2, 2.4, 7.6, v0_pp=-1.1, v1_pp=-1.3)

    # 10 uncommons
    for i in range(10):
        ata = 2.0 + i * 0.8
        val = 2.5 - 0.35 * ata
        add_card(f"Base Uncommon {i}", "uncommon", val, ata, ata)

    # 8 rares + 5 mythics sharing rare+mythic curve
    for i in range(8):
        ata = 1.5 + i * 0.8
        val = 3.5 - 0.4 * ata
        add_card(f"Base Rare {i}", "rare", val, ata, ata)
    for i in range(5):
        ata = 1.4 + i * 0.9
        val = 3.8 - 0.4 * ata
        add_card(f"Base Mythic {i}", "mythic", val, ata, ata)

    cards_df = pl.DataFrame(cards_rows)
    takes_df = pl.DataFrame(takes_rows, schema={"day": pl.String, "pick": pl.String, "pick_number": pl.Int32, "n": pl.Int64})
    sees_df = pl.DataFrame(sees_rows, schema={"day": pl.String, "name": pl.String, "seen": pl.Int64})
    return cards_df, takes_df, sees_df


# --------------------------------------------------------------------------- core math & fixture tests
def test_pav_decreasing_is_monotone_non_increasing():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([2.0, 3.0, 1.5, 2.2, -1.0, 0.0])
    w = np.ones_like(x)
    fitted = D.pav_decreasing(x, y, w)
    assert np.all(np.diff(fitted) <= 1e-12)
    # Weighted mean is preserved
    assert np.isclose(fitted.mean(), y.mean(), atol=1e-9)


def test_known_sleepers_overdrafted_split_half_and_mythic():
    cards_df, takes_df, sees_df = _synthetic_fixture()
    tbl, d0, d1, early_end = D.build_table(cards_df, takes_df, sees_df)
    res = D.analyze_table(
        tbl,
        set_code="TST",
        first_day=d0,
        last_day=d1,
        early_end=early_end,
        n_boot=60,
    )
    assert res["available"] is True
    assert res["first_day"] == "2026-06-01"
    assert res["last_day"] == "2026-06-25"

    sleeper_names = [r["name"] for r in res["sleepers_robust"]]
    over_names = [r["name"] for r in res["overdrafted_robust"]]
    assert "Common Sleeper" in sleeper_names
    assert "Common Trap" in over_names
    # Split-half & eligibility guards
    assert "Unstable Sleeper" not in sleeper_names
    assert "Tiny Gap Card" not in sleeper_names
    assert "Low Expo Sleeper" not in sleeper_names

    # Check CI fields on a robust sleeper
    cs = next(r for r in res["sleepers_robust"] if r["name"] == "Common Sleeper")
    assert cs["resid"] > 1.0
    assert cs["r0"] >= 1.0 and cs["r1"] >= 1.0
    assert cs["gap_ci_lo"] < cs["resid"] < cs["gap_ci_hi"]
    assert cs["ci_lo"] < cs["value_pp"] < cs["ci_hi"]

    # Rarity reporting: rare+mythic merged AND mythic separately reported
    br = res["by_rarity"]
    assert set(br.keys()) == {"common", "uncommon", "rare+mythic", "mythic"}
    assert br["rare+mythic"]["n_eligible"] == 13
    assert br["mythic"]["n_eligible"] == 5
    assert br["mythic"]["corr_ata"] is not None
    assert br["common"]["corr_ata"]["spearman"] < 0
    assert br["uncommon"]["corr_ata"]["spearman"] < -0.9

    # Early-vs-late drift
    el = res["early_late"]
    assert el["available"] is True
    earlier_names = [r["name"] for r in el["biggest_earlier"]]
    later_names = [r["name"] for r in el["biggest_later"]]
    assert "Drifter Earlier" in earlier_names
    assert "Drifter Later" in later_names
    de = next(r for r in el["biggest_earlier"] if r["name"] == "Drifter Earlier")
    assert de["d_ata"] < -4.0


def test_short_format_handles_missing_late_period_cleanly():
    cards_df, takes_df, sees_df = _synthetic_fixture()
    # Keep only early day picks (format < 14 days old)
    takes_early = takes_df.filter(pl.col("day") == "2026-06-01")
    sees_early = sees_df.filter(pl.col("day") == "2026-06-01")
    tbl, d0, d1, early_end = D.build_table(cards_df, takes_early, sees_early, min_takes=200)
    res = D.analyze_table(tbl, set_code="NEW", first_day=d0, last_day=d1, early_end=early_end, min_takes=200, n_boot=40)
    assert res["available"] is True
    assert res["early_late"]["available"] is False
    assert "14 days" in res["early_late"]["reason"]
    assert res["early_late"]["biggest_earlier"] == []
    assert res["early_late"]["biggest_later"] == []


# --------------------------------------------------------------------------- pick streaming, caching & bundling
def _write_pick_csv_gz(path: Path, card_names: list[str], extra_sleeper_picks: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["draft_id", "draft_time", "pack_number", "pick_number", "pick"] + [f"pack_card_{n}" for n in card_names]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for day_i, day in enumerate(("2026-06-01 12:00:00", "2026-06-22 12:00:00")):
            for idx, name in enumerate(card_names):
                n_rows = 300 + (extra_sleeper_picks if idx == len(card_names) - 1 else 0)
                pnum = min(13, idx)
                if name == "Card 0" and day_i == 1:
                    pnum = 6  # drifts later
                pack_vec = [1] * len(card_names)
                for k in range(n_rows):
                    w.writerow([f"d{day_i}_{idx}_{k}", day, 0, pnum, name] + pack_vec)


def _setup_fake_home_set(
    home: Path,
    data_dir: Path,
    code: str,
    *,
    provisional: bool = False,
    with_pick_file: bool = True,
) -> Path | None:
    from sealed_lab.ingest import registry

    n_cards = 12
    names = [f"Card {i}" for i in range(n_cards)]
    names[5] = 'Dr. "Quotes" Card'  # regression test: embedded double quotes in pack_card_* CSV headers
    # Card 11 is a clear sleeper (picked at pick_number=11 -> ATA 12, high beta)
    betas = np.linspace(0.08, -0.06, n_cards)
    betas[-1] = 0.095
    betas[0] = -0.075  # Card 0 is overdrafted (ATA 1, low beta)
    ses = np.full(n_cards, 0.01)
    expos = np.full(n_cards, 5000.0)

    mdir = home / "sets" / code / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    cards = pl.DataFrame(
        dict(
            card_id=list(range(n_cards)),
            name=names,
            front_name=names,
            rarity=["common"] * n_cards,
            is_basic=[False] * n_cards,
            is_land=[False] * n_cards,
            colors=["U"] * n_cards,
            mana_value=[3.0] * n_cards,
            beta_bo1=betas.tolist(),
            se_bo1=ses.tolist(),
            exposure_bo1=expos.tolist(),
        )
    )
    cards.write_parquet(mdir / "cards.parquet")
    # 17 draws: index 0 = point estimate, 1..16 = slightly noisy replicates agreeing in sign
    rng = np.random.default_rng(42)
    draws = np.vstack([betas] + [betas + rng.normal(0, 0.004, size=n_cards) for _ in range(16)])
    np.savez(mdir / "draws.npz", bo1__beta=draws)
    meta = dict(built_at="2026-07-01 10:00", formats=["bo1"])
    if provisional:
        meta["provisional"] = dict(source="PremierDraft", n_drafts=5000, built_at="2026-07-01 10:00")
    (mdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (home / "sets" / code / "cache").mkdir(parents=True, exist_ok=True)

    pick_path: Path | None = None
    if with_pick_file:
        pick_path = data_dir / code / f"draft_data_public.{code}.PremierDraft.csv.gz"
        _write_pick_csv_gz(pick_path, names)
        registry.register(
            registry.FileInfo(
                path=str(pick_path),
                kind="draft",
                set_code=code,
                event="PremierDraft",
                fmt="draft",
                size=pick_path.stat().st_size,
                fingerprint=f"fp_{code}_v1",
            )
        )
    return pick_path


def test_caching_invalidation_write_dir_bundled_and_provisional(tmp_path, monkeypatch):
    home = tmp_path / "app_home"
    data = tmp_path / "raw_data"
    monkeypatch.setenv("SEALED_LAB_HOME", str(home))

    # 1) Missing pick file -> unavailable, no cache file written
    _setup_fake_home_set(home, data, "NOP", provisional=False, with_pick_file=False)
    res_no = D.draft_sleepers("NOP", n_boot=30)
    assert res_no["available"] is False
    assert "No PremierDraft pick file" in res_no["reason"]
    assert not (home / "sets" / "NOP" / "model" / "draft_sleepers.json").exists()

    # 2) Provisional set with pick file -> computes with provisional values & bootstrap halves, portable cache
    _setup_fake_home_set(home, data, "PRV", provisional=True, with_pick_file=True)
    # read_only=True writes nothing
    res_ro = D.draft_sleepers("PRV", read_only=True, n_boot=30)
    assert res_ro["available"] is True
    assert res_ro["provisional"] is True
    assert res_ro["value_source"] == "provisional"
    assert res_ro["halves_source"] == "bootstrap_draws"
    assert not (home / "sets" / "PRV" / "model" / "draft_sleepers.json").exists()

    # write_dir writes ONLY to external dir, never APP_HOME
    ext = tmp_path / "bundle_model"
    res_wd = D.draft_sleepers("PRV", write_dir=ext, n_boot=30)
    assert res_wd["available"] is True
    assert (ext / "draft_sleepers.json").exists()
    assert not (home / "sets" / "PRV" / "model" / "draft_sleepers.json").exists()
    # Portability: no absolute paths in JSON
    raw_txt = (ext / "draft_sleepers.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in raw_txt
    assert str(data) not in raw_txt
    assert "draft_data_public" not in raw_txt

    # Normal call writes cache, second call hits cache
    res1 = D.draft_sleepers("PRV", n_boot=30)
    assert res1["cached"] is False
    assert "Card 11" in [r["name"] for r in res1["sleepers_robust"]]
    assert "Card 0" in [r["name"] for r in res1["overdrafted_robust"]]
    res2 = D.draft_sleepers("PRV", n_boot=30)
    assert res2["cached"] is True

    # Invalidates when pick file fingerprint changes
    from sealed_lab.ingest import registry

    pick_p = data / "PRV" / "draft_data_public.PRV.PremierDraft.csv.gz"
    registry.register(
        registry.FileInfo(
            path=str(pick_p),
            kind="draft",
            set_code="PRV",
            event="PremierDraft",
            fmt="draft",
            size=pick_p.stat().st_size,
            fingerprint="fp_PRV_v2",
        )
    )
    res3 = D.draft_sleepers("PRV", n_boot=30)
    assert res3["cached"] is False
    assert res3["pick_fingerprints"] == [["PremierDraft", "fp_PRV_v2"]]

    # Bundled / ready-made set: remove pick file from disk -> still serves model/draft_sleepers.json!
    pick_p.unlink()
    res_bundled = D.draft_sleepers("PRV", n_boot=30)
    assert res_bundled["available"] is True
    assert res_bundled["cached"] is True
    assert "Card 11" in [r["name"] for r in res_bundled["sleepers_robust"]]


def test_bundle_sets_precomputes_draft_sleepers_without_touching_home(tmp_path, monkeypatch):
    home = tmp_path / "app_home"
    data = tmp_path / "raw_data"
    out = tmp_path / "bundle_out"
    out.mkdir()
    monkeypatch.setenv("SEALED_LAB_HOME", str(home))
    _setup_fake_home_set(home, data, "TRK", provisional=True, with_pick_file=True)

    sys.path.insert(0, str(ROOT / "tools"))
    import bundle_sets  # noqa: E402
    from sealed_lab import deck_search

    # this fixture has no pick-deck tables (the real tool refuses such a set earlier, see test_bundled_provisional.py); only the
    # draft-sleepers cache is under test here
    monkeypatch.setattr(deck_search, "model_scores", lambda *a, **k: None)
    bundle_sets._copy_set("TRK", out, with_written=False)
    bundled_json = out / "TRK" / "model" / "draft_sleepers.json"
    assert bundled_json.exists()
    assert not (home / "sets" / "TRK" / "model" / "draft_sleepers.json").exists()
    loaded = json.loads(bundled_json.read_text(encoding="utf-8"))
    assert loaded["available"] is True
    assert loaded["provisional"] is True
    assert str(tmp_path) not in bundled_json.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- Report tab GUI tests
def _wait_until(qapp, pred, timeout: float = 10.0, what: str = "condition"):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        qapp.processEvents()
        if pred():
            qapp.processEvents()
            return
        time.sleep(0.005)
    raise AssertionError(f"timeout waiting for {what}")


def test_report_tab_draft_sleepers_normal_missing_and_provisional(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.path.isdir(r"C:\Windows\Fonts"):
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    os.environ["SEALED_LAB_FAKE_API"] = "1"
    os.environ["SEALED_LAB_FAKE_DELAY"] = "0.002"
    cfg = tempfile.mkdtemp(prefix="sealed_lab_ds_cfg_")
    monkeypatch.setenv("SEALED_LAB_SETTINGS", os.path.join(cfg, "settings.ini"))

    from PySide6.QtWidgets import QApplication, QLabel

    from sealed_lab import provisional as prov
    from sealed_lab.gui import fake_api, workers
    from sealed_lab.gui.report_tab import ReportTab
    from sealed_lab.gui.widgets import apply_theme, new_settings

    monkeypatch.setattr(workers, "_closing", False)
    app = QApplication.instance() or QApplication([])
    apply_theme(app, "light")
    fake_api.reset()
    fake_api.add_pick_only_set("PRV", calculated=True)
    new_settings().clear()

    rt = ReportTab(fake_api)
    try:
        rt.show()
        rt.refresh_sets(fake_api.list_sets())
        # 1) Normal set (TST): section rendered with robust lists, explanation, and drift table
        rt.select_set("TST")
        _wait_until(app, lambda: rt.current is not None and not rt.task.running, what="TST report")
        assert "draft_sleepers" in rt._secs
        sec = rt._secs["draft_sleepers"]
        txt = " ".join(lab.text() for lab in sec.findChildren(QLabel))
        assert "Drafting: value vs pick position" in sec.title
        assert "expected from pick position" in txt.lower()
        assert "Drafted later than its value" in txt
        assert "Drafted earlier than its value" in txt
        assert "Early-vs-late pick position drift" in txt
        assert "Drafted earlier over time" in txt
        assert "Drafted later over time" in txt

        # 2) Set without PremierDraft pick file: plain message, no empty tables
        orig_ds = fake_api.draft_sleepers
        monkeypatch.setattr(
            fake_api,
            "draft_sleepers",
            lambda code, refresh=False: dict(
                available=False,
                set_code=code,
                reason=f"No PremierDraft pick file is available for {code}.",
            ),
        )
        rt.draft_sleepers.clear()
        rt.reports.clear()
        rt._loaded_key = None
        rt.load()
        _wait_until(app, lambda: rt.current is not None and not rt.task.running, what="TST no-pick report")
        sec_no = rt._secs["draft_sleepers"]
        txt_no = " ".join(lab.text() for lab in sec_no.findChildren(QLabel))
        assert "No PremierDraft pick file is available for TST." in txt_no
        assert "Drafted later than its value" not in txt_no
        assert "Early-vs-late pick position drift" not in txt_no

        # 3) Provisional set (PRV): unavailable banner stays visible, set_report NOT called,
        #    and Drafting section renders labelled "Provisional values"
        monkeypatch.setattr(fake_api, "draft_sleepers", orig_ds)
        orig_sr = fake_api.set_report

        def guarded(code, *a, **k):
            if code == "PRV":
                pytest.fail("a provisional set must not call set_report")
            return orig_sr(code, *a, **k)

        monkeypatch.setattr(fake_api, "set_report", guarded)
        rt.select_set("PRV")
        _wait_until(
            app,
            lambda: "draft_sleepers" in rt._secs and not rt.task.running,
            what="PRV provisional draft_sleepers",
        )
        assert rt.message.isVisible()
        assert rt.message.text() == prov.UNAVAILABLE
        assert not rt.refresh_btn.isEnabled()
        sec_p = rt._secs["draft_sleepers"]
        txt_p = " ".join(lab.text() for lab in sec_p.findChildren(QLabel))
        assert "Provisional values" in txt_p
        assert "Drafted later than its value" in txt_p
        assert "Drafted earlier than its value" in txt_p
    finally:
        rt.wait(3000)
        rt.close()
        app.processEvents()
        monkeypatch.setattr(workers, "_closing", False)
