"""Ready-made sets in the release zip: sealed_lab.bundled (install at start), tools/bundle_sets.py (build the folder) and
tools/release_sign.py's merge of that folder into the signed zip.  Fake sets (no model is fitted), no network, no gh."""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import pytest

from sealed_lab import api, bundled, config, updater, written_report
from sealed_lab.ingest import registry
from sealed_lab.types import SetBusy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bundle_sets  # noqa: E402
import export_public  # noqa: E402
import release_sign  # noqa: E402

FP = "fp0000000000000001"
TABLES = ("cards", "games", "decks", "deck_cards", "builds", "hands", "pools", "pool_audit")


def make_set(home: Path, code: str, built_at: str = "2026-11-01T00:00:00+00:00", own_path: str = "C:" + r"\Users\RealName\data\g.csv.gz",
             register: bool = True, written: bool = True, report_stamp: str | None = None) -> None:
    """A set as the owner's app home holds it: model, corpus tables + manifest (with a personal path), cache, registry, written report."""
    sd = home / "sets" / code
    for sub in ("model", "corpus", "cache", "corpus/_shards", "cache/validation"):
        (sd / sub).mkdir(parents=True, exist_ok=True)
    corpus_at = "2026-10-31T00:00:00+00:00"
    (sd / "model" / "meta.json").write_text(json.dumps(dict(set_code=code, built_at=built_at, formats=["bo3"], built_corpus_at=corpus_at,
                                                           source_files=[f"game:{FP}"])), encoding="utf-8")
    import numpy as np
    import polars as pl

    np.savez(sd / "model" / "draws.npz", beta=np.arange(3.0), names=np.array([code, "card"]))   # real files: the scan decodes them
    for t in TABLES:
        pl.DataFrame({"table": [t], "set": [code]}).write_parquet(sd / "corpus" / f"{t}.parquet")
    (sd / "corpus" / "_shards" / "s.parquet").write_bytes(b"shard")
    (sd / "corpus" / "manifest.json").write_text(json.dumps(dict(
        set=code, built_at=corpus_at, files=[dict(path=own_path, fingerprint=FP, kind="game", event="TradSealed", fmt="bo3", rows=5)],
        audit=dict(n_games_by_fmt=dict(bo3=5)), warnings=[f"read {own_path} ok"])), encoding="utf-8")
    stamp = report_stamp or f"{built_at}|v7"
    (sd / "cache" / "report_bo3.json").write_text(json.dumps(dict(stamp=stamp, set_code=code, fmt="bo3")), encoding="utf-8")
    for name in ("scryfall.json", "creature_support.json", "archetype_proj_bo3.json", "phi_cache.npz", "cast_cache_x.npz", "sim_keep.json"):
        (sd / "cache" / name).write_bytes(b"{}")
    (sd / "cache" / "validation" / "v.json").write_text("{}", encoding="utf-8")
    if register:
        (sd / "registry.json").write_text(json.dumps(dict(version=1, files=[dict(path=own_path, kind="game", set_code=code, event="TradSealed",
                                                                                 fmt="bo3", size=1, fingerprint=FP, mtime=1.0)])), encoding="utf-8")
    if written:
        rep, facts = home / f"{code}.rep.md", home / f"{code}.facts.json"
        rep.write_text("A plain sentence.\n", encoding="utf-8")
        facts.write_text(json.dumps(dict(set=code, stamps=dict(bo3=stamp))), encoding="utf-8")
        old = config.app_home
        config.app_home = lambda: home                       # written_report.install writes to the app home
        try:
            written_report.install(code, rep, facts)
        finally:
            config.app_home = old


def dates(home: Path, **d: str) -> None:
    (home / "scryfall_sets.json").write_text(json.dumps(dict(fetched_at=time.time(), sets=d)), encoding="utf-8")


def build_bundle(tmp_path: Path, name: str, codes: dict[str, str]) -> Path:
    """A bundle folder for fake sets {code: built_at}, made by the real tool from a scratch owner home."""
    owner = tmp_path / f"owner_{name}"
    for c, b in codes.items():
        make_set(owner, c, b)
    dates(owner, **{c: f"2026-0{i + 1}-01" for i, c in enumerate(codes)})
    import os

    old = os.environ.get("SEALED_LAB_HOME")
    os.environ["SEALED_LAB_HOME"] = str(owner)
    try:
        bundle_sets.build(tmp_path / name, list(codes), 5, False)
    finally:
        if old is None:
            os.environ.pop("SEALED_LAB_HOME", None)
        else:
            os.environ["SEALED_LAB_HOME"] = old
    return tmp_path / name


def put_in_app(bundle: Path, app: Path) -> Path:
    import shutil

    app.mkdir(exist_ok=True)
    shutil.copytree(bundle, app / "bundled_sets")
    return app


# ------------------------------------------------------------------------------------------------ the bundle
def test_bundle_contents_and_sanitised_manifest(tmp_path, home):
    b = build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00"})
    files = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    assert "bundle.json" in files and "AAA/model/meta.json" in files and "AAA/written/AAA.md" in files
    assert "AAA/cache/report_bo3.json" in files and "AAA/cache/scryfall.json" in files and "AAA/cache/archetype_proj_bo3.json" in files
    assert not [f for f in files if "_shards" in f or "phi_cache" in f or "cast_cache" in f or "validation" in f or "sim_keep" in f
                or "registry" in f]
    man = json.loads((b / "AAA" / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    assert man["files"][0]["path"] == "g.csv.gz" and man["files"][0]["fingerprint"] == FP and man["files"][0]["kind"] == "game"
    assert "RealName" not in json.dumps(man) and man["warnings"] == ["read g.csv.gz ok"]
    bj = json.loads((b / "bundle.json").read_text(encoding="utf-8"))
    assert bj["sets"]["AAA"]["built_at"] == "2026-11-01T00:00:00+00:00" and bj["sets"]["AAA"]["release_date"] == "2026-01-01"


def test_sanitize_manifest_keeps_everything_but_paths():
    m = dict(set="X", files=[dict(path="/home" + "/bob/a.csv", fingerprint="f", n=3), dict(src_path="D:\\x\\y.gz")], note="from C:" + "\\Users\\bob\\z.csv here",
             nested=dict(folder="/mnt/data/dir/"))
    out = bundled.sanitize_manifest(m)
    assert out == dict(set="X", files=[dict(path="a.csv", fingerprint="f", n=3), dict(src_path="y.gz")], note="from z.csv here", nested=dict(folder="dir"))


def test_bundle_sets_picks_the_newest_and_refuses_problems(tmp_path, monkeypatch):
    owner = tmp_path / "owner"
    for c in ("AAA", "BBB", "CCC"):
        make_set(owner, c)
    dates(owner, AAA="2026-01-01", BBB="2026-03-01", CCC="2026-02-01")
    monkeypatch.setenv("SEALED_LAB_HOME", str(owner))
    res = bundle_sets.build(tmp_path / "o1", None, 2, False)
    assert sorted(res["sets"]) == ["BBB", "CCC"]
    # a stale set (a registered file the model was not computed from)
    reg = owner / "sets" / "BBB" / "registry.json"
    d = json.loads(reg.read_text(encoding="utf-8"))
    d["files"].append(dict(d["files"][0], fingerprint="other", event="Sealed", fmt="bo1"))
    reg.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SystemExit):
        bundle_sets.build(tmp_path / "o2", None, 2, False)
    assert not (tmp_path / "o2").exists()
    reg.write_text(json.dumps(dict(d, files=d["files"][:1])), encoding="utf-8")
    # no written report / outdated stamp
    (owner / "written_reports" / "CCC.md").unlink()
    with pytest.raises(SystemExit):
        bundle_sets.build(tmp_path / "o3", None, 2, False)
    res = bundle_sets.build(tmp_path / "o4", None, 2, True)
    assert "CCC" in res["notes"] and not (tmp_path / "o4" / "CCC" / "written").exists() and (tmp_path / "o4" / "BBB" / "written" / "BBB.md").exists()
    cache = owner / "sets" / "BBB" / "cache" / "report_bo3.json"
    cache.write_text(json.dumps(dict(stamp="newer|v7", set_code="BBB", fmt="bo3")), encoding="utf-8")        # the written text is now outdated
    with pytest.raises(SystemExit):
        bundle_sets.build(tmp_path / "o5", ["BBB"], 5, False)
    # failing the checker
    cache.write_text(json.dumps(dict(stamp="2026-11-01T00:00:00+00:00|v7", set_code="BBB", fmt="bo3")), encoding="utf-8")
    w = owner / "written_reports" / "BBB.md"
    w.write_text("Coffee beats tea by 12% (win rate).\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        bundle_sets.build(tmp_path / "o6", ["BBB"], 5, False)
    # a set without a model
    (owner / "sets" / "AAA" / "model" / "meta.json").unlink()
    with pytest.raises(SystemExit):
        bundle_sets.build(tmp_path / "o7", ["AAA"], 5, True)


def test_bundle_sets_refuses_a_personal_string_and_cleans_up(tmp_path, monkeypatch):
    owner = tmp_path / "owner"
    make_set(owner, "AAA")
    dates(owner, AAA="2026-01-01")
    (owner / "sets" / "AAA" / "model" / "extra.json").write_text(json.dumps(dict(p="C:" + r"\Users\RealName\x")), encoding="utf-8")
    monkeypatch.setenv("SEALED_LAB_HOME", str(owner))
    with pytest.raises(SystemExit):
        bundle_sets.build(tmp_path / "out", ["AAA"], 5, False)
    assert not (tmp_path / "out").exists()
    (owner / "sets" / "AAA" / "model" / "extra.json").unlink()
    (owner / "sets" / "AAA" / "model" / "blob.npz").write_bytes(b"\0\0C:" + b"\\Users\\RealName\\x\0\0")      # in a binary file too
    with pytest.raises(SystemExit):
        bundle_sets.build(tmp_path / "out", ["AAA"], 5, False)
    with pytest.raises(SystemExit):                                                         # a folder that is not empty is never mixed into
        (tmp_path / "full").mkdir()
        (tmp_path / "full" / "x").write_text("x", encoding="utf-8")
        bundle_sets.build(tmp_path / "full", ["AAA"], 5, False)


# ------------------------------------------------------------------------------------------------ installation
def test_install_into_an_empty_home(tmp_path, home):
    app = put_in_app(build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00"}), tmp_path / "app")
    res = bundled.install_bundled(app)
    assert res["installed"] == ["AAA"] and not res["errors"]
    assert not (app / "bundled_sets").exists()
    sd = home / "sets" / "AAA"
    assert (sd / "model" / "meta.json").is_file() and (sd / "corpus" / "games.parquet").is_file() and (sd / "cache" / "report_bo3.json").is_file()
    mk = json.loads((sd / "bundled.json").read_text(encoding="utf-8"))
    assert mk["built_at"] == "2026-11-01T00:00:00+00:00" and mk["complete"] and mk["written"] and mk["bundle_version"]
    assert registry.known_sets() == ["AAA"] and registry.files("AAA", include_shared=False) == []
    assert written_report.load("AAA") is not None
    st = api.set_status("AAA")
    assert st.has_model and st.has_corpus and not st.stale and not st.pending and st.formats == ["bo3"]
    assert st.note.startswith("ready-made model from sealed_lab ") and "(built 2026-11-01)" in st.note
    assert [s.set_code for s in api.list_sets()] == ["AAA"]
    assert bundled.install_bundled(app)["installed"] == []                               # nothing to do without a bundle


def test_a_set_with_own_files_or_an_own_model_is_never_touched(tmp_path, home):
    app = put_in_app(build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00", "BBB": "2026-11-01T00:00:00+00:00",
                                                  "CCC": "2026-11-01T00:00:00+00:00"}), tmp_path / "app")
    make_set(home, "AAA", built_at="2026-05-05T00:00:00+00:00", own_path="/home/someone/a.gz", written=False)          # registered files
    make_set(home, "BBB", register=False, written=False)                                                               # own model, no files
    (home / "sets" / "CCC").mkdir(parents=True)                                                                         # nothing there yet
    (home / "written_reports").mkdir()
    (home / "written_reports" / "CCC.md").write_text("mine\n", encoding="utf-8")
    (home / "written_reports" / "CCC.json").write_text("{}", encoding="utf-8")
    res = bundled.install_bundled(app)
    assert sorted(res["own"]) == ["AAA", "BBB"] and res["installed"] == ["CCC"]
    assert json.loads((home / "sets" / "AAA" / "model" / "meta.json").read_text(encoding="utf-8"))["built_at"] == "2026-05-05T00:00:00+00:00"
    assert not (home / "sets" / "AAA" / "bundled.json").exists() and not (home / "sets" / "BBB" / "bundled.json").exists()
    assert (home / "written_reports" / "CCC.md").read_text(encoding="utf-8") == "mine\n"          # the friend's own written report stays
    assert not json.loads((home / "sets" / "CCC" / "bundled.json").read_text(encoding="utf-8"))["written"]


def test_a_newer_bundle_replaces_the_old_ready_made_set_and_deletes_what_it_dropped(tmp_path, home):
    b1 = build_bundle(tmp_path, "b1", {"AAA": "2026-11-01T00:00:00+00:00", "OLD": "2026-11-01T00:00:00+00:00"})
    bundled.install_bundled(put_in_app(b1, tmp_path / "app1"))
    (home / "sets" / "AAA" / "model" / "leftover.bin").write_bytes(b"old")
    (home / "sets" / "AAA" / "cache" / "phi_cache.npz").write_bytes(b"old")
    assert (home / "sets" / "OLD" / "model").is_dir() and written_report.load("OLD") is not None
    b2 = build_bundle(tmp_path, "b2", {"AAA": "2026-11-20T00:00:00+00:00", "NEW": "2026-11-20T00:00:00+00:00"})
    res = bundled.install_bundled(put_in_app(b2, tmp_path / "app2"))
    assert sorted(res["installed"]) == ["AAA", "NEW"] and res["dropped"] == ["OLD"]
    assert not (home / "sets" / "AAA" / "model" / "leftover.bin").exists() and not (home / "sets" / "AAA" / "cache" / "phi_cache.npz").exists()
    assert json.loads((home / "sets" / "AAA" / "model" / "meta.json").read_text(encoding="utf-8"))["built_at"] == "2026-11-20T00:00:00+00:00"
    assert not (home / "sets" / "OLD").exists() and written_report.load("OLD") is None
    assert registry.known_sets() == ["AAA", "NEW"]


def test_the_same_version_is_left_alone(tmp_path, home):
    b = build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00"})
    bundled.install_bundled(put_in_app(b, tmp_path / "app1"))
    (home / "sets" / "AAA" / "extra.txt").write_text("x", encoding="utf-8")
    res = bundled.install_bundled(put_in_app(b, tmp_path / "app2"))
    assert res["current"] == ["AAA"] and (home / "sets" / "AAA" / "extra.txt").exists() and not (tmp_path / "app2" / "bundled_sets").exists()


def test_a_dropped_set_the_user_registered_files_for_is_kept(tmp_path, home):
    b1 = build_bundle(tmp_path, "b1", {"AAA": "2026-11-01T00:00:00+00:00", "OLD": "2026-11-01T00:00:00+00:00"})
    bundled.install_bundled(put_in_app(b1, tmp_path / "app1"))
    reg = home / "sets" / "OLD" / "registry.json"
    reg.write_text(json.dumps(dict(version=1, files=[dict(path="/home/someone/o.gz", kind="game", set_code="OLD", event="Sealed", fmt="bo1",
                                                          size=1, fingerprint="mine", mtime=1.0)])), encoding="utf-8")
    b2 = build_bundle(tmp_path, "b2", {"AAA": "2026-11-20T00:00:00+00:00"})
    res = bundled.install_bundled(put_in_app(b2, tmp_path / "app2"))
    assert res["dropped"] == []
    assert (home / "sets" / "OLD" / "model" / "meta.json").exists() and not (home / "sets" / "OLD" / "bundled.json").exists()
    assert api.set_status("OLD").note == "" and api.set_status("OLD").stale          # now an ordinary set whose files are not yet used


def test_a_set_being_calculated_waits_and_the_bundle_stays(tmp_path, home):
    app = put_in_app(build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00", "BBB": "2026-11-01T00:00:00+00:00"}), tmp_path / "app")
    with registry.calc_lock("AAA"):
        res = bundled.install_bundled(app)
    assert res["busy"] == ["AAA"] and res["installed"] == ["BBB"] and (app / "bundled_sets").exists()
    res = bundled.install_bundled(app)                      # the next start: BBB is current, AAA goes in, the bundle is deleted
    assert res["installed"] == ["AAA"] and res["current"] == ["BBB"] and not (app / "bundled_sets").exists()
    with pytest.raises(SetBusy):
        with registry.calc_lock("AAA"), registry.calc_lock("AAA"):
            pass


def test_an_incomplete_bundle_is_an_error_not_a_half_set(tmp_path, home):
    app = put_in_app(build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00"}), tmp_path / "app")
    (app / "bundled_sets" / "AAA" / "model" / "meta.json").unlink()
    res = bundled.install_bundled(app)
    assert res["errors"] and (app / "bundled_sets").exists() and not (home / "sets" / "AAA").exists()


def test_main_never_fails_over_the_bundle(monkeypatch):
    import sealed_lab.__main__ as m

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(bundled, "install_bundled", boom)
    m._install_bundled_sets()                              # logged, not raised


# ------------------------------------------------------------------------------------------------ status
def test_set_status_of_a_ready_made_set_and_after_the_user_registers_files(tmp_path, home):
    bundled.install_bundled(put_in_app(build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00"}), tmp_path / "app"))
    st = api.set_status("AAA")
    assert st.has_model and not st.stale and st.missing == [] and not any('source file' in w for w in st.warnings)
    # the user registers files of their own (not the ones the model was computed from): normal rules, the note stays until a calculation
    # (the registry file is written directly: there is no csv on disk)
    reg = home / "sets" / "AAA" / "registry.json"
    reg.write_text(json.dumps(dict(version=1, files=[dict(path=str(tmp_path / "mine.csv.gz"), kind="game", set_code="AAA", event="Sealed",
                                                          fmt="bo1", size=1, fingerprint="mine", mtime=1.0)])), encoding="utf-8")
    st = api.set_status("AAA")
    assert st.stale and st.pending and st.note
    # ... and the same files the model was computed from keep it up to date
    reg.write_text(json.dumps(dict(version=1, files=[dict(path=str(tmp_path / "g.csv.gz"), kind="game", set_code="AAA", event="TradSealed",
                                                          fmt="bo3", size=1, fingerprint=FP, mtime=1.0)])), encoding="utf-8")
    assert not api.set_status("AAA").stale
    bundled.clear_marker("AAA")                              # what a successful calculation does
    st = api.set_status("AAA")
    assert st.note == "" and not st.stale


def test_calculate_set_clears_the_marker(tmp_path, home, monkeypatch):
    bundled.install_bundled(put_in_app(build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00"}), tmp_path / "app"))
    import sealed_lab.ingest.corpus as corpus
    import sealed_lab.model.pipeline as pipeline

    monkeypatch.setattr(corpus, "build_corpus", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "calculate", lambda *a, **k: None)
    monkeypatch.setattr(api, "_precompute_analysis", lambda sc: None)
    assert bundled.read_marker("AAA") is not None
    api.calculate_set("AAA")
    assert bundled.read_marker("AAA") is None and not (home / "sets" / "AAA" / "bundled.json").exists()


def test_untracking_a_ready_made_set_removes_it(tmp_path, home):
    bundled.install_bundled(put_in_app(build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00"}), tmp_path / "app"))
    api.untrack_set("AAA")
    assert registry.known_sets() == [] and not (home / "sets" / "AAA").exists()


# ------------------------------------------------------------------------------------------------ release zip
def _ci_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, mode, data in (("sealed_lab-9.9.9/pyproject.toml", 0o644, b"[project]\n"), ("sealed_lab-9.9.9/run_linux.sh", 0o755, b"#!/bin/sh\n"),
                                 ("sealed_lab-9.9.9/sealed_lab/__init__.py", 0o644, b'__version__ = "9.9.9"\n')):
            zi = zipfile.ZipInfo(name, (2026, 11, 24, 10, 0, 0))
            zi.create_system, zi.external_attr = 3, (0o100000 | mode) << 16
            z.writestr(zi, data)
    return path


def test_release_sign_merges_the_bundle_into_the_ci_zip(tmp_path, home):
    b = build_bundle(tmp_path, "b", {"AAA": "2026-11-01T00:00:00+00:00", "BBB": "2026-11-02T00:00:00+00:00"})
    ci = _ci_zip(tmp_path / "ci.zip")
    out = release_sign.merge_bundle(ci, b, tmp_path / "m1" / "x.zip")
    out2 = release_sign.merge_bundle(ci, b, tmp_path / "m2" / "x.zip")
    assert out.read_bytes() == out2.read_bytes()                                       # deterministic
    with zipfile.ZipFile(out) as z, zipfile.ZipFile(ci) as c:
        names = [i.filename for i in z.infolist()]
        assert names[:3] == [i.filename for i in c.infolist()]                          # the CI entries first, unchanged
        for i in c.infolist():
            j = z.getinfo(i.filename)
            assert z.read(j) == c.read(i) and j.external_attr == i.external_attr and j.date_time == i.date_time
        bun = names[3:]
        assert bun == sorted(bun) and all(n.startswith("sealed_lab-9.9.9/bundled_sets/") for n in bun)
        assert "sealed_lab-9.9.9/bundled_sets/bundle.json" in bun and "sealed_lab-9.9.9/bundled_sets/AAA/model/meta.json" in bun
        assert {z.getinfo(n).date_time for n in bun} == {release_sign.BUNDLE_TIME}
    # the rebuild from source has no bundle: compared without it the zips are identical, and without skipping they are not
    assert release_sign.compare_zips(out, ci, skip_bundle=True) == []
    assert any("only in the release zip" in d for d in release_sign.compare_zips(out, ci))
    assert release_sign.compare_zips(_ci_zip(tmp_path / "ci2.zip"), out, skip_bundle=False)       # a zip lacking the bundle differs from one with it
    info = release_sign.bundle_info(out)
    assert sorted(info["sets"]) == ["AAA", "BBB"] and "AAA (built 2026-11-01)" in release_sign.describe_bundle(info)
    assert release_sign.bundle_info(ci) is None
    # merging again over an already merged zip replaces the bundle instead of duplicating it
    again = release_sign.merge_bundle(out, b, tmp_path / "m3" / "x.zip")
    assert release_sign.zip_duplicates(again) == [] and again.read_bytes() == out.read_bytes()
    # the merged zip is acceptable to the updater and the bundle lands where install_bundled looks for it
    staged_root = tmp_path / "staged"
    with zipfile.ZipFile(out) as z:
        z.extractall(staged_root)
    assert bundled.bundle_root(staged_root / "sealed_lab-9.9.9").joinpath("bundle.json").is_file()


def test_release_sign_refuses_a_bundle_folder_without_bundle_json_or_with_links(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        release_sign.merge_bundle(_ci_zip(tmp_path / "ci.zip"), tmp_path / "empty", tmp_path / "o.zip")


def test_the_updater_accepts_a_bundle_sized_zip_and_export_never_ships_it():
    assert updater.MAX_UNPACKED_BYTES >= 1024 * 1024 * 1024 and updater.MAX_MEMBERS >= 2000
    assert export_public.selected(["bundled_sets/x.json", "sealed_lab/bundled_sets/x.json", "sealed_lab/bundled.py"]) == ["sealed_lab/bundled.py"]
    assert "tools/bundle_sets.py" in export_public.ALLOW
    for ig in (ROOT / "tools" / "public.gitignore", ROOT.parent / ".gitignore"):       # (the second is the development repository's)
        if ig.is_file():
            assert "bundled_sets/" in ig.read_text(encoding="utf-8").splitlines()


def test_the_decoded_scan_finds_a_path_inside_parquet_and_npz_contents(tmp_path):
    """Compression can hide a path from the byte scan: the string columns of a .parquet and the string arrays of a .npz are decoded and
    scanned (bundle_sets._decoded_hits), and a clean file gives no hit."""
    import numpy as np
    import polars as pl

    ep, pats = export_public, export_public.deny_patterns()
    bad_pq, bad_npz, ok_pq = tmp_path / "bad.parquet", tmp_path / "bad.npz", tmp_path / "ok.parquet"
    pl.DataFrame({"src": ["C:" + r"\Users\RealName\data\game.csv.gz"], "n": [1]}).write_parquet(bad_pq)
    np.savez(bad_npz, names=np.array(["card", "/home" + "/realname/x"]))
    pl.DataFrame({"src": ["game_data_public.AAA.TradSealed.csv.gz"], "n": [1]}).write_parquet(ok_pq)
    assert bundle_sets._decoded_hits(bad_pq, "bad.parquet", ep, pats)
    assert bundle_sets._decoded_hits(bad_npz, "bad.npz", ep, pats)
    assert bundle_sets._decoded_hits(ok_pq, "ok.parquet", ep, pats) == []
