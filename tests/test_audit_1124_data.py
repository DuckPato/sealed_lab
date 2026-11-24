"""Audit 2026-11-24, data consistency (B1-B5) and the api / CLI items: card-id remap in the builder's corpus tables, all-or-nothing
corpus and model writes, the cross-process calculation lock, state files that are never silently emptied, corpus / model stamps,
cross_set from stored reports only, CLI messages.  Synthetic data only; nothing here fits a model."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from sealed_lab import api, calcproc, cli, config, cross_set, report
from sealed_lab.deck import builder
from sealed_lab.ingest import corpus, datasets, registry
from sealed_lab.model import artifact
from sealed_lab.types import Cancelled, SetBusy


def _write_corpus_cards(set_code: str, names: list[str]) -> None:
    cd = config.corpus_dir(set_code)
    cd.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"card_id": list(range(len(names))), "name": names}).write_parquet(cd / "cards.parquet")


def _model(set_code: str, names: list[str], creature: set[str]) -> SimpleNamespace:
    cards = pl.DataFrame({"card_id": list(range(len(names))), "name": names, "is_creature": [n in creature for n in names],
                          "is_land": [n == "Forest" for n in names]})
    return SimpleNamespace(set_code=set_code, cards=cards, meta={"set_code": set_code})


# ---------------------------------------------------------------------------------------------
# B1: the builder's creature table follows the model's card ids
# ---------------------------------------------------------------------------------------------

def test_creature_table_uses_remap_when_corpus_numbering_differs(home):
    model_names = ["Bear", "Forest", "Zap"]                       # model: dense over sorted names
    corpus_names = ["Ant", "Bear", "Forest", "Zap"]               # rebuilt corpus saw one more card
    model = _model("TST", model_names, {"Bear"})
    cd = config.corpus_dir("TST")
    _write_corpus_cards("TST", corpus_names)
    pl.DataFrame({"fmt": ["bo3"], "deck_key": [1]}).write_parquet(cd / "games.parquet")
    pl.DataFrame({"deck_key": [1], "n_cards": [40]}).write_parquet(cd / "decks.parquet")
    # deck: 2 Ant (corpus id 0, unknown to the model), 3 Bear (1), 5 Forest (2), 2 Zap (3)
    pl.DataFrame({"deck_key": [1] * 4, "card_id": [0, 1, 2, 3], "n": [2, 3, 5, 2]}).write_parquet(cd / "deck_cards.parquet")
    per, dcards = builder._corpus_creature_table(model, "bo3")
    assert per["cre"].to_list() == [3]                           # unmapped, corpus id 1 would have read as Forest: 0 creatures
    assert sorted(dcards["card_id"].to_list()) == [0, 1, 2]     # model ids; the card unknown to the model is dropped
    assert builder._corpus_remap_tag(model) != ""
    _write_corpus_cards("TST", model_names)                      # same numbering: no translation, empty tag
    assert builder._corpus_remap_tag(model) == ""


# ---------------------------------------------------------------------------------------------
# B2: the corpus tables are replaced together, the manifest last
# ---------------------------------------------------------------------------------------------

@pytest.fixture()
def tiny(home, tmp_path, monkeypatch):
    from sealed_lab.synth.generate import generate

    out = tmp_path / "syn"
    generate(str(out), set_code="TST", seed=1, scale=0.03)
    api.add_files([str(out)])
    monkeypatch.setattr(corpus, "run_ingest_tasks",
                        lambda tasks, n_jobs, progress, cancel: {t["fingerprint"]: corpus.ingest_worker(t) for t in tasks})
    corpus.build_corpus("TST", n_jobs=1, fetch_scryfall=False)
    return home


def _snapshot(set_code: str) -> dict[str, bytes]:
    cd = config.corpus_dir(set_code)
    return {t: (cd / f"{t}.parquet").read_bytes() for t in corpus.TABLES} | {"manifest": (cd / "manifest.json").read_bytes()}


def test_cancel_during_table_writes_leaves_the_old_corpus_whole(tiny):
    prev = json.loads(corpus.manifest_path("TST").read_text(encoding="utf-8"))
    prev["signature"] = "changed"                                 # forces a rebuild (no quick path), old tables stay in place
    corpus.manifest_path("TST").write_text(json.dumps(prev), encoding="utf-8")
    before = _snapshot("TST")
    cancel = threading.Event()

    def prog(frac, msg):
        if msg == "Writing tables":
            cancel.set()

    with pytest.raises(Cancelled):
        corpus.build_corpus("TST", n_jobs=1, fetch_scryfall=False, progress=prog, cancel=cancel)
    assert _snapshot("TST") == before
    assert not list(config.corpus_dir("TST").glob("*.stage"))


def test_failure_mid_swap_is_detected_by_the_missing_manifest(tiny, monkeypatch):
    prev = json.loads(corpus.manifest_path("TST").read_text(encoding="utf-8"))
    prev["signature"] = "changed"
    corpus.manifest_path("TST").write_text(json.dumps(prev), encoding="utf-8")
    real = os.replace

    def flaky(src, dst, *a, **k):
        if str(dst).endswith("deck_cards.parquet"):
            raise OSError("disk full")
        return real(src, dst, *a, **k)

    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(corpus.os, "replace", flaky)
        corpus.build_corpus("TST", n_jobs=1, fetch_scryfall=False)
    assert corpus.read_manifest("TST") is None                   # no commit marker: the mixed tables are not trusted
    assert api.set_status("TST").has_corpus is False


# ---------------------------------------------------------------------------------------------
# B3: meta.json is the last file of a model save
# ---------------------------------------------------------------------------------------------

def _save_args():
    cards = pl.DataFrame({"card_id": [0, 1], "name": ["A", "B"]})
    draws = {"bo3": dict(beta=np.zeros(2), theta=np.zeros(2), ref=np.zeros(2))}
    return cards, draws, {"bo3": np.zeros(3)}


def test_failed_save_leaves_no_meta(home, monkeypatch):
    d = config.model_dir("TST")
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"formats": ["bo3"], "built_at": "old"}), encoding="utf-8")
    assert api.set_status("TST").has_model
    cards, draws, ref = _save_args()
    real, n = np.savez_compressed, [0]

    def boom(*a, **k):
        n[0] += 1
        if n[0] == 2:                                            # the reference file
            raise OSError("killed")
        return real(*a, **k)

    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(artifact.np, "savez_compressed", boom)
        artifact.save_artifact("TST", {"formats": ["bo3"], "built_at": "new"}, cards, draws, ref, "<html/>", {})
    assert not (d / "meta.json").exists()
    assert api.set_status("TST").has_model is False
    with pytest.raises(FileNotFoundError):
        artifact.SetModel.load("TST")
    with pytest.raises(FileNotFoundError):
        builder.load_model("TST")
    artifact.save_artifact("TST", {"formats": ["bo3"], "built_at": "new"}, cards, draws, ref, "<html/>", {})
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["built_at"] == "new"


def test_save_extras_also_invalidates_meta(home):
    d = config.model_dir("TST")
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text("{}", encoding="utf-8")
    artifact.save_extras(d, None, None, None, None, ["bo3"])
    assert not (d / "meta.json").exists()


# ---------------------------------------------------------------------------------------------
# B4: calculation lock across processes, state files
# ---------------------------------------------------------------------------------------------

def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _lock_file(set_code: str, pid: int) -> None:
    p = registry.calc_lock_path(set_code)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pid": pid, "started": 0}), encoding="utf-8")


def test_calc_lock_is_exclusive_and_released(home):
    with registry.calc_lock("TST"):
        assert registry.calc_lock_holder("TST") == os.getpid()
        with pytest.raises(SetBusy):
            with registry.calc_lock("TST"):
                pass
    assert not registry.calc_lock_path("TST").exists()
    with registry.calc_lock("tst"):
        pass


def test_stale_lock_of_a_dead_process_is_taken_over(home):
    _lock_file("TST", _dead_pid())
    assert registry.calc_lock_holder("TST") is None
    _lock_file("TST", _dead_pid())
    with registry.calc_lock("TST"):
        assert registry.calc_lock_holder("TST") == os.getpid()


def test_other_process_lock_blocks_readers_calculation_and_scan(home):
    _lock_file("TST", os.getppid())                              # a live process that is not this one
    with pytest.raises(SetBusy):
        api._guard("TST")
    with pytest.raises(SetBusy):
        api.calculate_set("TST")
    with pytest.raises(SetBusy):
        api.calculate_set_isolated("TST")
    with pytest.raises(SetBusy):
        api.untrack_set("TST")
    rep = api.scan_data_folders(str(home / "data"), track=["TST"])
    assert "TST" in rep.deferred


def test_calcproc_reports_busy_with_its_own_exit_code(home, monkeypatch, capsys):
    _lock_file("TST", os.getppid())
    monkeypatch.setattr(sys, "argv", ["calcproc", json.dumps({"set_code": "TST", "options": {}, "cancel_file": None, "parent_pid": None})])
    assert calcproc.main() == calcproc.EXIT_BUSY
    assert "already being calculated" in capsys.readouterr().err


def test_unreadable_state_files_are_kept_not_emptied(home):
    rp = registry._reg_path("TST")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("{not json", encoding="utf-8")
    assert registry.entries("TST") == []
    assert rp.with_name(rp.name + ".corrupt").read_text(encoding="utf-8") == "{not json"
    sp = datasets.state_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text("[[", encoding="utf-8")
    assert datasets.load_state() == {}
    assert sp.with_name(sp.name + ".corrupt").exists()


def test_state_writes_use_unique_temp_names(home, monkeypatch):
    p = home / "x.json"
    seen = []
    real = os.replace
    monkeypatch.setattr(registry.os, "replace", lambda s, d: (seen.append(os.path.basename(s)), real(s, d))[1])
    registry.write_text_atomic(p, "{}")
    assert p.read_text(encoding="utf-8") == "{}"
    assert seen and str(os.getpid()) in seen[0] and seen[0] != "x.tmp"
    assert [f.name for f in home.iterdir() if f.name.endswith(".tmp")] == []


# ---------------------------------------------------------------------------------------------
# B5: corpus / model stamps
# ---------------------------------------------------------------------------------------------

def test_report_stamp_follows_the_corpus_and_mismatch_is_refused(home, caplog):
    names = ["Bear", "Forest", "Zap"]
    model = _model("TST", names, {"Bear"})
    model.meta.update(built_at="B", built_corpus_at="C1")
    model.formats = ["bo3"]
    cd = config.corpus_dir("TST")
    _write_corpus_cards("TST", names)
    (cd / "manifest.json").write_text(json.dumps({"built_at": "C1"}), encoding="utf-8")
    s1 = report.model_stamp(model)
    report.check_corpus_matches("TST", model)                    # same stamps: fine
    (cd / "manifest.json").write_text(json.dumps({"built_at": "C2"}), encoding="utf-8")
    assert report.model_stamp(model) != s1
    with caplog.at_level("WARNING"):
        report.check_corpus_matches("TST", model)                # newer corpus, same ids: only a warning
    assert "newer" in caplog.text
    _write_corpus_cards("TST", ["Ant"] + names)
    with pytest.raises(report.CorpusMismatch):
        report.check_corpus_matches("TST", model)
    with pytest.raises(report.CorpusMismatch):
        report.compute_report("TST", "bo3", model=model)


# ---------------------------------------------------------------------------------------------
# api: card-info cache keys, cross_set from stored reports
# ---------------------------------------------------------------------------------------------

def test_forget_card_caches_pops_the_path_keys(home):
    keys = [str(config.model_dir("TST") / "cards.parquet"), str(config.corpus_dir("TST") / "cards.parquet")]
    for k in keys:
        api._CARD_INFO_CACHE[k] = (1, {})
    api._forget_card_caches("tst")
    assert not any(k in api._CARD_INFO_CACHE for k in keys)


def test_cross_set_uses_stored_reports_and_lists_skipped_sets(home, monkeypatch):
    def put_model(code, cached):
        d = config.model_dir(code)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"set_code": code, "formats": ["bo3"], "built_at": "B"}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        if cached:
            report.write_cache(dict(set_code=code, fmt="bo3", stamp=report.model_stamp(meta)))

    put_model("AAA", True)
    put_model("BBB", False)                                      # model but no stored analysis
    put_model("CCC", True)                                       # busy
    from sealed_lab import report_facts

    monkeypatch.setattr(api, "set_report", lambda *a, **k: pytest.fail("a report was computed"))
    monkeypatch.setattr(report_facts, "facts_from_reports", lambda code, reps, b=None: {"facts": {"x": 1}, "noclear": {}, "set": code})
    monkeypatch.setattr(cross_set, "cross_from_facts", lambda per_set: {"facts": {"sets": sorted(per_set)}, "lessons": []})
    monkeypatch.setattr(api, "_calculating_elsewhere", lambda s: 1 if s == "CCC" else None)
    out = api.cross_set(["AAA", "BBB", "CCC", "DDD"])
    assert out["facts"]["sets"] == ["AAA"]
    assert {(s["set"], s["reason"]) for s in out["skipped"]} == {("BBB", "no stored bo3 analysis"), ("CCC", "being recalculated"),
                                                                  ("DDD", "no model")}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--jobs", "--boot"])
@pytest.mark.parametrize("bad", ["0", "-2"])
def test_cli_rejects_non_positive_counts(flag, bad, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["calc", "TST", flag, bad])
    assert e.value.code == 2
    assert "at least 1" in capsys.readouterr().err


def test_cli_unknown_set_and_missing_model_exit_nonzero(home, capsys):
    assert cli.main(["status", "XYZ"]) == 1
    assert "unknown set XYZ" in capsys.readouterr().err
    assert cli.main(["report", "XYZ"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "XYZ" in err and "Traceback" not in err
