"""Draft mode, app side (part 1: no GUI): the per-set option "Also build draft models", its stale hint, its way into the calculation,
the Sealed-only guards of the api, and the Arena-log import of draft courses."""
from __future__ import annotations

import json

import pytest

from sealed_lab import api, config
from sealed_lab.deck import arenalog
from tests.test_arenalog import HEADER, IDS, course, one_line, seed_ids


def _model(code: str, formats=("bo3", "bo1"), built_at="2026-10-20T10:00:00") -> None:
    d = config.model_dir(code)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(dict(set_code=code, formats=list(formats), built_at=built_at)), encoding="utf-8")


def test_option_roundtrip_and_defaults(home):
    assert api.get_option("MSH", "draft_formats") is False                      # default: off
    api.set_option("msh", "draft_formats", True)
    assert api.get_option("MSH", "draft_formats") is True
    p = config.set_dir("MSH") / "settings.json"
    assert json.loads(p.read_text(encoding="utf-8")) == {"draft_formats": True}
    api.set_option("MSH", "other", 3)                                           # other keys are kept
    api.set_option("MSH", "draft_formats", False)
    assert api.get_option("MSH", "draft_formats") is False and api.get_option("MSH", "other") == 3
    assert api.get_option("NEO", "draft_formats") is False                      # per set


def test_switch_on_without_draft_models_marks_the_set_pending(home):
    _model("MSH")
    assert not api.set_status("MSH").stale
    api.set_option("MSH", "draft_formats", True)
    st = api.set_status("MSH")
    assert st.stale and any("draft models" in p for p in st.pending)
    api.set_option("MSH", "draft_formats", False)
    assert not api.set_status("MSH").stale                                      # off again: nothing pending
    _model("MSH", formats=("bo3", "bo1", "dbo3", "dbo1"))
    api.set_option("MSH", "draft_formats", True)
    assert not api.set_status("MSH").stale                                      # the model has them


def test_a_provisional_set_with_the_draft_switch_on_is_not_stale(home):
    d = config.model_dir("PRV")
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(dict(set_code="PRV", formats=["bo3", "bo1"], built_at="2026-10-20T10:00:00",
                                                  provisional=dict(n_runs=1, n_games=1))), encoding="utf-8")
    api.set_option("PRV", "draft_formats", True)
    st = api.set_status("PRV")
    assert st.provisional and st.has_model and not any("draft models" in p for p in st.pending)
    assert api.get_option("PRV", "draft_formats") is True                       # the stored value is left alone


def test_calculate_set_passes_the_switch_and_a_set_without_draft_games_is_not_pending_forever(home, monkeypatch):
    import sealed_lab.ingest.corpus as corpus
    import sealed_lab.model.pipeline as pipeline

    seen: list[dict] = []
    monkeypatch.setattr(corpus, "build_corpus", lambda *a, **k: None)
    monkeypatch.setattr(api, "_precompute_analysis", lambda sc: None)

    def fake_calculate(set_code, **kw):
        seen.append(kw)
        _model(set_code, built_at="2026-10-21T10:00:00")                          # (no draft games: only the Sealed formats)
    monkeypatch.setattr(pipeline, "calculate", fake_calculate)
    api.set_option("MSH", "draft_formats", True)
    st = api.calculate_set("MSH", draft_formats=True)
    assert seen[-1]["draft_formats"] is True and seen[-1]["draft_boot"] == 16
    assert not st.stale                                                         # it asked, and there was nothing to fit
    api.calculate_set("MSH")
    assert seen[-1]["draft_formats"] is False


def test_the_report_and_archetype_line_are_sealed_only(home):
    with pytest.raises(ValueError, match="Sealed-only"):
        api.set_report("MSH", "dbo1")
    r = api.deck_archetype("MSH", "dbo3", {1: 4})
    assert r["available"] is False and "Sealed-only" in r["reason"]


def test_draft_formats_do_not_reach_the_export_or_report_facts(home):
    _model("MSH", formats=("bo3", "bo1", "dbo3", "dbo1"))
    from sealed_lab import report_facts

    assert report_facts.FORMATS == ("bo1", "bo3")
    from sealed_lab import report

    assert report.FORMATS == ("bo1", "bo3")


# --------------------------------------------------------------------------- Arena log: draft courses

def _write(logdir, *parts):
    (logdir / "Player.log").write_text(HEADER + "".join(parts), encoding="utf-8")


@pytest.fixture()
def logdir(tmp_path, home):
    d = tmp_path / "MTGA"
    d.mkdir()
    return d


def test_draft_event_helpers():
    assert arenalog.draft_format_of("PremierDraft_MSH_20261001") == "dbo1"
    assert arenalog.draft_format_of("QuickDraft_MSH_20261001") == "dbo1"
    assert arenalog.draft_format_of("TradDraft_MSH_20261001") == "dbo3"
    assert arenalog.draft_format_of("Sealed_MSH_20261001") == ""
    assert arenalog.event_set("PremierDraft_MSH_20261001", known=set()) == "MSH" and arenalog.event_set("TradDraft_FIN_1", known=set()) == "FIN"


def test_draft_course_is_imported_only_on_request_and_the_sealed_behaviour_is_unchanged(logdir):
    seed_ids()
    _write(logdir,
           one_line("24/10/2026 9:00:00", "<== EventJoin(1)", {"Course": course("Sealed_MSH_20261001", [101, 102])}),
           one_line("25/10/2026 9:00:00", "<== EventJoin(2)", {"Course": course("PremierDraft_MSH_20261002", [101, 101, 102, 103])}))
    r = arenalog.import_pool("MSH", log_dir=logdir, offline=True)                # default: Sealed only, as before
    assert r.event == "Sealed_MSH_20261001" and r.kind == "sealed" and r.draft_fmt == ""
    r = arenalog.import_pool("MSH", log_dir=logdir, offline=True, include_draft=True)
    assert r.event == "PremierDraft_MSH_20261002" and r.kind == "draft" and r.draft_fmt == "dbo1" and r.n_cards == 4
    assert any("not be finished" in w for w in r.warnings)                       # fewer than 40 cards
    assert r.text.startswith("Deck\n") and r.set_code == "MSH"


def test_a_more_recent_sealed_pool_wins_over_an_older_draft(logdir):
    seed_ids()
    _write(logdir,
           one_line("24/10/2026 9:00:00", "<== EventJoin(1)", {"Course": course("TradDraft_MSH_20261002", [101, 102])}),
           one_line("25/10/2026 9:00:00", "<== EventJoin(2)", {"Course": course("Sealed_MSH_20261001", [101, 102, 103])}))
    r = arenalog.import_pool("MSH", log_dir=logdir, offline=True, include_draft=True)
    assert r.event == "Sealed_MSH_20261001" and r.kind == "sealed"


def test_trad_draft_course_selects_the_bo3_draft_format(logdir):
    seed_ids()
    _write(logdir, one_line("25/10/2026 9:00:00", "<== EventJoin(2)", {"Course": course("TradDraft_MSH_20261002", [101, 102])}))
    r = arenalog.import_pool(None, log_dir=logdir, offline=True, include_draft=True)
    assert r.kind == "draft" and r.draft_fmt == "dbo3" and r.set_code == "MSH"
    with pytest.raises(arenalog.NoSealedPool, match="No sealed pool found"):      # without the flag a draft is no pool, as before
        arenalog.import_pool(None, log_dir=logdir, offline=True)


def test_nothing_found_message_names_draft_when_asked(logdir):
    _write(logdir, one_line("25/10/2026 9:00:00", "<== X", {"Course": course("Constructed_BestOf3", [101])}))
    with pytest.raises(arenalog.NoSealedPool, match="sealed or draft pool"):
        arenalog.import_pool(None, log_dir=logdir, offline=True, include_draft=True)


def test_api_wrapper_passes_include_draft(logdir, home):
    seed_ids()
    _write(logdir, one_line("25/10/2026 9:00:00", "<==", course("PremierDraft_MSH_1", [101, 102])))
    r = api.import_arena_log("msh", log_dir=str(logdir), include_draft=True)
    assert r.kind == "draft" and r.draft_fmt == "dbo1"
    with pytest.raises(arenalog.NoSealedPool):
        api.import_arena_log("msh", log_dir=str(logdir))
