"""Extra draft events (PickTwo / QuickDraft): the game filtering, the keep-or-drop decision, the report section and one small
end-to-end run on synthetic data (no process pool: n_jobs=1)."""
from __future__ import annotations

import json

import polars as pl
import pytest

from sealed_lab import api, config
from sealed_lab.model import pipeline, validate


# ------------------------------------------------------------------------------------------------ which events are "extra"

def test_extra_draft_events_are_the_draft_events_outside_premier_and_trad():
    assert not config.is_extra_draft_event("PremierDraft") and not config.is_extra_draft_event("TradDraft")
    assert not config.is_extra_draft_event("Sealed") and not config.is_extra_draft_event("TradSealed")
    assert all(config.is_extra_draft_event(e) for e in ("PickTwoDraft", "PickTwoTradDraft", "QuickDraft"))
    assert not config.is_extra_draft_event(None) and not config.is_extra_draft_event("")
    assert set(config.EXTRA_DRAFT_EVENTS) | set(config.CORE_DRAFT_EVENTS) == config.FORMAT_EVENTS["draft"]


def _inputs(events: list[tuple[str, str, int]]) -> pipeline.Inputs:
    """Inputs from [(event, fmt, deck_key)] game rows (one game each); every deck has one card."""
    games = pl.DataFrame(dict(event=[e for e, _f, _d in events], fmt=[f for _e, f, _d in events], deck_key=[d for _e, _f, d in events]))
    keys = sorted(set(d for _e, _f, d in events) | {99})                    # deck 99 is played in no game at all
    decks = pl.DataFrame(dict(deck_key=keys))
    deck_cards = pl.DataFrame(dict(deck_key=keys, card_id=[0] * len(keys), n=[1] * len(keys)))
    return pipeline.Inputs(set_code="TST", cards=pl.DataFrame(), games=games, decks=decks, deck_cards=deck_cards, builds=pl.DataFrame(),
                           manifest={}, lazy={})


ROWS = [("PremierDraft", "draft", 1), ("TradDraft", "draft", 2), ("PickTwoDraft", "draft", 3), ("PickTwoDraft", "draft", 1),
        ("QuickDraft", "draft", 4), ("Sealed", "bo1", 5), ("TradSealed", "bo3", 6)]


def test_extra_draft_events_counts_only_the_extra_draft_games():
    assert pipeline.extra_draft_events(_inputs(ROWS).games) == {"PickTwoDraft": 2, "QuickDraft": 1}
    assert pipeline.extra_draft_events(_inputs(ROWS[:2] + ROWS[5:]).games) == {}                       # a set without any


def test_without_extra_draft_keeps_core_and_sealed_games_and_drops_decks_nobody_plays_any_more():
    inp = pipeline.without_extra_draft(_inputs(ROWS))
    assert sorted(inp.games["event"].to_list()) == ["PremierDraft", "Sealed", "TradDraft", "TradSealed"]
    # decks 3 and 4 only existed in extra events; 1 is shared with PremierDraft (stays); 99 is played in no game (never in a game: goes)
    assert inp.decks["deck_key"].to_list() == [1, 2, 5, 6]
    assert sorted(inp.deck_cards["deck_key"].unique().to_list()) == [1, 2, 5, 6]


def test_without_extra_draft_when_the_set_has_no_standard_draft_games_leaves_no_draft_stage():
    om1_like = [("PickTwoDraft", "draft", 3), ("PickTwoTradDraft", "draft", 4), ("Sealed", "bo1", 5), ("TradSealed", "bo3", 6)]
    inp = _inputs(om1_like)
    assert pipeline.extra_draft_events(inp.games) == {"PickTwoDraft": 1, "PickTwoTradDraft": 1}
    out = pipeline.without_extra_draft(inp)
    assert out.games.filter(pl.col("fmt") == "draft").height == 0                            # bo1 from scratch, as for any set without drafts
    assert pipeline.extra_draft_events(out.games) == {}


# ------------------------------------------------------------------------------------------------ the decision

def _half(gain, lo, hi, lo90):
    return dict(gain=gain, lo=lo, hi=hi, lo90=lo90, hi90=hi, n_events=10, n_games=100)


def _entry(gain, lo, hi, a=None, b=None):
    return dict(gain=gain, lo=lo, hi=hi, logloss_with=0.6, logloss_without=0.6 + gain, n_holdout_events=40, n_holdout_games=300,
                halves=dict(A=a or _half(gain, lo, hi, lo), B=b or _half(gain, lo, hi, lo)))


EVENTS = {"PickTwoDraft": 1500, "QuickDraft": 500}


def test_decision_not_available_without_extra_events():
    d = pipeline.decide_extra_draft("auto", {}, 1000, None)
    assert d["available"] is False and d["applied"] is False and d["games"] == 0 and d["gain"] == {} and "no extra draft" in d["reason"]
    assert pipeline.decide_extra_draft("on", {}, 1000, None)["applied"] is False            # nothing to keep


def test_decision_on_keeps_untested_and_off_drops():
    on = pipeline.decide_extra_draft("on", EVENTS, 1000, None)
    assert on["applied"] is True and on["available"] and on["games"] == 2000 and on["gain"] == {} and "forced on" in on["reason"]
    off = pipeline.decide_extra_draft("off", EVENTS, 1000, {"bo1": _entry(0.01, 0.005, 0.015)})
    assert off["applied"] is False and "switched off" in off["reason"]


def test_decision_auto_keeps_events_only_with_a_confirmed_material_gain():
    ok = pipeline.decide_extra_draft("auto", EVENTS, 1000, {"bo3": _entry(0.002, 0.001, 0.003), "bo1": _entry(0.0, -0.001, 0.001)})
    assert ok["applied"] is True and ok["formats"] == ["bo3", "bo1"] and "bo3" in ok["reason"] and ok["confirmation"]["confirm"]
    noise = pipeline.decide_extra_draft("auto", EVENTS, 1000, {"bo3": _entry(0.001, -0.001, 0.003)})                  # CI includes zero
    assert noise["applied"] is False and "cannot distinguish" in noise["reason"]
    tiny = pipeline.decide_extra_draft("auto", EVENTS, 1000, {"bo3": _entry(5e-5, 2e-5, 8e-5)})                       # below MIN_GAIN
    assert tiny["applied"] is False and "materiality" in tiny["reason"]
    worse = pipeline.decide_extra_draft("auto", EVENTS, 1000, {"bo3": _entry(0.002, 0.001, 0.003), "bo1": _entry(-0.002, -0.003, -0.001)})
    assert worse["applied"] is False and "WORSE" in worse["reason"] and "bo1" in worse["reason"]


def test_decision_auto_needs_the_later_half_to_confirm():
    shaky = _entry(0.002, 0.001, 0.003, b=_half(0.0004, -0.002, 0.003, -0.001))                  # later half: 90% lower bound below zero
    d = pipeline.decide_extra_draft("auto", EVENTS, 1000, {"bo3": shaky})
    assert d["applied"] is False and "NOT confirmed" in d["reason"]
    assert pipeline.decide_extra_draft("auto", EVENTS, 1000, {"bo3": shaky}, confirm=False)["applied"] is True


def test_decision_auto_with_no_testable_format_does_not_keep_the_events():
    d = pipeline.decide_extra_draft("auto", EVENTS, 0, {})
    assert d["applied"] is False and "enough data" in d["reason"] and "no draft stage at all" in d["reason"]       # OM1-like: no core draft


def test_meta_summary_and_report_section():
    d = pipeline.decide_extra_draft("auto", EVENTS, 1000, {"bo3": _entry(0.002, 0.001, 0.003)})
    m = pipeline._extra_draft_meta(d)
    assert m["mode"] == "auto" and m["applied"] is True and m["available"] is True and m["events"] == EVENTS and m["games"] == 2000
    assert set(m["gain"]["bo3"]) == {"gain", "lo", "hi"} and set(m["halves"]["bo3"]) == {"A", "B"} and "lo90" in m["halves"]["bo3"]["B"]
    json.dumps(m)
    html = validate.render_extra_draft(d)
    assert "Extra draft events" in html and "PickTwoDraft" in html and "APPLIED" in html and "Bo3" in html
    off = validate.render_extra_draft(pipeline.decide_extra_draft("off", EVENTS, 1000, None))
    assert "NOT APPLIED" in off and "switched off" in off
    assert validate.render_extra_draft(pipeline.decide_extra_draft("auto", {}, 1000, None)) == ""        # nothing to say without such data
    assert pipeline._extra_draft_meta(pipeline.decide_extra_draft("auto", {}, 1000, None))["available"] is False


def test_api_labels_and_cli_flag():
    from sealed_lab import cli

    assert api.COMPONENT_LABELS["extra_draft"] == "Extra draft events (PickTwo / QuickDraft)"
    with pytest.raises(ValueError):
        pipeline.calculate("TST", extra_draft="sometimes")
    with pytest.raises(SystemExit):
        cli.main(["calc", "TST", "--extra-draft", "maybe"])


# ------------------------------------------------------------------------------------------------ end to end (synthetic, no pool)

def test_pipeline_end_to_end_on_a_set_with_an_extra_draft_event(tmp_path, monkeypatch):
    from sealed_lab.ingest.corpus import build_corpus
    from sealed_lab.synth.generate import generate

    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    syn = tmp_path / "syn"
    generate(syn, set_code="TST", seed=1, scale=0.05, events=("PremierDraft", "PickTwoDraft", "Sealed"))
    api.add_files([str(syn)])
    build_corpus("TST", n_jobs=1, fetch_scryfall=False)
    kw = dict(n_jobs=1, n_boot=2, n_rep=40, recency="off", attributes="off", crossset="off", synergy="off", archcond="off",
              matchups="off", manasim="v1", castability="off")
    meta_on = pipeline.calculate("TST", extra_draft="on", **kw)
    n_all = meta_on["n_games"]["draft"]
    meta_off = pipeline.calculate("TST", extra_draft="off", **kw)
    n_core = meta_off["n_games"]["draft"]
    assert n_all > n_core > 0 and meta_on["extra_draft"]["games"] == n_all - n_core == meta_off["extra_draft"]["games"]
    assert meta_on["extra_draft"]["applied"] is True and meta_on["extra_draft"]["gain"] == {}
    assert meta_off["extra_draft"]["applied"] is False and meta_off["extra_draft"]["events"] == {"PickTwoDraft": n_all - n_core}
    assert meta_off["formats"] == ["bo1"]
    meta = pipeline.calculate("TST", extra_draft="auto", **kw)                                   # the test, on the plain chain
    xd = meta["extra_draft"]
    assert xd["mode"] == "auto" and xd["available"] is True and xd["events"] == {"PickTwoDraft": n_all - n_core}
    assert set(xd["gain"]) == set(xd["halves"]) == {"bo1"} and set(xd["gain"]["bo1"]) == {"gain", "lo", "hi"}
    assert meta["n_games"]["draft"] == (n_all if xd["applied"] else n_core)                      # the chosen draft stage is what was fitted
    report = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))
    assert report["extra_draft"]["applied"] == xd["applied"]
    assert "Extra draft events" in (config.model_dir("TST") / "report.html").read_text(encoding="utf-8")
    ov = api.set_overview("TST")["model"]["components"]
    assert "extra_draft" in {c["key"] for c in ov}
    # a set without such games: not tested, meta says so, the overview hides the component
    syn2 = tmp_path / "syn2"
    generate(syn2, set_code="TS2", seed=2, scale=0.05, events=("PremierDraft", "Sealed"))
    api.add_files([str(syn2)])
    build_corpus("TS2", n_jobs=1, fetch_scryfall=False)
    m2 = pipeline.calculate("TS2", extra_draft="auto", **kw)
    assert m2["extra_draft"]["available"] is False and m2["extra_draft"]["applied"] is False and m2["extra_draft"]["gain"] == {}
    assert "extra_draft" not in {c["key"] for c in api.set_overview("TS2")["model"]["components"]}
