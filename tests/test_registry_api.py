from __future__ import annotations

import importlib.util
import os
import shutil
import time

import pytest

from sealed_lab import api
from sealed_lab.config import FORMAT_EVENTS, event_format, set_dir
from sealed_lab.ingest import registry
from sealed_lab.ingest.detect import detect_file


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _touch(p, t):
    os.utime(p, (t, t))


def test_config_event_format():
    assert event_format("PremierDraft") == "draft" and event_format("TradDraft") == "draft"
    assert event_format("Sealed") == "bo1" and event_format("TradSealed") == "bo3"
    assert event_format("WeirdTradSealed") == "bo3" and event_format("FooSealed") == "bo1"
    assert event_format("Whatever") == "draft" and event_format(None) is None
    assert FORMAT_EVENTS["bo3"] == {"TradSealed"}


def test_app_home_env(home):
    from sealed_lab import config

    assert config.APP_HOME == home
    assert config.set_dir("tst") == home / "sets" / "TST"


def test_registry_replacement(synth_dir, home, tmp_path):
    a = tmp_path / "a" / "game_data_public.TST.TradSealed.csv.gz"
    b = tmp_path / "b" / "game_data_public.TST.TradSealed.csv.gz"
    a.parent.mkdir()
    b.parent.mkdir()
    shutil.copy(synth_dir / "game_data_public.TST.TradSealed.csv.gz", a)
    shutil.copy(synth_dir / "game_data_public.TST.Sealed.csv.gz", b)  # different content, forced to same event below
    now = time.time()
    _touch(a, now - 1000)
    _touch(b, now)
    ia = detect_file(a)
    ib = detect_file(b)
    ib.event = "TradSealed"
    ib.fmt = "bo3"
    r = registry.register(ia)
    assert r.registered and not r.replaced
    # same file again -> no-op
    assert not registry.register(detect_file(a)).registered
    # newer file with same (set, kind, event) replaces
    r = registry.register(ib)
    assert r.registered and [x.path for x in r.replaced] == [str(a)]
    files = [f for f in registry.files("TST") if f.kind == "game"]
    assert [f.path for f in files] == [str(b)]
    # an older file does not replace a newer one
    r = registry.register(ia)
    assert not r.registered and "older" in r.note
    assert [f.path for f in registry.files("TST") if f.kind == "game"] == [str(b)]
    assert registry.known_sets() == ["TST"]
    assert registry.remove("TST", ib.fingerprint)
    assert not [f for f in registry.files("TST") if f.kind == "game"]


def test_registry_different_events_coexist(synth_dir, home):
    api.add_files([str(synth_dir / "game_data_public.TST.Sealed.csv.gz"),
                   str(synth_dir / "game_data_public.TST.TradSealed.csv.gz"),
                   str(synth_dir / "draft_data_public.TST.PremierDraft.csv.gz")])
    kinds = sorted((f.kind, f.event) for f in registry.files("TST"))
    assert kinds == [("draft", "PremierDraft"), ("game", "Sealed"), ("game", "TradSealed")]


def test_shared_cards_file(synth_dir, home, tmp_path):
    # a cards.csv with several expansions has no set -> shared across sets
    import polars as pl

    df = pl.read_csv(synth_dir / "cards.csv")
    other = df.head(5).with_columns(pl.lit("OTH").alias("expansion"))
    p = tmp_path / "cards.csv"
    pl.concat([df, other]).write_csv(p)
    info = api.add_files([str(p)])[0]
    assert info.kind == "cards" and info.set_code is None
    api.add_files([str(synth_dir / "game_data_public.TST.Sealed.csv.gz")])
    assert any(f.kind == "cards" for f in registry.files("TST"))
    assert any(f.kind == "cards" for f in api.set_status("TST").files)


def test_api_add_status_remove(synth_dir, home):
    infos = api.add_files([str(synth_dir)])
    kinds = {i.kind for i in infos}
    assert {"game", "draft", "cards", "scryfall", "unknown"} <= kinds
    sets = api.list_sets()
    assert [s.set_code for s in sets] == ["TST"]
    st = sets[0]
    assert not st.has_corpus and not st.has_model and st.stale
    assert len([f for f in st.files if f.kind == "game"]) == 4
    fp = next(f.fingerprint for f in st.files if f.event == "Sealed" and f.kind == "game")
    api.remove_file("TST", fp)
    assert len([f for f in api.set_status("TST").files if f.kind == "game"]) == 3
    assert (set_dir("TST") / "registry.json").exists()


@pytest.mark.skipif(_has("sealed_lab.model.pipeline"), reason="model implemented")
def test_calculate_without_model_returns_warning(synth_dir, home):
    api.add_files([str(synth_dir / "game_data_public.TST.TradSealed.csv.gz")])
    seen = []
    st = api.calculate_set("TST", progress=lambda f, m: seen.append(f), n_jobs=1, fetch_scryfall=False)
    assert st.has_corpus and not st.has_model
    assert any("model not implemented yet" in w for w in st.warnings)
    assert st.n_games == {"bo3": st.n_games["bo3"]}
    assert seen == sorted(seen) and seen[-1] == 1.0


@pytest.mark.skipif(_has("sealed_lab.deck.builder") or _has("sealed_lab.deck.score"), reason="deck tools implemented")
def test_deck_tools_not_implemented(home):
    with pytest.raises(NotImplementedError):
        api.build_pool("TST", "1 Plains")
    with pytest.raises(NotImplementedError):
        api.score_deck("TST", "1 Plains")
    assert api.report_path("TST") is None
