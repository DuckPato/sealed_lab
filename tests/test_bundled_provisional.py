"""Ready-made sets that include a provisional set (model from 17Lands pick files only): bundle, install, status, deck build."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from sealed_lab import api, bundled, provisional as prov

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_sets  # noqa: E402

import test_bundled as tb  # noqa: E402
import test_provisional as tp  # noqa: E402

SET = tp.SET


@pytest.fixture(scope="module")
def owner(tmp_path_factory):
    """An owner home with one calculated provisional set (and its bundle)."""
    mp = pytest.MonkeyPatch()
    h = tmp_path_factory.mktemp("owner_prov")
    mp.setenv("SEALED_LAB_HOME", str(h))
    d = tmp_path_factory.mktemp("pickdata_b")
    from sealed_lab.synth.generate import _scryfall_objects, _write_cards_csv, build_world

    tp.write_pick_file(d / f"draft_data_public.{SET}.PremierDraft.csv.gz")
    w = build_world(SET, 0)
    (d / f"scryfall_{SET}.json").write_text(json.dumps(_scryfall_objects(w), ensure_ascii=False), encoding="utf-8")
    _write_cards_csv(d / "cards.csv", w)
    api.add_files([str(d)])
    prov.calculate(SET, n_jobs=1, n_boot=2, fetch_scryfall=False)
    tb.dates(h, **{SET: "2026-11-01"})
    out = tmp_path_factory.mktemp("bundle_prov") / "b"
    res = bundle_sets.build(out, None, 5, False)
    yield dict(home=h, bundle=out, res=res)
    mp.undo()


def test_problems_of_notes_not_refusals(owner):
    bad, notes = bundle_sets.problems_of(SET, False)          # even without --skip-written
    assert bad == [] and len(notes) == 1 and "provisional model" in notes[0]


def test_problems_of_still_refuses_stale(owner):
    reg = owner["home"] / "sets" / SET / "registry.json"
    saved = reg.read_text(encoding="utf-8")
    try:
        d = json.loads(saved)
        d["files"].append(dict(path="x.csv.gz", kind="game", set_code=SET, event="Sealed", fmt="bo1", size=1, fingerprint="zz", mtime=1.0))
        reg.write_text(json.dumps(d), encoding="utf-8")
        bad, _ = bundle_sets.problems_of(SET, True)
        assert any("stale" in b for b in bad)
    finally:
        reg.write_text(saved, encoding="utf-8")


def test_bundle_has_model_and_no_corpus_or_written(owner):
    b = owner["bundle"] / SET
    assert (b / "model" / "meta.json").is_file() and (b / "model" / "draws.npz").is_file()
    assert not (b / "corpus").exists() and not (b / "written").exists()
    assert owner["res"]["sets"][SET]["built_at"] and SET in owner["res"]["notes"]


def test_bundle_has_the_pick_decks_and_their_scores_and_no_paths(owner):
    m = owner["bundle"] / SET / "model"
    for fn in (prov.PICK_DECKS, prov.PICK_DECK_CARDS, prov.PICK_POOLS, prov.PICK_DECKS_META, "deck_scores.parquet", "deck_scores.json"):
        assert (m / fn).is_file(), fn
    assert bundle_sets.scan_folder(owner["bundle"]) == []                       # no personal string or path in the new files
    # the owner's own home got no score file from the bundle step beyond what it had (the bundle writes into its own folder)
    assert json.loads((m / prov.PICK_DECKS_META).read_text(encoding="utf-8"))["built_at"] == owner["res"]["sets"][SET]["built_at"]


def test_bundle_refuses_a_provisional_set_without_its_pick_decks(owner):
    d = owner["home"] / "sets" / SET / "model"
    saved = (d / prov.PICK_DECKS_META).read_bytes()
    try:
        (d / prov.PICK_DECKS_META).unlink()
        bad, _ = bundle_sets.problems_of(SET, False)
        assert any("pick_decks" in b for b in bad)
    finally:
        (d / prov.PICK_DECKS_META).write_bytes(saved)


def _app_with(bundle: Path, tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir(exist_ok=True)
    shutil.rmtree(app / "bundled_sets", ignore_errors=True)
    shutil.copytree(bundle, app / "bundled_sets")
    return app


def test_install_on_a_friend_home_and_build_a_deck(owner, tmp_path, monkeypatch):
    friend = tmp_path / "friend"
    monkeypatch.setenv("SEALED_LAB_HOME", str(friend))
    res = bundled.install_bundled(_app_with(owner["bundle"], tmp_path))
    assert res["installed"] == [SET] and not res["errors"]
    st = api.set_status(SET)
    assert st.has_model and st.provisional and not st.stale and not st.has_corpus and st.note.startswith("ready-made model")
    with pytest.raises(ValueError, match="Not available yet"):
        api.set_report(SET, "bo1")
    with pytest.raises(ValueError, match="Not available yet"):
        api.search_rebuilds(SET)
    # Deck search works from the shipped pick-deck tables and cached model scores (this home has no pick file)
    assert prov.pick_files(SET) == []
    res_s = api.search_decks(SET, min_games=1)
    assert res_s["total"] > 0 and res_s["results"][0]["drafted_pool"]
    assert api.has_deck_model_scores(SET)
    assert api.search_decks(SET, min_games=1, sort="model")["results"][0]["model_score"] is not None
    pool: dict[int, int] = {}
    for cid, c in api.card_info(SET).items():
        if not c["is_land"] and not c["is_basic"] and set(c["colors"]) <= set("WU") and len(pool) < 50:
            pool[cid] = 1
    r = api.build_from_pool(SET, pool, "bo1", n_decks=1)
    assert r.decks and sum(c.count for c in r.decks[0].cards) == 40
    # the user registers a game file: normal staleness rules apply
    reg = friend / "sets" / SET / "registry.json"
    reg.write_text(json.dumps(dict(version=1, files=[dict(path=str(tmp_path / "g.csv.gz"), kind="game", set_code=SET, event="Sealed",
                                                          fmt="bo1", size=1, fingerprint="mine", mtime=1.0)])), encoding="utf-8")
    assert api.set_status(SET).stale
    assert bundled._is_own(SET)


def test_a_later_normal_bundle_replaces_the_provisional_set(owner, tmp_path, monkeypatch):
    friend = tmp_path / "friend2"
    monkeypatch.setenv("SEALED_LAB_HOME", str(friend))
    bundled.install_bundled(_app_with(owner["bundle"], tmp_path))
    assert api.set_status(SET).provisional
    normal = tb.build_bundle(tmp_path, "nb", {SET: "2026-11-20T00:00:00+00:00"})
    monkeypatch.setenv("SEALED_LAB_HOME", str(friend))              # build_bundle restores the env it found
    assert bundled.install_bundled(_app_with(normal, tmp_path))["installed"] == [SET]
    st = api.set_status(SET)
    assert st.has_corpus and not st.provisional and st.has_model


def test_a_normal_set_without_corpus_is_still_rejected(tmp_path, monkeypatch):
    friend = tmp_path / "friend3"
    b = tb.build_bundle(tmp_path, "nb2", {"AAA": "2026-11-01T00:00:00+00:00"})
    monkeypatch.setenv("SEALED_LAB_HOME", str(friend))
    shutil.rmtree(b / "AAA" / "corpus")
    res = bundled.install_bundled(_app_with(b, tmp_path))
    assert res["installed"] == [] and res["errors"] and not (friend / "sets" / "AAA" / "model").exists()
