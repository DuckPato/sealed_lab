"""Backend pieces behind the reworked GUI: guessed card names, the tracked data folder, set status reasons, the statistics of a set,
per-set refusal while a set is recalculated, deck + sideboard exports (the user's own deck) and the fast score."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import threading
import time
from pathlib import Path

import polars as pl
import pytest

from sealed_lab import api, config
from sealed_lab.deck import arena, builder
from sealed_lab.deck.names import NameIndex, analyse_pool, apply_overrides
from sealed_lab.ingest import folders, registry
from sealed_lab.types import SetBusy

BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")


# ------------------------------------------------------------------------------------------ guessed card names

def _index() -> NameIndex:
    names = ["Lightning Strike", "Giant Growth", "Fire // Ice", "Llanowar Elves", "Jötun's Rage, the Unbowed", "Forest", "Shock"]
    return NameIndex(pl.DataFrame({"card_id": list(range(len(names))), "name": names, "is_basic": [n == "Forest" for n in names]}))


def test_candidates_rank_by_similarity_and_skip_basics():
    idx = _index()
    c = idx.candidates("Lightning Strke")
    assert c[0][1] == "Lightning Strike" and c[0][2] > 0.9
    assert all(n != "Forest" for _i, n, _s in idx.candidates("Forrest", 5))
    assert [n for _i, n, _s in idx.candidates("Gaint Growht", 3)][0] == "Giant Growth"
    half = idx.candidates("Lightning Str")                        # a half-typed name still finds its card
    assert half and half[0][1] == "Lightning Strike"
    assert idx.candidates("") == [] and len(idx.candidates("Fire", 2)) <= 2
    scores = [s for _i, _n, s in idx.candidates("Lightning", 5)]
    assert scores == sorted(scores, reverse=True)


def test_analyse_pool_lists_guesses_unmatched_and_the_own_deck_flag():
    idx = _index()
    text = ("Deck\n2 Lightning Strke\n1 Giant Growht\n1 Zzzzzz Qqqq\n1 Llanowar Elves\n4 Forest\n\n"
            "Sideboard\n1 Lightning Strke\n1 Fire\n1 Jotuns Rage, the Unbowed\n")
    r = analyse_pool(text, idx)
    g = {x["raw"]: x for x in r["guesses"]}
    assert set(g) == {"Lightning Strke", "Giant Growht"}
    assert g["Lightning Strke"]["name"] == "Lightning Strike" and g["Lightning Strke"]["how"] == "fuzzy"
    assert g["Lightning Strke"]["count"] == 3 and g["Lightning Strke"]["section"] == "deck"   # both sections, copies summed
    assert g["Giant Growht"]["how"] == "closest" and g["Giant Growht"]["card_id"] == 1
    assert g["Giant Growht"]["candidates"][0]["name"] == "Giant Growth" and 1 <= len(g["Giant Growht"]["candidates"]) <= 5
    assert [u["raw"] for u in r["unmatched"]] == ["Zzzzzz Qqqq"] and r["unmatched"][0]["candidates"] == []
    assert r["has_deck"] is True and r["n_main"] == 9 and r["n_side"] == 3
    assert r["matched"] == 3 and r["n_copies"] == 8                                        # Llanowar + Fire + the accented card; Forest does not count
    assert analyse_pool("Deck\n1 Shock\n", idx)["has_deck"] is False                      # no Sideboard section: a plain pool
    assert analyse_pool("1 Shock\n\n1 Giant Growth\n", idx)["has_deck"] is False             # blank-line split without headers is not an export


def test_apply_overrides_replaces_sums_and_removes():
    named = {"Lightning Strke": 2, "Lightning Strike": 1, "Giant Growht": 1, "Shock": 1}
    out = apply_overrides(named, {"Lightning Strke": "Lightning Strike", "Giant Growht": ""})
    assert out == {"Lightning Strike": 3, "Shock": 1}
    assert apply_overrides(named, None) == named and apply_overrides(named, {}) == named


# ------------------------------------------------------------------------------------------ tracked data folder

def _gz(path: Path, header: str, row: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write(header + "\n" + row + "\n")
    return path


def game(folder: Path, set_code: str, event: str, extra: str = "") -> Path:
    return _gz(folder / f"game_data_public.{set_code}.{event}.csv.gz", "expansion,event_type,draft_id,won,deck_Plains",
               f"{set_code},{event},d{extra},True,8")


def replay(folder: Path, set_code: str, event: str) -> Path:
    return _gz(folder / f"replay_data_public.{set_code}.{event}.csv.gz", "expansion,event_type,history_id,won,deck_Plains,user_turn_1_cards",
               f"{set_code},{event},h1,True,8,x")


def test_scan_registers_only_tracked_sets_and_keeps_replays_replays(home, tmp_path):
    root = tmp_path / "root"
    seed = game(tmp_path / "seed", "TST", "Sealed")
    api.add_files([str(seed)])                                       # TST is known now (it has a registry)
    game(root / "TST", "TST", "TradSealed")
    replay(root / "TST", "TST", "TradSealed")
    game(root / "OTH", "OTH", "Sealed")                              # a set the app does not track
    rep = api.scan_data_folders(str(root))
    names = sorted(Path(f.path).name for f in rep.added)
    assert names == ["game_data_public.TST.TradSealed.csv.gz", "replay_data_public.TST.TradSealed.csv.gz"]
    assert rep.changed_sets == ["TST"] and "OTH" not in registry.known_sets()
    kinds = {(f.kind, f.event) for f in registry.files("TST")}
    assert ("replay", "TradSealed") in kinds and ("game", "TradSealed") in kinds and ("game", "Sealed") in kinds
    # unchanged files are skipped the second time; a changed file replaces the old version
    rep2 = api.scan_data_folders(str(root))
    assert rep2.added == [] and rep2.skipped >= 2 and rep2.scanned == 0
    time.sleep(0.02)
    game(root / "TST", "TST", "TradSealed", extra="2")
    rep3 = api.scan_data_folders(str(root))
    assert [Path(f.path).name for f in rep3.added] == ["game_data_public.TST.TradSealed.csv.gz"]
    assert "replaces older file" in rep3.added[0].note
    assert len([f for f in registry.files("TST") if f.kind == "game" and f.event == "TradSealed"]) == 1


def draft_picks(folder: Path, set_code: str, event: str) -> Path:
    return _gz(folder / f"draft_data_public.{set_code}.{event}.csv.gz", "expansion,event_type,draft_id,pick,pack_card_Plains",
               f"{set_code},{event},d1,Plains,1")


def test_scan_registers_game_files_of_extra_draft_events_but_not_their_other_files(home, tmp_path):
    root = tmp_path / "root"
    api.add_files([str(game(tmp_path / "seed", "TST", "Sealed"))])
    game(root / "TST", "TST", "PickTwoDraft")                       # extra draft events: the model decides whether to keep their games
    game(root / "TST", "TST", "PickTwoTradDraft")
    game(root / "TST", "TST", "QuickDraft")
    game(root / "TST", "TST", "TradDraft")
    game(root / "TST", "TST", "CubeDraft")                          # an event nothing knows about: only a dropped file gets in
    replay(root / "TST", "TST", "PickTwoDraft")                     # replay / draft-pick files feed statistics that only know the core events
    draft_picks(root / "TST", "TST", "PickTwoDraft")
    draft_picks(root / "TST", "TST", "PremierDraft")
    rep = api.scan_data_folders(str(root))
    assert sorted(Path(f.path).name for f in rep.added) == [
        "draft_data_public.TST.PremierDraft.csv.gz", "game_data_public.TST.PickTwoDraft.csv.gz",
        "game_data_public.TST.PickTwoTradDraft.csv.gz", "game_data_public.TST.QuickDraft.csv.gz", "game_data_public.TST.TradDraft.csv.gz"]
    kinds = {(f.kind, f.event) for f in registry.files("TST")}
    assert {("game", "PickTwoDraft"), ("game", "PickTwoTradDraft"), ("game", "QuickDraft")} <= kinds
    assert not {("replay", "PickTwoDraft"), ("draft", "PickTwoDraft"), ("game", "CubeDraft")} & kinds
    assert api.folder_set_codes(str(root / "TST")) == {"TST": 8}     # a dropped folder still counts every event ...
    api.add_files([str(root / "TST")])                              # ... and dropping registers them all
    assert {f.event for f in registry.files("TST")} >= {"PickTwoDraft", "QuickDraft", "TradDraft", "Sealed", "CubeDraft"}
    assert ("replay", "PickTwoDraft") in {(f.kind, f.event) for f in registry.files("TST")}


def test_scan_reports_missing_files_but_keeps_them_registered(home, tmp_path):
    root = tmp_path / "root"
    f = game(root / "TST", "TST", "TradSealed")
    api.add_files([str(f)])
    assert api.set_status("TST").missing == []
    f.unlink()
    rep = api.scan_data_folders(str(root))
    assert rep.missing == {"TST": [str(f)]}
    st = api.set_status("TST")
    assert st.missing == [str(f)] and len(st.files) == 1                           # still registered, flagged
    assert any("source file not found" in w for w in st.warnings)
    assert [x["missing"] for x in api.set_overview("TST")["files"]] == [True]


def test_scan_defers_sets_being_calculated_and_takes_track_requests(home, tmp_path):
    root = tmp_path / "root"
    api.add_files([str(game(tmp_path / "seed", "TST", "Sealed"))])
    game(root / "TST", "TST", "TradSealed")
    game(root / "NEW", "NEW", "Sealed")
    rep = api.scan_data_folders(str(root), skip_sets=["TST"])
    assert rep.deferred == ["TST"] and rep.added == []
    rep = api.scan_data_folders(str(root), track=["NEW"])                          # the user asked to track NEW
    assert sorted(rep.changed_sets) == ["NEW", "TST"] and "NEW" in registry.known_sets()


def test_a_set_can_have_its_own_folder(home, tmp_path):
    root, other = tmp_path / "root", tmp_path / "somewhere" / "else"
    api.add_files([str(game(tmp_path / "seed", "TST", "Sealed"))])
    game(other, "TST", "TradSealed")
    assert api.scan_data_folders(str(root)).added == []
    api.set_set_folder("TST", str(other))
    assert api.set_folders() == {"TST": str(other)} and api.set_status("TST").folder == str(other)
    rep = api.scan_data_folders(str(root))
    assert [Path(f.path).parent for f in rep.added] == [other]
    api.set_set_folder("TST", None)
    assert api.set_folders() == {}
    assert api.folder_set_codes(str(other)) == {"TST": 1}


def test_untracked_sets_from_the_folder_and_the_listing(home, tmp_path):
    root = tmp_path / "root"
    api.add_files([str(game(tmp_path / "seed", "TST", "Sealed"))])
    game(root / "ABC", "ABC", "TradSealed")
    replay(root / "ABC", "ABC", "TradSealed")
    (root / "EMPTY").mkdir()
    (root / "TST").mkdir()
    state = config.app_home() / "datasets_state.json"
    state.write_text(json.dumps({"remote": [
        dict(url="u1", name="game_data_public.XYZ.Sealed.csv.gz", kind="game", set_code="XYZ", event="Sealed", row_date=""),
        dict(url="u2", name="game_data_public.XYZ.TradSealed.csv.gz", kind="game", set_code="XYZ", event="TradSealed", row_date=""),
        dict(url="u3", name="game_data_public.ABC.TradSealed.csv.gz", kind="game", set_code="ABC", event="TradSealed", row_date=""),
        dict(url="u4", name="game_data_public.TST.Sealed.csv.gz", kind="game", set_code="TST", event="Sealed", row_date="")]}),
        encoding="utf-8")
    u = {x["set_code"]: x for x in api.untracked_sets(str(root))}
    assert sorted(u) == ["ABC", "XYZ"]                                              # TST is tracked, EMPTY holds nothing
    assert u["ABC"]["folder"] == str(root / "ABC") and u["ABC"]["n_local"] == 2 and u["ABC"]["n_remote"] == 1
    assert u["XYZ"]["folder"] == "" and u["XYZ"]["n_remote"] == 2 and sorted(u["XYZ"]["events"]) == ["Sealed", "TradSealed"]
    # the files 17lands lists for an untracked set, ready for the download flow (the one already on disk is left out)
    items = api.dataset_files_for_set("ABC", str(root))
    assert items == []                                                              # ABC's only listed file is on disk
    items = api.dataset_files_for_set("XYZ", str(root))
    assert [i.name for i in items] == ["game_data_public.XYZ.Sealed.csv.gz", "game_data_public.XYZ.TradSealed.csv.gz"]
    assert all(i.status == "new" and i.set_code == "XYZ" and i.needs_download for i in items)


def test_status_without_a_model_says_the_data_were_never_used(home, tmp_path):
    api.add_files([str(game(tmp_path / "a", "TST", "TradSealed")), str(game(tmp_path / "a", "TST", "Sealed"))])
    st = api.set_status("TST")
    assert st.stale and not st.has_model and st.pending == ["data never used yet: no model has been calculated for this set"]


# ------------------------------------------------------------------------------------------ with a real (synthetic) model

def test_pending_reasons_name_the_new_files(calc, tmp_path, monkeypatch):
    h = tmp_path / "copy_home"
    shutil.copytree(calc["home"], h)
    monkeypatch.setenv("SEALED_LAB_HOME", str(h))
    st = api.set_status("TST")
    assert st.has_model and not st.stale and st.pending == []
    src = next(calc["synth"].glob("game_data_public.TST.*TradSealed*.csv.gz"), None) or next(calc["synth"].glob("game_data_public.TST.*"))
    extra = tmp_path / "game_data_public.TST.PickTwoDraft.csv.gz"
    shutil.copy(src, extra)
    api.add_files([str(extra)])
    st = api.set_status("TST")
    assert st.stale and st.pending and st.pending[0] == "1 new or changed file: game_data_public.TST.PickTwoDraft.csv.gz"
    extra2 = tmp_path / "game_data_public.TST.TradDraft.csv.gz"
    shutil.copy(src, extra2)
    api.add_files([str(extra2)])
    assert api.set_status("TST").pending[0].startswith("2 new or changed files: game_data_public.TST.")
    assert not api.set_status("TST").pending[0].endswith("…")                    # two names fit
    extra3 = tmp_path / "game_data_public.TST.PremierDraft.csv.gz"
    shutil.copy(src, extra3)
    api.add_files([str(extra3)])
    assert api.set_status("TST").pending[0].startswith("3 new or changed files:") and api.set_status("TST").pending[0].endswith(" …")
    for f in api.set_status("TST").files:                                        # removing a file the model used: said as well
        if Path(f.path).name == "game_data_public.TST.PickTwoDraft.csv.gz":
            api.remove_file("TST", f.fingerprint)
    assert "new or changed" in api.set_status("TST").pending[0]


def test_set_overview_reads_the_model_cheaply(calc):
    ov = api.set_overview("TST")
    assert ov["set_code"] == "TST" and ov["n_cards"] and set(ov["games"]) >= {"bo3", "bo1"}
    assert ov["events"].get("bo3") and set(ov["date_range"]) >= {"bo3"} and all(len(v) == 2 for v in ov["date_range"].values())
    assert {f["kind"] for f in ov["files"]} >= {"game"} and all(not f["missing"] for f in ov["files"])
    m = ov["model"]
    assert m["built_at"] and m["seconds"] > 0 and set(m["formats"]) == set(m["validation"]) == set(m["top_cards"])
    for fmt, rows in m["top_cards"].items():
        assert 1 <= len(rows) <= 5 and [r["pp"] for r in rows] == sorted((r["pp"] for r in rows), reverse=True)
        assert all("name" in r and "pp" in r for r in rows)
    v = m["validation"][m["formats"][0]]
    assert v["status"] == "ok" and v["gain_vs_B0"] is not None and v["lo_vs_B0"] <= v["gain_vs_B0"] <= v["hi_vs_B0"]
    assert {c["key"] for c in m["components"]} >= {"synergy", "matchups", "manasim", "castability"}
    assert m["castability"] is not None and "applied" in m["castability"]
    assert set(m["lam"]) >= {"draft", "bo1", "bo3"}
    assert api.set_overview("NOPE")["model"] is None and api.set_overview("NOPE")["files"] == []


def test_readers_refuse_a_set_that_is_being_recalculated(calc, held_text):
    from sealed_lab.deck import builder as b

    ids = {0: 1}
    assert api.calculating_sets() == []
    with api._calculating("tst"):
        assert api.calculating_sets() == ["TST"]
        for call in (lambda: api.evaluate_deck("TST", ids), lambda: api.build_pool("TST", held_text), lambda: api.score_deck("TST", held_text),
                     lambda: api.card_values("TST"), lambda: api.explain_deck("TST", ids), lambda: api.build_from_pool("TST", ids),
                     lambda: api.synergies("TST", ids), lambda: api.archetype_of("TST", ids), lambda: api.swap_analysis("TST", ids, ids),
                     lambda: api.sideboard_plan("TST", ids, ids)):
            with pytest.raises(SetBusy, match="TST is being recalculated"):
                call()
        assert b.load_model("TST") is not None                                      # only the API entry points are guarded, not other sets
        api.card_info("TST")                                                        # names / images are still available meanwhile
    assert api.calculating_sets() == []
    assert api.evaluate_deck("TST", {int(k): 1 for k in list(api.card_info("TST"))[:3]}) is not None


def test_nested_calculations_and_other_sets_are_independent(home):
    with api._calculating("AAA"), api._calculating("AAA"):
        assert api.calculating_sets() == ["AAA"]
    assert api.calculating_sets() == []
    with api._calculating("AAA"):
        api._guard("BBB")                                                           # no error: another set


def test_shared_caches_are_safe_across_threads(calc, held_text):
    errs: list[BaseException] = []

    def work():
        try:
            for _ in range(15):
                api.card_info("TST")
                api.check_pool("TST", held_text)
                api.set_overview("TST")
        except BaseException as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errs == []


@pytest.fixture(scope="module")
def held(calc):
    return json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def held_text(held):
    h = held[0]
    return "Deck\n" + "".join(f"{n} {k}\n" for k, n in h["pool"].items())


def _own_text(h: dict, typo_in_side: bool = False) -> tuple[str, str | None, int]:
    deck = h["player_deck"]
    side = {n: c - deck.get(n, 0) for n, c in h["pool"].items() if c - deck.get(n, 0) > 0 and n not in BASICS}
    typo = None
    n_typo = 0
    if typo_in_side:
        victim = next(n for n in side if len(n) > 7 and " // " not in n)
        typo = victim[:-2] + victim[-1]                                            # one letter lost
        n_typo = side.pop(victim)
        side[typo] = n_typo
    text = "Deck\n" + "".join(f"{c} {n}\n" for n, c in deck.items()) + "\nSideboard\n" + "".join(f"{c} {n}\n" for n, c in side.items())
    return text, typo, n_typo


def test_check_pool_and_build_with_confirmed_names_and_the_own_deck(calc, held):
    h = held[0]
    text, typo, n_typo = _own_text(h, typo_in_side=True)
    victim_name = next(n for n in h["pool"] if n[:-2] + n[-1] == typo)
    chk = api.check_pool("TST", text)
    assert chk["has_deck"] is True and chk["n_main"] == sum(h["player_deck"].values())
    assert typo in [g["raw"] for g in chk["guesses"]]
    g = next(x for x in chk["guesses"] if x["raw"] == typo)
    assert g["name"] == victim_name and g["candidates"][0]["name"] == victim_name and g["count"] == n_typo
    base = api.build_pool("TST", text, "bo3", n_decks=1, name_overrides={typo: victim_name})
    assert base.user_deck is not None and not base.user_unmatched and base.pool_unmatched == []
    mine = {c.name: c.count for c in base.user_deck.cards}
    assert mine == {n: c for n, c in h["player_deck"].items()}                        # the Deck section, as pasted
    assert sum(base.pool.values()) == sum(c for n, c in h["pool"].items() if n not in BASICS)
    gone = api.build_pool("TST", text, "bo3", n_decks=1, name_overrides={typo: ""})        # "remove this card"
    assert sum(gone.pool.values()) == sum(base.pool.values()) - n_typo
    # a pool without a Sideboard section has no "own deck"
    only = api.build_pool("TST", "Deck\n" + "".join(f"{c} {n}\n" for n, c in h["pool"].items()), "bo3", n_decks=1)
    assert only.user_deck is None and only.user_unmatched == []


def test_score_without_search_is_the_fast_path_and_returns_the_pool(calc, held, monkeypatch):
    h = held[1]
    text, _t, _n = _own_text(h)
    called = []
    monkeypatch.setattr(builder, "build", lambda *a, **k: called.append(1) or (_ for _ in ()).throw(AssertionError("searched")))
    fast = api.score_deck("TST", text, "bo3", search=False)
    assert called == [] and fast.efficiency_pp is None and fast.best_alternative is None
    assert not any(w.startswith("Tip:") for w in fast.warnings)
    expect = {n: c for n, c in h["pool"].items() if n not in BASICS}
    cards = calc and builder.load_model("TST").cards.sort("card_id")["name"].to_list()
    assert {cards[i]: c for i, c in fast.pool.items()} == expect                      # deck + sideboard, basics left out
    assert fast.deck.game_wr > 0
    monkeypatch.undo()
    full = api.score_deck("TST", text, "bo3")                                         # the old behaviour (search) is still the default
    assert full.efficiency_pp is not None and full.deck.game_wr == pytest.approx(fast.deck.game_wr)
    # a corrected name
    text2, typo, _n = _own_text(h, typo_in_side=True)
    victim_name = next(n for n in h["pool"] if n[:-2] + n[-1] == typo)
    r = api.score_deck("TST", text2, "bo3", name_overrides={typo: victim_name}, search=False)
    assert sum(r.pool.values()) == sum(fast.pool.values())
    assert os.environ.get("SEALED_LAB_HOME")
