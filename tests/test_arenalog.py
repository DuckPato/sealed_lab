"""Arena log import: synthetic Player.log text, id translation with a fake Scryfall session, error messages."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from sealed_lab.deck import arena, arenalog

HEADER = "Mono path[0] = 'C:/x'\nDETAILED LOGS: ENABLED\nStartup Timestamp: 24/10/2026 11:03:05\n"
IDS = {101: "Alpha Hero", 102: "Beta Villain", 103: "Gamma // Delta", 104: "Plains"}


def course(event: str, pool: list[int], main=None, side=None) -> dict:
    return {"CourseId": "c-" + event, "InternalEventName": event, "CurrentModule": "DeckSelect", "ModulePayload": "",
            "CourseDeckSummary": {"Attributes": []},
            "CourseDeck": {"MainDeck": [{"cardId": c, "quantity": n} for c, n in (main or {}).items()],
                           "Sideboard": [{"cardId": c, "quantity": n} for c, n in (side or {}).items()]},
            "CardPool": pool}


def one_line(stamp: str, prefix: str, doc: dict) -> str:
    return f"[UnityCrossThreadLogger]{stamp}\n{prefix}\n{json.dumps(doc, separators=(',', ':'))}\n"


@pytest.fixture()
def logdir(tmp_path, home):
    d = tmp_path / "MTGA"
    d.mkdir()
    return d


def seed_ids(set_code="MSH", ids=IDS):
    arenalog.save_ids(set_code, dict(ids), searched=True)


def test_parses_latest_sealed_course_in_a_courses_list(logdir):
    seed_ids()
    text = HEADER
    text += one_line("24/10/2026 9:03:22", "<== EventGetCoursesV2(1)",
                     {"Courses": [course("Constructed_BestOf3", []), course("Sealed_MSH_20261001", [101, 101, 102])]})
    text += one_line("24/10/2026 10:15:00", "<== EventGetCoursesV2(2)",
                     {"Courses": [course("Draft_MSH_20261002", [101, 101, 101, 101]),
                                  course("Sealed_MSH_20261001", [101, 101, 102, 103, 104])]})
    (logdir / "Player.log").write_text(text, encoding="utf-8")
    r = arenalog.import_pool("MSH", log_dir=logdir, offline=True)
    assert r.event == "Sealed_MSH_20261001" and r.set_code == "MSH" and r.source == "Player.log"
    assert r.timestamp == "2026-10-24T10:15:00"
    main, side = arena.parse_arena(r.text)
    assert main == {"Alpha Hero": 2, "Beta Villain": 1, "Gamma": 1, "Plains": 1} and not side   # front face, copies counted
    assert r.n_cards == 5 and not r.unknown_ids and not r.warnings


def test_single_course_and_trad_sealed_and_set_from_event_name(logdir):
    seed_ids()
    doc = {"Course": course("TradSealed_MSH_20261001", [101, 102]), "InventoryInfo": {"Gems": 5}}
    (logdir / "Player.log").write_text(HEADER + one_line("24/10/2026 9:00:00", "<== EventJoin(3)", doc), encoding="utf-8")
    r = arenalog.import_pool(None, log_dir=logdir, offline=True)
    assert r.event == "TradSealed_MSH_20261001" and r.set_code == "MSH" and r.n_cards == 2


def test_pretty_printed_json_and_bare_course(logdir):
    seed_ids()
    pretty = json.dumps(course("Sealed_MSH_20261005", [102, 102, 101]), indent=2)
    text = HEADER + "[UnityCrossThreadLogger]25/10/2026 8:30:15 AM\n<== EventSetDeckV3(9)\n" + pretty + "\n"
    (logdir / "Player.log").write_text(text, encoding="utf-8")
    r = arenalog.import_pool("MSH", log_dir=logdir, offline=True)
    assert r.event == "Sealed_MSH_20261005" and r.n_cards == 3
    assert r.timestamp.startswith("2026-10-25T08:30:15")


def test_current_log_wins_over_previous_but_previous_is_used_when_needed(logdir):
    seed_ids()
    new = HEADER + one_line("24/10/2026 9:00:00", "<==", {"Courses": [course("Constructed_BestOf3", [])]})
    old = HEADER + one_line("23/10/2026 9:00:00", "<==", course("Sealed_MSH_20261001", [101, 102, 103]))
    (logdir / "Player.log").write_text(new, encoding="utf-8")
    (logdir / "Player-prev.log").write_text(old, encoding="utf-8")
    r = arenalog.import_pool("MSH", log_dir=logdir, offline=True)
    assert r.source == "Player-prev.log" and r.n_cards == 3 and r.timestamp.startswith("2026-10-23")
    both = HEADER + one_line("24/10/2026 10:00:00", "<==", course("Sealed_MSH_20261008", [101]))
    (logdir / "Player.log").write_text(both, encoding="utf-8")
    r = arenalog.import_pool("MSH", log_dir=logdir, offline=True)
    assert r.source == "Player.log" and r.event == "Sealed_MSH_20261008"


def test_error_messages(logdir, tmp_path):
    with pytest.raises(arenalog.LogNotFound, match="not found"):
        arenalog.import_pool("MSH", log_dir=tmp_path / "nowhere", offline=True)
    # logs without any course data / header: detailed logs are probably off
    (logdir / "Player.log").write_text("Mono path[0] = 'x'\nsome unity line\n", encoding="utf-8")
    with pytest.raises(arenalog.DetailedLogsOff, match="Detailed Logs"):
        arenalog.import_pool("MSH", log_dir=logdir, offline=True)
    (logdir / "Player.log").write_text("DETAILED LOGS: DISABLED\n", encoding="utf-8")
    with pytest.raises(arenalog.DetailedLogsOff):
        arenalog.import_pool("MSH", log_dir=logdir, offline=True)
    # detailed logs on, courses present but no sealed pool (empty pool of a sealed event, or non-sealed pools)
    doc = {"Courses": [course("Sealed_MSH_20261001", []), course("Constructed_BestOf3", []), course("Draft_X", [101])]}
    (logdir / "Player.log").write_text(HEADER + one_line("24/10/2026 9:00:00", "<==", doc), encoding="utf-8")
    with pytest.raises(arenalog.NoSealedPool, match="No sealed pool found"):
        arenalog.import_pool("MSH", log_dir=logdir, offline=True)
    # sealed pool whose ids cannot be translated
    (logdir / "Player.log").write_text(HEADER + one_line("24/10/2026 9:00:00", "<==", course("Sealed_ZZZ_1", [9001, 9002])),
                                       encoding="utf-8")
    with pytest.raises(arenalog.UnknownCards, match="card ids"):
        arenalog.import_pool("ZZZ", log_dir=logdir, offline=True)


def test_some_unknown_ids_are_reported_not_fatal(logdir):
    seed_ids()
    (logdir / "Player.log").write_text(HEADER + one_line("24/10/2026 9:00:00", "<==",
                                                           course("Sealed_MSH_1", [101, 555, 556, 102])), encoding="utf-8")
    r = arenalog.import_pool("MSH", log_dir=logdir, offline=True)
    assert r.n_cards == 2 and r.unknown_ids == [555, 556]
    assert any("2 card id(s)" in w and "555" in w for w in r.warnings)


class FakeResp:
    def __init__(self, status: int, payload: dict | None = None):
        self.status_code = status
        self._p = payload or {}

    def json(self):
        return self._p


class FakeHttp:
    """Scryfall stand-in: a 2-page `e:msh` search and /cards/arena/{id}."""

    def __init__(self, search_ok=True, singles: dict[int, str] | None = None):
        self.calls: list[str] = []
        self.search_ok = search_ok
        self.singles = singles or {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        if url.endswith("/cards/search"):
            assert params["q"] == "e:msh" and params["unique"] == "prints"
            if not self.search_ok:
                return FakeResp(500)
            return FakeResp(200, {"has_more": True, "next_page": "https://api.scryfall.com/page2",
                                  "data": [{"name": "Alpha Hero", "arena_id": 101}, {"name": "No Arena Id"}]})
        if url.endswith("/page2"):
            return FakeResp(200, {"has_more": False, "data": [{"name": "Beta Villain", "arena_id": 102}]})
        aid = int(url.rsplit("/", 1)[1])
        return FakeResp(200, {"name": self.singles[aid]}) if aid in self.singles else FakeResp(404)


def test_translation_via_set_search_cache_and_single_lookups(logdir, home):
    sleeps: list[float] = []
    http = FakeHttp(singles={103: "Gamma // Delta"})
    names, unknown, notes = arenalog.resolve_ids([101, 102, 103, 999], "MSH", http=http, sleep=sleeps.append)
    assert names == {101: "Alpha Hero", 102: "Beta Villain", 103: "Gamma // Delta"} and unknown == [999] and not notes
    assert sum(u.endswith("/cards/search") for u in http.calls) == 1 and any(u.endswith("/cards/arena/103") for u in http.calls)
    assert any(abs(s - arenalog.FALLBACK_INTERVAL) < 1e-9 for s in sleeps)          # <= 2 requests/s on the single lookups
    cached = json.loads(arenalog.ids_file("MSH").read_text(encoding="utf-8"))
    assert cached["searched"] and cached["ids"]["101"] == "Alpha Hero" and cached["ids"]["103"] == "Gamma // Delta"
    # second import: everything cached for 101/102/103, no new search; only the unknown id is tried again
    http2 = FakeHttp()
    names2, unknown2, _ = arenalog.resolve_ids([101, 102, 103], "MSH", http=http2, sleep=lambda s: None)
    assert names2 == names and not http2.calls and not unknown2
    # offline never touches the network
    http3 = FakeHttp()
    n3, u3, _ = arenalog.resolve_ids([101, 999], "MSH", offline=True, http=http3)
    assert n3 == {101: "Alpha Hero"} and u3 == [999] and not http3.calls


def test_network_failures_are_survivable(logdir, home):
    class Boom:
        def get(self, *a, **k):
            raise ConnectionError("offline")
    names, unknown, notes = arenalog.resolve_ids([101], "MSH", http=Boom(), sleep=lambda s: None)
    assert names == {} and unknown == [101] and notes
    names, unknown, notes = arenalog.resolve_ids([101], "MSH", http=FakeHttp(search_ok=False), sleep=lambda s: None)
    assert unknown == [101]


def test_timestamp_parsing():
    ref = datetime(2026, 10, 24, 12, 0, 0)
    assert arenalog._parse_ts("24/10/2026 9:03:22", ref) == "2026-10-24T09:03:22"
    assert arenalog._parse_ts("10/24/2026 9:03:22 PM", ref) == "2026-10-24T21:03:22"
    assert arenalog._parse_ts("2026-10-24T11:03:43", ref) == "2026-10-24T11:03:43"
    # ambiguous day/month: the reading that is not after the file's mtime and closest to it wins
    assert arenalog._parse_ts("03/10/2026 9:00:00", ref) == "2026-10-03T09:00:00"
    assert arenalog._parse_ts("garbage", ref) == ""
    assert arenalog.event_set("Sealed_MSH_20261001") == "MSH" and arenalog.event_set("TradSealed_SOS_1") == "SOS"
    assert arenalog.event_set("Sealed") is None


def test_real_log_if_present_reports_cleanly():
    """The user's real log (when present) must not crash the importer; with no sealed event it says so."""
    if not arenalog.find_logs():
        pytest.skip("no Arena log on this machine")
    try:
        r = arenalog.import_pool("MSH", offline=True)
    except arenalog.ArenaLogError as e:
        assert str(e)
    else:
        assert r.n_cards > 0


def test_api_wrapper(logdir, home):
    from sealed_lab import api

    seed_ids()
    (logdir / "Player.log").write_text(HEADER + one_line("24/10/2026 9:00:00", "<==", course("Sealed_MSH_1", [101, 102])),
                                       encoding="utf-8")
    r = api.import_arena_log("msh", log_dir=str(logdir))
    assert r.text.startswith("Deck\n") and r.n_cards == 2
