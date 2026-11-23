"""Arena log watcher: incremental reads, partial JSON, rotation, duplicates, start-up rule (synthetic logs only)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from sealed_lab.deck.arenawatch import ArenaWatcher

NOW = datetime(2026, 10, 24, 12, 0, 0)
HEADER = "Mono path[0] = 'C:/x'\nDETAILED LOGS: ENABLED\nStartup Timestamp: 24/10/2026 11:03:05\n"


def course(event, pool, module="DeckSelect", played=None):
    attrs = [{"name": "LastPlayed", "value": played}] if played else []
    return {"CourseId": "c-" + event, "InternalEventName": event, "CurrentModule": module,
            "CourseDeckSummary": {"Attributes": attrs}, "CourseDeck": {"MainDeck": [], "Sideboard": []}, "CardPool": pool}


def entry(stamp, doc, pretty=False):
    body = json.dumps(doc, indent=2) if pretty else json.dumps(doc, separators=(",", ":"))
    return f"[UnityCrossThreadLogger]{stamp}\n<== EventGetCoursesV2(1)\n{body}\n"


def stamp(minutes_ago):
    t = NOW - timedelta(minutes=minutes_ago)
    return f"{t.day}/{t.month}/{t.year} {t.hour}:{t.minute:02d}:{t.second:02d}"


def make(tmp_path, text="", **kw):
    d = tmp_path / "MTGA"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Player.log").write_bytes(text.encode("utf-8"))
    return d, ArenaWatcher(d, now=lambda: NOW, **kw)


def append(d, text):
    with open(d / "Player.log", "ab") as f:
        f.write(text.encode("utf-8"))


def test_missing_log_is_quiet_then_appears(tmp_path):
    w = ArenaWatcher(tmp_path / "nope", now=lambda: NOW)
    assert w.poll() == [] and w.state == "no_log"
    d = tmp_path / "MTGA"
    d.mkdir()
    w2 = ArenaWatcher(d, now=lambda: NOW)
    assert w2.poll() == []
    append(d, HEADER)
    assert w2.poll() == [] and w2.state == "ok" and w2.detailed is True


def test_new_pool_reported_once(tmp_path):
    d, w = make(tmp_path, HEADER)
    assert w.poll() == [] and not w.changed()
    append(d, entry(stamp(1), {"Courses": [course("Draft_MSH_1", [1, 2]), course("Sealed_MSH_20261001", [101, 101, 102])]}))
    assert w.changed()
    got = w.poll()
    assert [c.event for c in got] == ["Sealed_MSH_20261001"] and got[0].pool == [101, 101, 102]
    assert w.poll() == [] and not w.changed()
    # the same course re-sent: not reported again; a different pool for the event is
    append(d, entry(stamp(0), {"Courses": [course("Sealed_MSH_20261001", [101, 101, 102])]}))
    assert w.poll() == []
    append(d, entry(stamp(0), {"Course": course("Sealed_MSH_20261001", [101, 101, 102, 103])}))
    assert [len(c.pool) for c in w.poll()] == [4]


def test_completed_and_non_sealed_not_reported(tmp_path):
    d, w = make(tmp_path, HEADER)
    w.poll()
    append(d, entry(stamp(1), {"Courses": [course("Sealed_MSH_1", [1, 2], module="Complete"), course("Draft_MSH_2", [3])]}))
    assert w.poll() == []


def test_only_appended_bytes_are_read(tmp_path, monkeypatch):
    d, w = make(tmp_path, HEADER + "x" * 5000 + "\n")
    w.poll()
    seen = []
    import builtins
    real = builtins.open

    class Spy:
        def __init__(self, f):
            self.f = f

        def read(self, n=-1):
            seen.append(n)
            return self.f.read(n)

        def __getattr__(self, k):
            return getattr(self.f, k)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

    monkeypatch.setattr("sealed_lab.deck.arenawatch.open", lambda *a, **k: Spy(real(*a, **k)), raising=False)
    line = entry(stamp(1), {"Courses": [course("Sealed_MSH_1", [5, 6])]})
    append(d, line)
    assert len(w.poll()) == 1
    assert sum(seen) <= len(line.encode()) + 256 and max(seen) <= len(line.encode())


def test_partial_line_and_pretty_json_across_polls(tmp_path):
    d, w = make(tmp_path, HEADER)
    w.poll()
    text = entry(stamp(1), {"Courses": [course("Sealed_MSH_1", [5, 6, 7])]}, pretty=True)
    cuts = [len(text) // 3, 2 * len(text) // 3, len(text) - 1, len(text)]
    prev, out = 0, []
    for c in cuts:
        append(d, text[prev:c])
        prev = c
        out += w.poll()
    assert len(out) == 1 and out[0].pool == [5, 6, 7] and out[0].timestamp.startswith("2026-10-24T11:59")
    # one-line JSON cut in the middle of a line
    text = entry(stamp(0), {"Courses": [course("Sealed_MSH_2", [8, 9])]})
    append(d, text[:-20])
    assert w.poll() == []
    append(d, text[-20:])
    assert [c.event for c in w.poll()] == ["Sealed_MSH_2"]


def test_pretty_json_split_on_line_boundaries(tmp_path):
    d, w = make(tmp_path, HEADER)
    w.poll()
    lines = entry(stamp(1), {"Courses": [course("Sealed_MSH_1", [5, 6, 7])]}, pretty=True).splitlines(True)
    out = []
    for i in range(0, len(lines), 7):
        append(d, "".join(lines[i:i + 7]))
        out += w.poll()
    assert [c.event for c in out] == ["Sealed_MSH_1"]


def test_rotation_and_truncation_reset(tmp_path):
    d, w = make(tmp_path, HEADER + entry(stamp(500), {"Courses": [course("Sealed_MSH_old", [1, 2])]}))
    assert w.poll() == []                                        # old event: start-up rule
    (d / "Player.log").write_bytes((HEADER.replace("11:03:05", "11:59:00") + "pad\n" * 5).encode())   # shorter: restart
    assert w.poll() == []
    append(d, entry(stamp(0), {"Courses": [course("Sealed_MSH_old", [1, 2]), course("Sealed_MSH_new", [3, 4, 5])]}))
    got = w.poll()
    assert [c.event for c in got] == ["Sealed_MSH_new"]          # the old one was seen: not reported again
    # replaced by a longer file with another header
    body = HEADER.replace("11:03:05", "12:00:01") + "pad\n" * 5000 + entry(stamp(0), {"Courses": [course("Sealed_MSH_3", [9])]})
    (d / "Player.log").write_bytes(body.encode())
    assert [c.event for c in w.poll()] == ["Sealed_MSH_3"]


def _startup(tmp_path, name, age_min, module="DeckSelect", played=None, extra=()):
    text = HEADER + entry(stamp(age_min), {"Courses": [course("Sealed_MSH_1", [1, 2], module=module, played=played), *extra]})
    d, w = make(tmp_path / name, text)
    return w.poll(), w


def test_startup_rule(tmp_path):
    got, w = _startup(tmp_path, "a", 30)
    assert [c.event for c in got] == ["Sealed_MSH_1"]               # recent and open: offered once
    assert w.poll() == []
    assert _startup(tmp_path, "b", 180)[0] == []                    # older than two hours
    assert _startup(tmp_path, "c", 30, module="Complete")[0] == []  # completed
    # Arena re-sent an old event just now: its own LastPlayed date decides
    assert _startup(tmp_path, "d", 1, played="2026-09-01T10:00:00")[0] == []
    # several pools: only one (the most recently played open one) is offered
    got, _ = _startup(tmp_path, "e", 20, extra=(course("Sealed_MSH_0", [7, 8]),))
    assert len(got) == 1
    # Player-prev.log is never read
    d2 = tmp_path / "prev"
    d2.mkdir()
    (d2 / "Player-prev.log").write_text(HEADER + entry(stamp(5), {"Courses": [course("Sealed_MSH_1", [1])]}), encoding="utf-8")
    assert ArenaWatcher(d2, now=lambda: NOW).poll() == []
