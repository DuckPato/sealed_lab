"""Watch Arena's Player.log for a new Sealed pool (pure Python, no Qt).

`ArenaWatcher.poll()` reads only the bytes Arena appended since the previous poll (the log is 10+ MB; it is read in full
once, at the first poll, and never again), keeps the incomplete last line and the lines of an unfinished multi-line JSON
document for the next poll, and returns the Sealed courses with a non-empty card pool that it has not reported yet.  Parsing is
arenalog's (`scan_courses`).  `changed()` is a single `os.stat`, cheap enough for a GUI timer; `poll()` may parse megabytes the
first time, so a GUI should call it off the UI thread.

Start-up rule (the first read of a file, and the first read after the file was replaced or truncated, i.e. Arena restarted and
`Player.log` was rotated to `Player-prev.log`): Arena re-sends every course of the account at each login, so the log holds old
pools too.  None of them is reported, except the one `pick_sealed` would import (the sealed course most recently played), and
only if that event is not completed and was last played less than `START_MAX_AGE` (2 hours) ago.  So a pool opened shortly
before the app started is still offered, and a month-old event is not.  Every course read at that moment counts as already seen.
Later reads report every sealed course with a pool whose (event, pool) pair is new and whose event is not completed.
`Player-prev.log` is never read.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sealed_lab.deck import arenalog
from sealed_lab.deck.arenalog import CourseEntry

log = logging.getLogger(__name__)

START_MAX_AGE = timedelta(hours=2)
CHUNK = 16 * 1024 * 1024        # bytes parsed at a time (bounds the memory of the first read of a very large log)
MAX_CARRY = 8 * 1024 * 1024     # a single line longer than this without a newline is dropped
HEAD = 256                      # bytes compared to notice a replaced file that is already longer than the old one
CARRY_LINES = arenalog.MAX_JSON_LINES + 3


def course_key(c: CourseEntry) -> tuple:
    return c.event, tuple(sorted(c.pool))


def _stamp_of(line: str) -> bool:
    return bool(arenalog._line_ts(line))


class ArenaWatcher:
    def __init__(self, log_dir: str | Path | None = None, now=datetime.now, max_age: timedelta = START_MAX_AGE):
        self._dir = Path(log_dir) if log_dir else None
        self._now = now
        self.max_age = max_age
        self.seen: set[tuple] = set()
        self.detailed: bool | None = None       # the log header says whether detailed logs are on (None: unknown / no log yet)
        self.state = "no_log"                   # "no_log" | "ok"
        self._reset()
        self._stat: tuple | None = None

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        self._offset = 0
        self._head = b""
        self._bytes = b""                       # incomplete last line
        self._lines: list[str] = []             # unfinished multi-line document (plus the stamp line before it)
        self._stamp = ""                        # last line carrying a timestamp, prepended so documents keep their time
        self._from_start = True                 # next read starts at offset 0: apply the start-up rule

    @property
    def path(self) -> Path:
        return (self._dir or arenalog.default_log_dir()) / "Player.log"

    def _stat_key(self) -> tuple | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_size, st.st_mtime_ns

    def changed(self) -> bool:
        """One os.stat: did the file appear, grow, shrink or change since the last poll?"""
        return self._stat_key() != self._stat

    # ------------------------------------------------------------------ polling
    def poll(self) -> list[CourseEntry]:
        """New sealed courses (oldest first).  Never raises: an unreadable log is a quiet empty result."""
        try:
            return self._poll()
        except OSError as e:
            log.info("Arena watcher cannot read %s: %s", self.path, e)
            return []

    def _poll(self) -> list[CourseEntry]:
        key = self._stat_key()
        self._stat = key
        if key is None:
            self.state = "no_log"
            self._reset()
            return []
        self.state = "ok"
        size = key[0]
        with open(self.path, "rb") as fh:
            if self._offset and (size < self._offset or fh.read(len(self._head)) != self._head):
                log.info("Arena log was replaced or truncated: reading it from the start")
                self._reset()
            if size == self._offset:
                return []
            fh.seek(self._offset)
            found: list[CourseEntry] = []
            first = self._from_start
            while self._offset < size:
                data = fh.read(min(CHUNK, size - self._offset))
                if not data:
                    break
                if self._offset == 0:
                    self._head = data[:HEAD]
                    self.detailed = arenalog.detailed_logs_state(data[:20000].decode("utf-8", errors="replace"))
                self._offset += len(data)
                found += self._feed(data)
        self._from_start = False
        return self._select(found, first)

    def _feed(self, data: bytes) -> list[CourseEntry]:
        data = self._bytes + data
        cut = data.rfind(b"\n")
        if cut < 0:
            self._bytes = data if len(data) <= MAX_CARRY else b""
            return []
        self._bytes = data[cut + 1:]
        lines = data[:cut].decode("utf-8", errors="replace").splitlines()
        lines = ([self._stamp] if self._stamp else []) + self._lines + lines
        # the lines of a document still open at the end are kept for the next read (with the two lines before it, which hold
        # its prefix and its timestamp)
        depth, start = 0, None
        for i, ln in enumerate(lines):
            if depth <= 0:
                if "{" in ln:
                    start, depth = i, ln.count("{") - ln.count("}")
            else:
                depth += ln.count("{") - ln.count("}")
        for ln in reversed(lines):
            if _stamp_of(ln):
                self._stamp = ln
                break
        if depth > 0 and start is not None and len(lines) - start <= arenalog.MAX_JSON_LINES:
            self._lines = lines[max(0, start - 2):]
        else:
            self._lines = []
        return arenalog.scan_courses("\n".join(lines), "Player.log", self._now())

    def _select(self, found: list[CourseEntry], first: bool) -> list[CourseEntry]:
        out: list[CourseEntry] = []
        sealed = [c for c in found if arenalog.SEALED_RE.search(c.event)]
        if first:
            best = arenalog.pick_sealed(found)
            fresh = [c for c in sealed if course_key(c) not in self.seen]
            self.seen.update(course_key(c) for c in sealed)
            if best is not None and best in fresh and self._recent(best):
                out.append(best)
            return out
        for c in sealed:
            k = course_key(c)
            if k in self.seen:
                continue
            self.seen.add(k)
            if not c.completed:
                out.append(c)
        return out

    def _recent(self, c: CourseEntry) -> bool:
        if c.completed:
            return False
        when = c.last_played or c.timestamp
        if not when:
            return False
        try:
            t = datetime.fromisoformat(when)
        except ValueError:
            return False
        return self._now() - t <= self.max_age
