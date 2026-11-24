"""New-17lands-data check / download (all network calls mocked) and pool set detection."""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
from pathlib import Path

import polars as pl
import pytest

from sealed_lab import api
from sealed_lab.deck import setdetect
from sealed_lab.deck.names import NameIndex
from sealed_lab.ingest import datasets, registry
from sealed_lab.ingest.detect import detect_file

S3 = "https://data.17landscapes.com/analysis_data"


# ---------------------------------------------------------------------------------------------
# fake network
# ---------------------------------------------------------------------------------------------

def _p(text: str, url: str | None = None) -> list[dict]:
    span = [] if url is None else [{"start": 0, "end": len(text), "type": "hyperlink", "data": {"link_type": "Web", "url": url}}]
    return [{"type": "paragraph", "text": text, "spans": span}]


def _row(code: str, event: str, date: str, game=True, replay=True, draft=True, pad: str = "") -> dict:
    def link(kind, on):
        if not on:
            return _p("-")
        return _p("link", f"{pad}{S3}/{kind}_data/{kind}_data_public.{code}.{event}.csv.gz")
    return {"expansion": _p(code), "format": _p(event), "last_updated": _p(date),
            "draft_data": link("draft", draft), "game_data": link("game", game), "replay_data": link("replay", replay)}


def prismic_doc(rows: list[dict], published: str = "2026-10-20T04:12:10+0000") -> dict:
    return {"results": [{"last_publication_date": published, "data": {
        "datasets": rows, "helper_files": [],
        "card_lists": [{"dataset": _p("Cards", f"{S3}/cards/cards.csv")}]}}]}


class Resp:
    def __init__(self, status=200, headers=None, json_data=None, body: bytes = b""):
        self.status_code, self.headers, self._json, self.body = status, headers or {}, json_data, body

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeNet:
    """requests.Session stand-in: Prismic documents, HEAD metadata and file bodies by URL."""

    def __init__(self, doc: dict, files: dict[str, dict] | None = None):
        self.doc, self.files = doc, files or {}
        self.gets: list[str] = []
        self.heads: list[str] = []
        self.offline = False
        self.chunk_hook = None
        self.api = None                 # JSON of the 17lands API (None: the API answers 404 and the Prismic fallback is used)

    def get(self, url, params=None, timeout=None, stream=False):
        self.gets.append(url)
        if self.offline:
            raise ConnectionError("offline")
        if url == datasets.LISTING_API and self.api is not None:
            return Resp(json_data=self.api)
        if url.endswith("/api/v2"):
            return Resp(json_data={"refs": [{"id": "master", "ref": "REF1", "isMasterRef": True}]})
        if "documents/search" in url:
            assert params and params["ref"] == "REF1" and "public-data" in params["q"]
            return Resp(json_data=self.doc)
        f = self.files.get(url)
        if f is None:
            return Resp(404)
        body = f.get("body", b"x" * f["size"])
        if self.chunk_hook:
            body = self.chunk_hook(body)
        return Resp(headers={"Content-Length": str(f.get("declared", len(body)))}, body=body)

    def head(self, url, timeout=None, allow_redirects=True):
        self.heads.append(url)
        if self.offline:
            raise ConnectionError("offline")
        f = self.files.get(url)
        if f is None:
            return Resp(403)
        return Resp(headers={"Content-Length": str(f["size"]), "ETag": f'"{f.get("etag", "e1")}"',
                             "Last-Modified": f.get("lm", "Mon, 27 Jul 2026 18:11:23 GMT")})


def url(kind: str, code: str, event: str) -> str:
    return f"{S3}/{kind}_data/{kind}_data_public.{code}.{event}.csv.gz"


def write_game_gz(path: Path, code: str = "AAA", event: str = "TradSealed") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("expansion,event_type,draft_id,won,deck_Plains\n")
        f.write(f"{code},{event},d1,True,1\n")
    return path


@pytest.fixture()
def data_dir(tmp_path, home):
    d = tmp_path / "data"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------------------------

def test_parse_listing_rows_dates_and_odd_urls():
    doc = prismic_doc([
        _row("AAA", "TradSealed", "2026-10-17", draft=False, pad=" "),            # leading space in the link
        _row("AAA", "Sealed", "2026-10-17 (x)", replay=False),                    # date with a suffix
        {"expansion": _p("ZZZ"), "format": _p("PremierDraft"), "last_updated": _p(""), "game_data": _p("-")},
    ])
    doc["results"][0]["data"]["unlisted"] = [{"x": _p("again", url("replay", "BBB", "Sealed"))}]    # found by the regex fallback only
    lst = datasets.parse_listing(doc)
    by_name = {f.name: f for f in lst.files}
    assert lst.published.startswith("2026-10-20")
    assert "game_data_public.AAA.TradSealed.csv.gz" in by_name and by_name["game_data_public.AAA.TradSealed.csv.gz"].url.startswith("https://")
    assert by_name["game_data_public.AAA.TradSealed.csv.gz"].row_date == "2026-10-17"
    assert "draft_data_public.AAA.TradSealed.csv.gz" not in by_name                  # '-' = no file
    assert by_name["game_data_public.AAA.Sealed.csv.gz"].row_date == "2026-10-17"
    assert "replay_data_public.BBB.Sealed.csv.gz" in by_name and by_name["replay_data_public.BBB.Sealed.csv.gz"].row_date == ""
    assert not any("cards.csv" in f.name for f in lst.files)                          # helper files are not datasets
    f = by_name["replay_data_public.AAA.TradSealed.csv.gz"]
    assert (f.kind, f.set_code, f.event) == ("replay", "AAA", "TradSealed")


def test_list_remote_uses_master_ref():
    net = FakeNet(prismic_doc([_row("AAA", "Sealed", "2026-10-17")]))
    lst = datasets.list_remote(net)
    assert len(lst.files) == 3 and len(net.gets) == 3          # the API (404) first, then the two Prismic requests


# ---------------------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------------------

def _world(data_dir: Path):
    """AAA = newest set (priority), BBB = in the app, DDD = old set only in the data folder."""
    rows = [_row("AAA", "TradSealed", "2026-10-17", draft=False), _row("AAA", "Sealed", "2026-10-17", draft=False),
            _row("AAA", "PremierDraft", "2026-10-17"), _row("AAA", "PickTwoDraft", "2026-10-17"),
            _row("BBB", "TradSealed", "2026-08-01", draft=False), _row("BBB", "Sealed", "2026-08-01", draft=False),
            _row("DDD", "Sealed", "2025-01-01", replay=False, draft=False),
            _row("EEE", "Sealed", "2024-01-01", replay=False, draft=False)]
    files = {}
    for r in rows:
        for key in ("game_data", "replay_data", "draft_data"):
            for u in datasets._links(r[key]):
                files[u.strip()] = {"size": 1000, "etag": "e1"}
    return prismic_doc(rows), files


def test_check_new_updated_current_and_priority(data_dir, monkeypatch):
    monkeypatch.setattr(datasets, "NEWEST_SETS", 1)                       # priority = AAA (newest) + the app's sets
    doc, files = _world(data_dir)
    # local state: AAA TradSealed game current, AAA Sealed game outdated (size), AAA nothing else
    write_game_gz(data_dir / "AAA" / "game_data_public.AAA.TradSealed.csv.gz")
    cur = data_dir / "AAA" / "game_data_public.AAA.TradSealed.csv.gz"
    files[url("game", "AAA", "TradSealed")]["size"] = cur.stat().st_size
    old = write_game_gz(data_dir / "AAA" / "game_data_public.AAA.Sealed.csv.gz")
    files[url("game", "AAA", "Sealed")]["size"] = old.stat().st_size + 500
    # uncompressed copies: older than the server file -> updated; newer -> current
    csv_old = data_dir / "BBB" / "game_data_public.BBB.TradSealed.csv"
    csv_old.parent.mkdir()
    csv_old.write_text("won,deck_Plains\n1,1\n")
    os.utime(csv_old, (1_700_000_000, 1_700_000_000))                     # 2023, server file is from 2026
    csv_new = data_dir / "BBB" / "game_data_public.BBB.Sealed.csv"
    csv_new.write_text("won,deck_Plains\n1,1\n")
    os.utime(csv_new, (1_788_000_000, 1_788_000_000))                     # 2026-08-29 > server Last-Modified (2026-07-27)
    # old set, only in the data folder, 17lands' date is older than our file: no HEAD at all
    write_game_gz(data_dir / "DDD" / "game_data_public.DDD.Sealed.csv.gz", "DDD", "Sealed")
    # old set whose 17lands date is newer than our file: HEAD, size differs -> updated
    ee = write_game_gz(data_dir / "EEE" / "game_data_public.EEE.Sealed.csv.gz", "EEE", "Sealed")
    os.utime(ee, (1_600_000_000, 1_600_000_000))
    files[url("game", "EEE", "Sealed")]["size"] = ee.stat().st_size + 1
    # BBB is known to the app (registry), so it is a priority set
    registry.register(detect_file(csv_old))
    net = FakeNet(doc, files)
    res = datasets.check(str(data_dir), force=True, session=net)
    assert not res.error and res.n_remote > 10
    got = {(i.set_code, i.kind, i.event): i.status for i in res.items}
    assert got[("AAA", "game", "Sealed")] == "updated"
    assert ("AAA", "game", "TradSealed") not in got                      # current
    assert got[("AAA", "replay", "TradSealed")] == "new"
    assert got[("AAA", "game", "PremierDraft")] == "new"
    assert ("AAA", "game", "PickTwoDraft") not in got                    # not a format the app cares about, none on disk
    assert got[("AAA", "draft", "PremierDraft")] == "new"
    assert next(i for i in res.items if i.kind == "draft").optional
    assert got[("BBB", "game", "TradSealed")] == "updated"
    assert "uncompressed" in next(i for i in res.items if i.set_code == "BBB" and i.event == "TradSealed" and i.kind == "game").reason
    assert ("BBB", "game", "Sealed") not in got                          # uncompressed but newer than the server file
    assert got[("EEE", "game", "Sealed")] == "updated"
    assert not any(i.set_code == "DDD" for i in res.items)
    assert all("DDD" not in h for h in net.heads)
    assert all(i.size == 1000 or i.status == "updated" for i in res.items)
    # items are sorted by set, then game before replay before draft
    keys = [(i.set_code, datasets.KIND_ORDER[i.kind]) for i in res.items]
    assert keys == sorted(keys)


def test_check_flags_unregistered_current_copy_and_ignores_current_registered(data_dir, home):
    doc = prismic_doc([_row("AAA", "TradSealed", "2026-10-17", replay=False, draft=False),
                       _row("AAA", "Sealed", "2026-10-17", replay=False, draft=False)])
    # TradSealed: registered file is an old uncompressed copy, a current .csv.gz sits next to it
    old_csv = data_dir / "AAA" / "game_data_public.AAA.TradSealed.csv"
    old_csv.parent.mkdir()
    old_csv.write_text("expansion,event_type,draft_id,won,deck_Plains\nAAA,TradSealed,d,True,1\n")
    os.utime(old_csv, (1_700_000_000, 1_700_000_000))
    registry.register(detect_file(old_csv))
    gz = write_game_gz(data_dir / "AAA" / "game_data_public.AAA.TradSealed.csv.gz")
    # Sealed: registered uncompressed copy is NEWER than the server file and holds the SAME data as the .gz next to it -> nothing
    # to do (when the data differ the check reports it: see test_zz_audit_fixes)
    import gzip as _gzip

    s_gz = write_game_gz(data_dir / "AAA" / "game_data_public.AAA.Sealed.csv.gz", "AAA", "Sealed")
    s_csv = data_dir / "AAA" / "game_data_public.AAA.Sealed.csv"
    with _gzip.open(s_gz, "rb") as fh:
        s_csv.write_bytes(fh.read())
    os.utime(s_csv, (1_788_000_000, 1_788_000_000))
    registry.register(detect_file(s_csv))
    net = FakeNet(doc, {url("game", "AAA", "TradSealed"): {"size": gz.stat().st_size},
                        url("game", "AAA", "Sealed"): {"size": s_gz.stat().st_size}})
    res = datasets.check(str(data_dir), force=True, session=net)
    assert [(i.event, i.status) for i in res.items] == [("TradSealed", "unregistered")]
    it = res.items[0]
    assert it.local_path == str(gz) and not it.needs_download and it.set_in_app


def test_check_offline_and_head_failures_never_raise(data_dir):
    doc, files = _world(data_dir)
    net = FakeNet(doc, files)
    net.offline = True
    res = datasets.check(str(data_dir), force=True, session=net)
    assert res.error and res.items == []
    # listing fine, HEAD of some files fails (403) -> those are skipped, others reported
    net.offline = False
    del files[url("game", "AAA", "Sealed")]
    res = datasets.check(str(data_dir), force=True, session=net)
    assert not res.error and res.n_head_failed >= 1
    assert ("AAA", "game", "Sealed") not in {(i.set_code, i.kind, i.event) for i in res.items}
    assert ("AAA", "game", "TradSealed") in {(i.set_code, i.kind, i.event) for i in res.items}


def test_check_cache_dismiss_and_force(data_dir):
    doc, files = _world(data_dir)
    net = FakeNet(doc, files)
    r1 = datasets.check(str(data_dir), session=net)
    n_net = len(net.gets) + len(net.heads)
    assert r1.items and not r1.from_cache and n_net > 2
    r2 = datasets.check(str(data_dir), session=net)                       # within the TTL: no network at all
    assert r2.from_cache and len(net.gets) + len(net.heads) == n_net
    assert [i.url for i in r2.items] == [i.url for i in r1.items]
    st = datasets.load_state()
    assert st["last_check"] and st["published"] and st["files"]
    # don't ask again: this version of the files stays hidden, forced (manual) checks still show them
    datasets.dismiss(r2.items)
    r3 = datasets.check(str(data_dir), session=net)
    assert r3.items == [] and r3.n_dismissed == len(r1.items)
    assert len(datasets.check(str(data_dir), force=True, session=net).items) == len(r1.items)
    # a new upload (other ETag/size) of a dismissed file is announced again
    files[url("game", "AAA", "TradSealed")].update(size=2222, etag="e2")
    r4 = datasets.check(str(data_dir), force=True, session=net)
    datasets.dismiss([i for i in r4.items if i.url != url("game", "AAA", "TradSealed")])
    r5 = datasets.check(str(data_dir), force=True, session=net)
    assert len(r5.items) == len(r4.items)                                  # force shows everything
    r6 = datasets.check(str(data_dir), session=net)
    assert [i.url for i in r6.items] == [url("game", "AAA", "TradSealed")]
    datasets.clear_dismissed()


def test_head_budget_is_respected(data_dir):
    doc, files = _world(data_dir)

    class Slow(FakeNet):
        def head(self, *a, **k):
            time.sleep(0.6)
            return super().head(*a, **k)
    t0 = time.monotonic()
    res = datasets.check(str(data_dir), force=True, session=Slow(doc, files), budget=0.3)
    assert time.monotonic() - t0 < 3
    assert res.n_head_failed >= 1 and any("timed out" in n for n in res.notes)


# ---------------------------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------------------------

def _item(code="AAA", kind="game", event="TradSealed", size=3_000_000, **kw):
    d = dict(url=url(kind, code, event), name=f"{kind}_data_public.{code}.{event}.csv.gz", kind=kind, set_code=code, event=event,
             status="new", size=size, etag="e9", last_modified="2026-10-20T00:00:00+00:00")
    d.update(kw)
    return datasets.DatasetItem(**d)


def test_download_streams_part_file_then_renames(data_dir):
    it = _item(size=2_500_000)
    net = FakeNet(prismic_doc([]), {it.url: {"size": 2_500_000}})
    seen = []
    rep = datasets.download([it], str(data_dir), progress=lambda f, m: seen.append((f, m)), session=net)
    dest = data_dir / "AAA" / it.name
    assert rep.downloaded == [str(dest)] and dest.stat().st_size == 2_500_000 and not list(data_dir.rglob("*.part"))
    assert rep.nbytes == 2_500_000 and not rep.failed and seen[-1][0] == 1.0 and any(0 < f < 1 for f, _ in seen)
    assert datasets.load_state()["files"][it.url]["etag"] == "e9"
    # updating a compressed copy inside the data folder rewrites that same file
    other = data_dir / "elsewhere" / it.name
    other.parent.mkdir()
    other.write_bytes(b"old")
    upd = _item(status="updated", local_path=str(other))
    assert datasets.destination(upd, str(data_dir)) == other
    # a local .csv (uncompressed) is never overwritten: the .csv.gz goes to <data>/<SET>/
    csv = data_dir / "AAA" / "game_data_public.AAA.TradSealed.csv"
    assert datasets.destination(_item(status="updated", local_path=str(csv)), str(data_dir)) == data_dir / "AAA" / it.name


def test_download_failures_leave_no_part_and_keep_going(data_dir):
    bad, good = _item(event="Sealed"), _item(event="TradSealed")
    net = FakeNet(prismic_doc([]), {bad.url: {"size": 1000, "declared": 5000}, good.url: {"size": 1000}})
    rep = datasets.download([bad, good], str(data_dir), session=net)
    assert [n for n, _ in rep.failed] == [bad.name] and "incomplete" in rep.failed[0][1]
    assert not (data_dir / "AAA" / bad.name).exists() and (data_dir / "AAA" / good.name).exists()
    assert not list(data_dir.rglob("*.part"))
    missing = _item(event="TradDraft")
    rep = datasets.download([missing], str(data_dir), session=net)
    assert rep.failed and not rep.downloaded


def test_download_cancel(data_dir):
    a, b = _item(event="Sealed", size=4_000_000), _item(event="TradSealed", size=1000)
    net = FakeNet(prismic_doc([]), {a.url: {"size": 4_000_000}, b.url: {"size": 1000}})
    cancel = threading.Event()
    rep = datasets.download([a, b], str(data_dir), progress=lambda f, m: cancel.set(), cancel=cancel, session=net)
    assert rep.cancelled and not rep.downloaded
    assert not list(data_dir.rglob("*.part")) and not (data_dir / "AAA" / a.name).exists()


def test_api_download_registers_and_marks_sets_stale(data_dir, home, monkeypatch):
    it = _item(code="AAA", event="TradSealed", size=0)
    # a real (tiny) game file as the body
    p = write_game_gz(data_dir / "src.csv.gz")
    body = p.read_bytes()
    it.size = len(body)
    net = FakeNet(prismic_doc([]), {it.url: {"size": len(body), "body": body}})
    monkeypatch.setattr(datasets, "_session", lambda: net)
    rep = api.download_datasets([it.to_dict()], str(data_dir))
    dest = data_dir / "AAA" / it.name
    assert rep.downloaded == [str(dest)] and [f.kind for f in rep.registered] == ["game"]
    assert api.set_status("AAA").set_code == "AAA" and [f.event for f in registry.files("AAA", include_shared=False)] == ["TradSealed"]
    assert rep.stale_sets == ["AAA"] and "Calculate" in rep.message
    # "on disk, not loaded" items are registered without any download
    reg_only = _item(code="AAA", event="Sealed", status="unregistered", local_path=str(write_game_gz(data_dir / "AAA" / "game_data_public.AAA.Sealed.csv.gz", "AAA", "Sealed")))
    net.gets.clear()
    rep2 = api.download_datasets([reg_only], str(data_dir))
    assert not net.gets and not rep2.downloaded and len(rep2.registered) == 1
    assert sorted(f.event for f in registry.files("AAA", include_shared=False)) == ["Sealed", "TradSealed"]


def test_api_check_wrapper_does_not_raise_offline(data_dir, monkeypatch):
    net = FakeNet(prismic_doc([]))
    net.offline = True
    monkeypatch.setattr(datasets, "_session", lambda: net)
    res = api.check_new_datasets(str(data_dir), force=True)
    assert res.error and res.items == []


# ---------------------------------------------------------------------------------------------
# set detection
# ---------------------------------------------------------------------------------------------

def _idx(names: list[str]) -> NameIndex:
    return NameIndex(pl.DataFrame({"card_id": list(range(len(names))), "name": names, "is_basic": [False] * len(names)}))


A = [f"Alpha Card {i}" for i in range(60)]
B = [f"Beta Card {i}" for i in range(60)] + A[:3]            # shares 3 names with A (reprints)
INDEXES = {"AAA": _idx(A), "BBB": _idx(B)}


def _pool(names, suffix=None, n=40):
    return "\n".join(f"1 {x}" + (f" ({suffix}) {i + 1}" if suffix else "") for i, x in enumerate(names[:n]))


def test_detect_by_suffix_majority_and_case():
    g = setdetect.detect(_pool(A, "aaa"), INDEXES)
    assert (g.set_code, g.source, g.matched, g.total) == ("AAA", "suffix", 40, 40) and g.message == "Detected AAA (40/40 cards matched)"
    # basic lands (Arena writes the newest set's code on them) and special printings never vote
    text = _pool(A, "AAA") + "\n8 Forest (FDN) 280\n1 Alpha Card 41 (SPG) 7\n1 Alpha Card 42 (SPG) 8"
    assert setdetect.detect(text, INDEXES).source == "suffix"
    # a wrong suffix on a minority of lines does not matter
    mixed = _pool(A, "AAA", 30) + "\n" + "\n".join(f"1 Alpha Card {i} (BBB) 1" for i in range(30, 36))
    assert setdetect.detect(mixed, INDEXES).set_code == "AAA"
    # the Arena alias of Dominaria
    assert setdetect.detect(_pool(["Foo"] * 0 + A, "DAR"), {"DOM": _idx(A)}).set_code == "DOM"


def test_detect_suffix_for_a_set_the_app_does_not_have():
    g = setdetect.detect(_pool(A, "FDN"), INDEXES)
    assert g.set_code is None and g.detected_code == "FDN" and g.source == "suffix" and "no data" in g.message


def test_detect_suffix_that_does_not_match_the_names_falls_back_to_names():
    text = _pool(B[:40], "AAA")            # claims AAA but the cards are BBB's
    g = setdetect.detect(text, INDEXES)
    assert g.set_code == "BBB" and g.source == "names"


def test_detect_by_names_with_margin_and_runner_up():
    g = setdetect.detect(_pool(B[3:], None, 50) + "\n4 Plains", INDEXES)     # basics are ignored
    assert (g.set_code, g.source, g.matched, g.total) == ("BBB", "names", 50, 50)
    assert g.runner_up == "AAA" and g.runner_up_matched == 0
    assert setdetect.detect(_pool(A), INDEXES).set_code == "AAA"
    # reprints shared by two sets do not confuse a clear pool
    g = setdetect.detect(_pool(A[:40]) , INDEXES)
    assert g.set_code == "AAA" and g.runner_up == "BBB" and g.runner_up_matched == 3


def test_detect_ambiguous_and_unrecognised():
    half = "\n".join(f"1 {x}" for x in A[3:23] + B[:20])
    g = setdetect.detect(half, INDEXES)
    assert g.set_code is None and g.ambiguous and "Could not tell" in g.message
    low = "\n".join(f"1 {x}" for x in A[3:20]) + "\n" + "\n".join(f"1 Unknown {i}" for i in range(40))
    g = setdetect.detect(low, INDEXES)
    assert g.set_code is None and g.ambiguous                     # < 70% matched
    g = setdetect.detect("1 Nothing Here\n1 Nor Here", INDEXES)
    assert g.set_code is None and "No card" in g.message
    # a lead of one card is not "clearly ahead"
    twin = {"AAA": _idx(A), "BBB": _idx(A[:-1] + ["Other"])}
    assert setdetect.detect(_pool(A, None, 30), twin).set_code is None


def test_detect_no_models_and_empty_text():
    g = setdetect.detect(_pool(A), {})
    assert g.set_code is None and "No set with a card table" in g.message
    g = setdetect.detect("   \n", INDEXES)
    assert g.set_code is None and g.total == 0 and g.source == "none"
    assert setdetect.detect("", INDEXES).message == ""


def test_api_detect_set_without_any_set(home):
    r = api.detect_set("1 Shock (M21) 159\n1 Giant Growth")
    assert r["set_code"] is None and r["total"] == 2


def test_api_detect_set_against_a_real_card_table(built):
    names = [v["name"] for v in api.card_info("TST").values() if not v["is_basic"] and not v["is_land"]][:30]
    r = api.detect_set("\n".join(f"1 {n}" for n in names) + "\n5 Forest")
    assert r["set_code"] == "TST" and r["matched"] == r["total"] == 30 and r["source"] == "names"
    r = api.detect_set("\n".join(f"1 {n} (tst) {i}" for i, n in enumerate(names)))
    assert r["set_code"] == "TST" and r["source"] == "suffix"


# ---------------------------------------------------------------------------------------------
# file host move (S3 -> data.17landscapes.com), JSON API listing, state migration
# ---------------------------------------------------------------------------------------------

OLD = "https://17lands-public.s3.amazonaws.com/analysis_data"


def _api_row(code, event, date="2026-10-17", **urls):
    return {"expansion": code, "event_type": event, "last_updated": date,
            "urls": {k: f"{S3}/{k}_data/{k}_data_public.{code}.{event}.csv.gz" for k in (urls or {"game": 1, "draft": 1})}}


def test_canonical_url():
    assert datasets.DOWNLOAD_HOSTS == (datasets.DOWNLOAD_HOST, datasets.OLD_DOWNLOAD_HOST)
    assert datasets.canonical_url(OLD + "/game_data/x.csv.gz") == S3 + "/game_data/x.csv.gz"
    assert datasets.canonical_url(S3 + "/game_data/x.csv.gz") == S3 + "/game_data/x.csv.gz"
    for other in ("https://17lands-public.s3.amazonaws.com.evil.example/a", "http://17lands-public.s3.amazonaws.com/a",
                  "https://evil.example/https://17lands-public.s3.amazonaws.com/a", "", None, 5):
        assert datasets.canonical_url(other) == other


def test_api_listing_parses_rows_and_skips_odd_ones():
    rows = [_api_row("AAA", "PremierDraft"),
            _api_row("AAA", "Sealed", game=1, replay=1, extra=1),           # unknown key "extra": its file name is not a dataset
            {"expansion": "BBB", "event_type": "Sealed", "last_updated": "2026-01-02"},   # no urls
            {"expansion": "BBB", "event_type": "Sealed", "urls": None},
            "junk", None,
            {"expansion": "CCC", "event_type": "Sealed", "urls": {"game": "https://evil.example/game_data_public.CCC.Sealed.csv.gz",
                                                                   "draft": 7, "replay": f"{S3}/replay_data/notes.txt"}},
            {"expansion": "DDD", "event_type": "Sealed", "urls": {"game": f"{S3}/game_data/game_data_public.D:.Sealed.csv.gz"}},
            {"expansion": "EEE", "event_type": "Sealed", "last_updated": "2026-02-03",
             "urls": {"game": f"{OLD}/game_data/game_data_public.EEE.Sealed.csv.gz"}}]       # an old-host link is canonicalised
    lst = datasets.parse_api_listing(rows)
    got = {(f.set_code, f.kind, f.event): f for f in lst.files}
    assert set(got) == {("AAA", "game", "PremierDraft"), ("AAA", "draft", "PremierDraft"), ("AAA", "game", "Sealed"),
                        ("AAA", "replay", "Sealed"), ("EEE", "game", "Sealed")}
    assert got[("AAA", "game", "PremierDraft")].row_date == "2026-10-17"
    assert got[("EEE", "game", "Sealed")].url == f"{S3}/game_data/game_data_public.EEE.Sealed.csv.gz"
    assert all(f.url.startswith(datasets.DOWNLOAD_HOST) for f in lst.files)
    with pytest.raises(ValueError):
        datasets.parse_api_listing({"not": "a list"})


def test_list_remote_prefers_the_api_and_falls_back_to_prismic():
    net = FakeNet(prismic_doc([_row("PRI", "Sealed", "2026-10-17")]))
    net.api = [_api_row("API", "Sealed")]
    lst = datasets.list_remote(net)
    assert {f.set_code for f in lst.files} == {"API"} and net.gets == [datasets.LISTING_API]
    for bad in ([], [{"urls": {"game": "https://evil.example/x"}}], {"x": 1}):    # nothing valid / not a list
        net = FakeNet(prismic_doc([_row("PRI", "Sealed", "2026-10-17")]))
        net.api = bad
        assert {f.set_code for f in datasets.list_remote(net).files} == {"PRI"}
    net = FakeNet(prismic_doc([_row("PRI", "Sealed", "2026-10-17")]))            # non-200
    assert {f.set_code for f in datasets.list_remote(net).files} == {"PRI"}
    net.offline = True                                                           # both fail: raises as before
    with pytest.raises(Exception):
        datasets.list_remote(net)


def test_prismic_fallback_canonicalises_old_host_links():
    old_doc = prismic_doc([_row("OLD", "Sealed", "2026-10-17", draft=False)])
    raw = json.dumps(old_doc).replace("data.17landscapes.com", "17lands-public.s3.amazonaws.com")
    assert "17lands-public.s3.amazonaws.com" in raw
    lst = datasets.parse_listing(json.loads(raw))
    assert lst.files and all(f.url.startswith(datasets.DOWNLOAD_HOST) for f in lst.files)
    assert len({f.url for f in lst.files}) == len(lst.files)


def test_state_keys_migrate_to_the_new_host_keeping_download_info(home):
    f_old, f_new = OLD + "/game_data/a.csv.gz", S3 + "/game_data/a.csv.gz"
    g_old = OLD + "/game_data/b.csv.gz"
    g_new = S3 + "/game_data/b.csv.gz"
    state = {"files": {f_old: {"etag": "e1", "dl_etag": "d1", "dl_size": 10, "downloaded_at": 100.0, "seen_at": 100.0},
                       g_old: {"dl_etag": "OLDER", "downloaded_at": 50.0, "seen_at": 50.0},
                       g_new: {"dl_etag": "NEWER", "downloaded_at": 80.0, "seen_at": 80.0}},
             "dismissed": {f_old: "sig"},
             "remote": [{"url": f_old, "name": "a.csv.gz", "kind": "game", "set_code": "A", "event": "Sealed", "row_date": ""},
                        {"url": f_new, "name": "a.csv.gz", "kind": "game", "set_code": "A", "event": "Sealed", "row_date": ""}]}
    datasets.save_state(state)
    st = datasets.load_state()
    assert set(st["files"]) == {f_new, g_new}
    assert st["files"][f_new]["dl_etag"] == "d1" and st["files"][f_new]["dl_size"] == 10
    assert st["files"][g_new]["dl_etag"] == "NEWER"                  # both keys: the later download wins
    assert st["dismissed"] == {f_new: "sig"}
    assert [r["url"] for r in st["remote"]] == [f_new]
    state["files"][g_old]["downloaded_at"] = 500.0                    # ... whichever of the two it is
    datasets.save_state(state)
    assert datasets.load_state()["files"][g_new]["dl_etag"] == "OLDER"


def test_stale_state_dl_etag_still_flags_a_changed_server_file(data_dir):
    """A user who downloaded from the old host and kept the old state: no NEW on a current file, UPDATED on a changed ETag."""
    doc, files = _world(data_dir)
    key = url("game", "AAA", "TradSealed")
    local = write_game_gz(data_dir / "AAA" / "game_data_public.AAA.TradSealed.csv.gz")
    size = local.stat().st_size
    files[key] = {"size": size, "etag": "e1"}
    datasets.save_state({"files": {key.replace(S3, OLD): {"dl_etag": "e0", "dl_size": size, "downloaded_at": 1.0, "seen_at": 1.0}}})
    r = datasets.check(str(data_dir), force=True, session=FakeNet(doc, files))
    it = [i for i in r.items if i.url == key]
    assert it and it[0].status == "updated" and "changed since you downloaded" in it[0].reason
    files[key]["etag"] = "e0"
    r = datasets.check(str(data_dir), force=True, session=FakeNet(doc, files))
    assert not [i for i in r.items if i.url == key]


def test_cached_listing_with_old_host_urls_downloads_from_the_new_host(data_dir):
    key = url("game", "AAA", "TradSealed")
    rf = {"url": key.replace(S3, OLD), "name": "game_data_public.AAA.TradSealed.csv.gz", "kind": "game", "set_code": "AAA",
          "event": "TradSealed", "row_date": "2026-10-17"}
    datasets.save_state({"remote": [rf], "listing_at": time.time(), "published": "x"})
    net = FakeNet(prismic_doc([]), {key: {"size": 1000, "etag": "e1"}})
    r = datasets.check(str(data_dir), session=net)
    assert r.from_cache and [i.url for i in r.items] == [key]
    assert not any("amazonaws" in u for u in net.heads)
    rep = datasets.download(r.items, str(data_dir), session=net)
    assert rep.downloaded and not rep.failed and net.gets == [key]
    assert datasets.load_state()["files"][key]["dl_etag"] == "e1"
    # an item still carrying the old-host URL (stale GUI list) is fetched from the new host too
    net2 = FakeNet(prismic_doc([]), {key: {"size": 1000}})
    stale = _item(url=key.replace(S3, OLD), size=1000)
    datasets.download_one(stale, data_dir / "x" / stale.name, session=net2)
    assert net2.gets == [key]


def test_download_one_refuses_lookalike_hosts(data_dir):
    name = "game_data_public.AAA.TradSealed.csv.gz"
    for bad in (f"https://data.17landscapes.com.evil.example/analysis_data/game_data/{name}",
                f"https://data.17landscapes.com@evil.example/{name}", f"http://data.17landscapes.com/analysis_data/{name}",
                f"https://17lands-public.s3.amazonaws.com.evil.example/analysis_data/{name}",
                f"https://evil.example/https://data.17landscapes.com/{name}", f"https://xdata.17landscapes.com/{name}"):
        net = FakeNet(prismic_doc([]), {bad: {"size": 10}})
        with pytest.raises(ValueError):
            datasets.download_one(_item(url=bad), data_dir / "o.csv.gz", session=net)
        assert net.gets == []
