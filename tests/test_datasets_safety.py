"""17lands dataset downloads only come from 17lands' file host and only ever land inside the data folder, whatever the (CMS-made)
listing says."""
from __future__ import annotations

from pathlib import Path

import pytest

from sealed_lab.ingest import datasets as D

HOST = D.DOWNLOAD_HOST + "analysis_data/game_data/"


def test_listing_keeps_only_plain_files_on_the_17lands_host():
    ok = D._remote_file(HOST + "game_data_public.TST.PremierDraft.csv.gz")
    assert ok is not None and ok.set_code == "TST" and ok.event == "PremierDraft" and ok.name == "game_data_public.TST.PremierDraft.csv.gz"
    assert D._remote_file(HOST + "game_data_public.Y25_EOE.Pick-Two.csv") is not None
    for bad in ("https://evil.example/game_data_public.TST.Sealed.csv.gz",                 # another host
                "http://17lands-public.s3.amazonaws.com/game_data_public.TST.Sealed.csv.gz",  # not https
                "https://17lands-public.s3.amazonaws.com.evil.example/game_data_public.TST.Sealed.csv.gz",
                "http://data.17landscapes.com/game_data_public.TST.Sealed.csv.gz",                # not https
                "https://data.17landscapes.com.evil.example/game_data_public.TST.Sealed.csv.gz",  # lookalikes
                "https://data.17landscapes.com.evil/game_data_public.TST.Sealed.csv.gz",
                "https://xdata.17landscapes.com/game_data_public.TST.Sealed.csv.gz",
                "https://data.17landscapes.com@evil.example/game_data_public.TST.Sealed.csv.gz",
                "https://evil.example/https://data.17landscapes.com/game_data_public.TST.Sealed.csv.gz",
                HOST + "game_data_public.C:.Sealed.csv.gz",                                # a drive letter
                HOST + "game_data_public.a\\..\\b.Sealed.csv.gz",                          # a separator
                HOST + "game_data_public.TST.Sealed.csv.gz?x=1",
                HOST + "game_data_public..Sealed.csv.gz",
                HOST + "notes.txt"):
        assert D._remote_file(bad) is None, bad


def test_old_host_links_are_rewritten_to_the_new_host():
    old = "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.TST.Sealed.csv.gz"
    rf = D._remote_file(old)
    assert rf is not None and rf.url == HOST + "game_data_public.TST.Sealed.csv.gz"
    assert D._remote_file(old + "?x=1") is None


def test_parse_listing_ignores_foreign_links():
    doc = {"results": [{"data": {"datasets": [{"game_data": [{"spans": [
        {"type": "hyperlink", "data": {"url": "https://evil.example/game_data_public.TST.Sealed.csv.gz"}},
        {"type": "hyperlink", "data": {"url": HOST + "game_data_public.TST.Sealed.csv.gz"}}]}]}]}}]}
    files = D.parse_listing(doc).files
    assert [f.url for f in files] == [HOST + "game_data_public.TST.Sealed.csv.gz"]


def _item(**kw) -> D.DatasetItem:
    base = dict(url=HOST + "game_data_public.TST.Sealed.csv.gz", name="game_data_public.TST.Sealed.csv.gz", kind="game", set_code="TST",
                event="Sealed", status="new")
    base.update(kw)
    return D.DatasetItem(**base)


def test_destination_never_leaves_the_data_folder(tmp_path):
    assert D.destination(_item(), str(tmp_path)) == tmp_path / "TST" / "game_data_public.TST.Sealed.csv.gz"
    for kw in (dict(set_code="..", name="game_data_public.TST.Sealed.csv.gz"), dict(set_code="C:"), dict(name="../../evil.csv.gz"),
               dict(set_code="T/S")):
        with pytest.raises(ValueError):
            D.destination(_item(**kw), str(tmp_path))


def test_download_one_refuses_another_host(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        D.download_one(_item(url="https://evil.example/game_data_public.TST.Sealed.csv.gz"), tmp_path / "x.csv.gz", session=object())
    assert not any(Path(tmp_path).iterdir())


def test_download_one_refuses_cross_host_redirect_and_oversized_chunked_body(tmp_path):
    class RedirectResp:
        url = "https://evil.example/hijacked.csv.gz"
        status_code = 200
        headers = {"Content-Length": "4"}

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield b"evil"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class RedirectSess:
        def get(self, url, **kw):
            return RedirectResp()

        def head(self, url, **kw):
            return RedirectResp()

    with pytest.raises(ValueError, match="Refusing redirect"):
        D.download_one(_item(), tmp_path / "x.csv.gz", session=RedirectSess())
    assert not any(Path(tmp_path).iterdir())
    assert D.head(HOST + "game_data_public.TST.Sealed.csv.gz", session=RedirectSess()) is None

    # Chunked download (no Content-Length) exceeding max_bytes is aborted and .part deleted
    class BigChunkedResp:
        url = HOST + "game_data_public.TST.Sealed.csv.gz"
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            for _ in range(5):
                yield b"x" * 100

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class BigSess:
        def get(self, url, **kw):
            return BigChunkedResp()

    with pytest.raises(ValueError, match="byte limit"):
        D.download_one(_item(), tmp_path / "y.csv.gz", session=BigSess(), max_bytes=250)
    assert not any(Path(tmp_path).iterdir())


def test_check_handles_underscore_set_codes_without_crashing(tmp_path, home):
    # Y25_EOE passes _URL_FILE_RE but is not a plain valid_set_code ([A-Z0-9]{2,8}): check() must not crash at model_dir()
    u = HOST + "game_data_public.Y25_EOE.Sealed.csv.gz"
    api_rows = [{"expansion": "Y25_EOE", "event_type": "Sealed", "last_updated": "2026-10-20", "urls": {"game": u}}]

    class Sess:
        def get(self, url, **kw):
            class R:
                status_code = 200
                headers = {}
                def raise_for_status(self): pass
                def json(self): return api_rows
                def iter_content(self, n): return iter(())
            return R()

        def head(self, url, **kw):
            class R:
                url = u
                status_code = 200
                headers = {"Content-Length": "123", "ETag": '"e1"', "Last-Modified": "Mon, 27 Jul 2026 18:11:23 GMT"}
            return R()

    res = D.check(str(tmp_path), force=True, session=Sess())
    assert not res.error and len(res.items) == 1 and res.items[0].set_code == "Y25_EOE" and not res.items[0].set_has_model
