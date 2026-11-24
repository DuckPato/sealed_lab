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
