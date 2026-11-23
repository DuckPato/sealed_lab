"""Set release dates and the newest-first set order."""
from sealed_lab.ingest import setinfo


def test_order_sets_newest_first_undated_last():
    dates = {"AFR": "2021-07-23", "BLB": "2024-08-02", "DFT": "2025-02-14"}
    assert setinfo.order_sets(["AFR", "ZZZ", "DFT", "AAA", "BLB"], dates) == ["DFT", "BLB", "AFR", "AAA", "ZZZ"]


def test_release_dates_cached_and_offline_safe(home, monkeypatch):
    calls = []

    def fake_fetch():
        calls.append(1)
        return {"AFR": "2021-07-23", "BLB": "2024-08-02"}
    monkeypatch.setattr(setinfo, "_fetch", fake_fetch)
    assert setinfo.release_dates(["afr", "BLB", "XYZ"]) == {"AFR": "2021-07-23", "BLB": "2024-08-02"}
    assert setinfo.release_dates(["AFR", "XYZ"]) == {"AFR": "2021-07-23"} and len(calls) == 1     # unknown XYZ: no refetch within a day
    assert setinfo.release_dates(["BLB"], fetch=False) == {"BLB": "2024-08-02"}
    monkeypatch.setattr(setinfo, "_fetch", lambda: None)                                        # offline: keeps the cached dates
    monkeypatch.setattr(setinfo, "REFETCH_AFTER", -1)
    assert setinfo.release_dates(["AFR", "XYZ"]) == {"AFR": "2021-07-23"}


def test_untrack_set_removes_app_data_keeps_files(synth_dir, home):
    import pytest

    from sealed_lab import api, config
    from sealed_lab.ingest import registry
    from sealed_lab.types import SetBusy

    api.add_files([str(synth_dir)])
    assert "TST" in registry.known_sets() and config.set_dir("TST").exists()
    with api._calculating("TST"), pytest.raises(SetBusy):
        api.untrack_set("TST")
    assert "TST" in registry.known_sets()
    api.untrack_set("TST")
    assert "TST" not in registry.known_sets() and not config.set_dir("TST").exists()
    assert any(synth_dir.glob("game_data_public.TST.*.csv.gz"))
    api.add_files([str(synth_dir)])                                       # it can be tracked again
    assert "TST" in registry.known_sets()
