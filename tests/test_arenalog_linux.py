"""Finding Arena's Player.log on Linux (Steam/Proton, extra Steam libraries, Lutris/Wine): pure path logic, runs on every OS."""
from __future__ import annotations

import os
from pathlib import Path

from sealed_lab.deck import arenalog

REL = Path("AppData/LocalLow/Wizards Of The Coast/MTGA")


def steam_prefix(library: Path, app_id: str = "2141910") -> Path:
    return library / "steamapps" / "compatdata" / app_id / "pfx" / "drive_c" / "users" / "steamuser" / REL


def make_log(folder: Path, mtime: float, name: str = "Player.log") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text("x", encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def world(tmp_path: Path) -> dict[str, Path]:
    home = tmp_path / "home"
    lib2 = tmp_path / "bigdisk" / "SteamLibrary"
    steam = home / ".local" / "share" / "Steam"
    (steam / "steamapps").mkdir(parents=True)
    (steam / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n\t"1"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n' % (steam, lib2), encoding="utf-8")
    return {"home": home, "steam": steam, "lib2": lib2, "lutris": home / "Games" / "arena" / "drive_c" / "users" / "me" / REL,
            "default": steam_prefix(home / ".steam" / "steam")}


def use_linux(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(arenalog.sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("SEALED_LAB_ARENA_LOG_DIR", raising=False)
    monkeypatch.setattr(arenalog, "_LINUX_CACHE", {})              # the candidates are cached for a minute


def test_candidates_cover_steam_libraries_and_wine_prefixes(tmp_path):
    w = world(tmp_path)
    in_steam = steam_prefix(w["steam"])
    in_lib2 = steam_prefix(w["lib2"], "99999")                 # another app id: nothing is hard-coded
    for d in (in_steam, in_lib2, w["lutris"]):
        d.mkdir(parents=True)
    cands = arenalog.linux_log_dirs(w["home"])
    assert cands[0] == w["default"]                              # the default Steam folder comes first, even when absent
    assert in_steam in cands and in_lib2 in cands and w["lutris"] in cands
    assert len({str(c.resolve()) for c in cands}) == len(cands)  # de-duplicated


def test_newest_player_log_wins(tmp_path, monkeypatch):
    w = world(tmp_path)
    make_log(steam_prefix(w["steam"]), 1_000_000)
    mid = make_log(w["lutris"], 2_000_000).parent
    new = make_log(steam_prefix(w["lib2"], "42"), 3_000_000, "Player-prev.log").parent
    use_linux(monkeypatch, w["home"])
    assert arenalog.default_log_dir() == new
    os.utime(w["lutris"] / "Player.log", (4_000_000, 4_000_000))
    assert arenalog.default_log_dir() == mid


def test_nothing_present_gives_the_first_candidate(tmp_path, monkeypatch):
    w = world(tmp_path)
    use_linux(monkeypatch, w["home"])
    assert arenalog.default_log_dir() == w["default"]
    use_linux(monkeypatch, tmp_path / "empty_home")
    assert arenalog.default_log_dir() == steam_prefix(tmp_path / "empty_home" / ".steam" / "steam")


def test_env_override_wins(tmp_path, monkeypatch):
    w = world(tmp_path)
    make_log(w["lutris"], 2_000_000)
    use_linux(monkeypatch, w["home"])
    monkeypatch.setenv("SEALED_LAB_ARENA_LOG_DIR", str(tmp_path / "mine"))
    assert arenalog.default_log_dir() == tmp_path / "mine"


def test_windows_path_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(arenalog.sys, "platform", "win32")
    monkeypatch.delenv("SEALED_LAB_ARENA_LOG_DIR", raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert arenalog.default_log_dir() == tmp_path / "AppData" / "LocalLow" / "Wizards Of The Coast" / "MTGA"
