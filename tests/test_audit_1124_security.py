"""Audit 1124, section D (security / distribution): set codes as paths, release signing order and zip comparison, pull-request import,
the export scan of images, staged file modes, host and size limits for Scryfall downloads, the update marker, the legacy token."""
from __future__ import annotations

import importlib.util
import os
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

import pytest

from sealed_lab import config

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


ex = _load("export_public")
ip = _load("import_pr")


# ------------------------------------------------------------------------------------------------ D1
@pytest.mark.parametrize("code", ["TST", "tst", "OM1", "Y25", "ECL", "40K", " tdm "])
def test_real_set_codes_are_accepted(code, tmp_path, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path))
    assert config.set_dir(code) == tmp_path / "sets" / code.strip().upper()
    assert config.corpus_dir(code).parent == config.set_dir(code)


@pytest.mark.parametrize("code", ["..", "../x", "..\\..\\evil", "C:", "C:\\x", "/abs", "a/b", "A.B", "", "X", "TOOLONGCODE", "TS T", "TS\nT","ÄÖ1", "T\0S"])
def test_path_like_set_codes_are_refused(code, tmp_path, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path))
    for fn in (config.set_dir, config.corpus_dir, config.model_dir, config.cache_dir, config.ensure_set_dirs):
        with pytest.raises(ValueError):
            fn(code)
    assert config.valid_set_code(code) is None


def test_detect_treats_a_hostile_expansion_as_unknown_set(tmp_path):
    from sealed_lab.ingest import detect

    p = tmp_path / "export.csv"
    hdr = "expansion,event_type,draft_id,won,deck_Bolt,sideboard_Bolt,opening_hand,user_turn_1_cards_drawn,oppo_turn_1_cards_drawn\n"
    p.write_text(hdr + "../../evil,PremierDraft,abc,True,1,0,1|2,1,1\n", encoding="utf-8")
    info = detect.detect_file(p)
    assert info.kind in ("game", "replay") and info.set_code is None
    assert "not a valid set code" in info.note
    ok = tmp_path / "ok.csv"
    ok.write_text(hdr + "tst,PremierDraft,abc,True,1,0,1|2,1,1\n", encoding="utf-8")
    assert detect.detect_file(ok).set_code == "TST"


def test_detect_ignores_a_hostile_scryfall_set(tmp_path):
    from sealed_lab.ingest import detect

    for name, code in (("bad.json", "..\\x"), ("good.json", "tst")):
        (tmp_path / name).write_text('[{"name": "A", "type_line": "Creature", "set": "%s"}]' % code.replace("\\", "\\\\"), encoding="utf-8")
    assert detect.detect_file(tmp_path / "bad.json").set_code is None
    assert detect.detect_file(tmp_path / "good.json").set_code == "TST"


def test_hostile_cards_csv_set_is_not_a_single_set(tmp_path):
    from sealed_lab.ingest import detect

    p = tmp_path / "cards.csv"
    p.write_text("name,expansion\nA,..\\..\\x\n", encoding="utf-8")
    assert detect._cards_single_set(p) is None


# ------------------------------------------------------------------------------------------------ D2
def _zip(path: Path, entries: list[tuple[str, int, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, mode, data in entries:
            i = zipfile.ZipInfo(name)
            i.external_attr = mode << 16
            z.writestr(i, data)
    return path


def test_compare_zips_rejects_duplicates_and_checks_directories(tmp_path):
    rs = _load("release_sign")
    base = [("a/", 0o40755, b""), ("a/x.py", 0o100644, b"1")]
    ref = _zip(tmp_path / "ref.zip", base)
    assert rs.compare_zips(_zip(tmp_path / "same.zip", base), ref) == []
    dup = _zip(tmp_path / "dup.zip", base + [("a/x.py", 0o100644, b"2")])
    assert any("duplicate entry in the release zip: a/x.py" in d for d in rs.compare_zips(dup, ref))
    assert any("duplicate entry in the reference zip" in d for d in rs.compare_zips(ref, dup))
    nodir = _zip(tmp_path / "nodir.zip", [base[1]])
    assert rs.compare_zips(nodir, ref) == ["directory missing from the release zip: a/"]
    extra = _zip(tmp_path / "extra.zip", base + [("b/", 0o40755, b"")])
    assert rs.compare_zips(extra, ref) == ["directory only in the release zip: b/"]
    mode = _zip(tmp_path / "mode.zip", [("a/", 0o40700, b""), base[1]])
    assert rs.compare_zips(mode, ref) == ["directory mode differs: a/"]


def test_read_trusted_reads_data_without_running_the_file(tmp_path):
    rs = _load("release_sign")
    f = tmp_path / "release_keys.py"
    f.write_text('open("ran.txt", "w").write("x")\nTRUSTED_KEYS: dict[str, str] = {"k1": "v1"}\n', encoding="utf-8")
    assert rs.read_trusted(f) == {"k1": "v1"}
    assert not Path("ran.txt").exists()


def test_the_reference_zip_comes_from_this_repos_builder_with_the_tag_as_root(tmp_path):
    """export_public's functions take the source root: the tag checkout is only read, none of its code runs."""
    src = tmp_path / "src"
    for rel, text in (("sealed_lab/__init__.py", '__version__ = "9.9.9"\n'), ("pyproject.toml", "x\n"), ("apply_update.sh", "#!/bin/sh\n"),
                      ("tools/evil.py", "raise SystemExit(7)\n")):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text, encoding="utf-8", newline="\n")
    rels = ex.selected(["sealed_lab/__init__.py", "pyproject.toml", "apply_update.sh", "tools/evil.py"], root=src)
    z = ex.build_zip(tmp_path / "out", rels, "v9.9.9", root=src)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert "sealed_lab-9.9.9/sealed_lab/__init__.py" in names and not [n for n in names if "evil" in n]
        assert (zf.getinfo("sealed_lab-9.9.9/apply_update.sh").external_attr >> 16) == 0o100755


def test_sign_loads_the_key_only_after_the_comparison():
    src = (ROOT / "tools" / "release_sign.py").read_text(encoding="utf-8")
    body = src[src.index("def cmd_sign"):src.index("def cmd_verify")]
    assert body.index("compare_zips(") < body.index("load_private(")
    assert "tools/export_public.py" not in body.replace("# ", "")


# ------------------------------------------------------------------------------------------------ D3
def test_changed_paths_reads_rename_copy_and_file_headers():
    patch = ("diff --git a/old.txt b/new.txt\nsimilarity index 100%\nrename from old.txt\nrename to tools/evil.py\n"
             "diff --git a/x b/y\ncopy from x\ncopy to .github/workflows/w.yml\n--- a/p.py\n+++ b/q.py\n--- /dev/null\n+++ b/n.sh\n")
    got = ip.changed_paths(patch)
    for p in ("old.txt", "new.txt", "tools/evil.py", ".github/workflows/w.yml", "p.py", "q.py", "n.sh"):
        assert p in got
    assert "/dev/null" not in got


def test_git_attributes_lfsconfig_and_submodules_are_sensitive():
    _, warn = ip.review_notes([".gitattributes", "sealed_lab/.gitattributes", ".lfsconfig", "a/.lfsconfig", ".gitmodules"])
    assert len([w for w in warn if "read every line" in w]) == 5


def _git(cwd, *args, env=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=env).stdout


def test_git_runs_with_lfs_smudging_off(monkeypatch, tmp_path):
    seen = {}
    real = subprocess.run

    def spy(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw.get("env") or {}
        return real(cmd, **kw)

    monkeypatch.setattr(ip.subprocess, "run", spy)
    ip._git(["--version"], tmp_path)
    assert ip.NO_LFS == ["-c", "filter.lfs.smudge=", "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false"]
    assert seen["cmd"][:7] == ["git", *ip.NO_LFS] and seen["env"]["GIT_LFS_SKIP_SMUDGE"] == "1"


def test_a_rename_into_a_renamed_file_is_refused_after_applying(tmp_path, monkeypatch):
    """The patch headers are only read first; what git really changed decides (here a rename onto the private-name file)."""
    for k, v in (("GIT_AUTHOR_NAME", "O"), ("GIT_AUTHOR_EMAIL", "o@example.org"), ("GIT_COMMITTER_NAME", "O"), ("GIT_COMMITTER_EMAIL", "o@example.org")):
        monkeypatch.setenv(k, v)
    dev = tmp_path / "dev"
    dev.mkdir()
    _git(dev, "init", "-q", "-b", "main")
    (dev / "app").mkdir()
    (dev / "app" / "a.txt").write_text("a\n", encoding="utf-8", newline="\n")
    (dev / "outside.txt").write_text("o\n", encoding="utf-8", newline="\n")
    _git(dev, "add", "-A")
    _git(dev, "commit", "-q", "-m", "base")
    base = _git(dev, "rev-parse", "HEAD").strip()
    wt = tmp_path / "wt"
    _git(dev, "worktree", "add", "-b", "tmp", str(wt), base)
    (wt / "app" / "b.txt").write_text("b\n", encoding="utf-8", newline="\n")
    (wt / "outside.txt").write_text("changed\n", encoding="utf-8", newline="\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "x")
    assert ip.actual_paths(wt, base, "tmp", "app") == ["b.txt", "/outside.txt"]
    assert ip.review_notes(ip.actual_paths(wt, base, "tmp", "app"))[0]


# ------------------------------------------------------------------------------------------------ D4
def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _png(*extra: tuple[bytes, bytes], trailing: bytes = b"") -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return (ex.PNG_SIG + _chunk(b"IHDR", ihdr) + b"".join(_chunk(k, b) for k, b in extra) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"") + trailing)


BOB = "C:" + "\\Users\\" + "bob\\Pictures"


def _scan_file(tmp_path, name: str, data: bytes) -> list[str]:
    (tmp_path / name).write_bytes(data)
    return ex.scan([name], root=tmp_path, patterns=ex.deny_patterns())


@pytest.mark.parametrize("make", [
    lambda: _png((b"tEXt", b"Source\0" + BOB.encode())),
    lambda: _png((b"zTXt", b"Source\0\0" + zlib.compress(BOB.encode()))),
    lambda: _png((b"iTXt", b"Source\0\1\0\0\0" + zlib.compress(BOB.encode()))),
    lambda: _png((b"iTXt", b"Source\0\0\0\0\0" + BOB.encode())),
    lambda: _png((b"eXIf", b"II*\0" + BOB.encode("utf-16-le"))),
    lambda: _png(trailing=BOB.encode()),
])
def test_png_metadata_is_scanned(tmp_path, make):
    hits = _scan_file(tmp_path, "a.png", make())
    assert hits and "bob" in hits[0]


def test_clean_png_and_placeholder_png_pass(tmp_path):
    assert _scan_file(tmp_path, "a.png", _png()) == []
    assert _scan_file(tmp_path, "b.png", _png((b"tEXt", b"Source\0C:/Users/you/Documents"))) == []


def test_ico_is_scanned_as_bytes(tmp_path):
    assert _scan_file(tmp_path, "a.ico", b"\0\0\1\0" + BOB.encode()) != []
    assert _scan_file(tmp_path, "b.ico", b"\0\0\1\0" + b"nothing here") == []


@pytest.mark.parametrize("text, hit", [
    ("/c/Users/" + "bob/Documents", True), ("/D/Users/" + "bob", True), ("/home/" + "bob/x", True), ("/Users/" + "bob/x", True),
    ("C%3A%5CUsers%5C" + "bob%5Cx", True), ("c%3A%5cusers%5c" + "bob", True),
    ("/home/runner/work", False), ("/home/you/x", False), ("/home/someone/x", False), ("/Users/you/x", False),
    ("C:/Users/you/Documents", False), ("C:/Users/someone/Documents", False), ("C%3A%5CUsers%5Cyou%5Cx", False),
    ("/c/Users/someone/x", False), ("see /usr/home/page", False), ("https://example.org/home/page", False),
])
def test_new_builtin_patterns(text, hit):
    pats = [__import__("re").compile(p, __import__("re").I) for p in ex.DENY_BUILTIN]
    assert any(p.search(text) for p in pats) is hit


# ------------------------------------------------------------------------------------------------ D5
@pytest.mark.skipif(os.name != "posix", reason="Unix modes")
def test_stage_restores_exec_bits_and_never_setuid(tmp_path, monkeypatch):
    from sealed_lab import updater

    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    z = tmp_path / "r.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, mode in (("pyproject.toml", 0o100644), ("run_linux.sh", 0o100755), ("evil.sh", 0o104777), ("w.txt", 0o100666)):
            i = zipfile.ZipInfo(f"sealed_lab-1.0.0/{name}")
            i.create_system = 3
            i.external_attr = mode << 16
            zf.writestr(i, "x\n")
    root = updater.stage(z, "1.0.0")
    assert (root / "run_linux.sh").stat().st_mode & 0o7777 == 0o755
    assert (root / "evil.sh").stat().st_mode & 0o7777 == 0o755
    assert (root / "w.txt").stat().st_mode & 0o7777 == 0o644
    assert (root / "pyproject.toml").stat().st_mode & 0o7777 == 0o644


# ------------------------------------------------------------------------------------------------ D6
@pytest.mark.parametrize("url, ok", [
    ("https://api.scryfall.com/cards/search?page=2", True), ("https://cards.scryfall.io/normal/front/a/b.jpg", True),
    ("https://scryfall.io/x", True), ("https://c1.scryfall.io/x.jpg", True),
    ("http://api.scryfall.com/x", False), ("https://scryfall.io.evil.com/x", False), ("https://evilscryfall.io/x", False),
    ("https://api.scryfall.com@evil.com/x", False), ("https://evil.com/api.scryfall.com", False), ("file:///etc/passwd", False),
    ("https://api.scryfall.com:8443/x", False), ("//api.scryfall.com/x", False), ("", False), (None, False),
])
def test_allowed_url(url, ok):
    from sealed_lab.ingest.scryfall import allowed_url

    assert allowed_url(url) is ok


class _Stream:
    def __init__(self, body: bytes, status=200, headers=None, url="https://cards.scryfall.io/x.jpg"):
        self.body, self.status_code, self.headers, self.url = body, status, headers or {}, url
        self.closed = False
        self.read = 0

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            self.read += n
            yield self.body[i:i + n]

    def close(self):
        self.closed = True


def test_read_capped_stops_streaming_beyond_the_cap():
    from sealed_lab.ingest.scryfall import read_capped

    assert read_capped(_Stream(b"x" * 1000), 1000) == b"x" * 1000
    with pytest.raises(ValueError):
        read_capped(_Stream(b"x" * 1001), 1000)
    big = _Stream(b"x" * 10_000_000)
    with pytest.raises(ValueError):
        read_capped(big, 100_000)
    assert big.read < 1_000_000                                   # aborted early, not read to the end
    with pytest.raises(ValueError):
        read_capped(_Stream(b"x", headers={"Content-Length": "99999999"}), 1000)


class _Session:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return self.resp


def test_fetch_image_refuses_other_hosts_and_oversize(tmp_path, monkeypatch):
    from sealed_lab.ingest import images

    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path))
    monkeypatch.delenv("SEALED_LAB_NO_IMAGES", raising=False)
    monkeypatch.setattr(images, "_wait", lambda kind: None)
    monkeypatch.setattr(images, "_urls", {})
    monkeypatch.setattr(images, "_failed", set())
    m = images._url_map("TST")
    m["Evil"], m["Big"], m["Fine"] = "https://evil.example/x.jpg", "https://cards.scryfall.io/big.jpg", "https://cards.scryfall.io/ok.jpg"
    sess = _Session(None)
    monkeypatch.setattr(images, "_http", lambda: sess)
    assert images.fetch_image("TST", "Evil") is None and sess.calls == []
    sess.resp = _Stream(b"x" * (images.MAX_IMAGE_BYTES + 1))
    assert images.fetch_image("TST", "Big") is None and sess.resp.closed and sess.calls[0][1]["stream"] is True
    sess.resp = _Stream(b"jpg-bytes")
    p = images.fetch_image("TST", "Fine")
    assert p is not None and p.read_bytes() == b"jpg-bytes"
    assert images.fetch_image("../..", "X") is None


def test_scryfall_next_page_must_stay_on_scryfall():
    from sealed_lab.ingest import scryfall

    class Resp:
        status_code = 200
        headers: dict = {}
        url = "https://api.scryfall.com/cards/search"

        def json(self):
            return {"data": [{"name": "A"}], "has_more": True, "next_page": "https://evil.example/steal"}

    sess = _Session(Resp())
    http = scryfall._Http(sess, 0.0, None)
    assert scryfall.fetch_set_search("tst", http) is None
    assert [c[0] for c in sess.calls] == ["https://api.scryfall.com/cards/search"]
    with pytest.raises(ValueError):
        http.get("https://evil.example/x")


# ------------------------------------------------------------------------------------------------ D7 / D8
def test_interrupted_update_marker_warns_once_it_is_old(tmp_path, monkeypatch, capsys):
    from sealed_lab import __main__ as m

    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path))
    assert m._warn_interrupted_update() is None                      # no marker
    mk = tmp_path / m.UPDATE_MARKER
    mk.write_text("started", encoding="utf-8")
    assert m._warn_interrupted_update() is None                      # brand new: an update may be running
    old = time.time() - 3600
    os.utime(mk, (old, old))
    msg = m._warn_interrupted_update()
    assert msg and "did not finish" in msg and str(mk) in msg
    assert "did not finish" in capsys.readouterr().err
    assert mk.exists()


def test_legacy_token_is_forgotten_by_the_cli_path_too(monkeypatch):
    from sealed_lab import __main__ as m
    from sealed_lab.gui import update_ui

    calls = []
    monkeypatch.setattr(update_ui, "forget_legacy_token", lambda: calls.append(1))
    monkeypatch.setitem(sys.modules, "sealed_lab.cli", type(sys)("sealed_lab.cli"))
    sys.modules["sealed_lab.cli"].main = lambda argv: 0
    import sealed_lab
    monkeypatch.setattr(sealed_lab, "cli", sys.modules["sealed_lab.cli"], raising=False)
    assert m.main(["cli", "status"]) == 0 and calls == [1]
    monkeypatch.setattr(update_ui, "forget_legacy_token", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert m.main(["cli", "status"]) == 0                            # never fails the start
