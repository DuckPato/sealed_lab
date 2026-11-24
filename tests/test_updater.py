"""sealed_lab.updater: versions, the anonymous release check, the signed-manifest download, staging, the handover command.
No network, no gh, no Popen; signatures are made with a throw-away Ed25519 key."""
from __future__ import annotations

import io
import stat
import threading
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sealed_lab import release_sig, updater
from sealed_lab.release_sig import ManifestError
from sealed_lab.types import Cancelled

REPO = "a/b"
API = f"https://api.github.com/repos/{REPO}"


# --------------------------------------------------------------------------- fakes

class FakeResp:
    def __init__(self, status=200, json_data=None, content=b"", headers=None):
        self.status_code = status
        self._json = json_data
        self._content = content
        self.headers = headers or {"Content-Length": str(len(content))}
        self.closed = False

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), 10):
            yield self._content[i:i + 10]

    def close(self):
        self.closed = True


class FakeSession:
    """`resp`: one answer for every url, or a dict url -> answer."""

    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        if self.exc:
            raise self.exc
        if isinstance(self.resp, dict):
            return self.resp.get(url) or FakeResp(404)
        return self.resp


def make_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    return path


def _zip_bytes(members) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    return b.getvalue()


GOOD = {"sealed_lab-0.2.0/pyproject.toml": "x", "sealed_lab-0.2.0/sealed_lab/__init__.py": "", "sealed_lab-0.2.0/apply_update.ps1": "# new"}


def _key():
    k = Ed25519PrivateKey.generate()
    raw = k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return k, {release_sig.key_id(raw): release_sig._b64(raw)}


KEY, TRUSTED = _key()
OTHER_KEY, OTHER_TRUSTED = _key()


def manifest_for(data: bytes, version="0.2.0", key=None, repo=REPO, asset=None, size=None) -> bytes:
    import hashlib

    p = release_sig.make_payload(version, repo, asset or release_sig.zip_name(version), len(data) if size is None else size,
                                 hashlib.sha256(data).hexdigest(), "2026-11-24T00:00:00+00:00")
    return release_sig.sign_payload(p, key or KEY)


def release(tag="v0.2.0", data: bytes | None = None, manifest: bytes | None = None, signed=True, digest=None, extra=None, body="notes"):
    """(release JSON, url -> response) for a release whose zip is `data` (default: a good zip) and whose manifest is `manifest`
    (default: signed with KEY over `data`; signed=False: none attached)."""
    ver = tag.lstrip("v")
    data = _zip_bytes(GOOD) if data is None else data
    assets = [{"name": f"sealed_lab-{ver}.zip", "url": f"{API}/releases/assets/1", "size": len(data)}]
    if digest is not None:
        assets[0]["digest"] = digest
    routes = {f"{API}/releases/assets/1": FakeResp(content=data)}
    if signed:
        man = manifest_for(data, ver) if manifest is None else manifest
        assets.append({"name": f"sealed_lab-{ver}.manifest.json", "url": f"{API}/releases/assets/2", "size": len(man)})
        routes[f"{API}/releases/assets/2"] = FakeResp(content=man)
    assets += extra or []
    rel = {"tag_name": tag, "body": body, "html_url": f"https://github.com/{REPO}/releases/tag/{tag}", "assets": assets}
    routes[f"{API}/releases/latest"] = FakeResp(json_data=rel)
    return rel, routes


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for n in ("SEALED_LAB_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "SEALED_LAB_UPDATE_REPO"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))


def _checked(routes, current="0.1.0"):
    s = FakeSession(routes)
    return updater.check(s, repo=REPO, current=current), s


# --------------------------------------------------------------------------- versions

def test_parse_version():
    assert updater.parse_version("v0.2.0") == (0, 2, 0)
    assert updater.parse_version("0.10") == (0, 10)
    assert updater.parse_version(" V1.2.3 ") == (1, 2, 3)
    for junk in (None, "", "latest", "v1.2.x", "1.0-rc1", "v"):
        assert updater.parse_version(junk) is None


def test_is_newer():
    assert updater.is_newer("v0.2.0", "0.1.0")
    assert updater.is_newer("0.10.0", "0.9.9")
    assert updater.is_newer("1.0.1", "1.0")
    assert not updater.is_newer("0.1.0", "0.1.0")
    assert not updater.is_newer("0.1", "0.1.0")
    assert not updater.is_newer("0.0.9", "0.1.0")
    assert not updater.is_newer("junk", "0.1.0") and not updater.is_newer("0.2.0", "junk")


# --------------------------------------------------------------------------- check

def test_check_reads_zip_manifest_and_digest():
    _, routes = release(digest="sha256:" + "AB" * 32)
    info, s = _checked(routes)
    assert info.available and info.latest == "v0.2.0" and info.notes == "notes" and info.error == ""
    assert info.asset_name == "sealed_lab-0.2.0.zip" and info.asset_url == f"{API}/releases/assets/1" and info.asset_size > 0
    assert info.manifest_url == f"{API}/releases/assets/2" and info.signed and info.asset_digest == "sha256:" + "ab" * 32
    assert s.calls[0][0] == f"{API}/releases/latest"


def test_check_newer_same_older():
    assert not _checked(release("v0.1.0")[1], current="0.1.0")[0].available
    assert not _checked(release("v0.0.5")[1], current="0.1.0")[0].available
    assert _checked(release("v0.2.0")[1], current="0.1.0")[0].available


def test_check_default_repo_and_env(monkeypatch):
    s = FakeSession(FakeResp(404))
    updater.check(s, current="0.1.0")
    assert s.calls[0][0] == f"https://api.github.com/repos/{updater.DEFAULT_REPO}/releases/latest"
    monkeypatch.setenv("SEALED_LAB_UPDATE_REPO", "me/fork")
    s = FakeSession(FakeResp(404))
    updater.check(s, current="0.1.0")
    assert "/repos/me/fork/" in s.calls[0][0]


def test_check_unsigned_release_is_offered_but_not_signed():
    info, _ = _checked(release(signed=False)[1])
    assert info.available and not info.signed and info.manifest_url == "" and info.error == ""


def test_check_rejects_missing_zip_and_foreign_urls():
    rel, routes = release()
    rel["assets"] = [{"name": "notes.txt", "url": f"{API}/releases/assets/9"}]
    info, _ = _checked(routes)
    assert not info.available and "sealed_lab-0.2.0.zip" in info.error
    rel, routes = release()
    rel["assets"][0]["url"] = "https://evil.example/sealed_lab-0.2.0.zip"
    info, _ = _checked(routes)
    assert not info.available and "outside" in info.error
    rel, routes = release()
    rel["assets"][1]["url"] = "https://api.github.com/repos/someone/else/releases/assets/2"
    assert not _checked(routes)[0].signed                         # a manifest of another repository is not used


def test_check_errors():
    assert "not found" in updater.check(FakeSession(FakeResp(404)), current="0.1.0").error
    for code in (403, 429):
        assert "limit" in updater.check(FakeSession(FakeResp(code)), current="0.1.0").error
    assert "500" in updater.check(FakeSession(FakeResp(500)), current="0.1.0").error
    assert "could not reach github" in updater.check(FakeSession(exc=ConnectionError("down")), current="0.1.0").error.lower()
    assert "not understood" in updater.check(FakeSession(FakeResp(json_data=["x"])), current="0.1.0").error
    rel, routes = release()
    rel["tag_name"] = "nightly"
    assert "not a version" in _checked(routes)[0].error


def test_requests_are_anonymous_and_no_token_is_ever_looked_for(monkeypatch, tmp_path):
    for n in ("SEALED_LAB_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.setenv(n, "secret-" + n)

    def no_subprocess(*a, **k):
        raise AssertionError("the updater must not run gh or any other program while checking / downloading")
    monkeypatch.setattr(updater.subprocess, "run", no_subprocess)
    monkeypatch.setattr(updater.subprocess, "Popen", no_subprocess)
    info, s = _checked(release()[1])
    updater.download(info, tmp_path / "dl", session=s, repo=REPO, trusted=TRUSTED)
    assert len(s.calls) == 3
    for _, kw in s.calls:
        h = kw["headers"]
        assert "Authorization" not in h and not any("secret" in str(v) for v in h.values())
        assert h["User-Agent"].startswith("sealed_lab/") and h["X-GitHub-Api-Version"] == "2022-11-28"
    assert not hasattr(updater, "find_token")


def test_release_page_only_this_repository():
    info = updater.UpdateInfo(html_url=f"https://github.com/{REPO}/releases/tag/v0.2.0")
    assert updater.release_page(info, REPO) == info.html_url
    for bad in ("https://evil.example/", "javascript:alert(1)", f"https://github.com/{REPO}/releasesX", ""):
        assert updater.release_page(updater.UpdateInfo(html_url=bad), REPO) == f"https://github.com/{REPO}/releases"


# --------------------------------------------------------------------------- download (signed)

def _download(routes, tmp_path, trusted=TRUSTED, **kw):
    info, s = _checked(routes)
    return updater.download(info, tmp_path / "dl", session=s, repo=REPO, trusted=trusted, **kw), s


def test_download_verifies_and_streams(tmp_path):
    data = _zip_bytes(GOOD)
    seen = []
    (out, payload), s = _download(release(data=data, digest="sha256:" + __import__("hashlib").sha256(data).hexdigest())[1], tmp_path,
                                  progress=lambda f, m: seen.append(f))
    assert out == tmp_path / "dl" / "sealed_lab-0.2.0.zip" and out.read_bytes() == data
    assert payload["tag"] == "v0.2.0" and payload["repo"] == REPO and payload["size"] == len(data)
    assert not list((tmp_path / "dl").glob("*.part"))
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    (murl, mkw), (zurl, zkw) = s.calls[1], s.calls[2]                 # the manifest first, then the zip
    assert murl.endswith("/assets/2") and zurl.endswith("/assets/1") and zkw["stream"] is True
    assert mkw["headers"]["Accept"] == zkw["headers"]["Accept"] == "application/octet-stream"


@pytest.mark.parametrize("case", ["unsigned", "other_key", "no_trusted_keys", "tampered_zip", "other_tag", "other_repo", "other_asset",
                                  "bigger_than_signed", "github_digest", "github_size", "garbage_manifest", "bad_signature"])
def test_download_refuses_anything_not_exactly_signed(tmp_path, case):
    data = _zip_bytes(GOOD)
    trusted = TRUSTED
    kw = {}
    if case == "unsigned":
        kw = dict(signed=False)
    elif case == "other_key":
        kw = dict(manifest=manifest_for(data, key=OTHER_KEY))
    elif case == "no_trusted_keys":
        trusted = {}
    elif case == "tampered_zip":
        kw = dict(manifest=manifest_for(data), data=_zip_bytes({**GOOD, "sealed_lab-0.2.0/evil.py": "import os"}))
    elif case == "other_tag":                                         # a valid manifest of an older release replayed on this one
        kw = dict(manifest=manifest_for(data, version="0.1.9", asset="sealed_lab-0.1.9.zip"))
    elif case == "other_repo":
        kw = dict(manifest=manifest_for(data, repo="someone/fork"))
    elif case == "other_asset":
        pass                                                          # the app asks for another file name below
    elif case == "bigger_than_signed":
        kw = dict(manifest=manifest_for(data, size=len(data) - 5))
    elif case == "github_digest":
        kw = dict(digest="sha256:" + "0" * 64)
    elif case == "github_size":
        pass
    elif case == "garbage_manifest":
        kw = dict(manifest=b"{not json")
    elif case == "bad_signature":
        import base64
        import json as _j
        doc = _j.loads(manifest_for(data))
        sig = bytearray(base64.b64decode(doc["signature"]))
        sig[0] ^= 1
        doc["signature"] = base64.b64encode(bytes(sig)).decode()
        kw = dict(manifest=_j.dumps(doc).encode())
    rel, routes = release(**kw)
    if case == "github_size":
        rel["assets"][0]["size"] += 1
    info, s = _checked(routes)
    if case == "other_asset":
        info.asset_name = "sealed_lab-0.2.0-other.zip"
    with pytest.raises(ManifestError):
        updater.download(info, tmp_path / "dl", session=s, repo=REPO, trusted=trusted)
    assert not (tmp_path / "dl").exists() or list((tmp_path / "dl").iterdir()) == []


def test_download_signed_but_not_a_sealed_lab_zip(tmp_path):
    for data, msg in ((b"<html>nope</html>", "not a zip"), (_zip_bytes({"a/readme.txt": "x"}), "does not look like")):
        with pytest.raises(ValueError, match=msg):
            _download(release(data=data)[1], tmp_path)
        assert list((tmp_path / "dl").iterdir()) == []


def test_download_cancel_leaves_nothing(tmp_path):
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        _download(release()[1], tmp_path, cancel=ev)
    assert list((tmp_path / "dl").iterdir()) == []


def test_download_http_and_network_errors(tmp_path):
    rel, routes = release()
    routes[f"{API}/releases/assets/2"] = FakeResp(404)
    with pytest.raises(RuntimeError, match="not found"):
        _download(routes, tmp_path)
    rel, routes = release()
    routes[f"{API}/releases/assets/1"] = FakeResp(500)
    with pytest.raises(RuntimeError, match="500"):
        _download(routes, tmp_path)
    info, _ = _checked(release()[1])
    with pytest.raises(RuntimeError, match="reach GitHub"):
        updater.download(info, tmp_path, session=FakeSession(exc=ConnectionError("x")), repo=REPO, trusted=TRUSTED)


def test_oversized_manifest_is_refused(tmp_path):
    rel, routes = release()
    routes[f"{API}/releases/assets/2"] = FakeResp(content=b" " * (release_sig.MAX_MANIFEST_BYTES + 10))
    with pytest.raises(ManifestError, match="too large"):
        _download(routes, tmp_path)


# --------------------------------------------------------------------------- stage: extraction limits

def test_stage_checks_the_hash_again_and_refuses_links_and_bombs(tmp_path, monkeypatch):
    z = make_zip(tmp_path / "a.zip", GOOD)
    with pytest.raises(ValueError, match="changed"):
        updater.stage(z, "v0.2.0", "0" * 64)
    assert updater.stage(z, "v0.2.0", updater.sha256_file(z)).name == "sealed_lab-0.2.0"
    link = tmp_path / "l.zip"
    with zipfile.ZipFile(link, "w") as zf:
        zf.writestr("sealed_lab-0.2.0/pyproject.toml", "x")
        info = zipfile.ZipInfo("sealed_lab-0.2.0/evil")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(ValueError, match="Link"):
        updater.stage(link, "v0.2.0")
    monkeypatch.setattr(updater, "MAX_UNPACKED_BYTES", 2)
    with pytest.raises(ValueError, match="larger"):
        updater.stage(z, "v0.2.0")

# --------------------------------------------------------------------------- stage

def test_stage_top_folder_and_flat(tmp_path):
    z = make_zip(tmp_path / "a.zip", GOOD)
    out = updater.stage(z, "v0.2.0")
    assert out.name == "sealed_lab-0.2.0" and (out / "pyproject.toml").is_file()
    assert out.parent == (tmp_path / "home" / "update" / "staged" / "0.2.0").resolve()
    flat = make_zip(tmp_path / "b.zip", {"pyproject.toml": "x", "sealed_lab/__init__.py": ""})
    stale = tmp_path / "home" / "update" / "staged" / "0.2.0" / "stale.txt"
    stale.write_text("old")
    out2 = updater.stage(flat, "0.2.0")
    assert (out2 / "pyproject.toml").is_file() and not stale.exists()


def test_stage_rejects_zip_slip(tmp_path):
    for bad in ("../evil.txt", "a/../../evil.txt", "/abs.txt", "C:/abs.txt"):
        z = make_zip(tmp_path / "bad.zip", {"top/pyproject.toml": "x", bad: "boom"})
        with pytest.raises(ValueError, match="Unsafe"):
            updater.stage(z, "9.9.9")
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "home" / "update" / "evil.txt").exists()


def test_stage_without_pyproject(tmp_path):
    with pytest.raises(ValueError):
        updater.stage(make_zip(tmp_path / "c.zip", {"x/readme.txt": "x"}), "1.0.0")


# --------------------------------------------------------------------------- can_self_update, launch

def _install(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("x")
    (root / "run_windows.bat").write_text("x")
    (root / "run_linux.sh").write_text("x")
    return root


def test_can_self_update(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.sys, "platform", "win32")
    app = _install(tmp_path / "app")
    assert updater.can_self_update(app) == (True, "")
    (app / ".git").mkdir()
    ok, why = updater.can_self_update(app)
    assert not ok and "git" in why
    (app / ".git").rmdir()
    (tmp_path / ".git").mkdir()                              # a parent that is a checkout counts too
    assert "git" in updater.can_self_update(app)[1]
    (tmp_path / ".git").rmdir()
    (app / "run_windows.bat").unlink()
    ok, why = updater.can_self_update(app)
    assert not ok and "run_windows.bat" in why
    ok, why = updater.can_self_update(tmp_path / "missing")
    assert not ok


def test_can_self_update_needs_the_launcher_of_the_platform(tmp_path, monkeypatch):
    app = _install(tmp_path / "app")
    (app / "run_windows.bat").unlink()
    monkeypatch.setattr(updater.sys, "platform", "linux")
    assert updater.can_self_update(app) == (True, "")
    (app / "run_linux.sh").unlink()
    ok, why = updater.can_self_update(app)
    assert not ok and "run_linux.sh" in why
    (app / "run_windows.bat").write_text("x")
    monkeypatch.setattr(updater.sys, "platform", "win32")
    assert updater.can_self_update(app) == (True, "")
    monkeypatch.setattr(updater.sys, "platform", "linux")
    assert not updater.can_self_update(app)[0]


def test_can_self_update_not_writable(tmp_path, monkeypatch):
    app = _install(tmp_path / "app")
    monkeypatch.setattr(updater.os, "access", lambda *a, **k: False)
    ok, why = updater.can_self_update(app)
    assert not ok and "writable" in why


def test_app_dir_is_the_checkout_root():
    assert (updater.app_dir() / "sealed_lab" / "updater.py").is_file()


def test_launch_apply_command_and_script_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.sys, "platform", "win32")
    if not hasattr(updater.subprocess, "CREATE_NEW_CONSOLE"):
        monkeypatch.setattr(updater.subprocess, "CREATE_NEW_CONSOLE", 0x10, raising=False)
        monkeypatch.setattr(updater.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    staged = tmp_path / "staged" / "sealed_lab-0.2.0"
    staged.mkdir(parents=True)
    (staged / "apply_update.ps1").write_text("# new version's script")
    calls = []
    cmd = updater.launch_apply(staged, tmp_path / "app", 4242, popen=lambda c, **kw: calls.append((c, kw)))
    script = tmp_path / "home" / "update" / "apply_update.ps1"
    assert script.read_text() == "# new version's script"
    assert cmd == ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                   "-Source", str(staged), "-AppDir", str(tmp_path / "app"), "-WaitPid", "4242"]
    assert calls[0][0] == cmd
    flags = calls[0][1]["creationflags"]
    assert flags & updater.subprocess.CREATE_NEW_CONSOLE and flags & updater.subprocess.CREATE_NEW_PROCESS_GROUP


def test_launch_apply_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.sys, "platform", "linux")
    staged = tmp_path / "staged" / "sealed_lab-0.2.0"
    staged.mkdir(parents=True)
    (staged / "apply_update.sh").write_text("# new version's script")
    calls = []
    cmd = updater.launch_apply(staged, tmp_path / "app", 4242, popen=lambda c, **kw: calls.append((c, kw)))
    script = tmp_path / "home" / "update" / "apply_update.sh"
    assert script.read_text() == "# new version's script"
    assert cmd == ["bash", str(script), "--source", str(staged), "--app-dir", str(tmp_path / "app"), "--wait-pid", "4242"]
    assert calls[0][0] == cmd
    kw = calls[0][1]
    assert kw["start_new_session"] is True and kw["close_fds"] is True and "creationflags" not in kw
    assert kw["stdin"] == kw["stdout"] == kw["stderr"] == updater.subprocess.DEVNULL


# --------------------------------------------------------------------------- due

def test_due():
    now = datetime(2026, 5, 10, 12, 0, 0)
    assert updater.due(None, now) and updater.due("", now) and updater.due("garbage", now)
    assert not updater.due((now - timedelta(hours=3)).isoformat(), now)
    assert updater.due((now - timedelta(hours=24)).isoformat(), now)
    assert updater.due((now - timedelta(days=3)).isoformat(), now)
    assert updater.due((now - timedelta(hours=2)).isoformat(), now, hours=1)
