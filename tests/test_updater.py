"""sealed_lab.updater: versions, tokens, the release check, download, staging, the handover command.  No network, no gh, no Popen."""
from __future__ import annotations

import io
import threading
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sealed_lab import updater
from sealed_lab.types import Cancelled


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
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        if self.exc:
            raise self.exc
        return self.resp


def release(tag="v0.2.0", assets=None, body="notes"):
    if assets is None:
        assets = [{"name": f"sealed_lab-{tag.lstrip('v')}.zip", "url": "https://api.github.com/assets/1"}]
    return {"tag_name": tag, "body": body, "html_url": "https://github.com/x/y/releases/tag/" + tag, "assets": assets}


def make_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    return path


GOOD = {"sealed_lab-0.2.0/pyproject.toml": "x", "sealed_lab-0.2.0/sealed_lab/__init__.py": "", "sealed_lab-0.2.0/apply_update.ps1": "# new"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for n in ("SEALED_LAB_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "SEALED_LAB_UPDATE_REPO"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))


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


# --------------------------------------------------------------------------- token

def test_find_token_precedence(monkeypatch):
    gh = lambda: "from-gh"  # noqa: E731
    assert updater.find_token("stored", gh) == "stored"
    assert updater.find_token(None, gh) == "from-gh"
    monkeypatch.setenv("GITHUB_TOKEN", "e3")
    assert updater.find_token("stored", gh) == "e3"
    monkeypatch.setenv("GH_TOKEN", "e2")
    assert updater.find_token("stored", gh) == "e2"
    monkeypatch.setenv("SEALED_LAB_GITHUB_TOKEN", "e1")
    assert updater.find_token("stored", gh) == "e1"


def test_find_token_blank_and_gh_missing(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "   ")
    assert updater.find_token("  ", lambda: None) is None
    assert updater.find_token("", lambda: "  ") is None


def test_gh_auth_token_missing_or_failing(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("gh")
    monkeypatch.setattr(updater.subprocess, "run", boom)
    assert updater._gh_auth_token() is None

    class R:
        returncode, stdout = 1, ""
    monkeypatch.setattr(updater.subprocess, "run", lambda *a, **k: R())
    assert updater._gh_auth_token() is None
    R.returncode, R.stdout = 0, "tok\n"
    assert updater._gh_auth_token() == "tok"


# --------------------------------------------------------------------------- check

def test_check_newer_same_older():
    s = FakeSession(FakeResp(json_data=release("v0.2.0")))
    info = updater.check(s, token=None, repo="a/b", current="0.1.0")
    assert info.available and info.latest == "v0.2.0" and info.notes == "notes" and info.error == ""
    assert info.asset_name == "sealed_lab-0.2.0.zip" and info.asset_url == "https://api.github.com/assets/1"
    assert s.calls[0][0] == "https://api.github.com/repos/a/b/releases/latest"
    assert not updater.check(FakeSession(FakeResp(json_data=release("v0.1.0"))), current="0.1.0").available
    assert not updater.check(FakeSession(FakeResp(json_data=release("v0.0.5"))), current="0.1.0").available


def test_check_default_repo_and_env(monkeypatch):
    s = FakeSession(FakeResp(json_data=release()))
    updater.check(s, current="0.1.0")
    assert f"/repos/{updater.DEFAULT_REPO}/" in s.calls[0][0]
    monkeypatch.setenv("SEALED_LAB_UPDATE_REPO", "me/fork")
    s = FakeSession(FakeResp(json_data=release()))
    updater.check(s, current="0.1.0")
    assert "/repos/me/fork/" in s.calls[0][0]


def test_check_no_zip_asset():
    info = updater.check(FakeSession(FakeResp(json_data=release(assets=[{"name": "notes.txt", "url": "u"}]))), current="0.1.0")
    assert not info.available and "zip" in info.error


def test_check_404_without_and_with_token():
    no = updater.check(FakeSession(FakeResp(404)), token=None, current="0.1.0")
    assert "private" in no.error and "gh auth login" in no.error
    yes = updater.check(FakeSession(FakeResp(404)), token="t", current="0.1.0")
    assert "private" not in yes.error and "not found" in yes.error


def test_check_401_403_and_connection_error():
    for code in (401, 403):
        e = updater.check(FakeSession(FakeResp(code)), token="t", current="0.1.0").error
        assert "invalid or expired" in e
    e = updater.check(FakeSession(exc=ConnectionError("down")), current="0.1.0").error
    assert "could not reach github" in e.lower()
    assert updater.check(FakeSession(FakeResp(500)), current="0.1.0").error


def test_check_auth_header_only_with_token():
    s = FakeSession(FakeResp(json_data=release()))
    updater.check(s, token=None, current="0.1.0")
    h = s.calls[0][1]["headers"]
    assert "Authorization" not in h and h["User-Agent"] == "sealed_lab-updater"
    assert h["Accept"] == "application/vnd.github+json" and h["X-GitHub-Api-Version"] == "2022-11-28"
    s = FakeSession(FakeResp(json_data=release()))
    updater.check(s, token="abc", current="0.1.0")
    assert s.calls[0][1]["headers"]["Authorization"] == "Bearer abc"


# --------------------------------------------------------------------------- download

def _zip_bytes(members) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    return b.getvalue()


def _info():
    return updater.UpdateInfo(available=True, latest="v0.2.0", asset_name="sealed_lab-0.2.0.zip", asset_url="https://api.github.com/assets/1")


def test_download_writes_streams_and_reports_progress(tmp_path):
    data = _zip_bytes(GOOD)
    s = FakeSession(FakeResp(content=data))
    seen = []
    out = updater.download(_info(), tmp_path / "dl", session=s, token="tok", progress=lambda f, m: seen.append(f))
    assert out == tmp_path / "dl" / "sealed_lab-0.2.0.zip" and out.read_bytes() == data
    assert not list((tmp_path / "dl").glob("*.part"))
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    url, kw = s.calls[0]
    assert url == "https://api.github.com/assets/1" and kw["stream"] is True
    assert kw["headers"]["Accept"] == "application/octet-stream" and kw["headers"]["Authorization"] == "Bearer tok"
    assert s.resp.closed


def test_download_cancel_leaves_nothing(tmp_path):
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        updater.download(_info(), tmp_path, session=FakeSession(FakeResp(content=_zip_bytes(GOOD))), cancel=ev)
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_non_zip_and_wrong_layout(tmp_path):
    with pytest.raises(ValueError, match="not a zip"):
        updater.download(_info(), tmp_path, session=FakeSession(FakeResp(content=b"<html>nope</html>")))
    with pytest.raises(ValueError, match="does not look like"):
        updater.download(_info(), tmp_path, session=FakeSession(FakeResp(content=_zip_bytes({"a/readme.txt": "x"}))))
    assert list(tmp_path.iterdir()) == []


def test_download_http_and_network_errors(tmp_path):
    with pytest.raises(RuntimeError, match="private"):
        updater.download(_info(), tmp_path, session=FakeSession(FakeResp(404)))
    with pytest.raises(RuntimeError, match="reach GitHub"):
        updater.download(_info(), tmp_path, session=FakeSession(exc=ConnectionError("x")))


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
    return root


def test_can_self_update(tmp_path):
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


def test_can_self_update_not_writable(tmp_path, monkeypatch):
    app = _install(tmp_path / "app")
    monkeypatch.setattr(updater.os, "access", lambda *a, **k: False)
    ok, why = updater.can_self_update(app)
    assert not ok and "writable" in why


def test_app_dir_is_the_checkout_root():
    assert (updater.app_dir() / "sealed_lab" / "updater.py").is_file()


def test_launch_apply_command_and_script_copy(tmp_path):
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
    if updater.sys.platform == "win32":
        flags = calls[0][1]["creationflags"]
        assert flags & updater.subprocess.CREATE_NEW_CONSOLE and flags & updater.subprocess.CREATE_NEW_PROCESS_GROUP


# --------------------------------------------------------------------------- due

def test_due():
    now = datetime(2026, 5, 10, 12, 0, 0)
    assert updater.due(None, now) and updater.due("", now) and updater.due("garbage", now)
    assert not updater.due((now - timedelta(hours=3)).isoformat(), now)
    assert updater.due((now - timedelta(hours=24)).isoformat(), now)
    assert updater.due((now - timedelta(days=3)).isoformat(), now)
    assert updater.due((now - timedelta(hours=2)).isoformat(), now, hours=1)
