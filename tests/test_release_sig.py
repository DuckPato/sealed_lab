"""Signed release manifests (sealed_lab.release_sig) and the maintainer's signing tool (tools/release_sign.py), without gh or network."""
from __future__ import annotations

import base64
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sealed_lab import release_sig as rs
from sealed_lab.release_sig import ManifestError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release_sign  # noqa: E402


def keypair():
    k = Ed25519PrivateKey.generate()
    raw = k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return k, {rs.key_id(raw): rs._b64(raw)}


def payload(**over):
    p = dict(version="0.3.0", repo="a/b", asset="sealed_lab-0.3.0.zip", size=123, sha256="ab" * 32, created="2026-11-24T00:00:00+00:00")
    p.update(over)
    return rs.make_payload(**p)


def test_roundtrip_and_fields():
    k, trusted = keypair()
    p = rs.verify_manifest(rs.sign_payload(payload(), k), trusted)
    assert p == {"format": 1, "app": "sealed_lab", "repo": "a/b", "version": "0.3.0", "tag": "v0.3.0", "asset": "sealed_lab-0.3.0.zip",
                 "size": 123, "sha256": "ab" * 32, "created": "2026-11-24T00:00:00+00:00"}
    rs.check_payload(p, repo="A/B", tag="v0.3.0", asset="sealed_lab-0.3.0.zip")       # GitHub names are case-insensitive
    assert rs.manifest_name("v0.3.0") == "sealed_lab-0.3.0.manifest.json" and rs.zip_name("0.3.0") == "sealed_lab-0.3.0.zip"


def test_signature_is_over_the_exact_payload_bytes():
    k, trusted = keypair()
    doc = json.loads(rs.sign_payload(payload(), k))
    raw = base64.b64decode(doc["payload"]).replace(b'"size":123', b'"size":124')
    doc["payload"] = base64.b64encode(raw).decode()
    with pytest.raises(ManifestError, match="signature is not valid"):
        rs.verify_manifest(json.dumps(doc).encode(), trusted)


def test_unknown_or_missing_keys():
    k, trusted = keypair()
    _, other = keypair()
    m = rs.sign_payload(payload(), k)
    with pytest.raises(ManifestError, match="does not trust"):
        rs.verify_manifest(m, other)
    with pytest.raises(ManifestError, match="no release key"):
        rs.verify_manifest(m, {})
    kid = next(iter(trusted))
    with pytest.raises(ManifestError, match="inconsistent"):                       # a key id that is not the hash of its key
        rs.verify_manifest(m, {kid: next(iter(other.values()))})


@pytest.mark.parametrize("doc", [b"", b"[]", b"{not json", json.dumps({"format": 2}).encode(),
                                 json.dumps({"format": 1, "key_id": "x"}).encode()])
def test_malformed_manifests(doc):
    _, trusted = keypair()
    with pytest.raises(ManifestError):
        rs.verify_manifest(doc, trusted)


def test_bad_base64_and_oversize():
    k, trusted = keypair()
    doc = json.loads(rs.sign_payload(payload(), k))
    doc["signature"] = "!!!"
    with pytest.raises(ManifestError, match="base64"):
        rs.verify_manifest(json.dumps(doc).encode(), trusted)
    with pytest.raises(ManifestError, match="too large"):
        rs.verify_manifest(b" " * (rs.MAX_MANIFEST_BYTES + 1), trusted)


@pytest.mark.parametrize("over", [dict(asset="other.zip"), dict(size=0), dict(size=rs.MAX_ZIP_BYTES + 1), dict(sha256="xyz"),
                                  dict(version="1.0-rc1")])
def test_payload_shape_is_enforced_when_signing(over):
    with pytest.raises(ManifestError):
        payload(**over)


def test_check_payload_mismatches():
    k, trusted = keypair()
    p = rs.verify_manifest(rs.sign_payload(payload(), k), trusted)
    for kw in (dict(repo="x/y", tag="v0.3.0", asset="sealed_lab-0.3.0.zip"), dict(repo="a/b", tag="v0.3.1", asset="sealed_lab-0.3.0.zip"),
               dict(repo="a/b", tag="v0.3.0", asset="sealed_lab-0.3.1.zip")):
        with pytest.raises(ManifestError):
            rs.check_payload(p, **kw)


def test_shipped_key_list_is_well_formed():
    from sealed_lab.release_keys import TRUSTED_KEYS

    for kid, pub in TRUSTED_KEYS.items():
        raw = base64.b64decode(pub)
        assert len(raw) == 32 and rs.key_id(raw) == kid


# --------------------------------------------------------------------------- tools/release_sign.py

def test_init_writes_a_protected_key_and_its_public_half(tmp_path, monkeypatch):
    keys = tmp_path / "release_keys.py"
    keys.write_text((ROOT / "sealed_lab" / "release_keys.py").read_text(encoding="utf-8"), encoding="utf-8")
    before = release_sign.read_trusted(keys)                                       # the real keys stay; one is added
    key = tmp_path / "k" / "release.pem"
    answers = iter(["correct horse", "correct horse"])
    monkeypatch.setattr(release_sign.getpass, "getpass", lambda prompt="": next(answers))
    assert release_sign.main(["init", "--key", str(key), "--keys-file", str(keys)]) == 0
    assert b"ENCRYPTED PRIVATE KEY" in key.read_bytes()
    if os.name != "nt":
        assert (key.stat().st_mode & 0o777) == 0o600
    trusted = release_sign.read_trusted(keys)
    assert len(trusted) == len(before) + 1 and before.items() <= trusted.items()
    monkeypatch.setattr(release_sign.getpass, "getpass", lambda prompt="": "correct horse")
    priv = release_sign.load_private(key)
    assert rs.key_id(release_sign._public_raw(priv)) in trusted
    assert rs.verify_manifest(rs.sign_payload(payload(), priv), trusted)["version"] == "0.3.0"
    with pytest.raises(SystemExit):                                                 # never overwrites a key
        release_sign.main(["init", "--key", str(key), "--keys-file", str(keys), "--no-passphrase"])
    monkeypatch.setattr(release_sign.getpass, "getpass", lambda prompt="": "wrong one")
    with pytest.raises(SystemExit):
        release_sign.load_private(key)


def test_init_refuses_short_or_mismatched_passphrases(tmp_path, monkeypatch):
    keys = tmp_path / "release_keys.py"
    keys.write_text("TRUSTED_KEYS: dict[str, str] = {}\n", encoding="utf-8")
    for answers in (["short", "short"], ["long enough 1", "long enough 2"]):
        it = iter(answers)
        monkeypatch.setattr(release_sign.getpass, "getpass", lambda prompt="", it=it: next(it))
        with pytest.raises(SystemExit):
            release_sign.main(["init", "--key", str(tmp_path / "k.pem"), "--keys-file", str(keys)])
        assert not (tmp_path / "k.pem").exists()


def _zip(path: Path, members: dict[str, tuple[int, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, (mode, data) in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o100000 | mode) << 16
            z.writestr(info, data)
    return path


def test_compare_zips(tmp_path):
    base = {"a/x.py": (0o644, b"print(1)\n"), "a/run.sh": (0o755, b"#!/bin/sh\n")}
    ref = _zip(tmp_path / "ref.zip", base)
    assert release_sign.compare_zips(_zip(tmp_path / "same.zip", base), ref) == []
    diff = release_sign.compare_zips(_zip(tmp_path / "d.zip", {**base, "a/x.py": (0o644, b"print(2)\n"), "a/new.py": (0o644, b"")}), ref)
    assert "content differs: a/x.py" in diff and "only in the release zip: a/new.py" in diff
    assert release_sign.compare_zips(_zip(tmp_path / "m.zip", {**base, "a/run.sh": (0o644, b"#!/bin/sh\n")}), ref) == \
        ["mode differs: a/run.sh (0o644 vs 0o755)"]
    assert release_sign.compare_zips(_zip(tmp_path / "s.zip", {"a/x.py": base["a/x.py"]}), ref) == ["missing from the release zip: a/run.sh"]
