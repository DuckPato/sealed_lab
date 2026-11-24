"""Signed release manifests: what the updater checks before it extracts or runs anything from a downloaded release.

A release carries, next to `sealed_lab-X.Y.Z.zip`, the file `sealed_lab-X.Y.Z.manifest.json`:

    {"format": 1, "key_id": "<16 hex>", "payload": "<base64 of the payload JSON bytes>", "signature": "<base64 Ed25519 signature>"}

The signature covers the payload bytes exactly as stored (no re-serialisation, so no canonical-JSON pitfalls).  The payload names the
app, the repository, the version and tag, the zip's file name, its size and its SHA-256.  `verify_manifest` returns the payload only if
the signature is valid for a key in `release_keys.TRUSTED_KEYS` and every field has the expected shape; the updater then checks the
values against the release it is installing (`check_payload`) and the zip against the size and hash while it downloads.

Signing (`sign_payload`) is used only by `tools/release_sign.py` on the maintainer's computer.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from typing import Any

FORMAT = 1
APP = "sealed_lab"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_ZIP_BYTES = 200 * 1024 * 1024          # a release zip is < 1 MB today; anything near this is not ours
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID = re.compile(r"^[0-9a-f]{16}$")
_VERSION = re.compile(r"^\d+(?:\.\d+)*$")


class ManifestError(ValueError):
    """The manifest is missing, malformed, signed by an unknown key, has a bad signature or does not match the release."""


def manifest_name(version: str) -> str:
    return f"{APP}-{version.lstrip('vV')}.manifest.json"


def zip_name(version: str) -> str:
    return f"{APP}-{version.lstrip('vV')}.zip"


def key_id(public_raw: bytes) -> str:
    return hashlib.sha256(public_raw).hexdigest()[:16]


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(s: Any, what: str) -> bytes:
    if not isinstance(s, str):
        raise ManifestError(f"The manifest's {what} is missing.")
    try:
        return base64.b64decode(s.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError) as exc:
        raise ManifestError(f"The manifest's {what} is not valid base64.") from exc


def make_payload(version: str, repo: str, asset: str, size: int, sha256: str, created: str) -> bytes:
    p = {"format": FORMAT, "app": APP, "repo": repo, "version": version.lstrip("vV"), "tag": "v" + version.lstrip("vV"),
         "asset": asset, "size": int(size), "sha256": sha256.lower(), "created": created}
    _check_shape(p)
    return json.dumps(p, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_payload(payload: bytes, private_key: Any) -> bytes:
    """The manifest file's bytes for `payload`, signed with an Ed25519 private key object (cryptography)."""
    from cryptography.hazmat.primitives import serialization

    pub = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    doc = {"format": FORMAT, "key_id": key_id(pub), "payload": _b64(payload), "signature": _b64(private_key.sign(payload))}
    return (json.dumps(doc, indent=1) + "\n").encode("utf-8")


def _check_shape(p: Any) -> None:
    if not isinstance(p, dict):
        raise ManifestError("The manifest payload is not an object.")
    want = {"format": int, "app": str, "repo": str, "version": str, "tag": str, "asset": str, "size": int, "sha256": str, "created": str}
    for k, t in want.items():
        if not isinstance(p.get(k), t) or isinstance(p.get(k), bool):
            raise ManifestError(f"The manifest payload has no valid {k!r}.")
    if p["format"] != FORMAT or p["app"] != APP:
        raise ManifestError("The manifest is not a sealed_lab release manifest of a known format.")
    if not _VERSION.match(p["version"]) or p["tag"] != "v" + p["version"]:
        raise ManifestError("The manifest's version / tag are not a release version.")
    if p["asset"] != zip_name(p["version"]):
        raise ManifestError("The manifest names an unexpected file.")
    if not (0 < p["size"] <= MAX_ZIP_BYTES):
        raise ManifestError("The manifest's file size is out of range.")
    if not _HEX64.match(p["sha256"]):
        raise ManifestError("The manifest's SHA-256 is not a hex digest.")


def verify_manifest(data: bytes, trusted: dict[str, str] | None = None) -> dict:
    """The verified payload (dict) of a manifest file; raises ManifestError for anything short of a valid signature by a trusted key."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if trusted is None:
        from sealed_lab.release_keys import TRUSTED_KEYS as trusted
    if not trusted:
        raise ManifestError("This copy of sealed_lab has no release key, so it cannot check an update. Download it from the release page.")
    if len(data) > MAX_MANIFEST_BYTES:
        raise ManifestError("The manifest is too large.")
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestError("The manifest is not valid JSON.") from exc
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ManifestError("The manifest is not of a known format.")
    kid = doc.get("key_id")
    if not isinstance(kid, str) or not _KEY_ID.match(kid) or kid not in trusted:
        raise ManifestError("The update is signed with a key this copy of sealed_lab does not trust.")
    pub_raw = _unb64(trusted[kid], "trusted key")
    if key_id(pub_raw) != kid:
        raise ManifestError("The trusted key list is inconsistent.")
    payload = _unb64(doc.get("payload"), "payload")
    sig = _unb64(doc.get("signature"), "signature")
    try:
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(sig, payload)
    except (InvalidSignature, ValueError) as exc:
        raise ManifestError("The update's signature is not valid: it was not signed by the sealed_lab maintainer.") from exc
    try:
        p = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestError("The signed payload is not valid JSON.") from exc
    _check_shape(p)
    return p


def check_payload(p: dict, *, repo: str, tag: str, asset: str) -> None:
    """The signed payload is for exactly this release (repo, tag, file name): a valid manifest of another release is rejected."""
    if p["repo"].lower() != repo.lower():
        raise ManifestError(f"The signed manifest is for {p['repo']}, not {repo}.")
    if p["tag"] != tag:
        raise ManifestError(f"The signed manifest is for {p['tag']}, not {tag}.")
    if p["asset"] != asset:
        raise ManifestError(f"The signed manifest is for {p['asset']}, not {asset}.")
