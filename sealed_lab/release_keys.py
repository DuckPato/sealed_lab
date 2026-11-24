"""Public keys whose signature makes a release installable by the in-app updater (key id -> raw Ed25519 public key, base64).

Written by `python tools/release_sign.py init`; the matching private key stays on the maintainer's computer and is never committed or
uploaded.  An update is installed only if its manifest is signed by one of these keys, so a hijacked GitHub account or release cannot push
code to existing installs.  To rotate: add the new key here in a release signed with the old one, then drop the old key a release later.
"""
from __future__ import annotations

TRUSTED_KEYS: dict[str, str] = {
    "5d532547db0a8bc2": "YCVwukhWwW2MpeH1Ha+nWoXhtI222w4bMnIAY73ErXw=",
}
