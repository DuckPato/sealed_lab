# Security

## How sealed_lab installs and updates itself

**First install.** You download `sealed_lab-X.Y.Z.zip` from this repository's Releases page and run `run_windows.bat` or
`run_linux.sh`. If `uv` is not installed yet, `get_uv.ps1` / `get_uv.sh` download one pinned uv release from uv's GitHub releases and
check it against the SHA-256 written in the script before unpacking it into the app folder (`.uv`); nothing is piped into a shell
and no PATH or profile is changed. uv then installs Python and the exact library versions recorded, with their hashes, in `uv.lock`.

**Updates.** Once a day (Help > Check for updates automatically) the app asks GitHub's public API for the latest release, without any
account or token. When you click *Update now*:

1. It downloads `sealed_lab-X.Y.Z.manifest.json` and checks its Ed25519 signature against the public key(s) built into the app
   (`sealed_lab/release_keys.py`). The private key is kept offline by the maintainer: it is never on GitHub, so a compromised GitHub
   account or a replaced release file cannot produce a valid signature.
2. The signed manifest must name this repository, this exact tag and file; a manifest of another release is rejected.
3. The zip is downloaded only up to the signed size and must match the signed SHA-256 (and GitHub's own digest); otherwise it is
   deleted. Extraction refuses absolute or `..` paths, links and oversized archives.
4. The update script (from the verified zip) backs up the current version, copies the new files, syncs the libraries and checks that
   the new version starts. If any step fails, the previous version is restored exactly and started again.

An unsigned release is never installed by the app; it can still be downloaded by hand from the release page.

**Releases** are built from the tagged source by GitHub Actions (read-only token except for creating the draft release; actions
pinned to commit SHAs) and published only after the maintainer has checked, on their own computer, that the zip is identical to the one
the tagged source builds and signed it (`tools/release_sign.py`). Anyone can repeat that check:
`python tools/release_sign.py verify --tag vX.Y.Z`.

## What the app sends and stores

- To GitHub: anonymous requests for the latest release and its files. To 17Lands: downloads of its public dataset files (only from
  `data.17landscapes.com`). To Scryfall: card data and images. No account, password or token is used anywhere.
- Your data, models and settings stay on your computer (`%LOCALAPPDATA%\sealed_lab` or `~/.sealed_lab`). Older versions (0.1.1-0.1.2)
  could store a GitHub token in the app settings; the current version deletes it at start.

## Reporting a problem

Please report security issues privately through GitHub (**Security > Report a vulnerability** on this repository) rather than in a
public issue.
