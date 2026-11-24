# Developing and releasing sealed_lab

## Run from source
`run_windows.bat` works the same from a checkout: `uv sync --frozen --inexact` keeps `.venv` at the versions in `uv.lock` (uv itself:
the one on PATH, else a pinned version that `get_uv.ps1` / `get_uv.sh` download and check against a SHA-256 into `.uv`).
`uv sync --frozen --inexact --extra dev` adds pytest; run the tests with `.venv\Scripts\python -m pytest`. Refresh the
dependency versions with `uv lock --upgrade` and run the whole suite before a release.

## Linux
Run from source: `uv sync --frozen --extra dev`, then `uv run python -m sealed_lab` (or `bash run_linux.sh`, which also adds the
app-menu entry). Tests: `QT_QPA_PLATFORM=offscreen uv run --frozen --no-sync python -m pytest -q`; on a bare Ubuntu or WSL install
`libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libfontconfig1 libdbus-1-3` first. GitHub runs the quick subset on
both Windows and Ubuntu. The `.sh` files must keep LF line endings (`.gitattributes`; the zip build refuses CRLF). The updater
starts `apply_update.ps1` or `apply_update.sh` from the NEW zip, so `apply_update.ps1` must keep shipping for old Windows installs.

## Releasing (owner)
Releases are signed: the app installs an update only if its `sealed_lab-X.Y.Z.manifest.json` is signed by a key listed in
`sealed_lab/release_keys.py` and the zip matches the signed size and SHA-256 (see `SECURITY.md`). The private key lives only on the
owner's computer (`~/.sealed_lab_release/release_ed25519.pem`, passphrase-protected); keep a backup of it. Create it once with
`uv run python tools/release_sign.py init` and commit `sealed_lab/release_keys.py`.

0. **Ready-made sets** (friends then need not download and calculate the newest sets): recalculate the five newest sets one at a time
   (`run_windows_cli.bat calc SET`), refresh their written reports until `cli report-install` accepts them and they are current, then
   `uv run python tools/bundle_sets.py --out <bundle folder>` (add `--home <app home>` for another home, `--n`, `--sets`). It refuses
   when a chosen set is stale, has no model or has a missing / outdated / failing written report (`--skip-written` ships it without
   one), reduces the corpus manifests to file names, scans every file (decoded parquet / npz text too) for personal strings and prints
   the sizes (15-50 MB per set, about 200 MB for five; GitHub allows 2 GB per release file). The folder is only read in step 4;
   `bundled_sets/` is git-ignored and never exported. Every app update re-downloads the whole bundle, unchanged sets included.
   **Stepping stone:** apps up to 0.1.4 refuse a zip over 200 MB (400 MB unpacked, 5000 files). The first release with the larger limits
   (`release_sig.MAX_ZIP_BYTES`, `updater.MAX_UNPACKED_BYTES`) ships WITHOUT `--bundle`; bundle from the next release on, once friends
   run it (`sign` refuses a bundle over the old limits unless `--allow-large`).
1. Run the **full** suite locally (about 30 minutes): GitHub only runs the quick subset (`SEALED_LAB_QUICK_TESTS=1` skips the
   tests that fit models, which take over an hour on a free runner). Then raise `__version__` in `sealed_lab/__init__.py`, add a
   `## X.Y.Z (date)` section at the top of `CHANGELOG.md` (the release workflow uses it as the GitHub release text; GitHub's generated
   notes when the section is missing) and commit.
   **Urgent, narrow fix** (one area, e.g. a changed download address): run that area's tests and a real check instead, release once
   GitHub's quick tests pass, and run the full suite in the background afterwards (a failure there means a follow-up release).
2. In the development repository: `python tools/export_public.py --dest <public clone>`. It copies only the shareable files
   (an allowlist in the script) and refuses if any contains a personal string (built-in patterns plus the untracked
   `tools/.export_deny.txt`). In the public clone: commit, `git tag vX.Y.Z`, `git push --follow-tags`.
3. GitHub Actions (`.github/workflows/release.yml`) runs the quick tests on Windows and Ubuntu and attaches `sealed_lab-X.Y.Z.zip` to a
   **draft** release (nobody sees it yet). The workflow has a read-only token except in the release job, and every action is pinned to
   a commit SHA: bump those deliberately, never to a tag.
4. `uv run python tools/release_sign.py sign --tag vX.Y.Z --public <public clone> --bundle <bundle folder>` (asks for the key's passphrase): it downloads the draft's
   zip, rebuilds the zip from a fresh checkout of the tag and refuses unless the two are identical, then adds the bundle as `bundled_sets/...`
   entries to the CI zip, signs THAT zip (replacing the CI asset) and attaches the manifest,
   publishes the release and checks it the way the app will (`release_sign.py verify --tag vX.Y.Z [--public <public clone>]` repeats that check
   any time; with `--public` it also compares everything except `bundled_sets/` with the tag's rebuild and lists the bundle's sets).
   Without `--bundle` the release carries no ready-made sets. At the friends' next start `sealed_lab/bundled.py` moves the sets into their
   app home, deletes the previous ready-made ones and `bundled_sets/` itself; a set they calculated from their own files is never touched.
   Friends' apps offer the release on their next start.

Changing the key: add the new public key to `release_keys.py` in a release signed with the old key, sign later releases with the new
one, and drop the old key a release after that.

## Accepting a pull request (owner)
The public repository is an export of this one, so a pull request merged only on GitHub would be overwritten by the next export.
Bring it here instead (friends' instructions are in `CONTRIBUTING.md`):

1. `uv run python tools/import_pr.py <number>` applies the pull request's commits, with their authors, on a branch `pr-<number>` in a
   separate worktree (`.pr/pr-<number>` at the repository root), so the working tree the app runs from is not touched. It runs nothing
   from the pull request and lists the install / update / release files it changes.
2. Read the whole change (`git log -p HEAD..pr-<number>`) **before** running its tests: tests run the contributor's code. Once it is
   released and signed, every friend's app installs it, so the signature is only as good as this review.
3. Accept it with `git merge --ff-only pr-<number>` (app closed), then remove the worktree and branch. It reaches GitHub with the next
   export and release; close the pull request with a link to that release.

## Screenshots of GETTING_STARTED.md
`uv run python scripts/guide_screenshots.py` retakes them into `guide_images/` from the built-in example backend (made-up set
TST), so they never show real data or folders. Rerun it when a tab changes and check the pictures before committing.

## Command line
`run_windows_cli.bat status`, `run_windows_cli.bat calc SOS --jobs 16 --boot 16`,
`run_windows_cli.bat build SOS pool.txt --fmt bo3`, `run_windows_cli.bat score SOS deck.txt`.
`calc SOS --quick` is the GUI's Quick calculation: it forces `--manasim v1 --recency off --attributes off --extra-draft off` (the four
hold-out tests that never changed a model; about 40% faster) and stores `quick: true` in the model's meta.json. Without `--quick`
(the default for the API, CLI and research scripts) nothing changes.

## Tests and fake data
`python -m pytest -q` (Linux or Windows). `python -m sealed_lab cli synth OUTDIR --scale 0.3` writes a synthetic
set in exact 17lands format with known card strengths, used by the tests. `SEALED_LAB_FAKE_API=1 python -m
sealed_lab` runs the GUI on fake data.
