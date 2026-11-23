# Developing and releasing sealed_lab

## Run from source
`run_windows.bat` works the same from a checkout: `uv sync --frozen --inexact` keeps `.venv` at the versions in `uv.lock`.
`uv sync --frozen --inexact --extra dev` adds pytest; run the tests with `.venv\Scripts\python -m pytest`. Refresh the
dependency versions with `uv lock --upgrade` and run the whole suite before a release.

## Linux
Run from source: `uv sync --frozen --extra dev`, then `uv run python -m sealed_lab` (or `bash run_linux.sh`, which also adds the
app-menu entry). Tests: `QT_QPA_PLATFORM=offscreen uv run --frozen --no-sync python -m pytest -q`; on a bare Ubuntu or WSL install
`libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libfontconfig1 libdbus-1-3` first. GitHub runs the quick subset on
both Windows and Ubuntu. The `.sh` files must keep LF line endings (`.gitattributes`; the zip build refuses CRLF). The updater
starts `apply_update.ps1` or `apply_update.sh` from the NEW zip, so `apply_update.ps1` must keep shipping for old Windows installs.

## Releasing (owner)
1. Run the **full** suite locally (about 30 minutes): GitHub only runs the quick subset (`SEALED_LAB_QUICK_TESTS=1` skips the
   tests that fit models, which take over an hour on a free runner). Then raise `__version__` in `sealed_lab/__init__.py` and commit.
2. In the development repository: `python tools/export_public.py --dest <public clone>`. It copies only the shareable files
   (an allowlist in the script) and refuses if any contains a personal string (built-in patterns plus the untracked
   `tools/.export_deny.txt`). In the public clone: commit, `git tag vX.Y.Z`, `git push --follow-tags`.
3. GitHub Actions (`.github/workflows/release.yml`) runs the quick tests on Windows and Ubuntu and publishes `sealed_lab-X.Y.Z.zip` as the
   release; friends' apps offer it on their next start.

## Command line
`run_windows_cli.bat status`, `run_windows_cli.bat calc SOS --jobs 16 --boot 16`,
`run_windows_cli.bat build SOS pool.txt --fmt bo3`, `run_windows_cli.bat score SOS deck.txt`.

## Tests and fake data
`python -m pytest -q` (Linux or Windows). `python -m sealed_lab cli synth OUTDIR --scale 0.3` writes a synthetic
set in exact 17lands format with known card strengths, used by the tests. `SEALED_LAB_FAKE_API=1 python -m
sealed_lab` runs the GUI on fake data.
