# Developing and releasing sealed_lab

## Run from source
`run_windows.bat` works the same from a checkout: `uv sync --frozen --inexact` keeps `.venv` at the versions in `uv.lock`.
`uv sync --frozen --inexact --extra dev` adds pytest; run the tests with `.venv\Scripts\python -m pytest`. Refresh the
dependency versions with `uv lock --upgrade` and run the whole suite before a release.

## Releasing (owner)
1. Raise `__version__` in `sealed_lab/__init__.py` and commit.
2. In the development repository: `python tools/export_public.py --dest <public clone>`. It copies only the shareable files
   (an allowlist in the script) and refuses if any contains a personal string (built-in patterns plus the untracked
   `tools/.export_deny.txt`). In the public clone: commit, `git tag vX.Y.Z`, `git push --follow-tags`.
3. GitHub Actions (`.github/workflows/release.yml`) runs the tests on Windows and publishes `sealed_lab-X.Y.Z.zip` as the
   release; friends' apps offer it on their next start.

## Command line
`run_windows_cli.bat status`, `run_windows_cli.bat calc SOS --jobs 16 --boot 16`,
`run_windows_cli.bat build SOS pool.txt --fmt bo3`, `run_windows_cli.bat score SOS deck.txt`.

## Tests and fake data
`python -m pytest -q` (Linux or Windows). `python -m sealed_lab cli synth OUTDIR --scale 0.3` writes a synthetic
set in exact 17lands format with known card strengths, used by the tests. `SEALED_LAB_FAKE_API=1 python -m
sealed_lab` runs the GUI on fake data.
