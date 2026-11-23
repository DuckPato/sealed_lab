from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# App preferences go to a throw-away .ini, never the user's registry (QSettings.setPath does not redirect the Windows
# registry-backed native format); see gui.widgets.new_settings.
os.environ.setdefault("SEALED_LAB_SETTINGS", os.path.join(tempfile.mkdtemp(prefix="sealed_lab_cfg_"), "settings.ini"))

# The Arena-log watcher of the Pool tab polls the log folder: tests never see the real one (they point it at their own folder).
os.environ["SEALED_LAB_ARENA_LOG_DIR"] = tempfile.mkdtemp(prefix="sealed_lab_arena_")

# The once-a-day "is there a new release" check goes to github.com: never from a test (test_update_ui drives a fake updater).
os.environ["SEALED_LAB_NO_UPDATE_CHECK"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `built` and `calc` are session fixtures and each points SEALED_LAB_HOME at its own home when it is first created.  Left alone,
# the home a test (and every fixture and worker process it starts) sees is the one of whichever was created last, i.e. it depends
# on the alphabetical order of the test files.  So each test is given the home of the one it asks for (`calc` when it asks for
# both): before any of its fixtures is set up (the hook, so module / session fixtures built for it see it too) and again once its
# session fixtures exist (`_session_home`).  A home set by a narrower fixture of the test's own module is left alone.
_HOMES: dict[str, str] = {}           # "built" / "calc" -> the home that fixture created
_FIXTURE_HOMES: set[str] = set()      # every home conftest itself put into SEALED_LAB_HOME


def _wanted_home(fixturenames) -> str | None:
    key = "calc" if "calc" in fixturenames else "built" if "built" in fixturenames else None
    home = _HOMES.get(key) if key else None
    return home if home and os.environ.get("SEALED_LAB_HOME") in _FIXTURE_HOMES else None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    home = _wanted_home(getattr(item, "fixturenames", ()))
    if home:
        os.environ["SEALED_LAB_HOME"] = home


@pytest.fixture(scope="session", autouse=True)
def _scratch_app_home(tmp_path_factory):
    """No test may touch the real app home (%LOCALAPPDATA%/sealed_lab: errors.log, registries, models): default the app home to a scratch
    folder for the whole session (tests that need their own home still monkeypatch SEALED_LAB_HOME)."""
    mp = pytest.MonkeyPatch()
    if not os.environ.get("SEALED_LAB_HOME"):
        mp.setenv("SEALED_LAB_HOME", str(tmp_path_factory.mktemp("scratch_home")))
        _FIXTURE_HOMES.add(os.environ["SEALED_LAB_HOME"])
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _session_home(request):
    """See `_HOMES`: the home of the session fixture this test asks for, now that it exists.  Its own MonkeyPatch, because tests
    call `monkeypatch.undo()` half-way."""
    home = _wanted_home(request.fixturenames)
    if home is None:
        yield
        return
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SEALED_LAB_HOME", home)
        yield


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    """Synthetic 17lands-format data set (scale 0.2), generated once per test session."""
    from sealed_lab.synth.generate import generate

    d = tmp_path_factory.mktemp("synth")
    generate(d, set_code="TST", seed=0, scale=0.2)
    return d


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("SEALED_LAB_HOME", str(h))
    return h


@pytest.fixture(scope="session")
def built(synth_dir, tmp_path_factory):
    """A corpus built once from the synthetic data (read-only for tests)."""
    mp = pytest.MonkeyPatch()
    h = tmp_path_factory.mktemp("built_home")
    mp.setenv("SEALED_LAB_HOME", str(h))
    _HOMES["built"] = str(h)
    _FIXTURE_HOMES.add(str(h))
    from sealed_lab import api
    from sealed_lab.ingest.corpus import build_corpus

    api.add_files([str(synth_dir)])
    man = build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    yield {"home": h, "manifest": man, "synth": synth_dir}
    mp.undo()


@pytest.fixture(scope="session")
def calc(tmp_path_factory):
    """Synthetic set -> corpus -> model, once per session (shared by the model and deck tests).

    Yields dict(home, synth, meta, truth, progress, seconds). SEALED_LAB_HOME points at the built home while it lives.
    """
    import json
    import time

    from sealed_lab import api
    from sealed_lab.model import pipeline
    from sealed_lab.synth.generate import generate

    mp = pytest.MonkeyPatch()
    home = tmp_path_factory.mktemp("model_home")
    synth = tmp_path_factory.mktemp("model_synth")
    mp.setenv("SEALED_LAB_HOME", str(home))
    _HOMES["calc"] = str(home)
    _FIXTURE_HOMES.add(str(home))
    generate(synth, set_code="TST", seed=0, scale=0.3)
    api.add_files([str(synth)])
    from sealed_lab.ingest.corpus import build_corpus

    build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    log: list[tuple[float, str]] = []
    t0 = time.time()
    meta = pipeline.calculate("TST", progress=lambda f, m: log.append((f, m)), n_jobs=2, n_boot=4)
    yield dict(home=home, synth=synth, meta=meta, truth=json.loads((synth / "truth.json").read_text(encoding="utf-8")),
               progress=log, seconds=time.time() - t0)
    mp.undo()


@pytest.fixture(scope="session")
def model(calc):
    from sealed_lab.model.artifact import SetModel

    return SetModel.load("TST")
