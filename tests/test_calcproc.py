"""Calculations in a child process: a native crash is retried, errors and cancels come back as before."""
import sys
import textwrap
import threading

import pytest

from sealed_lab import api
from sealed_lab.types import Cancelled


def _fake_child(monkeypatch, tmp_path, body: str):
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    monkeypatch.setattr(api, "_calc_child_cmd", lambda sc, o, cf: [sys.executable, str(script), str(tmp_path / "count"), cf])


def test_native_crash_is_retried_once(home, tmp_path, monkeypatch):
    _fake_child(monkeypatch, tmp_path, """
        import os, sys
        p = sys.argv[1]
        n = int(open(p).read()) if os.path.exists(p) else 0
        open(p, "w").write(str(n + 1))
        print("P 0.25\tIngesting", flush=True)
        print("INFO sealed_lab.model: a log line", file=sys.stderr, flush=True)
        if n == 0:
            os._exit(-1073741819 if os.name == "nt" else 139)      # like an access violation
        print("P 1.0\tDone", flush=True)
    """)
    seen = []
    st = api.calculate_set_isolated("TST", progress=lambda f, m: seen.append((f, m)))
    assert st.set_code == "TST" and (tmp_path / "count").read_text() == "2"
    assert ("The calculation process crashed; retrying" in [m for _, m in seen]) and seen[-1] == (1.0, "Done")


def test_two_crashes_are_reported(home, tmp_path, monkeypatch):
    _fake_child(monkeypatch, tmp_path, """
        import os
        print("Windows fatal exception: access violation", file=__import__("sys").stderr, flush=True)
        os._exit(-1073741819 if os.name == "nt" else 139)
    """)
    with pytest.raises(api.CalculationError) as e:
        api.calculate_set_isolated("TST")
    assert "access violation" in str(e.value) and str(e.value).splitlines()[-1].endswith("2 times")


def test_child_error_keeps_its_message(home, tmp_path, monkeypatch):
    _fake_child(monkeypatch, tmp_path, """
        import sys
        print("Traceback (most recent call last):", file=sys.stderr)
        print("ValueError: AFR: no Sealed (Bo1) or TradSealed (Bo3) games", file=sys.stderr, flush=True)
        sys.exit(1)
    """)
    with pytest.raises(api.CalculationError) as e:
        api.calculate_set_isolated("AFR")
    assert str(e.value).splitlines()[-1] == "ValueError: AFR: no Sealed (Bo1) or TradSealed (Bo3) games"


def test_cancel_reaches_the_child(home, tmp_path, monkeypatch):
    _fake_child(monkeypatch, tmp_path, """
        import os, sys, time
        print("P 0.1\tworking", flush=True)
        while not os.path.exists(sys.argv[2]):
            time.sleep(0.1)
        sys.exit(2)
    """)
    cancel = threading.Event()
    with pytest.raises(Cancelled):
        api.calculate_set_isolated("TST", progress=lambda f, m: cancel.set(), cancel=cancel)


def test_real_calculation_in_a_child_process(synth_dir, home):
    api.add_files([str(synth_dir)])
    seen = []
    st = api.calculate_set_isolated("TST", progress=lambda f, m: seen.append(f), n_jobs=2, fetch_scryfall=False)
    assert st.has_model and not st.stale and max(seen) == 1.0
