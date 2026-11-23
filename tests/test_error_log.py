from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_record_error_writes_and_rotates(home, monkeypatch):
    from sealed_lab.gui import workers

    workers.record_error("calculate HOB", "Traceback (most recent call last):\nValueError: boom")
    path = workers.error_log_path()
    assert path.parent == home / "logs"
    text = path.read_text(encoding="utf-8")
    assert "calculate HOB" in text and "ValueError: boom" in text

    monkeypatch.setattr(workers, "ERROR_LOG_BYTES", 10)
    workers.record_error("build pool", "second")
    workers.record_error("build pool", "third")
    assert (home / "logs" / "errors.1.log").exists()
    assert "third" in path.read_text(encoding="utf-8")


def test_worker_failure_reaches_error_log(home):
    from PySide6.QtWidgets import QApplication

    from sealed_lab.gui import workers

    app = QApplication.instance() or QApplication([])
    ctl = workers.TaskController()
    errors: list[str] = []
    ctl.error.connect(errors.append)

    def fail(progress, cancel):
        raise RuntimeError("calc exploded")

    assert ctl.run(fail, label="calculate TST")
    ctl._worker.wait(5000)
    app.processEvents()
    assert errors and "calc exploded" in errors[0]
    text = workers.error_log_path().read_text(encoding="utf-8")
    assert "calculate TST" in text and "RuntimeError: calc exploded" in text


def test_native_crash_of_previous_session_goes_to_errors_log(home):
    from sealed_lab import __main__ as entry
    from sealed_lab.gui.workers import error_log_path

    crash = home / "logs" / "crash.log"
    crash.parent.mkdir(parents=True)
    dump = "Windows fatal exception: access violation\n\nThread 0x0001 (most recent call first):\n  File \"handcheck.py\", line 35 in compute_psi\n"
    crash.write_text(dump + f"\n{entry.CLEAN_EXIT}\n", encoding="utf-8")       # a handled exception, then a normal exit: not a crash
    assert not entry._check_previous_crash(crash) and not error_log_path().exists()
    crash.write_text(dump, encoding="utf-8")                                     # no clean exit: the process died
    assert entry._check_previous_crash(crash)
    text = error_log_path().read_text(encoding="utf-8")
    assert "native crash in the previous session" in text and "compute_psi" in text
    crash.write_text("", encoding="utf-8")
    assert not entry._check_previous_crash(crash)
