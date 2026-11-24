"""Entry point: `python -m sealed_lab` (GUI) or `python -m sealed_lab cli ...`.

Started with pythonw.exe (the desktop shortcut) there is no console: stdout/stderr are None. Output then goes to
`APP_HOME/logs/sealed_lab.log`, and uncaught errors are logged and shown in a message box instead of vanishing.
"""
from __future__ import annotations

import multiprocessing
import sys

MAX_LOG_BYTES = 5_000_000


def _windowless_logging() -> None:
    """Without a console, send stdout/stderr and logging to a log file (rotated at ~5 MB)."""
    import logging

    from sealed_lab.config import app_home

    log_dir = app_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "sealed_lab.log"
    try:
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            path.replace(path.with_suffix(".old.log"))
    except OSError:
        pass
    fh = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or fh
    sys.stderr = sys.stderr or fh
    logging.basicConfig(level=logging.INFO, stream=fh, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def hook(exc_type, exc, tb):
        import traceback

        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("sealed_lab").error("uncaught exception\n%s", text)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is not None:
                box = QMessageBox(QMessageBox.Icon.Critical, "sealed_lab", f"Unexpected error: {exc}")
                box.setDetailedText(text + f"\nLog file: {path}")
                box.exec()
        except Exception:  # noqa: BLE001 - never fail inside the error handler
            pass

    sys.excepthook = hook


def _record_uncaught() -> None:
    """Uncaught exceptions (e.g. in Qt slots) also go to logs/errors.log, with or without a console."""
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        import traceback

        from sealed_lab.gui.workers import record_error

        record_error("uncaught exception", "".join(traceback.format_exception(exc_type, exc, tb)))
        previous(exc_type, exc, tb)

    sys.excepthook = hook


CLEAN_EXIT = "== clean exit"
_crash_fh = None        # kept open for faulthandler for the whole session


def _check_previous_crash(path) -> bool:
    """A native crash (access violation in a compiled library) kills the process before any Python handler runs; faulthandler
    then left the Python stacks of all threads in `path`. Copy them to errors.log unless that session ended normally."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if ("Fatal Python error" not in text and "Windows fatal exception" not in text) or text.rstrip().endswith(CLEAN_EXIT):
        return False
    from sealed_lab.gui.workers import record_error

    record_error("native crash in the previous session (the app closed without an error message)", text[-20000:])
    return True


def _native_crash_log() -> None:
    import faulthandler

    from sealed_lab.config import app_home

    global _crash_fh
    path = app_home() / "logs" / "crash.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_previous_crash(path)
    _crash_fh = open(path, "w", encoding="utf-8")
    faulthandler.enable(file=_crash_fh, all_threads=True)


def _mark_clean_exit() -> None:
    if _crash_fh is not None:
        try:
            _crash_fh.write(f"\n{CLEAN_EXIT}\n")
            _crash_fh.flush()
        except OSError:
            pass


UPDATE_MARKER = "update_in_progress.txt"
MARKER_GRACE_SECONDS = 15 * 60       # a marker younger than this is probably an update that is still running


def _warn_interrupted_update() -> str | None:
    """apply_update.ps1 / .sh write `<app home>/update_in_progress.txt` before changing files and delete it when the update succeeded
    or was rolled back.  Found at start (and not brand new), the update was interrupted and the files may be a mix of two versions:
    say so on stderr and in the error log, and keep the marker's text for the user.  Never fails the start."""
    try:
        import time

        from sealed_lab.config import app_home

        path = app_home() / UPDATE_MARKER
        if not path.is_file() or time.time() - path.stat().st_mtime < MARKER_GRACE_SECONDS:
            return None
        msg = (f"An earlier update of sealed_lab did not finish ({path}). The program files may be a mix of two versions: if the app "
               f"misbehaves, download sealed_lab again from its release page. Delete {path} to stop this warning.")
        try:
            print(msg, file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
        try:
            from sealed_lab.gui.workers import record_error

            record_error("interrupted update", msg)
        except Exception:  # noqa: BLE001
            pass
        return msg
    except Exception:  # noqa: BLE001 - a warning must never stop the app
        return None


def _forget_legacy_token() -> None:
    """Versions 0.1.1-0.1.2 may have saved a GitHub token in the settings: delete it for the CLI as well as the GUI."""
    try:
        from sealed_lab.gui.update_ui import forget_legacy_token

        forget_legacy_token()
    except Exception:  # noqa: BLE001 - never fail the start over it
        pass


def _install_bundled_sets() -> None:
    """Install the ready-made sets the release zip carries (`bundled_sets/`), if any.  Never blocks the start: a failure is logged and
    the bundle stays for the next start.  Not run by the update script's smoke test (that only imports the package)."""
    try:
        from sealed_lab import bundled

        bundled.install_bundled()
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).exception("ready-made sets could not be installed")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _forget_legacy_token()
    if argv and argv[0] == "cli":
        _warn_interrupted_update()
        _install_bundled_sets()
        from sealed_lab import cli

        return int(cli.main(argv[1:]) or 0)
    if sys.stdout is None or sys.stderr is None:
        _windowless_logging()
    _warn_interrupted_update()
    _record_uncaught()
    _native_crash_log()
    _install_bundled_sets()
    from sealed_lab.gui.app import main as gui_main

    try:
        return int(gui_main() or 0)
    finally:
        _mark_clean_exit()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
