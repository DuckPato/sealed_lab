"""The "Export…" button of the Report tab: save the set's report page (`api.export_set_html`) as one HTML file.

`get_save_file_name`, `ask_open` and `open_file` are module-level hooks so that tests can replace them (nothing blocks)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QStandardPaths, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton, QWidget

from .widgets import new_settings

get_save_file_name = QFileDialog.getSaveFileName


def ask_open(parent: QWidget, path: str) -> bool:
    return QMessageBox.question(parent, "Export", f"Saved {path}\n\nOpen it now?") == QMessageBox.StandardButton.Yes


def open_file(path: str) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(path))


EXPORT_TIP = "Save the written report and the analysis of this set (both formats) as one HTML file: open it in a browser, share it, or print it to PDF"
NOT_FOR_ALL = "Not available for All sets"


def set_enabled(btn: QPushButton, has_set: bool, available: bool = True) -> None:
    """Enable the button for a chosen set; "All sets" (available False) keeps it disabled with an explaining tooltip."""
    btn.setEnabled(bool(has_set) and available)
    btn.setToolTip(EXPORT_TIP if available else NOT_FOR_ALL)


CHEAT_TIP = ("Save one printable page for this set and format (the best color pairs, the best commons and uncommons of each color, the deck shape "
             "to aim for, play or draw, cards to pick up and to avoid) and open it in your browser: it fits one sheet of paper")


def set_cheat_enabled(btn: QPushButton, enabled: bool, every: bool = False, shown: bool = True) -> None:
    """The cheat sheet needs a report on screen; "All sets" and a report that is not there yet keep it disabled with the reason as tooltip."""
    btn.setEnabled(bool(enabled))
    btn.setToolTip(NOT_FOR_ALL if every else CHEAT_TIP if shown else CHEAT_TIP + ".  Available once the report is shown")


def make_cheat_button(tab: Any) -> QPushButton:
    b = QPushButton("Cheat sheet\u2026")
    b.setToolTip(CHEAT_TIP)
    b.setEnabled(False)
    b.clicked.connect(lambda: cheat_sheet(tab))
    return b


def make_button(tab: Any) -> QPushButton:
    b = QPushButton("Export…")
    b.setToolTip(EXPORT_TIP)
    b.clicked.connect(lambda: export_set(tab))
    return b


def default_folder() -> str:
    last = str(new_settings().value("export/folder", "") or "")
    if last and Path(last).is_dir():
        return last
    return QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation) or str(Path.home())


def cheat_sheet(tab: Any) -> str | None:
    """Ask for a file name (default `<SET>_<format>_cheatsheet.html` in the export folder), make the page in the background and open it in the
    browser when it is saved.  Returns the path asked for, or None when cancelled."""
    code, fmt = tab.set_code(), tab.fmt()
    if not code or code == "ALL":
        return None
    name, _ = get_save_file_name(tab, f"Cheat sheet {code}", str(Path(default_folder()) / f"{code}_{fmt}_cheatsheet.html"), "HTML page (*.html)")
    if not name:
        return None
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".html")
    new_settings().setValue("export/folder", str(path.parent))
    return str(path) if tab.start_cheat_sheet(code, fmt, str(path)) else None


def export_set(tab: Any) -> str | None:
    """Ask for a file name, write the page, tell the user (`tab.logMessage`) and offer to open it.  Returns the path written, or None."""
    code = tab.set_code()
    if not code or code == "ALL":
        return None
    name, _ = get_save_file_name(tab, f"Export {code}", str(Path(default_folder()) / f"{code}_sealed_report.html"), "HTML page (*.html)")
    if not name:
        return None
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".html")
    try:
        html = tab.api.export_set_html(code)
        path.write_text(html, encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - shown, never crashes the window
        tab.errorOccurred.emit(f"Could not export {code}: {e}", "")
        return None
    new_settings().setValue("export/folder", str(path.parent))
    msg = f"Exported {code} to {path} ({len(html.encode('utf-8')) / 1024:.0f} KB)"
    tab.logMessage.emit(msg)
    tab.statusMessage.emit(msg)
    if ask_open(tab, str(path)):
        open_file(str(path))
    return str(path)
