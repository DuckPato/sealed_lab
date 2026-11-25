"""Shared widgets, theme helpers and formatting helpers for the sealed_lab GUI."""
from __future__ import annotations

import os
import weakref
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QStyleOptionToolButton,
    QStatusBar,
    QStyledItemDelegate,
    QStylePainter,
    QTableWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

# --------------------------------------------------------------------------- constants

RUN_RULES = {"bo1": (7, 3), "bo3": (4, 2)}  # mirrors config.RUN_RULES (wins, losses to end a run)
BASIC_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
               "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
               "Snow-Covered Mountain", "Snow-Covered Forest"}
COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
MTG_COLORS = {"W": "#f1ead0", "U": "#4a90d9", "B": "#5b5160", "R": "#d9573f", "G": "#3f9d62"}

ARENA_PLACEHOLDER = (
    "Paste an Arena export here, e.g.\n\n"
    "Deck\n"
    "1 Shock (M21) 159\n"
    "2 Llanowar Elves (M19) 314\n"
    "4 Forest\n\n"
    "Sideboard\n"
    "1 Giant Growth\n\n"
    "(Plain \"2 Card Name\" lines work too.)"
)


def run_rule_text(fmt: str) -> str:
    wins, losses = RUN_RULES.get(fmt, (7, 3))
    return f"{fmt.capitalize()}: run ends at {wins} wins or {losses} losses"


# --------------------------------------------------------------------------- settings

def new_settings() -> QSettings:
    """App preferences. SEALED_LAB_SETTINGS (a file path) keeps them in an .ini file instead of the Windows registry:
    tests use it, because QSettings.setPath() has no effect on the registry-backed native format on Windows."""
    ini = os.environ.get("SEALED_LAB_SETTINGS")
    if ini:
        return QSettings(ini, QSettings.Format.IniFormat)
    return QSettings("sealed_lab", "sealed_lab")


def remember_splitter(split, key: str) -> None:
    """Restore a QSplitter's sizes from QSettings and save them whenever the user drags a handle."""
    st = new_settings().value(f"split/{key}")
    if st is not None:
        try:
            split.restoreState(st)
        except TypeError:
            pass
    split.splitterMoved.connect(lambda *_a: new_settings().setValue(f"split/{key}", split.saveState()))


# --------------------------------------------------------------------------- theme

def is_dark(pal: QPalette | None = None) -> bool:
    pal = pal or QApplication.palette()
    return pal.color(QPalette.ColorRole.Window).lightness() < 128


def semantic() -> dict[str, QColor]:
    """Theme-aware semantic colors (good / bad / warn / accent / muted)."""
    if is_dark():
        return {"good": QColor("#56d08f"), "bad": QColor("#ff7f78"), "warn": QColor("#e6b94d"),
                "accent": QColor("#5aa9ff")}
    return {"good": QColor("#1a7f4b"), "bad": QColor("#c62828"), "warn": QColor("#a35f00"),
            "accent": QColor("#1f6feb")}


def _dark_palette() -> QPalette:
    p = QPalette()
    c = QColor
    p.setColor(QPalette.Window, c(37, 38, 41))
    p.setColor(QPalette.WindowText, c(228, 230, 233))
    p.setColor(QPalette.Base, c(28, 29, 32))
    p.setColor(QPalette.AlternateBase, c(46, 48, 52))
    p.setColor(QPalette.ToolTipBase, c(46, 48, 52))
    p.setColor(QPalette.ToolTipText, c(228, 230, 233))
    p.setColor(QPalette.Text, c(228, 230, 233))
    p.setColor(QPalette.PlaceholderText, c(140, 144, 150))
    p.setColor(QPalette.Button, c(52, 54, 59))
    p.setColor(QPalette.ButtonText, c(228, 230, 233))
    p.setColor(QPalette.BrightText, c(255, 90, 90))
    p.setColor(QPalette.Link, c(90, 169, 255))
    p.setColor(QPalette.Highlight, c(48, 108, 196))
    p.setColor(QPalette.HighlightedText, c(255, 255, 255))
    p.setColor(QPalette.Light, c(70, 72, 78))
    p.setColor(QPalette.Midlight, c(60, 62, 67))
    p.setColor(QPalette.Mid, c(96, 100, 109))             # outlines of badges / chips / axes: visible on the dark window (it was nearly the window color)
    p.setColor(QPalette.Dark, c(20, 21, 23))
    p.setColor(QPalette.Shadow, c(10, 10, 11))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c(120, 123, 128))
    return p


ACCENT = "#2f6fd6"            # primary buttons, the selected tab, focus rings (white text on it: contrast 4.9)

# Color tokens of the two looks; everything neutral that is not here comes from the palette.
_TOKENS = {
    False: dict(accent=ACCENT, accent_hover="#3d7fe8", accent_press="#2559ad", accent_text="#1b5bc4",
                accent_tint="rgba(47,111,214,0.10)", border="#b9c0cb", border_soft="#d3d8e0", btn="#f6f7f9", btn_hover="#e9eef9",
                btn_press="#dbe4f6", hover="#e9eef9", scroll="#b4bbc7", scroll_hover="#8f98a8", tab_idle="#40474f", btn_off="#e4e7ec", off_text="#848c98",
                ind_border="#7b8594", ind_fill="#ffffff", ind_off="#c9ced6"),
    True: dict(accent=ACCENT, accent_hover="#3d7fe8", accent_press="#2559ad", accent_text="#7db3ff",
               accent_tint="rgba(90,169,255,0.14)", border="#5a5e66", border_soft="#44474e", btn="#3a3c42", btn_hover="#454a57",
               btn_press="#343b4d", hover="#3b4254", scroll="#5d616b", scroll_hover="#838895", tab_idle="#c3c7ce", btn_off="#2c2e33", off_text="#8b909a",
               ind_border="#959ba6", ind_fill="#1c1d20", ind_off="#55595f"),
}


def _chevron(color: str) -> str:
    """Path of a small down-chevron PNG in `color` (style sheets can only point to files); written once into the temp folder.
    "" when it cannot be made: the combo boxes then keep the style's own arrow."""
    import tempfile

    try:
        d = Path(tempfile.gettempdir()) / "sealed_lab_qss"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"chevron_{color.lstrip('#')}.png"
        if not f.exists():
            pm = QPixmap(48, 48)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(color), 5.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawPolyline([QPointF(10, 17), QPointF(24, 31), QPointF(38, 17)])
            p.end()
            if not pm.save(str(f), "PNG"):
                return ""
        return f.as_posix()
    except Exception:  # noqa: BLE001 - cosmetic
        return ""


def _hairline(color: str, horizontal: bool) -> str:
    """A 3 px line of `color` centred in a transparent strip (a splitter handle that is a slim divider, not a grey slab)."""
    axis = "x1:0, y1:0, x2:0, y2:1" if horizontal else "x1:0, y1:0, x2:1, y2:0"
    return (f"qlineargradient({axis}, stop:0 transparent, stop:0.33 transparent, stop:0.34 {color}, stop:0.66 {color}, "
            f"stop:0.67 transparent, stop:1 transparent)")


def _indicator_images(t: dict) -> dict[str, str]:
    """Paths of the check box / radio button indicator PNGs for the color tokens `t` (written once into the temp folder).  Qt's own
    indicators vanish on the dark window (their outline is darker than it); these have a clear outline in both looks.  {} when they
    cannot be made: the controls then keep the style's own look."""
    import tempfile

    try:
        d = Path(tempfile.gettempdir()) / "sealed_lab_qss"
        d.mkdir(parents=True, exist_ok=True)
        key = f"{t['ind_border'].lstrip('#')}_{t['ind_fill'].lstrip('#')}_{t['accent'].lstrip('#')}"
        names = {}

        def make(kind: str, state: str) -> str:
            f = d / f"ind_{kind}_{state}_{key}.png"
            names[f"{kind}_{state}"] = f.as_posix()
            if f.exists():
                return f.as_posix()
            pm = QPixmap(64, 64)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            on = state.startswith("on")
            off = state.endswith("off")
            edge = QColor(t["ind_off"] if off else (t["accent"] if on or state == "hover" else t["ind_border"]))
            fill = QColor(t["ind_fill"])
            if on and kind == "check":
                fill = QColor(t["accent"])
            p.setPen(QPen(edge, 5.0))
            p.setBrush(QBrush(fill))
            r = QRectF(4, 4, 56, 56)
            if kind == "radio":
                p.drawEllipse(r)
                if on:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(QColor(t["accent"])))
                    p.drawEllipse(QRectF(19, 19, 26, 26))
            else:
                p.drawRoundedRect(r, 10, 10)
                if on:
                    pen = QPen(QColor("white"), 7.0)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    p.setPen(pen)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPolyline([QPointF(16, 33), QPointF(27, 44), QPointF(48, 20)])
            p.end()
            if not pm.save(str(f), "PNG"):
                raise OSError(str(f))
            return f.as_posix()

        for kind in ("check", "radio"):
            for state in ("off", "hover", "on", "off_off", "on_off"):
                make(kind, state)
        return names
    except Exception:  # noqa: BLE001 - cosmetic
        return {}


def app_qss(dark: bool) -> str:
    """The application style sheet, for the light or the dark palette: one accent color (primary buttons, the selected tab, focus),
    larger main tabs, rounded panels and fields, clear hover / focus states, slim scroll bars.  Sizes are in em (they follow the font);
    no rule sets a text color on labels, so the palette-based greys and the green / red of the values are untouched."""
    t = _TOKENS[dark]
    chev = _chevron("#c3c7ce" if dark else "#40474f")
    combo_arrow = (f"QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 1.6em; border: 0; }} "
                   f"QComboBox::down-arrow {{ image: url({chev}); width: 0.8em; height: 0.8em; }}") if chev else ""
    im = _indicator_images(t)
    indicators = "".join(
        f"Q{cls}::indicator {{ width: 1.05em; height: 1.05em; image: url({im[kind + '_off']}); }} "
        f"Q{cls}::indicator:hover {{ image: url({im[kind + '_hover']}); }} "
        f"Q{cls}::indicator:checked {{ image: url({im[kind + '_on']}); }} "
        f"Q{cls}::indicator:disabled {{ image: url({im[kind + '_off_off']}); }} "
        f"Q{cls}::indicator:checked:disabled {{ image: url({im[kind + '_on_off']}); }} "
        for cls, kind in (("CheckBox", "check"), ("RadioButton", "radio"))) if im else ""
    return f"""
/* ---- buttons */
QPushButton {{ background-color: {t['btn']}; border: 1px solid {t['border']}; border-radius: 0.4em; padding: 0.3em 0.9em; }}
QPushButton:hover {{ background-color: {t['btn_hover']}; border-color: {t['accent']}; }}
QPushButton:pressed {{ background-color: {t['btn_press']}; }}
QPushButton:focus {{ border: 1px solid {t['accent']}; }}
QPushButton:checked {{ background-color: {t['accent_tint']}; border-color: {t['accent']}; }}
QPushButton:disabled {{ background-color: transparent; border-color: {t['border_soft']}; }}
QPushButton#primary {{ background-color: {t['accent']}; color: white; border: 1px solid {t['accent_press']};
    padding: 0.4em 1.5em; font-weight: 600; }}
QPushButton#primary:hover {{ background-color: {t['accent_hover']}; }}
QPushButton#primary:pressed {{ background-color: {t['accent_press']}; }}
QPushButton#primary:focus {{ border: 2px solid {t['accent_text']}; }}
QPushButton#primary:disabled {{ background-color: {t['btn_off']}; color: {t['off_text']}; border: 1px solid {t['border_soft']}; }}
QToolButton {{ border: 1px solid {t['border_soft']}; border-radius: 0.3em; padding: 0 2px; }}
QToolButton:disabled {{ border-color: transparent; }}
QToolButton:hover {{ background-color: {t['hover']}; border-color: {t['border']}; }}
QToolButton:pressed, QToolButton:checked {{ background-color: {t['btn_press']}; }}
QToolButton:focus {{ border-color: {t['accent']}; }}
QToolButton#foldWeak:checked, QToolButton#foldWeak:pressed {{ background-color: transparent; }}        /* a text link that folds rows, not a pressed button */

/* ---- tabs: the main ones large, the nested ones lighter; the selected one carries the accent */
QTabWidget::pane {{ border: 0; border-top: 1px solid {t['border_soft']}; }}
QTabBar::tab {{ padding: 0.4em 1em; margin: 0 2px 0 0; border: 0; border-bottom: 2px solid transparent; color: {t['tab_idle']}; }}
QTabBar::tab:hover {{ background-color: {t['hover']}; }}
QTabBar::tab:selected {{ color: {t['accent_text']}; border-bottom: 2px solid {t['accent']}; }}
QTabWidget#mainTabs > QStackedWidget {{ border: 0; border-top: 1px solid {t['border_soft']}; }}
QTabWidget#mainTabs::pane {{ top: -1px; }}
QTabWidget#mainTabs > QTabBar::tab {{ padding: 0.65em 1.7em; font-size: 1.12em; border-bottom: 3px solid transparent; }}
QTabWidget#mainTabs > QTabBar::tab:selected {{ border-bottom: 3px solid {t['accent']}; }}

/* ---- fields */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{ border: 1px solid {t['border']}; border-radius: 0.35em;
    padding: 0.15em 0.4em; }}
QPlainTextEdit, QTextEdit {{ padding: 0.3em; }}
QComboBox {{ padding: 0.15em 0.1em 0.15em 0.4em; }}
{combo_arrow}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {t['accent']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {t['accent']}; }}
QCheckBox:focus, QRadioButton:focus {{ color: {t['accent_text']}; }}
QCheckBox, QRadioButton {{ spacing: 0.45em; }}
{indicators}
QProgressBar#busyBar {{ border: 1px solid {t['ind_border']}; border-radius: 0.3em; background-color: palette(base); text-align: center; }}
QProgressBar#busyBar::chunk {{ background-color: {t['accent']}; border-radius: 0.2em; }}

/* ---- panels and lists */
QFrame[card="true"] {{ border: 1px solid {t['border_soft']}; border-radius: 0.5em; }}
QFrame[card="true"][flash="true"] {{ border: 2px solid {t['accent']}; background-color: {t['accent_tint']}; }}
QListWidget#reportNav {{ background: transparent; border: 0; outline: 0; }}
QListWidget#reportNav::item {{ padding: 0.3em 0.6em; border-radius: 0.4em; }}
QListWidget#reportNav::item:hover {{ background-color: {t['hover']}; }}
QListWidget#reportNav::item:selected {{ background-color: {t['accent_tint']}; color: {t['accent_text']}; font-weight: 600; }}
QFrame#datasetPanel {{ margin: 8px 12px 4px 12px; }}
QFrame#datasetPanel QTreeWidget::item {{ min-height: 1.7em; }}
QFrame#watchBanner {{ background-color: {t['accent_tint']}; border: 1px solid {t['accent']}; border-radius: 0.5em; }}
QFrame#poolStrip {{ background-color: {t['accent_tint']}; border: 1px solid {t['border_soft']}; border-radius: 0.5em; }}
QFrame#floatingPreview {{ background-color: palette(window); border: 1px solid {t['accent']}; border-radius: 0.6em; }}
QGroupBox {{ border: 1px solid {t['border_soft']}; border-radius: 0.5em; margin-top: 1.1em; padding: 0.5em; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 0.8em; padding: 0 0.3em; font-weight: 600; }}
QTableView, QTreeView, QListView {{ border: 1px solid {t['border_soft']}; border-radius: 0.4em; selection-background-color: {t['accent']};
    selection-color: white; }}
QHeaderView::section {{ background-color: palette(window); border: 0; border-bottom: 1px solid {t['border']}; padding: 0.3em 0.4em;
    font-weight: 600; }}
QMenuBar::item:selected, QMenu::item:selected {{ background-color: {t['accent']}; color: white; }}
QToolTip {{ background-color: palette(tooltip-base); color: palette(tooltip-text); border: 1px solid {t['border']}; padding: 0.25em; }}

/* ---- slim scroll bars */
QScrollBar:vertical {{ background: transparent; width: 0.9em; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 0.9em; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t['scroll']}; border-radius: 0.3em; min-height: 2em; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['scroll']}; border-radius: 0.3em; min-width: 2em; margin: 2px; }}
QScrollBar::handle:hover {{ background: {t['scroll_hover']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: 0; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- splitter handles (a drag affordance) */
QSplitter::handle:horizontal {{ width: 7px; background: {_hairline(t['border_soft'], False)}; }}
QSplitter::handle:vertical {{ height: 7px; background: {_hairline(t['border_soft'], True)}; }}
QSplitter::handle:horizontal:hover {{ background: {_hairline(t['accent'], False)}; }}
QSplitter::handle:vertical:hover {{ background: {_hairline(t['accent'], True)}; }}
"""


def apply_theme(app: QApplication, mode: str | None = None) -> None:
    """Fusion style; palette and style sheet follow the OS color scheme (or mode 'dark'/'light'/'auto')."""
    mode = (mode or os.environ.get("SEALED_LAB_THEME") or "auto").lower()
    app.setStyle("Fusion")

    def refresh(*_args) -> None:
        scheme = app.styleHints().colorScheme()
        want_dark = mode == "dark" or (mode == "auto" and scheme == Qt.ColorScheme.Dark)
        pal = _dark_palette() if want_dark else app.style().standardPalette()
        if not want_dark and is_dark(pal):
            pal = app.style().standardPalette()
        pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))          # selection = the accent color, in both looks
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
        link = QColor("#7fb2ff") if want_dark else QColor(ACCENT)          # rich-text links (the 17Lands credit) readable in both looks
        pal.setColor(QPalette.ColorRole.Link, link)
        pal.setColor(QPalette.ColorRole.LinkVisited, link)
        app.setPalette(pal)
        app.setStyleSheet(app_qss(want_dark))

    refresh()
    if mode == "auto":
        try:
            app.styleHints().colorSchemeChanged.connect(refresh)
        except Exception:  # pragma: no cover - old Qt
            pass


def mono_font(delta: float = 0.0) -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    for fam in ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Menlo"):
        if fam in QFontDatabase.families():
            f.setFamily(fam)
            break
    f.setStyleHint(QFont.StyleHint.Monospace)
    if delta:
        f.setPointSizeF(max(6.0, f.pointSizeF() + delta))
    return f


def sized_font(base: QFont, pt_delta: float = 0.0, bold: bool = False) -> QFont:
    f = QFont(base)
    f.setPointSizeF(max(6.0, base.pointSizeF() + pt_delta))
    f.setBold(bold)
    return f


def text_w(widget: QWidget, text: str, font: QFont | None = None) -> int:
    """Pixel width of `text` in the widget's (or the given) font: sizes of text containers come from here, never from constants."""
    return QFontMetrics(font or widget.font()).horizontalAdvance(text)


def row_height(font: QFont, min_px: int = 24) -> int:
    """Height of a table / list row: one line of `font` plus the style's padding (never below `min_px`)."""
    return max(min_px, QFontMetrics(font).lineSpacing() + 8)


def with_full_text(tip: str, text: str) -> str:
    """Tooltip that also shows `text` in full (for text that is elided): an existing tooltip keeps its content and gets the full
    text as a bold first paragraph; without a tooltip the text itself is the tooltip."""
    from html import escape

    if not tip:
        return text
    if text in tip or escape(text) in tip:
        return tip
    if tip.startswith("<html>") and "<td>" in tip:
        return tip.replace("<td>", f"<td><p style='margin:0 0 5px 0'><b>{escape(text)}</b></p>", 1)
    if "<" not in tip:
        return f"{text}\n{tip}"
    return f"<p style='margin:0 0 5px 0'><b>{escape(text)}</b></p>{tip}"


def elide_wrapped(fm: QFontMetrics, text: str, width: int, height: int) -> str:
    """`text` shortened with "…" so that it fits `width` x `height` when drawn word-wrapped (for text painted by hand)."""
    from PySide6.QtCore import QRect

    flags = int(Qt.TextFlag.TextWordWrap)

    def fits(t: str) -> bool:
        return fm.boundingRect(QRect(0, 0, max(width, 1), 100000), flags, t).height() <= height

    if fits(text):
        return text
    words = text.split()
    while len(words) > 1:
        words.pop()
        if fits(" ".join(words) + "…"):
            return " ".join(words) + "…"
    one = words[0] if words else ""
    while len(one) > 1 and not fits(one + "…"):
        one = one[:-1]
    return one + "…"


def muted(label: QLabel) -> QLabel:
    label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
    return label


def primary_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("primary")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


class ElidedLabel(QLabel):
    """One-line label for text that may be longer than its room (paths, card names, messages): shows "…" where it does not fit
    and then offers the full text as tooltip (an existing tooltip is kept and gets the full text added).  Its minimum width is a
    few characters, so layouts and splitters can squeeze it without cutting anything silently."""

    elides = True

    def __init__(self, text: str = "", mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight, parent: QWidget | None = None,
                 min_chars: int = 6):
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        self._min_chars = min_chars
        self._base_tip = ""
        self._elided = False
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def full_text(self) -> str:
        return self._full

    def is_elided(self) -> bool:
        return self._elided

    def setText(self, text: str) -> None:
        self._full = text
        super().setText(text)
        self.updateGeometry()
        self._sync()

    def setToolTip(self, tip: str) -> None:
        self._base_tip = tip
        self._sync()

    def minimumSizeHint(self) -> QSize:
        fm = self.fontMetrics()
        w = min(fm.horizontalAdvance(self._full), fm.horizontalAdvance("n" * self._min_chars) + fm.horizontalAdvance("…"))
        return QSize(w + 2 * self.frameWidth() + 2 * max(self.margin(), 0), super().minimumSizeHint().height())

    def _sync(self) -> None:
        w = self.contentsRect().width() - 2 * max(self.margin(), 0)
        self._elided = bool(self._full) and w > 0 and self.fontMetrics().horizontalAdvance(self._full) > w
        QLabel.setToolTip(self, with_full_text(self._base_tip, self._full) if self._elided else self._base_tip)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        self._sync()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        self.drawFrame(p)
        m = max(self.margin(), 0)
        r = self.contentsRect().adjusted(m, 0, -m, 0)
        p.setPen(self.palette().color(self.foregroundRole()))
        text = self.fontMetrics().elidedText(self._full, self._mode, r.width())
        align = self.alignment() | Qt.AlignmentFlag.AlignVCenter
        p.drawText(r, int(align), text)


class ElidedStatusBar(QStatusBar):
    """Status bar whose temporary message is elided with "…" when the window is too narrow for it (Qt would cut it off silently); the
    whole message is the bar's tooltip.  `currentMessage()` returns what is shown."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._full = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.clearMessage)

    def full_message(self) -> str:
        return self._full

    def _room(self) -> int:
        taken = sum(w.width() + 8 for w in self.findChildren(QLabel) if w.isVisible() and w.parent() is self)
        return max(self.width() - taken - 24, 60)

    def showMessage(self, text: str, timeout: int = 0) -> None:
        self._full = text
        self._timer.stop()
        self._show_fitted()
        if timeout > 0:
            self._timer.start(timeout)

    def clearMessage(self) -> None:
        self._full = ""
        QStatusBar.setToolTip(self, "")
        super().clearMessage()

    def _show_fitted(self) -> None:
        fitted = QFontMetrics(self.font()).elidedText(self._full, Qt.TextElideMode.ElideRight, self._room())
        super().showMessage(fitted)
        QStatusBar.setToolTip(self, self._full if fitted != self._full else "")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._full:
            self._show_fitted()

    def clipped_texts(self) -> list[tuple[str, float, float]]:
        shown = self.currentMessage()
        if self._full and shown != self._full and self._full not in self.toolTip():
            return [(self._full, float(QFontMetrics(self.font()).horizontalAdvance(self._full)), float(self._room()))]
        return []


class FlowLayout(QLayout):
    """Left-to-right layout that starts a new row when the widgets do not fit the width (a row of buttons in a narrow panel wraps
    instead of forcing the panel wider).  Honours height-for-width, so it works inside scroll areas and boxes."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6, clamp: bool = False):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self._clamp = clamp                  # True: a widget wider than the row is squeezed to the row (an elided label) instead of sticking out
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        w = h = 0
        for it in self._items:
            s = it.minimumSize()
            w, h = max(w, s.width()), max(h, s.height())
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def _do_layout(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = r.x(), r.y(), 0
        for it in self._items:
            if it.isEmpty():
                continue
            s = it.sizeHint()
            if self._clamp and r.width() > 0 and s.width() > r.width():
                s = QSize(r.width(), s.height())
            if it.hasHeightForWidth():                 # wrapped text may need more lines at this width than its size hint says
                s = QSize(s.width(), max(s.height(), it.heightForWidth(s.width())))
            if x > r.x() and x + s.width() > r.right() + 1:
                x, y, row_h = r.x(), y + row_h + self._spacing, 0
            if apply:
                it.setGeometry(QRect(QPoint(x, y), s))
            x += s.width() + self._spacing
            row_h = max(row_h, s.height())
        return y + row_h - rect.y() + m.bottom()


class HeaderLayout(QLayout):
    """A title and the widget that belongs beside it (basic-land steppers ...): on one row, the extra widget at the right, while the
    width allows; on two rows (the extra widget under the title) when it does not.  The title is never squeezed to make room.

    Unlike a flow layout it has a plain minimum height - the one of the mode the current width calls for - so a splitter that sizes
    the box from `minimumSizeHint` reserves exactly what is needed."""

    def __init__(self, parent: QWidget | None = None, gap: int = 12):
        super().__init__(parent)
        self._items: list = []
        self._gap = gap
        self._stacked = False
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def _parts(self):
        t = self._items[0]
        e = self._items[1] if len(self._items) > 1 else None
        return t, e

    def _height(self, stacked: bool) -> int:
        t, e = self._parts()
        th = t.sizeHint().height()
        eh = e.sizeHint().height() if e is not None else 0
        return th + self._gap // 2 + eh if (stacked and e is not None) else max(th, eh)

    def sizeHint(self) -> QSize:
        if not self._items:
            return QSize(0, 0)
        t, e = self._parts()
        w = t.sizeHint().width() + (self._gap + e.sizeHint().width() if e is not None else 0)
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), self._height(self._stacked) + m.top() + m.bottom())

    def minimumSize(self) -> QSize:
        if not self._items:
            return QSize(0, 0)
        t, e = self._parts()
        w = max(t.sizeHint().width(), e.minimumSize().width() if e is not None else 0)
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), self._height(self._stacked) + m.top() + m.bottom())

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        if not self._items:
            return
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        t, e = self._parts()
        tw, th = t.sizeHint().width(), t.sizeHint().height()
        ew, eh = (e.sizeHint().width(), e.sizeHint().height()) if e is not None else (0, 0)
        stacked = e is not None and tw + self._gap + ew > r.width()
        if stacked != self._stacked:
            self._stacked = stacked
            self.invalidate()                      # the parent asks for the new minimum height
        if e is None or not stacked:
            h = max(th, eh)
            t.setGeometry(QRect(r.x(), r.y() + (h - th) // 2, min(tw, r.width()), th))
            if e is not None:
                e.setGeometry(QRect(r.right() + 1 - ew, r.y() + (h - eh) // 2, ew, eh))
        else:
            t.setGeometry(QRect(r.x(), r.y(), min(tw, r.width()), th))
            e.setGeometry(QRect(r.x(), r.y() + th + self._gap // 2, min(ew, r.width()), eh))


class ElidedToolButton(QToolButton):
    """Tool button (the header of a Collapsible section) whose text is elided with "…" when the button is narrower than the text; the
    full text is added to the tooltip.  It can shrink to a few characters, so a long title never widens the panel it sits in."""

    elides = True

    def __init__(self, parent: QWidget | None = None, min_chars: int = 10):
        super().__init__(parent)
        self._base_tip = ""
        self._min_chars = min_chars
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)      # (Minimum would forbid shrinking below the full text)

    def _chrome(self) -> int:
        """Width of everything but the text (arrow / icon, margins)."""
        return max(super().sizeHint().width() - self.fontMetrics().horizontalAdvance(self.text()), 0)

    def full_text(self) -> str:
        return self.text()

    def is_elided(self) -> bool:
        return self.fontMetrics().horizontalAdvance(self.text()) > self.width() - self._chrome()

    def setText(self, text: str) -> None:
        super().setText(text)
        self.updateGeometry()
        self._sync()

    def setToolTip(self, tip: str) -> None:
        self._base_tip = tip
        self._sync()

    def _sync(self) -> None:
        QToolButton.setToolTip(self, with_full_text(self._base_tip, self.text()) if self.is_elided() else self._base_tip)

    def minimumSizeHint(self) -> QSize:
        fm = self.fontMetrics()
        w = min(fm.horizontalAdvance(self.text()), fm.horizontalAdvance("n" * self._min_chars) + fm.horizontalAdvance("…"))
        return QSize(self._chrome() + w, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync()

    def paintEvent(self, event) -> None:
        p = QStylePainter(self)
        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)
        room = max(self.width() - self._chrome(), 8)
        opt.text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, room)
        p.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, opt)


class ElidedCombo(QComboBox):
    """Combo box whose current text is elided with "…" when the field is narrower than the text (the full text is added to the
    tooltip) and that can shrink below the width of its longest item."""

    elides = True

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._base_tip = ""
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(10)
        self.currentIndexChanged.connect(lambda _i: self._sync())

    def setToolTip(self, tip: str) -> None:
        self._base_tip = tip
        self._sync()

    def _field_width(self) -> int:
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        r = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt, QStyle.SubControl.SC_ComboBoxEditField, self)
        return r.width() - 4

    def is_elided(self) -> bool:
        return bool(self.currentText()) and self.fontMetrics().horizontalAdvance(self.currentText()) > self._field_width()

    def _sync(self) -> None:
        QComboBox.setToolTip(self, with_full_text(self._base_tip, self.currentText()) if self.is_elided() else self._base_tip)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync()

    def sizeHint(self) -> QSize:
        """Wide enough for the longest item (the minimum stays a few characters: the text is elided when the field is narrower)."""
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        fm = self.fontMetrics()
        longest = max([fm.horizontalAdvance(self.itemText(i)) for i in range(self.count())] + [0])
        full = self.style().sizeFromContents(QStyle.ContentsType.CT_ComboBox, opt, QSize(longest, fm.height()), self)
        sh = super().sizeHint()
        return QSize(min(max(sh.width(), full.width()), self.maximumWidth()), sh.height())

    def paintEvent(self, event) -> None:
        p = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        p.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        r = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt, QStyle.SubControl.SC_ComboBoxEditField, self)
        icon_w = (opt.iconSize.width() + 4) if not opt.currentIcon.isNull() else 0
        opt.currentText = self.fontMetrics().elidedText(opt.currentText, Qt.TextElideMode.ElideRight, max(r.width() - 4 - icon_w, 8))
        p.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


# --------------------------------------------------------------------------- formatting

def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def pct_interval(lo: float, hi: float, digits: int = 1) -> str:
    return f"80%: {lo * 100:.{digits}f}–{hi * 100:.{digits}f}"


def signed(x: float | None, digits: int = 2) -> str:
    return "" if x is None else f"{x:+.{digits}f}"


def color_title(colors: str, splash: str) -> str:
    base = " / ".join(COLOR_NAMES.get(c, c) for c in colors) or "Colorless"
    if splash:
        base += "  + splash " + " / ".join(COLOR_NAMES.get(c, c) for c in splash)
    return base


def color_tag(colors: str, splash: str) -> str:
    """Short tab label like 'WU+r'."""
    s = colors or "—"
    if splash:
        s += "+" + splash.lower()
    return s


def short_error(tb: str, limit: int = 300) -> str:
    lines = [ln for ln in tb.strip().splitlines() if ln.strip()]
    msg = lines[-1] if lines else "Unknown error"
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


def show_message(parent: QWidget | None, title: str, text: str, details: str = "",
                 icon: QMessageBox.Icon = QMessageBox.Icon.Critical) -> QMessageBox:
    """Non-blocking (window-modal) message box."""
    box = QMessageBox(icon, title, text, QMessageBox.StandardButton.Ok, parent)
    if details:
        box.setDetailedText(details)
    box.setWindowModality(Qt.WindowModality.WindowModal)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    box.open()
    return box


# --------------------------------------------------------------------------- small widgets

class PasteEdit(QPlainTextEdit):
    """Monospace paste box for decklists.  `nameHovered(name)` fires when the mouse moves over a card line (the Pool tab shows that
    card in its preview); "" when it leaves the text."""

    nameHovered = Signal(str)

    def __init__(self, placeholder: str = ARENA_PLACEHOLDER, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFont(mono_font())
        self.setPlaceholderText(placeholder)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabChangesFocus(True)
        self.setAcceptDrops(False)
        self.viewport().setMouseTracking(True)
        self._hover_name = ""

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        from sealed_lab.deck.arena import card_name_of_line

        name = card_name_of_line(self.cursorForPosition(event.position().toPoint()).block().text()) or ""
        if name and name != self._hover_name:
            self._hover_name = name
            self.nameHovered.emit(name)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._hover_name = ""


class LogPane(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(mono_font(-1))
        self.setMaximumBlockCount(5000)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setAcceptDrops(False)
        self.setPlaceholderText("Log output appears here.")

    def append_line(self, text: str) -> None:
        self.appendPlainText(text.rstrip("\n"))
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class FullTextTipDelegate(QStyledItemDelegate):
    """Item delegate for views whose cells elide long text with "…": hovering a cell whose text does not fit shows the full text as
    tooltip (added to the cell's own tooltip, if it has one).  Cells that fit keep their normal tooltip."""

    full_text_tips = True

    def helpEvent(self, event, view, option, index) -> bool:
        if event.type() == QEvent.Type.ToolTip and index.isValid():
            text = index.data(Qt.ItemDataRole.DisplayRole)
            text = "" if text is None else str(text)
            if text:
                font = index.data(Qt.ItemDataRole.FontRole)
                fm = QFontMetrics(font if isinstance(font, QFont) else option.font)
                room = option.rect.width() - 2 * (view.style().pixelMetric(QStyle.PixelMetric.PM_FocusFrameHMargin) + 1)
                if not option.icon.isNull():
                    room -= option.decorationSize.width() + 4
                if fm.horizontalAdvance(text.splitlines()[0]) > room:
                    QToolTip.showText(event.globalPos(), with_full_text(str(index.data(Qt.ItemDataRole.ToolTipRole) or ""), text), view)
                    return True
        return super().helpEvent(event, view, option, index)


def make_table(columns: list[str], parent: QWidget | None = None, cls=QTableWidget, **kw) -> QTableWidget:
    t = cls(0, len(columns), parent, **kw) if cls is not QTableWidget else QTableWidget(0, len(columns), parent)
    t.setHorizontalHeaderLabels(columns)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(row_height(t.font()))
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    t.setShowGrid(False)
    t.setAlternatingRowColors(True)
    t.horizontalHeader().setHighlightSections(False)
    t.setWordWrap(False)
    t.setItemDelegate(FullTextTipDelegate(t))               # text that is elided in a narrow column shows in full on hover
    return t


class MetricCard(QFrame):
    """Caption, big value, small sub-line."""

    def __init__(self, caption: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setProperty("card", True)
        self.setBackgroundRole(QPalette.ColorRole.AlternateBase)
        self.setAutoFillBackground(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(1)
        self.caption = muted(QLabel(caption))
        self.caption.setFont(sized_font(self.font(), -1))
        self.caption.setWordWrap(True)                  # captions wrap; the big number below stays on one line
        self.value = QLabel("—")
        self.value.setFont(sized_font(self.font(), 7, bold=True))
        self.sub = muted(QLabel(""))
        self.sub.setFont(sized_font(self.font(), -1))
        self.sub.setWordWrap(True)
        lay.addWidget(self.caption)
        lay.addWidget(self.value)
        lay.addWidget(self.sub)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def set_tip(self, text: str) -> None:
        for w in (self, self.caption, self.value, self.sub):
            w.setToolTip(text)

    def set(self, value: str, sub: str = "", role: str | None = None, tip: str | None = None,
            sub_role: str | None = None) -> None:
        self.value.setText(value)
        self.sub.setText(sub)
        self.sub.setVisible(bool(sub))
        pal = self.value.palette()
        col = semantic().get(role) if role else None
        pal.setColor(QPalette.ColorRole.WindowText, col or QApplication.palette().color(QPalette.ColorRole.WindowText))
        self.value.setPalette(pal)
        sp = self.sub.palette()
        scol = semantic().get(sub_role) if sub_role else None
        sp.setColor(QPalette.ColorRole.PlaceholderText,
                    scol or QApplication.palette().color(QPalette.ColorRole.PlaceholderText))
        self.sub.setPalette(sp)
        if tip is not None:
            self.set_tip(tip)


class TipLabel(QLabel):
    """Rich-text label whose parts carry their own tooltips: wrap a part in ``tipped(text, key)`` and register
    ``tips[key]``; hovering the part shows the tooltip next to the mouse (internally the part is a link, drawn like
    normal text)."""

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.tips: dict[str, str] = {}
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.linkHovered.connect(self._hovered)

    def set_content(self, html: str, tips: dict[str, str] | None = None) -> None:
        self.tips = dict(tips or {})
        self.setText(html)

    def _hovered(self, url: str) -> None:
        if url.startswith("tip:") and url[4:] in self.tips:
            self.setCursor(Qt.CursorShape.WhatsThisCursor)
            QToolTip.showText(QCursor.pos(), self.tips[url[4:]], self)
        else:
            self.unsetCursor()
            QToolTip.hideText()


def tipped(text: str, key: str) -> str:
    """Mark `text` (HTML) as the hover target of tooltip `key` of a TipLabel."""
    col = QApplication.palette().color(QPalette.ColorRole.WindowText).name()
    return f"<a href='tip:{key}' style='text-decoration:none; color:{col}'>{text}</a>"


class ColorPips(QWidget):
    """Row of colored circles for a color string; splash colors drawn smaller."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._colors = ""
        self._splash = ""
        self._apply_size()

    def _diameter(self) -> float:
        """Circle size: scaled from the font (the letter inside is text), not fixed pixels."""
        return max(26.0, round(self.fontMetrics().height() * 1.7))

    def _apply_size(self) -> None:
        d = self._diameter()
        ds = round(d * 0.7)
        self.setFixedHeight(int(d) + 2)
        n, k = len(self._colors), len(self._splash)
        self.setFixedWidth(max(4, int(2 + n * (d + 4) + k * (ds + 4))))
        self.update()

    def set_colors(self, colors: str, splash: str) -> None:
        self._colors, self._splash = colors, splash
        self._apply_size()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._apply_size()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x = 2.0
        d0 = self._diameter()
        for ch, d in [(c, d0) for c in self._colors] + [(c, float(round(d0 * 0.7))) for c in self._splash]:
            col = QColor(MTG_COLORS.get(ch, "#999999"))
            y = (self.height() - d) / 2
            p.setBrush(QBrush(col))
            p.setPen(QPen(QColor(0, 0, 0, 90), 1))
            p.drawEllipse(QRectF(x, y, d, d))
            p.setPen(QColor("#222" if ch == "W" else "#fff"))
            f = p.font()
            f.setBold(True)
            f.setPixelSize(max(8, int(d * 0.5)))
            p.setFont(f)
            p.drawText(QRectF(x, y, d, d), Qt.AlignmentFlag.AlignCenter, ch)
            x += d + 4


class BreakdownBars(QWidget):
    """Diverging horizontal bars for the contributions to the game win rate, in percentage points (pp).

    The model adds log-odds; the bars show them as pp applied one after the other starting from 50% (the first bar, the
    baseline, goes from 50% to the baseline win rate), so the pp add up to the deck's win rate - 50%.  The log-odds are
    in the tooltip of each row.  Falls back to the local slope q(1-q) when the parts do not add up to the deck's
    log-odds."""

    ROW = 26                    # least row height; the real one grows with the font

    def row_h(self) -> int:
        return max(self.ROW, self.fontMetrics().lineSpacing() + 10)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._items: list[tuple[str, float]] = []      # (key, log-odds)
        self._pp: list[float] = []
        self._exact = True
        self._game_wr = 0.5
        self._hover = -1
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @staticmethod
    def label_of(key: str) -> str:
        from .helptext import BREAKDOWN_LABELS
        return BREAKDOWN_LABELS.get(key, key)

    def set_items(self, items: dict[str, float], game_wr: float = 0.5) -> None:
        import math

        self._items = list(items.items())
        self._game_wr = game_wr
        total = sum(v for _, v in self._items)
        q = min(max(game_wr, 1e-6), 1 - 1e-6)
        self._exact = abs(total - math.log(q / (1 - q))) < 0.02

        def sig(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))
        if self._exact:
            cum, prev, pp = 0.0, 0.5, []
            for _k, v in self._items:
                cum += v
                cur = sig(cum)
                pp.append(100.0 * (cur - prev))
                prev = cur
            self._pp = pp
        else:
            slope = max(0.01, game_wr * (1 - game_wr))
            self._pp = [v * slope * 100 for _, v in self._items]
        self._hover = -1
        self.setFixedHeight(max(1, len(self._items)) * self.row_h() + 4)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(max(260, self._min_width()), max(1, len(self._items)) * self.row_h() + 4)

    def minimumSizeHint(self) -> QSize:
        return QSize(self._min_width(), max(1, len(self._items)) * self.row_h() + 4)

    def _min_width(self) -> int:
        label_w, val_w, _ = self._geometry(self.fontMetrics())
        return int(label_w + val_w + 40)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self.setFixedHeight(max(1, len(self._items)) * self.row_h() + 4)
            self.updateGeometry()

    def clipped_texts(self) -> list[tuple[str, float, float]]:
        """Audit hook: labels / values need `label_w + val_w` plus a bar of at least 40 px."""
        need = self._min_width()
        return [("breakdown bars", need, self.width())] if self._items and need > self.width() + 1 else []

    def row_text(self, k: int) -> str:
        key, _v = self._items[k]
        if key == "intercept" and self._exact:
            return f"50% → {50.0 + self._pp[k]:.1f}%"
        return f"{self._pp[k]:+.1f} pp"

    def _geometry(self, fm) -> tuple[float, float, float]:
        label_w = max([fm.horizontalAdvance(self.label_of(k)) for k, _ in self._items] + [40]) + 10
        val_w = max([fm.horizontalAdvance(self.row_text(i)) for i in range(len(self._items))] + [50]) + 8
        return label_w, val_w, max(40.0, self.width() - label_w - val_w)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        sem = semantic()
        fm = p.fontMetrics()
        label_w, val_w, bar_w = self._geometry(fm)
        bar_x0 = label_w
        mid = bar_x0 + bar_w / 2
        scale = max([abs(v) for v in self._pp] + [1.0])
        p.setPen(QPen(pal.color(QPalette.ColorRole.Mid), 1))
        p.drawLine(QPointF(mid, 0), QPointF(mid, self.height()))
        for i, (k, _v) in enumerate(self._items):
            y = i * self.row_h() + 2
            if i == self._hover:
                p.fillRect(QRectF(0, y, self.width(), self.row_h()), QColor(128, 128, 128, 36))
            p.setPen(pal.color(QPalette.ColorRole.WindowText))
            p.drawText(QRectF(0, y, label_w - 6, self.row_h()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       self.label_of(k))
            pp = self._pp[i]
            w = abs(pp) / scale * (bar_w / 2 - 4)
            col = pal.color(QPalette.ColorRole.PlaceholderText) if k == "intercept" else (
                sem["good"] if pp >= 0 else sem["bad"])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawRoundedRect(QRectF(mid if pp >= 0 else mid - w, y + 5, max(w, 1.5), self.row_h() - 10), 3, 3)
            p.setPen(pal.color(QPalette.ColorRole.WindowText))
            p.drawText(QRectF(bar_x0 + bar_w + 6, y, val_w, self.row_h()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.row_text(i))

    def row_at(self, y: float) -> int:
        k = int((y - 2) // self.row_h())
        return k if 0 <= k < len(self._items) else -1

    def tip(self, k: int) -> str:
        from .helptext import breakdown_tip

        if not (0 <= k < len(self._items)):
            return ""
        key, v = self._items[k]
        note = "" if self._exact else ("<i>The parts do not add up exactly to this deck's log-odds here, so pp use the "
                                       "local slope q(1−q) of the win-rate curve.</i>")
        return breakdown_tip(key, v, self._pp[k], note)

    def mouseMoveEvent(self, event) -> None:
        k = self.row_at(event.position().y())
        if k != self._hover:
            self._hover = k
            self.setToolTip(self.tip(k))
            self.update()

    def leaveEvent(self, event) -> None:
        self._hover = -1
        self.update()


class WarningBox(QFrame):
    """List of warnings with a colored accent; hidden when empty.

    `compact=True`: one line (the title, the number of notes and the first note, elided with the full text in the tooltip) that
    expands to the whole list on click: notes that need no action do not push the result down."""

    def __init__(self, title: str = "Warnings", parent: QWidget | None = None, compact: bool = False):
        super().__init__(parent)
        self._title = title
        self.compact = compact
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setProperty("card", True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(2)
        self.toggle: ElidedToolButton | None = None
        if compact:
            self.toggle = ElidedToolButton(min_chars=12)
            self.toggle.setCheckable(True)
            self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.toggle.setArrowType(Qt.ArrowType.RightArrow)
            self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
            self.toggle.toggled.connect(self._on_toggled)
            lay.addWidget(self.toggle)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.label)
        self.label.setVisible(not compact)
        self.setVisible(False)
        self.lines: list[str] = []

    def _on_toggled(self, on: bool) -> None:
        assert self.toggle is not None
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.label.setVisible(on)

    def set_expanded(self, on: bool) -> None:
        if self.toggle is not None:
            self.toggle.setChecked(on)

    def is_expanded(self) -> bool:
        return self.toggle is None or self.toggle.isChecked()

    def set_lines(self, lines: list[str]) -> None:
        from html import escape
        self.lines = [ln for ln in lines if ln]
        col = semantic()["warn"].name()
        body = "".join(f"<li>{escape(ln)}</li>" for ln in self.lines)
        if self.toggle is None:
            self.label.setText(f"<b style='color:{col}'>⚠ {escape(self._title)}</b>"
                               f"<ul style='margin:2px 0 0 -18px'>{body}</ul>")
        else:
            self.label.setText(f"<ul style='margin:0 0 0 -18px'>{body}</ul>")
            n = len(self.lines)
            head = f"⚠ {self._title} ({n})" + (f"  ·  {self.lines[0]}" if n else "")
            self.toggle.setText(head)
            self.toggle.setToolTip("Click to show or hide the whole list")
            self.set_expanded(False)
        self.setVisible(bool(self.lines))


class Collapsible(QWidget):
    """Section with a toggle header; content shown when expanded."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.toggle = ElidedToolButton()
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setText(title)
        self.content = content
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(content)
        self.toggle.toggled.connect(self._on)
        self.toggle.setChecked(expanded)
        self._on(expanded)

    def _on(self, on: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.content.setVisible(on)

    def set_title(self, title: str) -> None:
        self.toggle.setText(title)

    def set_expanded(self, on: bool) -> None:
        self.toggle.setChecked(on)


class BusyStrip(QWidget):
    """Status/progress indicator for tabs: bar + message + optional Cancel."""

    cancelRequested = Signal()

    def __init__(self, cancellable: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        self.bar.setObjectName("busyBar")
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(max(10, round(self.fontMetrics().height() * 0.6)))      # a graphic: scales with the font
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.label = muted(ElidedLabel("Ready"))                # a long message is elided (full text in the tooltip), never cut
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(cancellable)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancelRequested)
        self.bar.setMinimumWidth(40)
        self.bar.setMaximumWidth(360)                           # a bar, not a rail: message and Cancel sit right beside it
        lay.setSpacing(10)
        lay.addWidget(self.bar, 1)
        lay.addWidget(self.label)
        lay.addWidget(self.cancel_btn)
        lay.addStretch(1)

    def begin(self, message: str = "Working…") -> None:
        self.bar.setRange(0, 0)  # indeterminate until first progress
        self.label.setText(message)
        self.cancel_btn.setEnabled(True)

    def update_progress(self, frac: float, msg: str) -> None:
        if self.bar.maximum() == 0:
            self.bar.setRange(0, 1000)
        self.bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        if msg:
            self.label.setText(msg)

    def end(self, message: str = "Ready") -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.label.setText(message)
        self.cancel_btn.setEnabled(False)


def temp_flash(label: QLabel, text: str, ms: int = 2500) -> None:
    label.setText(text)
    ref = weakref.ref(label)

    def clear() -> None:
        lab = ref()
        if lab is not None:
            try:
                lab.setText("")
            except RuntimeError:  # underlying C++ object already deleted
                pass

    QTimer.singleShot(ms, clear)


def home_dir_hint() -> str:  # pragma: no cover - cosmetic
    return str(Path.home())


class CardPreview(QWidget):
    """The large card image with a caption under it, scaled to the room it has (docked permanently in the Pool tab's left column).

    `show_card(set_code, name, caption_html)` fetches the image through the shared `ImageStore` (background thread, disk cached)
    and repaints when it arrives; without an image the card name is drawn instead."""

    RATIO = 680 / 488

    def __init__(self, store, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self.set_code = ""
        self.name = ""
        self.pm: QPixmap | None = None
        self.caption = muted(QLabel(""))
        self.caption.setWordWrap(True)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.caption.setTextFormat(Qt.TextFormat.RichText)
        self.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.art = _PreviewArt(self)
        lay.addWidget(self.art, 1)
        from PySide6.QtWidgets import QScrollArea                  # a long caption scrolls instead of squeezing the card

        self.caption_scroll = QScrollArea()
        self.caption_scroll.setWidgetResizable(True)
        self.caption_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.caption_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.caption_scroll.setWidget(self.caption)
        self.caption_scroll.setStyleSheet("QScrollArea { background: transparent; } QScrollArea > QWidget > QWidget { background: transparent; }")
        lay.addWidget(self.caption_scroll)
        self._fit_caption()
        store.ready.connect(self._on_image)

    CAPTION_LINES = 7            # the caption gets the height it needs up to this many lines, then scrolls

    def _fit_caption(self) -> None:
        """Height of the caption area: what the wrapped text needs at the current width (no fixed pixels), up to CAPTION_LINES lines."""
        ls = self.caption.fontMetrics().lineSpacing()
        need = self.caption.heightForWidth(max(self.width(), 60)) if self.caption.text() else 0
        h = min(need, self.CAPTION_LINES * ls) if need else 0
        self.caption_scroll.setFixedHeight(h)
        self.caption_scroll.setVisible(h > 0)
        self.setMinimumHeight(self.art.minimumHeight() + (self.layout().spacing() + h if h else 0))      # the card and the caption never overlap

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_caption()

    def show_card(self, set_code: str, name: str, caption: str = "") -> None:
        self.set_code, self.name = set_code, name
        self.caption.setText(caption)
        self.caption.setToolTip("")
        self.art.setToolTip(name)
        self._fit_caption()
        self.art.update()

    def clear(self) -> None:
        self.name = ""
        self.caption.setText("")
        self.art.setToolTip("")
        self._fit_caption()
        self.art.update()

    def _on_image(self, name: str) -> None:
        if name == self.name:
            self.art.update()


class _PreviewArt(QWidget):
    def __init__(self, owner: CardPreview):
        super().__init__(owner)
        self.owner = owner
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(self.READABLE_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to see the card full size")

    READABLE_H = 340             # about 245 px wide: the rules text stays legible; the full-size view is one click away

    def mousePressEvent(self, event) -> None:
        o = self.owner
        if o.name and event.button() == Qt.MouseButton.LeftButton:
            _CardPopup(o.store, o.set_code, o.name, self).show_near(self)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        o = self.owner
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = max(self.height() - 2, 10)
        w = min(self.width() - 2, int(h / o.RATIO))
        h = int(w * o.RATIO)
        x = (self.width() - w) // 2
        r = QRectF(x, 0, w, h)
        pal = self.palette()
        pm = o.store.pixmap(o.set_code, o.name, w, self.devicePixelRatioF()) if o.name else None
        if pm is not None:
            p.drawPixmap(r.toRect(), pm)
            return
        p.setPen(QPen(pal.color(QPalette.ColorRole.Mid), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, 8, 8)
        p.setPen(pal.color(QPalette.ColorRole.PlaceholderText))
        box = r.adjusted(8, 8, -8, -8)
        text = elide_wrapped(p.fontMetrics(), o.name or "Hover a card to see it here", int(box.width()), int(box.height()))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)


class _CardPopup(QWidget):
    """The card at its native size (488 x 680) in a popup; any click or Esc closes it."""

    W = 488

    def __init__(self, store, set_code: str, name: str, parent: QWidget):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.store, self.set_code, self.name = store, set_code, name
        self.setFixedSize(self.W, round(self.W * CardPreview.RATIO))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        store.ready.connect(self._on_image)

    def show_near(self, anchor: QWidget) -> None:
        scr = (anchor.screen() or QApplication.primaryScreen()).availableGeometry()
        top_left = anchor.mapToGlobal(QPoint(anchor.width(), 0))
        x = min(max(top_left.x(), scr.left()), scr.right() - self.width())
        y = min(max(top_left.y(), scr.top()), scr.bottom() - self.height())
        self.move(x, y)
        self.show()

    def _on_image(self, name: str) -> None:
        if name == self.name:
            self.update()

    def mousePressEvent(self, event) -> None:
        self.close()

    def keyPressEvent(self, event) -> None:
        self.close()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        pm = self.store.pixmap(self.set_code, self.name, self.width(), self.devicePixelRatioF())
        if pm is not None:
            p.drawPixmap(self.rect(), pm)
            return
        p.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Window))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.name)


class FloatingCardPreview(QFrame):
    """The card (image, or its text while the image is missing) in a small frameless window beside the mouse pointer.

    Used by the Pool tab while the pool input - and with it the docked preview - is hidden.  A tool window that takes no focus and
    no mouse input (so it never steals a click or flickers the hover under it); `show_card` places it next to the pointer, kept on
    the screen.  `hide()` it when the pointer leaves the card."""

    W = 300
    GAP = 22                     # distance from the pointer

    def __init__(self, store, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setObjectName("floatingPreview")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.card = CardPreview(store)
        self.card.art.setCursor(Qt.CursorShape.ArrowCursor)
        self.card.CAPTION_LINES = 10                       # the caption of a floating card is shown (nearly) whole
        self.card.caption_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(0)
        lay.addWidget(self.card)
        self.setFixedWidth(self.W)

    def show_card(self, set_code: str, name: str, caption: str = "", global_pos: QPoint | None = None) -> None:
        c = self.card
        c.show_card(set_code, name, caption)
        c.art.setToolTip("")
        inner = self.W - 12
        c.setFixedWidth(inner)
        c.art.setMinimumHeight(0)
        c.art.setFixedHeight(round(inner * CardPreview.RATIO))
        c.resize(inner, c.height())
        c._fit_caption()
        cap = c.caption_scroll.height() if c.caption_scroll.isVisibleTo(c) else 0
        c.setMinimumHeight(0)
        c.setFixedHeight(c.art.height() + (c.layout().spacing() + cap if cap else 0))
        self.setFixedHeight(c.height() + 12)
        self.place(global_pos if global_pos is not None else QCursor.pos())
        self.show()
        self.raise_()

    def place(self, pos: QPoint) -> None:
        """Right of the pointer, or left of it when there is no room; vertically centred on it; always inside the screen."""
        scr = QApplication.screenAt(pos) or QApplication.primaryScreen()
        area = scr.availableGeometry()
        x = pos.x() + self.GAP
        if x + self.width() > area.right():
            x = pos.x() - self.GAP - self.width()
        y = pos.y() - self.height() // 2
        x = min(max(x, area.left()), max(area.right() - self.width() + 1, area.left()))
        y = min(max(y, area.top()), max(area.bottom() - self.height() + 1, area.top()))
        self.move(x, y)
