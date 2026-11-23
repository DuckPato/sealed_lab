"""Clipping audit for Qt widget trees: finds text that does not fit the box it is drawn in.

    from sealed_lab.gui.clipcheck import audit, format_findings
    findings = audit(main_window)            # [] when every visible text fits (or is deliberately elided with a tooltip)

What counts as clipped (visible widgets only):

* QLabel          plain / rich text wider than the label (no wrap), or taller than the label (word wrap: the height
                  ``heightForWidth`` asks for at the label's real width; rich text is measured with QTextDocument).
* buttons         text (+ icon) wider than the button, or the line taller than it (check boxes / radio buttons: sizeHint).
* QComboBox       current text wider than the edit field.
* QTabBar         a tab squeezed below its size hint, or running past the end of the bar.
* QHeaderView     a section title wider than the section.  Allowed only when the owning view elides text AND the title
                  has a tooltip (header ``ToolTipRole``) that shows the full text.
* QGroupBox       title wider than the box.
* QLineEdit       placeholder text wider than the field while the field is empty.
* item views      a cell whose text is wider than its column.  Cells elide with "…" (that is deliberate: table cells are a
                  constrained space), which is only accepted when the cell's tooltip shows the full text.
* custom painted  any widget may define ``clipped_texts() -> list[(text, needed_px, available_px)]`` returning the texts
                  it draws that do not fit and are not elided-with-tooltip; the audit reports them.
* parent clipping a text widget whose rectangle is cut by an ancestor that is not a scroll area.

Each :class:`Finding` carries the widget class, a path of objectNames / classes, the text, and needed vs available pixels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6 import QtGui
from PySide6.QtCore import QModelIndex, QPoint, QRect, Qt
from PySide6.QtGui import QFontMetrics, QTextDocument
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QScrollArea,
    QStyle,
    QStyleOptionComboBox,
    QTabBar,
    QTableView,
    QToolButton,
    QTreeView,
    QWidget,
)

TOL = 1.01          # px of slack: rounding of glyph advances / frames
MAX_CELLS = 4000    # item-view cells examined per view


@dataclass
class Finding:
    kind: str            # label-width / label-height / button / combo / tab / header / group / placeholder / cell / custom / parent
    cls: str
    path: str
    text: str
    needed: float
    available: float

    def __str__(self) -> str:
        t = self.text if len(self.text) <= 70 else self.text[:67] + "..."
        return f"[{self.kind}] {self.cls} {self.path}: {t!r} needs {self.needed:.0f}px, has {self.available:.0f}px"


def _plain(text: str) -> str:
    """Plain text with white space collapsed (HTML collapses runs of blanks, so comparisons must too)."""
    if "<" in text and ">" in text:
        d = QTextDocument()
        d.setHtml(text)
        text = d.toPlainText()
    return " ".join(text.split())


def widget_path(w: QWidget, root: QWidget | None = None) -> str:
    parts: list[str] = []
    cur: QWidget | None = w
    while cur is not None:
        name = cur.objectName() or type(cur).__name__
        parts.append(name)
        if cur is root:
            break
        cur = cur.parentWidget()
    return "/".join(reversed(parts[:8]))


def _is_rich(label: QLabel) -> bool:
    fmt = label.textFormat()
    if fmt == Qt.TextFormat.RichText:
        return True
    if fmt == Qt.TextFormat.AutoText:
        return QtGui.Qt.mightBeRichText(label.text())
    return False


def _label_margins(label: QLabel) -> tuple[int, int]:
    """Horizontal / vertical room taken by frame, margin and indent."""
    m = max(label.margin(), 0)
    fw = label.frameWidth()
    ind = label.indent()
    if ind < 0:
        ind = 0
    return 2 * (fw + m) + (ind if label.alignment() & Qt.AlignmentFlag.AlignLeft and ind else 0), 2 * (fw + m)


def _tip_shows(widget: QWidget, text: str) -> bool:
    """Does the widget's tooltip show `text` in full?"""
    tip = widget.toolTip()
    return bool(tip) and _plain(text).strip() in _plain(tip)


def _label_findings(lab: QLabel, root: QWidget, out: list[Finding]) -> None:
    text = lab.text()
    if not text or lab.movie() is not None:
        return
    if getattr(lab, "elides", False):                       # ElidedLabel: "…" is deliberate when the tooltip has the whole text
        if lab.is_elided() and not _tip_shows(lab, lab.full_text()):
            out.append(Finding("elided-no-tip", "ElidedLabel", widget_path(lab, root), text, 1.0, 0.0))
        return
    shown = _plain(text)
    where = widget_path(lab, root)
    if lab.wordWrap():
        # Qt's own measure of the wrapped text at the width the label really has (rich text included)
        need_h = lab.heightForWidth(lab.width())
        if need_h > lab.height() + TOL:
            out.append(Finding("label-height", "QLabel", where, shown, need_h, lab.height()))
        else:
            fm = QFontMetrics(lab.font())            # one unbreakable word wider than the label cannot wrap: it is cut
            longest = max((fm.horizontalAdvance(w) for w in shown.split()), default=0)
            hm, _vm = _label_margins(lab)
            if not _is_rich(lab) and longest > lab.width() - hm + TOL:
                out.append(Finding("label-width", "QLabel", where, shown, longest + hm, lab.width()))
        return
    hint = lab.sizeHint()
    if hint.width() > lab.width() + TOL:
        out.append(Finding("label-width", "QLabel", where, shown, hint.width(), lab.width()))
    if hint.height() > lab.height() + TOL:
        out.append(Finding("label-height", "QLabel", where, shown, hint.height(), lab.height()))


def _button_findings(b: QAbstractButton, root: QWidget, out: list[Finding]) -> None:
    text = b.text().replace("&&", "\0").replace("&", "").replace("\0", "&")
    if not text:
        return
    if getattr(b, "elides", False):
        if b.is_elided() and not _tip_shows(b, b.full_text()):
            out.append(Finding("elided-no-tip", type(b).__name__, widget_path(b, root), text, 1.0, 0.0))
        return
    fm = QFontMetrics(b.font())
    tw = fm.horizontalAdvance(text)
    icon_w = 0
    if not b.icon().isNull():
        icon_w = b.iconSize().width() + 4
    if isinstance(b, QToolButton):
        if b.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly:
            return
        if b.arrowType() != Qt.ArrowType.NoArrow:
            icon_w = max(icon_w, 14)
    from PySide6.QtWidgets import QCheckBox, QPushButton, QRadioButton

    if isinstance(b, (QCheckBox, QRadioButton)):
        need = b.sizeHint().width()
        avail = b.width()
        if need > avail + TOL:
            out.append(Finding("button", type(b).__name__, widget_path(b, root), text, need, avail))
        return
    hint = b.sizeHint().width()
    if isinstance(b, QPushButton) and hint <= 80:
        need = tw + icon_w + 8            # Fusion gives every push button at least 80 px: that says nothing about its text
    else:
        need = max(hint, tw + icon_w + 4)
    if need > b.width() + TOL:
        out.append(Finding("button", type(b).__name__, widget_path(b, root), text, need, b.width()))
    elif fm.height() > b.height() + 2:
        out.append(Finding("button-height", type(b).__name__, widget_path(b, root), text, fm.height(), b.height()))


def _combo_findings(c: QComboBox, root: QWidget, out: list[Finding]) -> None:
    text = c.currentText()
    if not text:
        return
    if getattr(c, "elides", False):
        if c.is_elided() and not _tip_shows(c, text):
            out.append(Finding("elided-no-tip", type(c).__name__, widget_path(c, root), text, 1.0, 0.0))
        return
    opt = QStyleOptionComboBox()
    opt.initFrom(c)
    opt.editable = c.isEditable()
    opt.frame = c.hasFrame()
    r = c.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt, QStyle.SubControl.SC_ComboBoxEditField, c)
    icon = c.itemIcon(c.currentIndex())
    ic = (c.iconSize().width() + 4) if not icon.isNull() else 0
    # the edit field is the exact box the text is drawn in (the application style sheet gives the combo boxes no extra inner margin, where
    # the plain style kept 4 px of air): the text must fit it
    margin = 0
    avail = r.width() - margin - ic
    need = QFontMetrics(c.font()).horizontalAdvance(text)
    if need > avail + TOL:
        out.append(Finding("combo", "QComboBox", widget_path(c, root), text, need + margin + ic + (c.width() - r.width()), c.width()))


def _tab_findings(bar: QTabBar, root: QWidget, out: list[Finding]) -> None:
    for i in range(bar.count()):
        if not bar.isTabVisible(i):
            continue
        rect = bar.tabRect(i)
        hint = bar.tabSizeHint(i)
        horizontal = bar.shape() in (QTabBar.Shape.RoundedNorth, QTabBar.Shape.RoundedSouth,
                                     QTabBar.Shape.TriangularNorth, QTabBar.Shape.TriangularSouth)
        if horizontal:
            if hint.width() > rect.width() + TOL:
                # squeezed: allowed only when the bar elides the title with "…" and the tab's tooltip shows it in full
                if bar.elideMode() == Qt.TextElideMode.ElideNone or _plain(bar.tabText(i)) not in _plain(bar.tabToolTip(i)):
                    out.append(Finding("tab", "QTabBar", widget_path(bar, root), bar.tabText(i), hint.width(), rect.width()))
            elif rect.right() > bar.width() + TOL and not bar.usesScrollButtons():
                out.append(Finding("tab", "QTabBar", widget_path(bar, root), bar.tabText(i), rect.right(), bar.width()))
            elif rect.right() > bar.width() + TOL:
                out.append(Finding("tab-overflow", "QTabBar", widget_path(bar, root), bar.tabText(i), rect.right(), bar.width()))


def _header_findings(h: QHeaderView, root: QWidget, out: list[Finding]) -> None:
    model = h.model()
    if model is None:
        return
    view = h.parentWidget()
    while view is not None and not isinstance(view, QAbstractItemView):
        view = view.parentWidget()
    elides = view is not None and view.textElideMode() != Qt.TextElideMode.ElideNone
    horizontal = h.orientation() == Qt.Orientation.Horizontal
    if not horizontal:
        return
    fm = QFontMetrics(h.font())
    st = h.style()
    margin = st.pixelMetric(QStyle.PixelMetric.PM_HeaderMargin) * 2 + 4
    for logical in range(h.count()):
        if h.isSectionHidden(logical):
            continue
        text = model.headerData(logical, h.orientation(), Qt.ItemDataRole.DisplayRole)
        if not text:
            continue
        text = str(text)
        sz = h.sectionSize(logical)
        sort = (st.pixelMetric(QStyle.PixelMetric.PM_HeaderMarkSize) + 4) if h.isSortIndicatorShown() and h.sortIndicatorSection() == logical else 0
        need = fm.horizontalAdvance(text) + margin + sort
        if need > sz + TOL:
            tip = model.headerData(logical, h.orientation(), Qt.ItemDataRole.ToolTipRole)
            if not (elides and tip and _plain(text) in _plain(str(tip))):
                out.append(Finding("header", "QHeaderView", widget_path(h, root), text, need, sz))


def _group_findings(g: QGroupBox, root: QWidget, out: list[Finding]) -> None:
    if not g.title():
        return
    need = QFontMetrics(g.font()).horizontalAdvance(g.title()) + 30
    if need > g.width() + TOL:
        out.append(Finding("group", "QGroupBox", widget_path(g, root), g.title(), need, g.width()))


def _lineedit_findings(e: QLineEdit, root: QWidget, out: list[Finding]) -> None:
    if e.text() or not e.placeholderText():
        return
    need = QFontMetrics(e.font()).horizontalAdvance(e.placeholderText()) + 12
    if need > e.width() + TOL:
        out.append(Finding("placeholder", "QLineEdit", widget_path(e, root), e.placeholderText(), need, e.width()))


def _cell_ok(index: QModelIndex, text: str) -> bool:
    tip = index.data(Qt.ItemDataRole.ToolTipRole)
    return bool(tip) and _plain(text).strip() in _plain(str(tip))


def _item_view_findings(v: QAbstractItemView, root: QWidget, out: list[Finding]) -> None:
    model = v.model()
    if model is None or model.rowCount(v.rootIndex()) == 0:
        return
    vp = v.viewport().rect()
    indexes: list[QModelIndex] = []
    if isinstance(v, QTableView):
        first, last = v.rowAt(0), v.rowAt(max(vp.height() - 1, 0))
        if first < 0:
            first = 0
        if last < 0:
            last = model.rowCount() - 1
        for r in range(first, last + 1):
            if v.isRowHidden(r):
                continue
            for c in range(model.columnCount()):
                if not v.isColumnHidden(c):
                    indexes.append(model.index(r, c))
    elif isinstance(v, QTreeView):
        seen = set()
        y = 1
        while y < vp.height() and len(indexes) < MAX_CELLS:
            ix = v.indexAt(QPoint(4, y))
            if not ix.isValid():
                break
            for c in range(model.columnCount(ix.parent())):
                s = ix.sibling(ix.row(), c)
                key = (s.row(), s.column(), s.internalId())
                if key not in seen and not v.isColumnHidden(c):
                    seen.add(key)
                    indexes.append(s)
            y += max(v.visualRect(ix).height(), 8)
    else:
        for r in range(min(model.rowCount(v.rootIndex()), 300)):
            indexes.append(model.index(r, 0, v.rootIndex()))
    st = v.style()
    pad = 2 * (st.pixelMetric(QStyle.PixelMetric.PM_FocusFrameHMargin) + 1)
    full_tips = bool(getattr(v.itemDelegate(), "full_text_tips", False))       # the delegate shows elided text in full on hover
    for ix in indexes[:MAX_CELLS]:
        text = ix.data(Qt.ItemDataRole.DisplayRole)
        if text is None or text == "":
            continue
        text = str(text)
        if isinstance(v, QTableView):
            width = sum(v.columnWidth(c) for c in range(ix.column(), ix.column() + max(v.columnSpan(ix.row(), ix.column()), 1)))
        elif isinstance(v, QTreeView):
            width = v.columnWidth(ix.column())
            if ix.column() == 0:
                depth, p = 0, ix.parent()
                while p.isValid():
                    depth += 1
                    p = p.parent()
                width -= (depth + 1) * v.indentation()
            if v.header().stretchLastSection() and ix.column() == v.header().logicalIndex(v.header().count() - 1):
                width = max(width, vp.width() - v.header().sectionViewportPosition(ix.column()))
        else:
            width = vp.width()
        font = ix.data(Qt.ItemDataRole.FontRole)
        fm = QFontMetrics(font if font is not None else v.font())
        dec = ix.data(Qt.ItemDataRole.DecorationRole)
        dec_w = (v.iconSize().width() + 4) if dec is not None and not isinstance(dec, str) else 0
        need = fm.horizontalAdvance(text.split("\n")[0]) + pad + dec_w
        if need > width + TOL and not full_tips and not _cell_ok(ix, text):
            out.append(Finding("cell", type(v).__name__, widget_path(v, root) + f"[{ix.row()},{ix.column()}]", text, need, width))


def _parent_clip_findings(w: QWidget, root: QWidget, out: list[Finding], text: str) -> None:
    """The widget's own rectangle is cut by an ancestor (not a scroll area): its text is partly invisible."""
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QAbstractScrollArea):
            return
        p = p.parentWidget()
    vis = w.visibleRegion().boundingRect()
    r = w.rect()
    if vis.isEmpty() and (r.width() > 0 and r.height() > 0):
        return          # fully covered (e.g. a stacked page behind another): not our business
    if vis.width() < r.width() - 1 or vis.height() < r.height() - 1:
        out.append(Finding("parent", type(w).__name__, widget_path(w, root), text,
                           r.width() if vis.width() < r.width() else r.height(),
                           vis.width() if vis.width() < r.width() else vis.height()))


def _scroll_clip_findings(w: QWidget, root: QWidget, out: list[Finding], text: str) -> None:
    """A text widget that sticks out of the right edge of its (resizable) scroll area: content wider than the panel, which is either
    cut off (no horizontal scroll bar) or needs sideways scrolling to be read."""
    p = w.parentWidget()
    while p is not None and not (isinstance(p, QScrollArea) and p.widgetResizable()):
        p = p.parentWidget()
    if p is None:
        return
    vp = p.viewport()
    right = w.mapTo(vp, QPoint(0, 0)).x() + w.width()
    if right > vp.width() + 1:
        out.append(Finding("scroll-clip", type(w).__name__, widget_path(w, root), text, right, vp.width()))


def _overlap_findings(w: QWidget, root: QWidget, out: list[Finding], text: str, seen: set) -> None:
    """A text widget that overlaps a sibling (a layout squeezed below the widgets' minimum sizes draws them over each other)."""
    parent = w.parentWidget()
    if parent is None:
        return
    mine = w.geometry()
    for sib in parent.children():
        if not isinstance(sib, QWidget) or sib is w or sib.isWindow() or not sib.isVisible() or (id(sib), id(w)) in seen:
            continue
        inter = mine.intersected(sib.geometry())
        if inter.width() > 2 and inter.height() > 2:
            seen.add((id(w), id(sib)))
            out.append(Finding("overlap", type(w).__name__, widget_path(w, root), f"{text}  [overlaps {type(sib).__name__} "
                               f"{sib.objectName()} {inter.width()}x{inter.height()}px]", float(inter.width()), 0.0))


def _in_collapsed_pane(w: QWidget) -> bool:
    """Inside a container that has no area at all (a splitter pane the user collapsed on purpose): nothing there is shown."""
    p = w.parentWidget()
    while p is not None:
        if p.width() <= 0 or p.height() <= 0:
            return True
        p = p.parentWidget()
    return False


def audit(root: QWidget) -> list[Finding]:
    """Every clipped text in the visible part of `root`'s widget tree (see the module docstring)."""
    out: list[Finding] = []
    seen: set = set()
    widgets = [root] + root.findChildren(QWidget)
    for w in widgets:
        if not w.isVisible() or w.width() <= 0 and w.height() <= 0 or _in_collapsed_pane(w):
            continue
        # skip the internals of item views / combo popups / line edit helpers: the views are checked as a whole
        if isinstance(w, QLabel):
            _label_findings(w, root, out)
            if w.text():
                _parent_clip_findings(w, root, out, _plain(w.text()))
                _scroll_clip_findings(w, root, out, _plain(w.text()))
                _overlap_findings(w, root, out, _plain(w.text()), seen)
        elif isinstance(w, QAbstractButton):
            _button_findings(w, root, out)
            if w.text():
                _parent_clip_findings(w, root, out, w.text())
                _scroll_clip_findings(w, root, out, w.text())
                _overlap_findings(w, root, out, w.text(), seen)
        elif isinstance(w, QComboBox):
            _combo_findings(w, root, out)
            _scroll_clip_findings(w, root, out, w.currentText())
        elif isinstance(w, QTabBar):
            _tab_findings(w, root, out)
        elif isinstance(w, QHeaderView):
            _header_findings(w, root, out)
        elif isinstance(w, QGroupBox):
            _group_findings(w, root, out)
        elif isinstance(w, QLineEdit):
            _lineedit_findings(w, root, out)
        elif isinstance(w, QAbstractItemView):
            _item_view_findings(w, root, out)
        if not isinstance(w, (QLabel, QAbstractButton)) and not w.isWindow():
            _overlap_findings(w, root, out, type(w).__name__, seen)       # any widget: containers and custom-painted ones too
        hook = getattr(w, "clipped_texts", None)
        if callable(hook):
            try:
                items = hook()
            except Exception as exc:  # noqa: BLE001 - a broken hook must not hide the rest of the audit
                items = [(f"clipped_texts() raised {exc!r}", 0.0, 0.0)]
            for text, need, avail in items:
                out.append(Finding("custom", type(w).__name__, widget_path(w, root), str(text), float(need), float(avail)))
    return out


def format_findings(findings: list[Finding], limit: int = 40) -> str:
    lines = [str(f) for f in findings[:limit]]
    if len(findings) > limit:
        lines.append(f"... and {len(findings) - limit} more")
    return "\n".join(lines)


def scaled_font_app(app: QApplication, factor: float) -> None:
    """Stress helper: enlarge the application font by `factor` (Windows 'make text bigger' / other users' fonts)."""
    f = app.font()
    f.setPointSizeF(f.pointSizeF() * factor)
    app.setFont(f)
