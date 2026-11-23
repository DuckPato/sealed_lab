"""The written report of a set, as the "Summary" section of the Report tab (Markdown with hidden fact tags `<!--F:id-->`, see `sealed_lab.written_report`).

Every tagged sentence gets a hover tooltip with the facts behind it (text and "How sure?" badge, from the facts stored with the report,
so it works offline) and can be clicked: `SummaryPanel.factClicked` carries the fact ids, and the Report tab scrolls to the analysis section
that holds the figure (`fact_target`).  Card names of the set become hover targets (`SummaryPanel.cardHovered`).  The panel only READS what
`api.written_report` returns; installing is the CLI's job (`report-install`).
"""
from __future__ import annotations

import html
import math
import re
from typing import Any
from urllib.parse import quote, unquote

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import QLabel, QTextBrowser, QTextEdit, QToolTip, QVBoxLayout, QWidget

from sealed_lab.report_check import SENTENCE, tags_of

from .widgets import muted, semantic, sized_font

NO_REPORT = "No written report yet for this set: the sections below have the numbers."
NO_REPORT_ALL = "No written report yet for All sets: the sections below have the counts."
ALL = "ALL"
OUTDATED = "Outdated: written for older data; the sections below are current."
BADGE_WORD = {"strong": "Strong", "likely": "Likely", "weak": "Weak", "descriptive": "Descriptive"}
HINT = "Hover a sentence to see the figures behind it; click it to jump to them below."
MAX_TEXT_WIDTH = 820                 # px: the text column of the summary (longer lines read badly)
LINE_HEIGHT = 150                    # percent
MIN_CARD_NAME = 4                    # shorter card names are not looked for in the text

# Which section of the analysis holds the figures of a fact: id prefix -> (section key, "More detail" block to open or None).
# The keys are those of the Section objects of the Report tab (report_tab.py); XSET facts (All sets) are mapped to a lesson by the tab.
PREFIX_TARGET: dict[str, tuple[str, str | None]] = {
    "GLANCE": ("glance", None), "PLAY": ("glance", None),
    "PAIR": ("pairs", None), "PAIRDET": ("pairs", "pairs"), "SPLASH": ("pairs", None),
    "CARD": ("cards", None),
    "SPEC": ("archetypes", None),
    "SHAPE": ("shape", None), "SHAPEDET": ("shape", "shape"), "BUILDER": ("shape", None), "BUILDERCHECK": ("shape", None),
    "EVOL": ("evolution", None),
}
FORMATS = ("bo1", "bo3")


def fact_target(fid: str) -> tuple[str | None, str, str | None, str | None] | None:
    """(format or None, section key, detail block or None, colour pair or None) for a fact id such as `PAIR.bo3.WU`; None when unknown.
    XSET facts give the section "xset" (the tab maps them to a lesson)."""
    parts = fid.split(".")
    prefix = parts[0]
    if prefix == "XSET":
        return None, "xset", None, None
    hit = PREFIX_TARGET.get(prefix)
    if hit is None:
        return None
    fmt = parts[1] if len(parts) > 1 and parts[1] in FORMATS else None
    pair = parts[2] if prefix == "SPEC" and len(parts) > 2 else None
    return fmt, hit[0], hit[1], pair


def fact_tip(fid: str, facts: dict) -> str:
    f = (facts.get("facts") or {}).get(fid)
    if f is not None:
        badge = BADGE_WORD.get(f.get("badge", ""), "")
        return f"{f.get('text', fid)}" + (f"  [{badge}]" if badge else "")
    n = (facts.get("noclear") or {}).get(fid)
    if n is not None:
        return f"{n.get('text', fid)}  [no clear difference]"
    return fid


def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)


def card_pattern(names: list[str]) -> tuple[re.Pattern | None, dict[str, str]]:
    """A pattern for the card names (longest first, whole words) and {escaped name: name}; (None, {}) without names."""
    esc = {html.escape(n, quote=False): n for n in names if len(n) >= MIN_CARD_NAME}
    if not esc:
        return None, {}
    alt = "|".join(re.escape(e) for e in sorted(esc, key=lambda e: (-len(e), e)))
    return re.compile(r"(?<![\w'])(?:" + alt + r")(?![\w'])"), esc


def _link_cards(body: str, cards: tuple[re.Pattern | None, dict[str, str]] | None, owned: dict[str, int] | None = None) -> str:
    """Card names in the HTML of a sentence become `card:` anchors (never inside a tag); `owned` ({name: copies}): a star in front of the
    names of cards in the Deck builder's pool."""
    if not cards or cards[0] is None:
        return body
    pat, esc = cards

    def sub(m: re.Match) -> str:
        star = "\u2605\u00a0" if owned and esc[m.group(0)] in owned else ""
        return f'<a href="card:{quote(esc[m.group(0)], safe="")}">{star}{m.group(0)}</a>'

    return "".join(chunk if chunk.startswith("<") else pat.sub(sub, chunk) for chunk in re.split(r"(<[^>]*>)", body))


def _plain(body: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", body))


def _line_html(line: str, facts: dict, sink: list | None = None, cards=None, owned: dict[str, int] | None = None) -> str:
    out = []
    for m in SENTENCE.finditer(line):
        s = m.group(0).strip()
        if not s:
            continue
        ids = tags_of(s)
        body = _link_cards(_inline(re.sub(r"<!--.*?-->", "", s).strip()), cards, owned)
        if ids:
            if sink is not None:
                sink.append((_plain(body), ids))
            tip = html.escape("\n".join(fact_tip(i, facts) for i in ids), quote=True)
            body = f'<span title="{tip}">{body}</span>'
        out.append(body)
    return " ".join(out)


def render_markdown(text: str, facts: dict, cards=None, owned: dict[str, int] | None = None) -> tuple[str, list[tuple[str, list[str]]]]:
    """The small Markdown subset of the report (headings, bullets, paragraphs, bold / italics) as HTML, and the tagged sentences in
    document order as (plain text, fact ids).  Each tagged sentence becomes a span whose title is the tooltip; card names (`cards` =
    `card_pattern(...)`) become `card:` anchors."""
    parts: list[str] = []
    bullets: list[str] = []
    para: list[str] = []
    sents: list[tuple[str, list[str]]] = []

    def flush() -> None:
        nonlocal bullets, para
        if bullets:
            parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
        if para:
            parts.append("<p>" + " ".join(para) + "</p>")
        bullets, para = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        h = re.match(r"(#{1,6})\s+(.*)", line)
        if h:
            flush()
            lvl = min(len(h.group(1)) + 1, 4)
            parts.append(f"<h{lvl}>{_inline(h.group(2))}</h{lvl}>")
            continue
        b = re.match(r"(?:[-*+]|\d+[.)])\s+(.*)", line)
        if b:
            if para:
                flush()
            bullets.append(_line_html(b.group(1), facts, sents, cards, owned))
            continue
        if bullets:
            flush()
        para.append(_line_html(line, facts, sents, cards, owned))
    flush()
    return "".join(parts), sents


def markdown_to_html(text: str, facts: dict, cards=None, owned: dict[str, int] | None = None) -> str:
    return render_markdown(text, facts, cards, owned)[0]


class _Browser(QTextBrowser):
    """The summary text: tooltips of the tagged sentences, a hover highlight and a pointing hand over them, a click signal.  It never scrolls
    by itself (the panel gives it the height of its text), so the wheel goes to the page."""

    factClicked = Signal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.ranges: list[tuple[int, int, list[str]]] = []          # (start, end) of each tagged sentence in the document, its fact ids
        self.owned: dict[str, int] = {}                             # card name -> copies in the Deck builder's pool (tooltip of a starred name)
        self._hover: tuple[int, int, list[str]] | None = None
        self._press: QPoint | None = None
        self.setOpenLinks(False)
        self.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.setMouseTracking(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("QTextBrowser { background: transparent; border: 0; padding: 0; }")

    # --- where is the pointer
    def _hit(self, p: QPoint) -> int:
        return self.document().documentLayout().hitTest(QPointF(p), Qt.HitTestAccuracy.ExactHit)

    def sentence_at(self, p: QPoint) -> tuple[int, int, list[str]] | None:
        i = self._hit(p)
        if i < 0:
            return None
        return next((r for r in self.ranges if r[0] <= i <= r[1]), None)

    def sentence_rect_center(self, text_start: int) -> QPoint:
        """A point inside the character at document position `text_start` (for tests and screenshots)."""
        c = QTextCursor(self.document())
        c.setPosition(text_start + 1)
        return self.cursorRect(c).center()

    # --- events
    def viewportEvent(self, e) -> bool:
        t = e.type()
        if t == QEvent.Type.ToolTip:
            href = self.anchorAt(e.pos())
            name = unquote(href[len("card:"):]) if href.startswith("card:") else ""
            tip = self.cursorForPosition(e.pos()).charFormat().toolTip()
            if name in self.owned:
                n = self.owned[name]
                QToolTip.showText(e.globalPos(), f"{name}: in your pool ({n} cop{'y' if n == 1 else 'ies'})", self)
            elif tip and self.sentence_at(e.pos()) is not None:
                QToolTip.showText(e.globalPos(), tip, self)
            else:
                QToolTip.hideText()
            return True
        if t == QEvent.Type.Leave:
            self._set_hover(None)
        return super().viewportEvent(e)

    def wheelEvent(self, e) -> None:
        e.ignore()

    def mouseMoveEvent(self, e) -> None:
        super().mouseMoveEvent(e)
        p = e.position().toPoint()
        r = self.sentence_at(p)
        self._set_hover(r)
        self.viewport().setCursor(Qt.CursorShape.PointingHandCursor if (r is not None or self.anchorAt(p)) else Qt.CursorShape.IBeamCursor)

    def mousePressEvent(self, e) -> None:
        self._press = e.position().toPoint() if e.button() == Qt.MouseButton.LeftButton else None
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        super().mouseReleaseEvent(e)
        if e.button() != Qt.MouseButton.LeftButton:
            return
        p = e.position().toPoint()
        pressed, self._press = self._press, None
        if pressed is None or (p - pressed).manhattanLength() > 6 or self.textCursor().hasSelection():
            return                                                  # a drag selects text; it does not jump
        r = self.sentence_at(p)
        if r is not None and not self.anchorAt(p):
            self.factClicked.emit(list(r[2]))

    def _set_hover(self, r: tuple[int, int, list[str]] | None) -> None:
        if r == self._hover:
            return
        self._hover = r
        sels = []
        if r is not None:
            sel = QTextEdit.ExtraSelection()
            c = QTextCursor(self.document())
            c.setPosition(r[0])
            c.setPosition(r[1], QTextCursor.MoveMode.KeepAnchor)
            sel.cursor = c
            col = QColor(semantic()["accent"])
            col.setAlpha(38)
            sel.format.setBackground(col)
            sels.append(sel)
        self.setExtraSelections(sels)
        if r is None:
            self.viewport().setCursor(Qt.CursorShape.IBeamCursor)


class SummaryPanel(QWidget):
    """Banner (outdated), message (no written report), the text and a one-line hint.  `load(code)` reads `api.written_report(code)`."""

    factClicked = Signal(list)         # fact ids of the clicked sentence
    cardHovered = Signal(str)          # card name under the pointer; "" when it leaves
    errorOccurred = Signal(str, str)

    def __init__(self, api: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self.result: dict | None = None
        self._code = ""
        self._owned: dict[str, int] = {}
        self._cards: dict[str, tuple[re.Pattern | None, dict[str, str]]] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.banner = QLabel(OUTDATED)
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        lay.addWidget(self.banner)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        muted(self.message)
        lay.addWidget(self.message)
        self.view = _Browser()
        self.view.setFont(sized_font(self.font(), 1))
        self.view.setMaximumWidth(MAX_TEXT_WIDTH)
        self.view.factClicked.connect(self.factClicked)
        self.view.highlighted.connect(self._on_highlighted)
        lay.addWidget(self.view)
        self.hint = muted(QLabel(HINT))
        self.hint.setFont(sized_font(self.font(), -1))
        lay.addWidget(self.hint)
        self._shown_w = -1
        self.view.installEventFilter(self)

    # ------------------------------------------------------------------ loading
    def set_owned(self, owned: dict[str, int], render: bool = True) -> None:
        """Mark the names of cards in the Deck builder's pool with a star ({card name: copies}); draws the text again (no new reading) when it changed."""
        owned = dict(owned or {})
        if owned == self._owned:
            return
        self._owned = owned
        self.view.owned = owned
        if render and self.result and self.result.get("text") and self._code:
            self._render(self._code)

    def load(self, code: str) -> None:
        self._code = code
        if not code:
            self._show_message("")
            return
        try:
            self.result = self.api.written_report(code)
        except Exception as e:  # noqa: BLE001
            self.errorOccurred.emit(f"Could not read the written report: {e}", "")
            self._show_message(NO_REPORT)
            return
        if not self.result.get("text"):
            self._show_message(NO_REPORT_ALL if code == ALL else NO_REPORT)
            return
        self._render(code)

    def _show_message(self, text: str) -> None:
        self.result = None
        self.banner.setVisible(False)
        self.view.setVisible(False)
        self.hint.setVisible(False)
        self.message.setText(text)
        self.message.setVisible(True)

    def _cards_of(self, code: str):
        if code == ALL:
            return None
        if code not in self._cards:
            try:
                names = sorted({d["name"] for d in self.api.card_info(code).values() if not d.get("is_basic")})
            except Exception:  # noqa: BLE001 - the text is readable without the card links
                names = []
            self._cards[code] = card_pattern(names)
        return self._cards[code]

    def _stylesheet(self) -> str:
        acc = semantic()["accent"].name()
        lh = LINE_HEIGHT
        return (f"p {{ margin-top: 0px; margin-bottom: 10px; line-height: {lh}%; }} "
                f"li {{ margin-bottom: 5px; line-height: {lh}%; }} "
                f"h2 {{ margin-top: 16px; margin-bottom: 6px; }} h3 {{ margin-top: 12px; margin-bottom: 4px; }} h4 {{ margin-top: 10px; margin-bottom: 4px; }} "
                f"a {{ color: {acc}; text-decoration: none; }}")

    def _render(self, code: str) -> None:
        r = self.result or {}
        facts = (r.get("meta") or {}).get("facts") or {}
        self.message.setVisible(False)
        self.view.setVisible(True)
        self.hint.setVisible(True)
        warn = semantic()["warn"]
        self.banner.setStyleSheet(f"QLabel {{ border: 1px solid {warn.name()}; border-radius: 6px; padding: 6px 10px; }}")
        self.banner.setVisible(bool(r.get("outdated")))
        doc = self.view.document()
        doc.setDefaultStyleSheet(self._stylesheet())
        body, sents = render_markdown(r["text"], facts, self._cards_of(code), self._owned)
        self.view.setHtml(body)
        ranges, pos = [], 0
        for plain, ids in sents:                                   # where each tagged sentence is in the document (they come in order)
            c = doc.find(plain, pos)
            if c.isNull():
                continue
            ranges.append((c.selectionStart(), c.selectionEnd(), ids))
            pos = c.selectionEnd()
        self.view.ranges = ranges
        self.view._hover = None
        self._shown_w = -1
        QTimer.singleShot(0, self.fit)

    # ------------------------------------------------------------------ size
    def fit(self) -> None:
        """The view is as high as its text needs at its width (the page scrolls, not the view)."""
        v = self.view
        w = v.viewport().width()
        if w <= 0 or not v.isVisibleTo(self):
            return
        doc = v.document()
        doc.setTextWidth(w)
        h = int(math.ceil(doc.size().height())) + 6
        if h != v.height():
            v.setFixedHeight(h)
        self._shown_w = w

    def eventFilter(self, obj, e) -> bool:
        if obj is self.view and e.type() == QEvent.Type.Resize and self.view.viewport().width() != self._shown_w:
            QTimer.singleShot(0, self.fit)
        return False

    def showEvent(self, e) -> None:
        super().showEvent(e)
        QTimer.singleShot(0, self.fit)

    # ------------------------------------------------------------------ cards
    def _on_highlighted(self, url: QUrl) -> None:
        s = url.toString() if isinstance(url, QUrl) else str(url or "")
        self.cardHovered.emit(unquote(s[len("card:"):]) if s.startswith("card:") else "")
