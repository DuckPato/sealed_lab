"""Visual deck builder (MTGO style): card images stacked in columns, deck on top and the rest of the pool below.

Columns group by mana value / colour / type / rarity, optionally split into a creature row and a non-creature row.
Double-click, drag & drop or the context menu move cards between deck and sideboard; basics are added with the
steppers of the side panel. Every change re-evaluates the deck in the background (`api.evaluate_deck`, a few ms)
and the side panel shows win rate, score, curve, colour requirements and the mana simulation live.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import traceback
from typing import Any

from html import escape

from PySide6.QtCore import QEvent, QMimeData, QObject, QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QDrag, QFont, QFontMetrics, QImage, QKeySequence, QPainter, QPainterPath, QPalette, QPen,
                           QPixmap, QShortcut)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sealed_lab.types import DeckEval, SetBusy

from . import helptext as H
from .explain_widgets import SuggestedSwaps, WaterfallChart
from .widgets import (
    BASIC_NAMES,
    COLOR_NAMES,
    MTG_COLORS,
    BreakdownBars,
    Collapsible,
    ElidedCombo,
    ElidedLabel,
    FlowLayout,
    HeaderLayout,
    MetricCard,
    TipLabel,
    WarningBox,
    color_tag,
    elide_wrapped,
    muted,
    new_settings,
    pct,
    pct_interval,
    remember_splitter,
    run_rule_text,
    RUN_RULES,
    semantic,
    sized_font,
    temp_flash,
    tipped,
)
from .workers import TaskController, is_cancelled_exc, record_error

log = logging.getLogger(__name__)

MIME = "application/x-sealedlab-card"
CARD_RATIO = 680 / 488          # Scryfall "normal" images
PEEK = 0.155                     # visible share of a card covered by the next one (its title bar)
HEADER_H = 22
PREVIEW_W = 220
GAP = 10
MARGIN = 10
BASIC_OF = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}

GROUP_MODES = [("Mana value", "cmc"), ("Colour", "color"), ("Type", "type"), ("Rarity", "rarity")]
SORT_MODES = [("Mana value", "mv"), ("Value", "value"), ("Name", "name")]
BADGE_MODES = [("Value: vs average", "avg"), ("Value: above replacement", "var")]
CAST_WARN = 0.65                 # castable-on-time below this gets a warning chip (real on-colour cards: median ~0.9-0.95, 5th pct ~0.75)
MANA_STATS = [
    ("p_screw", "Mana screw", "P(fewer than 3 lands in play at the end of turn 3)"),
    ("p_flood", "Flood", "P(9 or more lands among the first 16 cards seen)"),
    ("p_color_screw", "Colour screw", "Share of spells the mana could pay for but the colours could not (turns 2-6)"),
    ("mana_spent", "Mana spent", "Mana spent on turns 1-7 as a share of the maximum (28)"),
    ("mull_rate", "Mulligan rate", "P(the opening seven is mulliganed)"),
]
TILE_W, ART_H = 60, 44            # basic-land tiles


# --------------------------------------------------------------------------- card classification (pure)

def card_group(d: dict, mode: str) -> tuple[int, str]:
    """(order, label) of the column a card goes to under grouping `mode`."""
    if d.get("is_land"):
        return 99, "Lands"
    if mode == "color":
        cols = d.get("colors") or ""
        if len(cols) > 1:
            return 6, "Multicolour"
        if not cols:
            return 7, "Colourless"
        return "WUBRG".index(cols), COLOR_NAMES.get(cols, cols)
    if mode == "type":
        t = (d.get("types") or "").lower()
        for i, (key, label) in enumerate((("creature", "Creatures"), ("instant", "Instants"), ("sorcery", "Sorceries"),
                                          ("planeswalker", "Planeswalkers"), ("artifact", "Artifacts"),
                                          ("enchantment", "Enchantments"))):
            if key in t:
                return i, label
        return (0, "Creatures") if d.get("is_creature") else (8, "Other")
    if mode == "rarity":
        r = (d.get("rarity") or "").lower()
        order = {"mythic": (0, "Mythic"), "rare": (1, "Rare"), "uncommon": (2, "Uncommon"), "common": (3, "Common")}
        return order.get(r, (4, "Other"))
    mv = int(round(float(d.get("mana_value") or 0.0)))
    if mv >= 7:
        return 7, "7+"
    return mv, str(mv)


def color_fill(d: dict) -> QColor:
    if d.get("is_land"):
        return QColor("#8a7a5c")
    cols = d.get("colors") or ""
    if len(cols) > 1:
        return QColor("#c9a43a")
    if not cols:
        return QColor("#9aa0a8")
    return QColor(MTG_COLORS.get(cols, "#999999"))


def unknown_info(name: str) -> dict:
    return dict(name=name, mana_value=0.0, is_creature=False, is_land=name in BASIC_NAMES, colors="", rarity="", types="",
                cost="", is_basic=name in BASIC_NAMES, produces="")


def muted_html(text: str) -> str:
    return f"<span style='color:gray'>{text}</span>"


def value_pp(beta: float, game_wr: float, in_deck: bool) -> float:
    """pp change of game win rate from removing (deck card) or adding (sideboard card) one copy."""
    import math

    q = min(max(game_wr, 1e-6), 1 - 1e-6)
    L = math.log(q / (1 - q))

    def s(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    return 100.0 * ((s(L) - s(L - beta)) if in_deck else (s(L + beta) - s(L)))


# --------------------------------------------------------------------------- images

MAX_IMAGES = 400                   # full-size card images kept in memory (least recently used go first)


class ImageStore(QObject):
    """Card pixmaps fetched in a background thread through `api.card_image` (disk-cached), scaled on demand."""

    ready = Signal(str)
    _loaded = Signal(str, str, QImage)
    _instances: dict[int, "ImageStore"] = {}

    @classmethod
    def for_api(cls, api: Any) -> "ImageStore":
        key = id(api)
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls._instances[key] = cls(api)
        return inst

    def __init__(self, api: Any):
        super().__init__()
        self.api = api
        self._images: collections.OrderedDict[tuple[str, str], QImage] = collections.OrderedDict()   # (set, name) -> image, least recently used first
        self._scaled: dict[tuple[str, str, int], QPixmap] = {}
        self._missing: set[tuple[str, str]] = set()
        self._pending: set[tuple[str, str]] = set()
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._loaded.connect(self._on_loaded)

    def pixmap(self, set_code: str, name: str, width: int, dpr: float = 1.0) -> QPixmap | None:
        ck = (set_code, name)
        img = self._images.get(ck)
        if img is None:
            if ck not in self._missing and ck not in self._pending and getattr(self.api, "card_image", None):
                self._pending.add(ck)
                self._q.put((set_code, name))
                self._ensure_thread()
            return None
        self._images.move_to_end(ck)
        px_w = max(8, int(round(width * dpr)))
        key = (set_code, name, px_w)
        pm = self._scaled.get(key)
        if pm is None:
            pm = QPixmap.fromImage(img.scaledToWidth(px_w, Qt.TransformationMode.SmoothTransformation))
            pm.setDevicePixelRatio(dpr)
            if len(self._scaled) > 1500:
                self._scaled.clear()
            self._scaled[key] = pm
        return pm

    def prefetch(self, set_code: str, names: list[str]) -> None:
        """Resolve many image URLs in one go (batched API calls) before the per-card downloads start."""
        names = [n for n in dict.fromkeys(names) if (set_code, n) not in self._images and (set_code, n) not in self._missing]
        if names and getattr(self.api, "prefetch_images", None):
            self._q.put((set_code, names))
            self._ensure_thread()

    def _ensure_thread(self) -> None:
        with self._thread_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="card-images", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                set_code, name = self._q.get(timeout=30)
            except queue.Empty:
                with self._thread_lock:
                    if self._q.empty():
                        self._thread = None              # (an item queued from now on starts a new thread)
                        return
                continue
            if isinstance(name, list):
                try:
                    self.api.prefetch_images(set_code, name)
                except Exception:  # noqa: BLE001 - images are cosmetic
                    log.debug("image prefetch failed", exc_info=True)
                continue
            img = QImage()
            try:
                path = self.api.card_image(set_code, name)
                if path:
                    img = QImage(str(path))
            except Exception:  # noqa: BLE001 - images are cosmetic
                log.debug("card image failed for %s", name, exc_info=True)
            self._loaded.emit(set_code, name, img)

    def _on_loaded(self, set_code: str, name: str, img: QImage) -> None:
        ck = (set_code, name)
        self._pending.discard(ck)
        if img.isNull():
            self._missing.add(ck)
        else:
            self._images[ck] = img
            self._images.move_to_end(ck)
            while len(self._images) > MAX_IMAGES:
                old, _ = self._images.popitem(last=False)
                for k in [k for k in self._scaled if k[:2] == old]:
                    del self._scaled[k]
        self.ready.emit(name)


# --------------------------------------------------------------------------- pile view

class PileView(QWidget):
    """Columns of stacked cards for one zone ("deck" or "side")."""

    activated = Signal(int)            # card_id double-clicked
    hovered = Signal(int)              # card_id under the mouse, -1 for none
    dropped = Signal(int, str)         # card_id, zone it came from
    moveRequested = Signal(int, bool)  # card_id, all copies
    lockRequested = Signal(int, str)   # card_id, "lock" | "lock1" | "exclude" | "clear"

    def __init__(self, zone: str, store: ImageStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.zone = zone
        self.store = store
        self.set_code = ""
        self.ids: list[int] = []
        self.info: dict[int, dict] = {}
        self.locks: dict[int, int] = {}         # card_id -> locked copies (padlock marker)
        self.excluded: set[int] = set()         # dimmed with a cross
        self.marks: dict[int, str] = {}         # card_id -> "+" / "-": difference to the compared deck
        self.chips: dict[int, str] = {}         # card_id -> text of a pill ("SB vs UB"): premium sideboard card
        self.tips: dict[int, str] = {}          # card_id -> tooltip shown while the mouse is over the card
        self.values: dict[int, tuple[float, bool]] = {}
        self.castable: dict[int, float] = {}    # card_id -> P(castable on time); a warning chip below CAST_WARN
        self.accepts: set[str] = {"side", "basic"} if zone == "deck" else {"deck"}   # zones whose cards may be dropped here
        self.opts = dict(group="cmc", split=True, card_w=130, show_values=True, sort="mv", badge="avg")
        self.empty_text = ""
        self._items: list[tuple[QRectF, int]] = []      # paint order
        self._headers: list[tuple[QRectF, str]] = []
        self._hover = -1
        self._press: tuple[QPoint, int] | None = None
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        store.ready.connect(self._on_image)

    # ------------------------------------------------------------------ content
    def set_content(self, set_code: str, ids: list[int], info: dict[int, dict], values: dict[int, tuple[float, bool]],
                    opts: dict) -> None:
        self.set_code, self.ids, self.info, self.values = set_code, list(ids), info, values
        self.opts = dict(opts)
        self._relayout()

    def _d(self, cid: int) -> dict:
        return self.info.get(cid) or unknown_info(str(cid))

    def columns(self) -> list[tuple[str | None, list[tuple[str, list[int]]]]]:
        """[(row label, [(column label, [card ids, one per copy])])] in display order."""
        mode, sort = self.opts["group"], self.opts["sort"]

        def key(cid: int):
            d = self._d(cid)
            mv = float(d.get("mana_value") or 0.0)
            name = d.get("name") or ""
            if sort == "value":
                return (-(self.values.get(cid, (0.0, False))[0]), mv, name)
            if sort == "name":
                return (name,)
            cols = d.get("colors") or ""
            return (mv, len(cols) > 1, "WUBRG".find(cols[:1]) if cols else 9, name)

        def build(ids: list[int]) -> list[tuple[str, list[int]]]:
            groups: dict[tuple[int, str], list[int]] = {}
            for cid in ids:
                groups.setdefault(card_group(self._d(cid), mode), []).append(cid)
            return [(g[1], sorted(v, key=key)) for g, v in sorted(groups.items())]

        if self.opts["split"] and mode != "type":
            cre = [c for c in self.ids if self._d(c).get("is_creature") and not self._d(c).get("is_land")]
            cre_set = set(cre)
            rest = [c for c in self.ids if c not in cre_set]
            rows = []
            if cre:
                rows.append(("Creatures", build(cre)))
            if rest:
                rows.append(("Non-creatures", build(rest)))
            return rows
        return [(None, build(self.ids))] if self.ids else []

    def _geometry(self) -> tuple[int, int, int]:
        w = int(self.opts["card_w"])
        h = int(round(w * CARD_RATIO))
        return w, h, max(14, int(round(h * PEEK)))

    def _relayout(self) -> None:
        w, h, peek = self._geometry()
        hh = self.header_h()
        self._items, self._headers = [], []
        y = MARGIN
        max_x = 0
        for row_label, cols in self.columns():
            if row_label:
                n = sum(len(c) for _, c in cols)
                self._headers.append((QRectF(MARGIN, y, 600, hh), f"{row_label} ({n})"))
                y += hh
            tallest = 0
            x = MARGIN
            for label, ids in cols:
                head = f"MV {label}" if self.opts["group"] == "cmc" and label != "Lands" else label
                self._headers.append((QRectF(x, y, w, hh - 4), f"{head}  ({len(ids)})"))
                cy = y + hh
                for i, cid in enumerate(ids):
                    self._items.append((QRectF(x, cy + i * peek, w, h), cid))
                tallest = max(tallest, hh + (len(ids) - 1) * peek + h)
                x += w + GAP
            max_x = max(max_x, x)
            y += tallest + GAP
        self.setMinimumSize(QSize(max(max_x + MARGIN, 200), max(int(y) + MARGIN, 120)))
        self.update()

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def card_at(self, pos) -> int:
        for r, cid in reversed(self._items):
            if r.contains(QRectF(pos.x(), pos.y(), 1, 1).topLeft()):
                return cid
        return -1

    def _on_image(self, name: str) -> None:
        if any((self.info.get(cid) or {}).get("name") == name for cid in set(self.ids)):
            self.update()

    # ------------------------------------------------------------------ painting
    def header_h(self) -> int:
        """Height of a column header: one line of the (bold) header font, never fewer pixels than HEADER_H."""
        return max(HEADER_H, QFontMetrics(sized_font(self.font(), 0, bold=True)).lineSpacing() + 6)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._relayout()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        pal = self.palette()
        p.fillRect(self.rect(), pal.color(QPalette.ColorRole.Base))
        hf = sized_font(self.font(), 0, bold=True)
        p.setFont(hf)
        hfm = QFontMetrics(hf)
        for r, text in self._headers:
            p.setPen(pal.color(QPalette.ColorRole.PlaceholderText))
            # a header never runs into the next column: elided with "…" (the full text is the tooltip over the header)
            p.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       hfm.elidedText(text, Qt.TextElideMode.ElideRight, int(r.width())))
        if not self._items and self.empty_text:
            p.setPen(pal.color(QPalette.ColorRole.PlaceholderText))
            box = self.rect().adjusted(20, 20, -20, -20)
            p.drawText(box, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       elide_wrapped(QFontMetrics(self.font()), self.empty_text, box.width(), box.height()))
        dpr = self.devicePixelRatioF()
        clip = event.rect()
        for r, cid in self._items:
            if not r.toAlignedRect().intersects(clip):
                continue
            self._paint_card(p, r, cid, dpr)

    def _paint_card(self, p: QPainter, r: QRectF, cid: int, dpr: float) -> None:
        d = self._d(cid)
        radius = r.width() * 0.045
        path = QPainterPath()
        path.addRoundedRect(r, radius, radius)
        pm = self.store.pixmap(self.set_code, d.get("name", ""), int(r.width()), dpr)
        plan = self._plan_chips(r, cid)
        p.save()
        p.setClipPath(path)
        if pm is not None:
            p.drawPixmap(r.topLeft(), pm)
        else:
            self._paint_placeholder(p, r, d, plan)
        p.restore()
        hover = cid == self._hover
        p.setPen(QPen(semantic()["accent"] if hover else QColor(0, 0, 0, 150), 2.5 if hover else 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        if cid in self.excluded:
            p.save()
            p.setClipPath(path)
            p.fillRect(r, QColor(20, 20, 24, 150))
            p.restore()
        for item in plan:
            if item["box"] is not None:
                self._draw_badge(p, r, item)

    # ---- badges: value, castability, lock / cross / difference marker, family pill (planned from the right edge, never off the card)
    def _chip_font(self, width: float) -> QFont:
        f = QFont(self.font())
        f.setPointSizeF(max(7.0, width / 16.0))
        f.setBold(True)
        return f

    def _plan_chips(self, r: QRectF, cid: int) -> list[dict]:
        """Badges of card `cid` in the card rectangle `r`, placed right to left.  Each item: kind, text (full), shown (what is drawn: a
        text pill shrinks / is elided with "…" when the card is too narrow for it), box (None: no room at all), fg, bg, label (what a
        dropped or elided badge means: it goes into the tooltip)."""
        fm = QFontMetrics(self._chip_font(r.width()))
        ph = fm.height() + 2
        top = r.top() + r.height() * PEEK - ph - 1
        limit = r.left() + 3
        left = r.right() - r.width() * 0.05
        out: list[dict] = []

        def place(kind: str, text: str, label: str, pad: int = 12, round_: bool = False, fg=None, bg=None, gap: int = 4) -> None:
            nonlocal left
            right = left - gap
            full = ph if round_ else fm.horizontalAdvance(text) + pad
            avail = right - limit
            item = dict(kind=kind, text=text, shown=text, box=None, fg=fg, bg=bg, label=label)
            if full <= avail:
                w = full
            elif round_ or avail - pad < fm.horizontalAdvance("…") + fm.horizontalAdvance("0"):
                out.append(item)                                    # no room even for "…": dropped (the tooltip says what it was)
                return
            else:
                item["shown"] = fm.elidedText(text, Qt.TextElideMode.ElideRight, int(avail - pad))
                w = fm.horizontalAdvance(item["shown"]) + pad
            item["box"] = QRectF(right - w, top, w, ph)
            left = item["box"].left()
            out.append(item)

        if self.opts.get("show_values") and cid in self.values:
            pp, low = self.values[cid]
            col = QColor("#7ee2a8") if pp >= 0.05 else (QColor("#ff9b94") if pp <= -0.05 else QColor("#d0d4da"))
            place("value", f"{pp:+.1f}", f"value {pp:+.1f} pp", pad=10, fg=QColor("#f2c46b") if low else col,
                  bg=QColor(15, 17, 20, 215), gap=0)
        cp = self.castable.get(cid)
        if cp is not None and cp < CAST_WARN:
            place("pill", f"⚠ {cp * 100:.0f}%", f"castable on time {cp * 100:.0f}%", fg=QColor("#ffd27a"), bg=QColor(120, 78, 12, 235))
        if cid in self.excluded:
            place("cross", "✕", "excluded from builds", round_=True, fg=QColor("#ff9b94"), bg=QColor(15, 17, 20, 225))
        elif cid in self.locks:
            place("lock", "", "locked in the deck", round_=True, fg=QColor("#f2c46b"), bg=QColor(15, 17, 20, 225))
        mark = self.marks.get(cid)
        if mark:
            place("mark", "+" if mark == "+" else "−", "differs from the compared deck", round_=True,
                  fg=QColor("#7ee2a8") if mark == "+" else QColor("#ff9b94"), bg=QColor(15, 17, 20, 225))
        pill = self.chips.get(cid)
        if pill:
            place("pill", pill, pill, fg=QColor("#9af0c4"), bg=QColor(38, 84, 66, 235))
        return out

    def _draw_badge(self, p: QPainter, r: QRectF, item: dict) -> None:
        p.setFont(self._chip_font(r.width()))
        box: QRectF = item["box"]
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(item["bg"])
        if item["kind"] in ("cross", "lock", "mark"):
            p.drawEllipse(box)
        else:
            p.drawRoundedRect(box, box.height() / 2, box.height() / 2)
        if item["kind"] == "lock":
            c, col = box.center(), item["fg"]
            w = box.width() * 0.46
            body = QRectF(c.x() - w / 2, c.y() - w * 0.05, w, w * 0.78)
            p.setBrush(col)
            p.drawRoundedRect(body, 1.5, 1.5)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(col, max(1.3, w * 0.16)))
            p.drawArc(QRectF(c.x() - w * 0.3, c.y() - w * 0.62, w * 0.6, w * 0.8), 0, 180 * 16)
            return
        p.setPen(item["fg"])
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, item["shown"])

    def _placeholder_title(self, r: QRectF, d: dict, plan: list[dict]) -> tuple[QFont, QRectF, str, str]:
        """(font, title rectangle, name as drawn, cost as drawn ("" when there is no room)) of a card without image."""
        inner = r.adjusted(r.width() * 0.04, r.width() * 0.04, -r.width() * 0.04, -r.width() * 0.04)
        title = QRectF(inner.left(), inner.top(), inner.width(), r.height() * PEEK - r.width() * 0.04)
        f = QFont(self.font())
        f.setPointSizeF(max(6.5, r.width() / 15.0))
        f.setBold(True)
        fm = QFontMetrics(f)
        cost = d.get("cost") or ""
        chips_left = min([it["box"].left() for it in plan if it["box"] is not None], default=None)
        if chips_left is not None:                          # the badges sit on the title bar: the name stops before them, no cost
            room = chips_left - 6 - (title.left() + 4)
            return f, title, fm.elidedText(d.get("name", ""), Qt.TextElideMode.ElideRight, int(room)), ""
        cost_w = fm.horizontalAdvance(cost) + 6 if cost else 0
        room = title.width() - 8 - cost_w - 6
        return f, title, fm.elidedText(d.get("name", ""), Qt.TextElideMode.ElideRight, int(room)), cost

    def _paint_placeholder(self, p: QPainter, r: QRectF, d: dict, plan: list[dict]) -> None:
        fill = color_fill(d)
        p.fillRect(r, QColor("#1b1b1d"))
        inner = r.adjusted(r.width() * 0.04, r.width() * 0.04, -r.width() * 0.04, -r.width() * 0.04)
        p.fillRect(inner, fill.lighter(115))
        f, title, name, cost = self._placeholder_title(r, d, plan)
        p.fillRect(title, fill.lighter(140))
        p.setFont(f)
        p.setPen(QColor("#111"))
        p.drawText(title.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)
        if cost:
            p.drawText(title.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, cost)
        f.setBold(False)
        f.setPointSizeF(max(6.0, r.width() / 17.0))
        p.setFont(f)
        body = QRectF(inner.left() + 6, inner.top() + inner.height() * 0.45, inner.width() - 12, inner.height() * 0.5)
        types = (d.get("types") or "").replace(" - ", " — ") or ("Land" if d.get("is_land") else "")
        p.drawText(body, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                   elide_wrapped(QFontMetrics(f), types, int(body.width()), int(body.height())))

    def _rect_of(self, cid: int) -> QRectF | None:
        return next((r for r, c in self._items if c == cid), None)

    def lost_texts(self, cid: int) -> list[str]:
        """Texts drawn on card `cid` that did not fit (an elided or dropped badge, an elided name / cost of an image-less card):
        the tooltip of the card shows them in full."""
        r = self._rect_of(cid)
        if r is None:
            return []
        d = self._d(cid)
        plan = self._plan_chips(r, cid)
        out = [it["label"] for it in plan if it["box"] is None or it["shown"] != it["text"]]
        if self.store.pixmap(self.set_code, d.get("name", ""), int(r.width()), 1.0) is None:
            _f, _t, name, cost = self._placeholder_title(r, d, plan)
            if name != d.get("name", "") or (d.get("cost") and not cost):
                out.insert(0, f"{d.get('name', '')}  {d.get('cost') or ''}".strip())
        return out

    def clipped_texts(self) -> list[tuple[str, float, float]]:
        """Audit hook: text on the cards that neither fits nor is shown in full by the card's tooltip."""
        out: list[tuple[str, float, float]] = []
        for _r, cid in self._items:
            tip = self.tip_for(cid)
            for t in self.lost_texts(cid):
                if t not in tip and escape(t) not in tip:
                    out.append((t, 1.0, 0.0))
        return out

    # ------------------------------------------------------------------ mouse
    def mouseMoveEvent(self, event) -> None:
        cid = self.card_at(event.position())
        if cid != self._hover:
            self._hover = cid
            self.hovered.emit(cid)
            self.update()
        self.setToolTip(self.tip_for(cid) if cid >= 0 else self._header_tip(event.position()))
        if self._press is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            start, pcid = self._press
            if (event.position().toPoint() - start).manhattanLength() >= QApplication.startDragDistance() and pcid >= 0:
                self._press = None
                self._start_drag(pcid)

    def _header_tip(self, pos) -> str:
        """Full text of the column header under the mouse (headers are elided when the column is narrow)."""
        for r, text in self._headers:
            if r.contains(pos):
                return text
        return ""

    def tip_for(self, cid: int) -> str:
        """Tooltip of a card: its value badge, a castability warning and the family / sideboard-plan remark (and, in bold on top,
        whatever of the card's own text did not fit: elided name, dropped badge)."""
        parts: list[str] = []
        lost = self.lost_texts(cid)
        if lost:
            parts.append("<b>" + " · ".join(escape(t) for t in lost) + "</b>")
        if self.opts.get("show_values") and cid in self.values and not self._d(cid).get("is_basic"):
            pp, low = self.values[cid]
            parts.append(H.value_badge_tip(pp, low, self.zone == "deck", self.opts.get("badge", "avg"), self.castable.get(cid)))
        cp = self.castable.get(cid)
        if cp is not None and cp < CAST_WARN:
            parts.append(H.castable_tip(cp))
        if cid in self.tips:
            parts.append(self.tips[cid])
        if not parts:
            return ""
        body = "<br><br>".join(x.replace("\n", "<br>") for x in parts)
        return f"<html><table width='{H.TIP_WIDTH}'><tr><td>{body}</td></tr></table></html>"

    def leaveEvent(self, event) -> None:
        if self._hover != -1:
            self._hover = -1
            self.hovered.emit(-1)
            self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = (event.position().toPoint(), self.card_at(event.position()))

    def mouseReleaseEvent(self, event) -> None:
        self._press = None

    def mouseDoubleClickEvent(self, event) -> None:
        cid = self.card_at(event.position())
        if cid >= 0:
            self.activated.emit(cid)

    def contextMenuEvent(self, event) -> None:
        cid = self.card_at(event.pos())
        if cid < 0:
            return
        name = self._d(cid).get("name", "")
        dest = "sideboard" if self.zone == "deck" else "deck"
        m = QMenu(self)
        one = QAction(f"Move {name} to {dest}", m)
        one.triggered.connect(lambda: self.moveRequested.emit(cid, False))
        m.addAction(one)
        if self.ids.count(cid) > 1:
            allc = QAction(f"Move all {self.ids.count(cid)} copies to {dest}", m)
            allc.triggered.connect(lambda: self.moveRequested.emit(cid, True))
            m.addAction(allc)
        if not self._d(cid).get("is_basic"):
            m.addSeparator()
            n_in = self.ids.count(cid) if self.zone == "deck" else 0
            lk = QAction("Lock in deck" + (f" (all {n_in} copies)" if n_in > 1 else ""), m)
            lk.setToolTip("Every deck built by 'Rebuild around locks' contains the locked copies")
            lk.triggered.connect(lambda: self.lockRequested.emit(cid, "lock"))
            m.addAction(lk)
            if n_in > 1:
                lk1 = QAction("Lock 1 copy in deck", m)
                lk1.triggered.connect(lambda: self.lockRequested.emit(cid, "lock1"))
                m.addAction(lk1)
            ex = QAction("Exclude from builds", m)
            ex.setToolTip("'Rebuild around locks' never uses this card; it also leaves the deck now")
            ex.triggered.connect(lambda: self.lockRequested.emit(cid, "exclude"))
            m.addAction(ex)
            if cid in self.locks or cid in self.excluded:
                clr = QAction("Clear lock" if cid in self.locks else "Clear exclusion", m)
                clr.triggered.connect(lambda: self.lockRequested.emit(cid, "clear"))
                m.addAction(clr)
        m.exec(event.globalPos())

    def _start_drag(self, cid: int) -> None:
        if self._hover != -1:                     # the pointer is busy dragging: the floating card preview goes away
            self._hover = -1
            self.hovered.emit(-1)
        md = QMimeData()
        md.setData(MIME, f"{self.zone}:{cid}".encode())
        drag = QDrag(self)
        drag.setMimeData(md)
        w, h, _ = self._geometry()
        pm = self.store.pixmap(self.set_code, self._d(cid).get("name", ""), max(60, w * 2 // 3), self.devicePixelRatioF())
        if pm is not None:
            drag.setPixmap(pm)
            drag.setHotSpot(QPoint(int(pm.width() / pm.devicePixelRatio() / 2), 12))
        drag.exec(Qt.DropAction.MoveAction)

    def _payload(self, event) -> tuple[str, int] | None:
        md = event.mimeData()
        if not md.hasFormat(MIME):
            return None
        try:
            zone, cid = bytes(md.data(MIME)).decode().split(":")
            return zone, int(cid)
        except ValueError:
            return None

    def dragEnterEvent(self, event) -> None:
        pl = self._payload(event)
        if pl is not None and pl[0] in self.accepts:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        pl = self._payload(event)
        if pl is not None and pl[0] in self.accepts:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        pl = self._payload(event)
        if pl is not None and pl[0] in self.accepts:
            event.acceptProposedAction()
            self.dropped.emit(pl[1], pl[0])


# --------------------------------------------------------------------------- basic lands

def stepper_size(button: QToolButton) -> tuple[int, int]:
    """Size of a small − / + button: the glyph's width and the line height of its font plus the button's frame."""
    fm = QFontMetrics(button.font())
    return max(18, fm.horizontalAdvance("−") + 8), max(18, fm.lineSpacing() + 4)


class _TileArt(QWidget):
    """Cropped top of a basic land's card image; double-click adds one to the deck, dragging it into the deck does too."""

    activated = Signal()

    def __init__(self, store: ImageStore, col: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.store, self.col = store, col
        self.set_code, self.name, self.cid = "", BASIC_OF[col], -1
        self._press: QPoint | None = None
        self._size_from_font()
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        store.ready.connect(lambda n: self.update() if n == self.name else None)

    def _size_from_font(self) -> None:
        """The tile is a picture, but the land's name is drawn on it when there is no image: wide enough for the longest name in
        the bold font (TILE_W at the default font), height in the same proportion."""
        f = QFont(self.font())
        f.setBold(True)
        w = max(TILE_W, QFontMetrics(f).horizontalAdvance("Mountain") + 6)
        self.setFixedSize(w, round(w * ART_H / TILE_W))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._size_from_font()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        path = QPainterPath()
        path.addRoundedRect(r, 4, 4)
        dpr = self.devicePixelRatioF()
        pm = self.store.pixmap(self.set_code, self.name, self.width(), dpr) if self.set_code else None
        p.setClipPath(path)
        if pm is not None:
            p.drawPixmap(QPoint(0, 0), pm, QRectF(0, 0, pm.width(), self.height() * dpr).toRect())
        else:
            fill = QColor(MTG_COLORS[self.col])
            p.fillRect(r, fill.lighter(112))
            p.setPen(QColor("#111"))
            f = QFont(self.font())
            f.setBold(True)
            p.setFont(f)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, QFontMetrics(f).elidedText(self.name, Qt.TextElideMode.ElideRight, int(r.width()) - 4))
        p.setClipping(False)
        p.setPen(QPen(QColor(0, 0, 0, 170), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position().toPoint()

    def mouseReleaseEvent(self, event) -> None:
        self._press = None

    def mouseDoubleClickEvent(self, event) -> None:
        self.activated.emit()

    def mouseMoveEvent(self, event) -> None:
        if self._press is not None and (event.position().toPoint() - self._press).manhattanLength() >= QApplication.startDragDistance() \
                and self.cid >= 0:
            self._press = None
            md = QMimeData()
            md.setData(MIME, f"basic:{self.cid}".encode())
            drag = QDrag(self)
            drag.setMimeData(md)
            pm = self.store.pixmap(self.set_code, self.name, 90, self.devicePixelRatioF())
            if pm is not None:
                drag.setPixmap(pm)
                drag.setHotSpot(QPoint(int(pm.width() / pm.devicePixelRatio() / 2), 12))
            drag.exec(Qt.DropAction.CopyAction)


class BasicLandStrip(QFrame):
    """Five basic lands (unlimited supply) beside the sideboard: double-click or drag one into the deck, − / + buttons,
    the number in the deck.  Basics dragged from the deck onto the strip leave the deck."""

    addRequested = Signal(str)          # colour letter
    removeRequested = Signal(str)
    dropRemove = Signal(int)            # card_id of a deck basic dropped on the strip

    def __init__(self, store: ImageStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAcceptDrops(True)
        self.arts: dict[str, _TileArt] = {}
        self.counts: dict[str, QLabel] = {}
        self.minus: dict[str, QToolButton] = {}
        self.accepts_basic = lambda cid: False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(6)
        cap = muted(QLabel("<b>Basic lands</b><br>double-click or<br>drag to the deck"))
        cap.setTextFormat(Qt.TextFormat.RichText)
        cap.setFont(sized_font(self.font(), -1))
        lay.addWidget(cap)
        for col in "WUBRG":
            tile = QVBoxLayout()
            tile.setSpacing(1)
            art = _TileArt(store, col)
            art.activated.connect(lambda c=col: self.addRequested.emit(c))
            row = QHBoxLayout()
            row.setSpacing(1)
            mn, pl = QToolButton(), QToolButton()
            mn.setText("−")
            pl.setText("+")
            for b in (mn, pl):
                b.setFixedSize(*stepper_size(b))
            n = QLabel("0")
            n.setAlignment(Qt.AlignmentFlag.AlignCenter)
            n.setFont(sized_font(self.font(), 0, bold=True))
            n.setMinimumWidth(QFontMetrics(n.font()).horizontalAdvance("00") + 2)
            mn.clicked.connect(lambda _=False, c=col: self.removeRequested.emit(c))
            pl.clicked.connect(lambda _=False, c=col: self.addRequested.emit(c))
            row.addWidget(mn)
            row.addWidget(n, 1)
            row.addWidget(pl)
            tile.addWidget(art)
            tile.addLayout(row)
            lay.addLayout(tile)
            self.arts[col], self.counts[col], self.minus[col] = art, n, mn
            for w in (art, mn, pl, n):
                w.setToolTip(H.BASIC_STRIP_TIP)
        lay.addStretch(1)
        self.setToolTip(H.BASIC_STRIP_TIP)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_state(self, set_code: str, ids: dict[str, int], counts: dict[str, int]) -> None:
        for col, art in self.arts.items():
            art.set_code, art.cid = set_code, ids.get(col, -1)
            art.setEnabled(col in ids)
            self.counts[col].setText(str(counts.get(col, 0)))
            self.minus[col].setEnabled(counts.get(col, 0) > 0)
            art.update()

    def _deck_basic(self, event) -> int | None:
        md = event.mimeData()
        if not md.hasFormat(MIME):
            return None
        try:
            zone, cid = bytes(md.data(MIME)).decode().split(":")
            cid = int(cid)
        except ValueError:
            return None
        return cid if zone == "deck" and self.accepts_basic(cid) else None

    def dragEnterEvent(self, event) -> None:
        if self._deck_basic(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._deck_basic(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        cid = self._deck_basic(event)
        if cid is not None:
            event.acceptProposedAction()
            self.dropRemove.emit(cid)


class BasicSteppers(QWidget):
    """Compact − / + per basic land next to the deck title (the quick way to change the mana base)."""

    addRequested = Signal(str)
    removeRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lab = muted(QLabel("Basics:"))
        lab.setToolTip(H.BASIC_STRIP_TIP)
        lay.addWidget(lab)
        self.counts: dict[str, QLabel] = {}
        self.minus: dict[str, QToolButton] = {}
        self.plus: dict[str, QToolButton] = {}
        for col in "WUBRG":
            mn, pl = QToolButton(), QToolButton()
            mn.setText("−")
            pl.setText("+")
            mn.setFixedSize(*stepper_size(mn))
            pl.setFixedSize(*stepper_size(pl))
            mn.setToolTip(f"Remove a {BASIC_OF[col]} from the deck")
            pl.setToolTip(f"Add a {BASIC_OF[col]} to the deck")
            n = QLabel(f"<span style='color:{MTG_COLORS[col]}; font-size:15px'>●</span> 0")
            n.setTextFormat(Qt.TextFormat.RichText)
            n.setMinimumWidth(QFontMetrics(n.font()).horizontalAdvance("00") + 16)
            n.setToolTip(f"{BASIC_OF[col]}s in the deck")
            mn.clicked.connect(lambda _=False, c=col: self.removeRequested.emit(c))
            pl.clicked.connect(lambda _=False, c=col: self.addRequested.emit(c))
            lay.addSpacing(3)
            lay.addWidget(mn)
            lay.addWidget(n)
            lay.addWidget(pl)
            self.counts[col], self.minus[col], self.plus[col] = n, mn, pl

    def set_counts(self, counts: dict[str, int], available: set[str]) -> None:
        for col, n in self.counts.items():
            c = counts.get(col, 0)
            n.setText(f"<span style='color:{MTG_COLORS[col]}; font-size:15px'>●</span> {c}")
            self.minus[col].setEnabled(c > 0)
            self.plus[col].setEnabled(col in available)


# --------------------------------------------------------------------------- side-panel widgets

class CurveChart(QWidget):
    """Stacked bars: creatures / non-creature spells per mana value (1-, 2, ..., 7+)."""

    LABELS = ["≤1", "2", "3", "4", "5", "6", "7+"]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.cre = [0] * 7
        self.non = [0] * 7
        self.setMinimumHeight(max(120, 16 + 2 * self.fontMetrics().height() + 70))     # plot area + the number above and the label below a bar
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._bar = -1
        self.setToolTip(H.CURVE_TIP)

    def set_counts(self, cre: list[int], non: list[int]) -> None:
        self.cre, self.non = list(cre), list(non)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        k = int((event.position().x() - 4) // max(1.0, (self.width() - 8) / len(self.LABELS)))
        k = k if 0 <= k < len(self.LABELS) else -1
        if k != self._bar:
            self._bar = k
            self.setToolTip(H.CURVE_TIP if k < 0 else H.tip(
                f"Mana value {self.LABELS[k]}", H.curve_bar_tip(self.LABELS[k], self.cre[k], self.non[k])))

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        fm = p.fontMetrics()
        top, bottom = 16, self.height() - fm.height() - 4
        n = len(self.LABELS)
        bw = (self.width() - 8) / n
        mx = max([c + s for c, s in zip(self.cre, self.non)] + [6])
        acc, mut = semantic()["accent"], pal.color(QPalette.ColorRole.PlaceholderText)
        for i in range(n):
            x = 4 + i * bw
            hc = (bottom - top) * self.cre[i] / mx
            hs = (bottom - top) * self.non[i] / mx
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(acc)
            p.drawRoundedRect(QRectF(x + 3, bottom - hc, bw - 6, hc), 2, 2)
            p.setBrush(mut)
            p.drawRoundedRect(QRectF(x + 3, bottom - hc - hs, bw - 6, hs), 2, 2)
            p.setPen(pal.color(QPalette.ColorRole.WindowText))
            tot = self.cre[i] + self.non[i]
            if tot:
                p.drawText(QRectF(x, bottom - hc - hs - fm.height() - 1, bw, fm.height()), Qt.AlignmentFlag.AlignCenter, str(tot))
            p.setPen(mut)
            p.drawText(QRectF(x, bottom + 2, bw, fm.height()), Qt.AlignmentFlag.AlignCenter, self.LABELS[i])


# --------------------------------------------------------------------------- evaluator

class _Evaluator(QObject):
    """Runs `fn(deck)` on a background thread, latest request wins; results come back on the GUI thread."""

    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, fn, label=lambda: "evaluate deck"):
        super().__init__()
        self._fn = fn
        self._label = label
        self._next: tuple[int, dict] | None = None
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None

    def submit(self, seq: int, deck: dict[int, int]) -> None:
        with self._cv:
            self._next = (seq, dict(deck))
            self._cv.notify()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="deck-eval", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._next is None:
                    self._cv.wait(timeout=30)
                    if self._next is None:
                        return
                seq, deck = self._next
                self._next = None
            try:
                res = self._fn(deck)
            except Exception as exc:  # noqa: BLE001
                if not isinstance(exc, SetBusy):
                    record_error(self._label(), traceback.format_exc())
                self._emit(self.failed, seq, f"{type(exc).__name__}: {exc}")
            else:
                self._emit(self.done, seq, res)

    @staticmethod
    def _emit(sig, *args) -> None:
        try:
            sig.emit(*args)
        except RuntimeError:             # the window was closed while the evaluation ran
            pass


class _SwapWorker(QObject):
    """Latest-wins background job that can be cancelled and reports partial results (swap analysis)."""

    partial = Signal(int, object)
    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, fn, label=lambda: "swap analysis"):
        super().__init__()
        self._fn = fn               # fn(payload, cancel_event, on_partial)
        self._label = label
        self._next: tuple[int, Any] | None = None
        self._cancel: threading.Event | None = None
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None

    def submit(self, seq: int, payload: Any) -> None:
        with self._cv:
            self._next = (seq, payload)
            if self._cancel is not None:
                self._cancel.set()
            self._cv.notify()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="swap-analysis", daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        with self._cv:
            self._next = None
            if self._cancel is not None:
                self._cancel.set()

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._next is None:
                    self._cv.wait(timeout=30)
                    if self._next is None:
                        return
                seq, payload = self._next
                self._next = None
                ev = self._cancel = threading.Event()
            try:
                res = self._fn(payload, ev, lambda r, s=seq: self.partial.emit(s, r))
            except Exception as exc:  # noqa: BLE001
                if not is_cancelled_exc(exc):
                    if not isinstance(exc, SetBusy):
                        record_error(self._label(), traceback.format_exc())
                    self.failed.emit(seq, f"{type(exc).__name__}: {exc}")
            else:
                if not ev.is_set():
                    self.done.emit(seq, res)


# --------------------------------------------------------------------------- the builder

class DeckBuilder(QWidget):
    """Interactive visual builder for one deck drawn from a pool."""

    errorOccurred = Signal(str, str)
    edited = Signal(bool)              # True once the deck differs from the build it started from
    evaluated = Signal(object)         # DeckEval of the current deck
    previewChanged = Signal(str, str, str, str)   # set code, card name, caption (HTML), tooltip: the Pool tab shows it in its docked preview
    previewCleared = Signal()          # the pointer left the cards (the Pool tab hides its floating preview)

    def __init__(self, api: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self.store = ImageStore.for_api(api)
        self.set_code = ""
        self.fmt = "bo3"
        self.info: dict[int, dict] = {}
        self.betas: dict[int, dict] = {}
        self.level_note: dict | None = None
        self.side_vals: dict[int, dict] | None = None      # sideboard card -> value of one added copy in the CURRENT deck
        self.pool: dict[int, int] = {}
        self.deck: dict[int, int] = {}
        self.orig: dict[int, int] = {}
        self.orig_eval: DeckEval | None = None
        self.current: DeckEval | None = None
        self._history: list[dict[int, int]] = []
        self._future: list[dict[int, int]] = []     # undone decks, for Redo (cleared by any new edit)
        self._seq = 0
        self._shown_seq = 0
        self._basic_ids: dict[str, int] = {}
        self.locked: dict[int, int] = {}          # card_id -> copies "Rebuild around locks" must keep
        self.excluded: set[int] = set()           # cards "Rebuild around locks" must not use
        self.expl: Any | None = None              # DeckExplanation of the current deck
        self.swaps: Any | None = None             # SwapAnalysis of the current deck (None while being computed)
        self.syn: Any | None = None               # insight.SynergyView of the current deck
        self.arch: Any | None = None              # insight.ArchetypeMatch of the current deck
        self.plan: Any | None = None              # matchups.SideboardPlan of the current deck (None while being computed)
        self.alts: list[tuple[str, DeckEval]] = []   # other builds of the same pool, for "Compare with"
        self.origin_label = "generated build"     # what "Reset" / the delta card compare with ("your pasted deck" for the user's deck)
        self.best_ref: DeckEval | None = None     # best build of the pool (family #1): the efficiency line compares with it
        self.best_label = "the best family (#1)"
        self._compare_default = "orig"
        self.frozen = ""                          # non-empty: the set is being recalculated; evaluation is paused (the note)
        self.family: dict | None = None           # deck family this build represents (BuildResult.families[i]) or None
        self.fam_meta: dict = {}                  # BuildResult.family_meta
        self._pre_rebuild: tuple[dict[int, int], DeckEval | None] | None = None
        self._smoothing_modelled = False          # does the shipped mana simulation include Arena's Bo1 hand smoother?
        self._evaluator = _Evaluator(self._evaluate, lambda: f"evaluate deck {self.set_code}")
        self._evaluator.done.connect(self._on_eval)
        self._evaluator.failed.connect(self._on_eval_failed)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._submit)
        self._swap_worker = _SwapWorker(self._analyse_swaps, lambda: f"swap analysis {self.set_code}")
        self._swap_worker.partial.connect(self._on_swaps_partial)
        self._swap_worker.done.connect(self._on_swaps_done)
        self._swap_worker.failed.connect(self._on_swaps_failed)
        self._swap_timer = QTimer(self)
        self._swap_timer.setSingleShot(True)
        self._swap_timer.setInterval(150)
        self._swap_timer.timeout.connect(self._submit_swaps)
        self._swap_seq = 0
        self._plan_worker = _SwapWorker(self._compute_plan, lambda: f"sideboard plan {self.set_code}")
        self._plan_worker.done.connect(self._on_plan_done)
        self._plan_worker.failed.connect(self._on_plan_failed)
        self._plan_seq = 0
        self._rebuild = TaskController(self)
        self._rebuild.progress.connect(lambda f, m: self.status.setText(f"Rebuilding… {m}"))
        self._rebuild.result.connect(self._on_rebuilt)
        self._rebuild.error.connect(self._on_rebuild_error)
        self._rebuild.cancelled.connect(self._on_rebuild_cancelled)
        self._rebuild.ended.connect(self._update_buttons)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        s = new_settings()
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 6, 4, 4)
        root.setSpacing(6)

        bar = FlowLayout(spacing=10)                   # one row where the width allows, wraps (whole groups) where it does not
        self.toolbar = bar
        self.group_combo = QComboBox()
        for label, key in GROUP_MODES:
            self.group_combo.addItem(label, key)
        self.sort_combo = QComboBox()
        for label, key in SORT_MODES:
            self.sort_combo.addItem(label, key)
        self.split_cb = QCheckBox("Split creatures")
        self.split_cb.setToolTip("Creatures in one row, everything else in a second row")
        self.values_cb = QCheckBox("Show values")
        self.values_cb.setToolTip("Badge on each card: change in game win rate (pp) from removing it (deck) or adding it "
                                  "(sideboard).  Yellow = little data in this format: the value leans on the Bo1 / draft prior.")
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(80, 240)
        self.size_slider.setFixedWidth(14 * QFontMetrics(self.font()).horizontalAdvance("0"))       # a slider: sized from the font
        self.size_slider.setToolTip("Card size")
        i = self.group_combo.findData(str(s.value("builder/group", "cmc")))
        self.group_combo.setCurrentIndex(max(i, 0))
        i = self.sort_combo.findData(str(s.value("builder/sort", "mv")))
        self.sort_combo.setCurrentIndex(max(i, 0))
        self.split_cb.setChecked(str(s.value("builder/split", "true")).lower() in ("true", "1"))
        self.values_cb.setChecked(str(s.value("builder/values", "true")).lower() in ("true", "1"))
        try:
            self.size_slider.setValue(int(s.value("builder/card_w", 130)))
        except (TypeError, ValueError):
            self.size_slider.setValue(130)
        for w, text in ((self.group_combo, "Group by"), (self.sort_combo, "Sort by")):
            bar.addWidget(self._tool_group(muted(QLabel(text)), w))
        bar.addWidget(self._tool_group(self.split_cb, self.values_cb))
        self.badge_combo = QComboBox()
        for label, key in BADGE_MODES:
            self.badge_combo.addItem(label, key)
        self.badge_combo.setToolTip("Badge value on the cards.  vs average: change of the game win rate (pp) when the card is "
                                    "removed (deck) / added (sideboard).  above replacement: deck cards show how much better "
                                    "they are than the best replacement from the sideboard (negative: a sideboard card is "
                                    "better); sideboard cards show the gain of their best slot.")
        i = self.badge_combo.findData(str(s.value("builder/badge", "avg")))
        self.badge_combo.setCurrentIndex(max(i, 0))
        bar.addWidget(self.badge_combo)
        bar.addWidget(self._tool_group(muted(QLabel("Size")), self.size_slider))
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Undo the last change to the deck (Ctrl+Z)")
        self.reset_btn = QPushButton("Reset to build")
        self.rebuild_btn = QPushButton("Rebuild around locks")
        self.rebuild_btn.setToolTip("Run the builder again for this pool: locked cards are always in the deck, excluded "
                                    "cards never.  Right-click a card to lock / exclude it.  The result replaces the deck "
                                    "(Undo brings the previous one back).")
        self.copy_btn = QPushButton("Copy Arena export")
        self.copy_btn.setToolTip("Deck and sideboard (the rest of the pool) in Arena import format (Ctrl+Shift+C)")
        self.copy_msg = muted(QLabel(""))
        bar.addWidget(self._tool_group(self.undo_btn, self.reset_btn, self.rebuild_btn, self.copy_btn, self.copy_msg))
        root.addLayout(bar)

        self.hsplit = QSplitter(Qt.Orientation.Horizontal)
        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        for sp in (self.hsplit, self.vsplit):
            sp.setChildrenCollapsible(False)                       # a pane never shrinks below what its labels and buttons need
        self.deck_title = ElidedLabel("Deck", min_chars=8)           # elided only when the column is narrower than the title alone (full text in the tooltip)
        self.deck_title.setFont(sized_font(self.font(), 1, bold=True))
        self.deck_title.setToolTip(H.COUNTS_TIP)
        self.side_title = ElidedLabel("Sideboard", min_chars=8)
        self.side_title.setFont(sized_font(self.font(), 1, bold=True))
        self.deck_pile = PileView("deck", self.store)
        self.deck_pile.empty_text = "Double-click or drag cards from the sideboard to add them."
        self.side_pile = PileView("side", self.store)
        self.side_pile.empty_text = "Every card of the pool is in the deck."
        self.steppers = BasicSteppers()
        self.basic_strip = BasicLandStrip(self.store)
        self.basic_strip.accepts_basic = self._is_basic
        for title, pile, extra in ((self.deck_title, self.deck_pile, self.steppers),
                                   (self.side_title, self.side_pile, self.basic_strip)):
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(2)
            head_host = QWidget()                               # the title with its basics beside it where they fit, under it where they do not
            head = HeaderLayout(head_host)                     # (the title is never elided to make room for them)
            head.addWidget(title)
            head.addWidget(extra)
            bl.addWidget(head_host)
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(pile)
            sc.setFrameShape(QFrame.Shape.StyledPanel)
            sc.setMinimumHeight(4 * QFontMetrics(self.font()).lineSpacing())
            bl.addWidget(sc, 1)
            self.vsplit.addWidget(box)
        self.vsplit.setStretchFactor(0, 3)
        self.vsplit.setStretchFactor(1, 2)
        self.hsplit.addWidget(self.vsplit)
        self.hsplit.addWidget(self._build_panel())
        self.hsplit.setStretchFactor(0, 1)
        self.hsplit.setStretchFactor(1, 0)
        self.hsplit.setSizes([1000, 420])
        self.vsplit.setSizes([480, 360])
        remember_splitter(self.hsplit, "builder/h")
        remember_splitter(self.vsplit, "builder/v")
        root.addWidget(self.hsplit, 1)

        for p in (self.deck_pile, self.side_pile):
            p.hovered.connect(lambda cid, z=p.zone: self._on_hover(cid, z))
            p.lockRequested.connect(self.lock_action)
            p.activated.connect(lambda cid, z=p.zone: self.move(cid, z))
            p.moveRequested.connect(lambda cid, all_, z=p.zone: self.move(cid, z, all_copies=all_))
            p.dropped.connect(lambda cid, src: self.move(cid, src))
        for w in (self.steppers, self.basic_strip):
            w.addRequested.connect(lambda c: self.add_basic(c, +1))
            w.removeRequested.connect(lambda c: self.add_basic(c, -1))
        self.basic_strip.dropRemove.connect(lambda cid: self.move(cid, "deck"))
        for w in (self.group_combo, self.sort_combo, self.badge_combo):
            w.currentIndexChanged.connect(self._opts_changed)
        for w in (self.split_cb, self.values_cb):
            w.toggled.connect(self._opts_changed)
        self.size_slider.valueChanged.connect(self._opts_changed)
        self.undo_btn.clicked.connect(self.undo)
        # keyboard shortcuts (active while this builder is the visible one: a hidden tab's builder does not take part)
        self.undo_shortcut = self._shortcut("Ctrl+Z", self.undo)
        self.redo_shortcut = self._shortcut("Ctrl+Y", self.redo)
        self.redo_shortcut2 = self._shortcut("Ctrl+Shift+Z", self.redo)
        self.copy_shortcut = self._shortcut("Ctrl+Shift+C", self.copy_export)
        self.sample_shortcut = self._shortcut("Ctrl+H", self.open_sample_hand)
        self.reset_btn.clicked.connect(self.reset)
        self.rebuild_btn.clicked.connect(self.rebuild)
        self.copy_btn.clicked.connect(self.copy_export)
        self._update_buttons()

    def _shortcut(self, keys: str, slot) -> QShortcut:
        sc = QShortcut(QKeySequence(keys), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(slot)
        return sc

    @staticmethod
    def _tool_group(*widgets: QWidget) -> QWidget:
        """Widgets that belong together (a label and its control) as one item of the wrapping toolbar."""
        host = QWidget()
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        for w in widgets:
            lay.addWidget(w)
        return host

    def _section(self, key: str, title: str, tip: str, default_open: bool = False) -> tuple[Collapsible, QVBoxLayout]:
        """Collapsible side-panel section (open / closed state remembered); returns (section, layout of its content)."""
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(14, 0, 0, 6)
        cl.setSpacing(6)
        s = new_settings()
        opened = str(s.value(f"builder/sec_{key}", "true" if default_open else "false")).lower() in ("true", "1")
        sec = Collapsible(title, content, expanded=opened)
        sec.toggle.setToolTip(tip)
        sec.toggle.toggled.connect(lambda on, k=key: new_settings().setValue(f"builder/sec_{k}", on))
        sec.toggle.toggled.connect(lambda _on: QTimer.singleShot(0, self._fit_panel_width))
        self.sections[key] = sec
        return sec, cl

    def _build_panel(self) -> QWidget:
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 0, 6, 6)
        lay.setSpacing(8)
        self.sections: dict[str, Collapsible] = {}

        # ---- always visible: the headline numbers, then the card under the mouse
        grid = QGridLayout()
        grid.setSpacing(8)
        self.m_game = MetricCard("Game win rate (avg pilot)")
        self.m_match = MetricCard("Match win rate")
        self.m_score = MetricCard("Score (0–100)")
        self.m_wins = MetricCard("Expected run wins")
        self.m_pmax = MetricCard("P(max wins)")
        self.m_delta = MetricCard("vs generated build")
        for i, m in enumerate((self.m_game, self.m_match, self.m_score, self.m_wins, self.m_pmax, self.m_delta)):
            m.value.setFont(sized_font(self.font(), 3, bold=True))
            grid.addWidget(m, i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.m_game.set_tip(H.game_wr_tip(self.fmt))
        self.m_match.set_tip(H.match_wr_tip(self.fmt))
        self.m_score.set_tip(H.score_tip(self.fmt))
        self.m_delta.set_tip(H.delta_tip())
        self.m_wins.set_tip(H.exp_wins_tip(self.fmt))
        self.m_pmax.set_tip(H.p_max_tip(self.fmt))
        lay.addLayout(grid)
        self.rule_label = muted(QLabel(run_rule_text(self.fmt)))
        self.rule_label.setWordWrap(True)
        self.rule_label.setToolTip(H.exp_wins_tip(self.fmt))
        lay.addWidget(self.rule_label)
        self.eff_label = QLabel("")
        self.eff_label.setWordWrap(True)
        self.eff_label.setTextFormat(Qt.TextFormat.RichText)
        self.eff_label.setVisible(False)
        lay.addWidget(self.eff_label)
        self.support_box = WarningBox("Estimate less reliable: this deck is outside what real decks look like")
        self.support_box.setToolTip(H.support_tip())
        lay.addWidget(self.support_box)
        self.status = muted(QLabel(""))
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.warnings = WarningBox("Warnings")
        lay.addWidget(self.warnings)

        self.counts = QLabel("")
        self.counts.setWordWrap(True)
        self.counts.setToolTip(H.COUNTS_TIP)
        lay.addWidget(self.counts)

        # the large card image lives in the Pool tab (docked under the pool list); these hold what it shows (not laid out here)
        self.preview = QLabel(self)
        self.preview_caption = QLabel(self)
        self.preview.setVisible(False)
        self.preview_caption.setVisible(False)

        # ---- sections
        sec, fl = self._section("family", "Deck family", H.FAMILY_HELP["head"], True)
        self.family_label = TipLabel("")
        fl.addWidget(self.family_label)
        lay.addWidget(sec)
        self.family_box = sec
        sec.setVisible(False)

        self.arch_label = QLabel("")
        self.arch_label.setWordWrap(True)
        self.arch_label.setTextFormat(Qt.TextFormat.RichText)
        self.arch_label.setToolTip(H.ARCH_TIP)
        lay.addWidget(self.arch_label)

        self.lock_info = QLabel("")
        self.lock_info.setWordWrap(True)
        self.clear_locks_btn = QToolButton()
        self.clear_locks_btn.setText("Clear all locks")
        self.clear_locks_btn.clicked.connect(self.clear_locks)
        lrow = QHBoxLayout()
        lrow.addWidget(self.lock_info, 1)
        lrow.addWidget(self.clear_locks_btn)
        lay.addLayout(lrow)

        sec, wl = self._section("why", "Why this deck", H.WATERFALL_TIP, True)
        self.why_caption = muted(QLabel(""))
        self.why_caption.setWordWrap(True)
        self.waterfall = WaterfallChart()
        wl.addWidget(self.why_caption)
        wl.addWidget(self.waterfall)
        self.why_caption.setToolTip(H.WATERFALL_TIP)
        lay.addWidget(sec)
        self.why_box = sec

        sec, sl = self._section("swaps", "Suggested swaps", H.tip(
            "Suggested swaps",
            "The best one-for-one swaps between the deck and your sideboard, each evaluated with the full model "
            "(curve, colours and mana simulation included).  Hover a swap for its gain and what P(better) means."), True)
        self.swap_list = SuggestedSwaps()
        self.swap_list.applyRequested.connect(self.apply_swap)
        sl.addWidget(self.swap_list)
        lay.addWidget(sec)
        self.swap_box = sec

        sec, cl = self._section("curve", "Mana curve (spells)", H.CURVE_TIP)
        self.curve = CurveChart()
        cl.addWidget(self.curve)
        leg = muted(QLabel("■ creatures (bottom, accent) · ■ other spells (top, grey)"))
        leg.setFont(sized_font(self.font(), -1))
        cl.addWidget(leg)
        lay.addWidget(sec)

        sec, gl_ = self._section("colours", "Colours and basic lands", H.tip(
            "Colours and basic lands",
            "For each colour: pips needed by the spells, sources that can make it, and basic lands of it in the deck.  "
            "Change the basics with the − / + buttons above the deck or the 'Basic lands' strip beside the sideboard."), True)
        gl = QGridLayout()
        gl.setHorizontalSpacing(14)
        for c, (h, tp) in enumerate((("", ""), ("Pips", H.PIPS_TIP), ("Sources", H.SOURCES_TIP), ("Basics", H.BASICS_TIP))):
            if h:
                lab = muted(QLabel(h))
                lab.setToolTip(tp)
                gl.addWidget(lab, 0, c)
        self._color_rows: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        for r, col in enumerate("WUBRG", start=1):
            dot = QLabel(f"<span style='color:{MTG_COLORS[col]}; font-size:16px'>●</span> {COLOR_NAMES[col]}")
            pips, srcs, nb = QLabel("0"), QLabel("0"), QLabel("0")
            for c, w in enumerate((dot, pips, srcs, nb)):
                gl.addWidget(w, r, c)
            pips.setToolTip(H.PIPS_TIP)
            srcs.setToolTip(H.SOURCES_TIP)
            nb.setToolTip(H.BASICS_TIP)
            self._color_rows[col] = (pips, srcs, nb)
        gl.setColumnStretch(4, 1)
        gl_.addLayout(gl)
        lay.addWidget(sec)

        sec, ml = self._section("mana", "Mana simulation", H.tip(
            "Mana simulation",
            "Monte-Carlo simulation of thousands of games' opening hands and draws with this exact deck: how often "
            "the mana fails.  Hover each line for what it means and how to read it."))
        form = QFormLayout()
        self._mana_labels: dict[str, QLabel] = {}
        self._mana_names: dict[str, QLabel] = {}
        for key, label, tip in MANA_STATS:
            lab, val = QLabel(label), QLabel("—")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._mana_labels[key] = val
            self._mana_names[key] = lab
            form.addRow(lab, val)
        ml.addLayout(form)
        self.mana_note = muted(QLabel(""))
        self.mana_note.setWordWrap(True)
        self.mana_note.setFont(sized_font(self.font(), -1))
        ml.addWidget(self.mana_note)
        self.sample_btn = QPushButton("Sample hand…")
        self.sample_btn.setToolTip(H.SAMPLE_HAND_TIP)
        self.sample_btn.clicked.connect(self.open_sample_hand)
        ml.addWidget(self.sample_btn)
        lay.addWidget(sec)
        self._refresh_mana_tips()

        sec, bl = self._section("breakdown", "Win-rate breakdown", H.BREAKDOWN_TITLE_TIP)
        self.bars = BreakdownBars()
        bl.addWidget(self.bars)
        self.bars_note = muted(QLabel("Percentage points of game win rate, applied one after the other from 50%.  "
                                      "Hover a bar for the log-odds."))
        self.bars_note.setWordWrap(True)
        self.bars_note.setFont(sized_font(self.font(), -1))
        bl.addWidget(self.bars_note)
        lay.addWidget(sec)

        sec, yl = self._section("synergies", "Synergies", H.SYNERGY_TIP)
        self.syn_label = QLabel("")
        self.syn_label.setWordWrap(True)
        self.syn_label.setTextFormat(Qt.TextFormat.RichText)
        self.syn_label.setToolTip(H.SYNERGY_TIP)
        yl.addWidget(self.syn_label)
        lay.addWidget(sec)
        self.syn_box = sec

        sec, pl_ = self._section("plan", "Sideboard plan (Bo3)", H.PLAN_TIP)
        self.plan_note = muted(QLabel(""))
        self.plan_note.setWordWrap(True)
        self.plan_combo = ElidedCombo()
        self.plan_combo.currentIndexChanged.connect(self._refresh_plan_ui)
        self.plan_swaps = SuggestedSwaps()
        self.plan_swaps.applyRequested.connect(self.apply_swap)
        pl_.addWidget(self.plan_note)
        pl_.addWidget(self.plan_combo)
        pl_.addWidget(self.plan_swaps)
        lay.addWidget(sec)
        self.plan_box = sec

        sec, cl_ = self._section("compare", "Compare with", H.COMPARE_TIP)
        self.compare_combo = ElidedCombo()
        self.compare_combo.currentIndexChanged.connect(self._update_compare)
        self.compare_label = QLabel("")
        self.compare_label.setWordWrap(True)
        self.compare_label.setTextFormat(Qt.TextFormat.RichText)
        self.compare_label.setToolTip(H.COMPARE_TIP)
        cl_.addWidget(self.compare_combo)
        cl_.addWidget(self.compare_label)
        lay.addWidget(sec)
        self.compare_box = sec

        lay.addStretch(1)
        sc.setWidget(body)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)      # content wider than the panel scrolls instead of being cut
        self._panel_scroll, self._panel_body = sc, body
        self._fit_panel_width()
        return sc

    def _fit_panel_width(self) -> None:
        """The side panel is never narrower than its widest unwrappable content (metric cards, section titles, buttons) needs."""
        sc, body = self._panel_scroll, self._panel_body
        need = body.minimumSizeHint().width() + sc.verticalScrollBar().sizeHint().width() + 2 * sc.frameWidth() + 2
        sc.setMinimumWidth(max(330, 24 * QFontMetrics(self.font()).horizontalAdvance("0"), need))

    def _refresh_mana_tips(self) -> None:
        sm = self._smoothing_modelled
        for key, lab in self._mana_names.items():
            tp = H.mana_tip(key, self.fmt, sm)
            lab.setToolTip(tp)
            self._mana_labels[key].setToolTip(tp)
        self.mana_note.setText(H.mana_regime_note(self.fmt, sm).replace("<b>", "").replace("</b>", ""))
        self.mana_note.setToolTip(H.tip("Format note", H.mana_regime_note(self.fmt, sm)))

    # ------------------------------------------------------------------ deck family
    def set_family(self, fam: dict | None, meta: dict | None = None) -> None:
        """Describe the generated build as a deck family (P(best), gap to #1, core and flex cards) and mark its flex cards."""
        self.family, self.fam_meta = (dict(fam) if fam else None), dict(meta or {})
        self._refresh_family()

    def set_origin_label(self, label: str) -> None:
        """Name of the deck this one started as ("generated build", "your pasted deck"): used by the delta card and Compare with."""
        self.origin_label = label
        self.m_delta.caption.setText(f"vs {label}")
        self.m_delta.set_tip(H.delta_tip(None, f"the {label}" if not label.startswith("your") else label))
        self._rebuild_compare_items()
        if self.current is not None:
            self._show_eval(self.current)

    def set_best_reference(self, best: DeckEval | None, label: str = "the best family (#1)") -> None:
        """The best build of the pool (family #1): the efficiency line says how far the deck being edited is from it."""
        self.best_ref, self.best_label = best, label
        if self.current is not None:
            self._show_eval(self.current)

    def set_frozen(self, note: str) -> None:
        """Pause (note != "") / resume (note == "") evaluation: the set is being recalculated and its model is unavailable."""
        was = bool(self.frozen)
        self.frozen = note or ""
        if self.frozen:
            self._swap_worker.cancel()
            self._plan_worker.cancel()
            self._swap_timer.stop()
            self.status.setText(self.frozen)
        elif was:
            self.status.setText("")
            self._seq += 1
            self._timer.start()

    def _refresh_family(self) -> None:
        f = self.family
        chips: dict[int, str] = {}
        tips: dict[int, str] = {}
        if not f:
            self.family_box.setVisible(False)
            self.deck_pile.chips, self.deck_pile.tips = chips, tips
            self.deck_pile.update()
            return
        good, bad = semantic()["good"].name(), semantic()["bad"].name()
        nb = int(self.fam_meta.get("n_draws") or 0)
        step = f"1/{nb} = {100.0 / nb:.1f}%" if nb else "?"
        lab = color_tag(f.get("colors", ""), f.get("splash", ""))
        ht = H.FAMILY_HELP
        ptips = {"head": ht["head"], "pbest": ht["pbest"].replace("(number of refits)", str(nb) if nb else "the number of refits"), "gap": ht["gap"],
                 "core": ht["core"], "flex": ht["flex"], "why": ht["why"], "extra": ht["extra"]}
        head = f"<b>Family #{int(f.get('rank', 0)) + 1}</b> · {lab}" + (f" · {f['archetype']}" if f.get("archetype") else "")
        lines = [tipped(head, "head")]
        lines.append(tipped(f"<b>P(best) {f['p_best'] * 100:.0f}%</b> <span style='color:gray'>= share of {nb} model refits in which "
                            f"the best deck of this family beats every other candidate (steps of {step})</span>", "pbest"))
        if int(f.get("rank", 0)) == 0:
            lines.append(tipped("Top family: best win rate under the full-data fit.", "gap"))
        else:
            g = f["gap_pp"]
            col = bad if f["gap_lo_pp"] > 0 else "gray"
            lines.append(tipped(f"Gap vs #1: <b style='color:{col}'>{-g:+.2f} pp</b> <span style='color:gray'>80% interval "
                                f"[{-f['gap_hi_pp']:+.2f}, {-f['gap_lo_pp']:+.2f}] over the refits; beats #1 in "
                                f"{f['p_beats_top'] * 100:.0f}% of them</span>", "gap"))
        lines.append(tipped(f"<span style='color:gray'>{f.get('n_members', 0)} similar decks found · shown because: "
                            f"{f.get('why', '')}</span>", "why"))
        core = f.get("core", [])
        n_core = sum(c["count"] for c in core)
        lines.append(tipped(f"<b>Core ({n_core} cards, in ≥ 90% of this family's best decks):</b>", "core") + " "
                     + (", ".join(f"{c['name']}" + (f" ×{c['count']}" if c["count"] > 1 else "") for c in core) or "—"))
        flex = f.get("flex", [])
        if flex:
            lines.append(tipped("<b>Flex slots:</b>", "flex"))
            for x in flex:
                alts = ", ".join(f"{a['name']} ({a['freq'] * 100:.0f}%)" for a in x.get("alts", []))
                lines.append(f"&nbsp;• {x['name']}" + (f" ×{x['count']}" if x["count"] > 1 else "")
                             + f" <span style='color:gray'>in {x['freq'] * 100:.0f}%</span>" + (f" · instead: {alts}" if alts else ""))
                tip = f"<b>Flex card:</b> in {x['freq'] * 100:.0f}% of this family's best decks over the model refits."
                if x.get("core_count"):
                    tip += f" (copy {x['core_count'] + 1}; the earlier copies are core)"
                if alts:
                    tip += "\nAlternatives chosen instead: " + alts
                chips[int(x["card_id"])] = "flex"
                tips[int(x["card_id"])] = tip
        else:
            lines.append(tipped("<span style='color:gray'>No flex slots: the same 40 cards (up to the basic split) in nearly "
                                "every refit.</span>", "flex"))
        extra = [e for e in f.get("extra", []) if e["card_id"] not in {x["card_id"] for x in flex}]
        if extra:
            lines.append(tipped("<span style='color:gray'>Also chosen in some refits: "
                                + ", ".join(f"{e['name']} ({e['freq'] * 100:.0f}%)" for e in extra[:5]) + "</span>", "extra"))
        self.family_label.set_content("<br>".join(lines), ptips)
        self.family_box.setVisible(True)
        self.family_box.set_title(f"Deck family #{int(f.get('rank', 0)) + 1} · P(best) {f['p_best'] * 100:.0f}%")
        self.deck_pile.chips, self.deck_pile.tips = chips, tips
        self.deck_pile.update()

    # ------------------------------------------------------------------ data in
    def load(self, set_code: str, fmt: str, deck: DeckEval, pool: dict[int, int] | None, info: dict[int, dict] | None,
             betas: dict[int, dict] | None = None) -> None:
        self.set_code, self.fmt = set_code, fmt
        self._smoothing_modelled = self._manasim_smooths(set_code)
        self._refresh_mana_tips()
        self.m_match.set_tip(H.match_wr_tip(fmt))
        self.m_score.set_tip(H.score_tip(fmt))
        self.m_wins.set_tip(H.exp_wins_tip(fmt))
        self.m_pmax.set_tip(H.p_max_tip(fmt))
        self.rule_label.setText(run_rule_text(fmt))
        self.rule_label.setToolTip(H.exp_wins_tip(fmt))
        self.info = dict(info or {})
        self.betas = dict(betas or {})
        self.side_vals = None
        self.level_note = None
        cn = getattr(self.api, "calibration_note", None)
        if cn is not None:
            try:
                self.level_note = cn(set_code, fmt)
            except Exception:  # noqa: BLE001 - cosmetic
                log.debug("calibration_note failed", exc_info=True)
        self.orig_eval = deck
        d: dict[int, int] = {}
        for c in deck.cards:
            if c.card_id is None:
                continue
            d[c.card_id] = d.get(c.card_id, 0) + c.count
            if c.card_id not in self.info:
                self.info[c.card_id] = unknown_info(c.name)
        self._basic_ids = {}
        for cid, dd in self.info.items():
            if dd.get("is_basic") or (dd.get("is_land") and dd.get("name") in BASIC_NAMES):
                col = next((k for k, v in BASIC_OF.items() if v == dd.get("name")), None)
                if col:
                    self._basic_ids[col] = cid
        self.pool = {int(k): int(v) for k, v in (pool or {}).items() if not self._is_basic(int(k))}
        for cid, n in d.items():   # the deck is always a subset of the pool
            if not self._is_basic(cid):
                self.pool[cid] = max(self.pool.get(cid, 0), n)
        self.deck = dict(d)
        self.orig = dict(d)
        self._history = []
        self._future = []
        self.current = deck
        self.locked, self.excluded = {}, set()
        self.expl = self.swaps = None
        self.syn = self.arch = self.plan = None
        self.family, self.fam_meta = None, {}
        self.best_ref = None
        self._refresh_family()
        self._plan_worker.cancel()
        self._pre_rebuild = None
        self._show_eval(deck)
        self.store.prefetch(set_code, [self.info[c]["name"] for c in list(self.pool) + list(self.deck) + list(self._basic_ids.values())
                                       if c in self.info])
        self._rebuild_compare_items()
        self._refresh()
        self._refresh_explain()
        self._refresh_insight()
        self._refresh_plan_ui()
        self._request_explain()

    def _manasim_smooths(self, set_code: str) -> bool:
        """True when the deployed mana simulation models Arena's Bo1 hand smoother (simulator v2 applied)."""
        fn = getattr(self.api, "model_components", None)
        if fn is None:
            return False
        try:
            ms = (fn(set_code) or {}).get("manasim") or {}
            return bool(ms.get("applied")) and int(ms.get("version") or 1) >= 2
        except Exception:  # noqa: BLE001 - cosmetic
            return False

    def _is_basic(self, cid: int) -> bool:
        d = self.info.get(cid)
        return bool(d and (d.get("is_basic") or d.get("name") in BASIC_NAMES))

    # ------------------------------------------------------------------ editing
    def side(self) -> dict[int, int]:
        return {cid: n - self.deck.get(cid, 0) for cid, n in self.pool.items() if n - self.deck.get(cid, 0) > 0}

    def move(self, cid: int, from_zone: str, all_copies: bool = False) -> bool:
        """Move one copy (or all) of `cid` out of `from_zone` ("deck" or "side") into the other zone; from "basic" (the
        basic-land strip, unlimited supply) one copy of the basic is added to the deck."""
        if from_zone == "basic":
            col = next((c for c, b in self._basic_ids.items() if b == cid), None)
            return col is not None and self.add_basic(col, +1)
        src = self.deck if from_zone == "deck" else self.side()
        n = src.get(cid, 0)
        if n <= 0:
            return False
        k = n if all_copies else 1
        self._push()
        if from_zone == "deck":
            self.deck[cid] = n - k
            if self.deck[cid] <= 0:
                del self.deck[cid]
        else:
            self.deck[cid] = self.deck.get(cid, 0) + k
        self._changed()
        return True

    def add_basic(self, color: str, delta: int) -> bool:
        cid = self._basic_ids.get(color)
        if cid is None:
            return False
        n = self.deck.get(cid, 0) + delta
        if n < 0:
            return False
        self._push()
        if n == 0:
            self.deck.pop(cid, None)
        else:
            self.deck[cid] = n
        self._changed()
        return True

    def undo(self) -> None:
        if self._history:
            self._future.append(dict(self.deck))
            self.deck = self._history.pop()
            self._changed(push_history=False)

    def redo(self) -> None:
        if self._future:
            self._history.append(dict(self.deck))
            self.deck = self._future.pop()
            self._changed(push_history=False)

    def reset(self) -> None:
        if self.deck != self.orig:
            self._push()
            self.deck = dict(self.orig)
            self._changed()

    def _push(self) -> None:
        self._future = []
        self._history.append(dict(self.deck))
        del self._history[:-200]

    def _changed(self, push_history: bool = True) -> None:
        self._normalise_locks()
        self.swaps = None
        self.plan = None
        self._swap_worker.cancel()
        self._plan_worker.cancel()
        self._swap_timer.stop()
        self._refresh()
        self._refresh_swaps_ui()
        self._refresh_plan_ui()
        self.edited.emit(self.deck != self.orig)
        self._seq += 1
        self.status.setText(self.frozen or "Evaluating…")
        self._timer.start()

    def apply_swap(self, out_id: int, in_id: int) -> bool:
        """Replace one copy of `out_id` in the deck by one copy of `in_id` from the sideboard (one undo step)."""
        if self.deck.get(out_id, 0) <= 0 or self.side().get(in_id, 0) <= 0:
            return False
        self._push()
        self.deck[out_id] -= 1
        if self.deck[out_id] <= 0:
            del self.deck[out_id]
        self.deck[in_id] = self.deck.get(in_id, 0) + 1
        self._changed()
        return True

    # ------------------------------------------------------------------ locks and exclusions
    def _normalise_locks(self) -> None:
        """Manual edits win over locks: a lock never exceeds the copies in the deck, a card in the deck is not excluded."""
        for cid in list(self.locked):
            n = min(self.locked[cid], self.deck.get(cid, 0))
            if n <= 0:
                del self.locked[cid]
            else:
                self.locked[cid] = n
        self.excluded -= {c for c in self.excluded if self.deck.get(c, 0) > 0}

    def lock_action(self, cid: int, action: str) -> None:
        if self._is_basic(cid):
            return
        if action in ("lock", "lock1"):
            self.lock_card(cid, 1 if action == "lock1" else None)
        elif action == "exclude":
            self.exclude_card(cid)
        elif action == "clear":
            self.clear_lock(cid)

    def lock_card(self, cid: int, copies: int | None = None) -> bool:
        """Lock `copies` (default: all copies now in the deck, at least one) of `cid`; a sideboard card enters the deck."""
        if self.deck.get(cid, 0) <= 0:
            if self.side().get(cid, 0) <= 0:
                return False
            self._push()
            self.deck[cid] = 1
            moved = True
        else:
            moved = False
        self.excluded.discard(cid)
        self.locked[cid] = max(1, min(copies if copies is not None else self.deck[cid], self.deck[cid]))
        if moved:
            self._changed()
        else:
            self._lock_state_changed()
        return True

    def exclude_card(self, cid: int) -> bool:
        """Mark `cid` as excluded from rebuilds; its copies leave the deck now."""
        if self.pool.get(cid, 0) <= 0:
            return False
        self.locked.pop(cid, None)
        n = self.deck.get(cid, 0)
        if n > 0:
            self._push()
            del self.deck[cid]
            self.excluded.add(cid)
            self._changed()
        else:
            self.excluded.add(cid)
            self._lock_state_changed()
        return True

    def clear_lock(self, cid: int) -> None:
        if self.locked.pop(cid, None) is not None or cid in self.excluded:
            self.excluded.discard(cid)
            self._lock_state_changed()

    def clear_locks(self) -> None:
        if self.locked or self.excluded:
            self.locked, self.excluded = {}, set()
            self._lock_state_changed()

    def _lock_state_changed(self) -> None:
        self._refresh()
        self._request_swaps()

    def rebuild(self) -> None:
        """Run the builder for this pool with the current locks / exclusions in a worker (button toggles to Cancel)."""
        if self._rebuild.running:
            self._rebuild.cancel()
            self.rebuild_btn.setEnabled(False)
            return
        fn = getattr(self.api, "build_from_pool", None)
        if fn is None or not self.pool or self.frozen:
            return
        api, sc, fmt = self.api, self.set_code, self.fmt
        pool, locked, excluded = dict(self.pool), dict(self.locked), sorted(self.excluded)

        def job(progress, cancel):
            return api.build_from_pool(sc, pool, fmt, n_decks=1, progress=progress, cancel=cancel,
                                       locked=locked or None, excluded=excluded or None)
        self.status.setText("Rebuilding…")
        self._rebuild.run(job, label=f"rebuild deck {sc} {fmt}")
        self._update_buttons()

    def _on_rebuilt(self, res: object) -> None:
        decks = getattr(res, "decks", None) or []
        if not decks:
            self.status.setText("The rebuild found no legal deck.")
            return
        ev: DeckEval = decks[0]
        new = {c.card_id: c.count for c in ev.cards if c.card_id is not None}
        before = self.current
        self._pre_rebuild = (dict(self.deck), before)
        if new == self.deck:
            self.status.setText("")
            temp_flash(self.copy_msg, "Already the best deck that respects the locks", 6000)
            self._rebuild_compare_items()
            return
        self._push()
        self.deck = new
        for c in ev.cards:
            if c.card_id is not None and c.card_id not in self.info:
                self.info[c.card_id] = unknown_info(c.name)
        self._changed()
        self._rebuild_compare_items(select_pre=True)
        notes = [w for w in getattr(res, "warnings", []) if "locked" in w or "lock" in w]
        gain = 100 * (ev.game_wr - before.game_wr) if before is not None else 0.0
        temp_flash(self.copy_msg, f"Rebuilt: {gain:+.1f} pp vs the previous deck" + (" · " + " ".join(notes) if notes else ""), 8000)

    def _on_rebuild_error(self, tb: str) -> None:
        from .widgets import short_error

        self.status.setText("Rebuild failed.")
        self.errorOccurred.emit(short_error(tb), tb)

    def _on_rebuild_cancelled(self) -> None:
        self.status.setText("Rebuild cancelled.")

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, deck: dict[int, int]) -> tuple[DeckEval, Any]:
        fn = getattr(self.api, "evaluate_deck", None)
        if fn is None:
            raise RuntimeError("this backend cannot evaluate decks")
        ev = fn(self.set_code, deck, self.fmt)
        expl = syn = arch = None
        side_vals = None
        cvf = getattr(self.api, "card_values", None)
        if cvf is not None and self.pool:
            side_ids = [c for c, n in self.pool.items() if n - deck.get(c, 0) > 0 and not self._is_basic(c)]
            try:                              # what one added copy of each sideboard card is worth in THIS deck (its castability here)
                side_vals = cvf(self.set_code, self.fmt, deck=deck, cids=side_ids)
            except Exception:  # noqa: BLE001 - the badges fall back to the plain beta
                log.debug("card_values for the sideboard failed", exc_info=True)
        ef = getattr(self.api, "explain_deck", None)
        if ef is not None:
            try:
                expl = ef(self.set_code, deck, self.fmt)
            except Exception:  # noqa: BLE001 - the explanation is optional
                log.debug("explain_deck failed", exc_info=True)
        sf = getattr(self.api, "synergies", None)
        if sf is not None:
            try:
                syn = sf(self.set_code, deck, dict(self.pool), self.fmt)
            except Exception:  # noqa: BLE001 - optional
                log.debug("synergies failed", exc_info=True)
        af = getattr(self.api, "archetype_of", None)
        if af is not None:
            try:
                arch = af(self.set_code, deck, self.fmt)
            except Exception:  # noqa: BLE001 - optional
                log.debug("archetype_of failed", exc_info=True)
        return ev, expl, syn, arch, side_vals

    def _submit(self) -> None:
        if self.frozen:
            self.status.setText(self.frozen)
            return
        if not self.deck:
            self.status.setText("The deck is empty.")
            return
        self._evaluator.submit(self._seq, self.deck)

    def _on_eval(self, seq: int, res: object) -> None:
        if seq < self._shown_seq:
            return
        self._shown_seq = seq
        res, self.expl, self.syn, self.arch, self.side_vals = res  # type: ignore[misc]
        self.current = res  # type: ignore[assignment]
        self._show_eval(res)  # type: ignore[arg-type]
        self._refresh_explain()
        self._refresh_insight()
        self._refresh_values()
        self._update_compare()
        if seq == self._seq:
            self.status.setText("")
            self._request_swaps()
        self.evaluated.emit(res)

    def _on_eval_failed(self, seq: int, msg: str) -> None:
        if seq == self._seq:
            self.status.setText(self.frozen or f"Could not evaluate: {msg}")

    def shutdown(self) -> None:
        """Stop this builder's background work (its page is being discarded)."""
        for t in (self._timer, self._swap_timer):
            t.stop()
        for w in (self._swap_worker, self._plan_worker):
            w.cancel()
        self._rebuild.cancel()
        line = getattr(self, "archetype_line", None)
        if line is not None:
            try:
                line.shutdown()
            except (AttributeError, RuntimeError):
                pass

    def wait_idle(self, ms: int = 3000, swaps: bool = False) -> None:
        """Block until the latest edit is evaluated (tests); `swaps=True` also waits for the swap analysis."""
        import time

        end = time.monotonic() + ms / 1000
        while time.monotonic() < end:
            QApplication.processEvents()
            if self._shown_seq == self._seq and not self._timer.isActive() and \
                    (not swaps or self.swaps_ready() or getattr(self.api, "swap_analysis", None) is None):
                return
            time.sleep(0.01)

    def swaps_ready(self) -> bool:
        return self.swaps is not None and bool(getattr(self.swaps, "complete", True)) and not self._swap_timer.isActive()

    # ------------------------------------------------------------------ explanation and swaps
    def _refresh_explain(self) -> None:
        e = self.expl
        self.waterfall.set_explanation(e)
        if e is None:
            self.why_caption.setText("")
            self.sections["why"].set_title("Why this deck")
            return
        self.why_caption.setText(f"Average real {self.fmt.upper()} deck {pct(e.start_wr)} → this deck {pct(e.end_wr)} "
                                 f"({e.total_pp:+.1f} pp) · average pilot")
        self.sections["why"].set_title(f"Why this deck  ·  {e.total_pp:+.1f} pp vs the average deck")

    def _request_explain(self) -> None:
        self._seq += 1
        self._timer.start()

    def _request_swaps(self) -> None:
        if self.frozen or getattr(self.api, "swap_analysis", None) is None or not self.pool or not self.deck:
            return
        self._swap_timer.start()

    def _analyse_swaps(self, payload: dict, cancel: threading.Event, on_partial) -> Any:
        return self.api.swap_analysis(payload["set"], payload["deck"], payload["pool"], payload["fmt"],
                                      locked=payload["locked"], excluded=payload["excluded"], cancel=cancel,
                                      on_partial=on_partial)

    def _compute_plan(self, payload: dict, cancel: threading.Event, on_partial) -> Any:
        return self.api.sideboard_plan(payload["set"], payload["deck"], payload["pool"], payload["fmt"],
                                       locked=payload["locked"], excluded=payload["excluded"], cancel=cancel)

    def _on_plan_done(self, seq: int, res: object) -> None:
        if seq != self._plan_seq:
            return
        self.plan = res
        self._refresh_plan_ui()

    def _on_plan_failed(self, seq: int, msg: str) -> None:
        if seq == self._plan_seq:
            self.plan = None
            self.plan_note.setText(f"Sideboard plan failed: {msg}")

    def _submit_swaps(self) -> None:
        if self.frozen or getattr(self.api, "swap_analysis", None) is None or not self.deck:
            return
        if getattr(self.api, "sideboard_plan", None) is not None and self.fmt == "bo3":
            self._plan_seq += 1
            self._plan_worker.submit(self._plan_seq, dict(set=self.set_code, deck=dict(self.deck), pool=dict(self.pool),
                                                          fmt=self.fmt, locked=dict(self.locked), excluded=set(self.excluded)))
        self._swap_seq += 1
        self._swap_worker.submit(self._swap_seq, dict(set=self.set_code, deck=dict(self.deck), pool=dict(self.pool),
                                                      fmt=self.fmt, locked=dict(self.locked), excluded=set(self.excluded)))
        self._refresh_swaps_ui()

    def _on_swaps_partial(self, seq: int, res: object) -> None:
        if seq != self._swap_seq:
            return
        self.swaps = res
        self._refresh_values()
        self._refresh_swaps_ui()

    def _on_swaps_done(self, seq: int, res: object) -> None:
        if seq != self._swap_seq:
            return
        self.swaps = res
        self._refresh_values()
        self._refresh_swaps_ui()

    def _on_swaps_failed(self, seq: int, msg: str) -> None:
        if seq == self._swap_seq:
            self.swap_list.set_swaps([], f"Swap analysis failed: {msg}")

    def _n_refits(self) -> int | None:
        """Bootstrap refits behind every P(better) / P(best): known from the family meta of the build."""
        n = int(self.fam_meta.get("n_draws") or 0)
        return n or None

    def _card_name(self, cid: int) -> str:
        return (self.info.get(cid) or {}).get("name") or str(cid)

    def _refresh_swaps_ui(self) -> None:
        sw = self.swaps
        if getattr(self.api, "swap_analysis", None) is None:
            self.swap_list.set_swaps([], "Swap analysis is not available.")
            return
        if sw is None:
            self.swap_list.set_swaps([], "Analysing swaps…")
            return
        items = [(x.out_id, x.in_id, f"{self._card_name(x.out_id)} → <b>{self._card_name(x.in_id)}</b>", x.gain_pp, x.p_better)
                 for x in sw.suggestions]
        note = ""
        if not items:
            note = "No swap from the sideboard improves this deck." if getattr(sw, "complete", True) else "Analysing swaps…"
        elif not getattr(sw, "complete", True):
            note = "Analysing swaps…"
        self.swap_list.set_swaps(items, note, self._n_refits())

    # ------------------------------------------------------------------ archetype, synergies, sideboard plan
    def _refresh_insight(self) -> None:
        a = self.arch
        if a is None or getattr(a, "status", "off") != "ok":
            self.arch_label.setText(muted_html(getattr(a, "message", "")) if a is not None else "")
            self.arch_label.setVisible(a is not None and bool(getattr(a, "message", "")))
        else:
            bits = [f"<b>Archetype:</b> {a.label}  <span style='color:gray'>(similarity {a.similarity:.2f})</span>"]
            r = a.rate
            if r is not None:
                bits.append(f"observed {a.rate_fmt.upper()} game win rate <b>{pct(r['shrunk'])}</b> "
                            f"[{pct(r['lo'])}, {pct(r['hi'])}] over {r['games']:,} games (shrunk)")
                if a.rate_note:
                    bits.append(f"<span style='color:gray'>{a.rate_note}</span>")
            if a.missing:
                bits.append("signature cards you do not play: " + ", ".join(
                    f"{m['name']} ({m['rate'] * 100:.0f}% of decks)" for m in a.missing[:4]))
            self.arch_label.setText("<br>".join(bits))
            self.arch_label.setVisible(True)
        v = self.syn
        good, bad = semantic()["good"].name(), semantic()["bad"].name()
        if v is None:
            self.syn_box.setVisible(False)
        elif v.status != "ok":
            self.syn_box.setVisible(True)
            self.syn_label.setText(muted_html(v.message))
        else:
            self.syn_box.setVisible(True)
            lines: list[str] = []
            for r in v.present[:8]:
                col = good if r.pp > 0 else bad
                lines.append(f"{self._card_name(r.a)} + {self._card_name(r.b)}  <b style='color:{col}'>{r.pp:+.2f} pp</b> "
                             f"<span style='color:gray'>P {r.p_pos * 100:.0f}% · {r.games:,.0f} games</span>")
            if not lines:
                lines.append("<span style='color:gray'>No pair in this deck has a reliable synergy.</span>")
            cand = v.candidates[:5]
            if cand:
                lines.append("<b>Sideboard cards that would add a synergy:</b>")
                for r in cand:
                    lines.append(f"{self._card_name(r.b)} with {self._card_name(r.a)}  <b style='color:{good}'>{r.pp:+.2f} pp</b> "
                                 f"<span style='color:gray'>P {r.p_pos * 100:.0f}%</span>")
            lines.append(f"<span style='color:gray'>Shown: P(sign) ≥ {v.p_min * 100:.0f}% and ≥ {v.min_pp:g} pp; "
                         f"{v.n_reliable} such pairs in the whole set.</span>")
            self.syn_label.setText("<br>".join(lines))
            self.syn_box.set_title(f"Synergies  ·  {len(v.present)} in this deck, {len(v.candidates)} to add")

    def _plan_visible(self) -> bool:
        return self.fmt == "bo3" and getattr(self.api, "sideboard_plan", None) is not None

    def _refresh_plan_ui(self, *_args) -> None:
        self.plan_box.setVisible(self._plan_visible())
        chips: dict[int, str] = {}
        tips: dict[int, str] = {}
        pl = self.plan
        if not self._plan_visible():
            self.side_pile.chips, self.side_pile.tips = {}, {}
            self.side_pile.update()
            return
        if pl is None:
            self.plan_note.setText("Computing the sideboard plan…" if self.deck else "")
            self.plan_combo.setVisible(False)
            self.plan_swaps.set_swaps([], "")
        elif pl.status != "ok":
            self.plan_note.setText(pl.message)
            self.plan_combo.setVisible(False)
            self.plan_swaps.set_swaps([], "")
        else:
            cur = self.plan_combo.currentData()
            self.plan_combo.blockSignals(True)
            self.plan_combo.clear()
            self.plan_combo.addItem(f"All opponents with a reliable swap ({len(pl.swaps)})", "all")
            for o in pl.opponents:
                n = len(pl.swaps.get(o, []))
                self.plan_combo.addItem(f"vs {o}: {n} reliable swap{'s' if n != 1 else ''}", o)
            self.plan_combo.setCurrentIndex(max(self.plan_combo.findData(cur), 0))
            self.plan_combo.blockSignals(False)
            self.plan_combo.setVisible(True)
            sel = self.plan_combo.currentData()
            opps = [o for o in pl.opponents if o in pl.swaps] if sel == "all" else [sel]
            items = []
            for o in opps:
                for sw in pl.swaps.get(o, []):
                    label = f"<b>vs {o}</b>: {self._card_name(sw.out_id)} → <b>{self._card_name(sw.in_id)}</b>"
                    items.append((sw.out_id, sw.in_id, label, sw.gain_pp, sw.p_better))
            note = pl.message if not pl.swaps else ""
            if sel not in ("all", None) and not items:
                note = f"No reliable swap against {sel} (needs ≥ {pl.thresholds.min_pp:g} pp and P ≥ {pl.thresholds.p_min:.0%})."
            elif items:
                note = (f"Reliable swaps: ≥ {pl.thresholds.min_pp:g} pp against that opponent, P ≥ {pl.thresholds.p_min:.0%}; "
                        "the card must also be reliably better against it than on average.")
            self.plan_note.setText(note)
            self.plan_swaps.set_swaps(items, "", self._n_refits())
            for cid, sws in pl.chips.items():
                if self.side().get(cid, 0) <= 0:
                    continue
                names_ = list(dict.fromkeys(sw.opp for sw in sws))
                opp_txt = ", ".join(names_[:2]) + (f" +{len(names_) - 2}" if len(names_) > 2 else "")
                chips[cid] = f"SB vs {opp_txt}"
                tips[cid] = "<b>Sideboard card vs specific opponents</b><br>" + "<br>".join(
                    f"Against {sw.opp}: take out {self._card_name(sw.out_id)} for {self._card_name(sw.in_id)}: "
                    f"{sw.gain_pp:+.2f} pp win rate vs {sw.opp} (P(better) {0 if sw.p_better is None else sw.p_better * 100:.0f}%); "
                    f"this card is {sw.delta_in_pp:+.2f} pp better vs {sw.opp} than on average (P {sw.p_delta_in * 100:.0f}%)"
                    for sw in sws[:4]) + ("<br><i>Only marked when both effects are reliable (P >= 90%, >= 0.5 pp) across the "
                                          "model refits; several opponents are tested, so treat it as a hint.</i>")
        self.side_pile.chips, self.side_pile.tips = chips, tips
        self.side_pile.update()

    # ------------------------------------------------------------------ comparison with other decks
    def set_alternatives(self, alts: list[tuple[str, DeckEval]], compare_first: bool = False) -> None:
        """Other builds of the same pool to compare with (label, DeckEval); `compare_first` preselects the first (the user's own
        deck is compared with the best family from the start)."""
        self.alts = list(alts)
        self._compare_default = "alt0" if (compare_first and self.alts) else "orig"
        self.compare_combo.blockSignals(True)
        self.compare_combo.clear()
        self.compare_combo.blockSignals(False)
        self._rebuild_compare_items()

    def _rebuild_compare_items(self, select_pre: bool = False) -> None:
        cur = self.compare_combo.currentData() if self.compare_combo.count() else self._compare_default
        self.compare_combo.blockSignals(True)
        self.compare_combo.clear()
        self.compare_combo.addItem(self.origin_label[:1].upper() + self.origin_label[1:], "orig")
        for k, (label, _ev) in enumerate(self.alts):
            self.compare_combo.addItem(label, f"alt{k}")
        if self._pre_rebuild is not None:
            self.compare_combo.addItem("Deck before the last rebuild", "pre")
        self.compare_combo.addItem("(none)", "none")
        want = "pre" if select_pre and self._pre_rebuild is not None else cur
        i = self.compare_combo.findData(want)
        self.compare_combo.setCurrentIndex(max(i, 0))
        self.compare_combo.blockSignals(False)
        self._update_compare()

    def _compare_target(self) -> tuple[str, dict[int, int], DeckEval | None] | None:
        key = self.compare_combo.currentData()
        if key == "orig":
            return f"the {self.origin_label}" if not self.origin_label.startswith("your") else self.origin_label, dict(self.orig), self.orig_eval
        if key == "pre" and self._pre_rebuild is not None:
            return "the deck before the rebuild", dict(self._pre_rebuild[0]), self._pre_rebuild[1]
        if isinstance(key, str) and key.startswith("alt"):
            label, ev = self.alts[int(key[3:])]
            d: dict[int, int] = {}
            for c in ev.cards:
                if c.card_id is not None:
                    d[c.card_id] = d.get(c.card_id, 0) + c.count
            return label, d, ev
        return None

    def diff_against(self, other: dict[int, int]) -> tuple[dict[int, int], dict[int, int]]:
        """(cards added, cards removed) going from `other` to the current deck (copies, basics included)."""
        add = {c: n - other.get(c, 0) for c, n in self.deck.items() if n > other.get(c, 0)}
        rem = {c: n - self.deck.get(c, 0) for c, n in other.items() if n > self.deck.get(c, 0)}
        return add, rem

    def _update_compare(self, *_args) -> None:
        tgt = self._compare_target()
        self.deck_pile.marks, self.side_pile.marks = {}, {}
        if tgt is None:
            self.compare_label.setText("")
            self.deck_pile.update()
            self.side_pile.update()
            return
        label, other, ev = tgt
        add, rem = self.diff_against(other)
        good, bad = semantic()["good"].name(), semantic()["bad"].name()

        def fmt(counts: dict[int, int], basics: bool) -> list[str]:
            return [f"{self._card_name(c)}" + (f" ×{n}" if n > 1 else "") for c, n in
                    sorted(counts.items(), key=lambda kv: self._card_name(kv[0])) if self._is_basic(c) == basics]
        lines = []
        if self.current is not None and ev is not None:
            d = 100 * (self.current.game_wr - ev.game_wr)
            col = good if d > 0.05 else (bad if d < -0.05 else "gray")
            lines.append(f"<b style='color:{col}'>{d:+.2f} pp</b> game win rate vs {label} "
                         f"({pct(self.current.game_wr)} vs {pct(ev.game_wr)})")
            if self.current.match_wr is not None and ev.match_wr is not None:
                lines.append(f"match {100 * (self.current.match_wr - ev.match_wr):+.2f} pp · score "
                             f"{self.current.score - ev.score:+.0f}")
        if not add and not rem:
            lines.append("Same cards.")
        else:
            if add:
                lines.append(f"<span style='color:{good}'>+ only in this deck:</span> "
                             + ", ".join(fmt(add, False) + fmt(add, True)))
            if rem:
                lines.append(f"<span style='color:{bad}'>− only in {label}:</span> "
                             + ", ".join(fmt(rem, False) + fmt(rem, True)))
        self.compare_label.setText("<br>".join(lines))
        if self.current is not None and ev is not None:
            self.compare_box.set_title(f"Compare with  ·  {100 * (self.current.game_wr - ev.game_wr):+.1f} pp vs {label}")
        for c in add:
            if not self._is_basic(c):
                self.deck_pile.marks[c] = "+"
        for c in rem:
            if not self._is_basic(c):
                self.side_pile.marks[c] = "-"
        self.deck_pile.update()
        self.side_pile.update()

    # ------------------------------------------------------------------ display
    def _opts(self) -> dict:
        return dict(group=self.group_combo.currentData(), sort=self.sort_combo.currentData(), split=self.split_cb.isChecked(),
                    card_w=self.size_slider.value(), show_values=self.values_cb.isChecked(),
                    badge=self.badge_combo.currentData())

    def _opts_changed(self, *_args) -> None:
        o = self._opts()
        s = new_settings()
        s.setValue("builder/group", o["group"])
        s.setValue("builder/sort", o["sort"])
        s.setValue("builder/split", o["split"])
        s.setValue("builder/values", o["show_values"])
        s.setValue("builder/card_w", o["card_w"])
        s.setValue("builder/badge", o["badge"])
        self._refresh()

    def _values(self, zone: str) -> dict[int, tuple[float, bool]]:
        if self.badge_combo.currentData() == "var":
            return self._values_var(zone)
        wr = self.current.game_wr if self.current is not None else 0.55
        out: dict[int, tuple[float, bool]] = {}
        if zone == "deck" and self.current is not None:
            for c in self.current.cards:
                if c.card_id is not None and c.value_pp is not None:
                    out[c.card_id] = (c.value_pp, c.low_data)
        ids = self.deck if zone == "deck" else self.side()
        for cid in ids:
            if cid in out or self._is_basic(cid):
                continue
            sv = (self.side_vals or {}).get(cid)
            if sv is not None and zone != "deck" and sv.get("value_pp") is not None:
                out[cid] = (float(sv["value_pp"]), bool(sv.get("low_data")))        # castability- and dead-cards-aware
                continue
            b = self.betas.get(cid)
            if b is not None:
                out[cid] = (value_pp(float(b["beta"]), wr, zone == "deck"), bool(b.get("low_data")))
        return out

    def _values_var(self, zone: str) -> dict[int, tuple[float, bool]]:
        """Above-replacement badges: deck cards = q(deck) - q(deck with the best sideboard replacement) in pp; sideboard
        cards = gain of their best slot.  Empty until the swap analysis has produced something."""
        sw = self.swaps
        out: dict[int, tuple[float, bool]] = {}
        if sw is None:
            return out
        ids = self.deck if zone == "deck" else self.side()
        for cid in ids:
            if self._is_basic(cid):
                continue
            low = bool((self.betas.get(cid) or {}).get("low_data"))
            if zone == "deck" and cid in sw.var_pp:
                out[cid] = (sw.var_pp[cid], low)
            elif zone != "deck" and cid in sw.side_best:
                out[cid] = (sw.side_best[cid].gain_pp, low)
        return out

    def _expand(self, counts: dict[int, int]) -> list[int]:
        return [cid for cid, n in counts.items() for _ in range(max(0, n))]

    def _refresh(self) -> None:
        o = self._opts()
        side = self.side()
        for pile in (self.deck_pile, self.side_pile):
            pile.locks, pile.excluded = dict(self.locked), set(self.excluded)
        self.deck_pile.set_content(self.set_code, self._expand(self.deck), self.info, self._values("deck"), o)
        self.side_pile.set_content(self.set_code, self._expand(side), self.info, self._values("side"), o)
        nl, ne = sum(self.locked.values()), len(self.excluded)
        self.lock_info.setText("" if not (nl or ne) else
                               f"🔒 {nl} locked cop{'y' if nl == 1 else 'ies'} · ✕ {ne} excluded card{'s' if ne != 1 else ''}")
        self.lock_info.setVisible(bool(nl or ne))
        self.clear_locks_btn.setVisible(bool(nl or ne))
        self._update_compare()
        n_deck = sum(self.deck.values())
        lands = sum(n for cid, n in self.deck.items() if (self.info.get(cid) or {}).get("is_land"))
        cre = sum(n for cid, n in self.deck.items() if (self.info.get(cid) or {}).get("is_creature")
                  and not (self.info.get(cid) or {}).get("is_land"))
        spells = n_deck - lands
        self.deck_title.setText(f"Deck ({n_deck})  ·  {lands} lands")
        self.deck_title.setToolTip(H.tip("Deck", f"{n_deck} cards: {spells} spells ({cre} creatures, {spells - cre} other) and "
                                                 f"{lands} lands.  The model was fitted on 40-card decks with 16–17 lands."))
        self.side_title.setText(f"Sideboard ({sum(side.values())})")
        self.side_title.setToolTip("The rest of your pool: every card that is not in the deck.  Basic lands are in the strip "
                                   "to the right (unlimited).")
        pal_warn = semantic()["warn"].name()
        size_txt = (f"<b>{n_deck}</b> / 40 cards" if n_deck == 40 else
                    f"<b style='color:{pal_warn}'>{n_deck} / 40 cards</b>")
        self.counts.setText(f"{size_txt} · {lands} lands · {cre} creatures · {spells - cre} other spells")
        # curve
        cre_c, non_c = [0] * 7, [0] * 7
        pips = {c: 0 for c in "WUBRG"}
        srcs = {c: 0 for c in "WUBRG"}
        for cid, n in self.deck.items():
            d = self.info.get(cid) or {}
            if d.get("is_land"):
                prod = d.get("produces") or ""
                if not prod and d.get("name") in BASIC_NAMES:
                    prod = next((k for k, v in BASIC_OF.items() if v in d.get("name", "")), "")
                for c in set(prod) & set("WUBRG"):
                    srcs[c] += n
                continue
            b = min(max(int(round(float(d.get("mana_value") or 0.0))), 1), 7) - 1
            (cre_c if d.get("is_creature") else non_c)[b] += n
            for ch in d.get("cost") or "":
                if ch in pips:
                    pips[ch] += n
            for c in set(d.get("produces") or "") & set("WUBRG"):   # mana rocks / dorks fix too
                srcs[c] += n
        self.curve.set_counts(cre_c, non_c)
        nbasic = {col: self.deck.get(self._basic_ids.get(col, -1), 0) for col in "WUBRG"}
        for col, (lp, ls, lb) in self._color_rows.items():
            lp.setText(str(pips[col]))
            ls.setText(str(srcs[col]))
            lb.setText(str(nbasic[col]))
        self.basic_strip.set_state(self.set_code, self._basic_ids, nbasic)
        self.steppers.set_counts(nbasic, set(self._basic_ids))
        self._update_buttons()

    def _refresh_values(self) -> None:
        o = self._opts()
        self.deck_pile.values = self._values("deck")
        self.side_pile.values = self._values("side")
        if o["show_values"]:
            self.deck_pile.update()
            self.side_pile.update()
        if o["sort"] == "value":
            self._refresh()

    def _show_eval(self, d: DeckEval) -> None:
        ref = self.expl.start_wr if self.expl is not None else None
        sup = list(getattr(d, "support", None) or [])
        sub = pct_interval(d.game_wr_lo, d.game_wr_hi)
        if ref is not None:
            sub += f"\navg real deck {pct(ref)}"
        if sup:
            sub += "\n⚠ less reliable"
        self.m_game.set(pct(d.game_wr), sub, tip=H.game_wr_tip(self.fmt, ref, d.game_wr, bool(sup), level=self.level_note),
                        sub_role="warn" if sup else None)
        if d.match_wr is None:
            self.m_match.set("—", "Bo1: no match", tip=H.match_wr_tip(self.fmt))
        else:
            self.m_match.set(pct(d.match_wr), "Bo3: p²(3−2p)", tip=H.match_wr_tip(self.fmt))
        self.m_score.set(f"{d.score:.0f}", f"percentile among real {self.fmt.upper()} decks · 80% interval {d.score_lo:.0f}–{d.score_hi:.0f}",
                         "good" if d.score >= 75 else ("bad" if d.score < 25 else None), tip=H.score_tip(self.fmt))
        wins, _losses = RUN_RULES.get(self.fmt, (7, 3))
        self.m_wins.set(f"{d.exp_wins:.2f}", "wins per run", tip=H.exp_wins_tip(self.fmt))
        self.m_pmax.set(pct(d.p_max_wins), f"P({wins} wins)", tip=H.p_max_tip(self.fmt))
        if self.best_ref is not None:
            e = 100.0 * (d.game_wr - self.best_ref.game_wr)
            col = (semantic()["good"] if e >= -0.5 else semantic()["warn"] if e >= -2.0 else semantic()["bad"]).name()
            self.eff_label.setText(f"<b>Efficiency vs {self.best_label}:</b> <span style='color:{col}; font-weight:600'>{e:+.2f} pp</span> "
                                   "game win rate" + (" (this is the best build)" if e >= -0.05 else ""))
            self.eff_label.setToolTip(H.tip("Efficiency", "How far the deck being edited is from the best build of the pool "
                                            f"({self.best_label}), in percentage points of game win rate.  0 or positive: nothing "
                                            "better was found; −2 or less: a clearly better build exists (see Suggested swaps)."))
            self.eff_label.setVisible(True)
        else:
            self.eff_label.setVisible(False)
        if self.orig_eval is not None:
            delta = 100 * (d.game_wr - self.orig_eval.game_wr)
            role = "good" if delta > 0.05 else ("bad" if delta < -0.05 else None)
            self.m_delta.set(f"{delta:+.1f} pp", "game win rate" + ("\n⚠ less reliable" if sup else ""), role,
                             tip=H.delta_tip(sup, f"the {self.origin_label}" if not self.origin_label.startswith("your") else self.origin_label),
                             sub_role="warn" if sup else None)
        self.support_box.set_lines(sup)
        for key, label, _t in MANA_STATS:
            v = d.mana.get(key)
            self._mana_labels[key].setText("—" if v is None else f"{v * 100:.1f}%")
        sm = [d.mana.get(k) for k in ("p_screw", "p_flood", "p_color_screw")]
        if all(v is not None for v in sm):
            self.sections["mana"].set_title(f"Mana simulation  ·  screw {sm[0] * 100:.0f}% · flood {sm[1] * 100:.0f}% "
                                            f"· colour {sm[2] * 100:.0f}%")
        self.bars.set_items(d.breakdown, d.game_wr)
        self.warnings.set_lines(list(d.warnings))
        self.deck_pile.castable = {c.card_id: c.castable for c in d.cards if c.card_id is not None and c.castable is not None}
        self.deck_pile.update()
        self.side_pile.castable = {c: v["castable"] for c, v in (self.side_vals or {}).items() if v.get("castable") is not None}
        self.side_pile.update()

    def _on_hover(self, cid: int, zone: str = "") -> None:
        if cid < 0:
            self.previewCleared.emit()
            return
        self.preview_caption.setToolTip("")
        d = self.info.get(cid) or {}
        name = d.get("name", "")
        self.preview.setText(name)
        bits = [f"<b>{name}</b>"]
        if d.get("types"):
            bits.append(d["types"])
        vals = {**self._values("side"), **self._values("deck")}
        if cid in vals:
            pp, low = vals[cid]
            where = "in the deck" if cid in self.deck else "if added"
            bits.append(f"value {pp:+.2f} pp {where}" + (" · ⚠ little data" if low else ""))
            cpv = (self.deck_pile if cid in self.deck else self.side_pile).castable.get(cid)
            self.preview_caption.setToolTip(H.tip("Value badge", H.value_badge_tip(pp, low, cid in self.deck,
                                                                                  self.badge_combo.currentData(), cpv)))
        cp = self.deck_pile.castable.get(cid)
        if cp is not None and (zone == "deck" or (not zone and cid in self.deck)):
            warn = semantic()["warn"].name()
            bits.append(f"castable on time: <b style='color:{warn if cp < CAST_WARN else 'inherit'}'>{cp * 100:.0f}%</b>"
                        + (" ⚠ the mana struggles to cast it" if cp < CAST_WARN else ""))
        sw = self.swaps
        if sw is not None and not self._is_basic(cid):
            in_deck = (zone == "deck") if zone else cid in self.deck
            good, bad = semantic()["good"].name(), semantic()["bad"].name()

            def gain(x) -> str:
                col = good if x.gain_pp > 0 else bad
                p = "" if x.p_better is None else f", P(better) {x.p_better * 100:.0f}%"
                return f"<span style='color:{col}'><b>{x.gain_pp:+.2f} pp</b></span>{p}"
            if in_deck and cid in sw.deck_best:
                x = sw.deck_best[cid]
                bits.append(f"best swap: → {self._card_name(x.in_id)} {gain(x)}")
                v = sw.var_pp.get(cid)
                if v is not None:
                    bits.append(f"value above replacement {v:+.2f} pp")
            elif not in_deck and cid in sw.side_best:
                x = sw.side_best[cid]
                bits.append(f"best slot: replaces {self._card_name(x.out_id)} {gain(x)}")
        sv = self.syn
        if sv is not None and getattr(sv, "status", "") == "ok" and not self._is_basic(cid):
            for r in sv.by_card.get(cid, [])[:3]:
                other = r.b if r.a == cid else r.a
                if r in sv.present:
                    bits.append(f"synergy with {self._card_name(other)}: {r.pp:+.2f} pp (P {r.p_pos * 100:.0f}%)")
                elif cid not in self.deck or zone == "side":
                    bits.append(f"would add a synergy with {self._card_name(r.a)}: {r.pp:+.2f} pp (P {r.p_pos * 100:.0f}%)")
        pl = self.plan
        if pl is not None and getattr(pl, "status", "") == "ok" and cid in pl.chips and zone != "deck":
            for sw in pl.chips[cid][:3]:
                bits.append(f"SB vs {sw.opp}: take out {self._card_name(sw.out_id)} {sw.gain_pp:+.2f} pp "
                            f"(P(better) {0 if sw.p_better is None else sw.p_better * 100:.0f}%)")
        if cid in self.locked:
            bits.append(f"🔒 locked ({self.locked[cid]} cop{'y' if self.locked[cid] == 1 else 'ies'})")
        elif cid in self.excluded:
            bits.append("✕ excluded from rebuilds")
        self.preview_caption.setText("<br>".join(bits))
        self.previewChanged.emit(self.set_code, name, "<br>".join(bits), self.preview_caption.toolTip())

    def open_sample_hand(self) -> None:
        """Non-modal sample-hand dialog for the deck as it is now (it reads the deck again at every "New hand")."""
        from .samplehand_dialog import SampleHandDialog

        dlg = getattr(self, "_sample_dlg", None)
        if dlg is None:
            dlg = self._sample_dlg = SampleHandDialog(self, self)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _update_buttons(self) -> None:
        self.undo_btn.setEnabled(bool(self._history))
        self.undo_btn.setToolTip("Undo the last change to the deck (Ctrl+Z)" + ("; Ctrl+Y redoes it" if self._future else ""))
        self.reset_btn.setEnabled(self.deck != self.orig)
        running = self._rebuild.running
        self.rebuild_btn.setText("Cancel rebuild" if running else "Rebuild around locks")
        self.rebuild_btn.setEnabled(getattr(self.api, "build_from_pool", None) is not None and bool(self.pool))

    # ------------------------------------------------------------------ export
    def export_text(self) -> str:
        side_names: dict[str, int] = {}
        for cid, n in self.side().items():
            nm = (self.info.get(cid) or {}).get("name", str(cid))
            side_names[nm] = side_names.get(nm, 0) + n
        ev = self.current
        if ev is None or sum(c.count for c in ev.cards if c.card_id is not None) != sum(self.deck.values()):
            ev = None
        if ev is not None and getattr(self.api, "export_arena", None):
            ids_eval = {c.card_id: c.count for c in ev.cards if c.card_id is not None}
            if ids_eval == self.deck:
                return self.api.export_arena(ev, side_names, set_code=self.set_code)
        lines = ["Deck"] + [f"{n} {(self.info.get(cid) or {}).get('name', cid).split(' // ')[0]}" for cid, n in self.deck.items()]
        if side_names:
            lines += ["", "Sideboard"] + [f"{n} {nm.split(' // ')[0]}" for nm, n in side_names.items()]
        return "\n".join(lines) + "\n"

    def copy_export(self) -> None:
        try:
            text = self.export_text()
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.errorOccurred.emit(f"Could not build the Arena export: {exc}", traceback.format_exc())
            return
        QApplication.clipboard().setText(text)
        temp_flash(self.copy_msg, "Copied")
