"""Widgets for the "More detail" blocks of the Report tab: the card explorer (a sortable, searchable table of
every card with enough data).  The data come from `report.explorer_rows` (columns EXPLORER_COLS of the stored report)."""
from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from . import helptext as H
from .widgets import FullTextTipDelegate, muted, semantic, sized_font

BADGE_TEXT = {"strong": "Strong", "likely": "Likely", "weak": "Weak"}
GROUP_TITLE = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green", "gold": "Multicolour", "colorless": "Artifacts & colourless",
               "land": "Lands"}
RARITY_TITLE = {"common": "Common", "uncommon": "Uncommon", "rare": "Rare", "mythic": "Mythic"}
Z_LIKELY, Z_STRONG = 2.0, 3.0
HEADERS = ["Card", "Colour", "Rarity", "Played in", "Value", "How sure?", "Decks", "Value Bo1", "Value Bo3", "Bo1 vs Bo3"]
HEADER_TIPS = [
    "Hover or click a card to see it",
    "Colour group the value is compared within",
    "Rarity",
    "Share of the decks that could play it (pool and colours) that played at least one copy",
    "Points of win rate per copy (pts) against the average card of its colour group that decks play (this format)",
    "How sure we are the value is not luck: Strong, Likely or Weak (hover a cell for the range)",
    "Decks that could have played it",
    "The card's value in Best of 1, same points",
    "The card's value in Best of 3, same points (largely borrowed from Best of 1 when there are few games)",
    "Flagged only when the two formats clearly differ (Likely or Strong, with enough data in both)",
]


def badge_of(z: float | None) -> str:
    if z is None:
        return "weak"
    a = abs(z)
    return "strong" if a >= Z_STRONG else "likely" if a >= Z_LIKELY else "weak"


def pts(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


class ExplorerModel(QAbstractTableModel):
    """One row per card; `Qt.UserRole` holds the sort key (numbers sort as numbers, '–' sorts last when descending)."""

    def __init__(self, ex: dict, fmt: str, parent=None):
        super().__init__(parent)
        self.owned: dict[str, int] = {}                    # card name -> copies in the pool of the Deck builder (a star in the Card column)
        c = {n: i for i, n in enumerate(ex["cols"])}
        self.rows: list[dict] = []
        for r in ex["rows"]:
            v, vo = r[c["value_pp"]], r[c["value_other"]]
            z, dz = r[c["z"]], r[c["dz"]]
            v1, v3 = (v, vo) if fmt == "bo1" else (vo, v)
            clear = dz is not None and abs(dz) >= Z_LIKELY and not r[c["thin"]] and not r[c["thin_other"]]
            self.rows.append(dict(id=r[c["id"]], name=r[c["name"]], group=r[c["group"]], rarity=r[c["rarity"]], pr=r[c["play_rate"]], v=v, z=z,
                                  badge=badge_of(z), n=r[c["n_elig"]], thin=bool(r[c["thin"]]), v1=v1, v3=v3, dz=dz, clear=clear,
                                  diff=None if (v1 is None or v3 is None) else v3 - v1))

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                return HEADERS[section]
            if role == Qt.ItemDataRole.ToolTipRole:
                return HEADER_TIPS[section]
        return None

    def set_owned(self, owned: dict[str, int]) -> None:
        owned = dict(owned or {})
        if owned == self.owned:
            return
        self.owned = owned
        if self.rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, 0))

    def _text(self, r: dict, c: int) -> str:
        if c == 0:
            return ("\u2605 " if r["name"] in self.owned else "") + r["name"]
        if c == 1:
            return GROUP_TITLE.get(r["group"], r["group"])
        if c == 2:
            return RARITY_TITLE.get(r["rarity"], r["rarity"])
        if c == 3:
            return f"{r['pr'] * 100:.0f}%"
        if c == 4:
            return pts(r["v"]) + (" *" if r["thin"] else "")
        if c == 5:
            return BADGE_TEXT[r["badge"]]
        if c == 6:
            return f"{r['n']:,}"
        if c == 7:
            return "–" if r["v1"] is None else pts(r["v1"])
        if c == 8:
            return "–" if r["v3"] is None else pts(r["v3"])
        if r["diff"] is None:
            return "–"
        return (f"Bo3 {pts(r['diff'])}  ({BADGE_TEXT[badge_of(r['dz'])]})" if r["clear"] else "no clear difference")

    def _key(self, r: dict, c: int):
        if c == 0:
            return r["name"].lower()
        if c == 1:
            return GROUP_TITLE.get(r["group"], r["group"])
        if c == 2:
            return ["common", "uncommon", "rare", "mythic"].index(r["rarity"]) if r["rarity"] in RARITY_TITLE else 9
        num = {3: r["pr"], 4: r["v"], 5: abs(r["z"]) * (1 if r["z"] >= 0 else -1), 6: r["n"], 7: r["v1"], 8: r["v3"]}
        if c in num:
            return float("-inf") if num[c] is None else float(num[c])
        return float("-inf") if not r["clear"] else float(abs(r["diff"]))

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r, c = self.rows[index.row()], index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._text(r, c)
        if role == Qt.ItemDataRole.UserRole:
            return self._key(r, c)
        if role == Qt.ItemDataRole.TextAlignmentRole and c in (3, 4, 6, 7, 8):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole and c in (4, 7, 8):
            v = {4: r["v"], 7: r["v1"], 8: r["v3"]}[c]
            if v is not None:
                return QBrush(semantic()["good" if v >= 0 else "bad"])
        if role == Qt.ItemDataRole.ToolTipRole:
            if c == 0 and r["name"] in self.owned:
                n = self.owned[r["name"]]
                return f"In your pool ({n} cop{'y' if n == 1 else 'ies'})"
            if c == 4 or c == 5:
                se_txt = f"z = {r['z']:+.1f}" + ("; limited data in this format: the value leans on the model's prior" if r["thin"] else "")
                return f"{r['name']}: {pts(r['v'])} points per copy against the average {GROUP_TITLE.get(r['group'], r['group']).lower()} card decks play; {se_txt}."
            if c == 9 and r["diff"] is not None:
                return (f"Bo3 minus Bo1: {pts(r['diff'])} points (z = {r['dz']:+.1f}).  "
                        + ("Clear." if r["clear"] else "Not clear, or limited data in one of the formats."))
        return None


class ExplorerProxy(QSortFilterProxyModel):
    """Filters on name text, colour group and rarity."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text, self.group, self.rarity = "", "", ""
        self.setSortRole(Qt.ItemDataRole.UserRole)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_filters(self, text: str, group: str, rarity: str) -> None:
        self.text, self.group, self.rarity = text.strip().casefold(), group, rarity
        self.invalidate()

    def filterAcceptsRow(self, row: int, parent) -> bool:
        r = self.sourceModel().rows[row]
        return ((not self.text or self.text in r["name"].casefold()) and (not self.group or r["group"] == self.group)
                and (not self.rarity or r["rarity"] == self.rarity))


class CardExplorer(QWidget):
    """Search box, group and rarity filters and the table; hovering or selecting a card emits `shown(name, caption)`."""

    shown = Signal(str, str)
    hidden = Signal()                  # the pointer left the table: the floating card goes away

    def __init__(self, ex: dict, fmt: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = ExplorerModel(ex, fmt, self)
        self.proxy = ExplorerProxy(self)
        self.proxy.setSourceModel(self.model)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search a card by name")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip("Type part of a card name")
        bar.addWidget(self.search, 1)
        self.group = QComboBox()
        self.group.addItem("All colours", "")
        present = {r["group"] for r in self.model.rows}
        for k, t in GROUP_TITLE.items():
            if k in present:
                self.group.addItem(t, k)
        self.group.setToolTip("Show one colour group only")
        bar.addWidget(self.group)
        self.rarity = QComboBox()
        self.rarity.addItem("All rarities", "")
        for k, t in RARITY_TITLE.items():
            if any(r["rarity"] == k for r in self.model.rows):
                self.rarity.addItem(t, k)
        self.rarity.setToolTip("Show one rarity only")
        bar.addWidget(self.rarity)
        v.addLayout(bar)
        self.count = muted(QLabel(""))
        self.count.setFont(sized_font(self.font(), -1))
        v.addWidget(self.count)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMouseTracking(True)
        self.table.viewport().installEventFilter(self)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        self.table.setItemDelegate(FullTextTipDelegate(self.table))
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setMinimumSectionSize(40)
        fm = hh.fontMetrics()
        for c in range(1, len(HEADERS)):                       # (room for the sort arrow next to the title)
            self.table.setColumnWidth(c, max(self.table.sizeHintForColumn(c), fm.horizontalAdvance(HEADERS[c]) + 44))
        self.table.setMinimumHeight(max(300, 12 * self.fontMetrics().height() + 20))
        self.table.setMaximumHeight(max(420, 18 * self.fontMetrics().height()))
        self.table.sortByColumn(4, Qt.SortOrder.DescendingOrder)
        v.addWidget(self.table)
        self.note = muted(QLabel("Click a column to sort.  * = limited data in this format: the value leans on the model's prior.  "
                                 "Hover or click a card to see it."))
        self.note.setWordWrap(True)
        self.note.setFont(sized_font(self.font(), -1))
        v.addWidget(self.note)
        self.search.textChanged.connect(self._filter)
        self.group.currentIndexChanged.connect(self._filter)
        self.rarity.currentIndexChanged.connect(self._filter)
        self.table.entered.connect(self._show)
        self.table.clicked.connect(self._show)
        self.table.selectionModel().currentRowChanged.connect(lambda cur, _prev: self._show(cur))
        self._filter()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.table.viewport() and event.type() == QEvent.Type.Leave:
            self.hidden.emit()
        return super().eventFilter(obj, event)

    def _filter(self) -> None:
        self.proxy.set_filters(self.search.text(), str(self.group.currentData() or ""), str(self.rarity.currentData() or ""))
        self.count.setText(f"{self.proxy.rowCount()} of {self.model.rowCount()} cards")

    def set_owned(self, owned: dict[str, int]) -> None:
        """Mark the cards of the Deck builder's pool with a star in the Card column."""
        self.model.set_owned(owned)

    def row_for(self, proxy_index: QModelIndex) -> dict | None:
        if not proxy_index.isValid():
            return None
        return self.model.rows[self.proxy.mapToSource(proxy_index).row()]

    def _show(self, idx: QModelIndex) -> None:
        r = self.row_for(idx)
        if r is None:
            return
        cap = (f"<b>{r['name']}</b><br>Played in {r['pr'] * 100:.0f}% of the decks that could play it; {pts(r['v'])} points per copy against the "
               f"average {GROUP_TITLE.get(r['group'], r['group']).lower()} card decks play ({BADGE_TEXT[r['badge']]})."
               + ("<br><i>Limited data in this format.</i>" if r["thin"] else ""))
        self.shown.emit(r["name"], cap)
