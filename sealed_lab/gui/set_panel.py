"""Statistics panel of the Sets tab: what the selected set's data and model look like, from cheap reads only
(`api.set_overview`: registry, corpus manifest, model meta.json / report.json).  Also the panel of a set that is not tracked yet
(track a folder / download)."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import helptext as H
from .datasets_ui import human_size
from .widgets import Collapsible, FlowLayout, make_table, muted, semantic, sized_font

FORMAT_NAMES = {"draft": "Draft (Premier / Trad)", "bo1": "Bo1 Sealed", "bo3": "Bo3 Traditional Sealed",
                "dbo1": "Draft Bo1 (Premier Draft)", "dbo3": "Draft Bo3 (Traditional Draft)"}
FILE_COLS = ["File", "Kind", "Event", "Size", "Date", "State"]


def fmt_time(s: str | None) -> str:
    """ISO time (UTC or naive) -> local 'YYYY-MM-DD HH:MM'; the text itself when it cannot be read."""
    if not s:
        return "—"
    try:
        d = datetime.fromisoformat(str(s))
        if d.tzinfo is not None:
            d = d.astimezone()
        return d.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(s)


def fmt_duration(sec: float | None) -> str:
    if not sec:
        return "—"
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def _signed(x: float | None, digits: int = 4) -> str:
    return "—" if x is None else f"{x:+.{digits}f}"


def verdict(v: dict) -> tuple[str, str]:
    """(text, color role) of one format's hold-out result against the baseline that uses no deck information (B0)."""
    if v.get("status") != "ok":
        return f"hold-out not run ({v.get('reason') or v.get('status') or 'too little data'})", "warn"
    g, lo, hi = v.get("gain_vs_B0"), v.get("lo_vs_B0"), v.get("hi_vs_B0")
    if g is None or lo is None or hi is None:
        return "hold-out ran; no comparison with the no-deck-information baseline stored", "warn"
    ci = f"{_signed(g)} log loss per game (95% CI {_signed(lo)} to {_signed(hi)})"
    if lo > 0:
        return f"beats the no-deck-information baseline: {ci}", "good"
    if hi < 0:
        return f"WORSE than the no-deck-information baseline: {ci}", "bad"
    return f"cannot be told from the no-deck-information baseline: {ci}", "warn"


class _Box(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setProperty("card", True)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(10, 6, 10, 8)
        self.lay.setSpacing(4)
        self.title = QLabel(title)
        self.title.setFont(sized_font(self.font(), 1, bold=True))
        self.lay.addWidget(self.title)

    def add_label(self, wrap: bool = True) -> QLabel:
        lab = QLabel("")
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setWordWrap(wrap)
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lay.addWidget(lab)
        return lab


class SetPanel(QScrollArea):
    """Right-hand side of the Sets tab."""

    trackRequested = Signal(str)            # track <data folder>/<SET> (the folder that exists on disk)
    chooseFolderRequested = Signal(str)     # pick a folder for the set
    downloadRequested = Signal(str)
    rescanRequested = Signal()
    draftOptionChanged = Signal(str, bool)  # set, "Also build draft models" switched

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.code = ""
        body = QWidget()
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(6, 0, 6, 6)
        self.lay.setSpacing(8)
        self.setWidget(body)

        self.header = QLabel("")
        self.header.setTextFormat(Qt.TextFormat.RichText)
        self.header.setWordWrap(True)
        self.lay.addWidget(self.header)

        self.failed_box = _Box("The last calculation failed")
        self.failed_label = self.failed_box.add_label()
        self.trace = QPlainTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setMinimumHeight(120)
        self.trace.setMaximumHeight(220)
        self.trace.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.failed_box.lay.addWidget(self.trace)
        self.lay.addWidget(self.failed_box)

        self.pending_box = _Box("Recalculation pending")
        self.pending_label = self.pending_box.add_label()
        self.lay.addWidget(self.pending_box)

        self.warn_box = _Box("Warnings")
        self.warn_label = self.warn_box.add_label()
        self.lay.addWidget(self.warn_box)

        self.data_box = _Box("Data")
        self.data_label = self.data_box.add_label()
        self.lay.addWidget(self.data_box)

        self.model_box = _Box("Model")
        self.model_label = self.model_box.add_label()
        self.lay.addWidget(self.model_box)

        self.draft_box = _Box("Draft models")
        self.draft_cb = QCheckBox(H.DRAFT_OPTION_LABEL)
        self.draft_cb.setToolTip(H.DRAFT_OPTION_TIP)
        self.draft_box.lay.addWidget(self.draft_cb)
        self.draft_label = self.draft_box.add_label()
        self.lay.addWidget(self.draft_box)
        self._draft_opt: tuple[str, bool, bool] | None = None      # (set, option value, can be changed now): set by the Sets tab
        self.draft_cb.toggled.connect(lambda on: self.draftOptionChanged.emit(self.code, bool(on)))

        self.top_box = _Box("Best cards by value")
        self.top_label = self.top_box.add_label()
        self.lay.addWidget(self.top_box)

        self.files_table = make_table(FILE_COLS)
        hh = self.files_table.horizontalHeader()
        hh.setStretchLastSection(True)
        self.files_table.setMinimumHeight(140)
        self.files = Collapsible("Files", self.files_table, expanded=False)
        self.lay.addWidget(self.files)

        # not tracked yet
        self.untracked_box = _Box("Not tracked yet")
        self.untracked_label = self.untracked_box.add_label()
        row = FlowLayout()                       # wraps to a second row in a narrow panel
        self.use_btn = QPushButton("Track this folder")
        self.choose_btn = QPushButton("Track a folder for this set…")
        self.download_btn = QPushButton("Download…")
        for b in (self.use_btn, self.choose_btn, self.download_btn):
            row.addWidget(b)
        self.untracked_box.lay.addLayout(row)
        self.lay.addWidget(self.untracked_box)
        self.use_btn.clicked.connect(lambda: self.trackRequested.emit(self.code))
        self.choose_btn.clicked.connect(lambda: self.chooseFolderRequested.emit(self.code))
        self.download_btn.clicked.connect(lambda: self.downloadRequested.emit(self.code))
        self.lay.addStretch(1)
        self.clear()

    # ------------------------------------------------------------------
    def fit_width(self) -> None:
        """The panel is never narrower than its widest unwrappable content (the statistics tables) needs: it would otherwise cut them off."""
        body = self.widget()
        need = body.minimumSizeHint().width() + self.verticalScrollBar().sizeHint().width() + 2 * self.frameWidth() + 2
        self.setMinimumWidth(max(330, 46 * self.fontMetrics().horizontalAdvance("0"), need))

    def clear(self) -> None:
        self.code = ""
        self.header.setText("<span style='color:gray'>Select a set.</span>")
        for w in (self.pending_box, self.failed_box, self.warn_box, self.data_box, self.model_box, self.draft_box, self.top_box,
                  self.files, self.untracked_box):
            w.setVisible(False)

    def set_status_pill(self, text: str, role: str | None) -> None:
        """Update only the status next to the set code (progress ticks)."""
        if self.code:
            self.header.setText(f"<span style='font-size:20pt; font-weight:600'>{escape(self.code)}</span> &nbsp; " + self._pill(text, role))

    def _pill(self, text: str, role: str | None) -> str:
        col = semantic().get(role).name() if role else "gray"
        return f"<span style='color:{col}; font-weight:600'>{escape(text)}</span>"

    # ------------------------------------------------------------------ an untracked set
    def show_untracked(self, entry: dict, root: str, can_download: bool) -> None:
        self.clear()
        self.code = entry["set_code"]
        self.header.setText(f"<span style='font-size:20pt; font-weight:600'>{escape(self.code)}</span> &nbsp; "
                            + self._pill("Not tracked", None))
        bits = []
        folder = entry.get("folder") or ""
        if folder:
            bits.append(f"The folder <b>{escape(folder)}</b> holds {entry.get('n_local', 0)} 17lands file(s) for this set.")
        if entry.get("n_remote"):
            ev = ", ".join(entry.get("events") or [])
            bits.append(f"17lands lists {entry['n_remote']} file(s) for this set" + (f" ({escape(ev)})." if ev else "."))
        if not bits:
            bits.append("The app does not track this set.")
        bits.append(f"Tracked sets are read from <b>{escape(root)}</b>/&lt;SET&gt; (or from a folder you pick for the set).")
        self.untracked_label.setText("<br>".join(bits))
        self.use_btn.setVisible(bool(folder))
        self.use_btn.setText("Track this folder" if folder else "")
        self.use_btn.setToolTip(folder)
        self.download_btn.setVisible(can_download)
        self.untracked_box.setVisible(True)

    # ------------------------------------------------------------------ a tracked set
    def set_draft_option(self, code: str, value: bool | None, editable: bool = True) -> None:
        """The value of "Also build draft models" for `code` (None: the backend has no per-set options, the box stays hidden).  Call it
        before `show_set`."""
        self._draft_opt = None if value is None else (code, bool(value), bool(editable))

    def _fill_draft(self, st: Any) -> None:
        opt = self._draft_opt
        if opt is None or opt[0] != st.set_code:
            return
        _code, on, editable = opt
        self.draft_cb.blockSignals(True)
        self.draft_cb.setChecked(on)
        self.draft_cb.blockSignals(False)
        prov = bool(getattr(st, "provisional", False))
        self.draft_cb.setEnabled(editable and not prov)
        have = [f for f in (st.formats or []) if f in ("dbo1", "dbo3")]
        if prov:
            text = "Needs game data: this set has pick data only."
        elif have:
            text = ("Built: " + ", ".join(FORMAT_NAMES[f] for f in have) + ".  The Deck builder offers them next to Sealed.")
        elif on and st.has_model:
            text = "Not built yet: press Calculate."
        elif on:
            text = "They are built with the next calculation."
        else:
            text = "Off: only the Sealed models are built."
        self.draft_label.setText(escape(text) + "<br><span style='color:gray'>" + escape(H.DRAFT_FORMATS_NOTE) + "</span>")
        self.draft_box.setVisible(True)

    def show_set(self, st: Any, ov: dict | None, state: dict | None) -> None:
        self._show_set(st, ov, state)
        self.fit_width()

    def _show_set(self, st: Any, ov: dict | None, state: dict | None) -> None:
        """`st` SetStatus, `ov` api.set_overview (None while loading), `state` the GUI's state of the set: {key, text, role, progress}."""
        self.clear()
        self.code = st.set_code
        state = state or {}
        self.header.setText(f"<span style='font-size:20pt; font-weight:600'>{escape(st.set_code)}</span> &nbsp; "
                            + self._pill(state.get("text", ""), state.get("role")))
        # pending changes
        if st.stale and state.get("key") not in ("running",):
            self.pending_label.setText("<ul style='margin:0 0 0 -18px'>" + "".join(f"<li>{escape(x)}</li>" for x in (st.pending or [
                "the data changed since the model was built"])) + "</ul>Press <b>Calculate</b> to refit the model.")
            self.pending_box.setVisible(True)
        fail = state.get("failed")
        if fail:
            self.failed_label.setText(f"<span style='color:{semantic()['bad'].name()}'><b>{escape(fail[0])}</b></span><br>"
                                      "The full traceback is also in <b>logs/errors.log</b> of the app folder.")
            self.trace.setPlainText(fail[1])
            self.failed_box.setVisible(True)
        warns = list(st.warnings)
        if st.missing:
            warns.append(f"{len(st.missing)} registered file(s) are no longer on disk (a model built from them stays valid; "
                         "a new Calculate needs them): " + ", ".join(Path(p).name for p in st.missing[:3]))
        if warns:
            self.warn_label.setText("<ul style='margin:0 0 0 -18px'>" + "".join(f"<li>{escape(w)}</li>" for w in warns) + "</ul>")
            self.warn_box.setVisible(True)
        self._fill_draft(st)
        if ov is None:
            self.data_label.setText("<span style='color:gray'>Reading the set…</span>")
            self.data_box.setVisible(True)
            return
        self._fill_data(st, ov)
        self._fill_model(st, ov)
        self._fill_files(ov)

    def _fill_data(self, st: Any, ov: dict) -> None:
        games, events, rng = ov.get("games") or {}, ov.get("events") or {}, ov.get("date_range") or {}
        rows = []
        for f in ("draft", "bo1", "bo3"):
            if f in games or f in events:
                r = rng.get(f)
                ev = f"{int(events[f]):,}" if f in events else "—"
                dates = f"{escape(r[0])} – {escape(r[1])}" if r else "—"
                rows.append(f"<tr><td><b>{FORMAT_NAMES[f]}</b></td><td align='right'>{int(games.get(f, 0)):,}</td>"
                            f"<td align='right'>{ev}</td><td>&nbsp;&nbsp;{dates}</td></tr>")
        if ov.get("replay_games"):
            rows.append(f"<tr><td><b>Replay games</b> (turn by turn)</td><td align='right'>{int(ov['replay_games']):,}</td><td></td><td></td></tr>")
        if not rows:
            self.data_label.setText("<span style='color:gray'>No game data has been read yet: Calculate builds the data tables.</span>")
        else:
            head = ("<tr style='color:gray'><td>Format</td><td align='right'>Games</td><td align='right'>Events</td>"
                    "<td>&nbsp;&nbsp;Dates of the games</td></tr>")
            n = ov.get("n_cards")
            self.data_label.setText("<table cellspacing='4'>" + head + "".join(rows) + "</table>"
                                    + (f"<br>Cards in the set: <b>{int(n)}</b>" if n else ""))
        self.data_box.setVisible(True)

    def _fill_model(self, st: Any, ov: dict) -> None:
        m = ov.get("model")
        if not m:
            self.model_label.setText("<span style='color:gray'>No model yet: press Calculate.</span>")
            self.model_box.setVisible(True)
            return
        sem = {k: v.name() for k, v in semantic().items()}
        lines = []
        pv = m.get("provisional")
        if pv or getattr(st, "provisional", False):
            from sealed_lab import provisional as _pv

            n = f" ({int((pv or {}).get('n_runs') or 0):,} drafted decks, {int((pv or {}).get('n_games') or 0):,} games)" if pv else ""
            lines.append(f"<b>Provisional model</b>{escape(n)}: <span style='color:{sem['warn']}'>{escape(_pv.BANNER)}</span>")
        sig = ("matches the tracked files" if not st.stale else "differs from the tracked files (recalculation pending)")
        sigc = sem["good"] if not st.stale else sem["warn"]
        lines.append(f"Built <b>{escape(fmt_time(m.get('built_at')))}</b> · took <b>{fmt_duration(m.get('seconds'))}</b> · "
                     f"{m.get('n_boot') or '?'} bootstrap replicates{' · quick calculation' if m.get('quick') else ''} · data signature <span style='color:{sigc}'><b>{sig}</b></span>")
        comp = m.get("components") or []
        if comp:
            chips = []
            for c in comp:
                if c.get("applied") or c["key"] == "castability" and (m.get("castability") or {}).get("applied"):
                    pooled = [f for f, s_ in (c.get("source") or {}).items() if s_ == "pooled"]       # applied from the evidence pooled across sets
                    tag = f" (pooled: {', '.join(pooled)})" if pooled else ""
                    chips.append(f"<span style='color:{sem['good']}'>✔&nbsp;{escape(c['label'] + tag).replace(' ', '&nbsp;')}</span>")
                else:
                    chips.append(f"<span style='color:gray'>✘&nbsp;{escape(c['label']).replace(' ', '&nbsp;')}</span>")
            lines.append("<b>Optional components</b> (✔ = passed its hold-out test and is applied): " + " &nbsp; ".join(chips))
        for f in m.get("formats", []):
            v = (m.get("validation") or {}).get(f)
            if not v:
                continue
            text, role = verdict(v)
            extra = []
            if v.get("calibration_slope") is not None:
                extra.append(f"calibration slope {v['calibration_slope']:.2f}")
            if v.get("level_pp") is not None:
                extra.append(f"absolute level {v['level_pp']:+.1f} pp (observed − predicted on the newest games)")
            if v.get("gain_vs_B1") is not None and v.get("ci_vs_B1"):
                lo, hi = v["ci_vs_B1"]
                extra.append(f"vs the card-win-rate baseline {_signed(v['gain_vs_B1'])} [{_signed(lo)}, {_signed(hi)}]")
            lines.append(f"<b>{FORMAT_NAMES.get(f, f)} hold-out:</b> <span style='color:{sem[role]}'>{escape(text)}</span>"
                         + (f"<br><span style='color:gray'>{escape(' · '.join(extra))}</span>" if extra else ""))
        cast = m.get("castability")
        if cast:
            lines.append("<b>Castability:</b> " + ("<span style='color:%s'>applied</span>" % sem["good"] if cast.get("applied") else "not applied")
                         + (f" <span style='color:gray'>— {escape(cast.get('reason') or '')}</span>" if cast.get("reason") else ""))
        dep = m.get("deployed")
        if dep:
            if dep.get("components"):
                per = []
                for f, o in (dep.get("formats") or {}).items():
                    if o.get("status") == "ok":
                        per.append(f"{f} {_signed(o['gain'])} [{_signed(o['lo'])}, {_signed(o['hi'])}]")
                col = sem["bad"] if dep.get("fallback") else sem["good"]
                lines.append(f"<b>Deployed combination</b> ({', '.join(dep['components'])}): <span style='color:{col}'>"
                             f"{escape(dep.get('reason') or '')}</span>" + (f"<br><span style='color:gray'>gain over the plain chain, log loss per game: "
                                                                          f"{escape('; '.join(per))}</span>" if per else ""))
            else:
                lines.append("<b>Deployed combination:</b> none of the optional components was applied: the plain chain is deployed.")
        lam = []
        names = {"bo1": "Bo1", "bo3": "Bo3", "draft": "Draft"}
        for stage in ("bo1", "bo3"):
            info = (m.get("lam") or {}).get(stage)
            if info and info.get("source"):
                if info.get("lam") is not None:
                    se = f" ± {info['lam_se']:.2f}" if info.get("lam_se") is not None else ""
                    lam.append(f"{names[info['source']]} → {names[stage]}: λ = {info['lam']:.2f}{se}")
                else:
                    lam.append(f"{names[info['source']]} → {names[stage]}: λ not estimated (prior used as it is)")
        if lam:
            lines.append("<b>Transfer from the previous stage</b> (how much of the earlier estimate carries over): " + " · ".join(lam))
        if m.get("degraded"):
            lines.append(f"<span style='color:{sem['warn']}'><b>Degraded:</b> {escape('; '.join(m['degraded']))}</span>")
        self.model_label.setToolTip(H.tip("Optional components", *[f"<b>{escape(c['label'])}:</b> {'applied' if c.get('applied') else 'not applied'}"
                                                                 f"{' — ' + escape(c['reason']) if c.get('reason') else ''}" for c in comp]) if comp else "")
        self.model_label.setText("".join(f"<p style='margin:0 0 7px 0'>{ln}</p>" for ln in lines))
        self.model_box.setVisible(True)
        tops = m.get("top_cards") or {}
        if tops:
            cols = []
            for f in m.get("formats", []):
                rows = tops.get(f) or []
                if not rows:
                    continue
                items = "".join(f"<tr><td>{k}.</td><td>{escape(r['name'])}</td><td align='right'><b>{r['pp']:+.2f} pp</b></td>"
                                f"<td style='color:gray'>{'⚠ little data' if r.get('low_data') else ''}</td></tr>"
                                for k, r in enumerate(rows, 1))
                cols.append(f"<td valign='top'><b>{FORMAT_NAMES.get(f, f)}</b><table cellspacing='3'>{items}</table></td>")
            if cols:
                self.top_label.setText("<table><tr>" + "<td>&nbsp;&nbsp;&nbsp;&nbsp;</td>".join(cols) + "</tr></table>"
                                       "<span style='color:gray'>Win-rate change from adding one copy to an average deck (pp); "
                                       "cards with little data come last.</span>")
                self.top_box.setVisible(True)

    def _fill_files(self, ov: dict) -> None:
        files = ov.get("files") or []
        t = self.files_table
        t.setRowCount(len(files))
        sem = semantic()
        gone = sum(1 for f in files if f.get("missing"))
        self.files.set_title(f"Files ({len(files)})" + (f" — {gone} missing" if gone else ""))
        for r, f in enumerate(files):
            vals = [f["name"], f["kind"], f.get("event") or "", human_size(f.get("size")), f.get("date") or "",
                    "MISSING" if f.get("missing") else (f.get("note") or "")]
            for c, text in enumerate(vals):
                it = QTableWidgetItem(text)
                if c == 0:
                    it.setToolTip(f["path"])
                if c == 3:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if f.get("missing"):
                    it.setForeground(QBrush(sem["bad"]))
                t.setItem(r, c, it)
        t.resizeColumnToContents(0)
        t.setColumnWidth(0, min(max(t.columnWidth(0), 200), 420))
        self.files.setVisible(True)
