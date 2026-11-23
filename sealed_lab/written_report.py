"""The written per-set report: stored text + meta in `<app home>/written_reports/<SET>.md|json`.

    install(set, report_path, facts_path)  -> meta   (runs the deterministic checker first; raises ReportRejected if it finds problems)
    load(set)                              -> {text, meta} | None
    available()                            -> sorted set codes that have a written report
    outdated(meta, current_stamps)         -> bool
    current_stamps(set)                    -> {fmt: stamp} of the cached analysis (the statistics the Analysis tab shows), if known

The meta file holds the stamps of the statistics the text was written from (`stamp`), `written_at` and the facts JSON itself, so the tab
can show the hover texts offline.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sealed_lab import config
from sealed_lab.report_check import check


class ReportRejected(ValueError):
    """The text failed the checker; `.problems` lists why."""

    def __init__(self, problems: list[str]):
        super().__init__(f"{len(problems)} problem(s):\n" + "\n".join("  " + p for p in problems))
        self.problems = problems


def folder() -> Path:
    return config.app_home() / "written_reports"


def install(set_code: str, report_path: str | Path, facts_path: str | Path) -> dict:
    set_code = set_code.upper()
    text = Path(report_path).read_text(encoding="utf-8")
    facts = json.loads(Path(facts_path).read_text(encoding="utf-8"))
    if facts.get("set") and facts["set"].upper() != set_code:
        raise ReportRejected([f"the facts file is for set {facts['set']}, not {set_code}"])
    problems = check(text, facts)
    if problems:
        raise ReportRejected(problems)
    meta = dict(set=set_code, stamp=facts.get("stamps") or {}, written_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), facts=facts)
    d = folder()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{set_code}.md").write_text(text, encoding="utf-8")
    (d / f"{set_code}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def load(set_code: str) -> dict | None:
    d = folder()
    try:
        text = (d / f"{set_code.upper()}.md").read_text(encoding="utf-8")
        meta = json.loads((d / f"{set_code.upper()}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(text=text, meta=meta)


def available() -> list[str]:
    d = folder()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.md") if (d / f"{p.stem}.json").exists())


def current_stamps(set_code: str, like: dict | None = None) -> dict[str, str]:
    """{fmt: stamp} of the cached analysis of the set (empty when none is stored).  For "ALL" the keys are "SET.fmt" for the sets named in
    `like` (the stamps the written text was made from)."""
    if set_code.upper() == "ALL":
        out: dict[str, str] = {}
        for key in like or {}:
            code, _, fmt = key.partition(".")
            if code and fmt:
                for f, st in current_stamps(code).items():
                    if f == fmt:
                        out[key] = st
        return out
    out = {}
    for fmt in ("bo1", "bo3"):
        try:
            d = json.loads((config.cache_dir(set_code.upper()) / f"report_{fmt}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("stamp"):
            out[fmt] = d["stamp"]
    return out


def outdated(meta: dict, current: dict[str, str]) -> bool:
    """True when a format the text was written from has a different stamp now (unknown current stamps do not count)."""
    return any(f in current and current[f] != s for f, s in (meta.get("stamp") or {}).items())
