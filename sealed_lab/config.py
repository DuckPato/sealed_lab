"""App home, constants."""
from __future__ import annotations

import os
import re
from pathlib import Path


def _app_home() -> Path:
    env = os.environ.get("SEALED_LAB_HOME")
    if env:
        return Path(env)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "sealed_lab"
    return Path.home() / ".sealed_lab"


PROJECT_URL = "https://github.com/DuckPato/sealed_lab"


def user_agent() -> str:
    """What the app sends as User-Agent to Scryfall, 17Lands' file list and GitHub: name, version and where to find the project
    (Scryfall asks API users to identify their app)."""
    from sealed_lab import __version__

    return f"sealed_lab/{__version__} (+{PROJECT_URL})"


def app_home() -> Path:
    """Resolve the app home at call time (so tests can monkeypatch SEALED_LAB_HOME)."""
    return _app_home()


def __getattr__(name: str):
    # `config.APP_HOME` is resolved lazily so that tests (and the GUI) can change SEALED_LAB_HOME at runtime.
    if name == "APP_HOME":
        return _app_home()
    raise AttributeError(name)

FORMAT_EVENTS: dict[str, set[str]] = {
    "draft": {"PremierDraft", "TradDraft", "QuickDraft", "PickTwoDraft", "PickTwoTradDraft"},
    "bo1": {"Sealed"},
    "bo3": {"TradSealed"},
}

# The draft stage's standard data is PremierDraft + TradDraft; every other draft event (PickTwo, QuickDraft ...) is an EXTRA draft
# event, which the model keeps only if it passes its hold-out test (model/pipeline.py, component "extra_draft").
CORE_DRAFT_EVENTS: tuple[str, ...] = ("PremierDraft", "TradDraft")
EXTRA_DRAFT_EVENTS: tuple[str, ...] = ("PickTwoDraft", "PickTwoTradDraft", "QuickDraft")      # the ones the folder scan registers

RUN_RULES: dict[str, tuple[int, int]] = {"bo1": (7, 3), "bo3": (4, 2), "dbo1": (7, 3), "dbo3": (4, 2)}

# Draft TARGET formats of the model (docs/draft_mode/RULE.md; behind `pipeline.calculate(draft_formats=True)`): `dbo1` = the games of
# event type PremierDraft (Bo1, Arena's hand smoother), `dbo3` = TradDraft (Bo3, plain shuffle).  Other draft events are no target.
DRAFT_FORMAT_EVENT: dict[str, str] = {"dbo1": "PremierDraft", "dbo3": "TradDraft"}
DRAFT_FORMATS: tuple[str, ...] = ("dbo3", "dbo1")          # model order, as FMT_ORDER (Bo3 first)


def is_draft_format(fmt: str) -> bool:
    """True for a draft TARGET format of the model ("dbo1" / "dbo3"), False for the Sealed formats (and the draft prior stage)."""
    return fmt in DRAFT_FORMAT_EVENT


def is_bo3_format(fmt: str) -> bool:
    """Best-of-three queues: Traditional Sealed ("bo3") and Traditional Draft ("dbo3")."""
    return fmt in ("bo3", "dbo3")

BASICS: dict[str, str] = {
    "Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G",
    "Wastes": "",
    "Snow-Covered Plains": "W", "Snow-Covered Island": "U", "Snow-Covered Swamp": "B",
    "Snow-Covered Mountain": "R", "Snow-Covered Forest": "G", "Snow-Covered Wastes": "",
}

COLORS = "WUBRG"

GAME_META_COLUMNS = [
    "expansion", "event_type", "draft_id", "draft_time", "game_time", "build_index", "match_number",
    "game_number", "rank", "opp_rank", "main_colors", "splash_colors", "on_play", "num_mulligans",
    "opp_num_mulligans", "opp_colors", "num_turns", "won", "user_n_games_bucket",
    "user_game_win_rate_bucket",
]

FILE_NAME_RE = re.compile(
    r"(game|draft|replay)_data_public\.([A-Za-z0-9]+)\.([A-Za-z0-9]+)\.csv(\.gz)?$", re.IGNORECASE
)


def event_format(event: str | None) -> str | None:
    """Map a 17lands event type to 'draft' | 'bo1' | 'bo3'."""
    if not event:
        return None
    for fmt, events in FORMAT_EVENTS.items():
        if event in events:
            return fmt
    low = event.lower()
    if "sealed" in low and "trad" in low:
        return "bo3"
    if "sealed" in low:
        return "bo1"
    return "draft"


def is_extra_draft_event(event: str | None) -> bool:
    """True for a draft event outside the standard draft data (PremierDraft / TradDraft)."""
    return bool(event) and event_format(event) == "draft" and event not in CORE_DRAFT_EVENTS


# Hand/draw regimes of the mana simulator (see model/manasim.py): how Arena shuffles in each queue.
#   random : best-of-three ("Trad") queues - a plain random shuffle, the real keep rule only
#   hand   : best-of-one draft queues (Premier / Quick / Pick-Two) - the Arena hand smoother
#   sealed : the best-of-one Sealed queue - hand smoother plus land-balanced draws
REGIMES = ("random", "hand", "sealed")


def event_regime(event: str | None) -> str:
    """Hand/draw regime of a 17lands event type. Unknown Trad events are random, other unknown events smoothed Bo1."""
    if not event:
        return "random"
    low = event.lower()
    if "trad" in low:
        return "random"
    if "sealed" in low:
        return "sealed"
    return "hand"


def format_regime(fmt: str) -> str:
    """Regime of the model format: bo3 -> random, bo1 -> sealed, draft (dbo1: Premier) -> hand, dbo3 (Traditional draft) -> random."""
    return {"bo3": "random", "dbo3": "random", "bo1": "sealed"}.get(fmt, "hand")


def sets_dir() -> Path:
    return app_home() / "sets"


SET_CODE_RE = re.compile(r"[A-Z0-9]{2,8}")


def valid_set_code(code) -> str | None:
    """The upper-cased set code if it is 2-8 ASCII letters/digits (no separators, dots, colons), else None.

    Set codes come from file contents (an `expansion` column, Scryfall JSON) and become folder names, so they are validated."""
    if not isinstance(code, str):
        return None
    c = code.strip().upper()
    return c if SET_CODE_RE.fullmatch(c) and c.isascii() else None


def set_dir(set_code: str) -> Path:
    code = valid_set_code(set_code)
    if code is None:
        raise ValueError(f"invalid set code: {set_code!r}")
    return sets_dir() / code


def corpus_dir(set_code: str) -> Path:
    return set_dir(set_code) / "corpus"


def model_dir(set_code: str) -> Path:
    return set_dir(set_code) / "model"


def cache_dir(set_code: str) -> Path:
    return set_dir(set_code) / "cache"


def ensure_set_dirs(set_code: str) -> Path:
    d = set_dir(set_code)
    for sub in ("corpus", "model", "cache"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d
