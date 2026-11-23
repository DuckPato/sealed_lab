"""Shared data contract between ingest, model, deck tools, API and GUI. Do not change field names."""
from __future__ import annotations

from dataclasses import dataclass, field


class Cancelled(Exception):
    """Raised by long-running work when its cancel event is set."""


class SetBusy(RuntimeError):
    """Raised by the model readers (build, evaluate, ...) of a set whose model is being recalculated right now."""


@dataclass
class FileInfo:
    path: str
    kind: str  # "game" | "draft" | "replay" | "cards" | "unknown"
    set_code: str | None
    event: str | None
    fmt: str | None  # "draft" | "bo1" | "bo3" | None
    size: int
    fingerprint: str
    note: str = ""


@dataclass
class SetStatus:
    set_code: str
    files: list[FileInfo]
    has_corpus: bool
    has_model: bool
    stale: bool
    formats: list[str]
    n_games: dict[str, int]
    built_at: str | None
    warnings: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)    # plain-language reasons a recalculation is pending ("2 new files: ..."); [] = up to date
    missing: list[str] = field(default_factory=list)    # registered files that are no longer on disk (paths)
    folder: str = ""                                    # the set's own tracked folder ("" = the sub-folder of the data folder)


@dataclass
class CardLine:
    card_id: int | None
    name: str
    count: int
    value_pp: float | None
    low_data: bool
    note: str = ""
    castable: float | None = None    # P(this card can be cast on time with this deck's mana), nonland cards; None = n/a


@dataclass
class DeckEval:
    cards: list[CardLine]
    colors: str
    splash: str
    game_wr: float
    game_wr_lo: float
    game_wr_hi: float
    match_wr: float | None
    score: float
    score_lo: float
    score_hi: float
    exp_wins: float
    p_max_wins: float
    breakdown: dict[str, float]
    mana: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    support: list[str] = field(default_factory=list)   # plain-language reasons the deck is outside what real decks
                                                       # look like (estimate less reliable); empty = within the data
    # builder.ARCH_WINDOW only: the creature range the deck was held to ({group, label, lo, hi, own}); None otherwise
    creature_window: dict | None = None


@dataclass
class BuildResult:
    set_code: str
    fmt: str
    decks: list[DeckEval]
    p_best: list[float]
    diffs: list[int]
    pool_unmatched: list[str]
    warnings: list[str] = field(default_factory=list)
    pool: dict[int, int] = field(default_factory=dict)   # card_id -> copies of the recognised non-basic pool cards
    # deck families (see deck/families.py); `decks` / `p_best` / `diffs` then describe the shown families' representatives.  One dict per
    # shown family, best first: rank, colors, splash, archetype, p_best, wr, gap_pp / gap_lo_pp / gap_hi_pp (pp behind #1, 80% interval
    # over the model draws), p_beats_top, n_members, why, diff_to_top, core [{card_id, name, count}], flex [{card_id, name, count,
    # core_count, freq, alts [{card_id, name, freq}]}], extra [{card_id, name, freq}], n_decks.  Empty without bootstrap draws.
    families: list[dict] = field(default_factory=list)
    family_meta: dict = field(default_factory=dict)      # n_draws (B), n_candidates, n_families, n_shown, n_hidden, thresholds, seconds
    # the user's own build when the pasted text was an Arena deck export with a "Deck" and a "Sideboard" section (None otherwise);
    # `user_unmatched` lists its card names that matched nothing
    user_deck: "DeckEval | None" = None
    user_unmatched: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    set_code: str
    fmt: str
    deck: DeckEval
    efficiency_pp: float | None
    best_alternative: DeckEval | None
    unmatched: list[str]
    warnings: list[str] = field(default_factory=list)
    pool: dict[int, int] = field(default_factory=dict)   # card_id -> copies of the recognised non-basic cards of deck + sideboard
