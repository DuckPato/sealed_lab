"""Facade used by the GUI and CLI. Heavy imports are lazy so importing this module is cheap."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sealed_lab import config
from sealed_lab.ingest import registry
from sealed_lab.ingest.detect import detect_file, expand_paths
from sealed_lab.types import BuildResult, Cancelled, DeckEval, FileInfo, ScoreResult, SetBusy, SetStatus

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]

# Sets whose model is being rewritten right now (calculate_set).  Every function that LOADS a set's model refuses with SetBusy
# meanwhile: the artifact is written file by file, so a reader could otherwise see half old, half new files.  Other sets are unaffected.
_CALC_LOCK = threading.Lock()
_CALCULATING: dict[str, int] = {}
_CACHE_LOCK = threading.RLock()        # guards the module-level caches below (card tables, name indexes) shared by GUI threads


@contextlib.contextmanager
def _calculating(set_code: str):
    sc = set_code.upper()
    with _CALC_LOCK:
        _CALCULATING[sc] = _CALCULATING.get(sc, 0) + 1
    try:
        yield
    finally:
        with _CALC_LOCK:
            _CALCULATING[sc] -= 1
            if _CALCULATING[sc] <= 0:
                del _CALCULATING[sc]


def calculating_sets() -> list[str]:
    """Sets being recalculated right now."""
    with _CALC_LOCK:
        return sorted(_CALCULATING)


def _calculating_elsewhere(set_code: str) -> int | None:
    """PID of ANOTHER live process calculating the set (sets/<SET>/calc.lock, e.g. the calculation child of this very app, a CLI run
    or a second app window), else None."""
    pid = registry.calc_lock_holder(set_code.upper())
    return pid if pid is not None and pid != os.getpid() else None


def _forget_card_caches(set_code: str) -> None:
    """Drop the cached card tables of the set (keyed by the table's path: the model's and the corpus's)."""
    sc = set_code.upper()
    with _CACHE_LOCK:
        for p in (config.model_dir(sc) / "cards.parquet", config.corpus_dir(sc) / "cards.parquet"):
            _CARD_INFO_CACHE.pop(str(p), None)
        _NAME_INDEX_CACHE.pop(sc, None)


def _guard(set_code: str) -> None:
    with _CALC_LOCK:
        busy = set_code.upper() in _CALCULATING
    if not busy:
        busy = _calculating_elsewhere(set_code) is not None
    if busy:
        raise SetBusy(f"{set_code.upper()} is being recalculated: its model is available again when the calculation finishes")


# ---------------------------------------------------------------------------------------------
# sets and files
# ---------------------------------------------------------------------------------------------

def add_files(paths: list[str]) -> list[FileInfo]:
    """Detect and register files (folders are scanned recursively). Returns one FileInfo per candidate file.

    The `note` of each returned FileInfo says what happened (e.g. "replaces older file", "ignored: ...").
    Set-less cards.csv / Scryfall files are shared by all sets.
    """
    out: list[FileInfo] = []
    for p in expand_paths(list(paths)):
        info = detect_file(p)
        res = registry.register(info)
        if res.note and res.note not in (info.note or ""):
            info.note = (info.note + "; " if info.note else "") + res.note
        if not res.registered and not res.note.startswith("already"):
            if "ignored" not in info.note:
                info.note = (info.note + "; " if info.note else "") + "not registered"
        out.append(info)
    return out


def remove_file(set_code: str, fingerprint: str) -> None:
    registry.remove(set_code.upper(), fingerprint)


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


MODEL_INPUT_KINDS = ("game", "replay", "cards", "scryfall")


def model_input_signature(files) -> list[str]:
    """['kind:fingerprint', ...] of the registered files a model is computed from (stored in the model's meta)."""
    return [f"{f.kind}:{f.fingerprint}" for f in files if f.kind in MODEL_INPUT_KINDS]


def _short_names(files, n: int = 2) -> str:
    names = [Path(f.path).name for f in files]
    return ", ".join(names[:n]) + (" …" if len(names) > n else "")


def set_status(set_code: str) -> SetStatus:
    """State of a set: registered files, corpus / model presence, `stale` (a recalculation is pending) and WHY (`pending`: plain-language
    reasons, e.g. "2 new files: game_data_public.HOB.TradSealed.csv.gz …"), files that vanished from disk (`missing`)."""
    from sealed_lab import bundled
    from sealed_lab.ingest import folders

    set_code = set_code.upper()
    files = registry.files(set_code)
    note = bundled.set_note(set_code)
    ready_made = bool(note) and not any(f.kind == "game" for f in files)     # a bundled set the user has not registered files for
    sd = config.set_dir(set_code)
    man = _read_json(config.corpus_dir(set_code) / "manifest.json")
    tables_ok = man is not None and all((config.corpus_dir(set_code) / f"{t}.parquet").exists()
                                        for t in ("cards", "games", "decks", "deck_cards", "builds", "hands"))
    meta = _read_json(config.model_dir(set_code) / "meta.json")
    has_model = meta is not None
    warnings: list[str] = []
    pending: list[str] = []
    cur = sorted(f.fingerprint for f in files if f.kind == "game")
    if tables_ok:
        used = sorted(f["fingerprint"] for f in man.get("files", []) if f.get("kind") == "game")
        stale = cur != used and not ready_made
        warnings += list(man.get("warnings", []))
    else:
        used = []
        stale = bool(cur)
    if has_model and tables_ok and not stale:
        built = meta.get("built_corpus_at") or meta.get("corpus_built_at")
        if built and built != man.get("built_at"):
            stale = True
            pending.append("the data tables were rebuilt after the model (an interrupted or later calculation): calculate again")
    if not has_model and cur:
        stale = True
        pending.append("data never used yet: no model has been calculated for this set")
    elif stale and not pending and tables_ok:
        new = [f for f in files if f.kind == "game" and f.fingerprint not in used]
        if new:
            pending.append(f"{len(new)} new or changed file{'s' if len(new) != 1 else ''}: {_short_names(new)}")
        if len(used) > len([f for f in files if f.kind == "game" and f.fingerprint in used]):
            pending.append("a game file the data tables were built from was replaced or removed")
    if has_model and meta.get("source_files") is not None and not ready_made:
        # everything the model is computed from: game data, replay data (keep rule, simulator calibration) and the card
        # metadata files; a change in any of them makes the model stale even if the corpus tables do not change
        now = sorted(model_input_signature(files))
        if now != sorted(meta["source_files"]):
            stale = True
            warnings.append("the files the model was computed from changed (game / replay data or card lists): Calculate again")
            old_sig = set(meta["source_files"])
            new = [f for f in files if f.kind in MODEL_INPUT_KINDS and f"{f.kind}:{f.fingerprint}" not in old_sig]
            gone = old_sig - set(now)
            if new and not pending:
                pending.append(f"{len(new)} new or changed file{'s' if len(new) != 1 else ''}: {_short_names(new)}")
            if gone and not new and not pending:
                pending.append(f"{len(gone)} file{'s' if len(gone) != 1 else ''} the model was computed from "
                               f"{'were' if len(gone) != 1 else 'was'} replaced or removed")
    if stale and not pending:
        pending.append("the data changed since the model was built")
    missing = folders.missing_files(set_code)
    for f in files:
        if f.kind in ("game", "cards", "scryfall") and not Path(f.path).exists():
            warnings.append(f"source file not found: {f.path}")
    for w in (meta or {}).get("degraded", []):
        warnings.append(f"Model: {w}")
    formats: list[str] = list(meta.get("formats", [])) if meta else []
    built_at = None
    if meta:
        built_at = meta.get("built_at")
        if not built_at:
            try:
                built_at = datetime.fromtimestamp((sd / "model" / "meta.json").stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            except OSError:
                built_at = None
    n_games = dict(man["audit"]["n_games_by_fmt"]) if tables_ok and man.get("audit") else {}
    return SetStatus(set_code=set_code, files=files, has_corpus=tables_ok, has_model=has_model, stale=stale,
                     formats=formats, n_games=n_games, built_at=built_at, warnings=warnings, pending=pending if stale else [],
                     missing=missing, folder=folders.set_folders().get(set_code, ""), note=note)


def list_sets() -> list[SetStatus]:
    return [set_status(s) for s in registry.known_sets()]


def set_release_dates(codes) -> dict[str, str]:
    """{set code: 'YYYY-MM-DD'} from Scryfall's set list (cached; may fetch it once per day; offline-safe)."""
    from sealed_lab.ingest import setinfo

    return setinfo.release_dates(codes)


# ---------------------------------------------------------------------------------------------
# calculation
# ---------------------------------------------------------------------------------------------

def calculate_set(set_code: str, progress: Progress | None = None, cancel: threading.Event | None = None,
                  n_jobs: int | None = None, n_boot: int = 16, fetch_scryfall: bool = True,
                  recency: str = "auto", attributes: str = "auto", crossset: str = "auto", synergy: str = "auto",
                  archcond: str = "auto", matchups: str = "pooled", manasim: str = "auto", castability: str = "auto",
                  sigma_cv: str = "time", stack: str = "auto", confirm: bool = True, recency_scope: str = "format",
                  leak_free: bool = False, extra_draft: str = "auto", theta_sigma: str = "rule",
                  playdraw: str = "auto", oppflags: str = "off", quick: bool = False) -> SetStatus:
    """Ingest (incremental) + model + validation report.

    `quick`: the quick calculation. Forces manasim="v1", recency="off", attributes="off", extra_draft="off" (whatever was passed):
    the four hold-out tests that never changed a deployed model in the maintainer's 11 sets, which cost about 40% of the time.
    Everything else keeps its value. Stored as meta["quick"]. Default False: every other caller keeps its behaviour.

    `extra_draft`: the draft games of the events other than PremierDraft / TradDraft (PickTwo, QuickDraft). "auto" keeps them only
    if the plain chain with them beats the one without them on the same time hold-out (the usual rule, decided before any other
    component); "on" keeps them untested, "off" drops them. Sets without such games are unaffected.

    `castability`: card copies weighted by the probability the deck's mana can cast them on time, plus an expected-dead-cards
    feature (Karsten-style, simulated). Unlike the other components "auto" ships it UNLESS a format is materially worse on the
    hold-out (a safety constraint against extrapolation); "on" forces it, "off" skips it.

    `manasim`: the calibrated mana simulator (v2: queue-specific hands and draws, real keep rule, card flow). "auto" deploys it
    only if the whole chain refitted on its features beats the v1 chain on the hold-out (95% CI above zero, at least the
    materiality floor in one format, no format materially worse); "v2" forces it, "v1" keeps the original simulator.

    `recency`: "auto" (weight recent events more only if that validates out of sample), "on" or "off"; `recency_scope` "format"
    chooses and tests the half-life per format (the draft stage is never weighted), "pooled" uses one half-life for all.
    `sigma_cv` "time" (forward-chaining folds over event start times) or "random" (original folds) chooses the Sealed stages'
    prior strength; `stack` "auto" also chooses the weight w of q = w M + (1 - w) P on the same folds ("off": deploy M).
    `theta_sigma` "rule" (default) sets the structure terms' prior SD from the card sigma; "cv" (expert) chooses it by the same CV.
    `confirm`: a component must pass on the earlier half of the hold-out events and be confirmed on the later half.
    `attributes`: card-attribute prior (card effects shrunk towards a function of mana value, rarity, types, keywords,
    text tags), same three modes. `crossset`: structure-term prior borrowed from the other sets' models, same modes.
    `synergy` (card-pair interactions), `archcond` (card values conditional on the deck's archetype) and `matchups`
    (card x opponent-colour interactions): each deployed in "auto" only if it improves held-out log loss with a 95% CI
    above zero (and is nowhere significantly worse). `matchups` also has "pooled" (the default): "auto", and in each format where
    "auto" does not apply the layer, the other sets' stored hold-out numbers pooled (random effects, at least 5 other sets,
    pre-registered rule) decide, unless this set's own numbers there are materially worse. The layer only feeds the matchup /
    sideboard advice, in the formats where it is applied."""
    from sealed_lab.ingest.corpus import build_corpus

    set_code = set_code.upper()
    if quick:
        from sealed_lab.model.pipeline import QUICK_OPTIONS
        manasim, recency, attributes, extra_draft = (QUICK_OPTIONS[k] for k in ("manasim", "recency", "attributes", "extra_draft"))
    with registry.calc_lock(set_code), _calculating(set_code):   # one calculation per set across processes; readers see it as busy
        last = [0.0]

        def mono(frac: float, msg: str) -> None:
            last[0] = max(last[0], frac)
            if progress:
                progress(last[0], msg)

        build_corpus(set_code, progress=lambda f, m: mono(0.35 * f, m), cancel=cancel, n_jobs=n_jobs,
                     fetch_scryfall=fetch_scryfall)
        try:
            from sealed_lab.model.pipeline import calculate
        except ImportError:
            st = set_status(set_code)
            st.warnings.append("model not implemented yet")
            mono(1.0, "Corpus built (model not implemented yet)")
            return st
        calculate(set_code, progress=mono, cancel=cancel, n_jobs=n_jobs, n_boot=n_boot, recency=recency, attributes=attributes,
                  crossset=crossset, synergy=synergy, archcond=archcond, matchups=matchups, manasim=manasim, castability=castability,
                  sigma_cv=sigma_cv, stack=stack, confirm=confirm, recency_scope=recency_scope, leak_free=leak_free,
                  extra_draft=extra_draft, theta_sigma=theta_sigma, playdraw=playdraw, oppflags=oppflags, quick=quick)
        from sealed_lab import bundled

        bundled.clear_marker(set_code)             # the model now comes from the user's own files: no longer a ready-made set
    mono(0.995, "Preparing the analysis")
    _precompute_analysis(set_code)
    mono(1.0, "Done")
    return set_status(set_code)


def _precompute_analysis(set_code: str) -> None:
    """Store the Analysis tab's statistics (both formats) right after a calculation, so the tab never computes them on open.
    A failure here never fails the calculation: the tab then computes them on first open, as before."""
    from sealed_lab import report

    for fmt in ("bo1", "bo3"):
        try:
            rep = report.get_report(set_code, fmt, refresh=True)
        except Exception:  # noqa: BLE001 - logged; the model itself is fine
            logging.getLogger(__name__).exception("could not precompute the %s analysis of %s", fmt, set_code)
            continue
        try:                                       # the style projector of the pairs' spectra (separate cache file, used by the Pool tab's link)
            from sealed_lab import archetype_link
            from sealed_lab.deck import builder

            archetype_link.forget(set_code)
            archetype_link.proj_path(set_code, fmt).unlink(missing_ok=True)
            archetype_link.get_projectors(set_code, fmt, builder.load_model(set_code.upper()), rep)
        except Exception:  # noqa: BLE001 - fitted on first request instead
            logging.getLogger(__name__).info("could not precompute the %s archetype projector of %s", fmt, set_code, exc_info=True)


class CalculationError(RuntimeError):
    """A calculation that failed in its child process; the message ends with the child's own error line."""


def _calc_child_cmd(set_code: str, options: dict, cancel_file: str) -> list[str]:
    import os
    import sys

    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        exe = exe.parent / "python.exe"                    # pipes need a console-type interpreter; no window is opened
    arg = json.dumps({"set_code": set_code, "options": options, "cancel_file": cancel_file, "parent_pid": os.getpid()})
    return [str(exe), "-X", "faulthandler", "-m", "sealed_lab.calcproc", arg]


def _run_calc_child(cmd: list[str], progress: Progress | None, cancel: threading.Event | None, cancel_file: str) -> tuple[int, str]:
    """Run one calculation child: forward its progress and log lines, pass a cancel on (by creating `cancel_file`).
    Returns (exit code, last stderr lines)."""
    import collections
    import os
    import subprocess
    import sys

    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         errors="replace", bufsize=1, creationflags=flags, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    tail: collections.deque[str] = collections.deque(maxlen=300)
    child_log = logging.getLogger("sealed_lab.calc")
    done = threading.Event()

    def read_stderr() -> None:
        for line in p.stderr:
            line = line.rstrip("\n")
            tail.append(line)
            child_log.info("%s", line)

    def pass_cancel() -> None:
        while not done.wait(0.5):
            if cancel is not None and cancel.is_set():
                try:
                    Path(cancel_file).touch()
                except OSError:
                    pass
                try:
                    p.wait(timeout=120)                    # stops at its next checkpoint
                except subprocess.TimeoutExpired:
                    p.kill()
                return

    err_t = threading.Thread(target=read_stderr, daemon=True)
    err_t.start()
    threading.Thread(target=pass_cancel, daemon=True).start()
    try:
        for line in p.stdout:
            if line.startswith("P ") and progress is not None:
                frac, _, msg = line[2:].rstrip("\n").partition("\t")
                try:
                    progress(float(frac), msg)
                except ValueError:
                    pass
        rc = p.wait()
    finally:
        done.set()
        if p.poll() is None:
            p.kill()
    err_t.join(timeout=10)
    return rc, "\n".join(tail)


def calculate_set_isolated(set_code: str, progress: Progress | None = None, cancel: threading.Event | None = None,
                           retries: int = 1, **options) -> SetStatus:
    """`calculate_set` in a child process (the GUI's way to calculate). A native crash in a compiled library (access violations
    inside polars were seen in the long-lived GUI process, never in one-calculation processes) then ends only the child, not
    the app: it is retried `retries` times (the ingested corpus and the validation cache make a retry cheaper) and otherwise
    reported with the child's last output. Results are those of `calculate_set`."""
    import os
    import tempfile
    import uuid

    from sealed_lab import calcproc

    set_code = set_code.upper()
    other = _calculating_elsewhere(set_code)      # the child takes the lock itself (taking it here would lock the child out)
    if other is not None:
        raise SetBusy(f"{set_code} is already being calculated (process {other}): wait for it to finish")
    cancel_file = os.path.join(tempfile.gettempdir(), f"sealed_lab_cancel_{os.getpid()}_{uuid.uuid4().hex}")
    with _calculating(set_code):       # readers in this process see the set as busy while the child rewrites it
        for attempt in range(retries + 1):
            try:
                rc, tail = _run_calc_child(_calc_child_cmd(set_code, options, cancel_file), progress, cancel, cancel_file)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(cancel_file)
            if rc == 0:
                _JSON_CACHE.clear()
                _RANGE_CACHE.clear()
                _forget_card_caches(set_code)
                return set_status(set_code)
            if rc == calcproc.EXIT_BUSY:
                raise SetBusy(tail.strip().splitlines()[-1] if tail.strip() else f"{set_code} is already being calculated")
            if rc == calcproc.EXIT_CANCELLED or (cancel is not None and cancel.is_set()):
                raise Cancelled()
            if rc == calcproc.EXIT_ERROR:
                raise CalculationError(tail or f"{set_code}: the calculation failed")
            what = f"{set_code}: the calculation process crashed (exit code {rc & 0xFFFFFFFF:#x}, a native error in a compiled library)"
            if attempt < retries:
                log.warning("%s; retrying\n%s", what, tail[-4000:])
                if progress:
                    progress(0.0, "The calculation process crashed; retrying")
                continue
            raise CalculationError(f"{tail[-8000:]}\n{what}, {retries + 1} times")
    raise AssertionError("unreachable")


_CARD_INFO_CACHE: dict[str, tuple[int, dict[int, dict]]] = {}


def card_info(set_code: str) -> dict[int, dict]:
    """{card_id: {name, mana_value, is_creature, is_land, colors, rarity, types, cost, is_basic, produces}} for every card
    of the set (basics included).

    Read from the MODEL's own card table when a model exists (its dense ids are the ones the model, the builder and the pool
    parser use; the corpus table can differ after a cancelled or failed Calculate rebuilt the corpus), else from the corpus card
    table (works before a model exists); cached until the table changes."""
    import polars as pl

    mp = config.model_dir(set_code.upper()) / "cards.parquet"
    p = mp if (mp.exists() and (config.model_dir(set_code.upper()) / "meta.json").exists()) else         config.corpus_dir(set_code.upper()) / "cards.parquet"
    if not p.exists():
        return {}
    key = str(p)
    stamp = p.stat().st_mtime_ns
    with _CACHE_LOCK:
        hit = _CARD_INFO_CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        df = pl.read_parquet(p)
    except Exception:  # noqa: BLE001 - a damaged model table: fall back to the corpus
        log.warning("card table %s unreadable; using the corpus card table", p, exc_info=True)
        p = config.corpus_dir(set_code.upper()) / "cards.parquet"
        if not p.exists():
            return {}
        df = pl.read_parquet(p)
        key, stamp = str(p), p.stat().st_mtime_ns

    def cost(r: dict) -> str:
        pips = "".join(c * int(r.get(f"pip_{c}") or 0) for c in "WUBRG")
        gen = int(r.get("generic") or 0)
        return (str(gen) if gen or not pips else "") + pips if not r.get("is_land") else ""

    info = {int(r["card_id"]): dict(name=r["name"], mana_value=float(r["mana_value"] or 0.0), is_creature=bool(r["is_creature"]),
                                    is_land=bool(r["is_land"]), colors=r["colors"] or "", rarity=r.get("rarity") or "",
                                    types=r.get("types") or "", cost=cost(r), is_basic=bool(r.get("is_basic")),
                                    produces=r.get("produces") or "")
            for r in df.iter_rows(named=True)}
    with _CACHE_LOCK:
        _CARD_INFO_CACHE[key] = (stamp, info)
    return info


# ---------------------------------------------------------------------------------------------
# card deck search (read-only over the corpus; sealed_lab/deck_search.py)
# ---------------------------------------------------------------------------------------------

def deck_search_cards(set_code: str) -> list[str]:
    """The set's non-basic card names in its corpus, sorted (the search box's autocomplete)."""
    from sealed_lab import deck_search

    _guard(set_code)
    return deck_search.card_names(set_code)


def deck_search_events(set_code: str) -> list[dict]:
    """[{event, label, fmt, bo3, max_wins, n_runs, n_games}]: the 17lands events in the set's corpus, Premier Draft first."""
    from sealed_lab import deck_search

    _guard(set_code)
    return deck_search.events(set_code)


def search_decks(set_code: str, card: str | int, events: list[str] | None = None, sort: str = "result", min_games: int = 3,
                 min_card_share: float = 0.5, limit: int = 200, offset: int = 0) -> dict:
    """Real decks (17lands runs) of the set that played `card`, filtered by event and sorted ("result" | "win_rate" | "newest" |
    "oldest"); see `deck_search.search` for the result's fields.  Raises ValueError for an unknown card name."""
    from sealed_lab import deck_search

    _guard(set_code)
    return deck_search.search(set_code, card, events=events, sort=sort, min_games=min_games, min_card_share=min_card_share,
                              limit=limit, offset=offset)


def deck_search_text(set_code: str, result: dict) -> str:
    """Arena import text of one `search_decks` result (main deck + the sideboard when stored)."""
    from sealed_lab import deck_search

    return deck_search.deck_text(result, set_code.upper())


def report_path(set_code: str) -> Path | None:
    p = config.model_dir(set_code.upper()) / "report.html"
    return p if p.exists() else None


# ---------------------------------------------------------------------------------------------
# deck tools (delegated; implemented in sealed_lab.deck)
# ---------------------------------------------------------------------------------------------

def build_pool(set_code: str, text: str, fmt: str = "bo3", n_decks: int = 3, min_diff: int = 10,
               progress: Progress | None = None, cancel: threading.Event | None = None,
               locked: dict | list | set | None = None, excluded: dict | list | set | None = None,
               name_overrides: dict[str, str] | None = None) -> BuildResult:
    """Deck FAMILIES from an Arena pool export (see deck/families.py): `n_decks` is the maximum number of families returned (the GUI
    uses 5); `BuildResult.decks` are their representatives, best first, `p_best` their P(best) over the model's bootstrap draws,
    `families` the details (core / flex cards, gap to #1 with interval).  With `n_decks=1`, or a model without draws, only the single
    best deck is searched (`min_diff` then only applies to the old "decks >= min_diff cards apart" fallback).
    `locked` ({name or card_id: copies}, or a list = 1 copy each) must be in every returned deck, `excluded` (names or card_ids) in none.
    `name_overrides` ({pasted name: card name the user confirmed, "" = leave the card out}) corrects misspelt names (see `check_pool`);
    when the text is an Arena deck export with Deck and Sideboard sections the Deck is evaluated too (`BuildResult.user_deck`).
    Raises `SetBusy` while the set is being recalculated."""
    _guard(set_code)
    try:
        from sealed_lab.deck import builder
    except ImportError as e:
        raise NotImplementedError("deck builder is not implemented yet") from e
    return builder.build_pool(set_code.upper(), text, fmt, n_decks=n_decks, min_diff=min_diff, progress=progress,
                              cancel=cancel, locked=locked, excluded=excluded, name_overrides=name_overrides)


def score_deck(set_code: str, text: str, fmt: str = "bo3", name_overrides: dict[str, str] | None = None,
               search: bool = True) -> ScoreResult:
    """Score a pasted deck (Deck section of the Arena export): percentile among real builds, win rates, expected wins.

    `search=False` is the fast path: the sideboard is not searched for a better build (no efficiency / alternative), but
    `ScoreResult.pool` still holds deck + sideboard for the visual builder.  `name_overrides` as in `build_pool`."""
    _guard(set_code)
    try:
        from sealed_lab.deck import score
    except ImportError as e:
        raise NotImplementedError("deck scorer is not implemented yet") from e
    return score.score_deck(set_code.upper(), text, fmt, name_overrides=name_overrides, search=search)


def export_arena(deck: DeckEval, sideboard: dict[str, int] | None = None, set_code: str | None = None) -> str:
    try:
        from sealed_lab.deck import arena
    except ImportError as e:
        raise NotImplementedError("Arena export is not implemented yet") from e
    text = arena.export_arena(deck, sideboard, set_code)
    return _real_names(text, set_code) if set_code else text


def _real_names(text: str, set_code: str) -> str:
    """Arena spelling for names 17lands wrote with '?' for a lost letter ("Bespoke B?" -> "Bespoke Bō", from `front_name`)."""
    if "?" not in text:
        return text
    try:
        from sealed_lab.deck import builder

        cards = builder.load_model(set_code.upper()).cards
        fix = {n.split(" // ")[0].strip(): f for n, f in zip(cards["name"].to_list(), cards["front_name"].to_list())
               if "?" in n and f and "?" not in f}
    except Exception:  # noqa: BLE001 - the export still works with the 17lands spelling
        return text
    out = []
    for line in text.split("\n"):
        head, _, nm = line.partition(" ")
        out.append(f"{head} {fix[nm]}" if head.isdigit() and nm in fix else line)
    return "\n".join(out)


def evaluate_deck(set_code: str, deck: dict[int, int], fmt: str = "bo3") -> DeckEval:
    """Evaluate a deck given as {card_id: copies} (basics included). Fast (~tens of ms): used for live editing."""
    from sealed_lab.deck import builder

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    if fmt not in model.formats:
        raise ValueError(f"format {fmt!r} is not available for {model.set_code}")
    deck = {int(k): int(v) for k, v in deck.items() if v > 0}
    if not deck:
        raise ValueError("the deck is empty")
    return model.evaluate(deck, fmt)


def card_values(set_code: str, fmt: str = "bo3", deck: dict[int, int] | None = None, cids: list[int] | None = None) -> dict[int, dict]:
    """{card_id: {beta, low_data}} for every non-basic card: beta is the per-copy log-odds effect (draw 0).

    With `deck` ({card_id: copies}) every entry (or only `cids`) also carries what ONE ADDED copy would contribute to THIS deck under
    the deployed model: `value_pp` (change of the game win rate), `castable` (its castability in this deck, None for lands or a
    model without the castability layer) and `contrib` (logit: beta x castability + the dead-cards term). It runs the castability
    engine once per card (a few ms each): call it from a background thread."""
    import numpy as np

    from sealed_lab.deck import builder

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    if fmt not in model.formats:
        return {}
    beta = model.card_beta(fmt)
    c = model.cards.sort("card_id")
    basic = c["is_basic"].to_numpy()
    expo = c[f"exposure_{fmt}"].to_numpy() if f"exposure_{fmt}" in c.columns else np.zeros(len(beta))
    ids = range(len(beta)) if cids is None else [int(i) for i in cids if 0 <= int(i) < len(beta)]
    out = {i: dict(beta=float(beta[i]), low_data=bool(expo[i] < 2000)) for i in ids if not basic[i]}
    if deck:
        for i, v in model.add_values({int(k): int(n) for k, n in deck.items() if n > 0}, fmt, list(out)).items():
            out[i].update(value_pp=v["value_pp"], castable=v["castable"], contrib=v["contrib"])
    return out


def set_report(set_code: str, fmt: str = "bo3", refresh: bool = False, progress: Progress | None = None,
               cancel: threading.Event | None = None) -> dict:
    """The "Set report" of a set and Sealed format as a JSON-serialisable dict (see `sealed_lab.report`): game length, colour pairs
    and splash (skill-adjusted), cards players misjudge.  Computed from the corpus and the model (read-only), cached in the set's cache
    folder under the model's built-at stamp (`refresh=True` recomputes).  Raises `SetBusy` while the set is being recalculated."""
    from sealed_lab import report
    from sealed_lab.deck import builder

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    return report.get_report(set_code.upper(), fmt, model=model, refresh=refresh, progress=progress, cancel=cancel)


def deck_archetype(set_code: str, fmt: str, deck: dict[int, int]) -> dict:
    """Where a deck sits in the Analysis tab's archetypes (observed in 17lands data; the builder does not use archetypes):
    {available, key, code, label, effect_pp, badge, range, spectrum{shown, end ('a'|'b'|'mid'), a_label, b_label, label, effect_pp, badge,
    states_effect}, text}; {available: False, reason} when there is nothing to say.  Never raises for missing data; reads the STORED report only
    (never computes it) and fits / stores the style projector on first use."""
    try:
        from sealed_lab import archetype_link, report
        from sealed_lab.deck import builder
        from sealed_lab.model import structure

        _guard(set_code)
        sc, deck = set_code.upper(), {int(k): int(v) for k, v in deck.items() if v > 0}
        model = builder.load_model(sc)
        if fmt not in model.formats:
            return dict(available=False, reason=f"No {fmt} model for {sc}.")
        rep = report.read_cached(sc, fmt, report.model_stamp(model))
        if rep is None or rep.get("too_few"):
            return dict(available=False, reason="The analysis of this set and format has not been computed yet.")
        main = structure.deck_colors(deck, model.card_arrays)[0]
        proj = archetype_link.get_projectors(sc, fmt, model, rep)
        return archetype_link.describe(rep, proj, deck, archetype_link._card_vectors(model), main)
    except SetBusy:
        return dict(available=False, reason="The set is being recalculated.")
    except Exception as exc:  # noqa: BLE001 - optional line; logged
        logging.getLogger(__name__).info("deck_archetype failed", exc_info=True)
        return dict(available=False, reason=f"Not available ({type(exc).__name__}).")


def cross_set(sets: list[str] | None = None, refresh: bool = False, progress: Progress | None = None) -> dict:
    """The cross-set lessons (Analysis tab, "All sets"): {facts, lessons, skipped}; see `sealed_lab.cross_set`.  Built from the per-set facts of the sets with a
    model (their STORED analyses only: nothing is calculated).  A set that is being recalculated or has no stored analysis is left out
    and listed in `skipped` ([{set, reason}]).  `refresh` is accepted for compatibility and ignored (it would compute reports)."""
    from sealed_lab import cross_set as X

    return X.build_cross(sets, refresh=False, progress=progress, cached_only=True, busy=lambda s: s in calculating_sets() or _calculating_elsewhere(s) is not None)


def written_report(set_code: str | None = None) -> dict:
    """The written report of a set (Report tab): {text, meta, outdated, available}.  `text`/`meta` are None when the set has none;
    `outdated`: the statistics it was written from differ from the cached analysis now; `available`: sets that have a written report."""
    from sealed_lab import written_report as W

    out: dict = dict(text=None, meta=None, outdated=False, available=W.available())
    if set_code:
        hit = W.load(set_code)
        if hit:
            out.update(text=hit["text"], meta=hit["meta"],
                       outdated=W.outdated(hit["meta"], W.current_stamps(set_code, like=hit["meta"].get("stamp"))))
    return out


def export_set_html(set_code: str) -> str:
    """One self-contained HTML page with the written report and the STORED Analysis (Bo1 and Bo3) of a set (see `sealed_lab.export_html`).
    Never computes anything: a format whose analysis is not stored is listed as "not computed"."""
    from sealed_lab import export_html, report

    sc = set_code.upper()
    st = set_status(sc)
    reports: dict = {}
    for fmt in st.formats or []:
        reports[fmt] = report.read_cached(sc, fmt, f"{st.built_at or ''}|v{report.REPORT_VERSION}")
    written = written_report(sc)
    return export_html.export_set_html(sc, written, reports, ((written.get("meta") or {}).get("facts")))


def cheat_sheet(set_code: str, fmt: str = "bo3") -> str:
    """The set's one-page printable cheat sheet (HTML, see `sealed_lab.cheatsheet`) for a Sealed format: from the set report (its cache when
    it is there), the card table and the card values of the deployed model.  Raises `SetBusy` while the set is being recalculated and
    ValueError when the format has no report."""
    from sealed_lab import cheatsheet

    sc = set_code.upper()
    rep = set_report(sc, fmt)
    if rep.get("too_few"):
        raise ValueError(f"not enough {fmt} games of {sc} for a cheat sheet")
    return cheatsheet.cheat_sheet_html(sc, fmt, rep, card_info(sc), card_values(sc, fmt))


def card_image(set_code: str, name: str, fetch: bool = True) -> str | None:
    """Local path of the card's image (downloaded from Scryfall and cached on first use), or None."""
    from sealed_lab.ingest.images import fetch_image

    p = fetch_image(set_code.upper(), name, fetch=fetch)
    return str(p) if p is not None else None


def prefetch_images(set_code: str, names: list[str]) -> int:
    """Resolve the image URLs of many cards at once (2 API calls per 75 cards) so later card_image calls only hit
    Scryfall's image CDN. Returns how many URLs were added. Never raises."""
    from sealed_lab.ingest.images import resolve_urls

    return resolve_urls(set_code.upper(), list(names))


def build_from_pool(set_code: str, pool: dict[int, int], fmt: str = "bo3", n_decks: int = 1, min_diff: int = 10,
                    progress: Progress | None = None, cancel: threading.Event | None = None,
                    locked: dict | list | set | None = None, excluded: dict | list | set | None = None) -> BuildResult:
    """`build_pool` for a pool already given as {card_id: copies} (the visual builder's "Rebuild around locks").

    Always the same search as the Pool tab (deck families over the bootstrap draws), also for n_decks=1: the classic
    single search is on average 0.5-0.8 pp worse, so a rebuild without locks could otherwise return a weaker deck than
    the one it started from."""
    from sealed_lab.deck import builder

    _guard(set_code)
    return builder.build_ids(set_code.upper(), {int(k): int(v) for k, v in pool.items()}, fmt, n_decks=n_decks,
                             min_diff=min_diff, progress=progress, cancel=cancel, locked=locked, excluded=excluded,
                             families=True)


def explain_deck(set_code: str, deck: dict[int, int], fmt: str = "bo3"):
    """`DeckExplanation`: waterfall from the average real build of the format to this deck (see deck/explain.py)."""
    from sealed_lab.deck import builder, explain

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    if fmt not in model.formats:
        raise ValueError(f"format {fmt!r} is not available for {model.set_code}")
    return explain.explain_deck(model, {int(k): int(v) for k, v in deck.items() if v > 0}, fmt)


def swap_analysis(set_code: str, deck: dict[int, int], pool: dict[int, int], fmt: str = "bo3",
                  locked: dict[int, int] | None = None, excluded: set[int] | None = None,
                  cancel: threading.Event | None = None, on_partial: Callable | None = None):
    """`SwapAnalysis`: best replacement of every deck card from the sideboard, best slot of every sideboard card,
    value above replacement and the top suggested swaps (progressive through `on_partial`; raises Cancelled)."""
    from sealed_lab.deck import builder, explain

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    if fmt not in model.formats:
        raise ValueError(f"format {fmt!r} is not available for {model.set_code}")
    return explain.swap_analysis(model, {int(k): int(v) for k, v in deck.items() if v > 0},
                                 {int(k): int(v) for k, v in pool.items() if v > 0}, fmt, locked=locked,
                                 excluded=excluded, cancel=cancel, on_partial=on_partial)


def model_components(set_code: str) -> dict:
    """What the optional model layers did in the last calculation: {name: {applied, mode, reason, gain: {fmt: {gain, lo, hi}}}}
    for 'synergy', 'archcond', 'matchups', 'attributes', 'crossset' and 'recency' (empty dict without a model)."""
    meta = _read_json(config.model_dir(set_code.upper()) / "meta.json") or {}
    return {k: meta[k] for k in ("synergy", "archcond", "matchups", "attributes", "crossset", "recency", "archetypes", "manasim", "castability", "extra_draft", "playdraw", "oppflags") if k in meta}


def calibration_note(set_code: str, fmt: str) -> dict | None:
    """Absolute-level error of the model's win rates on the newest events (the hold-out of the last Calculate):
    {level_pp (observed minus predicted game win rate, pp), lo, hi (95% CI), logit, n_games, holdout_start, significant,
    base_rate_train, base_rate_holdout}, or None when the format had no hold-out.  Positive = the model UNDER-predicted the level.
    The level matters for absolute numbers (game / match win rate, expected wins); rankings and scores do not depend on it."""
    meta = _read_json(config.model_dir(set_code.upper()) / "meta.json") or {}
    v = (meta.get("validation") or {}).get(fmt) or {}
    cal = v.get("calibration") or {}
    if v.get("status") != "ok" or cal.get("level_pp") is None:
        return None
    lo, hi = cal.get("level_pp_ci") or (None, None)
    return dict(level_pp=float(cal["level_pp"]), lo=lo, hi=hi, logit=cal.get("level_logit"), n_games=cal.get("n"),
                holdout_start=v.get("holdout_start"), significant=bool(lo is not None and (lo > 0 or hi < 0)),
                base_rate_train=v.get("base_rate_train"), base_rate_holdout=v.get("base_rate_holdout"))


def synergies(set_code: str, deck: dict[int, int], pool: dict[int, int] | None = None, fmt: str = "bo3"):
    """`insight.SynergyView`: reliable card-pair synergies inside the deck and sideboard cards that would add one."""
    from sealed_lab.deck import builder, insight

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    return insight.synergy_view(model, {int(k): int(v) for k, v in deck.items() if v > 0},
                                {int(k): int(v) for k, v in (pool or {}).items() if v > 0}, fmt)


def archetype_of(set_code: str, deck: dict[int, int], fmt: str = "bo3"):
    """`insight.ArchetypeMatch`: nearest archetype of a deck, similarity, its observed win rate (shrunk, with CI) and the
    signature cards the deck lacks."""
    from sealed_lab.deck import builder, insight

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    return insight.archetype_of(model, {int(k): int(v) for k, v in deck.items() if v > 0}, fmt)


def sideboard_plan(set_code: str, deck: dict[int, int], pool: dict[int, int], fmt: str = "bo3",
                   locked: dict[int, int] | None = None, excluded: set[int] | None = None,
                   cancel: threading.Event | None = None):
    """`matchups.SideboardPlan`: per opponent colour pair the reliable in/out swaps for this build, and the sideboard cards to
    mark. status 'not_supported' (with a plain-language message) when the matchup layer did not pass its hold-out test."""
    from sealed_lab.deck import builder, matchups

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    return matchups.plan_for(model, {int(k): int(v) for k, v in deck.items() if v > 0},
                             {int(k): int(v) for k, v in pool.items() if v > 0}, fmt, locked=locked, excluded=excluded, cancel=cancel)


def playdraw_plan(set_code: str, deck: dict[int, int], pool: dict[int, int], fmt: str = "bo3", on_play: bool | None = None,
                  locked: dict[int, int] | None = None, excluded: set[int] | None = None, cancel: threading.Event | None = None):
    """`playdraw.PlayDrawPlan`: Bo3 games-2-3 swaps X -> Y that reliably help on the play / on the draw for this build (>= 0.5 pp and
    P(gain > 0) >= 0.90 over the bootstrap draws). status 'not_supported' when the play/draw layer did not pass its hold-out test."""
    from sealed_lab.deck import builder, playdraw

    _guard(set_code)
    model = builder.load_model(set_code.upper())
    return playdraw.playdraw_plan(model, {int(k): int(v) for k, v in deck.items() if v > 0},
                                  {int(k): int(v) for k, v in pool.items() if v > 0}, fmt, on_play=on_play, locked=locked,
                                  excluded=excluded, cancel=cancel)


def import_arena_log(set_code: str | None = None, log_dir: str | None = None, course_key: tuple | None = None):
    """`ArenaPool` of the most recent Sealed event in Arena's Player.log (Arena-export text in `.text`); with `course_key` (the
    watcher's offer) exactly that event.

    Raises `sealed_lab.deck.arenalog.ArenaLogError` subclasses with user-facing messages."""
    from sealed_lab.deck import arenalog

    return arenalog.import_pool(set_code.upper() if set_code else None, log_dir=log_dir, course_key=course_key)


def replay_summary(set_code: str, progress: Progress | None = None, cancel: threading.Event | None = None,
                   n_jobs: int | None = None, refresh: bool = False) -> dict:
    """Replay (turn-by-turn) analysis of a set as a JSON-able dict: mana simulator versus real play, tempo / curving out,
    per-card highlights (see `model/replaystats.py`). Ingests the registered replay files incrementally (writes only
    `sets/<SET>/cache/replay/`); cached by file fingerprints. `{"available": False, "reason": ...}` without replay files or corpus."""
    from sealed_lab.model import replaystats

    return replaystats.replay_summary(set_code.upper(), progress, cancel, n_jobs, refresh)


def replay_card_stats(set_code: str, fmt: str = "bo3", n_jobs: int | None = None, refresh: bool = False,
                      progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[int, dict]:
    """{card_id: stats} from replay data for 'bo3' (TradSealed), 'bo1' (Sealed), 'draft' or an event name: n_seen, cast_rate,
    mean_cast_turn, open_cast_turn, wr_cast / wr_deck_nocast (shrunk), cast_gap ... (descriptive, selection-biased)."""
    from sealed_lab.model import replaystats

    return replaystats.replay_card_stats(set_code.upper(), fmt, n_jobs, refresh, progress, cancel)


def render_replay_section(set_code: str, summary: dict | None = None) -> str:
    """HTML fragment (same CSS classes as report.html) for the replay analysis; computes the summary if not given."""
    from sealed_lab.model import replaystats

    return replaystats.render_replay_section(summary if summary is not None else replay_summary(set_code))


# ---------------------------------------------------------------------------------------------
# new 17lands data: check on start, download
# ---------------------------------------------------------------------------------------------

def datasets_data_dir() -> str:
    """Default folder with the 17lands downloads (env SEALED_LAB_DATA_DIR, else the owner's folder, else under the app home)."""
    from sealed_lab.ingest import datasets

    return datasets.default_data_dir()


def check_new_datasets(data_dir: str | None = None, force: bool = False):
    """`datasets.DatasetCheck`: 17lands files that are new, updated (server copy differs from yours) or already on disk but not
    used by the app, for the sets that matter (sets in the app + the newest few; game / replay data of Sealed, TradSealed,
    PremierDraft, TradDraft, draft data optional).  ~2 s, a few requests (HEAD only); never raises: offline gives an empty check
    with `error` set.  `force` ignores the 15-minute cache and the "don't ask again" list."""
    from sealed_lab.ingest import datasets

    return datasets.check(data_dir, force=force)


def download_datasets(items, data_dir: str | None = None, progress: Progress | None = None,
                      cancel: threading.Event | None = None):
    """Download the given `DatasetItem`s (dicts accepted) into `<data_dir>/<SET>/` (stream to `.part`, then rename), register
    them with `add_files` (also the 'unregistered' ones that are already on disk) and report which sets are now stale.
    Returns `datasets.DownloadReport`; a cancel stops cleanly and keeps the files finished so far."""
    from sealed_lab.ingest import datasets

    its = [datasets.DatasetItem(**i) if isinstance(i, dict) else i for i in items]
    rep = datasets.download(its, str(data_dir or datasets.default_data_dir()), progress=progress, cancel=cancel)
    if rep.to_register:
        rep.registered = add_files(rep.to_register)
        sets = sorted({f.set_code for f in rep.registered if f.kind == "game" and f.set_code})
        rep.stale_sets = [s for s in sets if set_status(s).stale]
    return rep


def dismiss_datasets(items) -> None:
    """"Don't ask again for these": the current version of each file stays out of future checks until it changes."""
    from sealed_lab.ingest import datasets

    datasets.dismiss([datasets.DatasetItem(**i) if isinstance(i, dict) else i for i in items])


# ---------------------------------------------------------------------------------------------
# set detection for a pasted pool
# ---------------------------------------------------------------------------------------------

_NAME_INDEX_CACHE: dict[str, tuple[dict, object]] = {}


def _name_index(set_code: str):
    """NameIndex over the set's card table (cached while card_info's table is unchanged); None without a table.

    Built under the cache lock: the set detector and the card-name check of the Pool tab run on two threads at once, and polars
    frames are not safe to construct from several threads at the same time."""
    with _CACHE_LOCK:
        info = card_info(set_code)
        if not info:
            return None
        hit = _NAME_INDEX_CACHE.get(set_code)
        if hit is not None and hit[0] is info:
            return hit[1]
        import polars as pl

        from sealed_lab.deck.names import NameIndex

        ids = sorted(info)
        idx = NameIndex(pl.DataFrame({"card_id": ids, "name": [info[i]["name"] for i in ids],
                                      "is_basic": [bool(info[i].get("is_basic")) for i in ids]}))
        _NAME_INDEX_CACHE[set_code] = (info, idx)
        return idx


def detect_set(text: str) -> dict:
    """Which set does the pasted Arena pool / deck belong to?  {set_code (None: keep the current choice), matched, total,
    runner_up, runner_up_matched, source ('suffix' | 'names' | 'none'), ambiguous, detected_code, message, scores}.

    Order: the `(SET) 123` suffixes of the export (majority vote, mapped to the app's set codes); else name matching against every
    set that has a card table: needs >= 70% of the pool matched and a clear lead over the runner-up (see deck/setdetect.py)."""
    from sealed_lab.deck import setdetect

    indexes = {}
    for s in registry.known_sets():
        idx = _name_index(s)
        if idx is not None:
            indexes[s] = idx
    return setdetect.detect(text, indexes).to_dict()


# ---------------------------------------------------------------------------------------------
# pasted pool: guessed card names
# ---------------------------------------------------------------------------------------------

def check_pool(set_code: str, text: str) -> dict:
    """How the pasted pool / deck reads against the set's card table (see `deck.names.analyse_pool`): {n_copies, matched, guesses
    [{raw, count, section, card_id, name, score, how, candidates [{card_id, name, score}]}], unmatched [...], ignored, has_deck, n_main,
    n_side}.  A guess is a name that only matched approximately; pass the confirmed names to `build_pool(name_overrides=...)`."""
    from sealed_lab.deck import names

    idx = _name_index(set_code.upper())
    if idx is None:
        raise ValueError(f"set {set_code.upper()} has no card table yet: calculate it on the Sets tab")
    return names.analyse_pool(text, idx)


# ---------------------------------------------------------------------------------------------
# tracked data folder
# ---------------------------------------------------------------------------------------------

def set_folders() -> dict[str, str]:
    """{SET: folder} of the sets that read their files from their own folder instead of `<data folder>/<SET>`."""
    from sealed_lab.ingest import folders

    return folders.set_folders()


def set_set_folder(set_code: str, path: str | None) -> None:
    """Point a set at its own folder (None / "" goes back to `<data folder>/<SET>`).  Call `scan_data_folders(track=[set])` afterwards."""
    from sealed_lab.ingest import folders

    folders.set_set_folder(set_code, path)


def scan_data_folders(data_dir: str | None = None, track: list[str] | None = None, skip_sets: list[str] | None = None,
                      progress: Progress | None = None, cancel: threading.Event | None = None):
    """Register the new / changed 17lands files found in the tracked folders (the data folder's sub-folders of the sets the app knows,
    plus per-set folders), cheaply: files whose size and mtime were seen before are skipped.  `track` = sets to start tracking now,
    `skip_sets` = sets being calculated (left alone, reported in `deferred`).  Returns `folders.ScanReport` (added files, changed sets,
    registered files that disappeared = `missing`).  Replay files stay replay files (same detector as `add_files`)."""
    from sealed_lab.ingest import folders

    skip = {s.upper() for s in (skip_sets or ())}
    for s in set(registry.known_sets()) | {t.upper() for t in (track or ())}:      # sets another process is calculating right now
        if s not in skip and _calculating_elsewhere(s) is not None:
            skip.add(s)
    return folders.scan(data_dir or datasets_data_dir(), track=track or (), skip_sets=sorted(skip), progress=progress, cancel=cancel)


def untracked_sets(data_dir: str | None = None) -> list[dict]:
    """Sets that are not tracked yet: [{set_code, folder (a folder with dataset files, or ""), n_local, n_remote, events}] - sub-folders
    of the data folder and sets on 17lands' stored listing (no network)."""
    from sealed_lab.ingest import folders

    return folders.untracked_sets(data_dir or datasets_data_dir())


def untrack_set(set_code: str) -> None:
    """Stop tracking a set: delete what the app keeps for it (APP_HOME/sets/<SET>: registered files, ingested data, model, caches).
    The 17lands files on disk and the set's folder choice are kept, so it can be tracked again.  Raises SetBusy while it is being
    calculated."""
    import shutil

    sc = set_code.upper()
    _guard(sc)
    d = config.set_dir(sc)
    (d / "registry.json").unlink(missing_ok=True)       # untracked from here on, even if a file below cannot be deleted
    _forget_card_caches(sc)
    _JSON_CACHE.clear()
    _RANGE_CACHE.clear()
    try:
        shutil.rmtree(d, ignore_errors=False)
    except FileNotFoundError:
        pass
    except OSError as e:
        raise OSError(f"{sc} is no longer tracked, but some of its files could not be deleted ({e}). "
                      f"Close what uses them and delete {d}") from e


def dataset_files_for_set(set_code: str, data_dir: str | None = None):
    """`DatasetItem`s for the files 17lands lists for a set that are not in the data folder (for the download flow); [] if unlisted."""
    from sealed_lab.ingest import datasets

    return datasets.items_for_set(set_code, data_dir or datasets.default_data_dir())


# ---------------------------------------------------------------------------------------------
# cheap statistics of a set for the Sets tab (registry, corpus manifest, model meta / report: no corpus table is loaded, except
# one column of games.parquet for the date range, cached)
# ---------------------------------------------------------------------------------------------

_JSON_CACHE: dict[str, tuple[int, dict | None]] = {}
_RANGE_CACHE: dict[str, tuple[int, dict]] = {}
_KIND_ORDER = {"game": 0, "replay": 1, "draft": 2, "cards": 3, "scryfall": 4}
COMPONENT_LABELS = {"extra_draft": "Extra draft events (PickTwo / QuickDraft)", "recency": "Recency weighting", "attributes": "Card-attribute prior", "crossset": "Cross-set prior",
                    "synergy": "Card-pair synergies", "archcond": "Archetype-conditional values", "matchups": "Opponent matchups",
                    "manasim": "Mana simulator v2", "castability": "Castability", "playdraw": "Play/draw card effects",
                    "oppflags": "Opponent colour flags"}


def _file_date(path: str) -> str:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        return ""


def _cached_json(p: Path) -> dict | None:
    try:
        stamp = p.stat().st_mtime_ns
    except OSError:
        return None
    with _CACHE_LOCK:
        hit = _JSON_CACHE.get(str(p))
    if hit is not None and hit[0] == stamp:
        return hit[1]
    d = _read_json(p)
    with _CACHE_LOCK:
        _JSON_CACHE[str(p)] = (stamp, d)
    return d


def _date_ranges(set_code: str) -> dict:
    """{fmt: [first game date, last game date]} from two columns of the corpus games table (cached by file time)."""
    import polars as pl

    p = config.corpus_dir(set_code) / "games.parquet"
    try:
        stamp = p.stat().st_mtime_ns
    except OSError:
        return {}
    with _CACHE_LOCK:
        hit = _RANGE_CACHE.get(str(p))
    if hit is not None and hit[0] == stamp:
        return hit[1]
    out: dict = {}
    try:
        df = (pl.scan_parquet(p).select("fmt", "game_time").group_by("fmt")
              .agg(pl.col("game_time").min().alias("lo"), pl.col("game_time").max().alias("hi")).collect())
        for r in df.iter_rows(named=True):
            out[str(r["fmt"])] = [str(r["lo"])[:10], str(r["hi"])[:10]]
    except Exception:  # noqa: BLE001 - cosmetic
        log.debug("date range of %s unavailable", set_code, exc_info=True)
    with _CACHE_LOCK:
        _RANGE_CACHE[str(p)] = (stamp, out)
    return out


def _top_cards(set_code: str, meta: dict, n: int = 5) -> dict[str, list[dict]]:
    """Best cards per format by the per-copy effect of the model (value of one added copy to an average deck, pp); cards with enough
    data first, cards flagged low_data only to fill up to `n` (and marked)."""
    import math

    import polars as pl

    p = config.model_dir(set_code) / "cards.parquet"
    try:
        df = pl.read_parquet(p)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, list[dict]] = {}
    for fmt in meta.get("formats", []):
        col = f"beta_{fmt}"
        if col not in df.columns:
            continue
        ref = ((meta.get("ref") or {}).get(fmt) or {}).get("reference_wr")
        q = min(max(float(ref), 0.05), 0.95) if ref else 0.55
        base = math.log(q / (1 - q))

        def pp(b: float) -> float:
            return 100.0 * (1 / (1 + math.exp(-(base + b))) - q)
        d = df.filter(~pl.col("is_land")).sort(col, descending=True)
        low = f"low_data_{fmt}"
        flags = d[low].to_list() if low in d.columns else [False] * d.height
        rows = d.to_dicts()
        ok = [r for r, f in zip(rows, flags) if not f][:n]
        poor = [r for r, f in zip(rows, flags) if f][:max(0, n - len(ok))]          # only when fewer than n cards have enough data
        out[fmt] = [dict(name=r["name"], pp=pp(float(r[col])), beta=float(r[col]), exposure=int(r.get(f"exposure_{fmt}") or 0),
                         low_data=r in poor) for r in ok + poor]
    return out


def set_overview(set_code: str) -> dict:
    """Facts about a set for the Sets tab, read from the registry, the corpus manifest, the model's meta.json / report.json (cached by
    file time).  Keys: set_code, n_cards, games {fmt: n}, events {fmt: n}, date_range {fmt: [first, last]}, replay_games, files
    [{name, path, kind, event, size, missing}], model (None without a model, else built_at, seconds, formats, n_boot, code_version,
    components [{key, label, applied, mode, reason}], validation {fmt: {...}}, deployed, castability, lam {stage: {lam, lam_se, source}},
    top_cards {fmt: [{name, pp, ...}]}, degraded, has_report)."""
    from sealed_lab.ingest import folders

    set_code = set_code.upper()
    man = _cached_json(config.corpus_dir(set_code) / "manifest.json") or {}
    meta = _cached_json(config.model_dir(set_code) / "meta.json")
    files = registry.files(set_code, include_shared=False)
    gone = set(folders.missing_files(set_code))
    out: dict = dict(set_code=set_code, n_cards=(meta or {}).get("n_cards") or man.get("n_cards"),
                     games=dict(((meta or {}).get("n_games")) or ((man.get("audit") or {}).get("n_games_by_fmt")) or {}),
                     events={}, date_range=_date_ranges(set_code) if man else {}, replay_games=None,
                     files=[dict(name=Path(f.path).name, path=f.path, kind=f.kind, event=f.event or "", fmt=f.fmt or "",
                                 size=f.size, note=f.note, date=_file_date(f.path), missing=f.path in gone) for f in
                            sorted(files, key=lambda f: (_KIND_ORDER.get(f.kind, 9), f.event or "", f.path))],
                     model=None)
    rs = _cached_json(config.cache_dir(set_code) / "replay" / "summary.json")
    if rs and rs.get("events"):
        out["replay_games"] = sum(int(e.get("n_games") or 0) for e in rs["events"])
    if not meta:
        return out
    stages = meta.get("stages") or {}
    out["events"] = {f: int(st.get("n_events") or 0) for f, st in stages.items() if st.get("n_events") is not None}
    report = _cached_json(config.model_dir(set_code) / "report.json") or {}
    val: dict[str, dict] = {}
    for fmt in meta.get("formats", []):
        v = ((meta.get("validation") or {}).get(fmt)) or {}
        rv = ((report.get("validation") or {}).get(fmt)) or {}
        b0 = (((rv.get("metrics") or {}).get("B0") or {}).get("vs_M") or {}).get("logloss") or {}
        cal = v.get("calibration") or rv.get("calibration") or {}
        val[fmt] = dict(status=v.get("status"), reason=v.get("reason"),
                        gain_vs_B0=b0.get("diff"), lo_vs_B0=b0.get("lo"), hi_vs_B0=b0.get("hi"),
                        gain_vs_B1=v.get("gain_vs_B1"), ci_vs_B1=v.get("ci"), logloss=v.get("model_logloss"), auc=v.get("model_auc"),
                        calibration_slope=v.get("calibration_slope"), level_pp=cal.get("level_pp"), level_ci=cal.get("level_pp_ci"),
                        n_holdout_games=rv.get("n_holdout_games"), holdout_start=v.get("holdout_start") or rv.get("holdout_start"))
    comps = []
    for key, label in COMPONENT_LABELS.items():
        c = meta.get(key)
        if not isinstance(c, dict) or c.get("available") is False:         # (extra draft events: nothing to decide in most sets)
            continue
        comps.append(dict(key=key, label=label, applied=bool(c.get("applied")), mode=c.get("mode"), reason=c.get("reason") or "",
                          **({k_: c[k_] for k_ in ("applied_formats", "source") if k_ in c} if key == "matchups" else {})))
    cast = meta.get("castability") if isinstance(meta.get("castability"), dict) else None
    out["model"] = dict(
        built_at=meta.get("built_at"), seconds=meta.get("seconds"), formats=list(meta.get("formats", [])), n_boot=meta.get("n_boot"), quick=bool(meta.get("quick")),
        code_version=meta.get("code_version"), components=comps, validation=val, deployed=meta.get("deployed"),
        castability=None if cast is None else dict(applied=bool(cast.get("applied")), mode=cast.get("mode"), reason=cast.get("reason") or "",
                                                   rule=cast.get("rule")),
        lam={st: dict(lam=info.get("lam"), lam_se=info.get("lam_se"), source=info.get("lam_from"), sigma=info.get("sigma"))
             for st, info in stages.items()},
        top_cards=_top_cards(set_code, meta), degraded=list(meta.get("degraded") or []),
        has_report=(config.model_dir(set_code) / "report.html").exists())
    return out


def folder_set_codes(path: str) -> dict[str, int]:
    """{SET: number of 17lands dataset files} under a folder (a folder dropped on the window: one set = its folder, several = register all)."""
    from sealed_lab.ingest import folders

    return folders.set_codes_in(Path(path))
