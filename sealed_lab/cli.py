"""Command line interface (the one place where printing is allowed): `python -m sealed_lab cli ...`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sealed_lab import api


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 (got {v})")
    return v


def _p(*a) -> None:
    print(*a)


def _progress_printer():
    last = {"t": -1.0, "m": ""}

    def cb(frac: float, msg: str) -> None:
        pct = int(frac * 100)
        if pct != last["t"] or msg != last["m"]:
            if pct != last["t"] or pct % 5 == 0:
                print(f"[{pct:3d}%] {msg}", file=sys.stderr, flush=True)
            last["t"], last["m"] = pct, msg

    return cb


def _fmt_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return str(n)


def _print_status(st) -> None:
    _p(f"Set {st.set_code}: corpus={'yes' if st.has_corpus else 'no'} model={'yes' if st.has_model else 'no'}"
       f"{' (stale)' if st.stale else ''} formats={','.join(st.formats) or '-'} built={st.built_at or '-'}")
    if st.n_games:
        _p("  games: " + ", ".join(f"{k}={v}" for k, v in st.n_games.items()))
    for f in st.files:
        _p(f"  - {Path(f.path).name:55s} {f.kind:8s} {f.event or '-':14s} {_fmt_size(f.size):>9s}  {f.note}")
    for w in st.warnings:
        _p(f"  ! {w}")


def _print_deck(i: int, d, p_best=None, diff=None) -> None:
    head = f"Deck {i + 1}: {d.colors}{('+' + d.splash) if d.splash else ''}  game WR {d.game_wr:.1%} " \
           f"[{d.game_wr_lo:.1%}, {d.game_wr_hi:.1%}]"
    if d.match_wr is not None:
        head += f"  match WR {d.match_wr:.1%}"
    head += f"  score {d.score:.0f} [{d.score_lo:.0f}, {d.score_hi:.0f}]  exp wins {d.exp_wins:.2f}"
    if p_best is not None:
        head += f"  P(best) {p_best:.0%}"
    if diff:
        head += f"  differs by {diff}"
    _p(head)
    for c in sorted(d.cards, key=lambda c: (c.value_pp is None, -(c.value_pp or 0))):
        pp = "" if c.value_pp is None else f"{c.value_pp:+.2f}pp"
        _p(f"  {c.count} {c.name:40s} {pp:>9s}{' (low data)' if c.low_data else ''}{' ' + c.note if c.note else ''}")
    for w in d.warnings:
        _p(f"  ! {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sealed_lab cli", description="sealed_lab command line")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="register 17lands / Scryfall files or folders")
    a.add_argument("paths", nargs="+")
    s = sub.add_parser("status", help="show sets and their files")
    s.add_argument("set", nargs="?")
    c = sub.add_parser("calc", help="ingest + model for a set")
    c.add_argument("set")
    c.add_argument("--jobs", type=_positive_int, default=None)
    c.add_argument("--boot", type=_positive_int, default=16)
    c.add_argument("--no-scryfall", action="store_true")
    c.add_argument("--recency", default="auto", choices=["auto", "on", "off"],
                   help="weight recent events more: auto = only if it improves held-out predictions")
    c.add_argument("--attributes", default="auto", choices=["auto", "on", "off"],
                   help="card-attribute prior (mana value, rarity, types, keywords, text tags): auto = only if it improves "
                        "held-out predictions")
    c.add_argument("--crossset", default="auto", choices=["auto", "on", "off"],
                   help="borrow the structure-term prior from other sets' models: auto = only if it improves held-out predictions")
    for flag, what in (("synergy", "card-pair synergies"), ("archcond", "card values conditional on the deck's archetype")):
        c.add_argument("--" + flag, default="auto", choices=["auto", "on", "off"],
                       help=f"{what}: auto = only if it improves held-out predictions (95%% CI above zero)")
    c.add_argument("--matchups", default="pooled", choices=["auto", "pooled", "on", "off"],
                   help="card x opponent-color interactions (sideboard plans): auto = only if it improves this set's held-out predictions "
                        "(95%% CI above zero); pooled (default) = auto, plus, in each format where auto does not apply it, the hold-out "
                        "evidence pooled across the other sets (needs >= 5 other sets with a stored report; the set's own numbers must not "
                        "be materially worse)")
    c.add_argument("--playdraw", default="auto", choices=["auto", "on", "off"],
                   help="card x play/draw interactions (Bo3 games-2-3 advice only; never the deck score): auto = only if it improves "
                        "held-out predictions with the usual confirmation")
    c.add_argument("--manasim", default="auto", choices=["auto", "v2", "v1"],
                   help="calibrated mana simulator (queue-specific hands/draws, real keep rule, card flow): auto = only if it "
                        "improves held-out predictions (95%% CI above zero), v2 = always, v1 = the original simulator")
    c.add_argument("--castability", default="auto", choices=["auto", "on", "off"],
                   help="card copies weighted by castability + expected dead cards: auto = ships unless a format is materially "
                        "worse on the hold-out (safety constraint), on = always, off = never")
    c.add_argument("--oppflags", default="off", choices=["auto", "on", "off"],
                   help="opponent color flags (W U B R G multi-hot + number of colors) in the game covariates; default off because 17lands opp_colors are the colors shown during the game (docs/opp_flags/RESULT.md): auto = only if the "
                        "chain with them beats the chain without on the hold-out (confirmed on the later half), on = always, off = never")
    c.add_argument("--extra-draft", default="auto", choices=["auto", "on", "off"],
                   help="draft games of PickTwo / QuickDraft events: auto = keep them only if the plain chain with them beats the one "
                        "without them on the hold-out (decided first), on = always, off = never")
    c.add_argument("--draft-formats", action="store_true",
                   help="also fit the draft target formats dbo1 (PremierDraft) and dbo3 (TradDraft) for the Deck builder "
                        "(docs/draft_mode/RULE.md); the Sealed models do not change")
    c.add_argument("--draft-boot", type=int, default=16,
                   help="bootstrap draws of each draft format (default 16; 8 is the pre-registered fallback if the draft fit is too slow)")
    c.add_argument("--quick", action="store_true",
                   help="quick calculation: skip the four tests that never changed a model (mana simulator v2, recency, attribute prior, "
                        "extra draft); overrides --manasim/--recency/--attributes/--extra-draft; about 40%% faster")
    c.add_argument("--sigma-cv", default="time", choices=["time", "random"],
                   help="prior strength of the Sealed stages: time = forward-chaining folds over event start times, random = original folds")
    c.add_argument("--stack", default="auto", choices=["auto", "off"],
                   help="auto = also choose the weight w of q = w M + (1 - w) P on the same folds and deploy q; off = deploy M")
    c.add_argument("--no-confirm", action="store_true",
                   help="skip the confirmation split (a component must otherwise pass on the earlier half of the hold-out events "
                        "and be confirmed on the later half)")
    c.add_argument("--leak-free-folds", action="store_true",
                   help="in the time-split CV refit the prior per fold on earlier events only, instead of reusing the prior "
                        "fitted on all training data (slower; scored worse on the hold-out)")
    c.add_argument("--theta-sigma", default="rule", choices=["rule", "cv"],
                   help="prior SD of the structure / mana terms in the Sealed stages: rule = clip(4 x card sigma); "
                        "cv = chosen by the same cross-validation (expert option, slower)")
    c.add_argument("--recency-scope", default="format", choices=["format", "pooled"],
                   help="format = one half-life per format (draft never weighted), pooled = one half-life for all")
    b = sub.add_parser("build", help="build the best decks from a pool (Arena export text file)")
    b.add_argument("set")
    b.add_argument("pool")
    b.add_argument("--fmt", default="bo3", choices=["bo3", "bo1", "dbo3", "dbo1"])
    b.add_argument("--n-decks", type=int, default=5, help="maximum number of deck families to show (1 = only the best deck)")
    sc = sub.add_parser("score", help="score a deck (Arena export text file)")
    sc.add_argument("set")
    sc.add_argument("deck")
    sc.add_argument("--fmt", default="bo3", choices=["bo3", "bo1", "dbo3", "dbo1"])
    rp = sub.add_parser("report", help="the set report: game length, color pairs, cards players misjudge (cached per model)")
    rp.add_argument("set")
    rp.add_argument("--fmt", default="bo3", choices=["bo3", "bo1"])
    rp.add_argument("--out", default=None, help="write the full report as JSON to this file instead of printing a summary")
    rp.add_argument("--refresh", action="store_true", help="recompute even if a cached report matches the model")
    rf = sub.add_parser("report-facts", help="compact facts digest of the Bo1 and Bo3 reports for the report writer (text; --out also writes JSON)")
    rf.add_argument("set")
    rf.add_argument("--out", default=None, help="write the JSON to this file and the text next to it (same name, .txt)")
    rf.add_argument("--refresh", action="store_true", help="recompute the reports even if cached")
    ri = sub.add_parser("report-install", help="install a written report (Report tab): refused unless it passes the deterministic checker")
    ri.add_argument("set")
    ri.add_argument("report", help="the report text (Markdown with <!--F:id--> tags)")
    ri.add_argument("facts", help="the facts JSON (cli report-facts --out) the text was written from")
    sy = sub.add_parser("synth", help="write a synthetic 17lands-format data set")
    sy.add_argument("outdir")
    sy.add_argument("--scale", type=float, default=1.0)
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument("--set", default="TST")
    args = ap.parse_args(argv)
    from sealed_lab.api import CalculationError
    from sealed_lab.types import Cancelled, SetBusy

    try:
        return _run(args)
    except (FileNotFoundError, SetBusy, CalculationError) as e:     # expected conditions: a message and an exit code, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Cancelled:
        print("cancelled", file=sys.stderr)
        return 130


def _run(args) -> int:
    if args.cmd == "add":
        infos = api.add_files(args.paths)
        for f in infos:
            _p(f"{Path(f.path).name:55s} {f.kind:8s} set={f.set_code or '-':5s} event={f.event or '-':13s} "
               f"{_fmt_size(f.size):>9s}  {f.note}")
        n_ok = sum(1 for f in infos if f.kind != "unknown")
        _p(f"{n_ok} of {len(infos)} file(s) recognised")
        return 0 if infos else 1
    if args.cmd == "status":
        sets = [api.set_status(args.set)] if args.set else api.list_sets()
        if args.set and not (sets[0].files or sets[0].has_corpus or sets[0].has_model):
            print(f"error: unknown set {args.set.upper()}: no files registered (use: add <files or folders>)", file=sys.stderr)
            return 1
        if not sets:
            _p("No sets registered yet. Use: add <files or folders>")
        for st in sets:
            _print_status(st)
        return 0
    if args.cmd == "calc":
        st = api.calculate_set(args.set, progress=_progress_printer(), n_jobs=args.jobs, n_boot=args.boot,
                               fetch_scryfall=not args.no_scryfall, recency=args.recency, attributes=args.attributes,
                               crossset=args.crossset, synergy=args.synergy, archcond=args.archcond, matchups=args.matchups,
                               manasim=args.manasim, castability=args.castability, sigma_cv=args.sigma_cv, stack=args.stack,
                               confirm=not args.no_confirm, recency_scope=args.recency_scope,
                               leak_free=args.leak_free_folds, extra_draft=args.extra_draft,
                               theta_sigma=args.theta_sigma, playdraw=args.playdraw, oppflags=args.oppflags, quick=args.quick,
                               draft_formats=args.draft_formats, draft_boot=args.draft_boot)
        _print_status(st)
        return 0
    if args.cmd == "build":
        text = Path(args.pool).read_text(encoding="utf-8")
        res = api.build_pool(args.set, text, args.fmt, n_decks=args.n_decks, progress=_progress_printer())
        meta = res.family_meta
        if meta:
            _p(f"{meta['n_shown']} of {meta['n_families']} deck families shown ({meta['n_candidates']} candidate decks, "
               f"{meta['n_draws']} bootstrap draws: P(best) moves in steps of {meta['step']:.1%}; {meta['seconds']:.1f}s)")
            _p("")
        for i, d in enumerate(res.decks):
            _print_deck(i, d, res.p_best[i] if i < len(res.p_best) else None, res.diffs[i] if i < len(res.diffs) else None)
            fam = res.families[i] if i < len(res.families) else None
            if fam:
                gap = "" if i == 0 else (f"  gap to #1 {fam['gap_pp']:.2f} pp [80%: {fam['gap_lo_pp']:+.2f}, {fam['gap_hi_pp']:+.2f}]")
                _p(f"  Family: {fam['n_members']} similar decks, {fam['why']}{gap}" + (f"; archetype {fam['archetype']}" if fam['archetype'] else ""))
                _p("  Core: " + ", ".join(f"{c['count']}x {c['name']}" if c['count'] > 1 else c['name'] for c in fam['core']))
                for fx in fam['flex']:
                    alts = ", ".join(f"{a['name']} ({a['freq']:.0%})" for a in fx['alts'])
                    _p(f"  Flex: {fx['name']} ({fx['freq']:.0%})" + (f" <-> {alts}" if alts else ""))
            _p("")
        if res.pool_unmatched:
            _p("Unmatched pool names: " + ", ".join(res.pool_unmatched))
        for w in res.warnings:
            _p(f"! {w}")
        return 0
    if args.cmd == "score":
        text = Path(args.deck).read_text(encoding="utf-8")
        res = api.score_deck(args.set, text, args.fmt)
        _print_deck(0, res.deck)
        if res.efficiency_pp is not None:
            _p(f"Efficiency vs best build from the full pool: {res.efficiency_pp:+.2f}pp")
        return 0
    if args.cmd == "report":
        import json

        from sealed_lab import report

        rep = api.set_report(args.set, args.fmt, refresh=args.refresh, progress=_progress_printer())
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            _p(f"Wrote {args.out}")
        else:
            _p(report.summary_text(rep))
        return 0
    if args.cmd == "report-facts":
        import json

        from sealed_lab import report_facts

        facts = report_facts.build_facts(args.set, refresh=args.refresh, progress=_progress_printer())
        text = report_facts.render_text(facts)
        if args.out:
            out = Path(args.out)
            out.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
            out.with_suffix(".txt").write_text(text, encoding="utf-8")
            _p(f"Wrote {out} and {out.with_suffix('.txt')} ({len(text.encode('utf-8'))} bytes of text)")
        else:
            _p(text)
        return 0
    if args.cmd == "report-install":
        from sealed_lab import written_report

        try:
            meta = written_report.install(args.set, args.report, args.facts)
        except written_report.ReportRejected as e:
            _p(f"Not installed: the report fails the checker.\n{e}")
            return 1
        _p(f"Installed the written report of {meta['set']} in {written_report.folder()}")
        return 0
    if args.cmd == "synth":
        from sealed_lab.synth.generate import generate

        r = generate(args.outdir, set_code=args.set, seed=args.seed, scale=args.scale)
        _p(f"Wrote synthetic data to {r['out_dir']}: " + ", ".join(f"{k}={v}" for k, v in r["n_games"].items()) + " games")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
