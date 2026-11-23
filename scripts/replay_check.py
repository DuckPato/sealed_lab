"""Replay-data check: ingest the replay files of a set and compare the mana simulator with real play.

    python scripts/replay_check.py SET [--jobs N] [--refresh] [--events TradSealed Sealed ...] [--json OUT.json]

Runs `ingest.replay.ingest_replays` (incremental by file fingerprint, writes only sets/<SET>/cache/replay/) and
`model.replaystats.replay_summary` on the real app home (SEALED_LAB_HOME or %LOCALAPPDATA%\\sealed_lab) and prints the
key numbers. Needs the corpus (press Calculate in the app once) for the set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _pct(x, d=2):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _ci(r, fmt):
    return f"{fmt(r['real'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] vs sim {fmt(r['sim'])}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_code")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="recompute the statistics even if cached")
    ap.add_argument("--events", nargs="*", default=None, help="only ingest these events (default: all registered)")
    ap.add_argument("--json", default=None, help="write the full summary here")
    a = ap.parse_args(argv)

    from sealed_lab.ingest import replay
    from sealed_lab.model import replaystats

    sc = a.set_code.upper()
    t0 = time.time()
    last = [-1.0]

    def prog(f, m):
        if f - last[0] >= 0.1 or f >= 1.0:
            last[0] = f
            print(f"  [{100 * f:5.1f}%] {m}", flush=True)

    t = time.time()
    r = replay.ingest_replays(sc, prog, None, a.jobs, events=a.events)
    print(f"ingest: {r['ingested']} file(s) ingested, {r['cached']} cached, {time.time() - t:.1f}s")
    for m in r["files"]:
        print(f"  {m['event']:<13} {m['rows']:>8,} games  {m['n_draw_rows']:>10,} draw rows  {m['n_cast_rows']:>10,} cast rows")
    if a.events:
        print("(--events given: statistics are computed on all shards that exist)")
    t = time.time()
    s = replaystats.replay_summary(sc, prog, None, a.jobs, refresh=a.refresh)
    if not s.get("available"):
        print("not available:", s.get("reason"))
        return 1
    print(f"statistics: {time.time() - t:.1f}s (total {time.time() - t0:.1f}s); id map: {s['idmap']['n_mapped']}/{s['idmap']['n_ids']} ids, "
          f"{_pct(s['idmap']['mass_mapped'], 3)} of appearances")
    for ev in s["events"]:
        print(f"{ev['label']}: {ev['n_games']:,} games, win rate {_pct(ev['win_rate'])}, on play {_pct(ev['on_play_share'])}")
    print("\n== simulator vs real (real re-weighted to 50/50 play-draw; cluster-bootstrap 95% CI) ==")
    for r in s["simulator"]["rows"]:
        lab = f"{r['event']:<12} {('L=' + str(r['n_lands'])) if r['n_lands'] else 'all L':<6}"
        pad = " " * 19
        if r.get("mull"):
            x = r["mull"]
            print(f"{lab} mulligan rate:       n={x['n']:>7,} {_ci(x, _pct)}  diff {100 * x['diff']:+.2f}pp")
        if r.get("screw"):
            x = r["screw"]
            sh = r.get("screw_hand")
            sn = r.get("screw_hand_nd")
            extra = "" if sh is None else f"  | 0-mull real - random draws {100 * sh['diff']:+.2f}pp [{100 * sh['diff_lo']:+.2f}, {100 * sh['diff_hi']:+.2f}]"
            if sn is not None:
                extra += f" (with actual draw counts {100 * sn['diff']:+.2f}pp)"
            print(f"{pad} P(<3 lands after T3): n={x['n']:>7,} {_ci(x, _pct)}  diff {100 * x['diff']:+.2f}pp "
                  f"[{100 * x['diff_lo']:+.2f}, {100 * x['diff_hi']:+.2f}]  sim-with-real-keep-rule {_pct(x.get('rule'))}{extra}")
        if r.get("mana4"):
            x = r["mana4"]
            print(f"{pad} mana T1-T4:           n={x['n']:>7,} {_ci(x, lambda v: f'{v:.2f}')}  diff {x['diff']:+.2f} [{x['diff_lo']:+.2f}, {x['diff_hi']:+.2f}]")
        if r.get("mana4_long"):
            x = r["mana4_long"]
            print(f"{pad} mana T1-T4 (7+ turns): n={x['n']:>6,} {_ci(x, lambda v: f'{v:.2f}')}  diff {x['diff']:+.2f} [{x['diff_lo']:+.2f}, {x['diff_hi']:+.2f}]")
        if r.get("mana7"):
            x = r["mana7"]
            print(f"{pad} mana T1-T7/28:        n={x['n']:>7,} {_ci(x, lambda v: f'{v:.3f}')}  diff {x['diff']:+.3f} [{x['diff_lo']:+.3f}, {x['diff_hi']:+.3f}]")
        if r.get("flood16"):
            x = r["flood16"]
            print(f"{pad} P(>=9 lands/16):      n={x['n']:>7,} {_ci(x, _pct)} (random draws given hand)  unconditional={_pct(x['unconditional'])} sim-model={_pct(x['sim_model'])}")
        if r.get("first7_tail"):
            x = r["first7_tail"]
            print(f"{pad} first 7 outside 2-5:  n={x['n']:>7,} {_ci(x, _pct)}  random shuffle={_pct(x['exact'])}")
    print("\n== per-deck calibration of simulated screw ==")
    for ev, c in s["simulator"]["calibration"].items():
        sl = c["screw"]["slope"]
        m = c.get("mana7_slope")
        print(f"{ev:<12} decks={c['n_decks']:>6,} slope {sl['est']:.2f} [{sl['lo']:.2f}, {sl['hi']:.2f}]; sim SD within land count {c['sim_sd_within_lands']:.4f}"
              + ("" if not m else f"; mana T1-T7 slope {m['slope']['est']:.2f} [{m['slope']['lo']:.2f}, {m['slope']['hi']:.2f}]"))
    print("\n== P(first card drawn is a land) after a kept 7-card hand (0 mulligans): real vs random shuffle ==")
    for ev, lst in s["simulator"].get("draw_bias", {}).items():
        print(f"{ev:<12} " + "  ".join(f"{x['lands_in_hand']}L: {100 * x['real']:.1f}% vs {100 * x['sim']:.1f}% (n={x['n']:,})" for x in lst))
    print("\n== real mulligan probability by lands in first 7 (play / draw) ==")
    for ev, lst in s["simulator"]["keep_rules"].items():
        for play in (True, False):
            cells = []
            for k in range(8):
                x = next((z for z in lst if z["on_play"] == play and z["lands"] == k), None)
                cells.append("  -  " if not x or x["n"] < 30 else f"{100 * x['p_mull']:4.0f}%")
            print(f"{ev:<12} {'play' if play else 'draw'}: " + " ".join(cells))
    print("\n== tempo (games with >= 4 completed own turns) ==")
    for ev, t_ in s["tempo"].items():
        if "effects" not in t_:
            continue
        ef = t_["effects"]
        w = ef.get("within_build")

        def f3(v):
            return f"{v[0]:+.2f}pp [{v[1]:+.2f}, {v[2]:+.2f}]"
        print(f"{ev:<12} n={t_['n_games']:>7,} full-curve share {_pct(t_['share_full_curve'], 0)} | +1 mana T1-T4 {f3(ef['mana4_per_point']['ame_pp'])} "
              f"| same lands {f3(ef['mana4_given_lands']['ame_pp'])} | full curve {f3(ef['full_curve']['ame_pp'])}"
              + ("" if not w else f" | within-deck full curve {w['full_curve']['pp']:+.2f}pp [{w['full_curve']['lo']:+.2f}, {w['full_curve']['hi']:+.2f}]"))
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1), encoding="utf-8")
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
