"""Measure the mana simulator's parameters (keep rule, hand smoother, draw smoothing, card flow) on replay data.

    python scripts/fit_manasim.py MSH SOS [--out calibration.json]

Reads the replay shards of the given sets from the app home (`SEALED_LAB_HOME`; run Calculate or `ingest_replays` first), prints
the estimates per regime (random / hand / sealed) and optionally writes them as JSON. The shipped constants in
`sealed_lab/model/manasim.py` were measured this way on MSH + SOS; the realism table of the report shows how well they fit a set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from sealed_lab.model import simcalib

    r = simcalib.calibrate([s.upper() for s in args.sets])
    if r.get("status") != "ok":
        print(r.get("status"))
        return 1
    for reg, d in r["regimes"].items():
        h = d["hand"]
        print(f"== {reg}: {d['n_games']:,} games ({', '.join(d['events'])}), mulligan rate {d['mull_rate']:.3f}")
        if h.get("status") == "ok":
            print(f"   hand smoother: s = {h['s']:.3f}, g = {h['g']:.3f} ({'smoothed' if h['smoothed'] else 'plain shuffle'}); "
                  f"deviance vs saturated {h['deviance_fit']:.0f} (plain hypergeometric {h['deviance_plain']:.0f}), P(3 lands | 17) = {h['p3_at_17']:.3f}")
        t = d["draw_tilt"]
        if t.get("status") == "ok":
            print(f"   draw tilt per land of deviation: {t['b']:+.3f} (SE {t['se']:.3f}, {t['n']:,} draws)")
        rs = d.get("rescue", {})
        if rs.get("status") == "ok":
            print("   land rescue (no land in hand), lift in P(land) at turn 3/4/5+ (pp):",
                  {k: round(v["lift_pp"], 1) for k, v in rs["by_turn"].items()})
        print("   extra cards drawn by own turn k:", {k: round(v, 3) for k, v in d["extra_draws"].items() if v is not None})
        print("   P(mulligan) after the first hand, lands 0..7: on the draw", d["keep"][0][0], "on the play", d["keep"][0][1])
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
