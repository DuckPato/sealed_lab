"""Builder check: among real player Sealed decks, do decks shaped more like the builder's win more (same skill, same colour pair)?

    python scripts/builder_check.py SET [SET ...] OUTDIR

Writes OUTDIR/builder_check_<SET>.json per set (see sealed_lab/builder_check.py for the pre-registered design).  Reads the corpus and
docs/builder_composition/builder_composition_<SET>.json; does not touch the model or the app home.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from sealed_lab import builder_check as B

    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 2:
        print(__doc__)
        return 2
    outdir, sets = Path(a[-1]), a[:-1]
    for sc in sets:
        t0 = time.time()
        res = B.run(sc)
        p = B.write(res, outdir)
        print(f"{res['set']}: {p} ({time.time() - t0:.0f}s)" + (f"  SKIPPED: {res['skipped']}" if res.get("skipped") else ""))
        for fmt, f in res["formats"].items():
            c = f.get("combined")
            if c:
                print(f"  {fmt}: {f['n_games']} games, combined {c['pp']:+.2f} +- {c['se_total']:.2f} pp [{c['badge']}]  "
                      + " ".join(f"{k}={v['badge']}" for k, v in f["betas"].items()))
            else:
                print(f"  {fmt}: skipped: {f.get('skipped')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
