"""Benchmark ingest throughput on a synthetic wide game file.

    python scripts/bench_ingest.py [rows=100000] [pad_cards=144]

`pad_cards` adds filler card columns so the file is ~1,500 columns wide like a real Premier Draft file.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sealed_lab.ingest.detect import fingerprint  # noqa: E402
from sealed_lab.ingest.game_data import process_game_file  # noqa: E402
from sealed_lab.synth import generate as G  # noqa: E402


def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    pad = int(sys.argv[2]) if len(sys.argv) > 2 else 144
    tmp = Path(tempfile.mkdtemp(prefix="sealed_lab_bench_"))
    world = G.build_world("TST", 0)
    path = tmp / "game_data_public.TST.PremierDraft.csv.gz"
    t = time.time()
    n = G.generate_game_file(world, "PremierDraft", rows, path, np.random.default_rng(1), pad_cards=pad)
    print(f"generated {n} rows x {5 * (world.C + pad) + 20} columns in {time.time() - t:.1f}s "
          f"({path.stat().st_size / 1e6:.0f} MB gz)")
    fp = fingerprint(path)
    t = time.time()
    meta = process_game_file(str(path), fp, "PremierDraft", "TST", tmp / "corpus")
    dt = time.time() - t
    print(f"ingest: {meta['rows']} rows in {dt:.1f}s = {meta['rows'] / dt:,.0f} rows/s "
          f"({meta['n_decks']} unique decks)")


if __name__ == "__main__":
    main()
