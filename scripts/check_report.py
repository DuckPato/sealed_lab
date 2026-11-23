"""Deterministic checker of a written report against its facts file.

    python scripts/check_report.py REPORT.md FACTS.json [--length]

REPORT.md   the written report; every factual sentence carries one or more tags  <!--F:ID-->  or  <!--F:ID1,ID2-->  (before or after the full stop).
FACTS.json  the output of `cli report-facts SET --out FACTS.json` (claims under "facts", "no clear difference" items under "noclear").

Rules (one problem is listed per broken rule and sentence; exit status 1 if any, else 0):
  1. A sentence with a number or a direction word (more, less, fewer, better, worse, wins, loses, beats, higher, lower, best, worst,
     overrated, underrated) needs a tag with at least one known id.
  2. Every tagged id must exist (in the claims or the no-clear list).
  3. Every number of the sentence must match a number of a tagged claim within rounding (the digits as written: 1.4 matches 1.36 and 1.4;
     "5" matches 4.6; 12% matches 11.6; a leading + or minus sign must match the sign of the number).  Numbers in the names "Bo1" / "best of 3" and
     the thousands commas are ignored.  No-clear items carry no numbers.
  4. A sentence tagged with a no-clear id must not use direction words, except inside a "no clear ..." / "no difference" statement.
  --length  also requires 400-800 words (headings and tags not counted).
Headings (lines starting with #) are not sentences.  This checks the wording against the facts, not whether the facts are well chosen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sealed_lab.report_check import (  # noqa: E402,F401  (re-exported: the tests use them from this module)
    DIRECTION, LEN_MAX, LEN_MIN, NO_CLEAR, NUMBER, SENTENCE, SKIP_NUM, TAG,
    check, number_ok, numbers_of, plain, split_sentences, tags_of,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report")
    ap.add_argument("facts")
    ap.add_argument("--length", action="store_true", help="also require 400-800 words")
    args = ap.parse_args(argv)
    facts = json.loads(Path(args.facts).read_text(encoding="utf-8"))
    problems = check(Path(args.report).read_text(encoding="utf-8"), facts, check_length=args.length)
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
        return 1
    print("OK: every claim is tagged and matches its facts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
