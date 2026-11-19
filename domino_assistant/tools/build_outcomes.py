"""Self-play many rounds and store their results for the match-win table.

    python tools/build_outcomes.py [rounds]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino.engine.match import OUTCOMES_FILE, simulate_rounds  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    t0 = time.time()
    rounds = simulate_rounds(n, seed=12345)
    with open(OUTCOMES_FILE, "w", encoding="utf-8") as fh:
        json.dump({"rounds": rounds}, fh, separators=(",", ":"))
    won = sum(r[2] for r in rounds) / len(rounds)
    print(f"{n} rounds in {time.time() - t0:.1f}s, starter wins {won:.1%} -> {OUTCOMES_FILE}")


if __name__ == "__main__":
    main()
