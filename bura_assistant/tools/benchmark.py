"""How many deals this computer calculates per tip.

    python tools/benchmark.py            uses the workers setting from config/app.json
    python tools/benchmark.py 4          force 4 worker processes
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bura.app_core import _worker_count, load_config  # noqa: E402
from bura.engine.core import ME  # noqa: E402
from bura.engine.search import DecisionEngine, InfoState  # noqa: E402
from bura.rules import Rules  # noqa: E402


def positions(R: Rules) -> dict[str, InfoState]:
    P = R.parse_cards
    kc = R.parse_card("KC")
    return {
        "start of round (stock 26)": InfoState(R, R.suits.index("D"), P("6S AD 7H 10C KH"), 5, 26,
                                               trump_card=P("7D")[0], leader=ME),
        "middle (stock 16)": InfoState(R, R.suit_of[kc], P("6D 7D JD AD JC"), 5, 16, trump_card=kc, leader=ME,
                                       my_pile_known=P("KH QH 8C QC 8H 10H"), opp_pile_known=P("6S AS 9S 9D")),
        "short stock (stock 6)": InfoState(R, R.suit_of[kc], P("6D 7D JD AD JC"), 5, 6, trump_card=kc, leader=ME,
                                           my_pile_known=P("KH QH 8C QC 8H 10H 6H 7H 6C 7S QS 9H"),
                                           opp_pile_known=P("6S AS 9S 9D JH AH 7C 9C")),
    }


def main() -> None:
    cfg = load_config(os.path.join(ROOT, "config", "app.json"))
    ec = cfg["engine"]
    R = Rules.load(os.path.join(ROOT, cfg["paths"]["rules"]))
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else _worker_count(ec.get("workers", "auto"))
    eng = DecisionEngine(R, ec.get("time_budget_s", 2.5), ec.get("max_samples", 100000),
                         opponent_model=ec.get("opponent_model", "lookahead"))
    workers = eng.start_workers(workers)
    print(f"CPU threads: {os.cpu_count()}, worker processes: {workers}, time per tip: {eng.time_budget}s")
    try:
        for name, info in positions(R).items():
            counts = []
            for _ in range(3):
                t = time.perf_counter()
                eng.evaluate(info)
                counts.append(eng.last_checked)
            print(f"  {name:27s} {min(counts):6,} - {max(counts):6,} deals per tip "
                  f"(last took {time.perf_counter() - t:.1f}s)")
    finally:
        eng.stop_workers()


if __name__ == "__main__":
    main()
