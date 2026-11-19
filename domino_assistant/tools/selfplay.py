"""Strength check: the engine (player A) against the fast look-ahead policy (player B).

    python tools/selfplay.py [rounds] [seconds per move]

Both players see only their own tiles; A gets the same draw/pass information the live tracker
collects. Reports A's average points per round minus B's, with a 95% interval, and A's round wins.
Rounds are dealt in pairs with the hands swapped, so luck of the deal mostly cancels.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino.engine.game import ME, OPP, TILES, Board, Rules, pips, round_up5  # noqa: E402
from domino.engine.knowledge import OppKnowledge  # noqa: E402
from domino.engine.search import Engine, Info, _blocked, choose  # noqa: E402


def play_round(deck, starter, engine: Engine, budget: float, rng: random.Random):
    rules = engine.rules
    hands = [list(deck[:7]), list(deck[7:14])]
    bone = list(deck[14:])
    board = Board()
    gained = [0, 0]
    know = OppKnowledge(7)
    to_move, passes = starter, 0
    while True:
        h = hands[to_move]
        mv = board.moves(h)
        while not mv and len(bone) > rules.boneyard_reserve:
            if to_move == OPP:
                know.drew(board.open_values())
            h.append(bone.pop())
            mv = board.moves(h)
        if not mv:
            if to_move == OPP:
                know.passed(board.open_values())
            passes += 1
            if passes >= 2:
                w = _blocked(hands, gained)
                return gained, w
            to_move ^= 1
            continue
        passes = 0
        if to_move == ME and len(mv) > 1:
            seen = set(hands[ME]) | {t for t in TILES if t not in set(hands[OPP]) | set(bone)}
            unknown = [t for t in TILES if t not in seen]
            info = Info(board, list(hands[ME]), len(hands[OPP]), unknown, know.forbidden())
            res = engine.evaluate(info, mv, budget)
            t, arm = res[0].move
        else:
            t, arm = choose(board, h, hands[to_move ^ 1], mv)
        h.remove(t)
        if to_move == OPP:
            know.played(t)
        board = board.place(t, arm)
        gained[to_move] += board.points()
        if not h:
            gained[to_move] += round_up5(pips(hands[to_move ^ 1]))
            return gained, to_move
        to_move ^= 1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    rng = random.Random(seed)
    eng = Engine(Rules(), budget, seed=seed + 1)
    diffs, wins = [], 0
    t0 = time.time()
    for i in range(n):
        deck = list(TILES)
        rng.shuffle(deck)
        for swap in (False, True):
            d = deck[7:14] + deck[:7] + deck[14:] if swap else deck
            g, w = play_round(d, i % 2, eng, budget, rng)
            diffs.append(g[ME] - g[OPP])
            wins += w == ME
    m = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
    print(f"{len(diffs)} rounds in {time.time() - t0:.0f}s: engine minus policy {m:+.2f} points/round "
          f"(±{1.96 * sd / math.sqrt(len(diffs)):.2f}), engine won {wins}/{len(diffs)} rounds")


if __name__ == "__main__":
    main()
