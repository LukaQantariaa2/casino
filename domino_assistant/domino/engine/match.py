"""Chance of winning the match from a given score.

A match is a race to the target (75 / 175 / 255 / 355 on Adjarabet); it is decided at the end of a
round. The table is built from the distribution of round results (points for the player who starts
the round, points for the other one, whether the starter won the round and so starts again), which
comes from self-play of the playing policy and is stored in round_outcomes.json.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

import numpy as np

from .game import ME, OPP, TILES, Board, Rules

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTCOMES_FILE = os.path.join(_HERE, "round_outcomes.json")


def simulate_rounds(n: int, seed: int = 1) -> list[tuple[int, int, int]]:
    """Self-play n rounds; returns (starter points, other points, starter won) per round."""
    from .search import play_out
    rng = random.Random(seed)
    rules = Rules()
    out = []
    for _ in range(n):
        deck = list(TILES)
        rng.shuffle(deck)
        hands = [deck[:7], deck[7:14]]
        bone = deck[14:]
        gained = [0, 0]
        w = play_out(Board(), hands, bone, ME, gained, rules, rng)
        out.append((gained[ME], gained[OPP], 1 if w in (ME, None) else 0))
    return out


def load_outcomes() -> list[tuple[int, int, int, float]]:
    """(starter pts/5, other pts/5, starter won, weight)."""
    try:
        with open(OUTCOMES_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)["rounds"]
    except (OSError, ValueError, KeyError):
        raw = simulate_rounds(1500)
    c = Counter((a // 5, b // 5, s) for a, b, s in raw)
    total = sum(c.values())
    return [(a, b, s, n / total) for (a, b, s), n in c.items()]


class MatchTable:
    def __init__(self, target: int, outcomes=None):
        self.target = target
        self.n = max(1, -(-target // 5))           # score units below the target
        outs = outcomes if outcomes is not None else load_outcomes()
        outs = [o for o in outs if o[0] or o[1]]
        wsum = sum(o[3] for o in outs)
        self.outs = [(a, b, s, w / wsum) for a, b, s, w in outs]
        self.M = self._build()

    def _terminal(self, na, nb):
        return np.where(na > nb, 1.0, np.where(nb > na, 0.0, 0.5))

    def _build(self) -> np.ndarray:
        N = self.n
        M = np.zeros((N, N, 2))
        for tot in range(2 * N - 2, -1, -1):
            a = np.arange(max(0, tot - N + 1), min(N - 1, tot) + 1)
            b = tot - a
            for s in (0, 1):                       # 0: I start this round, 1: he starts
                acc = np.zeros(len(a))
                for ds, do, sw, w in self.outs:
                    if s == 0:
                        na, nb, ns = a + ds, b + do, (0 if sw else 1)
                    else:
                        na, nb, ns = a + do, b + ds, (1 if sw else 0)
                    term = (na >= N) | (nb >= N)
                    v = self._terminal(na, nb)
                    inside = ~term
                    if inside.any():
                        v = v.copy()
                        v[inside] = M[na[inside], nb[inside], ns]
                    acc += w * v
                M[a, b, s] = acc
        return M

    def after_round(self, my_score: int, opp_score: int, next_starter: int) -> float:
        """Match win chance once a round ended at this score; next_starter = the round winner."""
        T = self.target
        if my_score >= T or opp_score >= T:
            return 1.0 if my_score > opp_score else 0.0 if opp_score > my_score else 0.5
        return float(self.M[my_score // 5, opp_score // 5, 0 if next_starter == ME else 1])
