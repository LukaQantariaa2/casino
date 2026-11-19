"""Move evaluation: sample the hidden tiles, play every candidate out, average the result.

Hidden information is the opponent's hand and the boneyard order. Each sample ("deal") respects
what the opponent revealed: every time he drew or passed, none of the tiles he held at that moment
matched an open end. Within a deal both sides play a fast look-ahead policy until the round ends;
with no draws left and few tiles in hand the round is solved exactly instead. The value of a finished
round is the chance of winning the match from the resulting score (see match.py), or plain points
when the match score is unknown.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .game import ME, OPP, Board, Rules, pips, round_up5
from .match import MatchTable

EXACT_TILES = 9          # solve exactly when no draws are possible and both hands hold at most this many


@dataclass
class Info:
    """What I know at my turn."""
    board: Board
    my_hand: list
    opp_count: int
    unknown: list                          # tiles neither on the board nor in my hand
    opp_forbidden: list = field(default_factory=list)   # per opponent tile: values it cannot contain
    my_score: int | None = None
    opp_score: int | None = None
    target: int | None = None

    @property
    def boneyard(self) -> int:
        return len(self.unknown) - self.opp_count

    def check(self) -> list[str]:
        errs = []
        if self.boneyard < 0:
            errs.append(f"opponent holds {self.opp_count} but only {len(self.unknown)} tiles are unseen")
        if len(self.opp_forbidden) not in (0, self.opp_count):
            errs.append("opponent constraints out of sync")
        return errs


@dataclass
class MoveResult:
    move: tuple
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    pts: float = 0.0           # my round points minus his, summed
    win_round: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def se(self) -> float:
        if self.n < 2:
            return float("inf")
        var = max(0.0, self.total_sq / self.n - self.mean ** 2)
        return math.sqrt(var / (self.n - 1))

    @property
    def mean_pts(self) -> float:
        return self.pts / self.n if self.n else 0.0

    @property
    def round_win(self) -> float:
        return self.win_round / self.n if self.n else 0.0


# ---------------------------------------------------------------- sampling
def sample_deal(info: Info, rng: random.Random, tries: int = 60):
    """Opponent hand + boneyard order consistent with his draws and passes (None if impossible)."""
    pool = list(info.unknown)
    k = info.opp_count
    forb = list(info.opp_forbidden) if len(info.opp_forbidden) == k else [frozenset()] * k
    order = sorted(range(k), key=lambda i: -len(forb[i]))
    for _ in range(tries):
        rest = pool[:]
        hand = []
        ok = True
        for i in order:
            f = forb[i]
            allowed = [j for j, t in enumerate(rest) if t[0] not in f and t[1] not in f] if f else None
            if allowed is not None and not allowed:
                ok = False
                break
            j = rng.choice(allowed) if allowed is not None else rng.randrange(len(rest))
            hand.append(rest.pop(j))
        if ok:
            rng.shuffle(rest)
            return hand, rest
    return None


# ---------------------------------------------------------------- playing policy
def choose(board: Board, hand, other, moves, rng: random.Random | None = None):
    """Fast look-ahead: points now, minus the best reply he can score, plus shedding heavy tiles."""
    if len(moves) == 1:
        return moves[0]
    best, best_v = moves[0], -1e9
    last = len(hand) == 1
    for mv in moves:
        t, arm = mv
        b2 = board.place(t, arm)
        v = b2.points() + 0.12 * (t[0] + t[1]) + (0.6 if t[0] == t[1] else 0.0)
        if last:
            v += 100
        else:
            reply = 0
            for t2, a2 in b2.moves(other):
                p = b2.place(t2, a2).points()
                if p > reply:
                    reply = p
            v -= 0.8 * reply
        if rng is not None:
            v += rng.random() * 0.05
        if v > best_v:
            best, best_v = mv, v
    return best


def play_out(board: Board, hands, bone, to_move: int, gained, rules: Rules, rng=None, passes: int = 0):
    """Plays the round to the end in place. Returns the round winner (ME/OPP, None if blocked and tied)."""
    reserve = rules.boneyard_reserve
    while True:
        h = hands[to_move]
        mv = board.moves(h)
        while not mv and len(bone) > reserve:
            h.append(bone.pop())
            mv = board.moves(h)
        if not mv:
            passes += 1
            if passes >= 2:
                return _blocked(hands, gained)
            to_move ^= 1
            continue
        passes = 0
        t, arm = choose(board, h, hands[to_move ^ 1], mv, rng)
        h.remove(t)
        board = board.place(t, arm)
        gained[to_move] += board.points()
        if not h:
            gained[to_move] += round_up5(pips(hands[to_move ^ 1]))
            return to_move
        to_move ^= 1


def _blocked(hands, gained):
    a, b = pips(hands[ME]), pips(hands[OPP])
    if a == b:
        return None
    w = ME if a < b else OPP
    gained[w] += round_up5(pips(hands[w ^ 1]))
    return w


# ---------------------------------------------------------------- exact endgame (no draws left)
class Exact:
    def __init__(self, value_fn):
        self.value_fn = value_fn            # (gained_me, gained_opp, winner) -> value for me
        self.memo: dict = {}

    def solve(self, board: Board, hands, to_move: int, g0: int, g1: int, passes: int = 0) -> float:
        key = (board, frozenset(hands[0]), frozenset(hands[1]), to_move, g0, g1, passes)
        v = self.memo.get(key)
        if v is not None:
            return v
        h = hands[to_move]
        mv = board.moves(h)
        if not mv:
            if passes >= 1:
                gained = [g0, g1]
                w = _blocked(hands, gained)
                v = self.value_fn(gained[0], gained[1], w)
            else:
                v = self.solve(board, hands, to_move ^ 1, g0, g1, 1)
            self.memo[key] = v
            return v
        best = -1.0 if to_move == ME else 2.0
        for t, arm in mv:
            b2 = board.place(t, arm)
            p = b2.points()
            n0, n1 = (g0 + p, g1) if to_move == ME else (g0, g1 + p)
            rest = [x for x in h if x != t]
            if not rest:
                bonus = round_up5(pips(hands[to_move ^ 1]))
                if to_move == ME:
                    v = self.value_fn(n0 + bonus, n1, ME)
                else:
                    v = self.value_fn(n0, n1 + bonus, OPP)
            else:
                nh = [rest, hands[1]] if to_move == ME else [hands[0], rest]
                v = self.solve(b2, nh, to_move ^ 1, n0, n1, 0)
            if to_move == ME:
                best = max(best, v)
            else:
                best = min(best, v)
        self.memo[key] = best
        return best


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, rules: Rules, time_budget_s: float = 2.0, max_samples: int = 20000, seed: int | None = None):
        self.rules = rules
        self.budget = time_budget_s
        self.max_samples = max_samples
        self.rng = random.Random(seed)
        self.tables: dict[int, MatchTable] = {}
        self.last_deals = 0
        self.last_exact = 0
        self.last_unique = 0
        self.constraints_dropped = False

    def table(self, target: int) -> MatchTable:
        if target not in self.tables:
            self.tables[target] = MatchTable(target)
        return self.tables[target]

    def value_fn(self, info: Info):
        """Maps (round points me, round points opp, round winner) to a value for me in [0, 1]."""
        if info.my_score is None or info.opp_score is None or not info.target:
            def pts(g0, g1, _w):
                return 0.5 + max(-0.5, min(0.5, (g0 - g1) / 200.0))
            return pts
        tab = self.table(info.target)
        a, b = info.my_score, info.opp_score

        def match(g0, g1, w):
            return tab.after_round(a + g0, b + g1, ME if w is None else w)
        return match

    def evaluate(self, info: Info, moves, budget_s: float | None = None) -> list[MoveResult]:
        rules = self.rules
        budget = self.budget if budget_s is None else budget_s
        vf = self.value_fn(info)
        results = [MoveResult(m) for m in moves]
        t_end = time.time() + budget
        exact = Exact(vf)
        self.last_deals = 0
        self.last_exact = 0
        no_draws = not rules.can_draw(info.boneyard)
        seen = set()
        self.constraints_dropped = False
        while self.last_deals < self.max_samples and time.time() < t_end:
            deal = sample_deal(info, self.rng)
            if deal is None:
                if self.last_deals or not info.opp_forbidden:
                    break
                info = Info(info.board, info.my_hand, info.opp_count, info.unknown, [], info.my_score,
                            info.opp_score, info.target)
                self.constraints_dropped = True
                continue
            opp_hand, bone = deal
            self.last_deals += 1
            key = (frozenset(opp_hand), tuple(bone))
            seen.add(key)
            use_exact = no_draws and len(info.my_hand) <= EXACT_TILES and len(opp_hand) <= EXACT_TILES
            if use_exact:
                self.last_exact += 1
            for r in results:
                t, arm = r.move
                board = info.board.place(t, arm)
                p = board.points()
                mine = [x for x in info.my_hand if x != t]
                if not mine:
                    g = [p + round_up5(pips(opp_hand)), 0]
                    w = ME
                    v = vf(g[0], g[1], w)
                elif use_exact:
                    v = exact.solve(board, [mine, list(opp_hand)], OPP, p, 0)
                    g, w = None, None
                else:
                    g = [p, 0]
                    w = play_out(board, [mine, list(opp_hand)], list(bone), OPP, g, rules, self.rng)
                    v = vf(g[0], g[1], w)
                r.n += 1
                r.total += v
                r.total_sq += v * v
                if g is not None:
                    r.pts += g[0] - g[1]
                    r.win_round += 1.0 if w == ME else 0.0
                else:
                    r.win_round += v
            if use_exact and len(seen) >= self._deal_space(info):
                break
        self.last_unique = len(seen)
        results.sort(key=lambda r: -r.mean)
        return results

    @staticmethod
    def _deal_space(info: Info) -> int:
        n, k = len(info.unknown), info.opp_count
        return math.comb(n, k) * math.factorial(n - k)
