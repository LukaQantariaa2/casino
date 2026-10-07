"""Decision engine: evaluates every legal action from the information I have.

Method
------
* Hidden information (opponent hand, deck order, unrevealed pile cards) is
  sampled uniformly from the cards I have not seen, respecting every known
  constraint (sizes, trump card position, known opponent cards).
* While the deck still has cards each sample is played out with fast
  heuristic policies (Perfect-Information Monte Carlo).  The same samples are
  reused for every action (common random numbers) and weak actions are
  dropped early (successive halving) so the time budget goes to the close
  calls.
* Once the deck is empty every sample is a perfect-information end game; it
  is then solved exactly with minimax + memoisation, so late-round play has
  no approximation beyond the hidden-card sampling (and none at all when the
  opponent's hand is fully known).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..rules import Rules
from .core import ME, OPP, Ctx, SimState, apply_trick
from .policy import response_policy, rollout


@dataclass
class InfoState:
    """Everything *I* know about the current round."""
    rules: Rules
    trump_suit: int
    my_hand: list[int]
    opp_hand_size: int
    deck_size: int                       # cards still in the stock, trump card included
    trump_card: int | None = None        # the face-up trump card
    trump_card_in_deck: bool = True      # still lying under the stock
    opp_known: list[int] = field(default_factory=list)   # cards known to be in opp hand
    my_pile_known: list[int] = field(default_factory=list)
    my_pile_unknown: int = 0
    opp_pile_known: list[int] = field(default_factory=list)
    opp_pile_unknown: int = 0
    my_tricks: int = 0
    opp_tricks: int = 0
    leader: int = ME                     # who leads the current trick
    current_lead: list[int] = field(default_factory=list)  # set when I must respond
    out_of_play_unknown: int = 0         # unknown cards removed for other reasons

    def unseen(self) -> list[int]:
        seen = set(self.my_hand) | set(self.opp_known) | set(self.my_pile_known) | set(self.opp_pile_known) \
            | set(self.current_lead)
        if self.trump_card is not None and self.trump_card_in_deck:
            seen.add(self.trump_card)
        return [c for c in range(self.rules.n_cards) if c not in seen]

    def hidden_slots(self) -> int:
        deck_unknown = self.deck_size - (1 if (self.trump_card is not None and self.trump_card_in_deck
                                               and self.deck_size > 0) else 0)
        return (self.opp_hand_size - len(self.opp_known)) + deck_unknown + self.my_pile_unknown \
            + self.opp_pile_unknown + self.out_of_play_unknown

    def check(self) -> list[str]:
        """Consistency problems (empty list = consistent)."""
        errs = []
        u = self.unseen()
        if len(u) != self.hidden_slots():
            errs.append(f"card count mismatch: {len(u)} unseen cards vs {self.hidden_slots()} hidden slots")
        all_known = self.my_hand + self.opp_known + self.my_pile_known + self.opp_pile_known + self.current_lead
        if len(all_known) != len(set(all_known)):
            errs.append("a card is recorded in two places")
        if len(self.my_hand) > self.rules.hand_size or self.opp_hand_size > self.rules.hand_size:
            errs.append("hand larger than hand_size")
        if self.trump_card is not None and self.trump_card_in_deck and self.trump_card in all_known:
            errs.append("trump card is marked as in the deck but also seen elsewhere")
        return errs

    @property
    def my_points_known(self) -> int:
        return sum(self.rules.points[c] for c in self.my_pile_known)

    @property
    def opp_points_known(self) -> int:
        return sum(self.rules.points[c] for c in self.opp_pile_known)


@dataclass
class Sample:
    opp_hand: list[int]
    deck: list[int]
    my_pile_extra: int
    opp_pile_extra: int


def draw_sample(info: InfoState, rng: random.Random) -> Sample:
    pool = info.unseen()
    rng.shuffle(pool)
    i = 0
    n_opp = info.opp_hand_size - len(info.opp_known)
    opp = info.opp_known + pool[i:i + n_opp]
    i += n_opp
    trump_bottom = info.trump_card is not None and info.trump_card_in_deck and info.deck_size > 0
    n_deck = info.deck_size - (1 if trump_bottom else 0)
    deck = pool[i:i + n_deck]
    i += n_deck
    if trump_bottom:
        deck = [info.trump_card] + deck     # index 0 = drawn last
    pts = info.rules.points
    my_extra = sum(pts[c] for c in pool[i:i + info.my_pile_unknown])
    i += info.my_pile_unknown
    opp_extra = sum(pts[c] for c in pool[i:i + info.opp_pile_unknown])
    return Sample(opp, deck, my_extra, opp_extra)


def build_state(info: InfoState, smp: Sample) -> SimState:
    return SimState([list(info.my_hand), list(smp.opp_hand)], list(smp.deck),
                    [info.my_points_known + smp.my_pile_extra, info.opp_points_known + smp.opp_pile_extra],
                    [info.my_tricks, info.opp_tricks], info.leader)


MARGIN_EPS = 0.001


def outcome(s: SimState) -> float:
    """Result (+1 win / -1 loss / 0 draw, times multiplier) plus a tiny point-margin tie-breaker."""
    return s.result + MARGIN_EPS * (s.pts[0] - s.pts[1]) / 120.0


# ---------------------------------------------------------------- exact solver
class EndgameSolver:
    """Exact minimax for perfect-information positions with an empty stock."""

    def __init__(self, ctx: Ctx, node_limit: int = 200_000):
        self.ctx = ctx
        self.memo: dict = {}
        self.nodes = 0
        self.node_limit = node_limit

    def value(self, s: SimState) -> float:
        if s.done:
            return outcome(s)
        key = (tuple(sorted(s.hands[0])), tuple(sorted(s.hands[1])), s.leader, s.pts[0], s.pts[1],
               s.tricks[0] > 0, s.tricks[1] > 0)
        v = self.memo.get(key)
        if v is not None:
            return v
        self.nodes += 1
        if self.nodes > self.node_limit:
            raise TimeoutError
        ctx = self.ctx
        leader = s.leader
        best_lead = -math.inf if leader == ME else math.inf
        for lead in ctx.legal_leads(s.hands[leader]):
            worst = math.inf if leader == ME else -math.inf
            for resp in ctx.legal_responses(s.hands[1 - leader], lead):
                c = s.copy()
                apply_trick(ctx, c, leader, lead, resp)
                val = self.value(c)
                if leader == ME:
                    worst = min(worst, val)
                    if worst <= best_lead:
                        break
                else:
                    worst = max(worst, val)
                    if worst >= best_lead:
                        break
            if leader == ME:
                best_lead = max(best_lead, worst)
            else:
                best_lead = min(best_lead, worst)
        self.memo[key] = best_lead
        return best_lead


# ---------------------------------------------------------------- evaluation
@dataclass
class ActionResult:
    kind: str                    # lead | respond | claim | continue | accept | decline | raise | fold
    cards: tuple[int, ...] = ()
    n: int = 0
    win: float = 0.0
    draw: float = 0.0
    lose: float = 0.0
    value: float = 0.0           # expected result in stake units (+1 win / -1 loss)
    exact: bool = False
    note: str = ""

    @property
    def win_prob(self) -> float:
        return self.win / self.n if self.n else 0.0

    @property
    def lose_prob(self) -> float:
        return self.lose / self.n if self.n else 0.0

    @property
    def mean(self) -> float:
        return self.value / self.n if self.n else 0.0


class DecisionEngine:
    def __init__(self, rules: Rules, time_budget_s: float = 1.2, max_samples: int = 4000,
                 seed: int | None = None, opponent_model: str = "heuristic"):
        self.rules = rules
        self.time_budget = time_budget_s
        self.max_samples = max_samples
        self.rng = random.Random(seed)
        self.opponent_model = opponent_model

    # -- one playout of an action on a sample
    def _play(self, ctx: Ctx, info: InfoState, smp: Sample, action: tuple[int, ...], solver: EndgameSolver | None,
              rng: random.Random) -> float:
        s = build_state(info, smp)
        if info.leader == ME:
            if self.opponent_model == "lookahead" or (not s.deck and solver is not None):
                worst = math.inf
                for resp in ctx.legal_responses(s.hands[OPP], action):
                    c = s.copy()
                    apply_trick(ctx, c, ME, action, resp)
                    v = self._continue(ctx, c, solver, rng)
                    worst = min(worst, v)
                return worst
            resp = response_policy(ctx, s, OPP, action, rng)
            apply_trick(ctx, s, ME, action, resp)
        else:
            # the lead is already on the table (removed from opp hand by the tracker)
            s.hands[OPP] = s.hands[OPP] + list(info.current_lead)
            apply_trick(ctx, s, OPP, tuple(info.current_lead), action)
        return self._continue(ctx, s, solver, rng)

    def _continue(self, ctx: Ctx, s: SimState, solver: EndgameSolver | None, rng: random.Random) -> float:
        if s.done:
            return outcome(s)
        if not s.deck and solver is not None:
            try:
                return solver.value(s)
            except TimeoutError:
                pass
        rollout(ctx, s, rng)
        return outcome(s)

    def legal_actions(self, info: InfoState) -> list[tuple[int, ...]]:
        ctx = Ctx(self.rules, info.trump_suit)
        if info.leader == ME and not info.current_lead:
            return ctx.legal_leads(info.my_hand)
        return ctx.legal_responses(info.my_hand, info.current_lead)

    def evaluate(self, info: InfoState, deadline_s: float | None = None) -> list[ActionResult]:
        ctx = Ctx(self.rules, info.trump_suit)
        actions = self.legal_actions(info)
        kind = "lead" if (info.leader == ME and not info.current_lead) else "respond"
        results = {a: ActionResult(kind, a) for a in actions}
        if not actions:
            return []
        t_end = time.perf_counter() + (deadline_s if deadline_s is not None else self.time_budget)
        exact_world = (info.deck_size == 0 and len(info.unseen()) == info.opp_hand_size - len(info.opp_known)
                       and info.my_pile_unknown == 0 and info.opp_pile_unknown == 0)
        solver = EndgameSolver(ctx) if info.deck_size == 0 else None
        alive = list(actions)
        batch = 16
        n_samples = 0
        # exact case: one deterministic world, evaluate each action once
        if exact_world:
            smp = draw_sample(info, self.rng)
            for a in actions:
                v = self._play(ctx, info, smp, a, solver, self.rng)
                r = results[a]
                r.n, r.value, r.exact = 1, v, True
                r.win, r.lose, r.draw = (1, 0, 0) if v > 0.5 else ((0, 1, 0) if v < -0.5 else (0, 0, 1))
            return sorted(results.values(), key=lambda r: -r.mean)
        while n_samples < self.max_samples:
            samples = [draw_sample(info, self.rng) for _ in range(batch)]
            for smp in samples:
                seed = self.rng.random()
                for a in alive:
                    v = self._play(ctx, info, smp, a, solver, random.Random(seed))
                    r = results[a]
                    r.n += 1
                    r.value += v
                    if v > 0.5:
                        r.win += 1
                    elif v < -0.5:
                        r.lose += 1
                    else:
                        r.draw += 1
                if solver is not None and len(solver.memo) > 400_000:
                    solver.memo.clear()
            n_samples += batch
            if time.perf_counter() > t_end:
                break
            # successive halving: drop actions clearly worse than the leader
            if len(alive) > 2 and n_samples >= 64:
                ranked = sorted(alive, key=lambda a: -results[a].mean)
                best = results[ranked[0]]
                keep = []
                for a in ranked:
                    r = results[a]
                    se = math.sqrt(max(1e-9, (1.0 - r.mean ** 2)) / max(1, r.n)) * 2.5
                    if r.mean + se >= best.mean or len(keep) < 2:
                        keep.append(a)
                alive = keep
        out = sorted(results.values(), key=lambda r: (-(r.mean if r.n else -9), -r.n))
        return out

    # ---------------------------------------------------------------- utility decisions
    def win_probability(self, info: InfoState, budget_s: float | None = None) -> tuple[float, float, float]:
        """(P win, P draw, P lose) assuming best play from here, via the best action."""
        if info.leader != ME and not info.current_lead:
            return self._position_value(info, budget_s)
        res = self.evaluate(info, budget_s)
        if not res:
            return (0.0, 0.0, 0.0)
        b = res[0]
        n = max(1, b.n)
        return (b.win / n, b.draw / n, b.lose / n)

    def _position_value(self, info: InfoState, budget_s: float | None) -> tuple[float, float, float]:
        """Value of a position where the opponent is about to lead."""
        ctx = Ctx(self.rules, info.trump_suit)
        solver = EndgameSolver(ctx) if info.deck_size == 0 else None
        t_end = time.perf_counter() + (budget_s if budget_s is not None else self.time_budget)
        w = d = l = n = 0
        while n < self.max_samples and (n < 32 or time.perf_counter() < t_end):
            s = build_state(info, draw_sample(info, self.rng))
            v = self._continue(ctx, s, solver, self.rng)
            n += 1
            if v > 0.5:
                w += 1
            elif v < -0.5:
                l += 1
            else:
                d += 1
        return (w / n, d / n, l / n)

    def claim_probability(self, info: InfoState, samples: int = 4000) -> float:
        """Probability that my pile already holds >= win_points."""
        need = self.rules.win_points - info.my_points_known
        if need <= 0:
            return 1.0
        if info.my_pile_unknown == 0:
            return 0.0
        hit = 0
        for _ in range(samples):
            smp = draw_sample(info, self.rng)
            if smp.my_pile_extra >= need:
                hit += 1
        return hit / samples
