"""Decision engine: evaluates every legal action from the information I have.

Method
------
* Hidden cards are sampled from the posterior P(hand | observed opponent
  plays), not the uniform prior.  When the number of unknown opponent cards
  is small enough the posterior is enumerated; otherwise samples are drawn
  uniformly and reweighted (importance sampling).
* Observed leads are scored with a softmax Bura lead model.  Face-down dumps
  after the stock is empty down-weight worlds where the remaining hand still
  holds a cheap beat.
* The opponent's reply to *my* candidate action is a 1-ply greedy best
  response.  Further tricks use greedy play while the stock is short, then a
  fast heuristic.
* With at most EXACT_DECK_LIMIT cards in the stock every sampled world is
  solved with minimax + memoisation; all actions are screened with fast
  playouts first and only the best EXACT_MAX_ACTIONS are solved exactly.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Sequence

from itertools import combinations, permutations

from ..rules import Rules
from .core import ME, OPP, Ctx, SimState, apply_trick
from .policy import (GREEDY_DECK_LIMIT, fast_lead, fast_response, greedy_lead, greedy_response,
                     sample_weight)

POSTERIOR_ENUM_LIMIT = 12_000
EXACT_DECK_LIMIT = 6          # stock size from which a sampled world is solved exactly
EXACT_NODE_LIMIT = 30_000     # per-sample minimax budget before falling back to greedy play
EXACT_MAX_ACTIONS = 5         # with cards in the stock only this many screened candidates are solved exactly
SCREEN_FRACTION = 0.3         # share of the time budget spent screening with fast playouts
SMALL_WORLD_LIMIT = 120       # with at most this many possible deals every action is solved in every deal
PARALLEL_OVERHEAD_S = 0.08    # time kept back for sending work to the worker processes and merging
PARALLEL_ROUND_S = 0.4        # worker results are merged (and bad actions dropped) this often


@dataclass
class OppEvidence:
    """An opponent action that constrains the remaining-card posterior."""
    kind: str  # dump
    lead: tuple[int, ...]
    stock_empty: bool = False


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
    opp_evidence: list[OppEvidence] = field(default_factory=list)

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
    weight: float = 1.0
    from_posterior: bool = False


def _assemble_sample(info: InfoState, rng: random.Random, opp_unknown: Sequence[int]) -> Sample:
    used = set(opp_unknown)
    rest = [c for c in info.unseen() if c not in used]
    rng.shuffle(rest)
    trump_bottom = info.trump_card is not None and info.trump_card_in_deck and info.deck_size > 0
    n_deck = info.deck_size - (1 if trump_bottom else 0)
    i = 0
    deck = rest[i:i + n_deck]
    i += n_deck
    if trump_bottom:
        deck = [info.trump_card] + deck
    pts = info.rules.points
    my_extra = sum(pts[c] for c in rest[i:i + info.my_pile_unknown])
    i += info.my_pile_unknown
    opp_extra = sum(pts[c] for c in rest[i:i + info.opp_pile_unknown])
    return Sample(list(info.opp_known) + list(opp_unknown), deck, my_extra, opp_extra)


def draw_sample(info: InfoState, rng: random.Random) -> Sample:
    pool = info.unseen()
    rng.shuffle(pool)
    n_opp = info.opp_hand_size - len(info.opp_known)
    return _assemble_sample(info, rng, pool[:n_opp])


def _posterior_table(ctx: Ctx, info: InfoState) -> tuple[list[tuple[int, ...]], list[float]] | None:
    """Enumerate unknown opponent cards and P(plays | those cards), or None if too large."""
    k = info.opp_hand_size - len(info.opp_known)
    pool = info.unseen()
    if k < 0 or k > len(pool):
        return None
    if k == 0:
        return [()], [1.0]
    ncomb = math.comb(len(pool), k)
    if ncomb > POSTERIOR_ENUM_LIMIT:
        return None
    assigns = list(combinations(pool, k))
    weights = []
    for extra in assigns:
        dummy = Sample(list(info.opp_known) + list(extra), [], 0, 0)
        weights.append(max(1e-12, sample_weight(ctx, info, dummy)))
    return assigns, weights


def _world_shape(info: InfoState) -> tuple[int, int, int, bool]:
    """(unknown opponent cards, unknown stock cards, cards left for the unseen piles, trump at the bottom)."""
    k = info.opp_hand_size - len(info.opp_known)
    trump_bottom = info.trump_card is not None and info.trump_card_in_deck and info.deck_size > 0
    n_deck = info.deck_size - (1 if trump_bottom else 0)
    return k, n_deck, len(info.unseen()) - k - n_deck, trump_bottom


def _deals_per_hand(info: InfoState) -> int:
    """Stock orders x unseen pile contents, for one opponent hand."""
    k, n_deck, left, _ = _world_shape(info)
    mu, ou = info.my_pile_unknown, info.opp_pile_unknown
    if k < 0 or n_deck < 0 or left < mu + ou:
        return 0
    return math.perm(n_deck + left, n_deck) * math.comb(left, mu) * math.comb(left - mu, ou)


def count_worlds(info: InfoState) -> int:
    """Number of deals consistent with what I have seen (before weighting by the opponent's play)."""
    k, _, _, _ = _world_shape(info)
    n_unseen = len(info.unseen())
    if k < 0 or k > n_unseen:
        return 0
    return math.comb(n_unseen, k) * _deals_per_hand(info)


def world_key(smp: Sample) -> tuple:
    return tuple(sorted(smp.opp_hand)), tuple(smp.deck), smp.my_pile_extra, smp.opp_pile_extra


def enumerate_worlds(info: InfoState, table: tuple[list[tuple[int, ...]], list[float]] | None,
                     limit: int) -> list[Sample] | None:
    """Every possible deal (opponent hand, stock order, unseen pile cards) with its posterior
    weight, or None if there are more than ``limit``."""
    if table is None:
        return None
    assigns, weights = table
    per = _deals_per_hand(info)
    if per <= 0 or len(assigns) * per > limit:
        return None
    _, n_deck, _, trump_bottom = _world_shape(info)
    unseen = info.unseen()
    pts = info.rules.points
    mu, ou = info.my_pile_unknown, info.opp_pile_unknown
    out = []
    for extra, w in zip(assigns, weights):
        used = set(extra)
        rest = [c for c in unseen if c not in used]
        for stock in permutations(rest, n_deck):
            left = [c for c in rest if c not in stock]
            deck = ([info.trump_card] if trump_bottom else []) + list(stock)
            for mine in combinations(left, mu):
                left2 = [c for c in left if c not in mine]
                for theirs in combinations(left2, ou):
                    out.append(Sample(list(info.opp_known) + list(extra), deck,
                                      sum(pts[c] for c in mine), sum(pts[c] for c in theirs),
                                      weight=w / per, from_posterior=True))
    return out


def draw_posterior_sample(ctx: Ctx, info: InfoState, rng: random.Random,
                          table: tuple[list[tuple[int, ...]], list[float]] | None) -> Sample:
    if table is None:
        smp = draw_sample(info, rng)
        smp.weight = sample_weight(ctx, info, smp)
        return smp
    assigns, weights = table
    extra = rng.choices(assigns, weights=weights, k=1)[0]
    smp = _assemble_sample(info, rng, extra)
    smp.weight = 1.0
    smp.from_posterior = True
    return smp


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
    """Exact minimax for perfect-information positions.

    With an empty stock one instance can be shared by every sample.  With
    cards still in the stock the deck order is part of the position, so an
    instance must only be reused while the deck contents are the same (i.e.
    within one sample, across the candidate actions).
    """

    def __init__(self, ctx: Ctx, node_limit: int = 200_000):
        self.ctx = ctx
        self.memo: dict = {}
        self.nodes = 0
        self.node_limit = node_limit

    def _leads(self, hand: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Legal leads, those carrying points first (near 61 they are decisive and cut off early)."""
        cache = self.ctx.move_cache
        key = ("L", hand)
        v = cache.get(key)
        if v is None:
            pts = self.ctx.points
            v = cache[key] = sorted(self.ctx.legal_leads(hand), key=lambda l: -sum(pts[c] for c in l))
        return v

    def _responses(self, hand: tuple[int, ...], lead: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Legal replies: winning (beating) replies first, cheapest first; then dumps, cheapest first."""
        cache = self.ctx.move_cache
        key = (hand, lead)
        v = cache.get(key)
        if v is None:
            ctx = self.ctx
            pts = ctx.points
            v = cache[key] = sorted(ctx.legal_responses(hand, lead),
                                    key=lambda r: (0 if ctx.takes_trick(r, lead) else 1, sum(pts[c] for c in r)))
        return v

    def value(self, s: SimState) -> float:
        """Exact game value in {-mult, 0, +mult}; no point-margin tie-breaker so equal
        values cut off (ternary payoffs make this a near-complete alpha-beta)."""
        if s.done:
            return s.result
        h0, h1 = tuple(sorted(s.hands[0])), tuple(sorted(s.hands[1]))
        key = (h0, h1, s.leader, s.pts[0], s.pts[1], s.tricks[0] > 0, s.tricks[1] > 0, len(s.deck))
        v = self.memo.get(key)
        if v is not None:
            return v
        self.nodes += 1
        if self.nodes > self.node_limit:
            raise TimeoutError
        ctx = self.ctx
        leader = s.leader
        leads = self._leads(h0 if leader == ME else h1)
        if leader == ME:
            best_lead = -math.inf
            for lead in leads:
                worst = math.inf
                for resp in self._responses(h1, lead):
                    c = s.copy()
                    apply_trick(ctx, c, ME, lead, resp)
                    val = self.value(c)
                    if val < worst:
                        worst = val
                        if worst <= best_lead or worst <= -1.0:
                            break
                if worst > best_lead:
                    best_lead = worst
                    if best_lead >= 1.0:
                        break
        else:
            best_lead = math.inf
            for lead in leads:
                worst = -math.inf
                for resp in self._responses(h0, lead):
                    c = s.copy()
                    apply_trick(ctx, c, OPP, lead, resp)
                    val = self.value(c)
                    if val > worst:
                        worst = val
                        if worst >= best_lead or worst >= 1.0:
                            break
                if worst < best_lead:
                    best_lead = worst
                    if best_lead <= -1.0:
                        break
        self.memo[key] = best_lead
        return best_lead


# ---------------------------------------------------------------- evaluation
@dataclass
class ActionResult:
    kind: str                    # lead | respond | claim | continue | accept | decline | raise | fold
    cards: tuple[int, ...] = ()
    n: int = 0
    wsum: float = 0.0
    w2sum: float = 0.0           # sum of squared weights (for the effective sample size)
    win: float = 0.0
    draw: float = 0.0
    lose: float = 0.0
    value: float = 0.0           # weighted sum of results (+1 win / -1 loss)
    exact: bool = False
    worlds: int = 0              # > 0: every possible deal was solved (no sampling error)
    note: str = ""

    @property
    def _norm(self) -> float:
        return self.wsum if self.wsum else float(self.n)

    @property
    def n_eff(self) -> float:
        """Kish effective sample size of the (importance-)weighted samples."""
        if self.exact or self.worlds:
            return float("inf")
        if self.w2sum <= 0:
            return float(self.n)
        return self.wsum * self.wsum / self.w2sum

    @property
    def win_prob(self) -> float:
        n = self._norm
        return self.win / n if n else 0.0

    @property
    def lose_prob(self) -> float:
        n = self._norm
        return self.lose / n if n else 0.0

    @property
    def mean(self) -> float:
        n = self._norm
        return self.value / n if n else 0.0

    @property
    def mean_se(self) -> float:
        """Standard error of ``mean`` (results are in [-1, 1])."""
        if self.exact or self.worlds:
            return 0.0
        ne = max(1.0, self.n_eff)
        return math.sqrt(max(0.0, 1.0 - self.mean ** 2) / ne)

    @property
    def win_se(self) -> float:
        """Standard error of ``win_prob``."""
        if self.exact or self.worlds:
            return 0.0
        p = self.win_prob
        return math.sqrt(p * (1.0 - p) / max(1.0, self.n_eff))


MIN_ESS_FRACTION = 0.25   # below this, importance weights are tempered toward the uniform prior


class DecisionEngine:
    def __init__(self, rules: Rules, time_budget_s: float = 1.2, max_samples: int = 4000,
                 seed: int | None = None, opponent_model: str = "lookahead"):
        self.rules = rules
        self.time_budget = time_budget_s
        self.max_samples = max_samples
        self.fixed_seed = seed
        self.rng = random.Random(seed)
        self.opponent_model = opponent_model
        self.last_ess_fraction: float = 1.0   # diagnostics of the last evaluate()
        self.last_tempered: bool = False
        self.last_checked: int | None = None  # distinct deals calculated by the last search (None: no search)
        self.last_exhaustive: bool = False    # the last search covered every possible deal
        self._seen_worlds: set = set()
        self._pool = None                     # ProcessPoolExecutor once start_workers() succeeded
        self.workers = 1

    def _reseed(self, info: InfoState) -> None:
        """Deterministic RNG per position: re-evaluating an unchanged state gives the same answer."""
        if self.fixed_seed is not None:
            self.rng = random.Random(self.fixed_seed)
            return
        key = (tuple(sorted(info.my_hand)), tuple(sorted(info.opp_known)), tuple(sorted(info.my_pile_known)),
               tuple(sorted(info.opp_pile_known)), info.my_pile_unknown, info.opp_pile_unknown,
               info.opp_hand_size, info.deck_size, info.trump_suit, info.trump_card, info.trump_card_in_deck,
               info.leader, tuple(info.current_lead), info.my_tricks, info.opp_tricks,
               tuple((ev.kind, ev.lead, ev.stock_empty) for ev in info.opp_evidence))
        self.rng = random.Random(hash(key))

    @staticmethod
    def _exact_deck(info: InfoState) -> int:
        """Stock size from which sampled worlds are solved exactly for this root position."""
        return EXACT_DECK_LIMIT if info.deck_size <= EXACT_DECK_LIMIT else 0

    def _opponent_response(self, ctx: Ctx, s: SimState, action: tuple[int, ...],
                           solver: EndgameSolver, exact_deck: int) -> float:
        """Value of ``action`` after the opponent replies in this determined world."""
        if len(s.deck) <= exact_deck:
            worst = math.inf
            for resp in ctx.legal_responses(s.hands[OPP], action):
                c = s.copy()
                apply_trick(ctx, c, ME, action, resp)
                v = self._continue(ctx, c, solver, exact_deck)
                worst = min(worst, v)
                if worst <= -1.0 + 1e-9:
                    break
            return worst
        if self.opponent_model == "heuristic":
            resp = fast_response(ctx, s, OPP, action)
        else:
            # 1-ply: opponent minimises static value, then we roll the chosen line out.
            resp = greedy_response(ctx, s, ME, action)
        apply_trick(ctx, s, ME, action, resp)
        return self._continue(ctx, s, solver, exact_deck)

    # -- one playout of an action on a sample
    def _play(self, ctx: Ctx, info: InfoState, smp: Sample, action: tuple[int, ...], solver: EndgameSolver,
              exact_deck: int) -> float:
        s = build_state(info, smp)
        if info.leader == ME:
            if ctx.hand_special(s.hands[ME]) is not None:
                return 1.0 + MARGIN_EPS      # Bura in hand: the round is already won
            return self._opponent_response(ctx, s, action, solver, exact_deck)
        # the lead is already on the table (removed from opp hand by the tracker)
        s.hands[OPP] = s.hands[OPP] + list(info.current_lead)
        apply_trick(ctx, s, OPP, tuple(info.current_lead), action)
        return self._continue(ctx, s, solver, exact_deck)

    def _continue(self, ctx: Ctx, s: SimState, solver: EndgameSolver, exact_deck: int) -> float:
        """Play ``s`` to the end: exact minimax once the stock is down to ``exact_deck`` cards,
        greedy 1-ply play while the stock is short, fast heuristics before that."""
        greedy_left = 1 if self.opponent_model != "heuristic" else 0
        exact_ok = True
        n = 0
        while not s.done and n < 60:
            if exact_ok and len(s.deck) <= exact_deck:
                try:
                    return solver.value(s)
                except TimeoutError:
                    exact_ok = False
            leader = s.leader
            if not s.hands[leader]:
                from .core import end_by_cards
                end_by_cards(ctx, s)
                break
            if greedy_left > 0 or len(s.deck) <= GREEDY_DECK_LIMIT:
                lead = greedy_lead(ctx, s, leader)
                resp = greedy_response(ctx, s, leader, lead)
                greedy_left -= 1
            else:
                lead = fast_lead(ctx, s, leader)
                resp = fast_response(ctx, s, 1 - leader, lead)
            apply_trick(ctx, s, leader, lead, resp)
            n += 1
        return outcome(s)

    def legal_actions(self, info: InfoState) -> list[tuple[int, ...]]:
        ctx = Ctx(self.rules, info.trump_suit)
        if info.leader == ME and not info.current_lead:
            return ctx.legal_leads(info.my_hand)
        return ctx.legal_responses(info.my_hand, info.current_lead)

    def _prepare_posterior(self, ctx: Ctx, info: InfoState):
        """Posterior table (or None for importance sampling) plus a weight exponent.

        If the posterior is so concentrated that the effective sample size would fall below
        ``MIN_ESS_FRACTION`` of the draws, the weights are tempered (w ** 0.5) toward the prior:
        a collapsed posterior means the opponent model is being trusted more than it deserves.
        """
        table = _posterior_table(ctx, info)
        if table is not None:
            ws = table[1]
        else:
            ws = [sample_weight(ctx, info, draw_sample(info, self.rng)) for _ in range(128)]
            ws = [w for w in ws if w > 0.0] or [1.0]
        ess = sum(ws) ** 2 / max(1e-12, sum(w * w for w in ws))
        frac = ess / len(ws)
        alpha = 1.0
        if frac < MIN_ESS_FRACTION:
            alpha = 0.5
            if table is not None:
                table = (table[0], [w ** alpha for w in table[1]])
                ws2 = table[1]
                frac = (sum(ws2) ** 2 / max(1e-12, sum(w * w for w in ws2))) / len(ws2)
            else:
                ws2 = [w ** alpha for w in ws]
                frac = (sum(ws2) ** 2 / max(1e-12, sum(w * w for w in ws2))) / len(ws2)
        self.last_ess_fraction = frac
        self.last_tempered = alpha != 1.0
        return table, alpha

    def evaluate(self, info: InfoState, deadline_s: float | None = None) -> list[ActionResult]:
        self._seen_worlds = set()
        res = self._evaluate(info, deadline_s)
        if res:
            b = res[0]
            if b.exact or b.worlds:
                self.last_checked, self.last_exhaustive = (1 if b.exact else b.worlds), True
            else:
                self.last_checked, self.last_exhaustive = len(self._seen_worlds), False
        return res

    def _evaluate(self, info: InfoState, deadline_s: float | None) -> list[ActionResult]:
        ctx = Ctx(self.rules, info.trump_suit)
        actions = self.legal_actions(info)
        kind = "lead" if (info.leader == ME and not info.current_lead) else "respond"
        results = {a: ActionResult(kind, a) for a in actions}
        if not actions:
            return []
        self._reseed(info)
        t_end = time.perf_counter() + (deadline_s if deadline_s is not None else self.time_budget)
        exact_world = (info.deck_size == 0 and len(info.unseen()) == info.opp_hand_size - len(info.opp_known)
                       and info.my_pile_unknown == 0 and info.opp_pile_unknown == 0)
        # With an empty stock one solver serves every sample; otherwise the deck order differs per
        # sample, so a fresh solver is created for each sample and shared by its candidate actions.
        exact_deck = self._exact_deck(info)
        shared_solver = EndgameSolver(ctx) if info.deck_size == 0 else None
        # exact case: one deterministic world, evaluate each action once
        if exact_world:
            self.last_ess_fraction, self.last_tempered = 1.0, False
            smp = draw_sample(info, self.rng)
            for a in actions:
                v = self._play(ctx, info, smp, a, shared_solver, exact_deck)
                r = results[a]
                r.n, r.wsum, r.value, r.exact = 1, 1.0, v, True
                r.win, r.lose, r.draw = (1, 0, 0) if v > 0.5 else ((0, 1, 0) if v < -0.5 else (0, 0, 1))
            return sorted(results.values(), key=lambda r: -r.mean)
        table, alpha = self._prepare_posterior(ctx, info)

        def rank(rs):
            return sorted(rs, key=lambda r: (-(r.mean if r.n else -9), -r.n))

        worlds = enumerate_worlds(info, table, SMALL_WORLD_LIMIT) if exact_deck else None
        if worlds:
            for smp in worlds:
                solver = shared_solver if shared_solver is not None \
                    else EndgameSolver(ctx, node_limit=EXACT_NODE_LIMIT)
                for a in actions:
                    self._record(results[a], smp.weight, self._play(ctx, info, smp, a, solver, exact_deck))
            for r in results.values():
                r.worlds = len(worlds)
            return rank(results.values())

        if exact_deck and info.deck_size > 0 and len(actions) > EXACT_MAX_ACTIONS:
            # Exact solving with cards in the stock is expensive: screen every action with fast
            # playouts first, then spend the rest of the budget solving the best few exactly.
            budget = deadline_s if deadline_s is not None else self.time_budget
            t_screen = time.perf_counter() + SCREEN_FRACTION * budget
            self._run_loop(ctx, info, results, list(actions), table, alpha, 0, None, t_screen)
            screened = rank(results.values())
            finalists = [r.cards for r in screened[:EXACT_MAX_ACTIONS]]
            exact_results = {a: ActionResult(kind, a) for a in finalists}
            self._seen_worlds = set()      # count only the deals the finalists were solved on
            self._run_loop(ctx, info, exact_results, finalists, table, alpha, exact_deck, None, t_end)
            return rank(exact_results.values()) + screened[EXACT_MAX_ACTIONS:]
        self._run_loop(ctx, info, results, list(actions), table, alpha, exact_deck, shared_solver, t_end)
        return rank(results.values())

    @staticmethod
    def _record(r: ActionResult, w: float, v: float) -> None:
        r.n += 1
        r.wsum += w
        r.w2sum += w * w
        r.value += w * v
        if v > 0.5:
            r.win += w
        elif v < -0.5:
            r.lose += w
        else:
            r.draw += w

    def _sample_loop(self, ctx: Ctx, info: InfoState, results: dict, alive: list, table, alpha: float,
                     exact_deck: int, shared_solver: EndgameSolver | None, t_end: float,
                     halve: bool = True) -> None:
        """Evaluate ``alive`` actions on common posterior samples until ``t_end``, with successive halving.
        ``exact_deck`` 0 still solves the empty-stock part of each playout exactly."""
        batch = 16
        n_samples = 0
        while n_samples < self.max_samples:
            samples = [draw_posterior_sample(ctx, info, self.rng, table) for _ in range(batch)]
            for smp in samples:
                w = smp.weight
                if w <= 0.0:
                    continue          # impossible world (e.g. an undeclared Bura)
                if alpha != 1.0 and not smp.from_posterior:
                    w = w ** alpha
                self._seen_worlds.add(world_key(smp))
                solver = shared_solver if shared_solver is not None \
                    else EndgameSolver(ctx, node_limit=EXACT_NODE_LIMIT)
                for a in alive:
                    self._record(results[a], w, self._play(ctx, info, smp, a, solver, exact_deck))
                if shared_solver is not None and len(shared_solver.memo) > 400_000:
                    shared_solver.memo.clear()
                if time.perf_counter() > t_end:
                    return
            n_samples += batch
            if halve and len(alive) > 2 and n_samples >= 64:
                alive = self._halve(results, alive)

    # ---------------------------------------------------------------- utility decisions
    def win_probability(self, info: InfoState, budget_s: float | None = None) -> tuple[float, float, float]:
        """(P win, P draw, P lose) assuming best play from here, via the best action."""
        if info.leader != ME and not info.current_lead:
            return self._position_value(info, budget_s)
        res = self.evaluate(info, budget_s)
        if not res:
            return (0.0, 0.0, 0.0)
        b = res[0]
        n = max(1e-9, b._norm)
        return (b.win / n, b.draw / n, b.lose / n)

    def _position_value(self, info: InfoState, budget_s: float | None) -> tuple[float, float, float]:
        """Value of a position where the opponent is about to lead."""
        ctx = Ctx(self.rules, info.trump_suit)
        self._reseed(info)
        exact_deck = self._exact_deck(info)
        shared_solver = EndgameSolver(ctx) if info.deck_size == 0 else None
        t_end = time.perf_counter() + (budget_s if budget_s is not None else self.time_budget)
        table, alpha = self._prepare_posterior(ctx, info)
        worlds = enumerate_worlds(info, table, SMALL_WORLD_LIMIT) if exact_deck else None
        if worlds:
            acc = [0.0, 0.0, 0.0, 0.0]
            for smp in worlds:
                self._value_add(ctx, info, smp, smp.weight, shared_solver, exact_deck, acc)
            self.last_checked, self.last_exhaustive = len(worlds), True
        elif self._pool is not None:
            acc, seen = self._parallel_value(info, table, alpha, exact_deck, t_end)
            self.last_checked, self.last_exhaustive = len(seen), False
        else:
            acc, seen = self._value_loop(ctx, info, table, alpha, exact_deck, shared_solver, t_end, self.max_samples)
            self.last_checked, self.last_exhaustive = len(seen), False
        w, d, l, mass = acc
        if mass <= 0:
            return (0.0, 0.0, 0.0)
        return (w / mass, d / mass, l / mass)

    def _value_add(self, ctx: Ctx, info: InfoState, smp: Sample, wt: float, shared_solver: EndgameSolver | None,
                   exact_deck: int, acc: list[float]) -> None:
        """Play ``smp`` out from the current position; acc = [win, draw, lose, total weight]."""
        solver = shared_solver if shared_solver is not None else EndgameSolver(ctx, node_limit=EXACT_NODE_LIMIT)
        v = self._continue(ctx, build_state(info, smp), solver, exact_deck)
        acc[3] += wt
        acc[0 if v > 0.5 else (2 if v < -0.5 else 1)] += wt

    def _value_loop(self, ctx: Ctx, info: InfoState, table, alpha: float, exact_deck: int,
                    shared_solver: EndgameSolver | None, t_end: float, max_n: int) -> tuple[list[float], set]:
        acc = [0.0, 0.0, 0.0, 0.0]
        seen: set = set()
        n = 0
        while n < max_n and (n < 32 or time.perf_counter() < t_end):
            smp = draw_posterior_sample(ctx, info, self.rng, table)
            n += 1
            wt = smp.weight
            if wt <= 0.0:
                continue
            if alpha != 1.0 and not smp.from_posterior:
                wt = wt ** alpha
            seen.add(world_key(smp))
            self._value_add(ctx, info, smp, wt, shared_solver, exact_deck, acc)
        return acc, seen

    # ---------------------------------------------------------------- worker processes
    def start_workers(self, n: int) -> int:
        """Spread sampled deals over ``n`` worker processes (n <= 1: stay single-process).
        Returns the number of workers actually running."""
        self.stop_workers()
        if n <= 1:
            return 1
        try:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=n, initializer=_worker_init,
                                       initargs=(self.rules, self.opponent_model))
            for f in [pool.submit(_worker_ping) for _ in range(n)]:
                f.result(timeout=120)
        except Exception:
            return 1
        self._pool, self.workers = pool, n
        return n

    def stop_workers(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self._pool, self.workers = None, 1

    def _worker_seconds(self, t_end: float) -> float:
        return max(0.05, t_end - time.perf_counter() - PARALLEL_OVERHEAD_S)

    def _run_loop(self, ctx: Ctx, info: InfoState, results: dict, alive: list, table, alpha: float,
                  exact_deck: int, shared_solver: EndgameSolver | None, t_end: float) -> None:
        """``_sample_loop`` on every worker at once (each on its own deals), results summed."""
        if self._pool is None:
            self._sample_loop(ctx, info, results, alive, table, alpha, exact_deck, shared_solver, t_end)
            return
        kind = next(iter(results.values())).kind
        alive = list(alive)
        done = 0
        # Short rounds, so that clearly worse actions are dropped across all workers' results together.
        while done < self.max_samples:
            secs = min(PARALLEL_ROUND_S, self._worker_seconds(t_end))
            per = math.ceil((self.max_samples - done) / self.workers)
            try:
                futs = [self._pool.submit(_worker_actions, info, alive, kind, table, alpha, exact_deck,
                                          shared_solver is not None, secs, self.rng.getrandbits(64), per)
                        for _ in range(self.workers)]
                outs = [f.result() for f in futs]
            except Exception:
                self.stop_workers()          # a broken pool: carry on in this process
                self._sample_loop(ctx, info, results, alive, table, alpha, exact_deck, shared_solver, t_end)
                return
            for rows, seen in outs:
                for cards, n, wsum, w2sum, win, draw, lose, value in rows:
                    r = results[cards]
                    r.n += n
                    r.wsum += wsum
                    r.w2sum += w2sum
                    r.win += win
                    r.draw += draw
                    r.lose += lose
                    r.value += value
                self._seen_worlds |= seen
            done = max(results[a].n for a in alive)
            if time.perf_counter() + 0.05 >= t_end - PARALLEL_OVERHEAD_S:
                return
            if len(alive) > 2 and min(results[a].n for a in alive) >= 64:
                alive = self._halve(results, alive)

    @staticmethod
    def _halve(results: dict, alive: list) -> list:
        """Successive halving: keep the actions that could still be the best."""
        ranked = sorted(alive, key=lambda a: -results[a].mean)
        best = results[ranked[0]]
        keep = []
        for a in ranked:
            r = results[a]
            if r.mean + 2.5 * r.mean_se >= best.mean or len(keep) < 2:
                keep.append(a)
        return keep

    def _parallel_value(self, info: InfoState, table, alpha: float, exact_deck: int,
                        t_end: float) -> tuple[list[float], set]:
        per = math.ceil(self.max_samples / self.workers)
        secs = self._worker_seconds(t_end)
        try:
            futs = [self._pool.submit(_worker_value, info, table, alpha, exact_deck, secs,
                                      self.rng.getrandbits(64), per) for _ in range(self.workers)]
            outs = [f.result() for f in futs]
        except Exception:
            self.stop_workers()
            ctx = Ctx(self.rules, info.trump_suit)
            shared = EndgameSolver(ctx) if info.deck_size == 0 else None
            return self._value_loop(ctx, info, table, alpha, exact_deck, shared, t_end, self.max_samples)
        acc = [0.0, 0.0, 0.0, 0.0]
        seen: set = set()
        for a, s in outs:
            for i in range(4):
                acc[i] += a[i]
            seen |= s
        return acc, seen

    def claim_probability(self, info: InfoState, samples: int = 4000) -> float:
        """Probability that my pile already holds >= win_points."""
        need = self.rules.win_points - info.my_points_known
        if need <= 0:
            return 1.0
        if info.my_pile_unknown == 0:
            return 0.0
        self._reseed(info)
        hit = 0
        for _ in range(samples):
            smp = draw_sample(info, self.rng)
            if smp.my_pile_extra >= need:
                hit += 1
        return hit / samples


# ---------------------------------------------------------------- worker-process entry points
_WORKER: DecisionEngine | None = None


def _worker_init(rules: Rules, opponent_model: str) -> None:
    global _WORKER
    _WORKER = DecisionEngine(rules, opponent_model=opponent_model)


def _worker_ping() -> bool:
    time.sleep(0.3)              # keeps this worker busy so the pool starts every process up front
    return _WORKER is not None


def _worker_actions(info: InfoState, actions: list, kind: str, table, alpha: float, exact_deck: int,
                    shared: bool, seconds: float, seed: int, max_samples: int):
    e = _WORKER
    e.rng = random.Random(seed)
    e.max_samples = max_samples
    e._seen_worlds = set()
    ctx = Ctx(e.rules, info.trump_suit)
    results = {a: ActionResult(kind, a) for a in actions}
    solver = EndgameSolver(ctx) if shared else None
    e._sample_loop(ctx, info, results, list(actions), table, alpha, exact_deck, solver,
                   time.perf_counter() + seconds, halve=False)
    rows = [(r.cards, r.n, r.wsum, r.w2sum, r.win, r.draw, r.lose, r.value) for r in results.values()]
    return rows, e._seen_worlds


def _worker_value(info: InfoState, table, alpha: float, exact_deck: int, seconds: float, seed: int,
                  max_samples: int):
    e = _WORKER
    e.rng = random.Random(seed)
    ctx = Ctx(e.rules, info.trump_suit)
    solver = EndgameSolver(ctx) if info.deck_size == 0 else None
    return e._value_loop(ctx, info, table, alpha, exact_deck, solver, time.perf_counter() + seconds, max_samples)
