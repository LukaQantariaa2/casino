"""Fast playout policies and a cheap position evaluator.

Rollouts use one-ply greedy play with ``static_eval`` (from each acting
player's point of view).  That is still an approximation, but it is
consistent with the search's payoff and much closer to best play than
random "dump low / sometimes smash" heuristics.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from .core import ME, OPP, Ctx, SimState, apply_trick


def _cheap_key(ctx: Ctx, c: int):
    return (ctx.suit[c] == ctx.trump, ctx.points[c], ctx.strength[c])


def _hand_power(ctx: Ctx, cards: Sequence[int]) -> float:
    p = 0.0
    for c in cards:
        trump = ctx.suit[c] == ctx.trump
        p += 0.035 * ctx.points[c]
        p += 0.025 * ctx.strength[c]
        if trump:
            p += 0.14 + 0.03 * ctx.strength[c]
    return p


def static_eval(ctx: Ctx, s: SimState) -> float:
    """Heuristic expected result for player 0, in roughly [-1, 1]."""
    if s.done:
        return s.result
    a, b = s.pts[0], s.pts[1]
    if a >= ctx.win_points and a > b:
        return 1.0
    if b >= ctx.win_points and b > a:
        return -1.0
    race = (a - b) / float(max(1, ctx.win_points))
    power = _hand_power(ctx, s.hands[0]) - _hand_power(ctx, s.hands[1])
    x = 1.7 * race + 1.0 * power
    return math.tanh(x)


def cheapest_beat(ctx: Ctx, hand: Sequence[int], lead: Sequence[int]) -> list[int] | None:
    remaining = list(hand)
    chosen: list[int] = []
    order = sorted(lead, key=lambda a: (ctx.suit[a] == ctx.trump, ctx.strength[a]), reverse=True)
    for a in order:
        cands = [b for b in remaining if ctx.beats(b, a)]
        if not cands:
            return None
        b = min(cands, key=lambda b: (ctx.suit[b] == ctx.trump, ctx.points[b], ctx.strength[b]))
        remaining.remove(b)
        chosen.append(b)
    if not ctx.responder_wins(chosen, lead):
        return None
    return chosen


def _dump(ctx: Ctx, hand: Sequence[int], k: int) -> tuple[int, ...]:
    return tuple(sorted(hand, key=lambda c: _cheap_key(ctx, c))[:k])


def greedy_response(ctx: Ctx, s: SimState, leader: int, lead: Sequence[int]) -> tuple[int, ...]:
    """Pick a response that is best for the responder under ``static_eval``."""
    responder = 1 - leader
    hand = s.hands[responder]
    k = len(lead)
    if len(hand) <= k:
        return tuple(hand)
    opts = ctx.legal_responses(hand, lead)
    if len(opts) == 1:
        return opts[0]
    # Candidates: the cheapest beat, the cheapest dump, and other *winning* replies (a dump that
    # gives away more points than the cheapest dump is dominated, so it is never considered).
    beat = cheapest_beat(ctx, hand, lead)
    dump = _dump(ctx, hand, k)
    ranked = []
    if beat is not None:
        ranked.append(tuple(beat))
    ranked.append(dump)
    if beat is not None:
        for o in opts:
            if o not in ranked and ctx.responder_wins(o, lead):
                ranked.append(o)
            if len(ranked) >= 7:
                break
    if len(ranked) == 1:
        return ranked[0]
    best_o = ranked[0]
    best_v = -math.inf if responder == ME else math.inf
    for resp in ranked:
        c = s.copy()
        apply_trick(ctx, c, leader, lead, resp)
        v = static_eval(ctx, c)
        if responder == ME:
            if v > best_v:
                best_v, best_o = v, resp
        elif v < best_v:
            best_v, best_o = v, resp
    return best_o


def greedy_lead(ctx: Ctx, s: SimState, p: int) -> tuple[int, ...]:
    """Pick a lead that is best for ``p`` under 1-ply ``static_eval``."""
    opts = ctx.legal_leads(s.hands[p])
    if not opts:
        return ()
    if len(opts) == 1:
        return opts[0]
    # Score a bounded set: cheapest non-trump singles, strongest late leads, legal pairs.
    hand = s.hands[p]
    non_trump = [c for c in hand if ctx.suit[c] != ctx.trump]
    prefer: list[tuple[int, ...]] = []
    if non_trump:
        prefer.append((min(non_trump, key=lambda c: (ctx.points[c], ctx.strength[c])),))
        prefer.append((max(non_trump, key=lambda c: (ctx.strength[c], ctx.points[c])),))
    else:
        prefer.append((min(hand, key=lambda c: ctx.strength[c]),))
    zeros: dict[int, list[int]] = {}
    for c in non_trump:
        if ctx.points[c] == 0:
            zeros.setdefault(ctx.suit[c], []).append(c)
    for group in zeros.values():
        if len(group) >= 2 and ctx.lead_max >= 2:
            prefer.append(tuple(sorted(group, key=lambda c: ctx.strength[c])[:2]))
    for o in opts:
        if o not in prefer:
            prefer.append(o)
        if len(prefer) >= 7:
            break
    best_o = prefer[0]
    best_v = -math.inf if p == ME else math.inf
    for lead in prefer:
        resp = greedy_response(ctx, s, p, lead)
        c = s.copy()
        apply_trick(ctx, c, p, lead, resp)
        v = static_eval(ctx, c)
        if p == ME:
            if v > best_v:
                best_v, best_o = v, lead
        elif v < best_v:
            best_v, best_o = v, lead
    return best_o


def fast_lead(ctx: Ctx, s: SimState, p: int) -> tuple[int, ...]:
    hand = s.hands[p]
    non_trump = [c for c in hand if ctx.suit[c] != ctx.trump]
    if not non_trump:
        return (min(hand, key=lambda c: ctx.strength[c]),)
    if not s.deck:
        return (max(non_trump, key=lambda c: (ctx.strength[c], ctx.points[c])),)
    return (min(non_trump, key=lambda c: (ctx.points[c], ctx.strength[c])),)


def fast_response(ctx: Ctx, s: SimState, p: int, lead: Sequence[int]) -> tuple[int, ...]:
    hand = s.hands[p]
    k = len(lead)
    if len(hand) <= k:
        return tuple(hand)
    beat = cheapest_beat(ctx, hand, lead)
    dump = _dump(ctx, hand, k)
    if beat is None:
        if not ctx.beat_optional:
            opts = ctx.legal_responses(hand, lead)
            return opts[0]
        return dump
    trick_pts = sum(ctx.points[c] for c in lead) + sum(ctx.points[c] for c in beat)
    if s.pts[p] + trick_pts >= ctx.win_points:
        return tuple(beat)
    dump_pts = sum(ctx.points[c] for c in lead) + sum(ctx.points[c] for c in dump)
    if s.pts[1 - p] + dump_pts >= ctx.win_points:
        return tuple(beat)
    trumps = [c for c in beat if ctx.suit[c] == ctx.trump]
    trump_pts = max((ctx.points[c] for c in trumps), default=0)
    if not trumps:
        return tuple(beat)
    if trick_pts >= 10 and trump_pts < 11:
        return tuple(beat)
    if trump_pts >= 10 and trick_pts < 10:
        return dump
    if trick_pts >= 4 and trump_pts <= 4:
        return tuple(beat)
    return dump


GREEDY_DECK_LIMIT = 8   # stock size from which playouts switch from fast heuristics to greedy play


def _use_greedy(s: SimState, ply: int, greedy_plies: int = 1) -> bool:
    return ply < greedy_plies or len(s.deck) <= GREEDY_DECK_LIMIT


def lead_policy(ctx: Ctx, s: SimState, p: int, rng: random.Random) -> tuple[int, ...]:
    _ = rng
    return greedy_lead(ctx, s, p) if _use_greedy(s, 0) else fast_lead(ctx, s, p)


def response_policy(ctx: Ctx, s: SimState, p: int, lead: Sequence[int], rng: random.Random) -> tuple[int, ...]:
    _ = rng
    if _use_greedy(s, 0):
        return greedy_response(ctx, s, 1 - p, lead)
    return fast_response(ctx, s, p, lead)


def lead_utility(ctx: Ctx, hand: Sequence[int], lead: Sequence[int]) -> float:
    """Relative attractiveness of a lead from ``hand`` (higher = more likely).

    Matches standard Bura: dump low side-suit cards, keep trump, cash winners
    only when the rest of the hand is also strong in that suit.
    """
    pts = sum(ctx.points[c] for c in lead)
    trumps = sum(1 for c in lead if ctx.suit[c] == ctx.trump)
    strength = max(ctx.strength[c] for c in lead)
    u = -2.4 * pts - 7.5 * trumps - 0.25 * strength
    if len(lead) > 1:
        if pts == 0 and trumps == 0:
            u += 0.8 * (len(lead) - 1)
        else:
            u -= 3.5 * (len(lead) - 1)
    if trumps == 0:
        suit = ctx.suit[lead[0]]
        held = sum(1 for c in hand if ctx.suit[c] == suit)
        u += 0.35 * held
    return u


# Calibration of the opponent model.  Humans lead aces and tens from ordinary hands all the
# time, so no observed play may make a world more than a few times less likely than another.
LEAD_TEMPERATURE = 14.0   # regret scale: the worst possible lead (~30 utility) is ~1/8 as likely
LEAD_FLOOR = 0.12         # minimum weight from a single observed lead
DUMP_FLOOR = 0.15         # minimum weight from a single observed dump
WEIGHT_FLOOR = 0.05       # minimum total weight of any consistent world (Bura excluded)


def lead_likelihood(ctx: Ctx, hand: Sequence[int], actual: Sequence[int],
                    temperature: float = LEAD_TEMPERATURE) -> float:
    """Relative likelihood of leading ``actual`` from ``hand`` (1 = the model's preferred lead).

    Regret model: ``exp(-(u_best - u_actual) / T)``, floored.  Only ratios between hands
    matter, so this is a likelihood up to a constant.
    """
    opts = ctx.legal_leads(hand)
    if not opts:
        return 1.0
    want = tuple(sorted(actual))
    scores = [lead_utility(ctx, hand, lead) for lead in opts]
    idx = next((i for i, lead in enumerate(opts) if tuple(sorted(lead)) == want), -1)
    if idx < 0:
        return LEAD_FLOOR
    regret = max(scores) - scores[idx]
    return max(LEAD_FLOOR, math.exp(-regret / max(0.5, temperature)))


def dump_weight(ctx: Ctx, hand: Sequence[int], lead: Sequence[int]) -> float:
    """How plausible a face-down give-up is if ``hand`` still holds these cards."""
    beat = cheapest_beat(ctx, hand, lead)
    if beat is None:
        return 1.0
    trick_pts = sum(ctx.points[c] for c in lead) + sum(ctx.points[c] for c in beat)
    trumps = [c for c in beat if ctx.suit[c] == ctx.trump]
    if not trumps:
        return DUMP_FLOOR            # declined a free trick: unusual but it happens
    if trick_pts >= 10:
        return 0.25
    if sum(ctx.points[c] for c in trumps) >= 10:
        return 0.6
    return 0.35


def sample_weight(ctx: Ctx, info, smp) -> float:
    """Likelihood of the opponent's observed plays given his sampled remaining hand (up to a
    constant).  Never below ``WEIGHT_FLOOR`` for a consistent world; exactly 0 only for worlds
    that are impossible (an undeclared Bura)."""
    w = 1.0
    full = list(info.current_lead) + list(smp.opp_hand) if (info.current_lead and info.leader != ME) \
        else list(smp.opp_hand)
    if len(full) != len(set(full)):
        return 0.0
    # A full hand that forms Bura (etc.) would already have been declared: such worlds are impossible.
    if len(full) >= ctx.hand_size and ctx.hand_special(full) is not None:
        return 0.0
    if info.current_lead and info.leader != ME:
        w *= lead_likelihood(ctx, full, info.current_lead)
    for ev in getattr(info, "opp_evidence", ()) or ():
        if ev.kind != "dump" or not ev.stock_empty:
            continue
        w *= dump_weight(ctx, smp.opp_hand, ev.lead)
    return max(WEIGHT_FLOOR, w)


def rollout(ctx: Ctx, s: SimState, rng: random.Random, max_tricks: int = 60,
            greedy_plies: int = 1) -> float:
    _ = rng
    n = 0
    while not s.done and n < max_tricks:
        leader = s.leader
        if not s.hands[leader]:
            from .core import end_by_cards
            end_by_cards(ctx, s)
            break
        if _use_greedy(s, n, greedy_plies):
            lead = greedy_lead(ctx, s, leader)
            resp = greedy_response(ctx, s, leader, lead)
        else:
            lead = fast_lead(ctx, s, leader)
            resp = fast_response(ctx, s, 1 - leader, lead)
        apply_trick(ctx, s, leader, lead, resp)
        n += 1
    return s.result
