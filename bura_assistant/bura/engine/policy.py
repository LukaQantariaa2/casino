"""Fast heuristic playout policies (used inside Monte Carlo simulations).

The policies only look at the acting player's own hand plus public info,
which keeps playouts realistic and cheap.
"""
from __future__ import annotations

import random
from typing import Sequence

from .core import Ctx, SimState, apply_trick


def _cheap_key(ctx: Ctx, c: int):
    return (ctx.suit[c] == ctx.trump, ctx.points[c], ctx.strength[c])


def lead_policy(ctx: Ctx, s: SimState, p: int, rng: random.Random) -> tuple[int, ...]:
    hand = s.hands[p]
    non_trump = [c for c in hand if ctx.suit[c] != ctx.trump]
    if not non_trump:
        return (min(hand, key=lambda c: ctx.strength[c]),)
    # multi-card lead of a same-suit non-trump group sometimes
    if ctx.lead_max > 1 and rng.random() < 0.3:
        groups: dict[int, list[int]] = {}
        for c in non_trump:
            groups.setdefault(ctx.suit[c], []).append(c)
        best = max(groups.values(), key=len)
        if len(best) >= 2:
            k = min(len(best), ctx.lead_max)
            return tuple(sorted(best, key=lambda c: -ctx.strength[c])[:k])
    # strong single: an ace of a side suit is often a good lead late in the round
    if not s.deck and rng.random() < 0.5:
        return (max(non_trump, key=lambda c: (ctx.strength[c], ctx.points[c])),)
    return (min(non_trump, key=lambda c: (ctx.points[c], ctx.strength[c])),)


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
    if not ctx.responder_wins(chosen, lead):  # greedy can fail; verify
        return None
    return chosen


def response_policy(ctx: Ctx, s: SimState, p: int, lead: Sequence[int], rng: random.Random) -> tuple[int, ...]:
    hand = s.hands[p]
    k = len(lead)
    if len(hand) <= k:
        return tuple(hand)
    beat = cheapest_beat(ctx, hand, lead)
    if beat is not None:
        trick_pts = sum(ctx.points[c] for c in lead) + sum(ctx.points[c] for c in beat)
        trumps_used = [c for c in beat if ctx.suit[c] == ctx.trump]
        high_trump_pts = sum(ctx.points[c] for c in trumps_used)
        reaches_win = s.pts[p] + trick_pts >= ctx.win_points
        if (reaches_win or not trumps_used or trick_pts >= 10
                or (trick_pts >= 4 and high_trump_pts < 10) or not ctx.beat_optional
                or rng.random() < 0.15):
            return tuple(beat)
    if not ctx.beat_optional:
        opts = ctx.legal_responses(hand, lead)
        return opts[0]
    return tuple(sorted(hand, key=lambda c: _cheap_key(ctx, c))[:k])


def rollout(ctx: Ctx, s: SimState, rng: random.Random, max_tricks: int = 60) -> float:
    n = 0
    while not s.done and n < max_tricks:
        leader = s.leader
        if not s.hands[leader]:
            # should not happen; treat as end of cards
            from .core import end_by_cards
            end_by_cards(ctx, s)
            break
        lead = lead_policy(ctx, s, leader, rng)
        resp = response_policy(ctx, s, 1 - leader, lead, rng)
        apply_trick(ctx, s, leader, lead, resp)
        n += 1
    return s.result
