"""Measure how much the decision engine gains over a typical player.

    python tools/simulate_edge.py --rounds 400 --budget 0.05

Plays full rounds between
  A = the assistant's decision engine (sees only what a real player sees), and
  B = a heuristic "human-like" player,
alternating who leads first, and reports A's round win rate with a 95% interval.
Useful for the question "does a tool like this give a measurable edge?".
Claims are made automatically when a player knows he has enough points; stakes are ignored.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bura.engine.core import ME, OPP, Ctx, SimState, apply_trick  # noqa: E402
from bura.engine.policy import fast_lead, fast_response  # noqa: E402
from bura.engine.search import DecisionEngine, InfoState  # noqa: E402
from bura.rules import Rules  # noqa: E402


def info_for_A(rules, ctx, s: SimState, trump_card, trump_taken_by, piles, current_lead=None, leader=ME):
    trump_in_deck = bool(s.deck) and s.deck[0] == trump_card
    opp_known = [trump_card] if trump_taken_by == OPP and trump_card in s.hands[OPP] else []
    return InfoState(rules, ctx.trump, list(s.hands[ME]), len(s.hands[OPP]) - (0 if current_lead is None else 0),
                     len(s.deck), trump_card=trump_card, trump_card_in_deck=trump_in_deck, opp_known=opp_known,
                     my_pile_known=list(piles[ME]), opp_pile_known=list(piles[OPP]),
                     my_tricks=s.tricks[ME], opp_tricks=s.tricks[OPP], leader=leader,
                     current_lead=list(current_lead or []))


def play_round(rules, engine, rng, a_leads_first: bool) -> float:
    deck = list(range(rules.n_cards))
    rng.shuffle(deck)
    hs = rules.hand_size
    h0 = [deck.pop() for _ in range(hs)]
    h1 = [deck.pop() for _ in range(hs)]
    trump_card = deck[0]                     # bottom card, drawn last
    ctx = Ctx(rules, rules.suit_of[trump_card])
    s = SimState([h0, h1], deck, [0, 0], [0, 0], ME if a_leads_first else OPP)
    piles = {ME: [], OPP: []}
    trump_taken_by = None
    while not s.done:
        leader = s.leader
        if leader == ME:
            info = info_for_A(rules, ctx, s, trump_card, trump_taken_by, piles, leader=ME)
            lead = engine.evaluate(info)[0].cards
            resp = fast_response(ctx, s, OPP, lead)
        else:
            lead = fast_lead(ctx, s, OPP)
            # A sees the lead; the lead is no longer in B's hand
            s.hands[OPP] = [c for c in s.hands[OPP] if c not in lead]
            info = info_for_A(rules, ctx, s, trump_card, trump_taken_by, piles, current_lead=lead, leader=OPP)
            resp = engine.evaluate(info)[0].cards
            s.hands[OPP] += list(lead)
        before = (len(s.deck), s.deck[0] if s.deck else None)
        winner_cards = list(lead) + list(resp)
        w = apply_trick(ctx, s, leader, tuple(lead), tuple(resp))
        piles[w] += winner_cards
        if before[1] == trump_card and (not s.deck or s.deck[0] != trump_card):
            trump_taken_by = ME if trump_card in s.hands[ME] else OPP
    return s.result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=os.path.join(ROOT, "config", "rules", "bura5_adjarabet.json"))
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--budget", type=float, default=0.05, help="engine seconds per decision")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rules = Rules.load(args.rules)
    engine = DecisionEngine(rules, time_budget_s=args.budget, max_samples=2000, seed=args.seed)
    rng = random.Random(args.seed)
    wins = losses = draws = 0
    t0 = time.time()
    for i in range(args.rounds):
        r = play_round(rules, engine, rng, a_leads_first=(i % 2 == 0))
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1
        if (i + 1) % 25 == 0:
            n = i + 1
            print(f"{n:5d} rounds: engine wins {wins}, loses {losses}, draws {draws}  "
                  f"({100 * wins / n:.1f}% wins)  [{time.time() - t0:.0f}s]")
    n = args.rounds
    p = wins / n
    ci = 1.96 * math.sqrt(p * (1 - p) / n)
    ev = (wins - losses) / n
    print(f"\nEngine vs heuristic player over {n} rounds: win rate {100 * p:.1f}% ± {100 * ci:.1f}% (95% CI), "
          f"draws {draws}, expected result per round {ev:+.3f} stake units")


if __name__ == "__main__":
    main()
