"""Perfect-information game mechanics used by the search.

Player 0 is always "me", player 1 is the opponent.  Cards are integer ids
(see ``Rules``).  Everything here is deliberately plain Python with small
lists so the Monte Carlo search can run thousands of playouts per second.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

from ..rules import Rules

ME, OPP = 0, 1


class Ctx:
    """Rule tables flattened for speed, bound to one trump suit."""

    __slots__ = ("rules", "trump", "suit", "strength", "points", "hand_size", "win_points",
                 "lead_min", "lead_max", "same_suit", "mixed_trumps", "beat_optional",
                 "all_must_beat", "winner_draws_first", "winner_leads", "tie_result",
                 "out_rule", "claim_enabled", "zero_mult", "trump_beats", "specials", "draw_alternate",
                 "_rw_cache", "move_cache")

    def __init__(self, rules: Rules, trump_suit: int):
        self.rules = rules
        self.trump = trump_suit
        self.suit = rules.suit_of
        self.strength = rules.strength
        self.points = rules.points
        self.hand_size = rules.hand_size
        self.win_points = rules.win_points
        lead = rules.raw["lead"]
        self.lead_min = lead["min_cards"]
        self.lead_max = lead["max_cards"]
        self.same_suit = lead["same_suit_required"]
        self.mixed_trumps = lead["allow_mixed_suits_if_all_trumps"]
        resp = rules.raw["response"]
        self.beat_optional = resp["beating_is_optional"]
        self.all_must_beat = resp["responder_wins_only_if_all_cards_beaten"]
        self.winner_draws_first = rules.raw["draw"]["trick_winner_draws_first"]
        self.winner_leads = rules.raw["draw"]["trick_winner_leads_next"]
        self.draw_alternate = rules.raw["draw"].get("mode", "alternate_one_by_one") == "alternate_one_by_one"
        sc = rules.raw["scoring"]
        self.tie_result = sc["tie_result"]
        self.out_rule = sc["when_cards_run_out"]
        self.claim_enabled = sc["claim_enabled"]
        self.zero_mult = sc.get("loser_with_zero_tricks_multiplier", 1)
        self.trump_beats = rules.raw["trump"]["trump_beats_any_non_trump"]
        self.specials = rules.specials
        self._rw_cache: dict = {}
        self.move_cache: dict = {}     # per-trump caches for solvers (legal moves by hand)

    # ----- card relations -----
    def beats(self, b: int, a: int) -> bool:
        sb, sa = self.suit[b], self.suit[a]
        if sb == sa:
            return self.strength[b] > self.strength[a]
        return sb == self.trump and self.trump_beats

    def beat_assignment(self, response: Sequence[int], lead: Sequence[int]) -> list[tuple[int, int]] | None:
        """Return [(lead_card, response_card), ...] if every lead card is beaten, else None."""
        k = len(lead)
        if len(response) != k:
            return None
        # fast path for one card
        if k == 1:
            return [(lead[0], response[0])] if self.beats(response[0], lead[0]) else None
        # backtracking matching (k <= 5)
        used = [False] * k
        pairs: list[tuple[int, int]] = []
        order = sorted(range(k), key=lambda i: -self.strength[lead[i]] - (100 if self.suit[lead[i]] == self.trump else 0))

        def rec(pos: int) -> bool:
            if pos == k:
                return True
            a = lead[order[pos]]
            for j in range(k):
                if not used[j] and self.beats(response[j], a):
                    used[j] = True
                    pairs.append((a, response[j]))
                    if rec(pos + 1):
                        return True
                    pairs.pop()
                    used[j] = False
            return False

        return list(pairs) if rec(0) else None

    def responder_wins(self, response: Sequence[int], lead: Sequence[int]) -> bool:
        if len(lead) == 1:
            return len(response) == 1 and self.beats(response[0], lead[0])
        key = (tuple(response), tuple(lead))
        v = self._rw_cache.get(key)
        if v is None:
            v = self._rw_cache[key] = self.beat_assignment(response, lead) is not None
        return v

    # ----- legal actions -----
    def legal_leads(self, hand: Sequence[int]) -> list[tuple[int, ...]]:
        hand = sorted(hand)
        out: list[tuple[int, ...]] = []
        hi = min(self.lead_max, len(hand))
        lo = min(self.lead_min, len(hand))
        if self.same_suit:
            by_suit: dict[int, list[int]] = {}
            for c in hand:
                by_suit.setdefault(self.suit[c], []).append(c)
            for cards in by_suit.values():
                for k in range(lo, min(hi, len(cards)) + 1):
                    out.extend(combinations(cards, k))
            if self.mixed_trumps:
                pass  # all-trump leads are already same-suit
        else:
            for k in range(lo, hi + 1):
                out.extend(combinations(hand, k))
        return out

    def legal_responses(self, hand: Sequence[int], lead: Sequence[int]) -> list[tuple[int, ...]]:
        k = len(lead)
        hand = sorted(hand)
        if len(hand) < k:
            return [tuple(hand)]
        allr = list(combinations(hand, k))
        if self.beat_optional:
            return allr
        winning = [r for r in allr if self.responder_wins(r, lead)]
        return winning or allr

    # ----- specials -----
    def hand_special(self, h: Sequence[int]) -> str | None:
        """Name of the instant-win combination ``h`` holds (e.g. Bura), or None."""
        if not self.specials or not h:
            return None
        for sp in self.specials:
            if sp.ctype == "trumps_in_hand":
                if sum(1 for c in h if self.suit[c] == self.trump) >= sp.count:
                    return sp.name
            elif sp.ctype == "same_rank_in_hand":
                rk = self.rules.ranks.index(sp.rank) if sp.rank in self.rules.ranks else -1
                if sum(1 for c in h if self.rules.rank_of[c] == rk) >= sp.count:
                    return sp.name
            elif sp.ctype == "same_suit_in_hand":
                counts: dict[int, int] = {}
                for c in h:
                    counts[self.suit[c]] = counts.get(self.suit[c], 0) + 1
                if counts and max(counts.values()) >= sp.count:
                    return sp.name
        return None

    def special_winner(self, hands: Sequence[Sequence[int]], first: int) -> int | None:
        if not self.specials:
            return None
        for p in (first, 1 - first):
            if self.hand_special(hands[p]) is not None:
                return p
        return None


class SimState:
    """A fully determined position (one sample of the hidden information)."""

    __slots__ = ("hands", "deck", "pts", "tricks", "leader", "done", "result")

    def __init__(self, hands, deck, pts, tricks, leader):
        self.hands = hands          # [list, list]
        self.deck = deck            # list; draw from the end, last element = next card
        self.pts = pts              # [int, int]
        self.tricks = tricks        # [int, int] tricks won (for zero-trick multiplier)
        self.leader = leader
        self.done = False
        self.result = 0.0           # +1 I win, -1 I lose, 0 draw (times multiplier)

    def copy(self) -> "SimState":
        s = SimState([self.hands[0][:], self.hands[1][:]], self.deck[:], self.pts[:], self.tricks[:], self.leader)
        s.done, s.result = self.done, self.result
        return s


def finish(ctx: Ctx, s: SimState, winner: int | None) -> None:
    s.done = True
    if winner is None:
        s.result = 0.0
        return
    loser = 1 - winner
    mult = ctx.zero_mult if s.tricks[loser] == 0 else 1
    s.result = float(mult) if winner == ME else -float(mult)


def end_by_cards(ctx: Ctx, s: SimState) -> None:
    a, b = s.pts
    if a == b:
        finish(ctx, s, None)
    elif ctx.out_rule == "higher_points_wins":
        finish(ctx, s, ME if a > b else OPP)
    else:  # "threshold_or_draw": only a player with >= win_points wins
        if a >= ctx.win_points:
            finish(ctx, s, ME)
        elif b >= ctx.win_points:
            finish(ctx, s, OPP)
        else:
            finish(ctx, s, None)


def apply_trick(ctx: Ctx, s: SimState, leader: int, lead: Sequence[int], response: Sequence[int],
                claim_policy=None) -> int:
    """Play a full trick (lead + response), resolve, claim check, draw. Returns the trick winner."""
    hl, hr = s.hands[leader], s.hands[1 - leader]
    for c in lead:
        hl.remove(c)
    for c in response:
        hr.remove(c)
    winner = 1 - leader if ctx.responder_wins(response, lead) else leader
    gained = 0
    for c in lead:
        gained += ctx.points[c]
    for c in response:
        gained += ctx.points[c]
    s.pts[winner] += gained
    s.tricks[winner] += 1
    # claim check (players know their own pile exactly in a determinized world)
    if ctx.claim_enabled and s.pts[winner] >= ctx.win_points:
        if claim_policy is None or claim_policy(s, winner):
            finish(ctx, s, winner)
            return winner
    drew = draw_cards(ctx, s, winner)
    s.leader = winner if ctx.winner_leads else 1 - leader
    if not s.hands[0] and not s.hands[1] and not s.deck:
        end_by_cards(ctx, s)
        return winner
    # specials (Bura ...) need a full hand, which only a draw can produce
    if drew and ctx.specials:
        sw = ctx.special_winner(s.hands, winner)
        if sw is not None:
            finish(ctx, s, sw)
    return winner


def draw_cards(ctx: Ctx, s: SimState, winner: int) -> bool:
    """Refill both hands from the stock. Returns True if any card was drawn."""
    if not s.deck:
        return False
    order = (winner, 1 - winner) if ctx.winner_draws_first else (1 - winner, winner)
    need = [ctx.hand_size - len(s.hands[0]), ctx.hand_size - len(s.hands[1])]
    if ctx.draw_alternate:
        # one card at a time, first player first
        while s.deck and (need[0] > 0 or need[1] > 0):
            for p in order:
                if need[p] > 0 and s.deck:
                    s.hands[p].append(s.deck.pop())
                    need[p] -= 1
    else:
        for p in order:
            while need[p] > 0 and s.deck:
                s.hands[p].append(s.deck.pop())
                need[p] -= 1
    return True


def cards_points(ctx: Ctx, cards: Iterable[int]) -> int:
    return sum(ctx.points[c] for c in cards)
