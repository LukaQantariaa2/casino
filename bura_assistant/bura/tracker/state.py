"""Persistent game state for the current round + full history."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..engine.core import ME, OPP
from ..engine.search import InfoState
from ..rules import Rules


@dataclass
class Trick:
    leader: int
    lead: list[int] = field(default_factory=list)
    response: list[int] = field(default_factory=list)   # known response cards
    response_hidden: int = 0                             # face-down cards not (yet) revealed
    responded: bool = False
    face_down: bool | None = None                        # True = given up face down, False = beat attempt

    @property
    def k(self) -> int:
        return len(self.lead)


@dataclass
class Event:
    t: float
    kind: str
    text: str
    data: dict = field(default_factory=dict)


@dataclass
class RoundState:
    rules: Rules
    number: int = 0
    trump_card: int | None = None
    trump_suit: int | None = None
    my_hand: list[int] = field(default_factory=list)
    opp_hand_size: int = 0
    deck_size: int = 0
    trump_in_deck: bool = True
    opp_known: list[int] = field(default_factory=list)
    my_pile: list[int] = field(default_factory=list)
    my_pile_unknown: int = 0
    opp_pile: list[int] = field(default_factory=list)
    opp_pile_unknown: int = 0
    my_tricks: int = 0
    opp_tricks: int = 0
    trick: Trick | None = None
    next_leader: int | None = None
    last_trick_winner: int | None = None
    stake: int = 1
    stake_offered: int | None = None
    i_raised_last: bool = False
    finished: bool = False
    result: str = ""

    # ------------------------------------------------------------- derived
    def points(self, who: int) -> int:
        pile = self.my_pile if who == ME else self.opp_pile
        return sum(self.rules.points[c] for c in pile)

    def seen_cards(self) -> set[int]:
        s = set(self.my_hand) | set(self.opp_known) | set(self.my_pile) | set(self.opp_pile)
        if self.trick:
            s |= set(self.trick.lead) | set(self.trick.response)
        if self.trump_card is not None:
            s.add(self.trump_card)
        return s

    def unseen_cards(self) -> list[int]:
        seen = self.seen_cards()
        return [c for c in range(self.rules.n_cards) if c not in seen]

    def info(self) -> InfoState:
        """Engine view. When I must respond, the lead is on the table."""
        lead: list[int] = []
        leader = self.next_leader if self.next_leader is not None else ME
        if self.trick and not self.trick.responded:
            leader = self.trick.leader
            if self.trick.leader == OPP:
                lead = list(self.trick.lead)
        return InfoState(rules=self.rules, trump_suit=self.trump_suit if self.trump_suit is not None else 0,
                         my_hand=list(self.my_hand), opp_hand_size=self.opp_hand_size,
                         deck_size=self.deck_size, trump_card=self.trump_card,
                         trump_card_in_deck=self.trump_in_deck and self.deck_size > 0,
                         opp_known=[c for c in self.opp_known if c not in self.my_hand],
                         my_pile_known=list(self.my_pile), my_pile_unknown=self.my_pile_unknown,
                         opp_pile_known=list(self.opp_pile), opp_pile_unknown=self.opp_pile_unknown,
                         my_tricks=self.my_tricks, opp_tricks=self.opp_tricks, leader=leader,
                         current_lead=lead)

    def consistency_errors(self) -> list[str]:
        errs = []
        r = self.rules
        if self.trump_card is None:
            errs.append("trump card unknown")
        if self.deck_size < 0:
            errs.append("deck size negative")
        if self.my_pile_unknown < 0 or self.opp_pile_unknown < 0:
            errs.append("negative unknown pile count")
        on_table_hidden = self.trick.response_hidden if self.trick else 0
        total = (len(self.my_hand) + self.opp_hand_size + self.deck_size + len(self.my_pile) + self.my_pile_unknown
                 + len(self.opp_pile) + self.opp_pile_unknown)
        if self.trick:
            total += len(self.trick.lead) + len(self.trick.response) + on_table_hidden
            # an opponent lead/response is no longer in his hand; mine are no longer in my hand
        if total != r.n_cards:
            errs.append(f"card total is {total}, expected {r.n_cards}")
        known = self.my_hand + self.my_pile + self.opp_pile + (self.trick.lead + self.trick.response if self.trick else [])
        if len(known) != len(set(known)):
            dup = {r.card_name(c) for c in known if known.count(c) > 1}
            errs.append("card seen twice: " + " ".join(sorted(dup)))
        if self.trump_card is not None and self.trump_in_deck and self.deck_size > 0 and self.trump_card in known:
            errs.append("trump card is both under the deck and elsewhere")
        if len(self.my_hand) > r.hand_size or self.opp_hand_size > r.hand_size:
            errs.append("hand bigger than allowed")
        return errs


class History:
    def __init__(self, logfile=None):
        self.events: list[Event] = []
        self.logfile = logfile

    def add(self, kind: str, text: str, **data) -> Event:
        ev = Event(time.time(), kind, text, data)
        self.events.append(ev)
        if self.logfile is not None:
            try:
                import json
                self.logfile.write(json.dumps({"t": ev.t, "kind": kind, "text": text, **data},
                                              ensure_ascii=False) + "\n")
                self.logfile.flush()
            except Exception:
                pass
        return ev

    def tail(self, n: int = 12) -> list[Event]:
        return self.events[-n:]
