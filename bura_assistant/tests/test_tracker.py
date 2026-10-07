import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bura.engine.core import ME, OPP  # noqa: E402
from bura.rules import Rules  # noqa: E402
from bura.tracker.tracker import GameTracker, STATUS_OK, STATUS_PAUSED, PHASE_RESPOND, PHASE_LEAD  # noqa: E402
from bura.vision.recognizer import DetectedCard, Observation  # noqa: E402

R = Rules.load(os.path.join(ROOT, "config", "rules", "bura5_adjarabet.json"))
P = R.parse_cards
BLANK = np.zeros((34, 35), np.int16)


def ob(hand, opp, center=(), backs=0, trump="7D", deck=True, my_turn=False, opp_turn=False, buttons=(),
       my_pile=False, opp_pile=False, pile_me=None, pile_opp=None):
    o = Observation(t=0.0)
    o.my_hand = [DetectedCard(c, 0, 0, 0.99, 0.2) for c in P(hand)]
    o.opp_count = opp
    o.center = [DetectedCard(c, 0, 0, 0.99, 0.2) for c in P(" ".join(center))]
    o.center_backs = backs
    o.trump = DetectedCard(P(trump)[0], 0, 0, 0.99, 0.1) if trump else None
    o.deck_present = deck
    o.my_turn, o.opp_turn = my_turn, opp_turn
    o.buttons = set(buttons)
    o.my_pile_present, o.opp_pile_present = my_pile, opp_pile
    o.pile_sig = {"me": BLANK if pile_me is None else pile_me, "opp": BLANK if pile_opp is None else pile_opp}
    o.stake_badge = {"me": None, "opp": None}
    return o


CHANGED = np.full((34, 35), 120, np.int16)


class TrackerTests(unittest.TestCase):
    def test_full_trick_cycle(self):
        t = GameTracker(R)
        t.update(ob("6S AD 7H 10C KH", 5, my_turn=False, opp_turn=True))
        self.assertEqual(t.status.status, STATUS_OK)
        r = t.round
        self.assertEqual(r.deck_size, 26)
        # opponent leads 9C
        t.update(ob("6S AD 7H 10C KH", 4, center=["9C"], my_turn=True, buttons=["fold", "raise"]))
        self.assertEqual(t.status.phase, PHASE_RESPOND)
        info = r.info()
        self.assertEqual(info.current_lead, P("9C"))
        self.assertEqual(info.check(), [])
        # I beat with 10C
        t.update(ob("6S AD 7H KH", 4, center=["9C", "10C"]))
        # trick resolved, my pile badge changes
        t.update(ob("6S AD 7H KH", 4, my_pile=True, pile_me=CHANGED, buttons=["claim", "continue"],
                    my_turn=True))
        self.assertEqual(r.points(ME), 10)
        self.assertEqual(r.next_leader, ME)
        # draws: I get AC, opponent 1
        t.update(ob("6S AD 7H KH AC", 5, my_pile=True, pile_me=CHANGED, my_turn=True))
        self.assertEqual(r.deck_size, 24)
        self.assertEqual(t.status.phase, PHASE_LEAD)
        self.assertEqual(t.status.status, STATUS_OK)
        self.assertEqual(r.consistency_errors(), [])

    def test_responder_gives_up_with_beating_card(self):
        """Opponent may give a card that could beat; the pile badge decides the winner."""
        t = GameTracker(R)
        t.update(ob("6H 9D JD KC JS", 5, trump="KD", my_turn=True))
        t.update(ob("9D JD KC JS", 5, center=["6H"], trump="KD"))
        t.update(ob("9D JD KC JS", 4, center=["6H"], backs=1, trump="KD"))
        t.update(ob("9D JD KC JS", 4, center=["6H", "9H"], trump="KD"))
        t.update(ob("9D JD KC JS", 4, trump="KD", my_pile=True, pile_me=CHANGED))
        self.assertEqual(t.round.my_tricks, 1)          # I won although 9H beats 6H
        self.assertEqual(t.round.next_leader, ME)

    def test_unexplained_change_pauses(self):
        t = GameTracker(R)
        t.update(ob("6S AD 7H 10C KH", 5, opp_turn=True))
        for _ in range(5):
            t.update(ob("6S AD 7H", 5))               # two cards vanished without being played
        self.assertEqual(t.status.status, STATUS_PAUSED)

    def test_low_confidence_not_trusted(self):
        t = GameTracker(R)
        o = ob("6S AD 7H 10C KH", 5)
        o.problems = ["low confidence"]
        t.update(o)
        self.assertIsNone(t.round)


if __name__ == "__main__":
    unittest.main()
