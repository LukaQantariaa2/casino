import json
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bura.rules import Rules, RulesError  # noqa: E402
from bura.engine.core import Ctx, SimState, apply_trick, ME, OPP  # noqa: E402
from bura.engine.search import InfoState, DecisionEngine, EndgameSolver  # noqa: E402
from bura.engine.advisor import Advisor, PHASE_CLAIM, PHASE_RESPOND, PHASE_LEAD, PHASE_RAISE_OFFER, StakeInfo  # noqa

RULES_PATH = os.path.join(ROOT, "config", "rules", "bura5_adjarabet.json")


class RulesTests(unittest.TestCase):
    def setUp(self):
        self.R = Rules.load(RULES_PATH)

    def test_deck_and_points(self):
        self.assertEqual(self.R.n_cards, 36)
        self.assertEqual(self.R.total_points, 120)
        self.assertEqual(self.R.card_name(self.R.parse_card("10C")), "10♣")
        self.assertEqual(self.R.parse_card("A♦"), self.R.parse_card("ad"))
        self.assertEqual(self.R.parse_card("TS"), self.R.parse_card("10S"))

    def test_ten_beats_king(self):
        R = self.R
        P = R.parse_card
        trump = R.suits.index("D")
        self.assertTrue(R.beats(P("10C"), P("KC"), trump))
        self.assertFalse(R.beats(P("KC"), P("10C"), trump))
        self.assertTrue(R.beats(P("6D"), P("AC"), trump))   # trump beats side suit
        self.assertFalse(R.beats(P("AC"), P("6D"), trump))
        self.assertFalse(R.beats(P("AH"), P("6C"), trump))  # other side suit never beats

    def test_invalid_config(self):
        with self.assertRaises(RulesError):
            Rules.from_dict({"rank_order_low_to_high": ["6", "7"]})
        with self.assertRaises(RulesError):
            Rules.from_dict({"lead": {"max_cards": 9}})

    def test_three_card_variant(self):
        R = Rules.load(os.path.join(ROOT, "config", "rules", "bura3.json"))
        self.assertEqual(R.hand_size, 3)
        self.assertEqual(R.win_points, 31)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.R = Rules.load(RULES_PATH)
        self.P = self.R.parse_cards
        self.ctx = Ctx(self.R, self.R.suits.index("D"))

    def test_legal_leads_same_suit(self):
        leads = self.ctx.legal_leads(self.P("6S AS 7H 10C KH"))
        sizes = sorted(len(x) for x in leads)
        # singles: 5, pairs: 6S+AS, 7H+KH  -> 7 leads
        self.assertEqual(len(leads), 7)
        self.assertEqual(sizes.count(2), 2)

    def test_multi_card_beat_requires_matching(self):
        P = self.P
        lead = P("7H KH")
        self.assertIsNotNone(self.ctx.beat_assignment(P("AH 6D"), lead))   # A♥ on K♥/7♥, trump on other
        self.assertIsNone(self.ctx.beat_assignment(P("6D 6C"), lead))
        self.assertIsNone(self.ctx.beat_assignment(P("8H 9H"), lead))       # nothing beats K♥

    def test_trick_and_draw(self):
        P = self.P
        s = SimState([P("6S AD 7H 10C KH"), P("9C 6D 6C QS JS")], P("8S 9S AC 7D"), [0, 0], [0, 0], OPP)
        w = apply_trick(self.ctx, s, OPP, tuple(P("9C")), tuple(P("10C")))
        self.assertEqual(w, ME)
        self.assertEqual(s.pts, [10, 0])
        self.assertEqual(len(s.hands[0]), 5)
        self.assertEqual(len(s.hands[1]), 5)
        self.assertEqual(s.leader, ME)
        self.assertIn(P("7D")[0], s.hands[0])          # winner drew first (alternating)

    def test_claim_ends_round(self):
        P = self.P
        s = SimState([P("AH 10H"), P("6C 7C")], [], [55, 0], [3, 0], ME)
        apply_trick(self.ctx, s, ME, tuple(P("AH")), tuple(P("6C")))
        self.assertTrue(s.done)
        self.assertEqual(s.result, 1.0)


class SolverTests(unittest.TestCase):
    def test_endgame_exact(self):
        R = Rules.load(RULES_PATH)
        P = R.parse_cards
        ctx = Ctx(R, R.suits.index("D"))
        # I lead with A♥ and 6D; opponent has 10♥ and 7C. 50 vs 50 points.
        s = SimState([P("AH 6D"), P("10H 7C")], [], [50, 50], [1, 1], ME)
        v = EndgameSolver(ctx).value(s)
        self.assertGreater(v, 0.5)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.R = Rules.load(RULES_PATH)
        self.P = self.R.parse_cards
        self.E = DecisionEngine(self.R, time_budget_s=0.6, seed=3)
        self.A = Advisor(self.R, self.E)

    def test_respond_recording_position(self):
        P = self.P
        info = InfoState(self.R, self.R.suits.index("D"), P("6S AD 7H 10C KH"), 4, 26,
                         trump_card=P("7D")[0], leader=OPP, current_lead=P("9C"))
        self.assertEqual(info.check(), [])
        rec = self.A.recommend(PHASE_RESPOND, info)
        self.assertEqual(rec.cards, tuple(P("10C")))
        self.assertIn("BEAT 10♣ on 9♣", rec.headline)

    def test_claim(self):
        P = self.P
        info = InfoState(self.R, 0, P("6S"), 1, 0, trump_card=P("7D")[0], trump_card_in_deck=False,
                         my_pile_known=P("AH AC AS 10H 10C 10S"), opp_pile_known=[], leader=ME)
        rec = self.A.recommend(PHASE_CLAIM, info)
        self.assertTrue(rec.headline.startswith("CLAIM"))
        info2 = InfoState(self.R, 0, P("6S"), 1, 0, trump_card=P("7D")[0], trump_card_in_deck=False,
                          my_pile_known=P("AH AC"), leader=ME)
        self.assertTrue(self.A.recommend(PHASE_CLAIM, info2).headline.startswith("CONTINUE"))

    def test_consistency_check(self):
        P = self.P
        bad = InfoState(self.R, 0, P("6S AD"), 5, 10, trump_card=P("7D")[0])
        self.assertTrue(bad.check())

    def test_raise_offer_math(self):
        P = self.P
        info = InfoState(self.R, self.R.suits.index("D"), P("AD 10D KD QD JD"), 5, 26,
                         trump_card=P("7D")[0], leader=ME)
        rec = self.A.recommend(PHASE_RAISE_OFFER, info, StakeInfo(current=1, offered=2), budget_s=0.4)
        self.assertNotIn("DECLINE", rec.headline)

    def test_lead_returns_legal(self):
        P = self.P
        info = InfoState(self.R, self.R.suits.index("D"), P("6S AS 7H 10C KH"), 5, 26,
                         trump_card=P("7D")[0], leader=ME)
        rec = self.A.recommend(PHASE_LEAD, info, budget_s=0.4)
        ctx = Ctx(self.R, info.trump_suit)
        self.assertIn(tuple(sorted(rec.cards)), [tuple(sorted(x)) for x in ctx.legal_leads(info.my_hand)])


if __name__ == "__main__":
    unittest.main()
