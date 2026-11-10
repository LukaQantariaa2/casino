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

    def test_lead_likelihood_prefers_low_side_suit(self):
        from bura.engine.policy import LEAD_FLOOR, lead_likelihood
        ctx = Ctx(self.R, self.R.suits.index("D"))
        hand = self.P("6S AS 7H 10C KH")
        p_low = lead_likelihood(ctx, hand, self.P("6S"))
        p_ace = lead_likelihood(ctx, hand, self.P("AS"))
        self.assertGreater(p_low, p_ace)
        # calibrated: an "irrational" ace lead is a few times less likely, never ~impossible
        self.assertGreaterEqual(p_ace, LEAD_FLOOR)
        self.assertLess(p_low / p_ace, 10)

    def test_posterior_does_not_collapse_on_high_lead(self):
        import random
        from bura.engine.policy import WEIGHT_FLOOR, sample_weight
        from bura.engine.search import draw_sample
        ctx = Ctx(self.R, self.R.suits.index("D"))
        P = self.P
        for lead in ("10C", "AC"):
            info = InfoState(self.R, self.R.suits.index("D"), P("6S KD 7H 10H QH"), 4, 26,
                             trump_card=P("7D")[0], leader=OPP, current_lead=P(lead))
            rng = random.Random(1)
            ws = [sample_weight(ctx, info, draw_sample(info, rng)) for _ in range(1000)]
            ess = sum(ws) ** 2 / sum(w * w for w in ws)
            self.assertGreater(ess / 1000, 0.4, f"ESS collapsed after opponent led {lead}")
            self.assertGreaterEqual(min(ws), WEIGHT_FLOOR)

    def test_same_state_same_answer(self):
        P = self.P
        info = InfoState(self.R, self.R.suits.index("D"), P("6S AS 7H 10C KH"), 5, 26,
                         trump_card=P("7D")[0], leader=ME)
        eng = DecisionEngine(self.R, time_budget_s=30.0, max_samples=48)
        a = eng.evaluate(info)
        b = eng.evaluate(info)
        self.assertEqual([(r.cards, r.n, round(r.mean, 9)) for r in a],
                         [(r.cards, r.n, round(r.mean, 9)) for r in b])
        self.assertGreater(a[0].win_se, 0.0)

    def test_history_tail_hides_advice(self):
        from bura.tracker.state import History
        h = History()
        h.add("lead", "a")
        h.add("advice", "x")
        h.add("trick", "b")
        self.assertEqual([e.text for e in h.tail(5)], ["a", "b"])

    def test_raise_ev_helpers(self):
        from bura.engine.advisor import _ev_if_i_raise, _opponent_accepts
        self.assertTrue(_opponent_accepts(1, 2, 0.55, 0.45))
        self.assertFalse(_opponent_accepts(1, 2, 0.95, 0.05))
        self.assertGreater(_ev_if_i_raise(1, 2, 0.6, 0.3), 1 * (0.6 - 0.3))
        self.assertLess(_ev_if_i_raise(1, 2, 0.3, 0.6), 1 * (0.3 - 0.6))

    def test_dump_and_lead_posterior(self):
        from bura.engine.policy import dump_weight, sample_weight
        from bura.engine.search import OppEvidence, Sample
        ctx = Ctx(self.R, self.R.suits.index("D"))
        P = self.P
        self.assertLess(dump_weight(ctx, P("AC 7H"), P("10C")), dump_weight(ctx, P("6S 7H"), P("10C")))
        info = InfoState(self.R, self.R.suits.index("D"), P("6D 7D"), 2, 0,
                         trump_card=P("8D")[0], trump_card_in_deck=False, leader=OPP,
                         current_lead=P("6S"),
                         opp_evidence=[OppEvidence("dump", tuple(P("10C")), True)])
        w_ace = sample_weight(ctx, info, Sample(P("AS AC"), [], 0, 0))
        w_low = sample_weight(ctx, info, Sample(P("7S 8S"), [], 0, 0))
        self.assertGreater(w_low, w_ace)

    def test_bura_rules(self):
        from bura.engine.policy import sample_weight
        from bura.engine.search import Sample
        P = self.P
        ctx = Ctx(self.R, self.R.suits.index("D"))
        self.assertIsNotNone(ctx.hand_special(P("6D 7D 8D 9D JD")))
        self.assertIsNone(ctx.hand_special(P("6S 7S 8S 9S JS")))       # Malyutka is not an instant win
        # a full opponent hand that is already Bura would have been declared -> impossible world
        info = InfoState(self.R, self.R.suits.index("D"), P("6S 7S 8S 9S JS"), 5, 26,
                         trump_card=P("AD")[0], leader=ME)
        self.assertEqual(sample_weight(ctx, info, Sample(P("6D 7D 8D 9D JD"), [], 0, 0)), 0.0)
        self.assertGreater(sample_weight(ctx, info, Sample(P("6D 7D 8D 9D JC"), [], 0, 0)), 0.0)
        # drawing into Bura ends the round in the simulation
        s = SimState([P("6D 7D 8D 9D 6C"), P("6S 8C 9C JC QC")], P("KC JD"), [0, 0], [0, 0], ME)
        apply_trick(ctx, s, ME, tuple(P("6C")), tuple(P("6S")))   # I win the trick and draw J♦
        self.assertTrue(s.done)
        self.assertEqual(s.result, 1.0)
        # advisor tells me to play the Bura
        info2 = InfoState(self.R, self.R.suits.index("D"), P("6D 7D 8D 9D JD"), 5, 26,
                          trump_card=P("AD")[0], leader=ME)
        rec = self.A.recommend(PHASE_LEAD, info2, StakeInfo(current=1, can_raise=True))
        self.assertTrue(rec.headline.startswith("BURA"))
        self.assertTrue(rec.extra and rec.extra[0].startswith("RAISE"))

    def test_short_stock_exact_solver(self):
        P = self.P
        # stock of 2 (trump card drawn last): a decisive position must be resolved with certainty
        info = InfoState(self.R, self.R.suits.index("D"), P("AD 10D"), 2, 2, trump_card=P("6D")[0],
                         my_pile_known=P("AH AC AS 10H 10C"), my_pile_unknown=9,
                         opp_pile_known=P("10S KH KC KS QH QC QS JH"), opp_pile_unknown=8,
                         opp_known=P("6C 7C"), leader=ME)
        self.assertEqual(info.check(), [])
        res = self.E.evaluate(info, 0.5)
        self.assertGreater(res[0].win_prob, 0.99)

    def test_short_stock_screens_then_solves(self):
        """Stock of 4 with many legal leads: every action is reported, the finalists come first."""
        from bura.engine import search
        P = self.P
        info = InfoState(self.R, self.R.suits.index("D"), P("7H 9H JH KH 8C"), 5, 4, trump_card=P("6D")[0],
                         my_pile_unknown=11, opp_pile_unknown=11, leader=ME)
        self.assertEqual(info.check(), [])
        actions = self.E.legal_actions(info)
        self.assertGreater(len(actions), search.EXACT_MAX_ACTIONS)
        res = self.E.evaluate(info, 0.6)
        self.assertEqual({r.cards for r in res}, set(actions))
        finalists = res[:search.EXACT_MAX_ACTIONS]
        self.assertTrue(all(r.n > 0 for r in finalists))
        self.assertEqual(finalists, sorted(finalists, key=lambda r: -r.mean))

    def test_claim_skips_when_pile_short(self):
        P = self.P
        info = InfoState(self.R, 0, P("6S"), 1, 0, trump_card=P("7D")[0], trump_card_in_deck=False,
                         my_pile_known=P("AH"), leader=ME)
        rec = self.A.recommend(PHASE_CLAIM, info, budget_s=0.2)
        self.assertTrue(rec.headline.startswith("CONTINUE"))


if __name__ == "__main__":
    unittest.main()
