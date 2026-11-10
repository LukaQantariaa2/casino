"""Turns a stream of screen observations into game events and a RoundState.

Design rules
------------
* Only *explained* changes are applied.  Every change between two observations
  must be explained by a game event (lead, response, reveal, trick end,
  draw, new round).  Anything else makes the tracker UNCERTAIN; if it does not
  clear up quickly the tracker PAUSES and asks for a manual correction.
* My hand is read directly every frame, so it can never drift; the hidden
  parts (opponent hand, deck, unrevealed cards) are tracked by counting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..engine.core import ME, OPP, Ctx
from ..engine.search import OppEvidence
from ..rules import Rules
from ..vision.recognizer import Observation
from .state import History, RoundState, Trick

STATUS_WAITING = "WAITING"        # no round being tracked yet
STATUS_OK = "OK"
STATUS_UNCERTAIN = "UNCERTAIN"    # temporary: last observation could not be explained
STATUS_PAUSED = "PAUSED"          # needs manual correction (or a new round)

PHASE_NONE = "wait"
PHASE_LEAD = "lead"
PHASE_RESPOND = "respond"
PHASE_CLAIM = "claim"
PHASE_RAISE_OFFER = "raise_offer"
PHASE_ROUND_OVER = "round_over"


@dataclass
class TrackerConfig:
    uncertain_to_pause: int = 4         # consecutive unexplained observations before pausing
    lowconf_to_pause: int = 6


@dataclass
class TrackerStatus:
    status: str = STATUS_WAITING
    phase: str = PHASE_NONE
    reasons: list[str] = field(default_factory=list)


class GameTracker:
    def __init__(self, rules: Rules, history: History | None = None, cfg: TrackerConfig | None = None):
        self.rules = rules
        self.cfg = cfg or TrackerConfig()
        self.history = history or History()
        self.round: RoundState | None = None
        self.round_no = 0
        self.status = TrackerStatus()
        self._unexplained = 0
        self._lowconf = 0
        self._last: Observation | None = None
        self._modal_was_open = False
        self._last_trick: dict | None = None
        self._pile_ref: dict | None = None
        self._prev_pile_sig: dict | None = None
        self._badges: dict = {}
        self._badge_baseline_needed = True
        self.version = 0                     # bumps whenever the state changes

    # ================================================================ helpers
    def n(self, cards) -> str:
        return " ".join(self.rules.card_name(c) for c in cards)

    def _log(self, kind: str, text: str, **data):
        self.history.add(kind, text, round=self.round_no, **data)
        self.version += 1

    def _set(self, status: str, reasons: list[str] | None = None):
        if status != self.status.status or (reasons or []) != self.status.reasons:
            self.version += 1
        self.status.status = status
        self.status.reasons = reasons or []

    def _ctx(self) -> Ctx:
        return Ctx(self.rules, self.round.trump_suit)

    # ================================================================ public
    def manual_reset(self):
        self.round = None
        self._set(STATUS_WAITING, ["reset by user — waiting for the next deal"])
        self._log("reset", "tracker reset by user")

    def apply_manual(self, rs: RoundState):
        """Replace the round state with a user-corrected one."""
        errs = rs.consistency_errors()
        if errs:
            raise ValueError("; ".join(errs))
        self.round = rs
        self._unexplained = 0
        self._set(STATUS_OK)
        self._log("manual", "state corrected manually")

    def update(self, ob: Observation) -> None:
        self._last = ob

        # ---- raise offer modal: dims the table, so read nothing else
        if ob.raise_modal:
            if self.round and not self._modal_was_open:
                nxt = self.rules.next_stake(self.round.stake)
                self.round.stake_offered = nxt
                self._log("raise_offer", f"opponent offers to raise to {self.rules.stake_name(nxt or 0)} ({nxt})")
            self._modal_was_open = True
            self.status.phase = PHASE_RAISE_OFFER if self.round else PHASE_NONE
            return
        modal_just_closed = self._modal_was_open
        self._modal_was_open = False
        self._track_stake_badges(ob, modal_just_closed)

        # ---- end-of-round panel
        if ob.end_panel:
            if self.round and not self.round.finished:
                if self.round.trick is not None and self.round.trick.responded:
                    self._resolve_trick(ob)
                self.round.finished = True
                pts = sum(self.rules.points[d.card] for d in ob.end_panel_cards)
                self._log("round_end", f"round over (panel shows {len(ob.end_panel_cards)} cards / ≥{pts} pts)",
                          panel=[self.rules.card_name(d.card) for d in ob.end_panel_cards])
            self.status.phase = PHASE_ROUND_OVER
            if self.status.status != STATUS_PAUSED:
                self._set(STATUS_WAITING, ["round over — waiting for the next deal"])
            return

        # ---- low-confidence recognition: do not trust this frame
        if ob.problems:
            self._lowconf += 1
            if self._lowconf >= self.cfg.lowconf_to_pause:
                self._set(STATUS_PAUSED, ["recognition uncertain: " + "; ".join(ob.problems[:3])])
            else:
                self._set(STATUS_UNCERTAIN, ob.problems[:3])
            return
        self._lowconf = 0

        # ---- new round?
        if self._is_new_deal(ob):
            self._start_round(ob)
            return
        if self.round is None or self.round.finished:
            if self.round and self.round.finished and ob.table_empty:
                pass
            elif self.round and not self.round.finished:
                pass
            self.status.phase = PHASE_NONE
            if self.status.status not in (STATUS_PAUSED,):
                self._set(STATUS_WAITING, ["waiting for a new deal (start the tool before the cards are dealt)"])
            return
        if ob.table_empty and not ob.raise_modal:
            # all cards vanished: fold / declined raise / special win / last trick taken
            r = self.round
            if r.trick is not None and r.trick.responded:
                self._resolve_trick(ob)
            r.finished = True
            self._log("round_end", f"table cleared — round over (me {r.points(ME)} / opp {r.points(OPP)}"
                                   f"{', some cards unseen' if r.my_pile_unknown or r.opp_pile_unknown else ''})")
            self._set(STATUS_WAITING, ["round over — waiting for the next deal"])
            self.status.phase = PHASE_ROUND_OVER
            return

        if self.status.status == STATUS_PAUSED:
            # stay paused until a new round or manual correction
            self.status.phase = PHASE_NONE
            return

        ok, why = self._advance(ob)
        self._prev_pile_sig = ob.pile_sig
        if ok:
            self._unexplained = 0
            errs = self.round.consistency_errors()
            if errs:
                self._set(STATUS_PAUSED, errs)
            else:
                self._set(STATUS_OK)
        else:
            self._unexplained += 1
            if self._unexplained >= self.cfg.uncertain_to_pause:
                self._set(STATUS_PAUSED, [why or "screen does not match the tracked state"])
            elif self._unexplained >= 2:
                self._set(STATUS_UNCERTAIN, [why or "unexplained change"])
            # a single unexplained frame is usually a card in motion: ignore it silently
        self.status.phase = self._phase(ob)

    # ================================================================ round start
    def _is_new_deal(self, ob: Observation) -> bool:
        hs = self.rules.hand_size
        fresh = (len(ob.my_hand) == hs and ob.opp_count == hs and not ob.center and ob.center_backs == 0
                 and ob.trump is not None and ob.deck_present and not ob.my_pile_present and not ob.opp_pile_present)
        if not fresh:
            return False
        if self.round is None or self.round.finished:
            return True
        r = self.round
        # same round still at its start?  then it's not new
        if r.trump_card == ob.trump.card and set(r.my_hand) == set(ob.hand_ids()) and not r.my_pile \
                and not r.opp_pile and r.trick is None:
            return False
        return r.my_tricks + r.opp_tricks > 0 or r.trump_card != ob.trump.card

    def _start_round(self, ob: Observation):
        self.round_no += 1
        r = RoundState(self.rules, number=self.round_no)
        r.trump_card = ob.trump.card
        r.trump_suit = self.rules.suit_of[ob.trump.card]
        r.my_hand = ob.hand_ids()
        r.opp_hand_size = ob.opp_count
        r.deck_size = self.rules.n_cards - 2 * self.rules.hand_size
        r.trump_in_deck = True
        r.next_leader = ME if ob.my_turn else (OPP if ob.opp_turn else None)
        self.round = r
        self._last_trick = None
        self._unexplained = 0
        self._set(STATUS_OK)
        self._log("new_round", f"round {self.round_no}: trump {self.rules.card_name(r.trump_card)}, "
                               f"my hand {self.n(r.my_hand)}")
        self.status.phase = self._phase(ob)

    # ================================================================ transitions
    def _advance(self, ob: Observation) -> tuple[bool, str]:
        r = self.round
        hand_obs = set(ob.hand_ids())
        hand = set(r.my_hand)
        removed = hand - hand_obs
        added = hand_obs - hand
        center = set(ob.center_ids())
        opp_delta = ob.opp_count - r.opp_hand_size

        # 1) a lead appears
        if r.trick is None and (center or ob.center_backs):
            if removed and removed <= center:
                if not self._legal_lead(removed):
                    return False, f"illegal lead {self.n(removed)}?"
                r.my_hand = [c for c in r.my_hand if c not in removed]
                r.trick = Trick(ME, sorted(removed))
                self._pile_ref = self._prev_pile_sig or ob.pile_sig
                self._log("lead", f"I lead {self.n(sorted(removed))}")
                hand = set(r.my_hand)
                removed = set()
            elif not removed and opp_delta < 0 and len(center) >= -opp_delta:
                lead = self._opp_lead_cards(center, -opp_delta)
                if lead is None:
                    return False, "cannot tell which cards the opponent led"
                r.opp_hand_size += opp_delta
                opp_delta = 0
                for c in lead:
                    if c in r.opp_known:
                        r.opp_known.remove(c)
                r.trick = Trick(OPP, sorted(lead))
                self._pile_ref = self._prev_pile_sig or ob.pile_sig
                self._log("lead", f"opponent leads {self.n(sorted(lead))}")
            else:
                return False, "cards appeared in the middle but no hand lost them"

        t = r.trick
        # 2) the response appears
        if t is not None and not t.responded:
            if t.leader == OPP and removed:
                if len(removed) != t.k:
                    return False, f"I gave {len(removed)} cards to a {t.k}-card lead"
                r.my_hand = [c for c in r.my_hand if c not in removed]
                t.response = sorted(removed)
                t.responded = True
                if ob.center_backs > 0:
                    t.face_down = True
                elif removed <= center:
                    t.face_down = False
                beat = self._ctx().beat_assignment(t.response, t.lead)
                how = {True: " face down", False: " face up", None: ""}[t.face_down]
                self._log("response", f"I answer{how} {self.n(t.response)}" +
                          (" (beats)" if beat and t.face_down is not True else ""))
                removed = set()
            elif t.leader == ME and opp_delta < 0:
                if -opp_delta != t.k:
                    return False, f"opponent dropped {-opp_delta} cards to a {t.k}-card lead"
                r.opp_hand_size += opp_delta
                opp_delta = 0
                up = sorted(center - set(t.lead))
                t.responded = True
                if ob.center_backs > 0:
                    t.face_down = True
                    t.response = up[:max(0, t.k - ob.center_backs)]
                elif len(up) >= t.k:
                    t.face_down = False
                    t.response = up[:t.k]
                else:
                    t.response = up
                t.response_hidden = t.k - len(t.response)
                for c in t.response:
                    if c in r.opp_known:
                        r.opp_known.remove(c)
                if t.face_down:
                    self._log("response", f"opponent gives {t.k} card(s) face down")
                elif t.face_down is False:
                    beat = self._ctx().beat_assignment(t.response, t.lead)
                    self._log("response", f"opponent {'beats with' if beat else 'answers'} {self.n(t.response)}")
                else:
                    self._log("response", f"opponent answers ({t.k} card(s), not seen)")

        # 3) hidden response gets revealed
        if t is not None and t.responded and t.response_hidden:
            newly = sorted(center - set(t.lead) - set(t.response))
            if newly:
                take = newly[:t.response_hidden]
                t.response += take
                t.response_hidden -= len(take)
                for c in take:
                    if c in r.opp_known:
                        r.opp_known.remove(c)
                self._log("reveal", f"opponent's face-down cards: {self.n(take)}")

        # 4) trick resolved: table centre cleared
        if t is not None and t.responded and not center and ob.center_backs == 0:
            self._resolve_trick(ob)
            t = None
        elif t is not None and not t.responded and not center and ob.center_backs == 0:
            return False, "the trick vanished before the answer was seen"

        # 4b) evidence about who won the last trick
        self._verify_last_trick(ob)

        # 5) draws (only when no trick is open)
        if r.trick is None and (added or opp_delta > 0):
            if removed:
                return False, f"cards left my hand unexpectedly: {self.n(removed)}"
            n_draw = len(added) + max(0, opp_delta)
            if n_draw > r.deck_size:
                return False, f"{n_draw} cards drawn but only {r.deck_size} left in the deck"
            if opp_delta < 0:
                return False, "opponent lost cards without playing them"
            r.my_hand = sorted(hand_obs)
            r.opp_hand_size += opp_delta
            r.deck_size -= n_draw
            msg = f"I draw {self.n(sorted(added))}" if added else "I draw nothing"
            msg += f"; opponent draws {opp_delta}"
            if r.trump_card in added:
                r.trump_in_deck = False
                msg += " (I got the trump card)"
            elif r.trump_in_deck and r.deck_size == 0:
                r.trump_in_deck = False
                if r.trump_card not in r.opp_known:
                    r.opp_known.append(r.trump_card)
                msg += f" (opponent took the trump card {self.rules.card_name(r.trump_card)})"
            self._log("draw", msg + f"; deck {r.deck_size}")
            added, removed = set(), set()

        # anything left unexplained?
        if removed or added:
            return False, f"my hand changed unexpectedly (+{self.n(added)} -{self.n(removed)})"
        if ob.opp_count != r.opp_hand_size:
            return False, f"opponent shows {ob.opp_count} cards, expected {r.opp_hand_size}"
        if r.deck_size > 1 and not ob.deck_present:
            return False, "deck not visible but cards should remain"
        return True, ""

    def _track_stake_badges(self, ob: Observation, modal_just_closed: bool):
        """The green "X2/X3.." badge sits on the avatar of the player who raised last."""
        import numpy as np
        r = self.round
        prev = self._badges
        cur = ob.stake_badge or {}
        self._badges = {k: (None if v is None else v.copy()) for k, v in cur.items()}
        if r is not None and not prev and self._badge_baseline_needed:
            # first observation of this round: badges carried over from earlier are not raises
            self._badge_baseline_needed = False
            return
        if r is None or r.finished or not cur:
            return
        for who in ("me", "opp"):
            b, pb = cur.get(who), prev.get(who)
            if b is None:
                continue
            changed = pb is None or int(np.logical_xor(b, pb).sum()) > 12
            if not changed:
                continue
            if who == "opp":
                new = r.stake_offered or self.rules.next_stake(r.stake)
                r.i_raised_last = False
                text = "opponent's raise accepted" if r.stake_offered else "opponent raised"
            else:
                base = r.stake_offered if r.stake_offered else r.stake
                new = self.rules.next_stake(base)
                r.i_raised_last = True
                text = "I counter-raised" if r.stake_offered else "I raised"
            if new:
                r.stake = new
            r.stake_offered = None
            self._log("stake", f"{text}: stake now {self.rules.stake_name(r.stake)} ({r.stake})")
        if modal_just_closed and r.stake_offered:
            # modal closed without a badge change yet: accepted (badge will confirm) or declined (round ends)
            pass

    def _legal_lead(self, cards) -> bool:
        cards = tuple(sorted(cards))
        return cards in {tuple(sorted(x)) for x in self._ctx().legal_leads(self.round.my_hand)}

    def _opp_lead_cards(self, center: set[int], k: int) -> list[int] | None:
        if len(center) == k:
            return sorted(center)
        return None

    def _resolve_trick(self, ob: Observation):
        r = self.round
        t = r.trick
        ctx = self._ctx()
        beats = t.response_hidden == 0 and ctx.beat_assignment(t.response, t.lead) is not None
        # Cards are always played face up, so an answer we did not see (or only saw as a back in
        # flight) may well have beaten the lead.
        could_win = beats or t.response_hidden > 0
        # a responder may also give up cards that could beat, so the rules alone cannot decide;
        # the decisive evidence is which pile badge changed
        pile_winner = self._pile_change_winner(ob)
        if pile_winner is not None:
            winner, sure = pile_winner, True
        else:
            winner, sure = ((1 - t.leader) if beats else t.leader), False
        if winner != t.leader and not could_win:
            # the pile says the responder won but the cards cannot beat: trust the rules
            winner, sure = t.leader, False
        self._award(t.lead + t.response, t.response_hidden, winner)
        evidence_idx = None
        if t.leader == ME and winner == ME and not beats:
            evidence_idx = len(r.opp_evidence)
            r.opp_evidence.append(OppEvidence(
                kind="dump",
                lead=tuple(t.lead),
                stock_empty=r.deck_size == 0,
            ))
        r.next_leader = winner if self.rules.raw["draw"]["trick_winner_leads_next"] else 1 - t.leader
        self._last_trick = {"cards": t.lead + t.response, "hidden": t.response_hidden, "winner": winner,
                            "leader": t.leader, "sure": sure, "checked": sure, "beats": could_win,
                            "pile_ref": self._pile_ref, "evidence_idx": evidence_idx}
        pts = sum(self.rules.points[c] for c in t.lead + t.response)
        who = "I" if winner == ME else "opponent"
        self._log("trick", f"{who} take{'s' if winner == OPP else ''} the trick (+{pts} pts"
                           f"{', ' + str(t.response_hidden) + ' unseen' if t.response_hidden else ''}) — "
                           f"score me {r.points(ME)} / opp {r.points(OPP)}")
        r.trick = None

    def _award(self, cards: list[int], hidden: int, winner: int, sign: int = 1):
        r = self.round
        pile = r.my_pile if winner == ME else r.opp_pile
        if sign > 0:
            pile += cards
        else:
            for c in cards:
                pile.remove(c)
        if winner == ME:
            r.my_pile_unknown += sign * hidden
            r.my_tricks += sign
        else:
            r.opp_pile_unknown += sign * hidden
            r.opp_tricks += sign
        r.last_trick_winner = winner

    def _flip_last_trick(self, why: str):
        lt = self._last_trick
        r = self.round
        old = lt["winner"]
        self._award(lt["cards"], lt["hidden"], old, sign=-1)
        new = 1 - old
        self._award(lt["cards"], lt["hidden"], new)
        lt["winner"] = new
        lt["checked"] = True
        idx = lt.get("evidence_idx")
        if idx is not None and new != lt["leader"] and idx < len(r.opp_evidence):
            del r.opp_evidence[idx]           # he did not give the trick up after all
            lt["evidence_idx"] = None
        if self.rules.raw["draw"]["trick_winner_leads_next"]:
            r.next_leader = new
        self._log("correction", f"last trick actually won by {'me' if new == ME else 'opponent'} ({why}) — "
                                f"score me {r.points(ME)} / opp {r.points(OPP)}")

    def _pile_change_winner(self, ob: Observation, ref: dict | None = None) -> int | None:
        ref = ref if ref is not None else self._pile_ref
        if not ref or not ob.pile_sig:
            return None
        import numpy as np
        d_me = float(np.abs(ob.pile_sig["me"] - ref["me"]).mean())
        d_opp = float(np.abs(ob.pile_sig["opp"] - ref["opp"]).mean())
        if max(d_me, d_opp) < 4.0:
            return None
        if d_me > 2 * d_opp:
            return ME
        if d_opp > 2 * d_me:
            return OPP
        return None

    def _verify_last_trick(self, ob: Observation):
        lt = getattr(self, "_last_trick", None)
        r = self.round
        if not lt or lt["checked"]:
            return
        # (0) a pile badge changed after the trick
        if r.trick is None:
            pw = self._pile_change_winner(ob, lt.get("pile_ref"))
            if pw is not None:
                if pw != lt["winner"] and (pw == lt["leader"] or lt["beats"]):
                    self._flip_last_trick("pile badge changed")
                lt["checked"] = True
                return
        # (a) claim buttons are only offered to the trick winner
        if "claim" in ob.buttons and "continue" in ob.buttons:
            if lt["winner"] != ME and (lt["leader"] == ME or lt["beats"]):
                self._flip_last_trick("claim buttons shown to me")
            lt["checked"] = True
            return
        # (b) the pile badge appears with a player's first trick
        feasible_flip = lt["winner"] != lt["leader"] or lt["beats"]
        if feasible_flip and lt["winner"] == ME and r.my_tricks == 1 and not ob.my_pile_present \
                and ob.opp_pile_present:
            self._flip_last_trick("only the opponent has a pile")
            return
        if feasible_flip and lt["winner"] == OPP and r.opp_tricks == 1 and not ob.opp_pile_present \
                and ob.my_pile_present:
            self._flip_last_trick("only I have a pile")
            return
        # (c) the next lead comes from the winner
        if r.trick is not None and self.rules.raw["draw"]["trick_winner_leads_next"]:
            if r.trick.leader != lt["winner"] and (r.trick.leader == lt["leader"] or lt["beats"]):
                self._flip_last_trick("the other player leads next")
            lt["checked"] = True

    # ================================================================ phase
    def _phase(self, ob: Observation) -> str:
        r = self.round
        if r is None or r.finished:
            return PHASE_NONE
        if ob.raise_modal:
            return PHASE_RAISE_OFFER
        if "claim" in ob.buttons and "continue" in ob.buttons:
            return PHASE_CLAIM
        my_turn = ob.my_turn or "fold" in ob.buttons
        if not my_turn:
            return PHASE_NONE
        if r.trick is None and not ob.center:
            return PHASE_LEAD
        if r.trick is not None and r.trick.leader == OPP and not r.trick.responded:
            return PHASE_RESPOND
        return PHASE_NONE
