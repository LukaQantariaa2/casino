"""Turns engine numbers into a single recommendation for the overlay."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..rules import Rules
from .core import ME, Ctx
from .search import ActionResult, DecisionEngine, InfoState

PHASE_LEAD = "lead"
PHASE_RESPOND = "respond"
PHASE_CLAIM = "claim"  # I just won a trick: ვარ / გაგრძელება
PHASE_RAISE_OFFER = "raise_offer"  # opponent raised: accept / decline / counter


@dataclass
class StakeInfo:
    current: int = 1
    offered: int | None = None  # value offered by the opponent (raise offer phase)
    i_raised_last: bool = False
    can_raise: bool = True  # raise button visible


@dataclass
class Recommendation:
    phase: str
    headline: str  # big text, e.g. "PLAY 10♣ on 9♣"
    detail: str = ""
    cards: tuple[int, ...] = ()
    win_prob: float | None = None
    lose_prob: float | None = None
    expected: float | None = None  # in stake units
    alternatives: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)  # e.g. raise advice
    exact: bool = False


def _pct(x: float | None) -> str:
    return "?" if x is None else f"{100 * x:.0f}%"


class Advisor:
    def __init__(self, rules: Rules, engine: DecisionEngine):
        self.rules = rules
        self.engine = engine

    def names(self, cards) -> str:
        return " ".join(self.rules.card_name(c) for c in cards)

    def _describe(self, info: InfoState, r: ActionResult) -> str:
        if r.kind == "lead":
            return f"LEAD {self.names(r.cards)}"

        ctx = Ctx(self.rules, info.trump_suit)
        pairs = ctx.beat_assignment(r.cards, info.current_lead)

        if pairs:
            return "BEAT " + ", ".join(
                f"{self.rules.card_name(b)} on {self.rules.card_name(a)}"
                for a, b in pairs
            )

        return f"GIVE (face down) {self.names(r.cards)}"

    def recommend(
        self,
        phase: str,
        info: InfoState,
        stake: StakeInfo | None = None,
        budget_s: float | None = None,
    ) -> Recommendation:
        stake = stake or StakeInfo()

        if phase == PHASE_CLAIM:
            return self._claim(info)

        if phase == PHASE_RAISE_OFFER:
            return self._raise_offer(info, stake, budget_s)

        results = self.engine.evaluate(info, budget_s)

        if not results:
            return Recommendation(
                phase,
                "No legal action",
                "hand is empty",
            )

        best = results[0]

        rec = Recommendation(
            phase,
            self._describe(info, best),
            cards=best.cards,
            win_prob=best.win_prob,
            lose_prob=best.lose_prob,
            expected=best.mean * stake.current,
            exact=best.exact,
        )

        rec.detail = (
            f"win {_pct(best.win_prob)} · "
            f"lose {_pct(best.lose_prob)} · "
            f"{'exact' if best.exact else f'{best.n} samples'}"
        )

        # Check whether the second-best action is statistically close.
        #
        # `1 - mean**2` may become slightly negative because of floating-point
        # precision, or because mean is outside [-1, 1]. Taking ** 0.5 of a
        # negative value produces a complex number in Python, which later
        # cannot be compared with a float.
        #
        # Clamp the variance-like value to >= 0 to prevent that.
        if len(results) > 1 and results[1].n and not best.exact:
            second = results[1]

            best_var = max(0.0, 1.0 - best.mean ** 2)
            second_var = max(0.0, 1.0 - second.mean ** 2)

            best_se = math.sqrt(
                best_var / max(1, best.n)
            )

            second_se = math.sqrt(
                second_var / max(1, second.n)
            )

            se = best_se + second_se
            difference = best.mean - second.mean

            if difference < se:
                rec.detail += (
                    " · close call: the next option is about as good"
                )

        # Show up to three alternative actions.
        for r in results[1:4]:
            if r.n:
                rec.alternatives.append(
                    f"{self._describe(info, r)}  —  "
                    f"win {_pct(r.win_prob)}, "
                    f"lose {_pct(r.lose_prob)}"
                )

        # Raise advice on my own turn.
        if stake.can_raise and not stake.i_raised_last:
            nxt = self.rules.next_stake(stake.current)
            thr = self.rules.assistant(
                "raise_min_win_probability"
            )

            if (
                nxt is not None
                and best.win_prob >= thr
                and best.lose_prob < 1 - thr
            ):
                rec.extra.append(
                    f"RAISE ({self.rules.stake_name(nxt)}) — "
                    f"win chance {_pct(best.win_prob)} ≥ {_pct(thr)}"
                )

        return rec

    def _claim(self, info: InfoState) -> Recommendation:
        p = self.engine.claim_probability(info)
        need = self.rules.assistant(
            "claim_min_probability"
        )

        pts = info.my_points_known

        unk = (
            f" + {info.my_pile_unknown} unknown cards"
            if info.my_pile_unknown
            else ""
        )

        if p >= need:
            return Recommendation(
                PHASE_CLAIM,
                "CLAIM — press ვარ",
                (
                    f"you have {pts} pts{unk} "
                    f"(≥ {self.rules.win_points}); "
                    f"P={_pct(p)}"
                ),
                win_prob=p,
            )

        return Recommendation(
            PHASE_CLAIM,
            "CONTINUE — press გაგრძელება",
            (
                f"you have {pts} pts{unk}; "
                f"need {self.rules.win_points}; "
                f"P(enough)={_pct(p)}"
            ),
            win_prob=p,
        )

    def _raise_offer(
        self,
        info: InfoState,
        stake: StakeInfo,
        budget_s: float | None,
    ) -> Recommendation:
        pw, pd, pl = self.engine.win_probability(
            info,
            budget_s,
        )

        cur = stake.current

        new = (
            stake.offered
            or self.rules.next_stake(cur)
            or cur + 1
        )

        ev_accept = new * (pw - pl)
        ev_decline = -cur

        thr_counter = self.rules.assistant(
            "counter_raise_min_win_probability"
        )

        counter = self.rules.next_stake(new)

        detail = (
            f"win {_pct(pw)} · "
            f"lose {_pct(pl)} · "
            f"accept EV {ev_accept:+.2f} "
            f"vs decline {ev_decline:+.2f} "
            f"(break-even win chance "
            f"{_pct((new - cur) / (2 * new))} "
            f"with no draws)"
        )

        if counter is not None and pw >= thr_counter:
            return Recommendation(
                PHASE_RAISE_OFFER,
                (
                    f"COUNTER-RAISE to "
                    f"{self.rules.stake_name(counter)}"
                ),
                detail,
                win_prob=pw,
                lose_prob=pl,
                expected=counter * (pw - pl),
            )

        if ev_accept >= ev_decline:
            return Recommendation(
                PHASE_RAISE_OFFER,
                "ACCEPT (თანხმობა)",
                detail,
                win_prob=pw,
                lose_prob=pl,
                expected=ev_accept,
            )

        return Recommendation(
            PHASE_RAISE_OFFER,
            "DECLINE (არ მსურს)",
            detail,
            win_prob=pw,
            lose_prob=pl,
            expected=ev_decline,
        )