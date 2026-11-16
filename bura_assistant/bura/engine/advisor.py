"""Turns engine numbers into a single recommendation for the overlay."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..rules import Rules
from .core import Ctx, Give
from .search import ActionResult, DecisionEngine, InfoState, count_worlds

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
    deals_checked: int | None = None   # distinct deals the engine calculated (None: no search needed)
    deals_possible: int | None = None  # deals consistent with what I have seen
    deals_all: bool = False            # every possible deal was calculated


MIN_ALT_SAMPLES = 200   # alternatives with fewer effective samples get no percentages
MIN_SOLID_SAMPLES = 100  # below this the headline percentages are flagged as rough


def _pct(x: float | None) -> str:
    return "?" if x is None else f"{100 * x:.0f}%"


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _count(n: int) -> str:
    if n < 1_000_000:
        return f"{n:,}"
    if n < 1_000_000_000:
        return f"{n / 1e6:.1f} million"
    if n < 1_000_000_000_000:
        return f"{n / 1e9:.1f} billion"
    exp = len(str(n)) - 1
    return f"{n / 10 ** exp:.1f}×10{str(exp).translate(_SUPERSCRIPT)}"


def _edge(pw: float, pl: float) -> float:
    return pw - pl


def _opponent_accepts(current: int, offered: int, pw: float, pl: float) -> bool:
    """Zero-sum: they accept iff playing at ``offered`` beats folding ``current``."""
    return offered * (pl - pw) >= -current


def _ev_if_i_raise(current: int, offered: int, pw: float, pl: float) -> float:
    if _opponent_accepts(current, offered, pw, pl):
        return offered * _edge(pw, pl)
    return float(current)


class Advisor:
    def __init__(self, rules: Rules, engine: DecisionEngine):
        self.rules = rules
        self.engine = engine

    def names(self, cards) -> str:
        return " ".join(self.rules.card_name(c) for c in cards)

    def _describe(self, info: InfoState, r: ActionResult) -> str:
        if r.kind == "lead":
            return f"LEAD {self.names(r.cards)}"

        if isinstance(r.cards, Give):
            return f"GIVE (face down) {self.names(r.cards)}"
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
        self.engine.last_checked = None
        rec = self._recommend(phase, info, stake, budget_s)
        self._note_deals(info, rec)
        return rec

    def _note_deals(self, info: InfoState, rec: Recommendation) -> None:
        """Append how many possible deals were calculated, out of how many exist."""
        n = self.engine.last_checked
        sep = " · " if rec.detail else ""
        if n is None:
            rec.detail += f"{sep}no deal calculation needed"
            return
        total = count_worlds(info)
        rec.deals_checked, rec.deals_possible, rec.deals_all = n, total, self.engine.last_exhaustive
        if self.engine.last_exhaustive:
            text = ("the only possible deal calculated (all cards known)" if n == 1
                    else f"all {_count(n)} possible deals calculated")
        else:
            text = f"{_count(n)} of {_count(total)} possible deals calculated"
        rec.detail += sep + text

    def _recommend(
        self,
        phase: str,
        info: InfoState,
        stake: StakeInfo | None,
        budget_s: float | None,
    ) -> Recommendation:
        stake = stake or StakeInfo()

        if phase == PHASE_CLAIM:
            return self._claim(info, budget_s)

        if phase == PHASE_RAISE_OFFER:
            return self._raise_offer(info, stake, budget_s)

        special = Ctx(self.rules, info.trump_suit).hand_special(info.my_hand)
        if special is not None:
            rec = Recommendation(
                phase,
                f"{special.split(' (')[0].upper()} — play all {len(info.my_hand)} cards",
                f"{self.names(info.my_hand)} wins the round outright",
                cards=tuple(info.my_hand),
                win_prob=1.0,
                lose_prob=0.0,
                expected=float(stake.current),
                exact=True,
            )
            if stake.can_raise and not stake.i_raised_last:
                nxt = self.rules.next_stake(stake.current)
                if nxt is not None:
                    rec.extra.append(
                        f"RAISE ({self.rules.stake_name(nxt)}) first — "
                        "a declined raise still wins the current stake, an accepted one wins more"
                    )
            return rec

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

        if best.exact or best.worlds:
            rec.detail = f"win {_pct(best.win_prob)} · lose {_pct(best.lose_prob)}"
        elif best.n_eff < MIN_SOLID_SAMPLES:
            rec.detail = f"win ≈{_pct(best.win_prob)} · lose ≈{_pct(best.lose_prob)} · ROUGH (too few deals)"
        else:
            rec.detail = (
                f"win {_pct(best.win_prob)} ±{_pct(1.96 * best.win_se)} · "
                f"lose {_pct(best.lose_prob)}"
            )
            if self.engine.last_tempered:
                rec.detail += " · opponent model tempered (his play did not fit it)"

        # Is the second-best action statistically indistinguishable from the best?
        # The two share the same sampled worlds, so their difference has at most the
        # variance of the two independent estimates combined.
        if len(results) > 1 and results[1].n and not best.exact:
            second = results[1]
            se_diff = math.sqrt(best.mean_se ** 2 + second.mean_se ** 2)
            if best.mean - second.mean < 1.5 * se_diff:
                rec.detail += " · close call: the next option is about as good"

        # Show up to three alternative actions; percentages only when they rest on enough samples.
        for r in results[1:4]:
            if not r.n:
                continue
            if r.exact or r.n_eff >= MIN_ALT_SAMPLES:
                rec.alternatives.append(
                    f"{self._describe(info, r)}  —  win {_pct(r.win_prob)}, lose {_pct(r.lose_prob)}"
                )
            else:
                rec.alternatives.append(
                    f"{self._describe(info, r)}  —  dropped early (clearly worse)"
                )

        # Raise if taking the next stake has higher EV than playing on at ``current``.
        if stake.can_raise and not stake.i_raised_last:
            nxt = self.rules.next_stake(stake.current)
            margin = float(self.rules.assistant("ev_margin") or 0.0)
            if nxt is not None:
                ev_stay = stake.current * _edge(best.win_prob, best.lose_prob)
                ev_raise = _ev_if_i_raise(stake.current, nxt, best.win_prob, best.lose_prob)
                if ev_raise > ev_stay + margin:
                    rec.extra.append(
                        f"RAISE ({self.rules.stake_name(nxt)}) — "
                        f"EV {ev_raise:+.2f} vs stay {ev_stay:+.2f}"
                    )

        return rec

    def _claim(self, info: InfoState, budget_s: float | None = None) -> Recommendation:
        p = self.engine.claim_probability(info)
        pts = info.my_points_known
        unk = (
            f" + {info.my_pile_unknown} unknown cards"
            if info.my_pile_unknown
            else ""
        )
        ev_claim = 2.0 * p - 1.0
        # Certain (or empty-pile impossible) cases skip a full continue-search.
        if p >= 1.0 - 1e-12:
            return Recommendation(
                PHASE_CLAIM,
                "CLAIM — press ვარ",
                (
                    f"you have {pts} pts{unk} "
                    f"(≥ {self.rules.win_points}); "
                    f"P={_pct(p)} EV(claim) {ev_claim:+.2f}"
                ),
                win_prob=p,
                expected=ev_claim,
            )
        pw, _pd, pl = self.engine.win_probability(info, budget_s)
        ev_cont = _edge(pw, pl)
        margin = float(self.rules.assistant("ev_margin") or 0.0)
        if p <= 0.0 and info.my_pile_unknown == 0:
            return Recommendation(
                PHASE_CLAIM,
                "CONTINUE — press გაგრძელება",
                (
                    f"if you continue: win {_pct(pw)} · lose {_pct(pl)} — "
                    f"claiming now would lose ({pts} pts, need {self.rules.win_points})"
                ),
                win_prob=pw,
                lose_prob=pl,
                expected=ev_cont,
            )
        if ev_claim > ev_cont + margin:
            return Recommendation(
                PHASE_CLAIM,
                "CLAIM — press ვარ",
                (
                    f"you have {pts} pts{unk}; "
                    f"P(enough)={_pct(p)}; "
                    f"EV(claim) {ev_claim:+.2f} vs continue {ev_cont:+.2f}"
                ),
                win_prob=p,
                expected=ev_claim,
            )

        return Recommendation(
            PHASE_CLAIM,
            "CONTINUE — press გაგრძელება",
            (
                f"if you continue: win {_pct(pw)} · lose {_pct(pl)} — "
                f"claiming now wins only {_pct(p)} ({pts} pts{unk}, need {self.rules.win_points})"
            ),
            win_prob=pw,
            lose_prob=pl,
            expected=ev_cont,
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
        counter = self.rules.next_stake(new)
        margin = float(self.rules.assistant("ev_margin") or 0.0)

        ev_counter = None
        if counter is not None:
            ev_counter = _ev_if_i_raise(new, counter, pw, pl)

        detail = (
            f"win {_pct(pw)} · "
            f"lose {_pct(pl)} · "
            f"accept EV {ev_accept:+.2f} "
            f"vs decline {ev_decline:+.2f}"
            + (f" vs counter {ev_counter:+.2f}" if ev_counter is not None else "")
            + (
                f" (break-even win chance "
                f"{_pct((new - cur) / (2 * new))} "
                f"with no draws)"
            )
        )

        best_kind = "decline"
        best_ev = ev_decline
        if ev_accept >= best_ev - 1e-12:
            best_kind, best_ev = "accept", ev_accept
        if ev_counter is not None and ev_counter > best_ev + margin:
            best_kind, best_ev = "counter", ev_counter

        if best_kind == "counter":
            return Recommendation(
                PHASE_RAISE_OFFER,
                f"COUNTER-RAISE to {self.rules.stake_name(counter)}",
                detail,
                win_prob=pw,
                lose_prob=pl,
                expected=ev_counter,
            )
        if best_kind == "accept":
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