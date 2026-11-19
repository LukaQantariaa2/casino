"""Turns engine results into the text shown on the overlay."""
from __future__ import annotations

from dataclasses import dataclass, field

from .game import ME, OPP, tile_name
from .search import Engine, Info

ACCEPT_RAISE_FROM = 0.25     # declining is assumed to lose the match at the current stake


@dataclass
class Recommendation:
    phase: str
    headline: str
    detail: str = ""
    alternatives: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    move: tuple | None = None
    win_prob: float | None = None
    deals: int = 0
    exact: bool = False


def _count(n: int) -> str:
    return f"{n:,}".replace(",", " ")


class Advisor:
    def __init__(self, engine: Engine):
        self.engine = engine

    # ------------------------------------------------------------ my turn
    def recommend(self, info: Info, labels: dict | None = None, budget_s: float | None = None) -> Recommendation:
        moves = info.board.moves(info.my_hand)
        if not moves:
            return Recommendation("my_turn", "nothing fits — the game draws or passes for you")
        match_mode = info.my_score is not None and info.opp_score is not None and bool(info.target)
        budget = budget_s
        if len(moves) == 1:
            budget = min(0.4, budget or self.engine.budget)
        res = self.engine.evaluate(info, moves, budget)
        best = res[0]
        rec = Recommendation("my_turn", self._move_text(info, best.move, labels, moves), move=best.move,
                             deals=self.engine.last_deals, exact=self.engine.last_exact > 0)
        if len(moves) == 1:
            rec.headline += "  (only move)"
        if match_mode:
            rec.win_prob = best.mean
            rec.detail = f"match win {best.mean:.0%}"
            if self.engine.last_exact == 0:
                rec.detail += f" · this round {best.mean_pts:+.1f} pts vs him"
        else:
            rec.detail = f"this round {best.mean_pts:+.1f} pts vs him · wins the round {best.round_win:.0%}"
        rec.detail += f" · {self._deals_text()}"
        for r in res[1:4]:
            if match_mode:
                diff = (r.mean - best.mean) * 100
                txt = f"{r.mean:.0%} ({diff:+.1f}%)"
            else:
                txt = f"{r.mean_pts:+.1f} pts ({r.mean_pts - best.mean_pts:+.1f})"
            rec.alternatives.append(f"{self._move_text(info, r.move, labels, moves, short=True)}: {txt}")
        rec.extra = self._notes(info)
        if self.engine.constraints_dropped:
            rec.extra.append("what his draws/passes said contradicts the board — ignored for this tip")
        return rec

    def _deals_text(self) -> str:
        e = self.engine
        if e.last_exact:
            return f"{_count(e.last_unique)} deals solved exactly"
        return f"{_count(e.last_deals)} deals calculated"

    @staticmethod
    def _move_text(info: Info, move, labels, moves, short: bool = False) -> str:
        t, arm = move
        if arm == -1:
            return f"{'' if short else 'Play '}{tile_name(t)}"
        b2 = info.board.place(t, arm)
        pts = b2.points()
        a = info.board.arms[arm]
        same_tile = [m for m in moves if m[0] == t]
        where = f"on the {a[0]}"
        if len(same_tile) > 1:
            if labels and labels.get(arm):
                where += f" ({labels[arm]})"
            elif any(info.board.arms[m[1]][0] == a[0] for m in same_tile if m[1] != arm):
                where += " (end with a double)" if a[1] else " (plain end)"
        txt = f"{'' if short else 'Play '}{tile_name(t)} {where}"
        if pts:
            txt += f"  +{pts}"
        return txt

    @staticmethod
    def _notes(info: Info) -> list[str]:
        notes = []
        if info.opp_forbidden and len(info.opp_forbidden) == info.opp_count:
            common = set(info.opp_forbidden[0])
            for s in info.opp_forbidden[1:]:
                common &= s
            if common:
                notes.append("opponent has no " + ", ".join(str(v) for v in sorted(common)))
        return notes

    # ------------------------------------------------------------ raise offer
    def raise_offer(self, info: Info | None, my_score, opp_score, target, offer=None, bet=None) -> Recommendation:
        if my_score is None or opp_score is None or not target:
            return Recommendation("raise_offer", "raise offered — scores unreadable, no advice",
                                  "the match win chance needs both scores and the target")
        tab = self.engine.table(target)
        p = 0.5 * (tab.after_round(my_score, opp_score, ME) + tab.after_round(my_score, opp_score, OPP))
        accept = p >= ACCEPT_RAISE_FROM
        stake = f" to {offer:g}" if offer else ""
        head = f"ACCEPT the raise{stake}" if accept else "DECLINE the raise"
        detail = (f"match win ≈ {p:.0%} at {my_score}–{opp_score} (target {target}); "
                  f"accepting pays if above {ACCEPT_RAISE_FROM:.0%}")
        return Recommendation("raise_offer", head, detail, win_prob=p)
