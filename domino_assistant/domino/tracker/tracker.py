"""Turns stable screen observations into the game state.

Follows every play (who played which tile on which end), the opponent's draws and passes (what he
cannot hold), the round boundaries and the stake offer. Board changes are applied one tile at a
time; if the picture ever stops making sense (a tile that fits no end, tiles disappearing) the line
is rebuilt from the tile positions instead, which is also how a session started mid-round begins.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..engine.game import ARMS, ME, OPP, TILES, Board, Rules, tile_name
from ..engine.knowledge import OppKnowledge
from ..engine.search import Info
from ..vision.reader import Observation
from . import geometry

STATUS_OK, STATUS_UNCERTAIN, STATUS_PAUSED, STATUS_WAITING = "OK", "UNCERTAIN", "PAUSED", "WAITING"
PHASE_WAIT, PHASE_MY_TURN, PHASE_RAISE, PHASE_ROUND_OVER = "wait", "my_turn", "raise_offer", "round_over"


@dataclass
class Event:
    t: float
    kind: str
    text: str
    data: dict = field(default_factory=dict)


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

    def tail(self, n: int = 12, exclude: tuple[str, ...] = ("advice",)) -> list[Event]:
        out = [ev for ev in reversed(self.events) if ev.kind not in exclude][:n]
        out.reverse()
        return out


@dataclass
class Status:
    status: str = STATUS_WAITING
    phase: str = PHASE_WAIT
    reasons: list[str] = field(default_factory=list)


@dataclass
class RoundState:
    number: int = 0
    board: Board = field(default_factory=Board)
    tips: list = field(default_factory=lambda: [None] * 4)    # tile at the tip of each arm
    played: list = field(default_factory=list)                 # (who, tile, arm, points)
    my_hand: list = field(default_factory=list)
    opp_count: int = 7
    opp: OppKnowledge = field(default_factory=lambda: OppKnowledge(7))
    points: list = field(default_factory=lambda: [0, 0])       # points scored this round
    starter: int | None = None
    joined_late: bool = False
    seen_mine: set = field(default_factory=set)                # tiles seen in my hand (twice) this round
    hand_seen: dict = field(default_factory=dict)
    _board_tiles: list = field(default_factory=list)

    def unknown(self) -> list:
        known = set(self.board_tiles()) | set(self.my_hand)
        return [t for t in TILES if t not in known]

    def board_tiles(self) -> list:
        return list(self._board_tiles)

    @property
    def boneyard(self) -> int:
        return 28 - len(self._board_tiles) - len(self.my_hand) - self.opp_count


class Tracker:
    def __init__(self, rules: Rules, history: History, pause_after: int = 4):
        self.rules = rules
        self.history = history
        self.pause_after = pause_after
        self.round: RoundState | None = None
        self.status = Status()
        self.version = 0
        self.round_no = 0
        self.scores: dict = {"me": None, "opp": None, "target": None, "bet": None, "offer": None}
        self.last_obs: Observation | None = None
        self._bad = 0
        self._glitches = 0
        self._glitch_since = 0.0
        self._pending = None
        self._last_mover: int | None = None
        self._pass_noted = False
        self._raise_seen = False

    # ------------------------------------------------------------ helpers
    def _log(self, kind: str, text: str, **data):
        self.history.add(kind, text, **data)

    def _bump(self):
        self.version += 1

    def _set(self, status: str, phase: str | None = None, reasons=None):
        old = (self.status.status, self.status.phase, tuple(self.status.reasons))
        self.status.status = status
        if phase is not None:
            self.status.phase = phase
        self.status.reasons = list(reasons or [])
        if old != (self.status.status, self.status.phase, tuple(self.status.reasons)):
            self._bump()

    def manual_reset(self):
        self.round = None
        self._bad = 0
        self._log("info", "reset — re-reading the table")
        self._set(STATUS_WAITING, PHASE_WAIT, ["waiting for a clear view of the table"])

    def set_scores(self, values: dict):
        changed = any(self.scores.get(k) != v for k, v in values.items())
        self.scores.update(values)
        if changed:
            self._bump()

    # ------------------------------------------------------------ main entry
    def update(self, ob: Observation):
        self.last_obs = ob
        if ob.raise_offer:
            if not self._raise_seen:
                self._raise_seen = True
                self._log("stake", "opponent offers to double the stake")
            self._set(STATUS_OK, PHASE_RAISE)
            return
        self._raise_seen = False
        if ob.dimmed:
            self._set(STATUS_WAITING, PHASE_WAIT, ["table dimmed (round end or a message)"])
            return
        if ob.unreadable:
            return self._glitch(ob, f"{ob.unreadable} tile(s) could not be read")
        seen = ob.board_tiles()
        hand = ob.hand_tiles()
        if len(set(seen)) != len(seen) or len(set(hand)) != len(hand):
            return self._glitch(ob, "a tile was read twice")
        both = set(seen) & set(hand)
        if both:
            if self.round is None or not both <= set(self.round.board_tiles()):
                return self._glitch(ob, "a tile was read twice")
            hand = [t for t in hand if t not in both]      # the board copy is confirmed; the hand one is a misread
        if not seen:
            fresh = len(hand) == self.rules.hand_size and ob.opp_count == self.rules.hand_size
            mid = self.round is not None and (self.round.played or self.round.joined_late)
            if fresh and (self.round is None or mid):
                if mid and self._pending != frozenset():
                    self._pending = frozenset()       # an empty board must be seen twice
                    return
                self._pending = None
                if self.round is not None and self.round.played:
                    self._log("round", "round over")
                self._new_round(ob)
            elif self.round is not None:
                if len(hand) >= len(self.round.my_hand):
                    self.round.my_hand = hand          # a shorter hand here is a tile being dragged
                self._saw_hand(hand)
                self._sync_opp(ob)
            self._finish(ob)
            return
        if self.round is None:
            if ob.opp_count < 1 or not hand:
                return                                 # mid-round both players hold tiles; this frame is partial
            key = frozenset(seen) | {("opp", ob.opp_count), ("hand", len(hand))}
            if self._pending != key:
                self._pending = key                    # join only on a reading seen twice
                return
            self._pending = None
            if not self._join(ob):
                return self._glitch(ob, "joining mid-round: could not work out the line of play")
            self._finish(ob)
            return
        r = self.round
        known = set(r.board_tiles())
        new = [t for t in seen if t not in known]
        gone = [t for t in known if t not in set(seen)]
        if (new or gone) and self._pending != frozenset(seen):
            self._pending = frozenset(seen)           # a board change must be seen twice (tiles in flight)
            return
        self._pending = None
        if gone:
            return self._glitch(ob, "tiles vanished from the board: " + " ".join(tile_name(t) for t in gone))
        if new and not self._apply_new(ob, new):
            return self._glitch(ob, "could not place " + " ".join(tile_name(t) for t in new))
        if not new:
            self._note_draws(ob)
        if new or len(hand) >= len(r.my_hand):
            r.my_hand = hand                           # only a play makes my hand shorter
        self._saw_hand(hand)
        self._finish(ob)

    def _saw_hand(self, hand):
        r = self.round
        for t in hand:
            r.hand_seen[t] = r.hand_seen.get(t, 0) + 1
            if r.hand_seen[t] >= 2:
                r.seen_mine.add(t)

    # ------------------------------------------------------------ round boundaries
    def _new_round(self, ob: Observation):
        self.round_no += 1
        self.round = RoundState(number=self.round_no, my_hand=ob.hand_tiles(), opp_count=max(ob.opp_count, 0),
                                opp=OppKnowledge(ob.opp_count))
        self._saw_hand(ob.hand_tiles())
        self._last_mover = None
        self._pass_noted = False
        self._log("round", f"round {self.round_no}: my tiles " + " ".join(tile_name(t) for t in ob.hand_tiles()))
        self._bump()

    def _join(self, ob: Observation) -> bool:
        res = geometry.rebuild(ob.board)
        if res is None:
            return False
        board, tips = res
        self.round_no += 1
        r = RoundState(number=self.round_no, board=board, tips=tips, my_hand=ob.hand_tiles(),
                       opp_count=ob.opp_count, opp=OppKnowledge(ob.opp_count), joined_late=True)
        r._board_tiles = ob.board_tiles()
        self.round = r
        self._saw_hand(ob.hand_tiles())
        self._last_mover = None
        self._log("round", f"joined round in progress: {board.describe()}")
        self._bump()
        return True

    def _rebuild(self, ob: Observation) -> bool:
        res = geometry.rebuild(ob.board)
        if res is None:
            return False
        r = self.round
        old = set(r.board_tiles())
        r.board, r.tips = res
        r._board_tiles = ob.board_tiles()
        added = [t for t in r._board_tiles if t not in old]
        held = r.seen_mine | set(r.my_hand)
        for t in added:
            if t not in held:
                r.opp.played(t)
        if len(added) == 1:
            self._last_mover = ME if added[0] in held else OPP
        elif added:
            self._last_mover = None                    # several unseen plays: turn order unknown
        r.opp_count = ob.opp_count
        r.opp.resize(ob.opp_count)
        r.my_hand = [t for t in ob.hand_tiles() if t not in set(r._board_tiles)]
        self._log("info", f"line re-read from the screen: {r.board.describe()}")
        self._bump()
        return True

    def _glitch(self, ob: Observation, why: str):
        """A reading that does not fit. Usually a tile in motion: wait; if it persists, re-read the line."""
        r = self.round
        if r is not None and self.status.phase == PHASE_MY_TURN and len(ob.hand_tiles()) < len(r.my_hand):
            return                                     # I am dragging a tile; it hides part of the board
        if self._glitches == 0:
            self._glitch_since = ob.t
        self._glitches += 1
        if self._glitches < 4 or ob.t - self._glitch_since < 1.5:
            return
        if r is not None and not ob.unreadable:
            res = geometry.rebuild(ob.board)
            keeps_all = res is not None and set(r.board_tiles()) <= set(ob.board_tiles())
            if res is not None and (keeps_all or ob.t - self._glitch_since >= 8.0) and self._rebuild(ob):
                self._glitches = 0
                self._finish(ob)
                return
        self._trouble(why)

    # ------------------------------------------------------------ plays
    def _note_draws(self, ob: Observation):
        """Opponent count went up with no new tile: he drew because nothing fitted."""
        r = self.round
        k = ob.opp_count - r.opp_count
        if k > 0 and (k > r.boneyard - self.rules.boneyard_reserve or r.joined_late and not r.played):
            self._sync_opp(ob)                          # more than he could have drawn: a misread, resync
            return
        if k > 0 and r.board.n > 0:
            for _ in range(k):
                r.opp.drew(r.board.open_values())
            self._log("draw", f"opponent drew {ob.opp_count - r.opp_count} — he has no "
                              + ", ".join(str(v) for v in sorted(r.board.open_values())))
            r.opp_count = ob.opp_count
            self._bump()

    def _apply_new(self, ob: Observation, new: list) -> bool:
        r = self.round
        hand_now = set(ob.hand_tiles())
        opp_drop = r.opp_count - ob.opp_count
        i_shrank = len(hand_now) < len(r.my_hand)
        held = r.seen_mine | set(r.my_hand)
        mine = {t for t in new if t in held and t not in hand_now}
        if len(new) == 1 and opp_drop >= 1 and not i_shrank:
            mine = set()
        elif len(new) == 1 and i_shrank and opp_drop <= 0 and new[0] not in hand_now:
            mine = set(new)
        first = OPP if self._last_mover == ME else ME
        bone_before = r.boneyard
        bone_after = 28 - len(ob.board_tiles()) - len(hand_now) - ob.opp_count
        strict = (self._last_mover is not None and self.rules.can_draw(bone_before)
                  and self.rules.can_draw(bone_after))
        if strict:
            # nobody can pass while tiles can still be drawn, so turns alternate
            order = [first if k % 2 == 0 else first ^ 1 for k in range(len(new))]
            mine_list = [t for t in new if t in mine] + [t for t in new if t not in mine]
            if order[0] == OPP:
                mine_list.reverse()
            mine = {t for t, w in zip(mine_list, order) if w == ME}
            pending = mine_list
        else:
            pending = sorted(new, key=lambda t: (ME if t in mine else OPP) != first)
        opp_plays = sum(1 for t in new if t not in mine)
        drawn = max(0, ob.opp_count - (r.opp_count - opp_plays))
        while pending:
            progress = False
            for t in list(pending):
                arm = self._attach_arm(ob, t)
                if arm is None:
                    continue
                who = ME if t in mine else OPP
                if who == OPP and drawn > 0 and r.board.n > 0:
                    for _ in range(drawn):
                        r.opp.drew(r.board.open_values())
                    self._log("draw", f"opponent drew {drawn} — he has no "
                                      + ", ".join(str(v) for v in sorted(r.board.open_values())))
                    drawn = 0
                if (self._last_mover == who == ME and not self._pass_noted
                        and not self.rules.can_draw(r.boneyard)):
                    r.opp.passed(r.board.open_values())
                    self._log("pass", "opponent passed — he has no "
                                      + ", ".join(str(v) for v in sorted(r.board.open_values())))
                self._pass_noted = False
                self._play(who, t, arm, ob)
                pending.remove(t)
                progress = True
            if not progress:
                return False
        r.opp_count = ob.opp_count
        r.opp.resize(ob.opp_count)
        return True

    def _attach_arm(self, ob: Observation, t) -> int | None:
        r = self.round
        b = r.board
        if b.n == 0:
            return -1
        fits = [i for i, a in enumerate(b.arms) if a is not None and a[0] in t]
        if not fits:
            return None
        return self._touching_arm(ob, t, fits)

    def _touching_arm(self, ob: Observation, t, fits) -> int | None:
        r = self.round
        me = ob.find(t)
        if me is None:
            return None
        best = []
        for i in fits:
            tip = r.tips[i]
            st = ob.find(tip) if tip is not None else None
            if st is not None and geometry.touching(me, st):
                best.append(i)
        if len(best) == 1:
            return best[0]
        if best and r.board.spinner >= 0:                      # several arms start at the spinner
            sp = ob.find((r.board.spinner, r.board.spinner))
            if sp is not None:
                side = geometry.side_of(me.center, sp.center)
                vertical = sp.box[3] > sp.box[2]
                want = {"left": 0, "right": 1, "top": 2, "bottom": 3} if vertical else \
                    {"top": 0, "bottom": 1, "left": 2, "right": 3}
                k = want[side]
                if k in best:
                    return k
        return best[0] if best else None

    def _play(self, who: int, t, arm: int, ob: Observation):
        r = self.round
        r.board = r.board.place(t, arm)
        if arm == -1:
            r.tips = [t, t, None, None]
            r.starter = who
        else:
            r.tips[arm] = t
        if r.board.arms[2] is not None and r.tips[2] is None:
            sp = (r.board.spinner, r.board.spinner)
            r.tips[2] = r.tips[3] = sp
        r._board_tiles.append(t)
        pts = r.board.points()
        r.points[who] += pts
        r.played.append((who, t, arm, pts))
        if who == OPP:
            if not r.opp.played(t):
                self._log("info", f"opponent played {tile_name(t)}, which I thought he could not hold")
        self._last_mover = who
        name = "I" if who == ME else "opponent"
        where = "" if arm == -1 else f" on {ARMS[arm]}"
        self._log("play", f"{name} played {tile_name(t)}{where}" + (f" — scores {pts}" if pts else "")
                  + f"  [{r.board.describe()}]", who=who, tile=list(t), arm=arm, points=pts)
        self._bump()

    def _sync_opp(self, ob: Observation):
        r = self.round
        if ob.opp_count != r.opp_count:
            r.opp_count = ob.opp_count
            r.opp.resize(ob.opp_count)

    # ------------------------------------------------------------ status
    def _trouble(self, why: str):
        self._bad += 1
        if self._bad >= self.pause_after:
            self._set(STATUS_PAUSED, PHASE_WAIT, [why])
        else:
            self._set(STATUS_UNCERTAIN, None, [why])

    def _finish(self, ob: Observation):
        r = self.round
        self._bad = 0
        self._glitches = 0
        if r is None:
            self._set(STATUS_WAITING, PHASE_WAIT, ["waiting for a new round"])
            return
        if ob.my_turn:
            if (self._last_mover == ME and not self._pass_noted and r.board.n > 0
                    and ob.opp_count == r.opp_count and not self.rules.can_draw(r.boneyard)):
                r.opp.passed(r.board.open_values())
                self._pass_noted = True
                self._log("pass", "opponent passed — he has no "
                                  + ", ".join(str(v) for v in sorted(r.board.open_values())))
            legal = {m[0] for m in r.board.moves(r.my_hand)}
            shown = set(ob.playable())
            if legal != shown and r.board.n > 0:
                self._set(STATUS_UNCERTAIN, PHASE_MY_TURN,
                          ["the game allows " + " ".join(tile_name(t) for t in sorted(shown))
                           + ", my picture of the line allows " + " ".join(tile_name(t) for t in sorted(legal))])
                return
            self._set(STATUS_OK, PHASE_MY_TURN)
        else:
            self._set(STATUS_OK, PHASE_WAIT)

    def arm_labels(self) -> dict:
        """Screen side of each open end (left/right/top/bottom), for telling equal ends apart."""
        r, ob = self.round, self.last_obs
        if r is None or ob is None or not ob.board:
            return {}
        if r.board.spinner >= 0 and ob.find((r.board.spinner, r.board.spinner)) is not None:
            ref = ob.find((r.board.spinner, r.board.spinner)).center
        else:
            xs = [b.center[0] for b in ob.board]
            ys = [b.center[1] for b in ob.board]
            ref = (sum(xs) / len(xs), sum(ys) / len(ys))
        out = {}
        for i, tip in enumerate(r.tips):
            st = ob.find(tip) if tip is not None else None
            if st is None or r.board.arms[i] is None:
                continue
            if r.board.arms[i][2]:
                out[i] = geometry.side_of(st.center, ref) + " end"
            else:
                out[i] = {0: "left", 1: "right", 2: "top", 3: "bottom"}[i] + " of the spinner"
        return out

    # ------------------------------------------------------------ engine view
    def info(self) -> Info | None:
        r = self.round
        if r is None:
            return None
        on_board = set(r.board_tiles())
        hand = [t for t in r.my_hand if t not in on_board]        # a misread can never be a played tile
        return Info(r.board, hand, r.opp_count, r.unknown(), r.opp.forbidden(),
                    self.scores.get("me"), self.scores.get("opp"), self.scores.get("target"))
