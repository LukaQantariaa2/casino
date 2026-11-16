"""Runs capture -> recognition -> tracking -> engine in background threads.

The UI (overlay or console) only receives immutable snapshots through callbacks.
The application never clicks, types or otherwise interacts with the browser.
"""
from __future__ import annotations

import copy
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from .app_core import Pipeline
from .engine.advisor import Recommendation
from .engine.core import ME, OPP
from .tracker.state import RoundState


@dataclass
class Snapshot:
    t: float
    status: str
    phase: str
    reasons: list[str]
    source: str
    fps: float
    round_no: int = 0
    stake: int = 1
    stake_name: str = "1"
    trump: str = ""
    deck: int = 0
    opp_hand: int = 0
    my_points: int = 0
    my_unknown: int = 0
    opp_points: int = 0
    opp_unknown: int = 0
    my_hand: list[str] = field(default_factory=list)
    card_grid: list[list[tuple[str, str]]] = field(default_factory=list)   # rows of (label, state)
    history: list[str] = field(default_factory=list)
    hidden: bool = False
    win_points: int = 61
    match: dict = field(default_factory=dict)
    ocr_status: str = ""


class Controller:
    def __init__(self, pipeline: Pipeline, source, on_snapshot, on_recommendation, fps: float = 10.0):
        self.p = pipeline
        self.source = source
        self.on_snapshot = on_snapshot
        self.on_recommendation = on_recommendation
        self.period = 1.0 / max(1.0, fps)
        self.lock = threading.RLock()
        self.stop_flag = threading.Event()
        self.recs_hidden = False
        self._req: queue.Queue = queue.Queue(maxsize=1)
        self._last_key = None
        self._fps = 0.0
        self.error: str | None = None
        self.finished = threading.Event()

    # ------------------------------------------------------------------ threads
    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True, name="capture").start()
        threading.Thread(target=self._engine_loop, daemon=True, name="engine").start()

    def stop(self):
        self.stop_flag.set()
        self.p.engine.stop_workers()

    def _capture_loop(self):
        n, t0 = 0, time.time()
        while not self.stop_flag.is_set():
            t_start = time.time()
            try:
                frame, ts = self.source.read()
                if frame is None:
                    self.finished.set()
                    self._emit_snapshot()
                    break
                with self.lock:
                    changed = self.p.process_frame(frame, ts)
                n += 1
                if time.time() - t0 >= 1.0:
                    self._fps = n / (time.time() - t0)
                    n, t0 = 0, time.time()
                if changed:
                    self._emit_snapshot()
                    self._maybe_request()
            except Exception as exc:  # keep running; show the error
                self.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                time.sleep(0.5)
            dt = time.time() - t_start
            if dt < self.period:
                time.sleep(self.period - dt)

    def _maybe_request(self):
        with self.lock:
            tr = self.p.tracker
            key = (tr.version, tr.status.phase, tr.status.status)
            if key == self._last_key:
                return
            self._last_key = key
            if not self.p.decision_ready():
                self.on_recommendation(None, key)
                return
            info = tr.round.info()
            if info.check():
                self.on_recommendation(None, key)
                return
            job = (key, tr.status.phase, info, self.p.stake_info())
        self.on_recommendation(None, key)        # clear the old tip while calculating
        try:
            self._req.get_nowait()            # drop a stale job
        except queue.Empty:
            pass
        self._req.put(job)

    def _engine_loop(self):
        while not self.stop_flag.is_set():
            try:
                key, phase, info, stake = self._req.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                rec = self.p.advisor.recommend(phase, info, stake)
            except Exception as exc:
                traceback.print_exc()
                rec = Recommendation(phase, "engine error", f"{type(exc).__name__}: {exc}")
            current = key == self._last_key      # still the current situation?
            self._log_advice(phase, info, rec, time.time() - t0, current)
            if current:
                self.on_recommendation(rec, key)

    def _log_advice(self, phase: str, info, rec: Recommendation, secs: float, shown: bool):
        """Write the recommendation to the session log (file only) so odd episodes can be replayed."""
        R = self.p.rules
        try:
            self.p.history.add(
                "advice",
                f"[{phase}] {rec.headline} — {rec.detail}" + ("" if shown else " (stale, not shown)"),
                phase=phase, headline=rec.headline, detail=rec.detail,
                cards=[R.card_name(c) for c in rec.cards],
                win=rec.win_prob, lose=rec.lose_prob, expected=rec.expected, exact=rec.exact,
                alternatives=rec.alternatives, extra=rec.extra, seconds=round(secs, 2), shown=shown,
                my_hand=[R.card_name(c) for c in info.my_hand], deck=info.deck_size,
                opp_hand=info.opp_hand_size, lead=[R.card_name(c) for c in info.current_lead],
                my_pts=info.my_points_known, opp_pts=info.opp_points_known,
                my_unknown=info.my_pile_unknown, opp_unknown=info.opp_pile_unknown,
                deals_checked=rec.deals_checked, deals_possible=rec.deals_possible, deals_all=rec.deals_all,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ user actions (thread-safe)
    def toggle_recommendations(self):
        self.recs_hidden = not self.recs_hidden
        self._emit_snapshot()

    def reset_round(self):
        with self.lock:
            self.p.tracker.manual_reset()
            self._last_key = None
        self._emit_snapshot()

    def adjust_stake(self, delta_levels: int):
        with self.lock:
            r = self.p.tracker.round
            if r is None:
                return
            vals = sorted(lv["value"] for lv in self.p.rules.stake_levels)
            i = vals.index(r.stake) if r.stake in vals else 0
            i = max(0, min(len(vals) - 1, i + delta_levels))
            r.stake = vals[i]
            self.p.tracker._log("stake", f"stake set manually to {self.p.rules.stake_name(r.stake)} ({r.stake})")
            self._last_key = None
        self._emit_snapshot()
        self._maybe_request()

    def current_round_copy(self) -> RoundState | None:
        with self.lock:
            r = self.p.tracker.round
            return copy.deepcopy(r) if r is not None else None

    def last_hand(self) -> list[int]:
        with self.lock:
            ob = self.p.last_obs
            return ob.hand_ids() if ob else []

    def apply_correction(self, rs: RoundState):
        with self.lock:
            self.p.tracker.apply_manual(rs)
            self._last_key = None
        self._emit_snapshot()
        self._maybe_request()

    # ------------------------------------------------------------------ snapshot
    def _emit_snapshot(self):
        with self.lock:
            snap = self._build_snapshot()
        self.on_snapshot(snap)

    def _build_snapshot(self) -> Snapshot:
        p = self.p
        tr = p.tracker
        R = p.rules
        reasons = list(tr.status.reasons)
        if self.error:
            reasons = [self.error] + reasons
        src = self.source.status() if hasattr(self.source, "status") else ""
        if self.finished.is_set():
            src += " (finished)"
        s = Snapshot(time.time(), tr.status.status, tr.status.phase, reasons, src, self._fps,
                     hidden=self.recs_hidden, win_points=R.win_points)
        r = tr.round
        s.history = [ev.text for ev in p.history.tail(8)]
        if p.match is not None:
            s.match = dict(p.match.values)
            s.ocr_status = p.match.status
        if r is None:
            return s
        s.round_no = r.number
        s.stake = r.stake
        s.stake_name = R.stake_name(r.stake)
        s.trump = R.card_name(r.trump_card) if r.trump_card is not None else "?"
        s.deck = r.deck_size
        s.opp_hand = r.opp_hand_size
        s.my_points = r.points(ME)
        s.opp_points = r.points(OPP)
        s.my_unknown = r.my_pile_unknown
        s.opp_unknown = r.opp_pile_unknown
        s.my_hand = [R.card_name(c) for c in r.my_hand]
        # card grid: one row per suit, columns in strength order
        order = sorted(range(len(R.ranks)), key=lambda i: R.strength[i])
        on_table = set()
        if r.trick:
            on_table = set(r.trick.lead) | set(r.trick.response)
        for si, _suit in enumerate(R.suits):
            row = []
            for ri in order:
                c = si * len(R.ranks) + ri
                if c in r.my_hand:
                    st = "mine"
                elif c in on_table:
                    st = "table"
                elif c in r.my_pile:
                    st = "my_pile"
                elif c in r.opp_pile:
                    st = "opp_pile"
                elif c in r.opp_known:
                    st = "opp_hand"
                elif c == r.trump_card and r.trump_in_deck:
                    st = "trump"
                else:
                    st = "unseen"
                row.append((R.card_name(c), st))
            s.card_grid.append(row)
        return s
