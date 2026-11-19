"""Runs capture -> recognition -> tracking -> engine in background threads.

The UI (overlay or console) only receives immutable snapshots through callbacks.
The application never clicks, types or otherwise interacts with the browser.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from .app_core import Pipeline
from .engine.advisor import Recommendation
from .engine.game import ME, OPP, tile_name


@dataclass
class Snapshot:
    t: float
    status: str
    phase: str
    reasons: list[str]
    source: str
    fps: float
    round_no: int = 0
    board: str = ""
    count: int = 0
    boneyard: int = 0
    opp_count: int = 0
    round_points: tuple = (0, 0)
    opp_has_no: list[int] = field(default_factory=list)
    tile_grid: list[list[tuple[str, str]]] = field(default_factory=list)   # rows of (label, state)
    history: list[str] = field(default_factory=list)
    hidden: bool = False
    scores: dict = field(default_factory=dict)
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
            job = self.p.job() if self.p.decision_ready() else None
        self.on_recommendation(None, key)        # clear the old tip while calculating
        if job is None:
            return
        try:
            self._req.get_nowait()               # drop a stale job
        except queue.Empty:
            pass
        self._req.put((key, job))

    def _engine_loop(self):
        while not self.stop_flag.is_set():
            try:
                key, job = self._req.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                rec = self.p.run_job(job)
            except Exception as exc:
                traceback.print_exc()
                rec = Recommendation(job[0], "engine error", f"{type(exc).__name__}: {exc}")
            current = key == self._last_key      # still the current situation?
            self._log_advice(job, rec, time.time() - t0, current)
            if current:
                self.on_recommendation(rec, key)

    def _log_advice(self, job, rec: Recommendation, secs: float, shown: bool):
        """Write the recommendation to the session log (file only) so odd episodes can be replayed."""
        info = job[1]
        try:
            self.p.history.add(
                "advice", f"[{rec.phase}] {rec.headline} — {rec.detail}" + ("" if shown else " (stale, not shown)"),
                phase=rec.phase, headline=rec.headline, detail=rec.detail, win=rec.win_prob,
                alternatives=rec.alternatives, extra=rec.extra, seconds=round(secs, 2), shown=shown,
                deals=rec.deals, exact=rec.exact,
                board=info.board.describe() if info is not None else "",
                my_hand=[tile_name(t) for t in info.my_hand] if info is not None else [],
                opp_count=info.opp_count if info is not None else None,
                boneyard=info.boneyard if info is not None else None)
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

    # ------------------------------------------------------------------ snapshot
    def _emit_snapshot(self):
        with self.lock:
            snap = self._build_snapshot()
        self.on_snapshot(snap)

    def _build_snapshot(self) -> Snapshot:
        p = self.p
        tr = p.tracker
        reasons = list(tr.status.reasons)
        if self.error:
            reasons = [self.error] + reasons
        src = self.source.status() if hasattr(self.source, "status") else ""
        if self.finished.is_set():
            src += " (finished)"
        s = Snapshot(time.time(), tr.status.status, tr.status.phase, reasons, src, self._fps,
                     hidden=self.recs_hidden, scores=dict(tr.scores))
        if p.match is not None:
            s.ocr_status = p.match.status
        s.history = [ev.text for ev in p.history.tail(8)]
        r = tr.round
        if r is None:
            return s
        s.round_no = r.number
        s.board = r.board.describe() if r.board.n else "empty"
        s.count = r.board.count() if r.board.n else 0
        s.boneyard = max(0, r.boneyard)
        s.opp_count = r.opp_count
        s.round_points = (r.points[ME], r.points[OPP])
        s.opp_has_no = sorted(r.opp.missing_values())
        mine, on_board = set(r.my_hand), set(r.board_tiles())
        slots = r.opp.forbidden()
        for a in range(7):
            row = []
            for b in range(a, 7):
                t = (a, b)
                if t in mine:
                    st = "mine"
                elif t in on_board:
                    st = "board"
                elif slots and not any(a not in sl and b not in sl for sl in slots):
                    st = "boneyard"          # none of his tiles can be this one
                else:
                    st = "unseen"
                row.append((tile_name(t), st))
            s.tile_grid.append(row)
        return s
