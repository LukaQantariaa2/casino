"""GUI-independent pipeline: frame -> observation -> tracker -> recommendation."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .engine.advisor import Advisor, Recommendation
from .engine.game import Rules
from .engine.search import Engine
from .tracker.tracker import PHASE_MY_TURN, PHASE_RAISE, STATUS_OK, History, Tracker
from .vision.capture import StabilityDetector
from .vision.ocr import MatchPanelReader
from .vision.reader import Reader

DECISION_PHASES = (PHASE_MY_TURN, PHASE_RAISE)
RECHECK_S = 0.5          # re-read an unchanged still screen this often (confirms pending board changes)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve(root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


class Pipeline:
    def __init__(self, cfg: dict, root: str, log_to_file: bool = False):
        self.cfg = cfg
        self.root = root
        paths = cfg["paths"]
        with open(_resolve(root, paths["layout"]), "r", encoding="utf-8") as fh:
            self.layout = json.load(fh)
        rc = cfg.get("rules", {})
        self.rules = Rules(target=rc.get("default_target", 255), boneyard_reserve=rc.get("boneyard_reserve", 2))
        logfile = None
        if log_to_file:
            d = _resolve(root, paths.get("logs_dir", "logs"))
            os.makedirs(d, exist_ok=True)
            logfile = open(os.path.join(d, time.strftime("session-%Y%m%d-%H%M%S.jsonl")), "a", encoding="utf-8")
        self.history = History(logfile)
        self.reader = Reader(self.layout)
        self.tracker = Tracker(self.rules, self.history, cfg.get("tracker", {}).get("pause_after", 4))
        ec = cfg.get("engine", {})
        self.engine = Engine(self.rules, ec.get("time_budget_s", 2.0), ec.get("max_samples", 50000))
        self.advisor = Advisor(self.engine)
        cc = cfg.get("capture", {})
        self.stability = StabilityDetector(self.layout, cc.get("still_frames", 1), cc.get("diff_threshold", 2.5))
        self.match = MatchPanelReader(self.layout) if cfg.get("ocr", {}).get("enabled", True) else None
        self.last_img = None
        self._last_t = -1e9

    def process_frame(self, frame: np.ndarray, t: float) -> bool:
        """Feed one captured frame. Returns True if a new stable observation was processed."""
        changed = self.stability.feed(frame)
        recheck = (self.stability.still >= self.stability.still_frames and t - self._last_t >= RECHECK_S)
        if not changed and not recheck:
            return False
        self._last_t = t
        img = self.reader.normalize(frame)
        self.last_img = img
        ob = self.reader.observe(img, t)
        if self.match is not None and not ob.dimmed:
            try:
                if self.match.update(img, time.time()):
                    self._scores_changed()
            except Exception as exc:                       # OCR is optional; never break tracking
                self.match.ocr.error = f"OCR error: {exc}"
                self.match.ocr.engine = None
        if ob.raise_offer and self.match is not None and self.tracker.scores.get("offer") is None:
            self.tracker.set_scores({"offer": self.match.read_offer(img)})
        elif not ob.raise_offer and self.tracker.scores.get("offer") is not None:
            self.tracker.set_scores({"offer": None})
        self.tracker.update(ob)
        return True

    def _scores_changed(self):
        v = self.match.values
        target = v.get("target") or self.rules.target
        new = {"me": v.get("my_score"), "opp": v.get("opp_score"), "target": target, "bet": v.get("bet")}
        old = dict(self.tracker.scores)
        self.tracker.set_scores(new)
        if (old.get("me"), old.get("opp")) != (new["me"], new["opp"]):
            self.history.add("score", f"score: me {new['me']} – opp {new['opp']} (target {target}, bet {new['bet']})")

    def decision_ready(self) -> bool:
        st = self.tracker.status
        return st.status == STATUS_OK and st.phase in DECISION_PHASES

    def job(self):
        """Everything the engine thread needs, copied so the tracker can move on meanwhile."""
        tr = self.tracker
        if tr.status.phase == PHASE_RAISE:
            s = dict(tr.scores)
            return ("raise", tr.info(), s)
        info = tr.info()
        if info is None or info.check():
            return None
        return ("move", info, tr.arm_labels())

    def run_job(self, job, budget_s: float | None = None) -> Recommendation:
        kind = job[0]
        if kind == "raise":
            _, info, s = job
            return self.advisor.raise_offer(info, s.get("me"), s.get("opp"), s.get("target"), s.get("offer"),
                                            s.get("bet"))
        _, info, labels = job
        return self.advisor.recommend(info, labels, budget_s)
