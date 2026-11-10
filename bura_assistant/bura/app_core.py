"""GUI-independent pipeline: frame -> observation -> tracker -> recommendation."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .engine.advisor import Advisor, Recommendation, StakeInfo
from .engine.search import DecisionEngine
from .rules import Rules
from .tracker.state import History
from .tracker.tracker import (PHASE_CLAIM, PHASE_LEAD, PHASE_RAISE_OFFER, PHASE_RESPOND, STATUS_OK, GameTracker,
                              TrackerConfig)
from .vision.capture import StabilityDetector
from .vision.ocr import MatchPanelReader
from .vision.recognizer import Observation, Recognizer
from .vision.templates import CardTemplates

DECISION_PHASES = (PHASE_LEAD, PHASE_RESPOND, PHASE_CLAIM, PHASE_RAISE_OFFER)


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
        self.rules = Rules.load(_resolve(root, paths["rules"]))
        with open(_resolve(root, paths["layout"]), "r", encoding="utf-8") as fh:
            self.layout = json.load(fh)
        cards_dir = _resolve(root, paths["cards_dir"])
        if not os.path.isdir(cards_dir):
            alt = os.path.join(root, "cards")
            if os.path.isdir(alt):
                cards_dir = alt
        self.templates = CardTemplates(self.rules, self.layout, cards_dir)
        self.recognizer = Recognizer(self.rules, self.layout, self.templates)
        logfile = None
        if log_to_file:
            d = _resolve(root, paths.get("logs_dir", "logs"))
            os.makedirs(d, exist_ok=True)
            logfile = open(os.path.join(d, time.strftime("session-%Y%m%d-%H%M%S.jsonl")), "a", encoding="utf-8")
        self.history = History(logfile)
        tc = cfg.get("tracker", {})
        self.tracker = GameTracker(self.rules, self.history,
                                   TrackerConfig(tc.get("uncertain_to_pause", 4), tc.get("lowconf_to_pause", 6)))
        ec = cfg["engine"]
        self.engine = DecisionEngine(self.rules, ec.get("time_budget_s", 1.2), ec.get("max_samples", 6000),
                                     opponent_model=ec.get("opponent_model", "lookahead"))
        self.advisor = Advisor(self.rules, self.engine)
        cc = cfg.get("capture", {})
        self.stability = StabilityDetector(self.layout, cc.get("still_frames", 2), cc.get("diff_threshold", 1.5))
        self.last_obs: Observation | None = None
        self.paused_by_user = False
        self.match = MatchPanelReader(self.layout) if cfg.get("ocr", {}).get("enabled", True) else None

    def process_frame(self, frame: np.ndarray, t: float) -> bool:
        """Feed one captured frame. Returns True if a new stable observation was processed."""
        if not self.stability.feed(frame):
            return False
        img = self.recognizer.normalize(frame)
        ob = self.recognizer.observe(img, t)
        self.last_obs = ob
        if not self.paused_by_user:
            self.tracker.update(ob)
        if self.match is not None:
            old = dict(self.match.values)
            try:
                if self.match.update(img, time.time()):
                    self._match_changed(old, self.match.values)
            except Exception as exc:                      # OCR is optional; never break tracking
                self.match.ocr.error = f"OCR error: {exc}"
                self.match.ocr.engine = None
        return True

    def _match_changed(self, old: dict, new: dict):
        v = new
        self.history.add("match", f"match score me {v['my_score']} / opp {v['opp_score']}, "
                                  f"target {v['target']}, bet {v['bet']}")
        for who, key in (("I", "my_score"), ("opponent", "opp_score")):
            if old.get(key) is not None and new.get(key) is not None and new[key] > old[key]:
                self.history.add("match", f"{who} gained {new[key] - old[key]} match point(s) this round")

    def stake_info(self) -> StakeInfo:
        r = self.tracker.round
        ob = self.last_obs
        if r is None:
            return StakeInfo()
        return StakeInfo(current=r.stake, offered=r.stake_offered, i_raised_last=r.i_raised_last,
                         can_raise=bool(ob and "raise" in ob.buttons))

    def decision_ready(self) -> bool:
        st = self.tracker.status
        return (not self.paused_by_user and st.status == STATUS_OK and st.phase in DECISION_PHASES
                and self.tracker.round is not None)

    def recommend_now(self, budget_s: float | None = None) -> Recommendation | None:
        if not self.decision_ready():
            return None
        r = self.tracker.round
        info = r.info()
        errs = info.check()
        if errs:
            return None
        return self.advisor.recommend(self.tracker.status.phase, info, self.stake_info(), budget_s)
