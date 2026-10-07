"""Headless replay of a screen recording through the full pipeline.

    python tools/replay_report.py path/to/recording.mp4 [--fps 10] [--advice]

Prints every game event the tracker derives, status changes and (with
--advice) the recommendation at each decision point.  Use it to check the
recognizer and tracker against your own recordings before playing live.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bura.app_core import Pipeline, load_config  # noqa: E402
from bura.vision.capture import VideoSource  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--advice", action="store_true", help="also run the decision engine at decision points")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "app.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["engine"]["time_budget_s"] = args.budget
    pipe = Pipeline(cfg, root=ROOT)
    src = VideoSource(args.video, fps=args.fps)
    seen_events = 0
    last_status = None
    last_phase = None
    t_start = time.time()
    n_obs = 0
    while True:
        frame, t = src.read()
        if frame is None:
            break
        changed = pipe.process_frame(frame, t)
        if not changed:
            continue
        n_obs += 1
        evs = pipe.history.events[seen_events:]
        seen_events = len(pipe.history.events)
        for ev in evs:
            print(f"[{t:7.2f}s] {ev.kind:13s} {ev.text}")
        st = pipe.tracker.status
        if (st.status, tuple(st.reasons)) != last_status:
            last_status = (st.status, tuple(st.reasons))
            print(f"[{t:7.2f}s] -- status {st.status} {'; '.join(st.reasons)}")
        if st.phase != last_phase:
            last_phase = st.phase
            if args.advice and st.status == "OK" and st.phase in ("lead", "respond", "claim", "raise_offer"):
                rec = pipe.recommend_now()
                if rec:
                    print(f"[{t:7.2f}s] >> {st.phase.upper()}: {rec.headline}   ({rec.detail})")
                    for e in rec.extra:
                        print(f"             + {e}")
            elif st.phase != "wait":
                print(f"[{t:7.2f}s] -- phase {st.phase}")
    print(f"\nprocessed {src.idx} frames, {n_obs} stable observations in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
