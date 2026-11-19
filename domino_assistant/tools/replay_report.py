"""Text replay of a screen recording through the full pipeline (no overlay).

    python tools/replay_report.py recording.mp4 [--advice] [--budget 0.5] [--fps 10]

Prints every tracked event (plays, draws, passes, scores, rounds) with the video time, and with
--advice the recommendation at each of my turns.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino.app_core import Pipeline, load_config  # noqa: E402
from domino.vision.capture import VideoSource  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--advice", action="store_true")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=10)
    args = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, "config", "app.json"))
    pipe = Pipeline(cfg, ROOT)
    src = VideoSource(args.video, fps=args.fps)
    shown = 0
    last_key = None
    last_status = None
    while True:
        frame, t = src.read()
        if frame is None:
            break
        if not pipe.process_frame(frame, t):
            continue
        tr = pipe.tracker
        for ev in pipe.history.events[shown:]:
            print(f"{t:7.2f}  {ev.kind:6s} {ev.text}")
        shown = len(pipe.history.events)
        st = (tr.status.status, tr.status.phase, tuple(tr.status.reasons))
        if st != last_status and st[0] != "OK":
            print(f"{t:7.2f}  status {st[0]} {st[1]} {'; '.join(st[2])}")
        last_status = st
        if args.advice and pipe.decision_ready():
            key = (tr.version, tr.status.phase)
            if key != last_key:
                last_key = key
                job = pipe.job()
                if job is not None:
                    rec = pipe.run_job(job, args.budget)
                    print(f"{t:7.2f}  >>> {rec.headline} — {rec.detail}")
                    for a in rec.alternatives:
                        print(f"{'':9s}     · {a}")
                    for e in rec.extra:
                        print(f"{'':9s}     ! {e}")


if __name__ == "__main__":
    main()
