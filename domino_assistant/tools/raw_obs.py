"""Print raw reader observations for a time window of a recording.

    python tools/raw_obs.py recording.mp4 START END [--fps 4]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino.app_core import load_config  # noqa: E402
from domino.engine.game import tile_name  # noqa: E402
from domino.vision.capture import VideoSource  # noqa: E402
from domino.vision.reader import Reader  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("start", type=float)
    ap.add_argument("end", type=float)
    ap.add_argument("--fps", type=float, default=4)
    args = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, "config", "app.json"))
    import json
    with open(os.path.join(ROOT, cfg["paths"]["layout"]), encoding="utf-8") as f:
        layout = json.load(f)
    reader = Reader(layout)
    src = VideoSource(args.video, fps=args.fps)
    while True:
        frame, t = src.read()
        if frame is None or t > args.end:
            break
        if t < args.start:
            continue
        ob = reader.observe(frame, t)
        hand = " ".join(tile_name(s.tile) + ("*" if s.color == "white" else "") for s in ob.hand)
        board = " ".join(sorted(tile_name(t_) for t_ in ob.board_tiles()))
        print(f"{t:7.2f} opp={ob.opp_count} dim={int(ob.dimmed)} hand=[{hand}] board=[{board}]")


if __name__ == "__main__":
    main()
