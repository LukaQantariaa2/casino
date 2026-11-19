"""Entry point.

    python run.py                      live: watch Chrome, show the overlay
    python run.py --replay game.mp4    replay a recording through the overlay (testing)
    python run.py --console            no GUI, print recommendations in the terminal
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .app_core import Pipeline, load_config
from .ui import winapi

if getattr(sys, "frozen", False):          # PyInstaller build: config sits next to the exe
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_source(args, cfg):
    from .vision.capture import ScreenSource, VideoSource
    if args.replay:
        return VideoSource(args.replay, fps=cfg["capture"].get("fps", 10), realtime=True)
    cc = cfg["capture"]
    return ScreenSource(cc.get("window_title_hint", ""), cc.get("monitor_index", 1), cc.get("fixed_region"))


def run_console(args, cfg):
    from .controller import Controller
    pipe = Pipeline(cfg, ROOT, log_to_file=True)
    src = build_source(args, cfg)
    last = {"status": None, "hist": 0}

    def on_snap(s):
        if (s.status, tuple(s.reasons)) != last["status"]:
            last["status"] = (s.status, tuple(s.reasons))
            print(f"[{time.strftime('%H:%M:%S')}] {s.status} {'; '.join(s.reasons)}")
        evs = [ev for ev in pipe.history.events[last["hist"]:] if ev.kind != "advice"]
        last["hist"] = len(pipe.history.events)
        for ev in evs:
            print(f"[{time.strftime('%H:%M:%S')}]   {ev.text}")

    def on_rec(rec, key):
        if rec is not None:
            print(f"\n>>> {rec.headline}\n    {rec.detail}")
            for e in rec.extra:
                print(f"    + {e}")
            for a in rec.alternatives:
                print(f"    · {a}")
            print()

    ctrl = Controller(pipe, src, on_snap, on_rec, fps=cfg["capture"].get("fps", 10))
    ctrl.start()
    print("Domino Assistant running in console mode — Ctrl+C to stop. It never clicks or types for you.")
    try:
        while not ctrl.finished.is_set():
            time.sleep(0.3)
        time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    ctrl.stop()


def run_gui(args, cfg):
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        print("PySide6 is not installed - falling back to console mode (pip install PySide6 for the overlay).")
        return run_console(args, cfg)
    from .controller import Controller
    from .ui.overlay import Overlay

    app = QApplication(sys.argv)
    try:
        pipe = Pipeline(cfg, ROOT, log_to_file=True)
        src = build_source(args, cfg)
    except Exception as exc:
        QMessageBox.critical(None, "Domino Assistant", f"Start-up failed:\n{exc}")
        return 1
    holder = {}
    ctrl = Controller(pipe, src, lambda s: holder["ov"].push_snapshot(s),
                      lambda r, k: holder["ov"].push_recommendation(r, k), fps=cfg["capture"].get("fps", 10))
    ov = Overlay(ctrl, cfg)
    holder["ov"] = ov
    ov.show()
    hk = winapi.HotkeyThread(cfg.get("hotkeys", {}), ov.push_hotkey)
    hk.start()
    ctrl.start()
    code = app.exec()
    ctrl.stop()
    return code


def main(argv=None):
    winapi.set_dpi_aware()
    ap = argparse.ArgumentParser(description="Domino (All Fives) decision-support overlay (observes the screen only).")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "app.json"))
    ap.add_argument("--replay", help="replay a screen recording instead of capturing the screen")
    ap.add_argument("--console", action="store_true", help="no overlay; print to the terminal")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.console:
        return run_console(args, cfg)
    return run_gui(args, cfg)


if __name__ == "__main__":
    sys.exit(main() or 0)
