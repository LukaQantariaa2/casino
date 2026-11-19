"""Frame sources (live screen / recorded video) and the change detector."""
from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------- window lookup (Windows only)
@dataclass
class WindowInfo:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int]   # left, top, right, bottom (screen pixels)


def find_chrome_window(title_hint: str = "") -> WindowInfo | None:
    """Return the largest visible Chrome window (optionally whose title contains title_hint)."""
    if sys.platform != "win32":
        return None
    import ctypes.wintypes as wt
    user32 = ctypes.windll.user32
    found: list[WindowInfo] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

    def cb(hwnd, _lp):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        if cls.value != "Chrome_WidgetWin_1":
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        if not title or (title_hint and title_hint.lower() not in title.lower()):
            return True
        rect = wt.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        found.append(WindowInfo(int(hwnd or 0), title, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    if not found:
        return None
    return max(found, key=lambda w: (w.rect[2] - w.rect[0]) * (w.rect[3] - w.rect[1]))


# ---------------------------------------------------------------- sources
class ScreenSource:
    """Captures the Chrome window (or a fixed monitor region) with mss."""

    def __init__(self, title_hint: str = "", monitor_index: int = 1, region: tuple | None = None):
        import mss  # imported lazily so replay mode works without it
        self._mss_mod = mss
        self.mss = None          # created in the capture thread (mss handles are thread-bound)
        self.title_hint = title_hint
        self.monitor_index = monitor_index
        self.fixed_region = region
        self.window: WindowInfo | None = None
        self._last_lookup = 0.0
        self._last_size: tuple[int, int] | None = None

    def _region(self) -> dict:
        if self.fixed_region:
            l, t, w, h = self.fixed_region
            return {"left": l, "top": t, "width": w, "height": h}
        now = time.time()
        if now - self._last_lookup > 2.0:
            self.window = find_chrome_window(self.title_hint)
            self._last_lookup = now
        if self.window:
            l, t, r, b = self.window.rect
            mons = self.mss.monitors[1:]
            # clip to the monitor that holds the window centre
            cx, cy = (l + r) // 2, (t + b) // 2
            for m in mons:
                if m["left"] <= cx < m["left"] + m["width"] and m["top"] <= cy < m["top"] + m["height"]:
                    l, t = max(l, m["left"]), max(t, m["top"])
                    r, b = min(r, m["left"] + m["width"]), min(b, m["top"] + m["height"])
                    break
            if r - l > 200 and b - t > 200:
                return {"left": l, "top": t, "width": r - l, "height": b - t}
        return dict(self.mss.monitors[self.monitor_index])

    def status(self) -> str:
        if self.fixed_region:
            return "fixed region"
        size = f" {self._last_size[0]}x{self._last_size[1]}" if self._last_size else ""
        warn = ""
        if self._last_size and abs(self._last_size[0] / self._last_size[1] - 16 / 9) > 0.02:
            warn = "  ⚠ not 16:9 — put Chrome in fullscreen (F11)"
        if self.window:
            return f"Chrome{size}: {self.window.title[:40]}{warn}"
        return f"Chrome window not found — capturing monitor{size}{warn}"

    def read(self) -> tuple[np.ndarray | None, float]:
        if self.mss is None:
            self.mss = self._mss_mod.mss()
        reg = self._region()
        self._last_size = (reg["width"], reg["height"])
        shot = self.mss.grab(reg)
        frame = np.asarray(shot)[:, :, :3]          # BGRA -> BGR
        return np.ascontiguousarray(frame), time.time()


class VideoSource:
    """Replays a screen recording (for testing and QA)."""

    def __init__(self, path: str, fps: float = 10.0, realtime: bool = False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, int(round(self.src_fps / fps)))
        self.realtime = realtime
        self.idx = 0
        self._t0 = time.time()

    def status(self) -> str:
        return f"video frame {self.idx}"

    def read(self) -> tuple[np.ndarray | None, float]:
        frame = None
        for _ in range(self.step):
            ok, frame = self.cap.read()
            self.idx += 1
            if not ok:
                return None, 0.0
        t = self.idx / self.src_fps
        if self.realtime:
            delay = self._t0 + t - time.time()
            if delay > 0:
                time.sleep(delay)
        return frame, t


# ---------------------------------------------------------------- change detection
class StabilityDetector:
    """Reports when the watched regions changed and then stayed still."""

    def __init__(self, layout: dict, still_frames: int = 2, diff_threshold: float = 1.5):
        self.ref_w, self.ref_h = layout["reference_size"]
        self.regions = [tuple(layout["regions"][n]) for n in layout["stability_regions"]]
        self.still_frames = still_frames
        self.thr = diff_threshold
        self.prev = None
        self.still = 0
        self.last_emitted = None

    def _thumb(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != self.ref_w or frame.shape[0] != self.ref_h:
            frame = cv2.resize(frame, (self.ref_w, self.ref_h), interpolation=cv2.INTER_AREA)
        parts = []
        for x0, y0, x1, y1 in self.regions:
            p = cv2.resize(frame[y0:y1, x0:x1], ((x1 - x0) // 8, (y1 - y0) // 8), interpolation=cv2.INTER_AREA)
            parts.append(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).ravel())
        return np.concatenate(parts).astype(np.int16)

    def feed(self, frame: np.ndarray) -> bool:
        """True when this frame is (nearly) still and differs from the last emitted frame."""
        th = self._thumb(frame)
        if self.prev is not None and np.abs(th - self.prev).mean() < self.thr:
            self.still += 1
        else:
            self.still = 0
        self.prev = th
        if self.still >= self.still_frames:
            if self.last_emitted is None or np.abs(th - self.last_emitted).mean() >= 0.6:
                self.last_emitted = th
                return True
        return False
