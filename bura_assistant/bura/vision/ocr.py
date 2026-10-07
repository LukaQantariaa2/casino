"""Optional number reading for the left score panel (match scores, target, bet).

Uses the OCR engine built into Windows 10/11 through the `winrt` (or older `winsdk`) Python
projections.  If those packages are missing everything still works; the values just show as "?".
Nothing here is needed for card play; it is used for display and to cross-check the stake.
"""
from __future__ import annotations

import asyncio
import re
import threading

import cv2
import numpy as np


def _load_winrt():
    try:
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.security.cryptography import CryptographicBuffer
        return OcrEngine, Language, SoftwareBitmap, BitmapPixelFormat, CryptographicBuffer
    except Exception:
        pass
    try:
        from winsdk.windows.globalization import Language
        from winsdk.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.security.cryptography import CryptographicBuffer
        return OcrEngine, Language, SoftwareBitmap, BitmapPixelFormat, CryptographicBuffer
    except Exception:
        return None


class WindowsOcr:
    def __init__(self):
        self.mods = _load_winrt()
        self.engine = None
        self.error = ""
        if self.mods is None:
            self.error = "Windows OCR packages not installed (optional)"
            return
        OcrEngine, Language, *_ = self.mods
        try:
            self.engine = OcrEngine.try_create_from_language(Language("en-US")) \
                or OcrEngine.try_create_from_user_profile_languages()
        except Exception as exc:
            self.error = f"OCR unavailable: {exc}"
        if self.engine is None and not self.error:
            self.error = "no OCR language installed"

    @property
    def ok(self) -> bool:
        return self.engine is not None

    def read(self, bgr: np.ndarray) -> str:
        if not self.ok:
            return ""
        _OcrEngine, _Language, SoftwareBitmap, BitmapPixelFormat, CryptographicBuffer = self.mods
        # dark text on white, enlarged: much better for small digits
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        g = 255 - g
        bgra = cv2.cvtColor(g, cv2.COLOR_GRAY2BGRA)
        h, w = bgra.shape[:2]
        buf = CryptographicBuffer.create_from_byte_array(bytes(bgra.tobytes()))
        bmp = SoftwareBitmap.create_copy_from_buffer(buf, BitmapPixelFormat.BGRA8, w, h)

        async def run():
            return await self.engine.recognize_async(bmp)

        try:
            res = asyncio.run(run())
            return res.text or ""
        except Exception as exc:
            self.error = f"OCR failed: {exc}"
            return ""


class MatchPanelReader:
    """Reads the left panel at most every `interval` seconds and only when it changed."""

    def __init__(self, layout: dict, interval: float = 1.5):
        self.R = layout["regions"]
        self.ocr = WindowsOcr()
        self.interval = interval
        self.values: dict = {"my_score": None, "opp_score": None, "bet": None, "target": None}
        self._thumb = None
        self._last = 0.0
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        return "OCR on" if self.ocr.ok else self.ocr.error

    def _crop(self, img, name):
        x0, y0, x1, y1 = self.R[name]
        return img[y0:y1, x0:x1]

    def update(self, img: np.ndarray, now: float) -> bool:
        """Returns True when the values changed."""
        if not self.ocr.ok or now - self._last < self.interval:
            return False
        x0, y0 = self.R["opp_score"][:2]
        x1, y1 = self.R["my_score"][2:]
        th = cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (30, 98))
        if self._thumb is not None and float(np.abs(th.astype(int) - self._thumb.astype(int)).mean()) < 2.0:
            return False
        self._thumb = th
        self._last = now
        new = dict(self.values)
        new["opp_score"] = _first_int(self.ocr.read(self._crop(img, "opp_score")))
        new["my_score"] = _first_int(self.ocr.read(self._crop(img, "my_score")))
        txt = self.ocr.read(self._crop(img, "stake_box"))
        m = re.search(r"(\d+[.,]\d+)", txt)
        new["bet"] = float(m.group(1).replace(",", ".")) if m else None
        rest = txt[m.end():] if m else txt
        new["target"] = _first_int(rest)
        with self._lock:
            changed = new != self.values
            self.values = new
        return changed


def _first_int(text: str):
    m = re.search(r"\d+", text or "")
    return int(m.group(0)) if m else None
