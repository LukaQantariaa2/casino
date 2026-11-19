"""Number reading for the left score panel (match scores, target, bet) and the raise offer.

Uses the OCR engine built into Windows 10/11 through the `winrt` (or older `winsdk`) Python
projections. If those packages are missing everything still works, but the engine then plays for
points instead of for the match (it needs both scores and the target for that).
"""
from __future__ import annotations

import asyncio
import os
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

    def __init__(self, layout: dict, interval: float = 0.5):
        self.R = layout["regions"]
        self.ocr = WindowsOcr()
        self.digits = DigitReader()
        self.interval = interval
        self.values: dict = {"my_score": None, "opp_score": None, "bet": None, "target": None}
        self._thumb = None
        self._last = 0.0
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        if self.ocr.ok:
            return "scores: OCR on"
        return "scores: digit templates only" if self.digits.ok else f"scores off — {self.ocr.error}"

    def _score(self, img, name):
        crop = self._crop(img, name)
        v = self.digits.read(crop)
        if v is None and self.ocr.ok:
            v = _first_int(self.ocr.read(crop))
        return v

    def _crop(self, img, name):
        x0, y0, x1, y1 = self.R[name]
        return img[y0:y1, x0:x1]

    def update(self, img: np.ndarray, now: float) -> bool:
        """Returns True when the values changed."""
        if not (self.ocr.ok or self.digits.ok) or now - self._last < self.interval:
            return False
        x0, y0 = self.R["opp_score"][:2]
        x1, y1 = self.R["my_score"][2:]
        th = cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (30, 98))
        if self._thumb is not None and float(np.abs(th.astype(int) - self._thumb.astype(int)).mean()) < 2.0:
            return False
        self._thumb = th
        self._last = now
        new = dict(self.values)
        new["opp_score"] = self._score(img, "opp_score")
        new["my_score"] = self._score(img, "my_score")
        if self.ocr.ok:
            txt = self.ocr.read(self._crop(img, "stake_box"))
            m = re.search(r"(\d+[.,]\d+)", txt)
            new["bet"] = float(m.group(1).replace(",", ".")) if m else None
            rest = txt[m.end():] if m else txt
            new["target"] = _first_int(rest) or self.values.get("target")
        with self._lock:
            changed = new != self.values
            self.values = new
        return changed

    def read_offer(self, img: np.ndarray) -> float | None:
        """The stake shown in the opponent's raise offer."""
        if not self.ocr.ok or "offer" not in self.R:
            return None
        m = re.search(r"(\d+[.,]?\d*)", self.ocr.read(self._crop(img, "offer")))
        return float(m.group(1).replace(",", ".")) if m else None


def _first_int(text: str):
    m = re.search(r"\d+", text or "")
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------- score digits by template
DIGITS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digits.npz")
GLYPH_W, GLYPH_H = 20, 32


def glyphs(crop: np.ndarray) -> list[np.ndarray]:
    """Bright digit shapes in a score box, left to right, each scaled to 20x32 (0..1)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    m = (g > 140).astype(np.uint8)
    n, _lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = [st[i] for i in range(1, n) if st[i][4] >= 30 and st[i][3] >= 0.25 * crop.shape[0]]
    boxes.sort(key=lambda s: s[0])
    out = []
    for x, y, w, h, _a in boxes:
        piece = m[y:y + h, x:x + w].astype(np.float32)
        out.append(cv2.resize(piece, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA))
    return out


class DigitReader:
    def __init__(self):
        self.ok = False
        try:
            d = np.load(DIGITS_FILE)
            self.tmpl = d["templates"]
            self.have = d["have"]
            self.ok = bool(self.have.all())
        except (OSError, KeyError, ValueError):
            pass

    def read(self, crop: np.ndarray) -> int | None:
        if not self.ok:
            return None
        gl = glyphs(crop)
        if not gl or len(gl) > 3:
            return None
        digits = []
        for g in gl:
            err = ((self.tmpl - g[None]) ** 2).mean(axis=(1, 2))
            d = int(np.argmin(err))
            if err[d] > 0.08:
                return None
            digits.append(d)
        return int("".join(map(str, digits)))
