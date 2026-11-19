"""Reads the Adjarabet domino table from a screenshot (Chrome fullscreen, any 16:9 size).

Tiles are found as connected blobs of tile colour separated by the thin green gaps the game draws
between them: white (board tiles and my playable tiles), grey (my tiles that cannot be played now)
and black (the spinner, with white pips). Each tile is split in two halves and the pips counted;
pip size is judged relative to the tile because the board shrinks as the line grows. Position on the
table decides what a tile is: the top row is the opponent's face-down tiles, the bottom row my hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

REF_W, REF_H = 1920, 1080


@dataclass
class SeenTile:
    tile: tuple[int, int] | None          # None if the pips could not be read
    box: tuple[int, int, int, int]        # x, y, w, h in 1920x1080 coordinates
    color: str                            # "white", "grey" or "black"
    halves: tuple[int, int] = (0, 0)      # pips per half (left/top first)

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2

    @property
    def short(self) -> int:
        return min(self.box[2], self.box[3])


@dataclass
class Observation:
    t: float
    board: list[SeenTile] = field(default_factory=list)
    hand: list[SeenTile] = field(default_factory=list)
    opp_count: int = 0
    dimmed: bool = False
    raise_offer: bool = False
    unreadable: int = 0

    @property
    def my_turn(self) -> bool:
        return any(h.color == "white" for h in self.hand)

    def hand_tiles(self) -> list:
        return [h.tile for h in self.hand if h.tile is not None]

    def playable(self) -> list:
        return [h.tile for h in self.hand if h.tile is not None and h.color == "white"]

    def board_tiles(self) -> list:
        return [b.tile for b in self.board if b.tile is not None]

    def find(self, t) -> SeenTile | None:
        for b in self.board:
            if b.tile == t:
                return b
        return None


class Reader:
    def __init__(self, layout: dict | None = None):
        L = layout or {}
        self.table = tuple(L.get("table", (330, 160, 1720, 920)))         # x0, y0, x1, y1
        self.opp_row_bottom = L.get("opp_row_bottom", 300)
        self.hand_row_top = L.get("hand_row_top", 760)
        self.felt_probe = tuple(L.get("felt_probe", (345, 330, 440, 760)))
        self.no_button = tuple(L.get("raise_no_button", (1050, 425, 1170, 450)))
        self.yes_button = tuple(L.get("raise_yes_button", (880, 425, 990, 450)))

    @staticmethod
    def normalize(frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != REF_W or frame.shape[0] != REF_H:
            frame = cv2.resize(frame, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
        return frame

    # ------------------------------------------------------------ pips
    @staticmethod
    def _count(half: np.ndarray, s: int, dark: bool) -> int:
        g = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        m = ((g < 100) if dark else (g > 100)).astype(np.uint8)
        n, _lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        k = 0
        lo, hi = 0.012 * s * s, 0.06 * s * s
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if not lo <= a <= hi:
                continue
            if not 0.6 <= w / max(1, h) <= 1.6 or max(w, h) > 0.3 * s:
                continue
            if a / float(w * h) < 0.6:
                continue
            k += 1
        return k

    def _read(self, img, box, color) -> SeenTile:
        x, y, w, h = box
        t = img[y:y + h, x:x + w]
        s = min(w, h)
        if w >= h:
            a, b = t[:, : w // 2], t[:, w // 2:]
        else:
            a, b = t[: h // 2], t[h // 2:]
        dark = color != "black"
        p, q = self._count(a, s, dark), self._count(b, s, dark)
        tile = (min(p, q), max(p, q)) if p <= 6 and q <= 6 else None
        return SeenTile(tile, box, color, (p, q))

    # ------------------------------------------------------------ main
    def observe(self, frame: np.ndarray, t: float = 0.0) -> Observation:
        img = self.normalize(frame)
        ob = Observation(t)
        fx0, fy0, fx1, fy1 = self.felt_probe
        felt = img[fy0:fy1, fx0:fx1, 1]
        ob.dimmed = float(np.median(felt)) < 65
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ob.raise_offer = self._is_red(hsv, self.no_button) and self._is_green(hsv, self.yes_button)
        if ob.dimmed or ob.raise_offer:
            return ob
        x0, y0, x1, y1 = self.table
        sub = hsv[y0:y1, x0:x1]
        v, s = sub[..., 2], sub[..., 1]
        masks = (("white", (v > 215) & (s < 40)),
                 ("grey", (v > 120) & (v < 190) & (s < 25)),
                 ("black", v < 40))
        k3 = np.ones((3, 3), np.uint8)
        for color, m in masks:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k3)
            n, _lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n):
                bx, by, bw, bh, area = st[i]
                if area < 1200:
                    continue
                short, long_ = min(bw, bh), max(bw, bh)
                if not (30 <= short <= 75 and 1.6 <= long_ / short <= 2.4):
                    continue
                box = (int(bx + x0), int(by + y0), int(bw), int(bh))
                if box[1] + box[3] < self.opp_row_bottom:
                    if color == "white":
                        ob.opp_count += 1
                    continue
                if area < 0.75 * bw * bh:
                    continue
                st_ = self._read(img, box, color)
                if box[1] > self.hand_row_top:
                    if color != "black":
                        ob.hand.append(st_)
                else:
                    if color == "grey":
                        continue
                    ob.board.append(st_)
                if st_.tile is None:
                    ob.unreadable += 1
        ob.hand.sort(key=lambda h: h.box[0])
        return ob

    @staticmethod
    def _frac(hsv, box, cond) -> float:
        x0, y0, x1, y1 = box
        r = hsv[y0:y1, x0:x1]
        return float(cond(r).mean()) if r.size else 0.0

    def _is_red(self, hsv, box) -> bool:
        return self._frac(hsv, box, lambda r: ((r[..., 0] >= 165) | (r[..., 0] <= 5)) & (r[..., 1] > 150)
                          & (r[..., 2] > 90)) > 0.5

    def _is_green(self, hsv, box) -> bool:
        return self._frac(hsv, box, lambda r: (r[..., 0] >= 75) & (r[..., 0] <= 95) & (r[..., 1] > 100)
                          & (r[..., 1] < 175) & (r[..., 2] > 150)) > 0.3
