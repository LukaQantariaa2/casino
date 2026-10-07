"""Reads one screenshot of the game table into an ``Observation``.

All coordinates are in the layout's reference resolution (1920x1080); frames
of a different size are resized first.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..rules import Rules
from .templates import CardTemplates


@dataclass
class DetectedCard:
    card: int
    x: int
    y: int
    score: float
    margin: float          # best score minus best score of any other card

    def ok(self, smin: float, mmin: float) -> bool:
        return self.score >= smin and self.margin >= mmin


@dataclass
class Observation:
    t: float
    my_hand: list[DetectedCard] = field(default_factory=list)
    opp_count: int = 0
    center: list[DetectedCard] = field(default_factory=list)
    center_backs: int = 0
    trump: DetectedCard | None = None
    deck_present: bool = False
    my_turn: bool = False
    opp_turn: bool = False
    buttons: set = field(default_factory=set)      # claim, continue, raise, fold
    raise_modal: bool = False
    end_panel: bool = False
    end_panel_cards: list[DetectedCard] = field(default_factory=list)
    my_pile_present: bool = False
    opp_pile_present: bool = False
    problems: list[str] = field(default_factory=list)   # low-confidence detections
    pile_sig: dict = field(default_factory=dict)          # small grey thumbnails of both pile badges
    stake_badge: dict = field(default_factory=dict)       # who -> grey thumbnail of the "X2" badge (or None)

    # convenience
    def hand_ids(self) -> list[int]:
        return sorted(d.card for d in self.my_hand)

    def center_ids(self) -> list[int]:
        return sorted(d.card for d in self.center)

    def signature(self) -> tuple:
        """What matters for game-state tracking (used to detect real changes)."""
        return (tuple(self.hand_ids()), self.opp_count, tuple(self.center_ids()), self.center_backs,
                self.trump.card if self.trump else None, self.deck_present, tuple(sorted(self.buttons)),
                self.raise_modal, self.end_panel, self.my_turn, self.opp_turn)

    @property
    def table_empty(self) -> bool:
        return not self.my_hand and self.opp_count == 0 and not self.center and self.center_backs == 0


class Recognizer:
    def __init__(self, rules: Rules, layout: dict, templates: CardTemplates):
        self.rules = rules
        self.L = layout
        self.T = templates
        self.ref_w, self.ref_h = layout["reference_size"]
        self.R = {k: tuple(v) for k, v in layout["regions"].items()}
        self.C = {k: (np.array(v[0], np.uint8), np.array(v[1], np.uint8)) for k, v in layout["colors_hsv"].items()}
        self.th = layout["thresholds"]
        self.cw, self.ch = layout["card_size"]
        x0, y0, x1, y1 = layout["corner_box"]
        self.corner_off = (x0, y0)
        self.corner_wh = (x1 - x0, y1 - y0)
        self._cache: dict = {}

    # ------------------------------------------------------------------ helpers
    def normalize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != self.ref_w or frame.shape[0] != self.ref_h:
            frame = cv2.resize(frame, (self.ref_w, self.ref_h), interpolation=cv2.INTER_AREA)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def crop(self, img, name):
        x0, y0, x1, y1 = self.R[name]
        return img[y0:y1, x0:x1]

    def mask(self, hsv, color):
        lo, hi = self.C[color]
        return cv2.inRange(hsv, lo, hi)

    def blobs(self, m, min_area, max_w=10_000, max_h=10_000):
        n, _lab, st, _cen = cv2.connectedComponentsWithStats(m)
        return [tuple(int(v) for v in st[i][:5]) for i in range(1, n)
                if st[i][4] >= min_area and st[i][2] <= max_w and st[i][3] <= max_h]

    # ------------------------------------------------------------------ cache
    def _cached(self, img: np.ndarray, key: str, region: str, fn):
        """Re-run ``fn`` only when the region's pixels changed since the last call."""
        x0, y0, x1, y1 = self.R[region]
        thumb = cv2.resize(img[y0:y1, x0:x1], ((x1 - x0) // 8, (y1 - y0) // 8), interpolation=cv2.INTER_AREA)
        hit = self._cache.get(key)
        if hit is not None and hit[0].shape == thumb.shape:
            if float(cv2.absdiff(hit[0], thumb).mean()) < 1.0:
                return hit[1]
        val = fn()
        self._cache[key] = (thumb, val)
        return val

    # ------------------------------------------------------------------ cards
    def detect_cards(self, img: np.ndarray, region: str, thr1: float = 0.72, nms: int = 38) -> list[DetectedCard]:
        x0, y0, x1, y1 = self.R[region]
        reg = img[y0:y1, x0:x1]
        half = cv2.cvtColor(cv2.resize(reg, ((x1 - x0) // 2, (y1 - y0) // 2), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        cands = []
        kernel = np.ones((9, 9), np.uint8)
        for t in self.T.items:
            r = cv2.matchTemplate(half, t.corner_half_gray, cv2.TM_CCOEFF_NORMED)
            if float(r.max()) < thr1:
                continue
            # every local maximum above the threshold (a template may match several places)
            peaks = (r >= thr1) & (r >= cv2.dilate(r, kernel))
            ys, xs = np.nonzero(peaks)
            for yy, xx in zip(ys, xs):
                cands.append((float(r[yy, xx]), t.card, int(xx) * 2 + x0, int(yy) * 2 + y0))
        cands.sort(reverse=True)
        # group candidates by location
        groups: list[list] = []
        for c in cands:
            for g in groups:
                if abs(g[0][2] - c[2]) < nms and abs(g[0][3] - c[3]) < nms:
                    g.append(c)
                    break
            else:
                groups.append([c])
        out: list[DetectedCard] = []
        cw, ch = self.corner_wh
        H, W = img.shape[:2]
        for g in groups:
            gx, gy = g[0][2], g[0][3]
            px0, py0 = max(0, gx - 6), max(0, gy - 6)
            patch = img[py0:min(H, gy + ch + 6), px0:min(W, gx + cw + 6)]
            if patch.shape[0] < ch or patch.shape[1] < cw:
                continue
            scored = []
            tried = set()
            for _s, card, _x, _y in g:
                if card in tried or len(tried) >= 12:
                    continue
                tried.add(card)
                r = cv2.matchTemplate(patch, self.T.by_card[card].corner, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                scored.append((mx, card, px0 + loc[0], py0 + loc[1]))
            scored.sort(reverse=True)
            if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.06:
                scored = self._tiebreak(img, scored[:3])
            best = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if len(scored) == 1:
                # only one template passed stage 1; measure margin against a few look-alikes
                second = max(cv2.minMaxLoc(cv2.matchTemplate(patch, t.corner, cv2.TM_CCOEFF_NORMED))[1]
                             for t in self.T.items if t.card != best[1]
                             and self.rules.rank_of[t.card] == self.rules.rank_of[best[1]])
            if best[0] < self.th.get("card_detect_min", 0.80):
                continue
            out.append(DetectedCard(best[1], best[2] - self.corner_off[0], best[3] - self.corner_off[1],
                                    float(best[0]), float(best[0] - second)))
        # a card can only exist once: keep the best detection of each id
        uniq: dict[int, DetectedCard] = {}
        for d in out:
            if d.card not in uniq or d.score > uniq[d.card].score:
                uniq[d.card] = d
        return sorted(uniq.values(), key=lambda d: (d.y // 60, d.x))

    def _tiebreak(self, img: np.ndarray, top: list) -> list:
        """Close call between look-alike corners: compare the upper half of the whole card too."""
        H, W = img.shape[:2]
        rescored = []
        for sc, card, cx, cy in top:
            x, y = cx - self.corner_off[0], cy - self.corner_off[1]
            px0, py0 = max(0, x - 4), max(0, y - 4)
            patch = img[py0:min(H, y + self.ch // 2 + 4), px0:min(W, x + self.cw + 4)]
            tmpl = self.T.by_card[card].full[: self.ch // 2]
            if patch.shape[0] < tmpl.shape[0] or patch.shape[1] < tmpl.shape[1]:
                rescored.append((sc, card, cx, cy))
                continue
            full = cv2.minMaxLoc(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED))[1]
            rescored.append((0.5 * sc + 0.5 * full, card, cx, cy))
        rescored.sort(reverse=True)
        return rescored

    def detect_trump(self, img: np.ndarray) -> DetectedCard | None:
        reg = self.crop(img, "trump")
        best = []
        for t in self.T.items:
            r = cv2.matchTemplate(reg, t.trump_corner, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            best.append((mx, t.card, loc))
        best.sort(reverse=True)
        if not best or best[0][0] < self.th["trump_score_min"]:
            return None
        x0, y0 = self.R["trump"][:2]
        return DetectedCard(best[0][1], x0 + best[0][2][0], y0 + best[0][2][1], float(best[0][0]),
                            float(best[0][0] - best[1][0]))

    def count_backs(self, hsv: np.ndarray, region: str) -> list[tuple[int, int]]:
        x0, y0, x1, y1 = self.R[region]
        sub = hsv[y0:y1, x0:x1]
        red = self.mask(sub, "back_logo_red")
        dark = self.mask(sub, "back_dark")
        out = []
        for bx, by, bw, bh, area in self.blobs(red, 1200, 80, 80):
            if not (38 <= bw <= 72 and 38 <= bh <= 72):
                continue
            # a card back logo is surrounded by the dark back
            ring = dark[max(0, by - 25):by + bh + 25, max(0, bx - 25):bx + bw + 25]
            if ring.size and ring.mean() / 255 > 0.35:
                out.append((x0 + bx, y0 + by))
        return out

    # ------------------------------------------------------------------ ui elements
    def detect_buttons(self, hsv: np.ndarray) -> set:
        x0, y0, x1, y1 = self.R["buttons"]
        sub = hsv[y0:y1, x0:x1]
        a = self.th["button_min_area"]
        found = set()

        def pill(color):
            for bx, by, bw, bh, area in self.blobs(self.mask(sub, color), a, 260, 80):
                if 100 <= bw <= 230 and 30 <= bh <= 70 and area / float(bw * bh) > 0.72:
                    return True
            return False

        if pill("btn_orange"):
            found.add("claim")
        if pill("btn_green"):
            found.add("continue")
        if pill("btn_blue"):
            found.add("raise")
        black = cv2.inRange(sub, np.array((0, 0, 0), np.uint8), np.array((180, 255, 14), np.uint8))
        for bx, by, bw, bh, area in self.blobs(black, 3000, 260, 80):
            if 110 <= bw <= 230 and 30 <= bh <= 70:
                found.add("fold")
        return found

    def detect_modal(self, hsv: np.ndarray) -> bool:
        x0, y0, x1, y1 = self.R["modal"]
        sub = hsv[y0:y1, x0:x1]
        a = self.th["modal_min_area"]
        g = self.blobs(self.mask(sub, "btn_green"), a, 260, 90)
        r = self.blobs(self.mask(sub, "btn_red"), a, 260, 90)
        return bool(g and r)

    def ring_active(self, hsv: np.ndarray, region: str) -> bool:
        sub = self.crop(hsv, region)
        h, w = sub.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        ann = (rr > min(h, w) * 0.40) & (rr < min(h, w) * 0.50) & ~((yy > h * 0.55) & (xx > w * 0.55))
        g = self.mask(sub, "ring_green") > 0
        o = self.mask(sub, "ring_orange") > 0
        frac = ((g | o) & ann).sum() / max(1, ann.sum())
        return frac >= self.th["ring_min_fraction"]

    def panel_present(self, hsv: np.ndarray) -> bool:
        sub = self.crop(hsv, "end_panel")
        return self.mask(sub, "panel_green").mean() / 255 >= self.th["end_panel_min_fraction"]

    def pile_present(self, hsv: np.ndarray, region: str) -> bool:
        sub = self.crop(hsv, region)
        return bool(self.blobs(self.mask(sub, "btn_green"), 600, 70, 70))

    def read_badge(self, hsv: np.ndarray, center, radius: int):
        """Green "X2" stake badge: returns its white-text mask (bool array) or None if absent."""
        cx, cy = center
        sub = hsv[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disc = (xx * xx + yy * yy) <= radius * radius
        white = (sub[:, :, 2] > 190) & (sub[:, :, 1] < 70) & disc
        green = (self.mask(sub, "btn_green") > 0) & disc & ~white
        if green.sum() < 0.45 * (disc.sum() - white.sum()) or white.sum() < 10:
            return None
        return white

    def deck_present(self, hsv: np.ndarray) -> bool:
        sub = self.crop(hsv, "deck_back")
        return self.mask(sub, "back_dark").mean() / 255 > 0.45

    # ------------------------------------------------------------------ main
    def observe(self, frame: np.ndarray, t: float | None = None) -> Observation:
        img = self.normalize(frame)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ob = Observation(t=time.time() if t is None else t)
        smin, mmin = self.th["card_score_min"], self.th["card_margin_min"]
        ob.end_panel = self.panel_present(hsv)
        ob.raise_modal = self.detect_modal(hsv)
        if ob.end_panel:
            ob.end_panel_cards = self._cached(img, "end", "end_panel",
                                              lambda: self.detect_cards(img, "end_panel", nms=30))
        ob.my_hand = self._cached(img, "hand", "my_hand", lambda: self.detect_cards(img, "my_hand"))
        ob.opp_count = len(self.count_backs(hsv, "opp_hand"))
        ob.center = [] if ob.end_panel else self._cached(img, "center", "center",
                                                         lambda: self.detect_cards(img, "center"))
        ob.center_backs = 0 if ob.end_panel else len(self.count_backs(hsv, "center"))
        ob.trump = self._cached(img, "trump", "trump", lambda: self.detect_trump(img))
        ob.deck_present = self.deck_present(hsv)
        ob.my_turn = self.ring_active(hsv, "my_avatar")
        ob.opp_turn = self.ring_active(hsv, "opp_avatar")
        ob.buttons = self.detect_buttons(hsv)
        for key, region in (("me", "my_pile"), ("opp", "opp_pile")):
            g = cv2.cvtColor(self.crop(img, region), cv2.COLOR_BGR2GRAY)
            ob.pile_sig[key] = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4),
                                          interpolation=cv2.INTER_AREA).astype(np.int16)
        for key, spec in self.L.get("stake_badges", {}).items():
            ob.stake_badge[key] = self.read_badge(hsv, spec["center"], spec["radius"])
        ob.my_pile_present = self.pile_present(hsv, "my_pile")
        ob.opp_pile_present = self.pile_present(hsv, "opp_pile")
        for d in ob.my_hand + ob.center:
            if not d.ok(smin, mmin):
                ob.problems.append(f"low confidence {self.rules.card_name(d.card)} "
                                   f"(score {d.score:.2f}, margin {d.margin:.2f})")
        if len(ob.my_hand) > self.rules.hand_size:
            ob.problems.append(f"{len(ob.my_hand)} cards detected in my hand")
        if ob.opp_count > self.rules.hand_size:
            ob.problems.append(f"{ob.opp_count} opponent cards detected")
        return ob
