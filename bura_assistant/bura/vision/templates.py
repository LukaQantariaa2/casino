"""Card template loading (the exported .webp/.png card images)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from ..rules import Rules


def load_image_bgr_on_white(path: str) -> np.ndarray:
    """Load an image (webp/png, with or without alpha) composited on white, as BGR uint8."""
    arr = None
    try:
        from PIL import Image  # Pillow decodes webp alpha reliably
        im = Image.open(path).convert("RGBA")
        rgba = np.asarray(im).astype(np.float32)
        a = rgba[:, :, 3:4] / 255.0
        rgb = rgba[:, :, :3] * a + 255.0 * (1.0 - a)
        arr = np.ascontiguousarray(rgb[:, :, ::-1].astype(np.uint8))
    except Exception:
        pass
    if arr is None:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[2] == 4:
            a = im[:, :, 3:4].astype(np.float32) / 255.0
            arr = (im[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
        else:
            arr = im[:, :, :3].copy()
    return arr


@dataclass
class CardTemplate:
    card: int
    full: np.ndarray        # card_size BGR
    corner: np.ndarray      # corner crop
    corner_half: np.ndarray
    corner_half_gray: np.ndarray
    trump_full: np.ndarray  # rotated (lying) card, left visible part
    trump_corner: np.ndarray


class CardTemplates:
    def __init__(self, rules: Rules, layout: dict, cards_dir: str):
        self.rules = rules
        self.layout = layout
        cw, ch = layout["card_size"]
        x0, y0, x1, y1 = layout["corner_box"]
        tf = layout["template_files"]
        self.items: list[CardTemplate] = []
        missing = []
        for c in range(rules.n_cards):
            suit = rules.suits[rules.suit_of[c]]
            rank = rules.ranks[rules.rank_of[c]]
            name = tf["pattern"].format(suit=tf["suit_codes"][suit], rank=tf["rank_codes"][rank])
            path = os.path.join(cards_dir, name)
            if not os.path.exists(path):
                alt = os.path.splitext(path)[0] + ".png"
                if os.path.exists(alt):
                    path = alt
                else:
                    missing.append(name)
                    continue
            img = cv2.resize(load_image_bgr_on_white(path), (cw, ch), interpolation=cv2.INTER_AREA)
            corner = np.ascontiguousarray(img[y0:y1, x0:x1])
            half = cv2.resize(corner, ((x1 - x0) // 2, (y1 - y0) // 2), interpolation=cv2.INTER_AREA)
            rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)   # lying card: corner at bottom-left
            vis = int(layout.get("trump_visible_width", 110))
            trump_full = np.ascontiguousarray(rot[:, :vis])
            # the corner of a lying card: rotate the corner crop the same way
            trump_corner = np.ascontiguousarray(cv2.rotate(corner, cv2.ROTATE_90_COUNTERCLOCKWISE))
            half_gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
            self.items.append(CardTemplate(c, img, corner, half, half_gray, trump_full, trump_corner))
        if missing:
            raise FileNotFoundError(f"missing card images in {cards_dir}: {', '.join(missing[:6])}"
                                    f"{'...' if len(missing) > 6 else ''}")
        self.by_card = {t.card: t for t in self.items}
