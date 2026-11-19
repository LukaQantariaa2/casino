"""Builds the score digit templates (domino/vision/digits.npz) from screen recordings.

    python tools/build_digits.py rec1.mp4 [rec2.mp4 ...]

Windows OCR reads most score boxes but not "00"; wherever it does read a number whose length matches
the glyphs found, those glyphs become examples of their digits. The averaged glyphs are saved.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino.vision.ocr import DIGITS_FILE, WindowsOcr, glyphs  # noqa: E402

BOXES = [(70, 372, 160, 458), (70, 640, 160, 725)]


def main():
    ocr = WindowsOcr()
    samples: dict[int, list] = {d: [] for d in range(10)}
    for path in sys.argv[1:]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(0, n, int(fps * 2)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            if fr.shape[1] != 1920:
                fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
            for x0, y0, x1, y1 in BOXES:
                crop = fr[y0:y1, x0:x1]
                gl = glyphs(crop)
                txt = "".join(ch for ch in ocr.read(crop) if ch.isdigit())
                if txt and len(txt) == len(gl):
                    for ch, g in zip(txt, gl):
                        samples[int(ch)].append(g)
    missing = [d for d, s in samples.items() if not s]
    print({d: len(s) for d, s in samples.items()})
    if missing:
        print("no examples for digits", missing)
    tmpl = np.stack([np.mean(samples[d], axis=0) if samples[d] else np.zeros((32, 20)) for d in range(10)])
    np.savez_compressed(DIGITS_FILE, templates=tmpl.astype(np.float32), have=np.array([bool(samples[d])
                                                                                       for d in range(10)]))
    print("saved", DIGITS_FILE)


if __name__ == "__main__":
    main()
