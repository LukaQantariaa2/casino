"""All Fives (Muggins) rules as played on Adjarabet, 1 vs 1.

Tiles are (low, high) tuples of the double-six set. The line of play is kept as up to four arms:
L and R are the two ends of the line; U and D open on the spinner (the first double played)
once both of its sides are covered. An arm is (open value, tip is a double, has tiles beyond the
spinner). After every play the open ends are added up — a double at an end counts both halves, the
spinner counts both halves while one of its sides is still uncovered — and a multiple of five is
scored. A player who goes out scores the opponent's remaining pips rounded up to the next five.
"""
from __future__ import annotations

from dataclasses import dataclass

ME, OPP = 0, 1
TILES: tuple[tuple[int, int], ...] = tuple((a, b) for a in range(7) for b in range(a, 7))
ARMS = ("L", "R", "U", "D")
_L, _R, _U, _D = 0, 1, 2, 3


def tile(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def tile_name(t) -> str:
    return f"{t[0]}|{t[1]}"


def pips(tiles) -> int:
    return sum(a + b for a, b in tiles)


def round_up5(n: int) -> int:
    return -(-n // 5) * 5


@dataclass(frozen=True)
class Board:
    """Immutable line of play. ``arms[i]`` is None (not open) or (open, tip_double, has_tiles)."""
    n: int = 0
    spinner: int = -1
    arms: tuple = (None, None, None, None)

    # ------------------------------------------------------------ queries
    def open_values(self) -> set[int]:
        return {a[0] for a in self.arms if a is not None}

    def count(self) -> int:
        """Sum of the open ends after the last play."""
        if self.n == 0:
            return 0
        s = 0
        for a in self.arms:
            if a is not None and a[2]:
                s += a[0] * 2 if a[1] else a[0]
        if self.spinner >= 0 and not (self.arms[_L][2] and self.arms[_R][2]):
            s += 2 * self.spinner
        return s

    def points(self) -> int:
        s = self.count()
        return s if s and s % 5 == 0 else 0

    def moves(self, hand) -> list[tuple[tuple[int, int], int]]:
        """Legal (tile, arm) plays; arms with identical state are interchangeable, so only one is listed."""
        if self.n == 0:
            return [(t, -1) for t in hand]
        out = []
        for t in hand:
            seen = set()
            for i, a in enumerate(self.arms):
                if a is None or (a[0] != t[0] and a[0] != t[1]) or a in seen:
                    continue
                seen.add(a)
                out.append((t, i))
        return out

    def playable(self, hand) -> bool:
        if self.n == 0:
            return bool(hand)
        ov = self.open_values()
        return any(a in ov or b in ov for a, b in hand)

    # ------------------------------------------------------------ play
    def place(self, t: tuple[int, int], arm: int) -> "Board":
        x, y = t
        arms = list(self.arms)
        spinner = self.spinner
        if self.n == 0:
            if x == y:
                spinner = x
                arms = [(x, False, False), (x, False, False), None, None]
            else:
                arms = [(x, False, True), (y, False, True), None, None]
            return Board(1, spinner, tuple(arms))
        a = arms[arm]
        if a is None or (a[0] != x and a[0] != y):
            raise ValueError(f"{tile_name(t)} does not fit arm {ARMS[arm]}")
        if x == y and spinner < 0:
            spinner = x
            arms[arm] = (x, False, False)
        else:
            other = y if x == a[0] else x
            arms[arm] = (other, x == y, True)
        if spinner >= 0 and arms[_U] is None and arms[_L][2] and arms[_R][2]:
            arms[_U] = (spinner, False, False)
            arms[_D] = (spinner, False, False)
        return Board(self.n + 1, spinner, tuple(arms))

    def describe(self) -> str:
        if self.n == 0:
            return "empty"
        parts = []
        for name, a in zip(ARMS, self.arms):
            if a is None:
                continue
            if a[2]:
                parts.append(f"{name}:{a[0]}{'d' if a[1] else ''}")
            else:
                parts.append(f"{name}:({a[0]})")
        sp = f" spinner {self.spinner}|{self.spinner}" if self.spinner >= 0 else ""
        return " ".join(parts) + f"{sp} = {self.count()}"


@dataclass
class Rules:
    target: int = 255
    hand_size: int = 7
    boneyard_reserve: int = 2       # tiles that stay in the boneyard and can never be drawn
    multiple: int = 5

    def can_draw(self, boneyard_left: int) -> bool:
        return boneyard_left > self.boneyard_reserve
