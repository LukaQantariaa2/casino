"""What the opponent's draws and passes tell us about his hand.

He only draws (or passes when drawing is no longer allowed) when nothing in his hand fits an open
end. So after a draw, every tile he held at that moment contains none of the open values. Each of
his tiles carries the set of values it cannot contain; tiles held longer carry more. When he plays a
tile we drop the most restricted tile that could have been the one played, which keeps every
remaining statement true whichever tile it really was.
"""
from __future__ import annotations


class OppKnowledge:
    def __init__(self, n: int = 0):
        self.slots: list[frozenset] = [frozenset()] * n

    def __len__(self):
        return len(self.slots)

    def drew(self, open_values):
        ov = frozenset(open_values)
        self.slots = [s | ov for s in self.slots] + [frozenset()]

    def passed(self, open_values):
        ov = frozenset(open_values)
        self.slots = [s | ov for s in self.slots]

    def played(self, t) -> bool:
        """Remove the slot of tile t. False if no slot allowed it (our picture was wrong)."""
        if not self.slots:
            return False
        ok = [i for i, s in enumerate(self.slots) if t[0] not in s and t[1] not in s]
        if not ok:
            self.slots = [frozenset()] * (len(self.slots) - 1)
            return False
        i = max(ok, key=lambda j: len(self.slots[j]))
        self.slots.pop(i)
        return True

    def resize(self, n: int):
        """Resynchronise with the tile count seen on screen."""
        if n > len(self.slots):
            self.slots += [frozenset()] * (n - len(self.slots))
        while len(self.slots) > n:
            i = min(range(len(self.slots)), key=lambda j: len(self.slots[j]))
            self.slots.pop(i)

    def forbidden(self) -> list[frozenset]:
        return list(self.slots)

    def missing_values(self) -> set[int]:
        """Values none of his tiles can contain."""
        if not self.slots:
            return set()
        common = set(self.slots[0])
        for s in self.slots[1:]:
            common &= s
        return common
