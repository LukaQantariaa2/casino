"""Rebuilds the line of play from tile positions on screen.

Two board tiles are joined when their boxes touch and the halves facing each other show the same
number. From the spinner (the black tile) or, without one, from the two loose ends, each arm is
walked out to its tip; the tip's outer half is that arm's open value.
"""
from __future__ import annotations

from ..engine.game import Board
from ..vision.reader import SeenTile

_L, _R, _U, _D = 0, 1, 2, 3


def touching(a: SeenTile, b: SeenTile) -> bool:
    ax, ay, aw, ah = a.box
    bx, by, bw, bh = b.box
    s = min(a.short, b.short)
    g = 0.25 * s
    gap_x = max(ax, bx) - min(ax + aw, bx + bw)
    gap_y = max(ay, by) - min(ay + ah, by + bh)
    ov_x = min(ax + aw, bx + bw) - max(ax, bx)
    ov_y = min(ay + ah, by + bh) - max(ay, by)
    return (gap_x <= g and ov_y > 0.3 * s) or (gap_y <= g and ov_x > 0.3 * s)


def facing_half(t: SeenTile, other: SeenTile) -> int:
    cx, cy = t.center
    ox, oy = other.center
    if t.box[2] >= t.box[3]:
        return 0 if ox < cx else 1
    return 0 if oy < cy else 1


def joined(a: SeenTile, b: SeenTile) -> bool:
    return touching(a, b) and a.halves[facing_half(a, b)] == b.halves[facing_half(b, a)]


def is_double(t: SeenTile) -> bool:
    return t.tile is not None and t.tile[0] == t.tile[1]


def _walk(tiles, adj, start: int, prev: int):
    """Follow the chain from start (whose predecessor is prev) to its tip."""
    cur, p = start, prev
    seen = {prev}
    while True:
        seen.add(cur)
        nxt = [j for j in adj[cur] if j != p and j not in seen]
        if not nxt:
            return cur, p
        if len(nxt) > 1:
            return None
        cur, p = nxt[0], cur


def _arm(tiles, tip: int, prev: int):
    t = tiles[tip]
    if is_double(t):
        return (t.tile[0], True, True)
    f = facing_half(t, tiles[prev])
    return (t.halves[1 - f], False, True)


def side_of(center, ref) -> str:
    dx, dy = center[0] - ref[0], center[1] - ref[1]
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "top" if dy < 0 else "bottom"


def rebuild(seen: list[SeenTile]):
    """Returns (Board, tips) where tips[arm] is the tile at that arm's tip, or None if unclear."""
    tiles = [t for t in seen if t.tile is not None]
    if len(tiles) != len(seen):
        return None
    if not tiles:
        return Board(), [None] * 4
    if len(set(t.tile for t in tiles)) != len(tiles):
        return None
    if len(tiles) == 1:
        t = tiles[0]
        return Board().place(t.tile, -1), [t.tile, t.tile, None, None]
    n = len(tiles)
    adj = [[] for _ in range(n)]
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if joined(tiles[i], tiles[j]):
                adj[i].append(j)
                adj[j].append(i)
                edges += 1
    if edges != n - 1:
        return None
    blacks = [i for i, t in enumerate(tiles) if t.color == "black"]
    sp = blacks[0] if len(blacks) == 1 else None
    if sp is None:
        loose_doubles = [i for i, t in enumerate(tiles) if is_double(t) and len(adj[i]) == 1]
        doubles = [i for i, t in enumerate(tiles) if is_double(t)]
        if len(doubles) == 1 and loose_doubles:
            sp = loose_doubles[0]                 # first double, still white until both sides are covered
    for i in range(n):
        if i != sp and len(adj[i]) > 2:
            return None
    arms = [None, None, None, None]
    tips = [None, None, None, None]
    if sp is None:
        leaves = [i for i in range(n) if len(adj[i]) == 1]
        if len(leaves) != 2:
            return None
        leaves.sort(key=lambda i: tiles[i].center[0])
        for k, leaf in enumerate(leaves):
            arms[k] = _arm(tiles, leaf, adj[leaf][0])
            tips[k] = tiles[leaf].tile
        return Board(n, -1, tuple(arms)), tips
    S = tiles[sp]
    d = S.tile[0]
    sx, sy = S.center
    vertical = S.box[3] > S.box[2]
    side, cross = {}, {}
    for j in adj[sp]:
        ox, oy = tiles[j].center
        dx, dy = ox - sx, oy - sy
        along_sides = abs(dx) > abs(dy) if vertical else abs(dy) > abs(dx)
        key = (dx if vertical else dy) if along_sides else (dy if vertical else dx)
        (side if along_sides else cross)[0 if key < 0 else 1] = j
    if len(side) + len(cross) != len(adj[sp]):
        return None
    for k in (0, 1):
        j = side.get(k)
        if j is None:
            arms[k] = (d, False, False)
            tips[k] = S.tile
        else:
            w = _walk(tiles, adj, j, sp)
            if w is None:
                return None
            arms[k] = _arm(tiles, w[0], w[1])
            tips[k] = tiles[w[0]].tile
    if arms[_L][2] and arms[_R][2]:
        for k, a in ((0, _U), (1, _D)):
            j = cross.get(k)
            if j is None:
                arms[a] = (d, False, False)
                tips[a] = S.tile
            else:
                w = _walk(tiles, adj, j, sp)
                if w is None:
                    return None
                arms[a] = _arm(tiles, w[0], w[1])
                tips[a] = tiles[w[0]].tile
    elif cross:
        return None
    return Board(n, d, tuple(arms)), tips
