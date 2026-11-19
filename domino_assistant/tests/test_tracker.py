"""Tracker tests on synthetic observations (a short horizontal line with the spinner in the middle)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino.engine.game import ME, OPP, Rules  # noqa: E402
from domino.tracker.tracker import STATUS_OK, History, Tracker  # noqa: E402
from domino.vision.reader import Observation, SeenTile  # noqa: E402

S = 40                       # short side of a tile in pixels
Y = 500                      # top of the horizontal line


def spinner(a, black=True):
    return SeenTile((a, a), (900, Y - S // 2, S, 2 * S), "black" if black else "white", (a, a))


def right(k, a, b):
    """k-th tile to the right of the spinner, showing a then b."""
    return SeenTile(tuple(sorted((a, b))), (940 + 2 * S * k, Y, 2 * S, S), "white", (a, b))


def left(k, a, b):
    """k-th tile to the left of the spinner, outer value a, inner value b."""
    return SeenTile(tuple(sorted((a, b))), (900 - 2 * S * (k + 1), Y, 2 * S, S), "white", (a, b))


def hand(tiles, playable=()):
    return [SeenTile(t, (700 + i * 50, 800, S, 2 * S), "white" if t in playable else "grey", t)
            for i, t in enumerate(tiles)]


def obs(board, my, opp, playable=(), t=[0.0]):
    t[0] += 0.5
    return Observation(t[0], board=board, hand=hand(my, playable), opp_count=opp)


def feed(tr, o, times=2):
    for _ in range(times):
        o.t += 0.5
        tr.update(o)


def fresh_round(my):
    tr = Tracker(Rules(target=175), History())
    feed(tr, obs([], my, 7))
    assert tr.round is not None and tr.round.my_hand == my
    return tr


START = [(3, 3), (3, 4), (0, 1), (1, 2), (2, 6), (5, 6), (0, 0)]


def test_new_round_my_spinner_then_his_reply():
    tr = fresh_round(START)
    rest = [t for t in START if t != (3, 3)]
    feed(tr, obs([spinner(3)], rest, 7))
    assert tr.round.played[-1][:2] == (ME, (3, 3))
    feed(tr, obs([spinner(3), right(0, 3, 5)], rest, 6, playable=[(3, 4), (5, 6)]))
    assert tr.round.played[-1][:2] == (OPP, (3, 5))
    assert tr.round.board.count() == 6 + 5
    assert tr.status.status == STATUS_OK and tr.status.phase == "my_turn"


def test_tile_hidden_while_dragged_still_counts_as_mine():
    tr = fresh_round(START)
    rest = [t for t in START if t != (3, 3)]
    feed(tr, obs([], rest, 7, playable=START))              # 3|3 is under the cursor, in flight
    feed(tr, obs([spinner(3)], rest, 7))
    assert tr.round.played[-1][:2] == (ME, (3, 3))


def test_turns_alternate_while_the_boneyard_is_open():
    tr = fresh_round(START)
    h = [t for t in START if t != (3, 3)]
    feed(tr, obs([spinner(3)], h, 7))
    b = [spinner(3), right(0, 3, 5)]
    feed(tr, obs(b, h, 6))
    h2 = [t for t in h if t != (3, 4)]
    b2 = b + [left(0, 4, 3)]
    # my 3|4 and his next tile land between two readings: still mine first, then his
    b3 = b2 + [right(1, 5, 5)]
    feed(tr, obs(b3, h2, 5))
    who = [(w, t) for w, t, *_ in tr.round.played]
    assert who[-2:] == [(ME, (3, 4)), (OPP, (5, 5))], who


def test_opponent_draw_marks_missing_values():
    tr = fresh_round(START)
    h = [t for t in START if t != (3, 3)]
    feed(tr, obs([spinner(3)], h, 7))
    feed(tr, obs([spinner(3)], h, 9))                       # he drew two: no 3 except maybe the last one
    assert sum(3 in s for s in tr.round.opp.slots) == 8
    feed(tr, obs([spinner(3), right(0, 3, 6)], h, 8))
    assert tr.round.played[-1][:2] == (OPP, (3, 6))


def test_impossible_count_jump_is_not_a_draw():
    tr = fresh_round(START)
    h = [t for t in START if t != (3, 3)]
    feed(tr, obs([spinner(3)], h, 7))
    feed(tr, obs([spinner(3)], h, 30))                      # misread: more than the boneyard holds
    assert not tr.round.opp.missing_values()


def test_board_tile_never_reaches_the_engine_hand():
    tr = fresh_round(START)
    h = [t for t in START if t != (3, 3)]
    feed(tr, obs([spinner(3)], h, 7))
    tr.round.my_hand.append((3, 3))                         # stale misread
    assert (3, 3) not in tr.info().my_hand


def run_all():
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:
                failed += 1
                print("FAIL", name, type(exc).__name__, exc)
    print(failed, "failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
