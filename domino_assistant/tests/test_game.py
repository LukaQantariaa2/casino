import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino.engine.game import Board, Rules, round_up5  # noqa: E402
from domino.engine.match import MatchTable  # noqa: E402
from domino.engine.search import Engine, Info, sample_deal  # noqa: E402

L, R, U, D = 0, 1, 2, 3


def test_spinner_first_sequence_from_video1():
    b = Board().place((4, 4), -1)
    assert b.count() == 8 and b.points() == 0
    b = b.place((4, 5), R)
    assert b.count() == 13
    b = b.place((5, 6), R)
    assert b.count() == 14
    b = b.place((3, 4), L)
    assert b.count() == 9
    assert b.arms[U] == (4, False, False)
    b = b.place((4, 6), U)
    assert b.points() == 15


def test_spinner_later_in_the_line_from_video2():
    b = Board().place((0, 2), -1)
    assert b.count() == 2
    b = b.place((2, 2), R)
    assert b.spinner == 2 and b.count() == 4
    b = b.place((0, 6), L)
    assert b.points() == 10
    b = b.place((1, 6), L)
    assert b.points() == 5
    assert b.arms[U] is None                    # spinner's outer side still open
    b = b.place((2, 3), R)
    assert b.arms[U] == (2, False, False)


def test_double_at_an_end_counts_twice_and_first_double_scores():
    b = Board().place((5, 5), -1)
    assert b.points() == 10
    b = Board().place((1, 3), -1).place((3, 3), R)      # 3|3 becomes the spinner here
    assert b.count() == 1 + 6
    b = Board().place((4, 4), -1).place((4, 1), L).place((4, 2), R).place((1, 1), L)
    assert b.count() == 2 + 2


def test_moves_dedupe_identical_arms():
    b = Board().place((3, 3), -1)
    assert len(b.moves([(3, 5)])) == 1
    b = Board().place((3, 5), -1)
    assert len(b.moves([(3, 5)])) == 2


def test_round_up():
    assert [round_up5(x) for x in (0, 2, 4, 5, 16, 17, 28)] == [0, 5, 5, 5, 20, 20, 30]


def test_sampler_respects_forbidden_values():
    unknown = [(a, b) for a in range(7) for b in range(a, 7) if (a, b) not in [(0, 0), (1, 1)]]
    info = Info(Board().place((2, 2), -1), [(0, 0), (1, 1)], 3, unknown, [frozenset({2, 5})] * 3)
    rng = random.Random(1)
    for _ in range(50):
        hand, bone = sample_deal(info, rng)
        assert len(hand) == 3 and len(bone) == len(unknown) - 3
        assert all(2 not in t and 5 not in t for t in hand)


def test_match_table_is_sane():
    m = MatchTable(75)
    assert 0.0 < m.after_round(0, 0, 0) < 1.0
    assert m.after_round(70, 0, 0) > 0.9
    assert m.after_round(0, 70, 1) < 0.1
    assert m.after_round(80, 40, 0) == 1.0


def test_engine_takes_the_points():
    # ends 4 (left) and 6 (right): 1|4 on the 4 makes 1+6 = 7, 4|5... use a clear case: 5|6 on 6 -> 4+5 = 9,
    # 1|6 on the 6 -> 4+1 = 5 scores 5 and nothing else scores
    b = Board().place((4, 6), -1)
    hand = [(1, 6), (2, 3)]
    seen = set(hand) | {(4, 6)}
    unknown = [(a, c) for a in range(7) for c in range(a, 7) if (a, c) not in seen]
    eng = Engine(Rules(), 0.5, seed=3)
    info = Info(b, hand, 6, unknown, [], 0, 0, 255)
    res = eng.evaluate(info, b.moves(hand))
    assert res[0].move[0] == (1, 6)


def run_all(ns):
    failed = 0
    for name, fn in sorted(ns.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:
                failed += 1
                print("FAIL", name, type(exc).__name__, exc)
    print(f"{failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(run_all(dict(globals())))
