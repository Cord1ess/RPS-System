import numpy as np

from replay_eval import aggregate, landing_time, throw_time
from rps.hand_tracker import PAPER, ROCK, SCISSORS

FPS = 30.0


def rows_for(wrist, gestures=None, vy=None, pumps=None):
    n = len(wrist)
    return [{"t": i / FPS, "ref_wrist": wrist[i], "ref_gesture": gestures[i] if gestures else None,
             "vy": vy[i] if vy is not None else None, "pumps": pumps[i] if pumps else 2} for i in range(n)]


def throw_round():
    """Last pump counted at frame 5; the throw stroke lands at frame 20 (the wrist's first low point
    after it), although an earlier pump bottom (frame 2) was lower."""
    t = np.arange(40)
    wrist = list(0.6 + 0.1 * np.cos(2 * np.pi * (t - 20) / 24))       # low point (max y) at frame 20
    wrist[2] = 0.9                                                    # deeper pump bottom before the throw
    pumps = [1] * 5 + [2] * 35
    return wrist, pumps


def test_landing_is_the_first_low_point_after_the_last_pump():
    wrist, pumps = throw_round()
    rows = rows_for(wrist, pumps=pumps)
    assert abs(landing_time(rows, 18) - 20 / FPS) < 1e-9


def test_landing_without_the_hand_uses_the_end_of_the_down_stroke():
    vy = [0.0] * 5 + [1.0] * 8 + [-0.5] * 10                          # moving down until frame 12
    pumps = [1] * 3 + [2] * 20
    rows = rows_for([None] * len(vy), vy=vy, pumps=pumps)
    assert abs(landing_time(rows, 15) - 13 / FPS) < 1e-9


def test_throw_time_is_when_the_shape_first_shows_for_paper_and_scissors():
    wrist, pumps = throw_round()
    gestures = [ROCK] * 16 + [SCISSORS] * 24                          # fingers open at frame 16
    rows = rows_for(wrist, gestures, pumps=pumps)
    assert abs(throw_time(rows, 19, SCISSORS) - 16 / FPS) < 1e-9
    assert abs(throw_time(rows, 19, ROCK) - 20 / FPS) < 1e-9          # rock: the landing
    assert abs(throw_time(rows, 19, PAPER) - 20 / FPS) < 1e-9         # shape never seen: the landing


def test_aggregate_throw_accuracy_counts_decisions():
    scores = [{"type": "throws", "commits": 10, "throw_acc": 0.9, "throws_planned": 10},
              {"type": "throws", "commits": 0, "throw_acc": 0.0, "throws_planned": 15}]
    agg = aggregate(scores)
    assert agg["throw_commits"] == 10 and agg["throws_planned"] == 25 and abs(agg["throw_acc"] - 0.9) < 1e-9


def test_timing_pools_every_decision():
    # one good session with 28 decisions near +64 ms, one bad session with a single stray decision
    scores = [{"type": "throws", "commits": 28, "throw_acc": 1.0, "latencies_ms": [64.0] * 28,
               "visible_all_ms": [264.0] * 28},
              {"type": "throws", "commits": 1, "throw_acc": 0.0, "latencies_ms": [1232.0],
               "visible_all_ms": [1432.0]}]
    agg = aggregate(scores)
    assert agg["commit_vs_throw_ms"] == 64.0 and agg["visible_ms"] == 264.0
