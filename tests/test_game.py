"""Beat guide timing (rps.game) and the guided game mode of the decision engine."""

import pytest

from rps.config import DecisionConfig, VoteConfig
from rps.decision import ARMED, HOLD, READY, DecisionEngine, MotionObs
from rps.game import BeatSchedule
from rps.hand_tracker import BACKGROUND, PAPER, ROCK, SCISSORS, HandObs

FPS = 30.0


def test_beat_pattern_and_rounds():
    s = BeatSchedule(t0=10.0, beat_s=0.4, pumps=3, rounds=2, lead_beats=4, gap_beats=4)
    kinds = [s.beat(k).kind for k in range(s.total_beats)]
    labels = [s.beat(k).label for k in range(s.total_beats)]
    assert kinds[:4] == ["soft"] * 4                                  # get ready
    assert labels[4:8] == ["3", "2", "1", "SHOOT"] and kinds[4:8] == ["pump", "pump", "pump", "shoot"]
    assert kinds[8:12] == ["soft"] * 4                                # result shows, beat goes on
    assert labels[12:16] == ["3", "2", "1", "SHOOT"]
    assert s.shoot_time(0) == pytest.approx(10.0 + 7 * 0.4) and s.shoot_time(1) == pytest.approx(10.0 + 15 * 0.4)
    assert s.round_at(10.0) == -1 and s.round_at(s.t0 + 4 * 0.4) == 0 and s.round_at(s.shoot_time(1)) == 1
    assert s.round_at(s.end_time + 1.0) == 2                          # match over
    assert s.beat(0).t == 10.0 and s.beat(5).t == pytest.approx(12.0)


def hand(g, y=0.5, present=True):
    if not present:
        return HandObs(present=False, gesture=BACKGROUND, confidence=1.0)
    return HandObs(present=True, gesture=g, confidence=0.9, wrist_y=y, box=(0.2, 0.2, 0.8, 0.8))


def play(schedule, throws, use_cnn=False, offset=0.0):
    """
    A player who pumps on the pump beats and throws `throws[n]` on round n's throw beat (None = takes
    the hand away instead). Returns (engine, poses sent).
    """
    eng = DecisionEngine(DecisionConfig(mode="guided"), VoteConfig(k=2, min_confidence=0.7), use_cnn=use_cnn,
                         use_mp=True)
    eng.start_guided(schedule, offset)
    poses = []
    t_end = schedule.end_time + 0.5
    i = 0
    while True:
        t = schedule.t0 + i / FPS
        if t > t_end:
            break
        te = t - offset
        n = schedule.round_at(te)
        g, y = ROCK, 0.4
        if 0 <= n < schedule.rounds:
            shoot = schedule.shoot_time(n)
            phase = (te - schedule.t0) / schedule.beat_s
            y = 0.4 + 0.15 * abs(((phase + 0.5) % 1.0) - 0.5) * 2          # pumping on the beat
            if te >= shoot - 0.05:
                g, y = (throws[n], 0.55) if throws[n] is not None else (None, None)   # lands on the beat
        vy = None
        p = eng.on_motion(MotionObs(i, t, 300 if g is not None else 0, None,
                                    (g, 0.9) if use_cnn and g is not None else None, vy))
        if p:
            poses.append((round(te, 2), p))
        p = eng.on_hand(t, hand(g, y, present=g is not None), 300 if g is not None else 0)
        if p:
            poses.append((round(te, 2), p))
        i += 1
    return eng, poses


def test_guided_rounds_each_throw_decided_on_its_beat():
    s = BeatSchedule(t0=100.0, beat_s=0.4, pumps=3, rounds=4)
    eng, poses = play(s, [SCISSORS, PAPER, ROCK, SCISSORS])
    assert eng.round_results == {0: SCISSORS, 1: PAPER, 2: ROCK, 3: SCISSORS}
    moves = [p for _, p in poses if p != READY]
    assert moves == ["R", "S", "P", "R"]                              # the counter to each throw
    for n, (t, p) in enumerate([tp for tp in poses if tp[1] != READY]):
        assert -0.05 <= t - s.shoot_time(n) <= 0.45                   # decided within the throw window
    assert [p for _, p in poses].count(READY) == 3                     # back to ready at each later count-in


def test_guided_no_throw_is_a_missed_round_and_early_shapes_do_not_count():
    s = BeatSchedule(t0=0.0, beat_s=0.4, pumps=2, rounds=3)
    eng, poses = play(s, [None, PAPER, ROCK])                        # a fist kept on the beat is rock
    assert eng.round_results == {0: None, 1: PAPER, 2: ROCK}
    # a paper shown before the throw window opens is ignored
    eng = DecisionEngine(DecisionConfig(mode="guided"), VoteConfig(), use_cnn=False, use_mp=True)
    eng.start_guided(BeatSchedule(t0=0.0, beat_s=0.4, pumps=3, rounds=1), 0.0)
    first_pump = 4 * 0.4
    for i in range(12):                                               # paper during the count-in
        t = first_pump + i / FPS
        eng.on_motion(MotionObs(i, t, 300))
        eng.on_hand(t, hand(PAPER), 300)
    assert eng.state == ARMED and eng.round_results == {}


def test_guided_sync_offset_shifts_the_window():
    s = BeatSchedule(t0=0.0, beat_s=0.4, pumps=2, rounds=1)
    eng, poses = play(s, [SCISSORS], offset=0.12)                    # frames arrive 120 ms after the sound
    assert eng.round_results == {0: SCISSORS}
    snap = eng.snapshot()
    assert snap.state == HOLD and snap.round == 1 and snap.reason == "match over"


def test_guided_round_that_ends_before_its_window_closes_is_a_miss():
    # fast tempo, one beat between rounds: the next count-in starts before the 0.8 s throw window would close
    s = BeatSchedule(t0=0.0, beat_s=0.25, pumps=1, rounds=3, lead_beats=1, gap_beats=1)
    eng, _poses = play(s, [None, SCISSORS, None])
    assert eng.round_results == {0: None, 1: SCISSORS, 2: None}
