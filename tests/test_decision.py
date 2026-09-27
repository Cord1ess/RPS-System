import os

import numpy as np

from rps.config import DecisionConfig, VoteConfig
from rps.decision import (ARMED, HOLD, IDLE, READY, SHOOT, DecisionEngine, MotionObs)
from rps.hand_tracker import BACKGROUND, PAPER, ROCK, SCISSORS, HandObs

FPS = 30.0


def hand(gesture, y=0.5, conf=0.9):
    return HandObs(present=True, gesture=gesture, confidence=conf, wrist_y=y, box=(0.2, 0.2, 0.8, 0.8))


def engine(mode="continuous", use_cnn=True, use_mp=True, **kw):
    cfg = DecisionConfig(mode=mode, **kw)
    return DecisionEngine(cfg, VoteConfig(k=2, min_confidence=0.7), use_cnn=use_cnn, use_mp=use_mp)


def step(eng, i, events, cnn=None, mp=None, events_in_hand=None, centroid_y=None, vy=None):
    t = i / FPS
    poses = []
    p = eng.on_motion(MotionObs(i, t, events, centroid_y, cnn, vy))
    if p:
        poses.append((t, p))
    if mp is not None:
        p = eng.on_hand(t, mp, events if events_in_hand is None else events_in_hand)
        if p:
            poses.append((t, p))
    return poses


# ----------------------------------------------------------------------------- continuous

def test_continuous_cnn_commit_and_background_holds():
    eng = engine(use_mp=False)
    poses = []
    poses += step(eng, 0, 300, cnn=(PAPER, 0.9))
    poses += step(eng, 1, 300, cnn=(PAPER, 0.9))
    assert poses == [(1 / FPS, "S")]
    for i in range(2, 10):
        assert step(eng, i, 300, cnn=(BACKGROUND, 0.95)) == []
    assert eng.pose == "S"


def test_continuous_goes_ready_when_idle():
    eng = engine(use_mp=False, idle_timeout_s=0.5)
    step(eng, 0, 300, cnn=(ROCK, 0.9))
    step(eng, 1, 300, cnn=(ROCK, 0.9))
    assert eng.pose == "P"
    poses = []
    for i in range(2, 40):
        poses += step(eng, i, 0)
    assert poses and poses[-1][1] == READY


def test_still_hold_mediapipe_only_never_switches():
    eng = engine(use_cnn=False)
    rng = np.random.default_rng(0)
    for i in range(300):   # 10 s of a held rock with occasional unsure frames
        g = ROCK if rng.random() > 0.1 else -1
        step(eng, i, 5, mp=hand(g, conf=0.9 if g == ROCK else 0.0))
    assert eng.pose == "P" and eng.commits == 1 and eng.switches == 0


def test_stale_mediapipe_cannot_override_cnn_commit_during_motion():
    eng = engine()
    step(eng, 0, 400, cnn=(SCISSORS, 0.9), mp=hand(ROCK), events_in_hand=400)
    step(eng, 1, 400, cnn=(SCISSORS, 0.9), mp=hand(ROCK), events_in_hand=400)
    assert eng.pose == "R"
    for i in range(2, 6):   # hand still blurred/moving: MediaPipe says rock, must be ignored
        step(eng, i, 300, mp=hand(ROCK), events_in_hand=300)
    assert eng.pose == "R" and eng.human == SCISSORS


def test_steady_tracker_reading_blocks_a_contradicting_motion_vote():
    eng = engine()
    for i in range(5):                                  # tracker steadily sees paper while the hand moves
        step(eng, i, 300, mp=hand(PAPER), events_in_hand=300)
    for i in range(5, 9):                               # motion model insists on scissors
        step(eng, i, 300, cnn=(SCISSORS, 0.95), mp=hand(PAPER), events_in_hand=300)
    assert eng.human != SCISSORS
    eng = engine()
    for i in range(4):                                  # tracker has no hand: the motion model decides alone
        step(eng, i, 300, cnn=(SCISSORS, 0.95), mp=HandObs(False, BACKGROUND, 1.0))
    assert eng.human == SCISSORS


def test_countdown_tracker_fist_does_not_block_the_throw():
    # a pumping fist in the tracker must not stop the motion model's early paper/scissors
    eng = engine(mode="countdown", use_mp=True)
    eng.state, eng.pumps = "SHOOT", 3
    eng.shoot_t = 0.0
    for i in range(4):
        step(eng, i, 300, mp=hand(ROCK), events_in_hand=300)
    poses = []
    for i in range(4, 6):
        poses += step(eng, i, 300, cnn=(PAPER, 0.95), mp=hand(ROCK), events_in_hand=300)
    assert decided(poses) == ["S"]


def test_mediapipe_corrects_when_still_after_margin():
    eng = engine(mp_after_cnn_commit_s=0.15)
    step(eng, 0, 400, cnn=(SCISSORS, 0.9))
    step(eng, 1, 400, cnn=(SCISSORS, 0.9))
    assert eng.human == SCISSORS
    for i in range(2, 20):   # hand is now still and clearly paper
        step(eng, i, 0, mp=hand(PAPER), events_in_hand=0)
    assert eng.human == PAPER and eng.pose == "S"


# ----------------------------------------------------------------------------- countdown

def countdown_trajectory(final_gesture, pumps=3, period=0.4, top=0.35, bottom=0.65, hold_s=1.0, drop_pump=None):
    """Per-frame (y, vy, events, gesture) for pumps + final downstroke + hold. drop_pump hides one pump's motion."""
    frames = []
    n_pump = int(period * FPS)
    for p in range(pumps):
        for k in range(n_pump):
            y = top + (bottom - top) * 0.5 * (1 - np.cos(2 * np.pi * k / n_pump))
            frames.append([y, ROCK, p == drop_pump])
    n_down = max(4, int(period * FPS) // 2)
    for k in range(n_down + 1):
        y = top + (bottom - top) * 0.5 * (1 - np.cos(np.pi * k / n_down))
        frames.append([y, final_gesture if k > n_down // 2 else ROCK, False])
    for _ in range(int(hold_s * FPS)):
        frames.append([bottom, final_gesture, False])
    out, prev_y = [], frames[0][0]
    for y, g, hidden in frames:
        vy = (y - prev_y) * FPS
        events = int(abs(y - prev_y) * 6000)
        prev_y = y
        out.append((y, 0.0 if hidden else vy, events, g))
    return out


def run_countdown(final_gesture, use_cnn=True, use_mp=True, **kw):
    traj_kw = {k: kw.pop(k) for k in ("pumps", "period", "drop_pump") if k in kw}
    eng = engine(mode="countdown", use_cnn=use_cnn, use_mp=use_mp, **kw)
    poses, states = [], []
    for i, (y, vy, events, g) in enumerate(countdown_trajectory(final_gesture, **traj_kw)):
        cnn = (g, 0.9) if events >= 100 else None
        poses += step(eng, i, events, cnn=cnn, mp=hand(g, y=y) if use_mp else None, vy=vy)
        states.append(eng.state)
    return eng, poses, states


def decided(poses):
    return [p for _, p in poses if p != READY]


def test_countdown_scissors_commits_once_after_pumps():
    eng, poses, states = run_countdown(SCISSORS)
    assert decided(poses) == ["R"]
    assert eng.commits == 1 and eng.state == HOLD
    assert SHOOT in states and states.index(SHOOT) > states.index(ARMED)


def test_countdown_paper_commits_once():
    eng, poses, _ = run_countdown(PAPER)
    assert decided(poses) == ["S"]


def test_countdown_rock_only_after_the_throw_lands():
    eng, poses, states = run_countdown(ROCK)
    commits = [(t, p) for t, p in poses if p != READY]
    assert [p for _, p in commits] == ["P"]
    # never during the pumps: the commit comes after the third pump put us in SHOOT
    assert commits[0][0] * FPS > states.index(SHOOT)


def test_countdown_motion_model_only():
    eng, poses, _ = run_countdown(SCISSORS, use_mp=False)
    assert decided(poses) == ["R"]


def test_countdown_fast_and_slow_players():
    for period in (0.3, 0.7):
        for gesture, pose in ((PAPER, "S"), (ROCK, "P")):
            eng, poses, _ = run_countdown(gesture, period=period)
            assert decided(poses) == [pose], (period, gesture)
            assert eng.pump.tempo is not None


def test_countdown_missed_pump_still_catches_the_throw():
    # the second pump is invisible (tracking lost): only 2 pumps counted, the throw lands on the beat
    eng, poses, _ = run_countdown(ROCK, drop_pump=1)
    assert decided(poses) == ["P"] and eng.commits == 1


def test_countdown_extra_pump_is_not_the_throw():
    eng, poses, _ = run_countdown(PAPER, pumps=4)
    assert decided(poses) == ["S"] and eng.commits == 1


def test_countdown_starts_idle_and_arms():
    eng = engine(mode="countdown")
    assert eng.state == IDLE
    step(eng, 0, 0, mp=hand(ROCK))
    assert eng.state == ARMED and eng.pose == READY


# ----------------------------------------------------------------------------- several rounds in a row

def rounds_trajectory(gestures, pumps=2, period=0.4, top=0.35, bottom=0.65, land_s=0.35):
    """
    Back-to-back rounds the way people really play: `pumps` pumps, the throw on the next down
    stroke, a short look at the result, then straight up into the next round's pumps.
    """
    frames = []
    n = int(period * FPS)

    def move(y0, y1, count, g):
        for k in range(1, count + 1):
            frames.append((y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * k / count)), g))

    frames.append((top, ROCK))
    for g in gestures:
        for _ in range(pumps):
            move(top, bottom, n // 2, ROCK)
            move(bottom, top, n - n // 2, ROCK)
        move(top, bottom, n // 2, ROCK)
        frames[-2] = (frames[-2][0], g)            # the hand opens on the way down
        frames[-1] = (frames[-1][0], g)
        frames.extend([(bottom, g)] * int(land_s * FPS))
        move(bottom, top, n - n // 2, ROCK)
    out, prev = [], frames[0][0]
    for y, g in frames:
        out.append((y, (y - prev) * FPS, int(abs(y - prev) * 6000), g))
        prev = y
    return out


def run_rounds(gestures, use_cnn=False, **kw):
    traj_kw = {k: kw.pop(k) for k in ("pumps", "period", "land_s") if k in kw}
    eng = engine(mode="countdown", use_cnn=use_cnn, **kw)
    poses = []
    for i, (y, vy, events, g) in enumerate(rounds_trajectory(gestures, **traj_kw)):
        cnn = (g, 0.9) if use_cnn and events >= 100 else None
        poses += step(eng, i, events, cnn=cnn, mp=hand(g, y=y), vy=vy)
    return eng, poses


def test_back_to_back_rounds_each_decided_once():
    gestures = [SCISSORS, PAPER, ROCK, SCISSORS, ROCK, PAPER]
    counter = {ROCK: "P", PAPER: "S", SCISSORS: "R"}
    for period in (0.3, 0.4, 0.6):
        eng, poses = run_rounds(gestures, period=period)
        assert decided(poses) == [counter[g] for g in gestures], period
        assert eng.commits == len(gestures)
        # every round goes back to ready while pumping
        assert [p for _, p in poses].count(READY) >= len(gestures) - 1


def test_learned_tempo_is_the_pump_rhythm_not_the_round_length():
    for period in (0.3, 0.5):
        eng, _ = run_rounds([SCISSORS, PAPER, SCISSORS, PAPER, SCISSORS], period=period)
        assert eng.pump.tempo is not None and abs(eng.pump.tempo - period) < 0.08 * period + 0.04, period


def test_open_hand_after_two_pumps_is_the_throw_on_the_third_beat():
    # pumps_before_shoot=3 with one miss tolerated: a player who throws on the 3rd down stroke
    eng, poses = run_rounds([SCISSORS], pumps=2)
    assert decided(poses) == ["R"]
    eng, poses = run_rounds([PAPER], pumps=3)          # ... and one who throws on the 4th
    assert decided(poses) == ["S"]


def test_no_decision_from_pumping_alone():
    eng = engine(mode="countdown", use_cnn=False)
    poses = []
    traj = rounds_trajectory([ROCK], pumps=8, land_s=0.0)[:-20]     # 8 pumps, never lands
    for i, (y, vy, events, g) in enumerate(traj[:8 * 12]):
        poses += step(eng, i, events, mp=hand(ROCK, y=y), vy=vy)
    assert decided(poses) == []


# ----------------------------------------------------------------------------- real recording

def load_fixture(name):
    d = np.load(os.path.join(os.path.dirname(__file__), "fixtures", name))
    return d["motion"], d["hand"], str(d["label"])


def fixture_throw_times(h):
    """When the hand tracker first sees a stable paper/scissors right after a fist (the throw)."""
    throws, last_rock, run, prev = [], -1e9, 0, None
    for row in h:
        t, g = float(row[0]), (int(row[2]) if row[1] else None)
        run = run + 1 if g == prev else 1
        prev = g
        if g == ROCK and run >= 3:
            last_rock = t
        if g in (PAPER, SCISSORS) and run == 3 and t - last_rock < 1.5:
            throws.append(t - 2 / FPS)
            last_rock = -1e9
    return throws


def test_real_recording_every_throw_decided_once():
    """
    45 s of real countdown play (2 pumps + throw, a new round every ~1.6 s, pumps 0.37 s apart):
    per-frame inputs recorded from the pipeline on data/recordings (no images).
    """
    m, h, label = load_fixture("real_throws_scissors.npz")
    eng = DecisionEngine(DecisionConfig(mode="countdown"), VoteConfig(), use_cnn=False, use_mp=True)
    commits = []

    def nan_none(v):
        return None if np.isnan(v) else float(v)

    for i in range(len(m)):
        t = float(m[i, 0])
        eng.on_motion(MotionObs(i, t, int(m[i, 1]), nan_none(m[i, 2]), None, nan_none(m[i, 3])))
        r = h[i]
        box = None if np.isnan(r[5]) else tuple(float(x) for x in r[5:9])
        before = eng.commits
        eng.on_hand(t, HandObs(present=bool(r[1]), gesture=int(r[2]), confidence=float(r[3]),
                               wrist_y=nan_none(r[4]), box=box, skipped=bool(r[9])),
                    None if np.isnan(r[10]) else int(r[10]))
        if eng.commits > before:
            commits.append((t, eng.human))
    throws = fixture_throw_times(h)
    assert label == "scissors" and len(throws) == 28
    assert len(commits) == len(throws)
    assert all(g == SCISSORS for _, g in commits)
    for (tc, _), tt in zip(commits, throws):      # each decision belongs to its own throw
        assert -0.1 <= tc - tt <= 0.25
    assert 0.3 <= eng.pump.tempo <= 0.45
