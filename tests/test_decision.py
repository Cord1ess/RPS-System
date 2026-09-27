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
