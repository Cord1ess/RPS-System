"""
What the robot plays (win / draw / lose), each decision's timing (first camera image showing the throw
-> deciding image -> command sent), the Speed card's numbers, the Wi-Fi ping helper, and the camera's
Linux / IP-camera settings.
"""

import os
import sys

import cv2
import numpy as np
import pytest

from rps.camera import CameraSource, MockSource, backend_name, fourcc_for, set_auto_exposure, set_manual_exposure
from rps.config import Config, DecisionConfig, VoteConfig
from rps.decision import DRAW, READY, ROBOT, YOU, DecisionEngine, outcome
from rps.game import BeatSchedule
from rps.hand_tracker import PAPER, ROCK, SCISSORS, HandObs
from rps.netping import ping_ms, summarize
from rps.pipeline import Pipeline

FPS = 30.0


# ---------------------------------------------------------------- what the robot plays
def test_outcome_of_a_round():
    assert outcome(ROCK, "P") == ROBOT and outcome(ROCK, "R") == DRAW and outcome(ROCK, "S") == YOU
    assert outcome(PAPER, "S") == ROBOT and outcome(SCISSORS, "P") == YOU


def hand(g):
    return HandObs(present=True, gesture=g, confidence=0.9, wrist_y=0.5, box=(0.3, 0.3, 0.7, 0.7))


@pytest.mark.parametrize("plays, expected", [("win", {ROCK: "P", PAPER: "S", SCISSORS: "R"}),
                                             ("draw", {ROCK: "R", PAPER: "P", SCISSORS: "S"}),
                                             ("lose", {ROCK: "S", PAPER: "R", SCISSORS: "P"})])
def test_robot_plays_to_win_draw_or_lose(plays, expected):
    for gesture, pose in expected.items():
        eng = DecisionEngine(DecisionConfig(mode="continuous", robot_plays=plays), VoteConfig(), use_cnn=False,
                             use_mp=True)
        sent = [p for i in range(6) if (p := eng.on_hand(i / FPS, hand(gesture), 0))]
        assert sent == [pose], (plays, gesture)
        assert outcome(gesture, pose) == {"win": ROBOT, "draw": DRAW, "lose": YOU}[plays]


def test_guided_rounds_follow_the_chosen_play():
    from tests.test_game import play          # the scripted beat-guide player
    s = BeatSchedule(t0=0.0, beat_s=0.4, pumps=2, rounds=3)
    for plays, moves in (("win", ["R", "S", "P"]), ("draw", ["S", "P", "R"]), ("lose", ["P", "R", "S"])):
        eng, poses = play(s, [SCISSORS, PAPER, ROCK], robot_plays=plays)
        assert eng.round_results == {0: SCISSORS, 1: PAPER, 2: ROCK}      # the throws are read the same
        assert [p for _, p in poses if p != READY] == moves, plays


# ---------------------------------------------------------------- decision timing
def test_first_seen_is_the_first_image_showing_the_throw():
    eng = DecisionEngine(DecisionConfig(mode="continuous"), VoteConfig(), use_cnn=False, use_mp=True)
    for i in range(5):
        eng.on_hand(10.0 + i / FPS, hand(ROCK), 0)
    for i in range(5, 9):                                    # the hand opens at image 5
        eng.on_hand(10.0 + i / FPS, hand(PAPER), 0)
    d = eng.last_decision
    assert d.gesture == PAPER and d.source == "mp"
    assert d.t_first == pytest.approx(10.0 + 5 / FPS)
    assert d.t_frame - d.t_first >= 0.045 - 1e-6              # held for mp_stable_ms before deciding
    assert eng.snapshot().decision is d


def test_rock_is_timed_from_the_landing_not_from_the_pumps():
    from tests.test_game import play
    s = BeatSchedule(t0=0.0, beat_s=0.4, pumps=2, rounds=1)
    eng, _poses = play(s, [ROCK])                            # a fist all the way through
    d = eng.last_decision
    assert d.gesture == ROCK
    assert d.t_first >= s.shoot_time(0) - 0.35 - 1e-6        # not from the pumps before the throw window
    assert d.t_first <= d.t_frame


class FakeHand:
    """Stands in for Mediapipe: sees `gesture` from image `open_at` on (a fist before)."""

    def __init__(self, open_at=3, gesture=PAPER):
        self.n, self.open_at, self.gesture = 0, open_at, gesture

    def process(self, bgr, roi, t, allow_skip=True):
        self.n += 1
        g = self.gesture if self.n > self.open_at else ROCK
        return HandObs(present=True, gesture=g, confidence=0.9, wrist_y=0.5, box=(0.3, 0.3, 0.7, 0.7), ms=2.0)

    def close(self):
        pass


def test_pipeline_times_each_decision_on_the_camera_clock():
    cfg = Config()
    cfg.decision.mode = "continuous"
    sent = []
    pipe = Pipeline(cfg, None, FakeHand(open_at=3), pose_sink=sent.append)
    src = MockSource(realtime=True).start()                   # frames stamped on the camera clock
    timings = []
    for _ in range(10):
        r = pipe.step(src.read(), src.roi)
        if r.decision is not None:
            timings.append(r.decision)
    assert sent == ["P", "S"] and len(timings) == 2            # the fist (robot: paper), then paper (scissors)
    d = timings[1]
    assert d.gesture == PAPER and d.pose == "S" and d.t_sent is not None
    assert d.frames >= 2 and d.read_ms == pytest.approx((d.frames - 1) * 1000 / 30, abs=2.0)
    assert 0 <= d.process_ms < 50 and d.software_ms == pytest.approx(d.read_ms + d.process_ms)
    assert d.mp_ms == 2.0                                       # the command went out after Mediapipe


def test_offline_replay_has_no_send_time():
    cfg = Config()
    cfg.decision.mode = "continuous"
    pipe = Pipeline(cfg, None, FakeHand(open_at=1), pose_sink=lambda p: None)
    src = MockSource(realtime=False).start()                   # recorded clock: sending cannot be timed
    d = next(r.decision for r in (pipe.step(src.read(), src.roi) for _ in range(8)) if r.decision)
    assert d.t_sent is None and d.process_ms is None and d.software_ms is None and d.read_ms > 0


# ---------------------------------------------------------------- Wi-Fi ping
def test_ping_summary():
    s = summarize([4.0, 6.0, None, 5.0])
    assert s["answered"] == 3 and s["sent"] == 4 and s["median_ms"] == 5.0 and s["one_way_ms"] == 2.5
    assert not s["slow"] and summarize([60.0, 80.0, 70.0])["slow"]
    assert summarize([None, None]) == {"answered": 0, "sent": 2}


def test_ping_this_computer():
    rtts = ping_ms("127.0.0.1", count=3, interval_s=0.05)
    assert len(rtts) == 3 and sum(r is not None for r in rtts) >= 2 and all(r < 50 for r in rtts if r)


# ---------------------------------------------------------------- camera on Linux and IP cameras
def test_windows_drivers_become_v4l2_on_linux():
    assert backend_name("msmf", "linux") == "v4l2" and backend_name("dshow", "linux") == "v4l2"
    assert backend_name("msmf", "win32") == "msmf" and backend_name("any", "linux") == "any"
    assert backend_name("nonsense", "win32") == "any"
    assert fourcc_for("v4l2", "YUY2") == "YUYV" and fourcc_for("msmf", "YUY2") == "YUY2"


class FakeCap:
    def __init__(self):
        self.calls = []

    def set(self, prop, value):
        self.calls.append((prop, value))


def test_exposure_is_translated_for_v4l2():
    cap = FakeCap()
    set_manual_exposure(cap, "v4l2", -6)                        # 15.6 ms
    assert cap.calls == [(cv2.CAP_PROP_AUTO_EXPOSURE, 1), (cv2.CAP_PROP_EXPOSURE, 156)]
    cap = FakeCap()
    set_manual_exposure(cap, "dshow", -6)
    assert cap.calls == [(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25), (cv2.CAP_PROP_EXPOSURE, -6)]
    cap = FakeCap()
    set_auto_exposure(cap, "v4l2")
    set_auto_exposure(cap, "msmf")
    assert cap.calls == [(cv2.CAP_PROP_AUTO_EXPOSURE, 3), (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)]


def test_ip_camera_address_opens_through_ffmpeg(tmp_path):
    path = str(tmp_path / "stream.avi")                        # FFmpeg reads files like it reads streams
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for i in range(10):
        writer.write(np.full((240, 320, 3), 20 * i, np.uint8))
    writer.release()
    cfg = Config()
    cfg.camera.url = path
    src = CameraSource(cfg.camera, cfg.roi).start()
    try:
        frame = src.read(timeout=3.0)
        assert frame is not None and frame.bgr.shape == (240, 320, 3) and frame.live
        assert src.info["backend"] == "ffmpeg" and src.info["width"] == 320
        assert "nobuffer" in os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    finally:
        src.stop()


@pytest.mark.skipif(sys.platform != "win32", reason="the demo laptop's check")
def test_a_missing_camera_says_what_to_check():
    cfg = Config()
    cfg.camera.index = 97
    with pytest.raises(RuntimeError, match="Could not open camera 97"):
        CameraSource(cfg.camera, cfg.roi).open()


@pytest.mark.skipif(not os.path.exists("models/hand_landmarker.task"), reason="Mediapipe model not downloaded")
def test_benchmark_tool_measures_each_stage(tmp_path):
    import importlib.util
    from tests.test_camera import image, write_recording
    spec = importlib.util.spec_from_file_location("benchmark", "tools/benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    write_recording(str(tmp_path), [(i / 30, image(i, 64)) for i in range(12)])
    res = bench.run(Config(), str(tmp_path), "mediapipe", max_frames=10)    # loops the short recording
    assert res["frames"] == 10 and res["mediapipe_ms"]["n"] == 10 and res["total_ms"]["median"] > 0
    assert "keeps_up_at_30fps" in res and bench.machine()["cores"] >= 1
