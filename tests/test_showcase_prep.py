"""
Interface size, robot handover between computers, and the faster camera handling (live settings,
background restarts). Nothing here talks to the real robot or leaves this computer.
"""

import os
import shutil
import time

import cv2
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rps.camera import CameraSource  # noqa: E402
from rps.config import Config  # noqa: E402
from rps.handoff import Handoff, encode_take, parse_take  # noqa: E402
from rps.ui import scale  # noqa: E402


# ---------------------------------------------------------------- interface size
def test_scale_choices_and_fit():
    assert scale.parse("0.8") == 0.8 and scale.parse("80") == 0.8 and scale.parse("80%") == 0.8
    assert scale.parse("auto") is None and scale.parse("") is None
    assert scale.parse("9") == scale.MIN_SCALE and scale.parse("2.5") == scale.MAX_SCALE      # 9 = 9%
    assert scale.fit_scale(1366, 728) == 0.8          # a small laptop screen
    assert scale.fit_scale(1536, 824) == 0.9          # a 1080p laptop at 125%
    assert scale.fit_scale(1706, 1066) == 1.0         # the 2K demo laptop: never enlarged
    assert scale.fit_scale(400, 300) == scale.MIN_SCALE


def test_scale_is_set_before_qt_and_cleared_for_a_restart():
    env = {}
    assert scale.before_qt("0.8", env) is False and env[scale.ENV] == "0.8" and env[scale.MARK] == "0.8"
    assert scale.clean_env(env) == {}                   # the app's own size: the restart chooses again
    env = {}
    assert scale.before_qt("auto", env) is True and env == {}       # measured once Qt is up
    assert scale.before_qt("1.0", env) is False and env == {}       # 100%: nothing to set
    outside = {scale.ENV: "1.5"}
    assert scale.before_qt("0.8", outside) is False and outside == {scale.ENV: "1.5"}   # set by the user: kept
    assert scale.clean_env(outside) == outside
    relaunched = {scale.ENV: "0.85", scale.MARK: "0.85"}
    assert scale.before_qt("auto", relaunched) is False             # Auto already measured: no loop


def test_restart_drops_the_scale_argument():
    import app
    assert app.without_scale(["--page", "play", "--scale", "0.8", "--fullscreen"]) == ["--page", "play",
                                                                                         "--fullscreen"]
    assert app.without_scale(["--scale=70", "--mock"]) == ["--mock"]


# ---------------------------------------------------------------- robot handover
def test_takeover_message():
    data = encode_take("abc123", "raspberrypi", "192.168.0.126")
    assert data == b"RPS-CTRL:TAKE:abc123:raspberrypi:192.168.0.126"
    assert parse_take(data) == ("abc123", "raspberrypi", "192.168.0.126")
    for bad in (b"RPS:ROCK", b"RPS-CTRL:TAKE:only:two", b"\xff\xfe", b"RPS-CTRL:TAKE:a::c"):
        assert parse_take(bad) is None


def test_another_copy_taking_the_robot_is_heard_and_our_own_is_ignored():
    heard = []
    listener = Handoff(on_taken=lambda computer, robot: heard.append((computer, robot)), port=42291,
                       targets=[("127.0.0.1", 42291)])
    other = Handoff(on_taken=lambda *_: None, port=42291, targets=[("127.0.0.1", 42291)], listen=False)
    try:
        assert listener.listening
        assert listener.announce("10.0.0.5")             # our own announcement: ignored
        assert other.announce("192.168.0.126")
        end = time.time() + 2.0
        while time.time() < end and not heard:
            time.sleep(0.02)
        assert heard == [(other.computer, "192.168.0.126")]
    finally:
        listener.stop()


# ---------------------------------------------------------------- camera
class FakeCap:
    def __init__(self):
        self.calls = []

    def set(self, prop, value):
        self.calls.append((prop, value))


def test_exposure_and_colour_balance_apply_to_a_running_camera():
    cfg = Config()
    src = CameraSource(cfg.camera, cfg.roi)
    src.cap, src.backend = FakeCap(), "dshow"
    cfg.camera.lock_exposure, cfg.camera.exposure = True, -7
    cfg.camera.lock_white_balance, cfg.camera.wb_temperature = True, 5000
    src.request_apply()
    assert src._apply_pending
    src._apply_live()
    assert (cv2.CAP_PROP_EXPOSURE, -7) in src.cap.calls and (cv2.CAP_PROP_WB_TEMPERATURE, 5000) in src.cap.calls
    src.cap.calls.clear()
    cfg.camera.lock_exposure = cfg.camera.lock_white_balance = False      # back to automatic, still running
    src._apply_live()
    assert src.cap.calls == [(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75), (cv2.CAP_PROP_AUTO_WB, 1)]
    src.backend, src.cap.calls = "ffmpeg", []
    src._apply_live()                                     # an IP camera is set in its own settings
    assert src.cap.calls == []


# ---------------------------------------------------------------- the app
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from rps.ui.style import apply_theme
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def window(qapp, tmp_path):
    from rps.ui.common import AppState
    from rps.ui.main_window import MainWindow
    shutil.copy("config.json", tmp_path / "config.json")
    state = AppState(str(tmp_path / "config.json"), str(tmp_path / "data"))
    state.cfg.robot.mode, state.cfg.robot.port = "simulated", 42197
    w = MainWindow(state, start_kind="mock", remember=False)
    w.confirm_close = False
    w.handoff.targets = [("127.0.0.1", 42292)]            # announcements stay on this computer
    yield w
    w.close()
    pump(qapp, 0.3)


def test_the_window_fits_a_small_laptop_screen(qapp, window):
    m = window.minimumSizeHint()
    assert m.width() <= 1100 and m.height() <= 768, (m.width(), m.height())
    game = window.pages[6]
    game._set_cue("Waiting for camera", "#ffffff")         # long cue text shrinks instead of widening
    assert game.cue.minimumSizeHint().width() < 100


def test_camera_restarts_in_the_background(qapp, window):
    window.start_camera("mock")
    pump(qapp, 0.6)
    assert window.worker.isRunning()
    t0 = time.perf_counter()
    window.restart_camera("mock")                          # returns at once; reopens when the old one is free
    assert time.perf_counter() - t0 < 0.1
    pump(qapp, 0.8)
    frames = []
    window.worker.frame_ready.connect(lambda: frames.append(1))
    pump(qapp, 0.7)
    assert window.worker.isRunning() and not window.worker.stopping and len(frames) > 5   # images again
    t0 = time.perf_counter()
    window.stop_camera(wait=False)
    assert time.perf_counter() - t0 < 0.1
    pump(qapp, 1.0)
    assert not window.worker.isRunning()


@pytest.mark.skipif(not os.path.exists("models/hand_landmarker.task"), reason="Mediapipe model not downloaded")
def test_a_game_stops_when_another_computer_takes_the_robot(qapp, window):
    from rps.ui.common import select_data
    cfg = window.state.cfg.robot
    cfg.mode, cfg.host, cfg.port = "real", "127.0.0.1", 42197     # nothing listens there: no robot involved
    window.tabs.setCurrentIndex(7)
    debug = window.pages[7]
    select_data(debug.detector, "mediapipe")
    debug._toggle()
    pump(qapp, 3.0)
    assert debug.pipeline is not None and debug.link is not None
    window._robot_taken("raspberrypi", "10.9.9.9")               # another robot: ignored
    assert debug.pipeline is not None
    window._robot_taken("raspberrypi", "127.0.0.1")              # this robot: hand over
    assert debug.pipeline is None
    assert "raspberrypi took over the robot" in debug.log.toPlainText()
    cfg.handoff = False                                          # handover off: keeps playing
    debug._toggle()
    pump(qapp, 2.0)
    window._robot_taken("raspberrypi", "127.0.0.1")
    assert debug.pipeline is not None
    debug._toggle()
