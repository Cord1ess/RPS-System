"""
Offscreen checks of the desktop app's failure paths: a camera that cannot open while recording,
leaving Play while it runs, a stop requested right after start, job cancellation, and the parsing
of the lines the background jobs print.
"""

import os
import shutil
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from rps.ui.common import AppState, select_data  # noqa: E402
from rps.ui.main_window import MainWindow  # noqa: E402
from rps.ui.style import apply_theme  # noqa: E402
from rps.ui.tab_train import EPOCH, LOPO_ROW, MEAN_ROW  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
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
    shutil.copy("config.json", tmp_path / "config.json")
    state = AppState(str(tmp_path / "config.json"), str(tmp_path / "data"))
    state.cfg.robot.port = 42198
    w = MainWindow(state, start_kind="mock")
    yield w
    w.close()
    pump(qapp, 0.3)


def test_record_with_a_camera_that_cannot_open_unlocks_the_app(qapp, window, tmp_path):
    window.tabs.setCurrentIndex(1)
    rec = window.pages[1]
    window.stop_camera()
    window.default_kind, window.default_video = "video", str(tmp_path / "missing")   # opening fails
    rec.person.setEditText("tester")
    rec._start()
    pump(qapp, 1.5)
    assert rec._pending is None and rec._active is None
    assert rec.start_btn.isEnabled()
    assert all(window.tabs.isTabEnabled(i) for i in range(window.tabs.count()))


def test_leaving_play_stops_the_game(qapp, window):
    if not os.path.exists(window.state.cfg.hand.model_path):
        pytest.skip("hand tracker model not downloaded")
    window.tabs.setCurrentIndex(5)
    play = window.pages[5]
    select_data(play.detector, "mediapipe")
    play.mock_esp.setChecked(True)
    play._toggle()
    pump(qapp, 3.0)
    assert play.pipeline is not None and play.link is not None
    window.tabs.setCurrentIndex(0)
    pump(qapp, 0.3)
    assert play.pipeline is None and play.link is None and play.mock is None


def test_stop_right_after_start_is_not_lost(qapp, window):
    window.start_camera("mock")
    window.stop_camera()
    assert not window.worker.isRunning()


def test_cancel_is_reported_as_cancelled(qapp, window):
    ev = window.pages[4]
    ev.runner.start(["-c", "import time; time.sleep(20)"])
    ev._busy("compare", "Comparison running")
    pump(qapp, 0.5)
    ev.runner.kill()
    pump(qapp, 1.0)
    assert ev.verdict.text() == "Cancelled"


def test_job_output_lines_are_understood(qapp, window):
    # formats printed by train.py
    person, s, mean, std = "Mary Ann", 0.5, 0.417, 0.083
    assert LOPO_ROW.match(f"  {person:12s} balanced acc {s * 100:5.1f}%").groups() == ("Mary Ann", "50.0")
    assert MEAN_ROW.match(f"  mean {mean * 100:5.1f}% +/- {std * 100:4.1f}%").groups() == ("41.7", "8.3")
    m = EPOCH.search(f"Epoch [{3:02d}/{30}] loss {0.4321:.4f} acc {88.2:5.1f}% || val balanced acc {61.0:5.1f}%")
    assert m.group(1) == "03" and m.group(5) == "61.0"
    # Dextra check result line (tools/dextra_transfer.py) -> plain text and a working apply button
    ev = window.pages[4]
    ev._line('@@BEST {"rotate": 0, "flip": true, "event_count": 5000, "contrast_threshold": 0.2, "zoom": 1.0, '
             '"balanced": 0.69, "per_gesture": {"rock": 0.74, "paper": 0.63, "scissors": 0.7}, '
             '"current_balanced": 0.6}')
    assert "mirrored" in ev.transfer_result.text() and "69%" in ev.transfer_result.text()
    assert not ev.apply_btn.isHidden()
    ev._apply_best()
    cfg = window.state.cfg
    assert cfg.cnn.flip is True and cfg.dvs.event_count == 5000 and window.state.dirty


def test_play_starts_without_any_motion_model(qapp, window, monkeypatch):
    if not os.path.exists(window.state.cfg.hand.model_path):
        pytest.skip("hand tracker model not downloaded")
    import rps.ui.tab_play as tab_play
    monkeypatch.setattr(tab_play, "model_choices", lambda: [])        # nothing downloaded or trained
    window.state.cfg.cnn.model_path = "models/not_trained_yet.pth"
    window.tabs.setCurrentIndex(5)
    play = window.pages[5]
    select_data(play.detector, "fused")                                # the default
    play.mock_esp.setChecked(True)
    play._toggle()
    pump(qapp, 3.0)
    assert play.pipeline is not None and play.cnn is None and play.hand is not None
    assert "no motion model" in play.summary.text()
    play._toggle()
