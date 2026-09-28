"""
Offscreen smoke test of the desktop app: mock camera -> Setup readouts, a real Record session,
the Play loop with MediaPipe and a simulated robot, and saving settings. About 15 s.
"""

import os
import shutil
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from rps.recorder import list_recordings  # noqa: E402
from rps.ui.common import AppState, select_data  # noqa: E402
from rps.ui.main_window import MainWindow  # noqa: E402
from rps.ui.style import apply_theme  # noqa: E402


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


def test_app_end_to_end(qapp, tmp_path):
    shutil.copy("config.json", tmp_path / "config.json")
    state = AppState(str(tmp_path / "config.json"), str(tmp_path / "data"))
    state.cfg.robot.port = 42199                       # avoid clashing with a real mock ESP on 4210
    state.cfg.robot.mode = "simulated"                 # never the real robot in tests
    w = MainWindow(state, start_kind="mock", remember=False)
    w.confirm_close = False
    try:
        setup = w.pages[0]
        select_data(setup.source, "mock")
        setup._toggle_camera()
        pump(qapp, 1.5)
        assert "fps" in setup.chip_fps.text()

        w.tabs.setCurrentIndex(2)
        rec = w.pages[2]
        rec.person.setEditText("tester")
        select_data(rec.kind, "show")
        select_data(rec.label, "rock")
        rec.duration.setValue(2)
        rec.countdown_s = 0.2
        rec._start()
        pump(qapp, 4.0)
        recs = list_recordings(state.recordings_root)
        assert len(recs) == 1 and recs[0]["label"] == "rock" and recs[0]["video_check_ok"]
        assert "Saved" in rec.status.text()

        if os.path.exists(state.cfg.hand.model_path):
            w.tabs.setCurrentIndex(7)
            play = w.pages[7]                              # Play Debug
            select_data(play.detector, "mediapipe")
            play._toggle()                                 # models load in the background
            pump(qapp, 3.0)
            assert play.pipeline is not None and not play._loading
            assert play.chip_robot._level == "ok"
            assert play.m_round.value.text() != "Stopped"
            play._toggle()
            assert play.m_round.value.text() == "Stopped"

        state.cfg.vote.k = 3
        state.save()
        assert AppState(str(tmp_path / "config.json")).cfg.vote.k == 3
    finally:
        w.close()
        pump(qapp, 0.3)
