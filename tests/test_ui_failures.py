"""
Offscreen checks of the desktop app: failure paths (a camera that cannot open while recording,
leaving a Play page while it runs, a stop right after start, cancelled jobs), the pages' choices
and settings, the Bot tuning commands, the Play match and the lines the background jobs print.
Every robot here is the simulated one on this computer.
"""

import os
import shutil
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from rps.pipeline import RECOGNIZERS  # noqa: E402
from rps.ui.common import AppState, select_data  # noqa: E402
from rps.ui.main_window import MainWindow  # noqa: E402
from rps.ui.style import apply_theme  # noqa: E402
from rps.ui.tab_train import EPOCH, LOPO_ROW, MEAN_ROW  # noqa: E402

SETUP, BOT, RECORD, DATASET, TRAIN, EVALUATE, PLAY, DEBUG, SETTINGS = range(9)
HAVE_MP = os.path.exists("models/hand_landmarker.task")


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
    state.cfg.robot.mode = "simulated"                         # never the real robot in tests
    w = MainWindow(state, start_kind="mock", remember=False)
    w.confirm_close = False
    yield w
    w.close()
    pump(qapp, 0.3)


def test_pages_in_order(qapp, window):
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == ["1  Setup", "2  Bot tuning", "3  Record", "4  Dataset", "5  Train", "6  Evaluate", "7  Play",
                      "Play Debug", "Settings"]


def test_record_with_a_camera_that_cannot_open_unlocks_the_app(qapp, window, tmp_path):
    window.tabs.setCurrentIndex(RECORD)
    rec = window.pages[RECORD]
    window.stop_camera()
    window.default_kind, window.default_video = "video", str(tmp_path / "missing")   # opening fails
    rec.person.setEditText("tester")
    rec._start()
    pump(qapp, 1.5)
    assert rec._pending is None and rec._active is None
    assert rec.start_btn.isEnabled()
    assert all(window.tabs.isTabEnabled(i) for i in range(window.tabs.count()))


@pytest.mark.skipif(not HAVE_MP, reason="Mediapipe model not downloaded")
def test_leaving_play_debug_stops_the_game(qapp, window):
    window.tabs.setCurrentIndex(DEBUG)
    play = window.pages[DEBUG]
    select_data(play.detector, "mediapipe")
    play._toggle()
    pump(qapp, 3.0)
    assert play.pipeline is not None and play.link is not None and play.mock is not None
    window.tabs.setCurrentIndex(SETUP)
    pump(qapp, 0.3)
    assert play.pipeline is None and play.link is None and play.mock is None


def test_stop_right_after_start_is_not_lost(qapp, window):
    window.start_camera("mock")
    window.stop_camera()
    assert not window.worker.isRunning()


def test_cancel_is_reported_as_cancelled(qapp, window):
    ev = window.pages[EVALUATE]
    ev.runner.start(["-c", "import time; time.sleep(20)"])
    ev._busy("compare", "Running")
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
    # replay_eval.py result, skip and verdict lines -> table rows and the verdict
    ev = window.pages[EVALUATE]
    ev._line('@@RESULT {"setting": "", "source": "mediapipe", "show_hold_acc": 0.987, "show_switches_per_min": 5.3, '
             '"throw_acc": 1.0, "throw_commits": 28, "throws_planned": 30, "commit_vs_throw_ms": 64.0, '
             '"bg_false_commits_per_min": null}')
    ev._line('@@SKIP {"source": "dextra_tuned", "reason": "Dextra Tuned does not exist yet: tune Dextra on the Train '
             'page."}')
    ev._line("Best on these recordings: Mediapipe")
    ev._done(0)
    assert ev.table.item(0, 0).text() == "Mediapipe" and ev.table.item(0, 1).text() == "98.7%"
    assert ev.table.item(1, 0).text() == "Dextra Tuned" and ev.table.item(1, 1).text() == "not available"
    assert ev.verdict.text() == "Best on these recordings: Mediapipe"
    # Dextra Raw view check result line (tools/dextra_transfer.py) -> plain text and a working apply button
    ev._line('@@BEST {"rotate": 0, "flip": true, "event_count": 5000, "contrast_threshold": 0.2, "zoom": 1.0, '
             '"balanced": 0.69, "per_gesture": {"rock": 0.74, "paper": 0.63, "scissors": 0.7}, '
             '"current_balanced": 0.6}')
    assert "mirrored" in ev.transfer_result.text() and "69%" in ev.transfer_result.text()
    assert not ev.apply_btn.isHidden()
    ev._apply_best()
    cfg = window.state.cfg
    assert cfg.cnn.flip is True and cfg.dvs.event_count == 5000 and window.state.dirty


@pytest.mark.skipif(not HAVE_MP, reason="Mediapipe model not downloaded")
def test_both_runs_on_mediapipe_when_dextra_tuned_is_missing(qapp, window):
    window.state.cfg.cnn.tuned_model = "models/not_trained_yet.pth"
    window.tabs.setCurrentIndex(DEBUG)
    play = window.pages[DEBUG]
    select_data(play.detector, "both")                                 # the default
    play._toggle()
    pump(qapp, 3.0)
    assert play.pipeline is not None and play.cnn is None and play.hand is not None
    assert "Mediapipe only" in play.log.toPlainText()
    play._toggle()


def test_recognition_uses_the_four_names_and_greys_out_missing_models(qapp, window):
    window.state.cfg.cnn.tuned_model = "models/not_trained_yet.pth"
    for page in (PLAY, DEBUG):
        window.tabs.setCurrentIndex(page)
        combo = window.pages[page].detector if page == DEBUG else window.pages[page].recognizer
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert [t.split("  (")[0] for t in texts] == list(RECOGNIZERS.values())
        tuned = [combo.itemData(i) for i in range(combo.count())].index("dextra_tuned")
        assert not combo.model().item(tuned).isEnabled() and "Train page" in combo.itemData(tuned, 3)
    debug = window.pages[DEBUG]
    select_data(debug.detector, "mediapipe")
    assert not debug.dextra_card.isEnabled() and debug.mediapipe_card.isEnabled()
    select_data(debug.detector, "both")
    assert debug.dextra_card.isEnabled() and debug.mediapipe_card.isEnabled()


def test_evaluate_all_or_individual_models(qapp, window, tmp_path):
    window.state.cfg.cnn.tuned_model = "models/not_trained_yet.pth"
    window.tabs.setCurrentIndex(EVALUATE)
    ev = window.pages[EVALUATE]
    assert not ev.boxes["dextra_tuned"].isEnabled()                   # its file does not exist
    ev.target.addItem("x", str(tmp_path))
    ev.target.setCurrentIndex(ev.target.count() - 1)
    started = []
    ev.runner.start = lambda args: started.append(args)
    ev.all_box.setChecked(False)
    ev.all_box.setChecked(True)
    ev._run()
    expected = [k for k in RECOGNIZERS if ev.boxes[k].isEnabled()]
    assert started[-1][started[-1].index("--sources") + 1] == ",".join(expected)
    ev.all_box.setChecked(False)
    ev.boxes["mediapipe"].setChecked(True)
    ev._run()
    assert started[-1][started[-1].index("--sources") + 1] == "mediapipe"


def test_bot_tuning_sends_each_command(qapp, window):
    window.tabs.setCurrentIndex(BOT)
    bot = window.pages[BOT]
    assert window.state.cfg.robot.protocol == "rps_text"
    assert bot.pose_btns["N"].isHidden()                              # the team firmware has no ready command
    for pose, command in (("R", "RPS:ROCK"), ("P", "RPS:PAPER"), ("S", "RPS:SCISSORS")):
        bot._send(pose)
        pump(qapp, 0.7)
        assert bot.chip.text() == f"Simulated robot received {command}"
        assert command in bot.log.toPlainText()
    bot._send("test")
    pump(qapp, 2.0)
    assert bot.chip.text() == "Simulated robot received RPS:PAPER"


def test_bot_tuning_moves_each_finger_and_shows_the_confirmation(qapp, window):
    window.tabs.setCurrentIndex(BOT)
    bot = window.pages[BOT]
    assert bot.finger_card.isEnabled()                               # team firmware, simulated robot
    bot.angle_spins[1].setValue(120)
    bot.angle_btns[1].click()
    pump(qapp, 0.6)
    assert bot.tune_chip.text() == "Simulated robot confirmed: CONFIRM_ANGLE:ch=1(Index),deg=120"
    assert "ANGLE:1,120" in bot.log.toPlainText()
    assert bot._mock.angles == {1: 120}
    bot.fold_all.click()
    pump(qapp, 0.8)
    assert bot._mock.angles == {0: 180, 1: 180, 2: 180}
    cfg = window.state.cfg
    assert cfg.robot.finger_angles == [180, 180, 180] and window.state.dirty    # remembered when saved
    window.tabs.setCurrentIndex(SETUP)                               # leaving frees the port for the Play pages
    pump(qapp, 0.2)
    assert bot._mock is None
    cfg.robot.protocol = "ack"                                       # the reference firmware has no ANGLE
    window.tabs.setCurrentIndex(BOT)
    assert not bot.finger_card.isEnabled() and "Team firmware" in bot.tune_hint.text()


def test_settings_are_written_through_and_the_robot_is_only_on_bot_tuning(qapp, window):
    window.tabs.setCurrentIndex(PLAY)
    game = window.pages[PLAY]
    game.rounds10.setChecked(True)
    game.guide.setChecked(True)
    game.pumps.setValue(2)
    cfg = window.state.cfg
    assert cfg.game.rounds == 10 and cfg.decision.mode == "guided" and cfg.decision.pumps_before_shoot == 2
    assert window.state.dirty and window.save_btn.isVisibleTo(window)
    window.state.save()
    game.sound.setCurrentIndex(game.sound.findData("wood"))
    game.lead.setValue(6)
    game.gap.setValue(3)
    game.sound_delay.setValue(200)
    game.beat_volume.setValue(35)
    game.cue_volume.setValue(90)
    window.state.save()
    reopened = AppState(window.state.config_path)
    g = reopened.cfg.game
    assert g.rounds == 10 and reopened.cfg.decision.mode == "guided"
    assert (g.sound, g.lead_beats, g.gap_beats, g.audio_latency_ms) == ("wood", 6, 3, 200.0)
    assert (g.beat_volume, g.cue_volume) == (0.35, 0.9)
    game.rounds_endless.setChecked(True)
    assert cfg.game.rounds == 0
    settings = window.pages[SETTINGS]
    sections = [form.section for _card, form in settings.cards]
    assert "robot" not in sections and "game" not in sections


@pytest.mark.skipif(not HAVE_MP, reason="Mediapipe model not downloaded")
def test_play_match_does_not_start_without_a_camera(qapp, window, tmp_path):
    window.tabs.setCurrentIndex(PLAY)
    window.stop_camera()
    window.default_kind, window.default_video = "video", str(tmp_path / "missing")   # opening fails
    game = window.pages[PLAY]
    select_data(game.recognizer, "mediapipe")
    game.guide.setChecked(True)
    beats = []
    game.beats.beat.connect(lambda b: beats.append(b))
    game._toggle()
    pump(qapp, 3.0)
    assert game.pipeline is None and game.link is None and game.mock is None
    assert not beats and game.schedule is None                        # no beat played to nobody
    assert game.start_btn.isEnabled() and game.start_btn.text() == "Start match"
    assert game.cue.text() == "No camera" and "camera is not running" in game.log.toPlainText()


@pytest.mark.skipif(not HAVE_MP, reason="Mediapipe model not downloaded")
def test_play_match_with_the_beat_guide(qapp, window):
    window.state.cfg.game.beat_bpm = 240                               # a quick match
    window.state.cfg.game.lead_beats, window.state.cfg.game.gap_beats = 1, 1
    window.tabs.setCurrentIndex(PLAY)
    game = window.pages[PLAY]
    select_data(game.recognizer, "mediapipe")
    game.guide.setChecked(True)
    game.rounds5.setChecked(True)
    game.pumps.setValue(1)
    beats, frames = [], []
    game.beats.beat.connect(lambda b: beats.append(b.kind))
    game.beats.beat.connect(lambda b: frames.append(game._prev_frame_t))
    game._toggle()
    pump(qapp, 9.0)                                                    # 1 + 5 x 3 beats of 0.25 s, plus loading
    assert "pump" in beats and "shoot" in beats
    assert frames[0] is not None                                      # the beat began once camera frames arrived
    assert game.pipeline is None                                      # stopped when the beat finished
    assert len(game.results) == 5                                     # every round was scored
    # the engine played the rounds on the beat's clock (no hand in the mock scene: each one a miss), rather than
    # the page filling them in when the beat ended
    assert {0, 1, 2, 3} <= set(game._last_snap.round_results)
    assert all(winner == "you" for winner, *_ in game.results)
    assert game.cue.text() in ("Robot wins", "You win", "Draw")


def test_play_a_round_previews_the_beat_guide_without_a_match(qapp, window):
    window.state.cfg.game.beat_bpm = 240
    window.state.cfg.game.lead_beats = 1
    window.tabs.setCurrentIndex(PLAY)
    game = window.pages[PLAY]
    game.guide.setChecked(True)
    game.pumps.setValue(2)
    cues = []
    game.preview.beat.connect(lambda b: cues.append(game.cue.text()))
    game.preview_btn.click()
    assert game.preview.running and game.preview_btn.text() == "Stop sound" and game.pipeline is None
    pump(qapp, 1.6)                                                   # 1 + 2 + 1 + 1 beats of 0.25 s
    assert cues[1:4] == ["2", "1", "SHOOT"]
    assert not game.preview.running and game.preview_btn.text() == "Play a round"
    game.preview_btn.click()                                          # stops when the page is left
    window.tabs.setCurrentIndex(SETUP)
    assert not game.preview.running


@pytest.mark.skipif(not HAVE_MP, reason="Mediapipe model not downloaded")
def test_endless_match_keeps_the_beat_until_stop(qapp, window):
    g = window.state.cfg.game
    g.beat_bpm, g.lead_beats, g.gap_beats = 240, 1, 2
    window.tabs.setCurrentIndex(PLAY)
    game = window.pages[PLAY]
    select_data(game.recognizer, "mediapipe")
    game.guide.setChecked(True)
    game.rounds_endless.setChecked(True)
    game.pumps.setValue(1)
    game._toggle()
    pump(qapp, 8.0)                                                   # loading, then 4 beats (1 s) per round
    assert game.pipeline is not None and game.beats.running           # past 5 rounds and still going
    assert len(game.results) > 5 and game.m_round.value.text() == str(len(game.results) + 1)
    game.beat_volume.setValue(20)                                     # volumes apply during the match
    assert game.beats.beat_volume == 0.2
    game._toggle()
    assert game.pipeline is None and not game.beats.running
    assert game.cue.text() in ("Robot wins", "You win", "Draw")


def test_bot_tuning_sends_every_click_at_once_from_one_port(qapp, window):
    """Back-to-back commands must all go out, in order, from one socket (as the team's own tool sends
    them). A listener on this computer stands in for the robot."""
    import socket
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 42188))
    rx.settimeout(0.05)
    cfg = window.state.cfg.robot
    cfg.mode, cfg.host, cfg.port = "real", "127.0.0.1", 42188
    window.tabs.setCurrentIndex(BOT)
    bot = window.pages[BOT]
    try:
        for pose in ("S", "R", "P", "R", "S"):                         # faster than any person clicks
            bot.pose_btns[pose].click()
            pump(qapp, 0.05)
        bot.angle_btns[0].click()
        got = []
        end = time.time() + 1.0
        while time.time() < end:
            try:
                data, addr = rx.recvfrom(1024)
                got.append((data.decode(), addr[1]))
            except socket.timeout:
                pump(qapp, 0.01)
    finally:
        window.tabs.setCurrentIndex(SETUP)
        rx.close()
    assert [d for d, _ in got] == ["RPS:SCISSORS", "RPS:ROCK", "RPS:PAPER", "RPS:ROCK", "RPS:SCISSORS", "ANGLE:0,0"]
    assert len({port for _, port in got}) == 1
    assert bot.log.toPlainText().count("RPS:") == 5                    # every command is in the log


def test_play_scores_each_round_by_what_the_robot_plays(qapp, window):
    from rps.hand_tracker import PAPER, ROCK
    window.tabs.setCurrentIndex(PLAY)
    game = window.pages[PLAY]
    cfg = window.state.cfg
    for plays, winner in (("win", "robot"), ("draw", "draw"), ("lose", "you")):
        select_data(game.robot_plays, plays)
        assert cfg.decision.robot_plays == plays
        game.results = []
        game._record(PAPER, "Mediapipe")
        assert game.results[-1][0] == winner, plays
    game._record(None, "-")                                           # nothing read in time: your point
    game._record(ROCK, "Mediapipe")                                    # lose: the robot plays scissors
    assert [w for w, *_ in game.results] == ["you", "you", "you"] and game.results[-1][2] == "S"
    select_data(game.robot_plays, "draw")
    game.results = []
    game._record(ROCK, "Dextra Tuned")
    game._show_score()
    assert game.m_draws.value.text() == "1" and game._final_text() == "Draw"
    assert "draw" in game.result_line.text()


def test_speed_card_shows_where_the_time_went(qapp, window):
    from rps.hand_tracker import PAPER
    from rps.pipeline import DecisionTiming
    from rps.ui.delay_view import throw_delay
    cfg = window.state.cfg
    cfg.latency.camera_latency_ms, cfg.latency.network_ms = 50.0, 0.0
    cfg.latency.servo_transition_ms["R>S"] = 180.0
    window.tabs.setCurrentIndex(DEBUG)
    debug = window.pages[DEBUG]
    timing = DecisionTiming(PAPER, "S", "mp", t_first=10.0, t_frame=10.066, t_sent=10.080, frames=3,
                            dvs_ms=1.5, cnn_ms=0.0, mp_ms=11.0)
    d = throw_delay(timing, "R", cfg)
    assert d.compute_ms == pytest.approx(14.0) and d.to_command_ms == pytest.approx(80.0)
    assert d.hardware_ms == pytest.approx(230.0)                        # camera 50 + hand 180 (Wi-Fi unknown)
    debug.decisions.append((timing, "R"))                             # as the camera thread queues it
    debug.on_frame({"display": np.zeros((48, 64, 3), np.uint8), "running": False})
    card = debug.delay_card
    assert card.headline.text() == "Computed and sent in 14 ms  ·  throw -> command 80 ms"
    assert "2 more camera images at 33 ms each" in card.detail.text() and "wi-fi ?" in card.detail.text()
    assert card.table.rowCount() == 1 and card.table.item(0, 5).text() == "80"
    assert "computing median 14 ms" in card.summary.text()
    same = throw_delay(timing, "S", cfg)                               # the hand already showed it
    assert same.ms("Robot hand moves") == 0.0


def test_wifi_delay_on_the_simulated_robot_is_shown_but_not_kept(qapp, window):
    window.tabs.setCurrentIndex(BOT)
    bot = window.pages[BOT]
    window.state.cfg.latency.network_ms = 0.0
    bot.wifi_btn.click()
    end = time.time() + 6.0
    while time.time() < end and not bot.wifi_btn.isEnabled():
        pump(qapp, 0.05)
    assert "ms one way" in bot.wifi_chip.text() and "not kept" in bot.wifi_hint.text()
    assert window.state.cfg.latency.network_ms == 0.0


def test_the_app_can_open_straight_on_a_page(qapp, window):
    assert window.open_page("play") and window.tabs.currentIndex() == PLAY
    assert window.open_page("play-debug") and window.tabs.currentIndex() == DEBUG
    assert window.open_page("Bot tuning") and window.tabs.currentIndex() == BOT
    assert not window.open_page("nothing like it") and window.tabs.currentIndex() == BOT
