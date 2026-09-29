"""
Play Debug page: everything the recognition and the game rules are doing, for tuning.
  status bar (decision, read by, robot, round, tempo) | camera | delay graph | health
  (Speed under the camera: where each throw's time went)
  right column: Run | Last throws | Dextra | Mediapipe | Orientation and tuning | Log
The robot connection is set on the Bot tuning page.
"""

import os

from PySide6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from rps.cnn import list_models
from rps.decision import GESTURE_NAME
from rps.pipeline import RECOGNIZERS, reader_name
from rps.ui.common import ConfigForm, ProcessRunner, VideoView, select_data
from rps.ui.fields import CHOICES
from rps.ui.play_base import (POSE_GESTURE, Metric, PlayBase, recognizer_combo, refresh_recognizers)
from rps.ui.style import (GESTURE_COLOR, MOTION_COLOR, MUTED, TRACKER_COLOR, Card, Collapsible, button, caption, label,
                          page_header, set_kind, tip)


def model_choices():
    """Every usable Dextra model file in models/, described from its stored details (rps.cnn.list_models)."""
    return list_models("models")


def round_status(snap, pumps_needed: int, beat_label: str = "") -> str:
    if snap.mode == "continuous":
        return "Live" if snap.human is not None else "Waiting for hand"
    if snap.mode == "guided":
        if snap.round < 0:
            return "Get ready"
        if snap.reason == "match over":
            return "Match over"
        if snap.state == "HOLD":
            return "Result" if snap.round_results.get(snap.round) is not None else "No throw seen"
        return beat_label or f"Round {snap.round + 1}"
    if snap.state == "IDLE":
        return "Waiting for hand"
    if snap.state == "ARMED":
        return f"Pump {snap.pumps} of {pumps_needed}"
    if snap.state == "SHOOT":
        return "Throw"
    return "Result"


class PlayTab(PlayBase):
    title = "Play Debug"

    def __init__(self, main):
        super().__init__(main)
        self.runner = ProcessRunner(self)
        self.runner.line.connect(self.log.log)
        self.runner.finished.connect(self._import_done)
        self._start_after_import = False
        self._beat_label = ""
        self.beats.beat.connect(self._on_beat)

        # ---------------- left column: status bar, camera, delay graph, health
        bar = QFrame()
        bar.setObjectName("card")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8)
        bl.setSpacing(28)
        self.m_you = Metric("Decision", "The final answer after the game rules: the gesture you showed. The robot "
                                        "plays against this. The Dextra and Mediapipe cards show raw answers; they "
                                        "can differ from this for a moment.")
        self.m_source = Metric("Read by", "Which reader made the decision. In Both, Dextra Tuned decides while your "
                                          "hand moves and Mediapipe when it is steady (or when it clearly "
                                          "disagrees).")
        self.m_robot = Metric("Robot", "The move sent to the robot hand (Robot plays: to win, draw or lose).")
        self.m_round = Metric("Round", "Countdown: pumps counted, throw, result. Beat guide: the count 3, 2, 1, "
                                       "SHOOT.")
        self.m_tempo = Metric("Tempo", "Your pump speed, learned from your last pumps.")
        for m in (self.m_you, self.m_source, self.m_robot, self.m_round, self.m_tempo):
            bl.addWidget(m)
        bl.addStretch(1)

        self.view = VideoView(placeholder="Camera off")
        self.view.setToolTip("Live camera. The green square is the play zone; only it is analysed. In its top corner: "
                             "Mediapipe's answer (cyan, with its finger points) and Dextra's (violet).")
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(bar)
        left.addWidget(self.view, 1)
        left.addWidget(self.make_delay_card(table=True))
        left.addWidget(self.make_delay_plot(100))
        left.addLayout(self.make_chips())

        # ---------------- Run
        run = Card("Run", "Choose the recognition and the game, then Start. The robot is set on Bot tuning.")
        self.options = QWidget()
        g = QGridLayout(self.options)
        g.setContentsMargins(0, 0, 0, 0)
        g.setColumnStretch(1, 1)
        g.setVerticalSpacing(5)
        self.detector = recognizer_combo()
        self.detector.currentIndexChanged.connect(self._choice_changed)
        self.mode = tip(QComboBox(), "Countdown: pumps are counted from your hand. Beat guide: a drum beat leads each "
                                     "round and you throw on SHOOT. Live: the robot answers whatever it sees.")
        for data, text in CHOICES[("decision", "mode")]:
            self.mode.addItem(text, data)
        self.mode.currentIndexChanged.connect(self._choice_changed)
        self.model_detail = caption("", "Who Dextra Tuned was tuned on, when, and its accuracy on a person it never "
                                        "saw.")
        self.import_btn = button("Download Dextra", tooltip="Downloads Dextra's pretrained model (Dextra Raw) and "
                                                             "converts it for this app (about 1 MB).")
        self.import_btn.clicked.connect(self._import_dextra)
        g.addWidget(label("Recognition", self.detector.toolTip()), 0, 0)
        g.addWidget(self.detector, 0, 1)
        g.addWidget(self.model_detail, 1, 1)
        g.addWidget(label("Game", self.mode.toolTip()), 2, 0)
        g.addWidget(self.mode, 2, 1)
        self.robot_plays = tip(QComboBox(), "To win: the robot shows the move that beats your throw. To draw: the "
                                            "same move as you (in Live mode it mirrors your hand). To lose: the move "
                                            "your throw beats.")
        for data, text in CHOICES[("decision", "robot_plays")]:
            self.robot_plays.addItem(text, data)
        self.robot_plays.currentIndexChanged.connect(self._choice_changed)
        g.addWidget(label("Robot plays", self.robot_plays.toolTip()), 3, 0)
        g.addWidget(self.robot_plays, 3, 1)
        run.body.addWidget(self.options)
        self.summary = caption("", "Current run. Stop to change it.")
        self.summary.setVisible(False)
        run.body.addWidget(self.summary)
        br = QHBoxLayout()
        self.start_btn = button("Start", "primary", "Start or stop.")
        self.start_btn.setMinimumWidth(90)
        self.start_btn.clicked.connect(self._toggle)
        self.dextra_btn = button("Test Dextra Raw", tooltip="Run Dextra Raw alone in Live mode, to see how it "
                                                            "reads your gestures on this camera.")
        self.dextra_btn.clicked.connect(self._test_dextra)
        br.addWidget(self.start_btn)
        br.addWidget(self.dextra_btn)
        br.addWidget(self.import_btn)
        br.addStretch(1)
        run.body.addLayout(br)

        # ---------------- orientation + tuning, log (collapsed)
        adv = QWidget()
        al = QVBoxLayout(adv)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(ConfigForm(self.state, "cnn", keys=["rotate", "flip"]))
        for section, keys in (("dvs", ["contrast_threshold", "event_count"]),
                              ("vote", ["k", "min_confidence"]),
                              ("decision", ["still_frames", "pumps_before_shoot", "pump_min_rise"])):
            form = ConfigForm(self.state, section, keys=keys)
            form.changed.connect(lambda *_: setattr(self, "_rebuild", True))
            al.addWidget(form)
        al.addWidget(caption("Changes apply immediately. Save on the Settings page to keep them."))
        logw = QWidget()
        ll = QVBoxLayout(logw)
        ll.setContentsMargins(0, 0, 0, 0)
        self.counts = caption("", "Decisions made and times the decision changed while you held one gesture.")
        reset = button("Reset counts", tooltip="Set both counts back to zero.")
        reset.clicked.connect(self._reset_counters)
        cr = QHBoxLayout()
        cr.addWidget(self.counts, 1)
        cr.addWidget(reset)
        ll.addLayout(cr)
        ll.addWidget(self.log)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 6, 0)
        pl.addWidget(run)
        pl.addWidget(self.delay_card.table_card)
        pl.addWidget(self.make_dextra_card())
        pl.addWidget(self.make_mediapipe_card())
        pl.addWidget(Collapsible("Orientation and tuning", adv, tooltip="Turn or mirror the Dextra view for Dextra "
                                                                         "Raw (Dextra Tuned keeps its own), and "
                                                                         "adjust sensitivity while playing."))
        pl.addWidget(Collapsible("Log", logw, tooltip="Messages and decision counts."))
        pl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(500)

        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addWidget(scroll)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Play Debug", "Every reading and rule, for tuning. Press Start and play inside "
                                                 "the green square."))
        page.addLayout(body, 1)
        self._reset_display()

    # ------------------------------------------------------------------ choices
    def _choice_changed(self, *_):
        if self.pipeline is None and not self._loading:
            cfg = self.state.cfg
            if self.detector.currentData() and cfg.decision.recognizer != self.detector.currentData():
                cfg.decision.recognizer = self.detector.currentData()
                self.state.mark_dirty()
            if self.mode.currentData() and cfg.decision.mode != self.mode.currentData():
                cfg.decision.mode = self.mode.currentData()
                self.state.mark_dirty()
            if self.robot_plays.currentData() and cfg.decision.robot_plays != self.robot_plays.currentData():
                cfg.decision.robot_plays = self.robot_plays.currentData()
                self.state.mark_dirty()
        self._update_enabled()

    def _update_enabled(self, *_):
        idle = self.pipeline is None and not self._loading
        rec = self.detector.currentData() or "mediapipe"
        if idle:
            self.show_readers(rec)
        tuned = next((m for m in model_choices() if m["path"] == self.state.cfg.cnn.tuned_model), None)
        self.model_detail.setText(f"Dextra Tuned: tuned on {tuned['detail']}" if tuned and tuned["detail"]
                                  and rec in ("dextra_tuned", "both") else "")
        self.model_detail.setVisible(bool(self.model_detail.text()))

    def on_activated(self):
        cfg = self.state.cfg
        if self.pipeline is None and not self._loading:
            refresh_recognizers(self.detector, cfg)
            select_data(self.detector, cfg.decision.recognizer)
            refresh_recognizers(self.detector, cfg)
            select_data(self.mode, cfg.decision.mode)
            select_data(self.robot_plays, cfg.decision.robot_plays)
        self.import_btn.setVisible(not os.path.exists(cfg.cnn.raw_model))
        self._update_enabled()
        super().on_activated()

    def _import_dextra(self):
        if not self.runner.running():
            self.log.log("Downloading Dextra...")
            self.import_btn.setEnabled(False)
            self.runner.start(["tools/import_dextra.py", "--output", self.state.cfg.cnn.raw_model])

    def _import_done(self, code):
        self.import_btn.setEnabled(True)
        self.on_activated()
        if code == 0 and self._start_after_import:
            self._start_after_import = False
            self._test_dextra()
        elif code != 0:
            self.log.log("Download failed; check the internet connection.")

    def _test_dextra(self):
        if self._loading:
            return
        if not os.path.exists(self.state.cfg.cnn.raw_model):
            self._start_after_import = True
            self._import_dextra()
            return
        if self.pipeline is not None:
            self.stop_run()
        select_data(self.detector, "dextra_raw")
        select_data(self.mode, "continuous")
        cfg = self.state.cfg
        self.log.log(f"Dextra view turned {cfg.cnn.rotate} degrees, mirrored: {'yes' if cfg.cnn.flip else 'no'}. "
                     "The Dextra Raw check on the Evaluate page finds the best setting for this camera.")
        self._toggle()

    # ------------------------------------------------------------------ start / stop
    def _toggle(self):
        if self._loading:
            return
        if self.pipeline is not None:
            self.stop_run()
            return
        if self.start_run(self.detector.currentData(), self.mode.currentData()):
            self.start_btn.setText("Loading...")
            self.start_btn.setEnabled(False)
            self.dextra_btn.setEnabled(False)
            self._update_enabled()

    def on_run_started(self):
        self.start_btn.setEnabled(True)
        self.dextra_btn.setEnabled(True)
        self.start_btn.setText("Stop")
        set_kind(self.start_btn, "danger")
        self._show_options(False)

    def on_run_failed(self, reason: str = ""):
        self.start_btn.setEnabled(True)
        self.dextra_btn.setEnabled(True)
        self.start_btn.setText("Start")
        self._update_enabled()

    def on_run_stopped(self):
        self.start_btn.setText("Start")
        set_kind(self.start_btn, "primary")
        self._show_options(True)
        self._update_enabled()
        self._reset_display()

    def _show_options(self, show: bool):
        """While playing, the Run card shrinks to one summary line so the readings stay in view."""
        self.options.setVisible(show)
        self.dextra_btn.setVisible(show)
        self.summary.setVisible(not show)
        if not show:
            robot = {"real": "robot", "simulated": "simulated robot", "off": "robot off"}[self.state.cfg.robot.mode]
            game = self.mode.currentText().split(" (")[0]
            plays = self.robot_plays.currentText().split(" (")[0].lower()
            self.summary.setText(f"{RECOGNIZERS[self.running_recognizer]} · {game} · {robot} plays {plays}")

    def _reset_display(self):
        self._beat_label = ""
        self.m_you.set("-")
        self.m_source.set("-")
        self.m_robot.set("-")
        self.m_round.set("Stopped")
        self.m_tempo.set("-")

    def _reset_counters(self):
        if self.pipeline is not None:
            self.pipeline.engine.switches = self.pipeline.engine.commits = 0

    def _on_beat(self, beat):
        self._beat_label = beat.label

    # ------------------------------------------------------------------ UI thread
    def on_frame_ui(self, p, now):
        s = p["snap"]
        human = GESTURE_NAME.get(s.human) if s.human is not None else None
        robot = POSE_GESTURE.get(s.pose, "ready")
        self.m_you.set(human.upper() if human else "-", GESTURE_COLOR.get(human, MUTED))
        who = reader_name(s.source, self.cnn)
        if human and who:
            self.m_source.set(who, MOTION_COLOR if s.source == "cnn" else TRACKER_COLOR)
        else:
            self.m_source.set("-")
        self.m_robot.set(robot.upper(), GESTURE_COLOR.get(robot, MUTED))
        self.m_round.set(round_status(s, self.state.cfg.decision.pumps_before_shoot, self._beat_label), "#e4e7eb")
        self.m_tempo.set(f"{s.tempo:.2f} s/pump" if s.tempo else ("learning" if s.mode != "continuous" else "-"),
                         "#e4e7eb" if s.tempo else MUTED)
        self.counts.setText(f"Decisions {s.commits} · changes {s.switches}")

    def shutdown(self):
        super().shutdown()
        self.runner.kill()
