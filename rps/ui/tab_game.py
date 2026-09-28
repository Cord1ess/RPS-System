"""
Play page: the demo. A match of 5 or 10 rounds (or endless, until Stop) against the robot, with a
scoreboard, a big count and, optionally, the beat guide: a steady beat that plays all the time,
louder hits on the pumps (shown as 3, 2, 1) and a double hit on SHOOT, when you throw. Its sound,
tempo, both volumes, count-in, beats between rounds and sound delay are set here; "Play a round"
previews it without the camera, and the volumes can be changed during a match. The camera, the
delay graph and the Dextra view stay visible; everything else is on Play Debug.

Scoring: the robot answers every throw it reads with the winning move, so it scores the round.
A round where no throw was read in time goes to you.
"""

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QRadioButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget)

from rps.decision import GESTURE_NAME
from rps.game import ENDLESS_ROUNDS, BeatSchedule
from rps.pipeline import RECOGNIZERS, reader_name
from rps.ui.common import VideoView, select_data
from rps.ui.play_base import POSE_GESTURE, Metric, PlayBase, recognizer_combo, refresh_recognizers
from rps.ui.sound import SOUNDS, BeatPlayer
from rps.ui.style import (GESTURE_COLOR, MUTED, OK, WARN, Card, Collapsible, button, caption, label, page_header,
                          set_kind, tip)

CUE_STYLE = "font-size:40pt; font-weight:800; color:{};"
YOU, ROBOT = "you", "robot"


def percent_slider(tooltip: str):
    """A 0-100 % slider with its value shown next to it: (slider, row layout)."""
    slider = tip(QSlider(Qt.Orientation.Horizontal), tooltip)
    slider.setRange(0, 100)
    value = caption("")
    value.setMinimumWidth(34)
    slider.valueChanged.connect(lambda v: value.setText(f"{v}%"))
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(slider, 1)
    lay.addWidget(value)
    return slider, lay


class GameTab(PlayBase):
    title = "7  Play"

    def __init__(self, main):
        super().__init__(main)
        self.results = []                  # per round: (winner, your gesture or None, robot pose)
        self._seen_commits = self._seen_misses = 0
        self._last_round = -1
        self._last_snap = None
        self.beats.beat.connect(self._on_beat)
        self.beats.finished.connect(self._beats_done)
        self.preview = BeatPlayer(self)                 # "Play a round": the beat guide without a match
        self.preview.beat.connect(self._on_preview_beat)
        self.preview.finished.connect(self._preview_done)

        # ---------------- scoreboard
        board = QFrame()
        board.setObjectName("card")
        bl = QHBoxLayout(board)
        bl.setContentsMargins(18, 10, 18, 10)
        bl.setSpacing(34)
        self.m_robot_score = Metric("Robot", "Rounds the robot won: it read your throw and answered with the winning "
                                             "move.", big=True)
        self.m_you_score = Metric("You", "Rounds you won: the robot did not read a throw in time.", big=True)
        self.m_round = Metric("Round", "Round being played, of the match length (endless: until Stop).", big=True)
        for m in (self.m_robot_score, self.m_you_score, self.m_round):
            bl.addWidget(m)
        bl.addStretch(1)
        self.cue = QLabel("Press Start")
        self.cue.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.cue.setStyleSheet(CUE_STYLE.format("#e4e7eb"))
        tip(self.cue, "What to do now. With the beat guide: pump on 3, 2, 1 and throw on SHOOT.")
        bl.addWidget(self.cue)
        self.result_line = caption("", "The last round: what you threw, what the robot played, and who read it.")

        self.view = VideoView(placeholder="Camera off")
        self.view.setToolTip("Play inside the green square. In its corner: Mediapipe's answer (cyan) and Dextra's "
                             "(violet).")
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(board)
        left.addWidget(self.result_line)
        left.addWidget(self.view, 1)
        left.addWidget(self.make_delay_plot(110))
        left.addLayout(self.make_chips())

        # ---------------- match options
        match = Card("Match", "Choose, then Start. Everything here is remembered when you save the settings.")
        self.options = QWidget()
        g = QGridLayout(self.options)
        g.setContentsMargins(0, 0, 0, 0)
        g.setColumnStretch(1, 1)
        g.setVerticalSpacing(6)
        self.recognizer = recognizer_combo()
        self.recognizer.currentIndexChanged.connect(self._options_changed)
        g.addWidget(label("Recognition", self.recognizer.toolTip()), 0, 0)
        g.addWidget(self.recognizer, 0, 1)
        rounds = QHBoxLayout()
        self.rounds5 = tip(QRadioButton("5"), "A match of 5 rounds.")
        self.rounds10 = tip(QRadioButton("10"), "A match of 10 rounds.")
        self.rounds_endless = tip(QRadioButton("Endless"), "Rounds keep coming, with the beat, until you press Stop.")
        self._rounds_group = QButtonGroup(self)
        for b in (self.rounds5, self.rounds10, self.rounds_endless):
            self._rounds_group.addButton(b)
            rounds.addWidget(b)
            b.toggled.connect(self._options_changed)
        rounds.addStretch(1)
        g.addWidget(label("Rounds", "Match length."), 1, 0)
        g.addLayout(rounds, 1, 1)
        self.pumps = tip(QSpinBox(), "Pumps before the throw. With the beat guide: the loud hits before SHOOT.")
        self.pumps.setRange(1, 4)
        self.pumps.valueChanged.connect(self._options_changed)
        g.addWidget(label("Pumps", self.pumps.toolTip()), 2, 0)
        g.addWidget(self.pumps, 2, 1)
        self.guide = tip(QCheckBox("Beat guide"), "A beat leads the game: the steady beat keeps the tempo, louder "
                                                 "hits are your pumps (3, 2, 1) and a double hit is SHOOT: throw. "
                                                 "Off: pump at your own pace; pumps are counted from your hand.")
        self.guide.toggled.connect(self._options_changed)
        g.addWidget(self.guide, 3, 0, 1, 2)
        self.guide_box = QWidget()
        gg = QGridLayout(self.guide_box)
        gg.setContentsMargins(0, 0, 0, 0)
        gg.setColumnStretch(1, 1)
        self.sound = tip(QComboBox(), "The beat's sound. All three are made to be heard on laptop speakers; the wood "
                                      "block and the beep cut through a noisy room.")
        for key, name in SOUNDS.items():
            self.sound.addItem(name, key)
        self.sound.currentIndexChanged.connect(self._options_changed)
        self.bpm = tip(QDoubleSpinBox(), "Beats per minute. 150 = one beat every 0.4 s, a natural pump pace.")
        self.bpm.setRange(60, 240)
        self.bpm.setDecimals(0)
        self.bpm.setSingleStep(5)
        self.bpm.setSuffix(" bpm")
        self.bpm.valueChanged.connect(self._options_changed)
        self.lead = tip(QSpinBox(), "Steady beats before the first round, to find the tempo.")
        self.lead.setRange(1, 16)
        self.lead.setSuffix(" beats")
        self.lead.valueChanged.connect(self._options_changed)
        self.gap = tip(QSpinBox(), "Steady beats after SHOOT while the result shows, before the next count.")
        self.gap.setRange(2, 16)
        self.gap.setSuffix(" beats")
        self.gap.valueChanged.connect(self._options_changed)
        self.sound_delay = tip(QSpinBox(), "Time from the app playing a beat to you hearing it: about 40 ms on laptop "
                                           "speakers, 150-250 ms on Bluetooth. The throw window moves by this much, "
                                           "so throwing on the SHOOT you hear counts.")
        self.sound_delay.setRange(0, 500)
        self.sound_delay.setSingleStep(10)
        self.sound_delay.setSuffix(" ms")
        self.sound_delay.valueChanged.connect(self._options_changed)
        for r, (name, w) in enumerate((("Sound", self.sound), ("Tempo", self.bpm), ("Count-in", self.lead),
                                       ("Between rounds", self.gap), ("Sound delay", self.sound_delay))):
            gg.addWidget(label(name, w.toolTip()), r, 0)
            gg.addWidget(w, r, 1)
        self.preview_btn = button("Play a round", tooltip="Hear the beat guide once (count-in, the count and SHOOT) "
                                                          "with these settings. No camera or robot needed.")
        self.preview_btn.clicked.connect(self._preview_toggle)
        gg.addWidget(self.preview_btn, 5, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        g.addWidget(self.guide_box, 4, 0, 1, 2)
        match.body.addWidget(self.options)
        # the volumes stay here during a match, so they can be set while playing
        self.volume_box = QWidget()
        vg = QGridLayout(self.volume_box)
        vg.setContentsMargins(0, 0, 0, 0)
        vg.setColumnStretch(1, 1)
        self.beat_volume, beat_row = percent_slider("Volume of the steady beat that plays all the time.")
        self.cue_volume, cue_row = percent_slider("Volume of the count (3, 2, 1) and SHOOT. Keep it above the steady "
                                                  "beat, so the cues stand out.")
        for r, (name, slider, lay) in enumerate((("Steady beat", self.beat_volume, beat_row),
                                                 ("Count and SHOOT", self.cue_volume, cue_row))):
            slider.valueChanged.connect(self._volume_changed)
            vg.addWidget(label(name, slider.toolTip()), r, 0)
            vg.addLayout(lay, r, 1)
        match.body.addWidget(self.volume_box)
        self.summary = caption("", "The running match. Stop to change it.")
        self.summary.setVisible(False)
        match.body.addWidget(self.summary)
        self.start_btn = button("Start match", "primary", "Start or stop the match.")
        self.start_btn.setMinimumWidth(110)
        self.start_btn.clicked.connect(self._toggle)
        br = QHBoxLayout()
        br.addWidget(self.start_btn)
        br.addStretch(1)
        match.body.addLayout(br)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 6, 0)
        pl.addWidget(match)
        pl.addWidget(self.make_dextra_card(preview=160, references=False))
        pl.addWidget(Collapsible("Messages", self.log, tooltip="Notes from starting the match."))
        pl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(440)

        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addWidget(scroll)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Play", "A match against the robot. Play inside the green square."))
        page.addLayout(body, 1)
        self._loading_options = False
        self._show_score()

    # ------------------------------------------------------------------ options (written to the settings)
    def on_activated(self):
        cfg = self.state.cfg
        if self.pipeline is None and not self._loading:
            self._loading_options = True
            refresh_recognizers(self.recognizer, cfg)
            select_data(self.recognizer, cfg.decision.recognizer)
            refresh_recognizers(self.recognizer, cfg)
            g = cfg.game
            length = self.rounds_endless if g.rounds <= 0 else self.rounds10 if g.rounds >= 10 else self.rounds5
            length.setChecked(True)
            self.pumps.setValue(cfg.decision.pumps_before_shoot)
            self.guide.setChecked(cfg.decision.mode == "guided")
            select_data(self.sound, g.sound)
            self.bpm.setValue(g.beat_bpm)
            self.lead.setValue(g.lead_beats)
            self.gap.setValue(g.gap_beats)
            self.sound_delay.setValue(int(round(g.audio_latency_ms)))
            self.beat_volume.setValue(int(round(g.beat_volume * 100)))
            self.cue_volume.setValue(int(round(g.cue_volume * 100)))
            self._loading_options = False
            self._options_changed()
        super().on_activated()

    def on_deactivated(self):
        self._preview_stop()
        super().on_deactivated()

    def shutdown(self):
        self._preview_stop()
        super().shutdown()

    def _write(self, new: dict):
        """Writes {(section, key): value} into the settings; marks them unsaved if anything changed."""
        cfg = self.state.cfg
        changed = False
        for (section, key), value in new.items():
            if value is not None and getattr(getattr(cfg, section), key) != value:
                setattr(getattr(cfg, section), key, value)
                changed = True
        if changed:
            self.state.mark_dirty()

    def _options_changed(self, *_):
        guided = self.guide.isChecked()
        self.guide_box.setVisible(guided)
        self.volume_box.setVisible(guided)
        self.show_readers(self.recognizer.currentData() or "mediapipe")
        if self._loading_options or self.pipeline is not None or self._loading:
            return
        self._write({("decision", "recognizer"): self.recognizer.currentData(),
                     ("game", "rounds"): self.rounds(),
                     ("decision", "pumps_before_shoot"): self.pumps.value(),
                     ("decision", "mode"): "guided" if guided else "countdown",
                     ("game", "sound"): self.sound.currentData(),
                     ("game", "beat_bpm"): float(self.bpm.value()),
                     ("game", "lead_beats"): self.lead.value(),
                     ("game", "gap_beats"): self.gap.value(),
                     ("game", "audio_latency_ms"): float(self.sound_delay.value())})
        self._show_score()

    def _volume_changed(self, *_):
        """Volumes apply at once, also during a match or a preview."""
        beat, cue = self.beat_volume.value() / 100.0, self.cue_volume.value() / 100.0
        self.beats.set_volumes(beat, cue)
        self.preview.set_volumes(beat, cue)
        if not self._loading_options:
            self._write({("game", "beat_volume"): beat, ("game", "cue_volume"): cue})

    # ------------------------------------------------------------------ preview ("Play a round")
    def _preview_toggle(self):
        if self.preview.running:
            self._preview_stop()
            return
        if self.pipeline is not None or self._loading:
            return
        self._options_changed()
        g = self.state.cfg.game
        schedule = BeatSchedule(time.perf_counter() + 0.15, 60.0 / g.beat_bpm, self.pumps.value(), 1,
                                g.lead_beats, 1)
        self.preview.start(schedule, g.sound, self.beat_volume.value() / 100.0, self.cue_volume.value() / 100.0)
        self.preview_btn.setText("Stop sound")
        self._set_cue("Get ready", MUTED)

    def _on_preview_beat(self, beat):
        if beat.kind == "pump":
            self._set_cue(beat.label, "#e4e7eb")
        elif beat.kind == "shoot":
            self._set_cue("SHOOT", OK)

    def _preview_stop(self):
        if self.preview.running:
            self.preview.stop()
            self._preview_done()

    def _preview_done(self):
        self.preview_btn.setText("Play a round")
        if self.pipeline is None and not self._loading:
            self._set_cue("Press Start", "#e4e7eb")

    # ------------------------------------------------------------------ match
    def _toggle(self):
        if self._loading:
            return
        if self.pipeline is not None:
            self.stop_run()
            return
        self._options_changed()
        self._preview_stop()
        mode = "guided" if self.guide.isChecked() else "countdown"
        if self.start_run(self.recognizer.currentData(), mode):
            self.results, self._seen_commits, self._seen_misses, self._last_round = [], 0, 0, -1
            self.start_btn.setText("Loading...")
            self.start_btn.setEnabled(False)
            self._set_cue("Loading", MUTED)

    def on_run_started(self):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Stop")
        set_kind(self.start_btn, "danger")
        self.options.setVisible(False)
        self.summary.setVisible(True)
        guide = "beat guide" if self.guided else "own pace"
        length = f"{self.rounds()} rounds" if self.rounds() else "endless"
        self.summary.setText(f"{RECOGNIZERS[self.recognizer_key()]} · {length} · "
                             f"{self.state.cfg.decision.pumps_before_shoot} pumps · {guide}")
        self.result_line.setText("")
        self._set_cue("Waiting for camera" if self.guided else "Pump!", "#e4e7eb" if not self.guided else MUTED)
        self._show_score()

    def recognizer_key(self) -> str:
        return self.recognizer.currentData() or "mediapipe"

    def rounds(self) -> int:
        """Match length; 0 = endless."""
        return 0 if self.rounds_endless.isChecked() else 10 if self.rounds10.isChecked() else 5

    def _limit(self) -> int:
        return self.rounds() or ENDLESS_ROUNDS

    def on_run_failed(self, reason: str = ""):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start match")
        self._set_cue(reason or "Press Start", WARN if reason else "#e4e7eb")

    def on_run_stopped(self):
        self.start_btn.setText("Start match")
        set_kind(self.start_btn, "primary")
        self.options.setVisible(True)
        self.summary.setVisible(False)
        done = bool(self.results) and len(self.results) >= (self.rounds() or 1)     # endless: stopped after a round
        self._set_cue(self._final_text() if done else "Press Start", OK if done else "#e4e7eb")
        self.show_readers(self.recognizer_key())

    def _final_text(self) -> str:
        robot, you = self._score()
        return "Robot wins" if robot > you else ("You win" if you > robot else "Draw")

    def _score(self):
        return sum(1 for w, *_ in self.results if w == ROBOT), sum(1 for w, *_ in self.results if w == YOU)

    def _show_score(self):
        robot, you = self._score()
        self.m_robot_score.set(str(robot), "#e4e7eb")
        self.m_you_score.set(str(you), "#e4e7eb")
        current = str(min(len(self.results) + 1, self._limit())) if self.pipeline is not None else "-"
        self.m_round.set(f"{current} / {self.rounds()}" if self.rounds() else current, "#e4e7eb")

    def _set_cue(self, text: str, color: str):
        if self.cue.text() != text:
            self.cue.setText(text)
        self.cue.setStyleSheet(CUE_STYLE.format(color))

    def _record(self, winner: str, gesture, pose: str, source: str):
        self.results.append((winner, gesture, pose))
        if winner == ROBOT:
            you = GESTURE_NAME.get(gesture, "?")
            robot = POSE_GESTURE.get(pose, "?")
            self._set_cue(f"{robot.upper()}", GESTURE_COLOR.get(robot, "#e4e7eb"))
            self.result_line.setText(f"Round {len(self.results)}: you threw {you}, the robot played {robot} "
                                     f"(read by {source}).")
        else:
            self._set_cue("Missed", WARN)
            self.result_line.setText(f"Round {len(self.results)}: no throw was read in time. Your point.")
        self._show_score()
        if len(self.results) >= self._limit() and not self.guided:
            self.stop_run()                          # own pace: the match ends with its last round

    # ------------------------------------------------------------------ beat guide
    def on_beat_started(self):
        self._set_cue("Get ready", MUTED)

    def _on_beat(self, beat):
        if self.pipeline is None:
            return
        if beat.kind == "pump":
            self._set_cue(beat.label, "#e4e7eb")
        elif beat.kind == "shoot":
            self._set_cue("SHOOT", OK)
        elif beat.round < 0:
            self._set_cue("Get ready", MUTED)

    def _beats_done(self):
        if self.pipeline is not None:
            self._finish_rounds(self._last_snap)
            self.stop_run()

    def _finish_rounds(self, s, upto=None):
        """Scores every round up to `upto` (default: the whole match) that has not been scored yet."""
        if s is None:
            return
        end = self._limit() if upto is None else min(upto, self._limit())
        for n in range(len(self.results), end):
            g = (s.round_results or {}).get(n)
            self._record(ROBOT if g is not None else YOU, g, s.pose, reader_name(s.source, self.cnn) or "-")

    # ------------------------------------------------------------------ results, from the decision engine
    def on_frame_ui(self, p, now):
        s = p["snap"]
        self._last_snap = s
        source = reader_name(s.source, self.cnn) or "-"
        if s.mode == "guided":
            self._finish_rounds(s, upto=s.round)                     # rounds that ended since the last frame
            if 0 <= s.round < self._limit() and len(self.results) == s.round \
                    and s.round in s.round_results:                  # decided or missed: show it at once
                g = s.round_results[s.round]
                self._record(ROBOT if g is not None else YOU, g, s.pose, source)
            return
        while self._seen_commits < s.commits:
            self._seen_commits += 1
            self._record(ROBOT, s.human, s.pose, source)
        while self._seen_misses < s.misses:
            self._seen_misses += 1
            self._record(YOU, None, s.pose, source)
        if self.pipeline is not None and s.state == "ARMED":
            self._set_cue(f"Pump {s.pumps}" if s.pumps else "Pump!", "#e4e7eb")
        elif self.pipeline is not None and s.state == "SHOOT":
            self._set_cue("Throw!", OK)
