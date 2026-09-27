"""
Play page. Every value appears once:
  status bar (you, robot, round, tempo) | camera | delay graph | health
  right column: Run | Readings | Orientation and tuning | Log
"""

import glob
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QScrollArea, QVBoxLayout, QWidget)

from model import CLASS_NAMES
from rps.decision import GESTURE_NAME
from rps.hud import draw_hand, draw_roi
from rps.pipeline import Pipeline, load_models
from rps.robot_link import ESP_RST_BROWNOUT, MockEsp, RobotLink
from rps.timing import LatencyLog
from rps.ui.base import Tab
from rps.ui.common import ConfigForm, LogView, ProcessRunner, VideoView, select_data, to_pixmap
from rps.ui.fields import CHOICES
from rps.ui.style import (BORDER, GESTURE_COLOR, MUTED, PANEL, Card, Chip, Collapsible, button, caption, label,
                          page_header, set_kind, static_plot, tip)

CLASS_SHORT = [c.split("_", 1)[1] for c in CLASS_NAMES]
BAR_NAMES = ["Rock", "Paper", "Scissors", "None"]
POSE_GESTURE = {"R": "rock", "P": "paper", "S": "scissors", "N": "ready"}
SOURCE_NAME = {"cnn": "motion model", "mp": "hand tracker"}
DEXTRA_MODEL = "models/dextra_roshambo.pth"
DEXTRA_SAMPLES = "models/dextra/sample_frames"
UI_PERIOD_S, PLOT_PERIOD_S = 0.066, 0.1


def colorize(frame64: np.ndarray, size: int) -> np.ndarray:
    return cv2.applyColorMap(cv2.resize(frame64, (size, size), interpolation=cv2.INTER_NEAREST),
                             cv2.COLORMAP_INFERNO)


def model_choices():
    """(path, name) for every motion model file in models/."""
    out = []
    for path in sorted(glob.glob("models/*.pth")):
        name = os.path.basename(path)
        path = path.replace("\\", "/")
        if name == "dextra_roshambo.pth":
            out.insert(0, (path, "Dextra model (pretrained)"))
        else:
            out.append((path, f"Your model ({name})"))
    return out


def round_status(snap, pumps_needed: int) -> str:
    if snap.mode == "countdown":
        if snap.state == "IDLE":
            return "Waiting for hand"
        if snap.state == "ARMED":
            return f"Pump {snap.pumps} of {pumps_needed}"
        if snap.state == "SHOOT":
            return "Throw"
        return "Result"
    return "Live" if snap.human is not None else "Waiting for hand"


class Metric(QWidget):
    """Small caption over a value, e.g. 'You' / 'ROCK'. Restyles only on change."""

    def __init__(self, title: str, tooltip: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        t = QLabel(title)
        t.setObjectName("metricLabel")
        self.value = QLabel("-")
        self.value.setObjectName("metric")
        lay.addWidget(t)
        lay.addWidget(self.value)
        self.setToolTip(tooltip)
        self._text, self._color = None, None

    def set(self, text: str, color: str = MUTED):
        if text != self._text:
            self._text = text
            self.value.setText(text)
        if color != self._color:
            self._color = color
            self.value.setStyleSheet(f"color:{color};")


class PlayTab(Tab):
    title = "6  Play"
    uses_camera = True
    models_ready = Signal(object)

    def __init__(self, main):
        super().__init__(main)
        self.pipeline = None
        self.cnn = self.hand = self.link = self.mock = None
        self._lock = threading.Lock()
        self.latency = LatencyLog()
        self._rebuild = False
        self._loading = False
        self._start_after_import = False
        self._last_cnn_t = None
        self._cnn_times = deque(maxlen=60)
        self._last_ui = self._last_plot = 0.0
        self._y_max = 60.0
        self._closing = False
        self._prev_frame_t = None
        self._probs = None
        self.runner = ProcessRunner(self)
        self.log = LogView(500)
        self.runner.line.connect(self.log.log)
        self.runner.finished.connect(self._import_done)
        self.models_ready.connect(self._models_loaded)
        main.worker.state_changed.connect(self._camera_state)

        # ---------------- left column: status bar, camera, delay graph, health
        bar = QFrame()
        bar.setObjectName("card")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8)
        bl.setSpacing(28)
        self.m_you = Metric("You", "The gesture the system decided you showed.")
        self.m_robot = Metric("Robot", "The move sent to the robot hand (the one that beats yours).")
        self.m_round = Metric("Round", "Countdown progress: pumps counted, throw, result.")
        self.m_tempo = Metric("Tempo", "Your pump speed, learned from your last pumps. The throw is expected "
                                       "one beat after the last pump.")
        for m in (self.m_you, self.m_robot, self.m_round, self.m_tempo):
            bl.addWidget(m)
        bl.addStretch(1)

        self.view = VideoView(placeholder="Camera off")
        self.view.setToolTip("Live camera. The green square is the play zone; only it is analysed. Blue dots are "
                             "finger points from the hand tracker.")

        self.plot = static_plot(pg.PlotWidget())
        self.plot.setBackground(PANEL)
        self.plot.setFixedHeight(130)
        self.plot.setToolTip("Delay per camera frame over the last 5 seconds. Processing: time from a camera "
                             "frame arriving to the decision. Camera interval: time between camera frames "
                             "(33 ms = 30 fps).")
        self.plot.setLabel("left", "ms")
        self.plot.setYRange(0, 60)
        self.plot.showGrid(y=True, alpha=0.15)
        self.plot.addLegend(offset=(-8, 2), labelTextSize="8pt", colCount=4)
        self.curves = {k: self.plot.plot(pen=pg.mkPen(c, width=1.6), name=k) for k, c in
                       (("Processing", "#e0524a"), ("Motion model", "#e879b0"), ("Hand tracker", "#5b9cf0"),
                        ("Camera interval", "#9aa3ad"))}
        self.series = {k: deque(maxlen=150) for k in self.curves}
        self.delay_text = caption("", "Averages over the last 100 frames.")
        self.delay_text.setWordWrap(False)

        self.chip_cam = Chip("Camera off", "off", "Camera frames per second. 27 or more is good.")
        self.chip_proc = Chip("Processing -", "off", "Average processing time per frame. Under 25 ms is good.")
        self.chip_robot = Chip("Robot off", "off", "Connection to the robot hand and its reply time.")
        health = QHBoxLayout()
        for c in (self.chip_cam, self.chip_proc, self.chip_robot):
            health.addWidget(c)
        health.addStretch(1)
        health.addWidget(self.delay_text)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(bar)
        left.addWidget(self.view, 1)
        left.addWidget(self.plot)
        left.addLayout(health)

        # ---------------- Run
        run = Card("Run", "Choose what reads your hand and how the game works, then Start.")
        self.options = QWidget()
        g = QGridLayout(self.options)
        g.setContentsMargins(0, 0, 0, 0)
        g.setColumnStretch(1, 1)
        g.setVerticalSpacing(5)
        self.detector = tip(QComboBox(), "Motion model: reads movement (Dextra's method); keep the hand moving. "
                                         "Hand tracker: reads finger positions; works when still. Both: motion "
                                         "model while moving, tracker when still.")
        for data, text in CHOICES[("decision", "source")]:
            self.detector.addItem(text, data)
        self.detector.currentIndexChanged.connect(self._update_enabled)
        self.mode = tip(QComboBox(), "Countdown: pump 3 times, then throw; one decision per throw. Live: the robot "
                                     "answers whatever it sees, continuously.")
        for data, text in CHOICES[("decision", "mode")]:
            self.mode.addItem(text, data)
        self.model = tip(QComboBox(), "Which motion model file to use.")
        self.import_btn = button("Download Dextra model", tooltip="Downloads Dextra's pretrained weights and "
                                                                   "converts them for this app (about 1 MB).")
        self.import_btn.clicked.connect(self._import_dextra)
        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        mrow.addWidget(self.model, 1)
        mrow.addWidget(self.import_btn)
        g.addWidget(label("Recognition", self.detector.toolTip()), 0, 0)
        g.addWidget(self.detector, 0, 1)
        g.addWidget(label("Game", self.mode.toolTip()), 1, 0)
        g.addWidget(self.mode, 1, 1)
        g.addWidget(label("Motion model", self.model.toolTip()), 2, 0)
        g.addLayout(mrow, 2, 1)
        self.robot_on = tip(QCheckBox("Send moves to robot"), "Send each decision to the robot hand over Wi-Fi.")
        self.robot_on.setChecked(self.state.cfg.robot.enabled)
        self.robot_on.toggled.connect(lambda on: (setattr(self.state.cfg.robot, "enabled", bool(on)),
                                                  self.state.mark_dirty()))
        self.mock_esp = tip(QCheckBox("Simulated robot"), "Use a simulated robot on this computer instead of the "
                                                          "real hand (for testing without hardware).")
        cb = QHBoxLayout()
        cb.addWidget(self.robot_on)
        cb.addWidget(self.mock_esp)
        cb.addStretch(1)
        g.addLayout(cb, 3, 0, 1, 2)
        run.body.addWidget(self.options)
        self.summary = caption("", "Current run settings. Stop to change them.")
        self.summary.setVisible(False)
        run.body.addWidget(self.summary)
        br = QHBoxLayout()
        self.start_btn = button("Start", "primary", "Start or stop the game.")
        self.start_btn.setMinimumWidth(90)
        self.start_btn.clicked.connect(self._toggle)
        self.dextra_btn = button("Test Dextra model", tooltip="Run Dextra's pretrained motion model alone in Live "
                                                               "mode, to see whether it recognises your gestures on "
                                                               "this camera without any training.")
        self.dextra_btn.clicked.connect(self._test_dextra)
        br.addWidget(self.start_btn)
        br.addWidget(self.dextra_btn)
        br.addStretch(1)
        run.body.addLayout(br)

        # ---------------- Readings
        see = Card("Readings", "Raw readings before the game rules. The robot moves only when they agree for a "
                               "few frames.")
        imgs = QHBoxLayout()
        self.cnn_input = QLabel()
        self.cnn_input.setFixedSize(144, 144)
        self.cnn_input.setStyleSheet(f"background:#000; border:1px solid {BORDER};")
        tip(self.cnn_input, "Motion image: the movement in the play zone that the motion model reads. It is "
                            "empty while the hand is still.")
        imgs.addWidget(self.cnn_input)
        self.ref_box = QWidget()
        tip(self.ref_box, "Dextra's own motion images for each gesture. Yours should look similar; if not, try "
                          "another rotation under Orientation and tuning.")
        rg = QGridLayout(self.ref_box)
        rg.setContentsMargins(0, 0, 0, 0)
        rg.setSpacing(3)
        self.ref_labels = []
        for i, name in enumerate(BAR_NAMES):
            img = QLabel()
            img.setFixedSize(56, 56)
            img.setStyleSheet("background:#000;")
            rg.addWidget(img, (i // 2) * 2, i % 2)
            rg.addWidget(caption(name), (i // 2) * 2 + 1, i % 2)
            self.ref_labels.append(img)
        imgs.addWidget(self.ref_box)
        imgs.addStretch(1)
        self.ref_box.setVisible(False)
        see.body.addLayout(imgs)

        bars = QGridLayout()
        bars.setVerticalSpacing(3)
        self.bars = []
        for i, (name, key) in enumerate(zip(BAR_NAMES, CLASS_SHORT)):
            b = QProgressBar()
            b.setRange(0, 100)
            b.setFormat("%p%")
            color = GESTURE_COLOR.get(key, GESTURE_COLOR["none"])
            b.setStyleSheet(f"QProgressBar::chunk {{ background:{color}; border-radius:2px; }}")
            b.setToolTip(f"Motion model confidence that the movement is {name.lower()}.")
            bars.addWidget(label(name, b.toolTip()), i, 0)
            bars.addWidget(b, i, 1)
            self.bars.append(b)
        see.body.addLayout(bars)
        self.cnn_status = caption("Not running", "When the motion model last answered and how often.")
        see.body.addWidget(self.cnn_status)
        rows = QGridLayout()
        rows.setColumnStretch(1, 1)
        self.mp_value = label("off", "Hand tracker reading and its confidence.")
        self.decision_value = label("-", "The current decision and which reader made it.")
        for r, (name, w) in enumerate((("Hand tracker", self.mp_value), ("Decision", self.decision_value))):
            rows.addWidget(label(name, w.toolTip()), r, 0)
            rows.addWidget(w, r, 1)
        see.body.addLayout(rows)

        # ---------------- orientation + tuning, log (collapsed)
        adv = QWidget()
        al = QVBoxLayout(adv)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(ConfigForm(self.state, "cnn", keys=["rotate", "flip"]))
        for section, keys in (("dvs", ["contrast_threshold", "event_count"]),
                              ("vote", ["k", "min_confidence"]),
                              ("decision", ["still_frames", "pumps_before_shoot"])):
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
        pl.addWidget(see)
        pl.addWidget(Collapsible("Orientation and tuning", adv, tooltip="Rotate the motion image and adjust "
                                                                         "sensitivity while playing."))
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
        page.addWidget(page_header("Play", "Press Start and play inside the green square."))
        page.addLayout(body, 1)
        self._reset_display()

    # ------------------------------------------------------------------ choices
    def _update_enabled(self, *_):
        needs_model = self.detector.currentData() in ("fused", "cnn")
        self.model.setEnabled(needs_model and self.pipeline is None and not self._loading)

    def _refresh_models(self):
        current = self.model.currentData() or self.state.cfg.cnn.model_path
        self.model.clear()
        choices = model_choices()
        for path, name in choices:
            self.model.addItem(name, path)
        if not choices:
            self.model.addItem("No model yet", "")
        select_data(self.model, current)
        self.import_btn.setVisible(not os.path.exists(DEXTRA_MODEL))

    def on_activated(self):
        cfg = self.state.cfg
        self._refresh_models()
        if self.pipeline is None and not self._loading:
            select_data(self.detector, cfg.decision.source)
            select_data(self.mode, cfg.decision.mode)
        self._update_enabled()
        if not self.main.worker.isRunning():
            self.main.start_camera()

    def _import_dextra(self):
        if not self.runner.running():
            self.log.log("Downloading Dextra model...")
            self.import_btn.setEnabled(False)
            self.runner.start(["tools/import_dextra.py"])

    def _import_done(self, code):
        self.import_btn.setEnabled(True)
        self._refresh_models()
        if code == 0 and self._start_after_import:
            self._start_after_import = False
            self._test_dextra()
        elif code != 0:
            self.log.log("Download failed; check the internet connection.")

    def _test_dextra(self):
        if self._loading:
            return
        if not os.path.exists(DEXTRA_MODEL):
            self._start_after_import = True
            self._import_dextra()
            return
        if self.pipeline is not None:
            self._stop()
        self._refresh_models()
        select_data(self.model, DEXTRA_MODEL)
        select_data(self.detector, "cnn")
        select_data(self.mode, "continuous")
        cfg = self.state.cfg
        self.log.log(f"Motion image turned {cfg.cnn.rotate} degrees, mirrored: {'yes' if cfg.cnn.flip else 'no'}. "
                     "The Dextra model check on the Evaluate page finds the best setting for this camera.")
        self._toggle()

    # ------------------------------------------------------------------ start / stop
    def _toggle(self):
        if self._loading:
            return
        if self.pipeline is not None:
            self._stop()
            return
        cfg = self.state.cfg
        cfg.decision.source = self.detector.currentData()
        cfg.decision.mode = self.mode.currentData()
        if cfg.decision.source in ("fused", "cnn"):
            if self.model.currentData():
                cfg.cnn.model_path = self.model.currentData()
            elif cfg.decision.source == "cnn":
                QMessageBox.information(self, "No motion model", "Download the Dextra model or train one, or "
                                        "choose Hand tracker.")
                return
        self.state.mark_dirty()              # Play's choices are settings too; Save keeps them
        self._loading = True
        self.start_btn.setText("Loading...")
        self.start_btn.setEnabled(False)
        self.dextra_btn.setEnabled(False)
        self._update_enabled()
        if not self.main.worker.isRunning():
            self.main.start_camera()
        threading.Thread(target=self._load_models_thread, args=(cfg,), daemon=True).start()

    def _load_models_thread(self, cfg):
        """Loads the models off the UI thread so the window never freezes."""
        try:
            cnn, hand, messages = load_models(cfg, cfg.decision.source)
            refs = self._references(cnn)
            self.models_ready.emit((cnn, hand, messages, refs, None))
        except Exception as e:
            self.models_ready.emit((None, None, [], None, str(e)))

    def _models_loaded(self, result):
        cnn, hand, messages, refs, error = result
        self._loading = False
        self.start_btn.setEnabled(True)
        self.dextra_btn.setEnabled(True)
        if error:
            self.start_btn.setText("Start")
            self._update_enabled()
            QMessageBox.warning(self, "Cannot start", error)
            return
        if self._closing or self.main.current_page() is not self:   # the user left while the models loaded
            if hand is not None:
                hand.close()
            self.start_btn.setText("Start")
            self._update_enabled()
            return
        self.cnn, self.hand = cnn, hand
        for m in messages:
            self.log.log(m)
        self._show_references(refs)
        cfg = self.state.cfg
        if self.robot_on.isChecked():
            host = cfg.robot.host
            if self.mock_esp.isChecked():
                try:
                    self.mock = MockEsp(port=cfg.robot.port, verbose=False).start()
                    host = "127.0.0.1"
                except OSError as e:
                    self.log.log(f"Simulated robot could not start: {e}")
            self.link = RobotLink(host, cfg.robot.port, cfg.robot.heartbeat_s, cfg.robot.ack_timeout_s).start()
        self.latency = LatencyLog()
        self._rebuild = False
        self.pipeline = self._make_pipeline()
        self.start_btn.setText("Stop")
        set_kind(self.start_btn, "danger")
        self._show_options(False)

    def _references(self, cnn):
        """Dextra's most confident sample frame per class (computed in the loading thread)."""
        if cnn is None or cnn.arch != "dextra":
            return None
        files = sorted(glob.glob(os.path.join(DEXTRA_SAMPLES, "*.png")))
        if not files:
            return None
        rot, flip = cnn.rotate, cnn.flip
        cnn.rotate, cnn.flip = 0, False          # samples are already in Dextra's orientation
        best = [(-1.0, None)] * 4
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            lab, conf, _, _ = cnn.predict(img)
            if conf > best[lab][0]:
                best[lab] = (conf, img)
        cnn.rotate, cnn.flip = rot, flip
        return [colorize(img, 56) if img is not None else None for _c, img in best]

    def _show_references(self, refs):
        self.ref_box.setVisible(refs is not None)
        for lab, img in zip(self.ref_labels, refs or [None] * 4):
            if img is not None:
                lab.setPixmap(to_pixmap(img))
            else:
                lab.clear()

    def _show_options(self, show: bool):
        """While playing, the Run card shrinks to one summary line so the readings stay in view."""
        self.options.setVisible(show)
        self.dextra_btn.setVisible(show)
        self.summary.setVisible(not show)
        if not show:
            model = self.model.currentText() if self.cnn is not None else "no motion model"
            robot = ("simulated robot" if self.mock else "robot") if self.link else "robot off"
            self.summary.setText(f"{self.detector.currentText()} · {self.mode.currentText()} · {model} · {robot}")

    def _make_pipeline(self):
        return Pipeline(self.state.cfg, self.cnn, self.hand, pose_sink=self.link.send_pose if self.link else None)

    def _stop(self):
        with self._lock:
            self.pipeline = None
        if self.link:
            self.link.stop()
            self.link = None
        if self.mock:
            self.mock.stop()
            self.mock = None
        if self.hand:
            self.hand.close()
        self.cnn = self.hand = None
        self.start_btn.setText("Start")
        set_kind(self.start_btn, "primary")
        self._show_options(True)
        self._update_enabled()
        self._reset_display()

    def _reset_display(self):
        self._last_cnn_t = None
        self._cnn_times.clear()
        self._probs = None
        for b in self.bars:
            b.setValue(0)
        self.m_you.set("-")
        self.m_robot.set("-")
        self.m_round.set("Stopped")
        self.m_tempo.set("-")
        self.cnn_status.setText("Not running")
        self.mp_value.setText("off")
        self.decision_value.setText("-")
        self.chip_proc.set("Processing -", "off")
        self.chip_robot.set("Robot off", "off")
        for s in self.series.values():
            s.clear()

    def _reset_counters(self):
        if self.pipeline is not None:
            self.pipeline.engine.switches = self.pipeline.engine.commits = 0

    # ------------------------------------------------------------------ camera thread
    def processor(self, frame, src) -> dict:
        with self._lock:     # _stop() waits for the current frame before closing models
            return self._process_locked(frame, src)

    def _process_locked(self, frame, src) -> dict:
        interval = None if self._prev_frame_t is None else (frame.t - self._prev_frame_t) * 1000.0
        self._prev_frame_t = frame.t
        pipeline = self.pipeline
        img = frame.bgr.copy()
        draw_roi(img, src.roi)
        if pipeline is None:
            return {"display": img, "running": False}
        if self._rebuild:
            self._rebuild = False
            pipeline = self.pipeline = self._make_pipeline()
        if self.cnn is not None:            # orientation edits apply live
            self.cnn.rotate = int(self.state.cfg.cnn.rotate) % 360
            self.cnn.flip = bool(self.state.cfg.cnn.flip)
        r = pipeline.step(frame, src.roi)
        rec = pipeline.log_record(r)
        self.latency.add(rec)
        draw_hand(img, src.roi, r.hand)
        mp = None
        if r.hand is not None and not r.hand.skipped:
            mp = (r.hand.present, r.hand.gesture, r.hand.confidence)
        return {"display": img, "running": True, "snap": r.snapshot, "mp": mp, "rec": rec, "interval": interval,
                "link": self.link.stats() if self.link else None,
                "cnn_probs": r.cnn[2] if r.cnn is not None else None,
                "cnn_input": self.cnn.orient(r.dvs_frame.image) if (self.cnn and r.dvs_frame is not None) else None,
                "total_ms": self.latency.mean("total_ms")}

    # ------------------------------------------------------------------ UI thread
    def on_frame(self, p):
        self.view.show_image(p["display"])
        now = time.perf_counter()
        fps = p.get("fps", 0.0)
        if not p.get("running"):
            if now - self._last_ui >= UI_PERIOD_S:
                self._last_ui = now
                self.chip_cam.set(f"Camera {fps:.0f} fps", "ok" if fps >= 27 else ("warn" if fps >= 20 else "bad"))
            return

        rec = p["rec"]
        for key, field in (("Processing", "total_ms"), ("Motion model", "cnn_ms"), ("Hand tracker", "mp_ms")):
            v = rec.get(field)
            self.series[key].append(v if v is not None else np.nan)
        self.series["Camera interval"].append(p["interval"] if p.get("interval") is not None else np.nan)
        probs = p.get("cnn_probs")
        if probs is not None:
            self._last_cnn_t = now
            self._cnn_times.append(now)
            self._probs = probs
        if p.get("cnn_input") is not None:
            self.cnn_input.setPixmap(to_pixmap(colorize(p["cnn_input"], 144)))

        if now - self._last_plot >= PLOT_PERIOD_S:
            self._last_plot = now
            top = 0.0
            for key, curve in self.curves.items():
                data = np.asarray(self.series[key], dtype=float)
                curve.setData(data, connect="finite")
                if np.isfinite(data).any():
                    top = max(top, float(np.nanmax(data)))
            y_max = max(60.0, 1.15 * top)            # grow for spikes instead of cutting them off
            if abs(y_max - self._y_max) > 1.0:
                self._y_max = y_max
                self.plot.setYRange(0, y_max, padding=0)
        if now - self._last_ui < UI_PERIOD_S:
            return
        self._last_ui = now
        s = p["snap"]
        human = GESTURE_NAME.get(s.human) if s.human is not None else None
        robot = POSE_GESTURE.get(s.pose, "ready")
        self.m_you.set(human.upper() if human else "-", GESTURE_COLOR.get(human, MUTED))
        self.m_robot.set(robot.upper(), GESTURE_COLOR.get(robot, MUTED))
        self.m_round.set(round_status(s, self.state.cfg.decision.pumps_before_shoot), "#e4e7eb")
        self.m_tempo.set(f"{s.tempo:.2f} s/pump" if s.tempo else ("learning" if s.mode == "countdown" else "-"),
                         "#e4e7eb" if s.tempo else MUTED)
        if self._probs is not None:
            for b, v in zip(self.bars, self._probs):
                val = int(round(float(v) * 100))
                if b.value() != val:
                    b.setValue(val)
        if self.cnn is None:
            self.cnn_status.setText("Motion model off")
        else:
            while self._cnn_times and now - self._cnn_times[0] > 2.0:
                self._cnn_times.popleft()
            self.cnn_status.setText("Waiting for movement" if self._last_cnn_t is None else
                                    f"Updated {1000 * (now - self._last_cnn_t):.0f} ms ago · "
                                    f"{len(self._cnn_times) / 2:.0f} answers/s")
        mp = p.get("mp")
        if self.hand is None:
            self.mp_value.setText("off")
        elif mp is not None:
            present, gesture, conf = mp
            name = GESTURE_NAME.get(gesture, "unsure")
            self.mp_value.setText(f"{name.capitalize()} {conf:.0%}" if present else "no hand")
        who = SOURCE_NAME.get(s.source, "")
        self.decision_value.setText(f"{human.capitalize()} ({who})" if human and who else "-")
        self.counts.setText(f"Decisions {s.commits} · changes {s.switches}")

        ms = p.get("total_ms") or 0.0
        intervals = [v for v in self.series["Camera interval"] if v == v]
        cam_ms = float(np.mean(intervals)) if intervals else None
        self.chip_cam.set(f"Camera {fps:.0f} fps", "ok" if fps >= 27 else ("warn" if fps >= 20 else "bad"))
        self.chip_proc.set(f"Processing {ms:.0f} ms", "ok" if ms <= 25 else ("warn" if ms <= 33 else "bad"))
        self.delay_text.setText(f"avg processing {ms:.1f} ms" +
                                (f" · camera every {cam_ms:.0f} ms" if cam_ms is not None else ""))
        link = p.get("link")
        if link is None:
            self.chip_robot.set("Robot off", "off")
        elif link["reboots"]:
            self.chip_robot.set(f"Robot restarted {link['reboots']}x", "bad")
            self.chip_robot.setToolTip("The robot restarted during play"
                                       + (" because its power dipped: give the servos their own supply."
                                          if link["last_reset_reason"] == ESP_RST_BROWNOUT else "."))
        elif link["connected"]:
            name = "Simulated robot" if self.mock else "Robot"
            self.chip_robot.set(f"{name} {link['rtt_median_ms'] or 0:.1f} ms", "ok")
        else:
            self.chip_robot.set("Robot not replying", "bad")

    def on_deactivated(self):
        # Nothing processes frames while another page is open, so the robot must not keep repeating its
        # last move (and a second connection, e.g. Setup's connection test, must not fight this one).
        if self.pipeline is not None:
            self._stop()
            self.log.log("Stopped because another page was opened; the robot went to ready.")

    def _camera_state(self, s: str):
        self._prev_frame_t = None                    # a restarted camera starts a new frame clock
        if self.pipeline is not None and s != "running":
            self._stop()
            self.log.log("Stopped: the camera stopped sending images. The robot went to ready.")

    def shutdown(self):
        self._closing = True
        if self.pipeline is not None:
            self._stop()
        self.runner.kill()
