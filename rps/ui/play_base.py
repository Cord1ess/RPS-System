"""
What the Play and Play Debug pages share: loading the recognizer in the background, the robot
connection (set on Bot tuning), running the pipeline on camera frames, each reader's answer on the
video, the delay graph, the Dextra view and Mediapipe cards, the beat guide, and the per-throw
speed card (rps.ui.delay_view: how fast each throw was read and sent).

Names used everywhere: Dextra Raw, Dextra Tuned, Mediapipe, Both (Dextra Tuned + Mediapipe).
Each reader has one colour: violet = Dextra, cyan = Mediapipe.
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
from PySide6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QVBoxLayout, QWidget

from model import CLASS_NAMES
from rps.decision import GESTURE_NAME, READY
from rps.game import ENDLESS_ROUNDS, BeatSchedule
from rps.hud import draw_hand, draw_label, draw_roi
from rps.pipeline import RECOGNIZERS, Pipeline, load_models, missing_reason, reader_name
from rps.robot_link import ESP_RST_BROWNOUT, MockEsp, RobotLink
from rps.timing import LatencyLog
from rps.ui.base import Tab
from rps.ui.common import LogView, select_data, to_pixmap
from rps.ui.delay_view import DelayCard, throw_delay
from rps.ui.sound import BeatPlayer
from rps.ui.style import (BORDER, GESTURE_COLOR, MOTION_COLOR, MUTED, PANEL, TRACKER_COLOR, Card, Chip, caption,
                          label, static_plot, tip)

CLASS_SHORT = [c.split("_", 1)[1] for c in CLASS_NAMES]
BAR_NAMES = ["Rock", "Paper", "Scissors", "None"]
POSE_GESTURE = {"R": "rock", "P": "paper", "S": "scissors", "N": "ready"}
DEXTRA_SAMPLES = "models/dextra/sample_frames"
UI_PERIOD_S, PLOT_PERIOD_S = 0.066, 0.1
MOTION_STALE_S = 0.4                 # a Dextra answer older than this is shown as waiting
MOTION_BGR = tuple(int(MOTION_COLOR[i:i + 2], 16) for i in (5, 3, 1))     # '#rrggbb' -> OpenCV BGR
TRACKER_BGR = tuple(int(TRACKER_COLOR[i:i + 2], 16) for i in (5, 3, 1))
NO_CAMERA = "Not started: the camera is not running. Check the Setup page, then start again."
RECOGNIZER_TIP = ("Dextra Raw: Dextra's model as downloaded; reads movement. Dextra Tuned: Dextra tuned on your "
                  "recordings (Train page). Mediapipe: reads finger positions; works when the hand is still. Both: "
                  "Dextra Tuned while the hand moves, Mediapipe when it is steady or clearly disagrees.")


def colorize(frame64: np.ndarray, size: int) -> np.ndarray:
    return cv2.applyColorMap(cv2.resize(frame64, (size, size), interpolation=cv2.INTER_NEAREST),
                             cv2.COLORMAP_INFERNO)


def recognizer_combo(tooltip: str = RECOGNIZER_TIP) -> QComboBox:
    combo = tip(QComboBox(), tooltip)
    for key, name in RECOGNIZERS.items():
        combo.addItem(name, key)
    return combo


def refresh_recognizers(combo: QComboBox, cfg):
    """Greys out a recognizer whose model file is missing, with the reason on hover."""
    for i in range(combo.count()):
        key = combo.itemData(i)
        why = missing_reason(cfg, key) if key != "both" else None
        item = combo.model().item(i)
        item.setEnabled(why is None)
        combo.setItemData(i, why or RECOGNIZERS[key], 3)                    # hover (ToolTipRole)
        combo.setItemText(i, RECOGNIZERS[key] + ("  (not available)" if why else ""))
    if not combo.model().item(combo.currentIndex()).isEnabled():
        select_data(combo, "mediapipe")


class Metric(QWidget):
    """Small caption over a value, e.g. 'Decision' / 'ROCK'. Restyles only on change."""

    def __init__(self, title: str, tooltip: str, big: bool = False):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        t = QLabel(title)
        t.setObjectName("metricLabel")
        self.value = QLabel("-")
        self.value.setObjectName("metric")
        self._base = "font-size:26pt; font-weight:700;" if big else ""
        lay.addWidget(t)
        lay.addWidget(self.value)
        self.setToolTip(tooltip)
        self._text, self._color = None, None
        self.set("-")

    def set(self, text: str, color: str = MUTED):
        if text != self._text:
            self._text = text
            self.value.setText(text)
        if color != self._color:
            self._color = color
            self.value.setStyleSheet(f"{self._base} color:{color};")


class PlayBase(Tab):
    """Runs a recognizer on the camera; subclasses build the page and react to results."""
    uses_camera = True
    models_ready = Signal(object)

    def __init__(self, main):
        super().__init__(main)
        self.pipeline = None
        self.cnn = self.hand = self.link = self.mock = None
        self.running_recognizer = "mediapipe"     # the recognizer of the current run
        self._lock = threading.Lock()
        self.latency = LatencyLog()
        self._rebuild = False
        self._loading = False
        self._closing = False
        self._prev_frame_t = None
        self._last_cnn = None                    # (label, confidence, perf_counter) for the video label
        self._last_mp = (False, None, 0.0)
        self._last_cnn_t = None
        self._cnn_times = deque(maxlen=60)
        self._probs = None
        self._last_ui = self._last_plot = 0.0
        self._y_max = 60.0
        self.schedule = None
        self.guided = False                      # this run follows the beat guide
        self._beat_pending = False               # guided: the beat starts with the first processed frame
        self._camera_error = None                # the camera failed while the models loaded
        self.beats = BeatPlayer(self)
        self.decisions = deque()                 # (DecisionTiming, robot's move before it), camera -> UI thread
        self._robot_pose = READY                 # what the robot hand shows now (camera thread)
        self.log = LogView(500)
        self.models_ready.connect(self._models_loaded)
        main.worker.state_changed.connect(self._camera_state)

    # ------------------------------------------------------------------ shared widgets
    def make_delay_plot(self, height: int = 130):
        self.plot = static_plot(pg.PlotWidget())
        self.plot.setBackground(PANEL)
        self.plot.setFixedHeight(height)
        self.plot.setToolTip("Delay per camera frame over the last 5 seconds. Processing: time from a camera frame "
                             "arriving to the decision. Dextra and Mediapipe: time each took. Camera interval: time "
                             "between camera frames (33 ms = 30 fps).")
        self.plot.setLabel("left", "ms")
        self.plot.setYRange(0, 60)
        self.plot.showGrid(y=True, alpha=0.15)
        self.plot.addLegend(offset=(-8, 2), labelTextSize="8pt", colCount=4)
        self.curves = {k: self.plot.plot(pen=pg.mkPen(c, width=1.6), name=k) for k, c in
                       (("Processing", "#e0524a"), ("Dextra", MOTION_COLOR), ("Mediapipe", TRACKER_COLOR),
                        ("Camera interval", "#9aa3ad"))}
        self.series = {k: deque(maxlen=150) for k in self.curves}
        return self.plot

    def make_delay_card(self, table: bool = False) -> DelayCard:
        self.delay_card = DelayCard(table=table)
        return self.delay_card

    def make_chips(self) -> QHBoxLayout:
        self.chip_cam = Chip("Camera off", "off", "Camera frames per second. 27 or more is good.")
        self.chip_proc = Chip("Processing -", "off", "Average processing time per frame. Under 25 ms is good.")
        self.chip_robot = Chip("Robot off", "off", "The robot connection set on the Bot tuning page.")
        self.delay_text = caption("", "Averages over the last 100 frames.")
        self.delay_text.setWordWrap(False)
        row = QHBoxLayout()
        for c in (self.chip_cam, self.chip_proc, self.chip_robot):
            row.addWidget(c)
        row.addStretch(1)
        row.addWidget(self.delay_text)
        return row

    def make_dextra_card(self, preview: int = 144, references: bool = True) -> Card:
        card = Card("Dextra", "Dextra's raw answer, before the game rules. It reads the Dextra view (the movement "
                              "in the play zone), so it answers only while the hand moves.", color=MOTION_COLOR)
        self.dextra_card = card
        self.cnn_value = QLabel("-")
        self.cnn_value.setStyleSheet(f"font-size:13pt; font-weight:600; color:{MOTION_COLOR};")
        tip(self.cnn_value, "Dextra's latest answer and how sure it is.")
        card.body.addWidget(self.cnn_value)
        imgs = QHBoxLayout()
        self.cnn_input = QLabel()
        self.cnn_input.setFixedSize(preview, preview)
        self.cnn_input.setStyleSheet(f"background:#000; border:1px solid {BORDER};")
        tip(self.cnn_input, "Dextra view: the movement in the play zone that Dextra reads. Empty while the hand is "
                            "still.")
        imgs.addWidget(self.cnn_input)
        self.ref_box = QWidget()
        tip(self.ref_box, "Dextra's own training images for each gesture. Yours should look similar.")
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
        self._want_refs = references
        card.body.addLayout(imgs)
        bars = QGridLayout()
        bars.setVerticalSpacing(3)
        self.bars = []
        for i, (name, key) in enumerate(zip(BAR_NAMES, CLASS_SHORT)):
            b = QProgressBar()
            b.setRange(0, 100)
            b.setFormat("%p%")
            b.setStyleSheet(f"QProgressBar::chunk {{ background:{GESTURE_COLOR.get(key, GESTURE_COLOR['none'])}; "
                            f"border-radius:2px; }}")
            b.setToolTip(f"How sure Dextra is that the movement is {name.lower()}.")
            bars.addWidget(label(name, b.toolTip()), i, 0)
            bars.addWidget(b, i, 1)
            self.bars.append(b)
        card.body.addLayout(bars)
        self.cnn_status = caption("Not running", "When Dextra last answered, how often, and its time per image.")
        card.body.addWidget(self.cnn_status)
        return card

    def make_mediapipe_card(self) -> Card:
        card = Card("Mediapipe", "Mediapipe's raw answer, before the game rules. It finds the finger joints (the "
                                 "cyan points on the video), so it also works when the hand is still.",
                    color=TRACKER_COLOR)
        self.mediapipe_card = card
        self.mp_value = QLabel("-")
        self.mp_value.setStyleSheet(f"font-size:13pt; font-weight:600; color:{TRACKER_COLOR};")
        tip(self.mp_value, "Mediapipe's current answer and how sure it is.")
        card.body.addWidget(self.mp_value)
        self.mp_status = caption("Not running", "Time per camera frame, and how many matching frames a decision "
                                                "needs.")
        card.body.addWidget(self.mp_status)
        return card

    def show_readers(self, recognizer: str):
        """Enables the cards of the readers a recognizer uses (while stopped)."""
        uses_dextra = recognizer != "mediapipe"
        uses_mp = recognizer in ("mediapipe", "both")
        if hasattr(self, "dextra_card"):
            self.dextra_card.setEnabled(uses_dextra)
        if hasattr(self, "mediapipe_card"):
            self.mediapipe_card.setEnabled(uses_mp)

    # ------------------------------------------------------------------ start / stop
    @property
    def running(self) -> bool:
        return self.pipeline is not None

    def start_run(self, recognizer: str, mode: str) -> bool:
        """Loads the recognizer in the background; on_run_started() follows. False if it cannot start."""
        if self._loading or self.pipeline is not None:
            return False
        cfg = self.state.cfg
        why = missing_reason(cfg, recognizer) if recognizer != "both" else None
        if why:
            QMessageBox.information(self, "Not available", why)
            return False
        self.running_recognizer = recognizer
        cfg.decision.mode = mode
        self._loading = True
        self._camera_error = None
        if not self.main.worker.isRunning():
            self.main.start_camera()
        threading.Thread(target=self._load_models_thread, args=(cfg, recognizer), daemon=True).start()
        return True

    def _load_models_thread(self, cfg, recognizer):
        """Loads the models off the UI thread so the window never freezes."""
        try:
            cnn, hand, messages = load_models(cfg, recognizer)
            refs = self._references(cnn) if getattr(self, "_want_refs", False) else None
            self.models_ready.emit((cnn, hand, messages, refs, None))
        except Exception as e:
            self.models_ready.emit((None, None, [], None, str(e)))

    def _models_loaded(self, result):
        cnn, hand, messages, refs, error = result
        self._loading = False
        if error:
            self.on_run_failed()
            QMessageBox.warning(self, "Cannot start", error)
            return
        if self._closing or self.main.current_page() is not self:   # the user left while the models loaded
            if hand is not None:
                hand.close()
            self.on_run_failed()
            return
        if self._camera_error is not None or not self.main.worker.isRunning():
            if hand is not None:                  # no camera: a match would run blind
                hand.close()
            self.log.log(NO_CAMERA)
            self.main.statusBar().showMessage(NO_CAMERA, 8000)
            self.on_run_failed("No camera")
            return
        self.cnn, self.hand = cnn, hand
        for m in messages:
            self.log.log(m)
        if hasattr(self, "ref_box"):
            self._show_references(refs)
        if hasattr(self, "dextra_card"):
            self.dextra_card.head.setText(reader_name("cnn", cnn) if cnn is not None else "Dextra: off")
            self.dextra_card.setEnabled(cnn is not None)
        if hasattr(self, "mediapipe_card"):
            self.mediapipe_card.head.setText("Mediapipe" if hand is not None else "Mediapipe: off")
            self.mediapipe_card.setEnabled(hand is not None)
        self._start_robot()
        self.latency = LatencyLog()
        self.decisions.clear()
        self._robot_pose = READY
        if hasattr(self, "delay_card"):
            self.delay_card.reset()
        self._rebuild = False
        self.schedule = None
        self.guided = self._beat_pending = self.state.cfg.decision.mode == "guided"
        self.pipeline = self._make_pipeline()
        self.on_run_started()

    def _start_beat(self):
        """Guided: starts the beat and the engine's rounds on one clock, once camera frames arrive, so a
        slow-opening camera cannot eat the first round."""
        cfg = self.state.cfg
        g = cfg.game
        schedule = BeatSchedule(time.perf_counter() + 0.3, 60.0 / g.beat_bpm, cfg.decision.pumps_before_shoot,
                                g.rounds or ENDLESS_ROUNDS, g.lead_beats, g.gap_beats)
        with self._lock:                          # not while the camera thread is inside a frame
            if self.pipeline is None:
                return
            self.pipeline.engine.start_guided(schedule, (cfg.latency.camera_latency_ms + g.audio_latency_ms) / 1000.0)
            self.schedule = schedule
        self.beats.start(schedule, g.sound, g.beat_volume, g.cue_volume)
        self.on_beat_started()

    def _start_robot(self):
        cfg = self.state.cfg.robot
        if cfg.mode == "off":
            return
        host = None
        if cfg.mode == "simulated":
            try:
                self.mock = MockEsp(port=cfg.port, verbose=False).start()
                host = "127.0.0.1"
            except OSError as e:
                self.log.log(f"Simulated robot could not start: {e}")
                return
        self.link = RobotLink.from_config(cfg, host=host).start()

    def _references(self, cnn):
        """Dextra's most confident sample image per gesture (computed in the loading thread)."""
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

    def _make_pipeline(self):
        return Pipeline(self.state.cfg, self.cnn, self.hand, pose_sink=self.link.send_pose if self.link else None)

    def stop_run(self, reason: str = ""):
        self.beats.stop()
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
        self.schedule = None
        self.guided = self._beat_pending = False
        if reason:
            self.log.log(reason)
        self.reset_readers()
        self.on_run_stopped()

    def reset_readers(self):
        self._last_cnn_t = None
        self._cnn_times.clear()
        self._probs = None
        self._last_cnn = None
        self._last_mp = (False, None, 0.0)
        if hasattr(self, "bars"):
            for b in self.bars:
                b.setValue(0)
            self.cnn_value.setText("-")
            self.cnn_status.setText("Not running")
            self.dextra_card.head.setText("Dextra")
        if hasattr(self, "mp_value"):
            self.mp_value.setText("-")
            self.mp_status.setText("Not running")
            self.mediapipe_card.head.setText("Mediapipe")
        if hasattr(self, "chip_proc"):
            self.chip_proc.set("Processing -", "off")
            self.chip_robot.set("Robot off", "off")
        if hasattr(self, "series"):
            for s in self.series.values():
                s.clear()

    # hooks for the pages
    def on_run_started(self):
        pass

    def on_run_stopped(self):
        pass

    def on_run_failed(self, reason: str = ""):
        pass

    def on_beat_started(self):
        pass

    def on_frame_ui(self, p, now: float):
        pass

    # ------------------------------------------------------------------ camera thread
    def processor(self, frame, src) -> dict:
        with self._lock:     # stop_run() waits for the current frame before closing models
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
            engine = pipeline.engine
            pipeline = self.pipeline = self._make_pipeline()
            if engine.mode == "guided":                # keep the match running across a settings change
                pipeline.engine.start_guided(engine.schedule, engine.sync_offset_s)
                pipeline.engine.round_results = engine.round_results
        if self.cnn is not None and not self.cnn.fixed_orientation:   # Dextra Raw: orientation edits apply live
            self.cnn.rotate = int(self.state.cfg.cnn.rotate) % 360
            self.cnn.flip = bool(self.state.cfg.cnn.flip)
        r = pipeline.step(frame, src.roi)
        self._track_robot(r)
        rec = pipeline.log_record(r)
        self.latency.add(rec)
        now = time.perf_counter()
        if r.cnn is not None:
            self._last_cnn = (r.cnn[0], r.cnn[1], now)
        mp = None
        if r.hand is not None and not r.hand.skipped:
            mp = (r.hand.present, r.hand.gesture, r.hand.confidence)
            self._last_mp = mp
        self._draw_readers(img, src.roi, r.hand, now)
        return {"display": img, "running": True, "snap": r.snapshot, "mp": mp, "rec": rec, "interval": interval,
                "link": self.link.stats() if self.link else None,
                "cnn_probs": r.cnn[2] if r.cnn is not None else None,
                "cnn_input": self.cnn.orient(r.dvs_frame.image) if (self.cnn and r.dvs_frame is not None) else None,
                "total_ms": self.latency.mean("total_ms")}

    def _track_robot(self, r):
        """Follows what the hand shows (the team firmware keeps its last move on READY) and queues each
        decision with the move the hand came from, for its move time."""
        came_from = None
        for pose, _t in r.poses:
            if r.decision is not None and came_from is None and pose == r.decision.pose:
                came_from = self._robot_pose
            if pose != READY or self.state.cfg.robot.protocol == "ack":
                self._robot_pose = pose
        if r.decision is not None:
            self.decisions.append((r.decision, came_from if came_from is not None else self._robot_pose))

    def on_decision(self, delay):
        """A throw was decided (rps.ui.delay_view.ThrowDelay)."""
        if hasattr(self, "delay_card"):
            self.delay_card.add(delay)

    def _draw_readers(self, img, roi, hand, now):
        """Each reader's own answer on the video, in its colour, so the two are never confused."""
        draw_hand(img, roi, hand, TRACKER_BGR)
        x, y, _s = roi
        line = 20
        if self.hand is not None:
            present, gesture, conf = self._last_mp
            draw_label(img, f"Mediapipe: {GESTURE_NAME.get(gesture, 'unsure').upper()} {conf:.0%}" if present
                       else "Mediapipe: no hand", (x + 6, y + line), TRACKER_BGR)
            line += 22
        if self.cnn is not None:
            name = reader_name("cnn", self.cnn)
            if self._last_cnn is None or now - self._last_cnn[2] > MOTION_STALE_S:
                text = f"{name}: waiting for movement"
            else:
                text = f"{name}: {GESTURE_NAME.get(self._last_cnn[0], '?').upper()} {self._last_cnn[1]:.0%}"
            draw_label(img, text, (x + 6, y + line), MOTION_BGR)

    # ------------------------------------------------------------------ UI thread
    def on_frame(self, p):
        self.view.show_image(p["display"])
        while self.decisions:                    # every decision, even if its frame's picture was skipped
            timing, came_from = self.decisions.popleft()
            self.on_decision(throw_delay(timing, came_from, self.state.cfg, self.cnn))
        now = time.perf_counter()
        fps = p.get("fps", 0.0)
        if not p.get("running"):
            if hasattr(self, "chip_cam") and now - self._last_ui >= UI_PERIOD_S:
                self._last_ui = now
                self.chip_cam.set(f"Camera {fps:.0f} fps", "ok" if fps >= 27 else ("warn" if fps >= 20 else "bad"))
            return
        if self._beat_pending:
            self._beat_pending = False
            self._start_beat()
        if hasattr(self, "series"):
            rec = p["rec"]
            for key, field in (("Processing", "total_ms"), ("Dextra", "cnn_ms"), ("Mediapipe", "mp_ms")):
                v = rec.get(field)
                self.series[key].append(v if v is not None else np.nan)
            self.series["Camera interval"].append(p["interval"] if p.get("interval") is not None else np.nan)
        probs = p.get("cnn_probs")
        if probs is not None:
            self._last_cnn_t = now
            self._cnn_times.append(now)
            self._probs = probs
        if p.get("cnn_input") is not None and hasattr(self, "cnn_input"):
            self.cnn_input.setPixmap(to_pixmap(colorize(p["cnn_input"], self.cnn_input.width())))
        if hasattr(self, "plot") and now - self._last_plot >= PLOT_PERIOD_S:
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
        self._update_readers(p, now)
        self._update_chips(p, fps)
        self.on_frame_ui(p, now)

    def _update_readers(self, p, now):
        if hasattr(self, "bars"):
            if self._probs is not None:
                for b, v in zip(self.bars, self._probs):
                    val = int(round(float(v) * 100))
                    if b.value() != val:
                        b.setValue(val)
            if self.cnn is None:
                self.cnn_value.setText("Off")
                self.cnn_status.setText("Not used by this choice")
            else:
                while self._cnn_times and now - self._cnn_times[0] > 2.0:
                    self._cnn_times.popleft()
                if self._probs is not None and self._last_cnn_t is not None and \
                        now - self._last_cnn_t < MOTION_STALE_S:
                    k = int(np.argmax(self._probs))
                    self.cnn_value.setText(f"{BAR_NAMES[k].upper()} {float(self._probs[k]):.0%}")
                else:
                    self.cnn_value.setText("Waiting for movement")
                ms = self.latency.mean("cnn_ms")
                self.cnn_status.setText(("No movement yet" if self._last_cnn_t is None else
                                         f"{len(self._cnn_times) / 2:.0f} answers/s · last "
                                         f"{1000 * (now - self._last_cnn_t):.0f} ms ago")
                                        + (f" · {ms:.1f} ms each" if ms else "")
                                        + f" · decides after {self.state.cfg.vote.k} matching answers")
        if hasattr(self, "mp_value"):
            mp = p.get("mp")
            if self.hand is None:
                self.mp_value.setText("Off")
                self.mp_status.setText("Not used by this choice")
            else:
                if mp is not None:
                    present, gesture, conf = mp
                    self.mp_value.setText(f"{GESTURE_NAME.get(gesture, 'unsure').upper()} {conf:.0%}" if present
                                          else "No hand")
                ms = self.latency.mean("mp_ms")
                self.mp_status.setText((f"{ms:.0f} ms per camera frame · " if ms else "")
                                       + f"decides once steady for {self.state.cfg.decision.mp_stable_ms:.0f} ms")

    def _update_chips(self, p, fps):
        if not hasattr(self, "chip_cam"):
            return
        ms = p.get("total_ms") or 0.0
        intervals = [v for v in self.series["Camera interval"] if v == v] if hasattr(self, "series") else []
        cam_ms = float(np.mean(intervals)) if intervals else None
        self.chip_cam.set(f"Camera {fps:.0f} fps", "ok" if fps >= 27 else ("warn" if fps >= 20 else "bad"))
        self.chip_proc.set(f"Processing {ms:.0f} ms", "ok" if ms <= 25 else ("warn" if ms <= 33 else "bad"))
        self.delay_text.setText(f"avg processing {ms:.1f} ms" +
                                (f" · camera every {cam_ms:.0f} ms" if cam_ms is not None else ""))
        link = p.get("link")
        if link is None:
            self.chip_robot.set("Robot off", "off")
        elif not link["replies"]:                    # team firmware: RPS:<GESTURE>, cannot reply
            if self.mock is not None:
                self.chip_robot.set(f"Simulated robot received {self.mock.received}", "ok")
            else:
                self.chip_robot.set(f"Robot: {link['sent']} moves sent", "info")
            self.chip_robot.setToolTip("Moves are sent as RPS:ROCK, RPS:PAPER or RPS:SCISSORS when the robot's move "
                                       "changes. This firmware does not reply, so delivery and reply time cannot be "
                                       "shown; it has no ready position, so the hand keeps its last move.")
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

    # ------------------------------------------------------------------ page life
    def on_activated(self):
        if not self.main.worker.isRunning():
            self.main.start_camera()

    def on_deactivated(self):
        # Nothing processes frames while another page is open, so the robot must not keep repeating its
        # last move (and a second connection, e.g. Bot tuning's test, must not fight this one).
        if self.pipeline is not None:
            self.stop_run("Stopped because another page was opened.")

    def _camera_state(self, s: str):
        self._prev_frame_t = None                    # a restarted camera starts a new frame clock
        if self._loading:
            self._camera_error = None if s == "running" else s     # checked when the models are ready
        if self.pipeline is not None and s != "running" and self.main.current_page() is self:
            self.stop_run("Stopped: the camera stopped sending images.")

    def shutdown(self):
        self._closing = True
        if self.pipeline is not None:
            self.stop_run()
