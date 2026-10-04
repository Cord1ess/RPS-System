"""
Setup page: camera, light and play zone. Manual settings and tools are folded away.

The camera reacts at once: choosing another webcam or source restarts it in the background (the
window never waits for the old one to close), exposure and colour balance apply to the running camera
without reopening it, and other camera settings restart it by themselves.
"""

import re
import sys

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from rps.camera import crop_roi, margin_box
from rps.ui.base import Tab
from rps.ui.common import ConfigForm, LogView, ProcessRunner, VideoView
from rps.ui.style import Card, Chip, Collapsible, button, caption, page_header, row, tip

SOURCES = [("camera", "Webcam"), ("mock", "Simulated camera"), ("video", "Recorded session")]
LATENCY_RESULT = re.compile(r"^\[latency\] \w+: median ([\d.]+) ms")
RESTART_KEYS = ("index", "url", "backend", "width", "height", "fps", "fourcc")   # need the camera reopened


def list_webcams():
    """(camera number, name) for every webcam the system reports. On Linux the number comes from the
    device path (/dev/videoN); on Windows it is the position, which is how OpenCV numbers them."""
    try:
        from PySide6.QtMultimedia import QMediaDevices
        devices = QMediaDevices.videoInputs()
    except Exception:                      # no multimedia support: the camera number setting still works
        return []
    out = []
    for i, d in enumerate(devices):
        dev_id = bytes(d.id()).decode(errors="replace")
        m = re.search(r"/dev/video(\d+)", dev_id)
        out.append((int(m.group(1)) if m and sys.platform != "win32" else i, d.description() or f"Camera {i}"))
    return out


class SetupTab(Tab):
    title = "1  Setup"
    uses_camera = True

    def __init__(self, main):
        super().__init__(main)
        self.view = VideoView(placeholder="Camera off")
        tip(self.view, "Live camera. Green: play zone (the only area analysed). Grey: extra area saved when "
                       "recording. Use 'Set play zone' and drag to change it.")
        self.view.roi_selected.connect(self._roi_selected)
        self.runner = ProcessRunner(self)
        self.log = LogView()
        self.runner.line.connect(self.log.log)
        self.runner.line.connect(self._tool_line)
        self.runner.finished.connect(self._process_done)
        self._after = ""

        # --- camera
        cam = Card("Camera", "Start the webcam and check frame rate and light.")
        self.source = tip(QComboBox(), "Webcam: the real camera. Simulated camera: a moving test pattern. "
                                       "Recorded session: replay a recording.")
        for data, text in SOURCES:
            self.source.addItem(text, data)
        self.cam_btn = button("Start camera", "primary", "Open or close the selected camera.")
        self.cam_btn.clicked.connect(self._toggle_camera)
        r = QHBoxLayout()
        r.addWidget(self.source, 1)
        r.addWidget(self.cam_btn)
        cam.body.addLayout(r)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.device = tip(QComboBox(), "Which webcam (built-in or USB). Changing it switches the running camera.")
        self.device.currentIndexChanged.connect(self._device_changed)
        refresh = button("Find cameras", tooltip="List the webcams again (after plugging one in).")
        refresh.clicked.connect(lambda: self._fill_devices(force=True))
        cam.body.addLayout(row(self.device, refresh))
        self.device.setVisible(False)                 # shown once listed, if there is a choice
        self._devices_listed = False
        self._restart_timer = QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.setInterval(700)            # one restart after a burst of edits
        self._restart_timer.timeout.connect(self._reopen)
        self.chip_fps = Chip("Frame rate -", "off", "New images per second. 27 or more is good; this camera's "
                                                     "maximum is 30. Images the camera sends twice count once.")
        self.chip_light = Chip("Light -", "off", "Average brightness inside the play zone (0-255). 60-200 is good.")
        self.chip_clip = Chip("Overexposed -", "off", "Share of the play zone that is pure white. Under 5% is good.")
        cam.body.addLayout(row(self.chip_fps, self.chip_light, self.chip_clip))
        self.light_hint = caption("")
        cam.body.addWidget(self.light_hint)
        probe = button("Auto-configure", tooltip="Measures every camera mode in the current light (about 10-15 s) "
                                                 "and saves the best one. Run it again when the lighting changes.")
        probe.clicked.connect(self._probe)
        cam.body.addLayout(row(probe))
        manual = QWidget()
        ml = QVBoxLayout(manual)
        ml.setContentsMargins(0, 0, 0, 0)
        self.cam_form = ConfigForm(self.state, "camera")
        self.cam_form.changed.connect(self._camera_setting_changed)
        ml.addWidget(self.cam_form)
        ml.addWidget(caption("Exposure, colour balance and mirroring apply to the running camera at once; other "
                             "settings restart it by themselves."))
        reopen = button("Apply and restart camera", tooltip="Restart the camera with the values above.")
        reopen.clicked.connect(self._reopen)
        ml.addLayout(row(reopen))
        cam.body.addWidget(Collapsible("Manual camera settings", manual))

        # --- play zone
        zone = Card("Play zone", "The square the app analyses. Frame only the hand, not your face.")
        self.roi_btn = button("Set play zone", tooltip="Click, then drag a square on the video. Click again to "
                                                       "cancel.")
        self.roi_btn.setCheckable(True)
        self.roi_btn.toggled.connect(self._roi_mode)
        self.zone_label = caption("", "Current play zone position and size in camera pixels.")
        self.zone_label.setWordWrap(False)
        zone.body.addLayout(row(self.roi_btn, self.zone_label))

        # --- camera delay (the robot itself is on the Bot tuning page)
        lat = QWidget()
        ll = QHBoxLayout(lat)
        ll.setContentsMargins(0, 0, 0, 0)
        for text, mode, t in (("Mirror test", "screen", "This screen flashes; hold a mirror so the camera sees it. "
                                                        "Includes the screen's own delay."),
                              ("LED test", "led", "The robot flashes its LED inside the play zone; measures how late "
                                                  "the camera sees it. Needs the reference firmware.")):
            b = button(text, tooltip=t)
            b.clicked.connect(lambda _c=False, m=mode: self._latency(m))
            ll.addWidget(b)
        ll.addStretch(1)
        cam.body.addWidget(Collapsible("Camera delay tests", lat, tooltip="Measure how many milliseconds the camera "
                                                                           "image lags behind reality."))

        save = button("Save setup", "primary", "Save the camera and play zone settings to config.json.")
        save.clicked.connect(self.state.save)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 6, 0)
        for w in (cam, zone):
            pl.addWidget(w)
        pl.addLayout(row(save))
        pl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(500)

        left = QVBoxLayout()
        left.addWidget(self.view, 1)
        self.output = Collapsible("Tool output", self.log, tooltip="Output of auto-configure and delay tests.")
        left.addWidget(self.output)
        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addWidget(scroll)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Setup", "Once per location: camera, light and play zone. Then Save setup. "
                                            "The robot is on Bot tuning."))
        page.addLayout(body, 1)
        main.worker.state_changed.connect(self._cam_state)
        self._zone_text()

    # ------------------------------------------------------------------ camera thread
    def processor(self, frame, src) -> dict:
        img = frame.bgr.copy()
        h, w = img.shape[:2]
        roi = src.roi
        x0, y0, x1, y1 = margin_box(roi, self.state.cfg.roi.record_margin, w, h)
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (110, 110, 110), 1)
        x, y, s = roi
        cv2.rectangle(img, (x, y), (x + s, y + s), (0, 220, 0), 2)
        gray = cv2.cvtColor(crop_roi(frame.bgr, roi), cv2.COLOR_BGR2GRAY)
        return {"display": img, "brightness": float(gray.mean()), "clipped": float(np.mean(gray >= 250) * 100)}

    # ------------------------------------------------------------------ UI thread
    def on_frame(self, p):
        self.view.show_image(p["display"])
        fps, b, clip = p.get("fps", 0.0), p.get("brightness", 0.0), p.get("clipped", 0.0)
        self.chip_fps.set(f"Frame rate {fps:.0f} fps", "ok" if fps >= 27 else ("warn" if fps >= 20 else "bad"))
        self.chip_light.set(f"Light {b:.0f}", "ok" if 60 <= b <= 200 else ("warn" if b >= 40 else "bad"))
        self.chip_clip.set(f"Overexposed {clip:.0f}%", "ok" if clip < 5 else "warn")
        if b < 40:
            hint = "Too dark: add a lamp over the play zone, then Auto-configure."
        elif b < 60:
            hint = "Dim: more light allows a shorter exposure and less motion blur."
        elif clip >= 5:
            hint = "Parts of the hand are washed out: move the lamp back or Auto-configure."
        elif fps < 27 and p.get("repeats", 0) > 30:
            hint = (f"The camera sends many images twice, so only {fps:.0f} are new each second: add light, then "
                    "run Auto-configure (it compares both camera drivers).")
        elif fps < 27:
            hint = "Low frame rate: run Auto-configure."
        else:
            hint = ""
        if hint != self.light_hint.text():
            self.light_hint.setText(hint)
        self._zone_text(p.get("roi"))

    def _zone_text(self, roi=None):
        x, y, s = roi or (self.state.cfg.roi.x, self.state.cfg.roi.y, self.state.cfg.roi.size)
        text = f"{s} x {s} px at ({x}, {y})"
        if text != self.zone_label.text():
            self.zone_label.setText(text)

    def _cam_state(self, s):
        running = s == "running"
        self.cam_btn.setText("Stop camera" if running else "Start camera")
        if not running:
            for c, name in ((self.chip_fps, "Frame rate"), (self.chip_light, "Light"),
                            (self.chip_clip, "Overexposed")):
                c.set(f"{name} -", "off")
        if s.startswith("error"):
            self.log.log(f"[camera] {s}")
            self.light_hint.setText("The camera could not be opened. Close other apps using it (OBS, Teams, "
                                    "browser) and check Windows camera privacy settings.")

    def on_activated(self):
        if not self._devices_listed:
            QTimer.singleShot(0, self._fill_devices)       # listing takes ~1 s the first time: after showing

    def _fill_devices(self, force: bool = False):
        self._devices_listed = True
        cams = list_webcams()
        self.device.blockSignals(True)
        self.device.clear()
        for index, name in cams:
            self.device.addItem(f"{name} (camera {index})", index)
        current = self.state.cfg.camera.index
        if self.device.findData(current) < 0:
            self.device.addItem(f"Camera {current}", current)
        self.device.setCurrentIndex(self.device.findData(current))
        self.device.blockSignals(False)
        self.device.setVisible(self.device.count() > 1 or force)

    def _device_changed(self, _i):
        index = self.device.currentData()
        cfg = self.state.cfg.camera
        if index is None or index == cfg.index:
            return
        cfg.index = int(index)
        self.state.mark_dirty()
        self.cam_form.refresh()
        if self.main.worker.isRunning() and self.main.worker.kind == "camera":
            self.main.restart_camera("camera")

    def _source_changed(self, _i):
        """A running camera switches to the newly chosen source straight away."""
        if not self.main.worker.isRunning() or self.main.worker.stopping:
            return
        kind = self.source.currentData()
        if kind == self.main.worker.kind and kind != "video":
            return
        video = None
        if kind == "video":
            video = QFileDialog.getExistingDirectory(self, "Choose a recording folder", self.state.recordings_root)
            if not video:
                return
        self.main.restart_camera(kind, video)

    def _camera_setting_changed(self, _section: str, key: str):
        worker = self.main.worker
        if not worker.isRunning() or worker.kind != "camera":
            return
        if key in RESTART_KEYS:
            self._restart_timer.start()
        else:
            worker.apply_camera_settings()

    def _toggle_camera(self):
        if self.main.worker.isRunning():
            self.main.stop_camera(wait=False)              # released in the background
            return
        kind, video = self.source.currentData(), None
        if kind == "video":
            video = QFileDialog.getExistingDirectory(self, "Choose a recording folder", self.state.recordings_root)
            if not video:
                return
        self.main.start_camera(kind, video)

    def _reopen(self):
        if self.main.worker.isRunning():
            self.main.restart_camera(self.main.worker.kind, self.main.worker.video_path)

    def _roi_mode(self, on: bool):
        self.view.roi_edit = on
        self.roi_btn.setText("Drag on the video" if on else "Set play zone")

    def _roi_selected(self, x, y, size):
        self.main.worker.set_roi(x, y, size)
        src = self.main.worker.source
        if src is not None:
            x, y, size = src.roi
        self.roi_btn.setChecked(False)
        if self.main.worker.kind != "camera":
            # a recording or the simulated camera has its own frame: its coordinates are not the webcam's
            self.light_hint.setText("Play zone changed for this view only. Set it on the webcam to keep it.")
            return
        roi = self.state.cfg.roi
        roi.x, roi.y, roi.size = int(x), int(y), int(size)
        self.state.mark_dirty()

    def _run_tool(self, after: str, args):
        """Runs a Setup tool that needs the webcam to itself; the camera restarts when it finishes."""
        if self.runner.running():
            return
        self.main.stop_camera()
        self.main.camera_blocked = True
        self.output.toggle.setChecked(True)
        self._after = after
        self.runner.start(args)

    def _probe(self):
        self.log.log("Auto-configure: measuring each camera mode...")
        self._run_tool("probe", ["tools/camera_probe.py", "--write-config", "--config", self.state.config_path])

    def _latency(self, mode: str):
        if mode == "led" and self.state.cfg.robot.protocol != "ack":
            self.output.toggle.setChecked(True)
            self.log.log("The LED test needs the reference firmware; the team firmware has no LED command. "
                         "Use the mirror test.")
            return
        self._run_tool("latency", ["tools/latency_test.py", "--mode", mode, "--config", self.state.run_config()])

    def _tool_line(self, line: str):
        m = LATENCY_RESULT.search(line)
        if m and self._after == "latency":
            ms = float(m.group(1))
            self.state.cfg.latency.camera_latency_ms = round(ms, 1)
            self.state.mark_dirty()
            self.log.log(f"Camera delay set to {ms:.0f} ms. Save setup to keep it.")

    def _process_done(self, code):
        self.main.camera_blocked = False
        if self._after == "probe":
            if code == 0:
                self.state.reload(["camera"])
                self.log.log("Camera settings saved.")
            else:
                self.log.log("Auto-configure did not finish; camera settings unchanged.")
        if self.main.current_page().uses_camera:
            self.main.start_camera()

    def shutdown(self):
        self.runner.kill()
