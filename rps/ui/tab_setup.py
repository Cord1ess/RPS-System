"""Setup page: camera, play zone, robot hand. Manual settings and tools are folded away."""

import re

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from rps.camera import crop_roi, margin_box
from rps.robot_link import ESP_RST_BROWNOUT, MockEsp, RobotLink
from rps.ui.base import Tab
from rps.ui.common import ConfigForm, LogView, ProcessRunner, VideoView
from rps.ui.style import Card, Chip, Collapsible, button, caption, page_header, row, tip

SOURCES = [("camera", "Webcam"), ("mock", "Simulated camera"), ("video", "Recorded session")]
LATENCY_RESULT = re.compile(r"^\[latency\] \w+: median ([\d.]+) ms")


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
        self._link = self._mock = None
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
        self.chip_fps = Chip("Frame rate -", "off", "Frames per second. 27 or more is good; this camera's maximum "
                                                     "is 30.")
        self.chip_light = Chip("Light -", "off", "Average brightness inside the play zone (0-255). 60-200 is good.")
        self.chip_clip = Chip("Overexposed -", "off", "Share of the play zone that is pure white. Under 5% is good.")
        cam.body.addLayout(row(self.chip_fps, self.chip_light, self.chip_clip))
        self.light_hint = caption("")
        cam.body.addWidget(self.light_hint)
        probe = button("Auto-configure", tooltip="Measures every camera mode in the current light (about 25 s) "
                                                 "and saves the best one. Run it again when the lighting changes.")
        probe.clicked.connect(self._probe)
        cam.body.addLayout(row(probe))
        manual = QWidget()
        ml = QVBoxLayout(manual)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(ConfigForm(self.state, "camera"))
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

        # --- robot
        robot = Card("Robot hand", "Connection to the ESP32. The laptop must be on the same network as the robot.")
        robot.body.addWidget(ConfigForm(self.state, "robot", keys=["protocol", "host", "port"]))
        self.mock_chk = tip(QCheckBox("Simulated robot"), "Test against a simulated robot on this computer.")
        test = button("Test connection", tooltip="Team firmware: sends RPS:PAPER once, so the hand should open "
                                                 "(it cannot reply). Reference firmware: sends a message and "
                                                 "waits 1.5 s for its reply.")
        test.clicked.connect(self._test_link)
        self.chip_link = Chip("Not tested", "off", "Result of the last connection test.")
        robot.body.addLayout(row(test, self.mock_chk, self.chip_link))
        self.link_hint = caption("")
        robot.body.addWidget(self.link_hint)
        lat = QWidget()
        ll = QHBoxLayout(lat)
        ll.setContentsMargins(0, 0, 0, 0)
        for text, mode, t in (("LED test", "led", "The robot flashes its LED inside the play zone; measures how "
                                                  "late the camera sees it. Needs the robot."),
                              ("Mirror test", "screen", "This screen flashes; hold a mirror so the camera sees it. "
                                                        "Includes the screen's own delay.")):
            b = button(text, tooltip=t)
            b.clicked.connect(lambda _c=False, m=mode: self._latency(m))
            ll.addWidget(b)
        ll.addStretch(1)
        robot.body.addWidget(Collapsible("Camera delay tests", lat, tooltip="Measure how many milliseconds the "
                                                                             "camera image lags behind reality."))

        save = button("Save setup", "primary", "Save camera, play zone and robot settings to config.json.")
        save.clicked.connect(self.state.save)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 6, 0)
        for w in (cam, zone, robot):
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
        page.addWidget(page_header("Setup", "Once per location: camera, light, play zone, robot. Then Save setup."))
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

    def _toggle_camera(self):
        if self.main.worker.isRunning():
            self.main.stop_camera()
            return
        kind, video = self.source.currentData(), None
        if kind == "video":
            video = QFileDialog.getExistingDirectory(self, "Choose a recording folder", self.state.recordings_root)
            if not video:
                return
        self.main.start_camera(kind, video)

    def _reopen(self):
        if self.main.worker.isRunning():
            kind, video = self.main.worker.kind, self.main.worker.video_path
            self.main.stop_camera()
            self.main.start_camera(kind, video)

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

    def _test_link(self):
        if self._link is not None:
            return
        host, port = self.state.cfg.robot.host, self.state.cfg.robot.port
        if self.mock_chk.isChecked():
            try:
                self._mock = MockEsp(port=port, verbose=False).start()
            except OSError:
                self.chip_link.set("Simulated robot failed", "bad")
                self.link_hint.setText(f"Port {port} is already in use on this computer, probably by a running "
                                       f"game or another simulated robot. Stop it and test again.")
                return
            host = "127.0.0.1"
        cfg = self.state.cfg.robot
        self._link = RobotLink(host, port, heartbeat_s=0.05, protocol=cfg.protocol).start()
        # reference firmware: READY, then wait for replies; team firmware: one visible move (it cannot reply)
        self._link.send_pose("N" if cfg.protocol == "ack" else "P")
        self.chip_link.set("Testing...", "info")
        QTimer.singleShot(1500, self._link_result)

    def _link_result(self):
        stats = self._link.stats()
        self._link.stop(send_ready=False)
        self._link = None
        simulated = self._mock is not None
        received = self._mock.received if simulated else 0
        if self._mock is not None:
            self._mock.stop()
            self._mock = None
        if not stats["replies"]:                   # team firmware: RPS:<GESTURE>, no replies
            where = f"{self.state.cfg.robot.host}:{self.state.cfg.robot.port}"
            if stats["sent"] == 0:
                self.chip_link.set("Could not send", "bad")
                self.link_hint.setText("Check that the laptop is on the same network as the robot.")
            elif simulated:
                self.chip_link.set("Simulated robot received RPS:PAPER" if received else "Simulated robot got "
                                   "nothing", "ok" if received else "bad")
                self.link_hint.setText("")
            else:
                self.chip_link.set("Sent RPS:PAPER", "info")
                self.link_hint.setText(f"Sent to {where}. This firmware does not reply, so delivery cannot be "
                                       f"confirmed here: check that the hand opened.")
            return
        if stats["acked"]:
            name = "Simulated robot" if simulated else "Robot"
            rtt = stats["rtt_median_ms"]
            self.chip_link.set(f"{name} replied {stats['acked']}/{stats['sent']}"
                               + (f" · {rtt:.1f} ms" if rtt is not None else ""), "ok")
            self.link_hint.setText("The robot last restarted from a power dip: give the servos their own supply."
                                   if stats["last_reset_reason"] == ESP_RST_BROWNOUT else "")
        else:
            self.chip_link.set(f"No reply (0/{stats['sent']})", "bad")
            self.link_hint.setText("Check: laptop on the RPS-HAND Wi-Fi, address and port match the firmware, "
                                   "Windows Firewall allows Python.")

    def shutdown(self):
        self.runner.kill()
