"""
Bot tuning page: everything about the robot hand. Connection (real, simulated or off; commands;
address), a connection test, sending each command on its own, finger tuning (the team firmware's
ANGLE:<channel>,<angle>: one servo straight to an angle, confirmed by the robot), the hand's move
times and the Wi-Fi delay (ping), which the Play pages' Speed card uses.

While the page is open it keeps ONE connection to the robot (one socket, so every command comes
from the same port, as the team's own tool sends them). Every click is sent at once, in order, and
logged; anything the robot sends back is logged too.
"""

import threading
import time
from collections import deque

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QGridLayout, QHBoxLayout, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget

from rps.robot_link import (ANGLE_MAX, ANGLE_MIN, ESP_RST_BROWNOUT, FINGER_CHANNELS, TEXT_COMMAND, MockEsp, RobotLink,
                            encode_angle)
from rps.netping import ping_ms, summarize
from rps.ui.base import Tab
from rps.ui.common import ConfigForm, LogView
from rps.ui.style import Card, Chip, Collapsible, button, caption, label, page_header, row, tip

POSE_NAME = {"R": "Rock", "P": "Paper", "S": "Scissors", "N": "Ready"}
ANGLE_REPLY_S = 1.5              # how long to wait for the robot's confirmation of an ANGLE command


class BotTab(Tab):
    title = "2  Bot tuning"
    uses_camera = False
    robot_said = Signal(str)                     # text the robot sent back (from the link's thread)
    wifi_done = Signal(object)                   # ping summary dict, or an error message

    def __init__(self, main):
        super().__init__(main)
        self._link = None                        # this page's connection, while the page is open
        self._link_key = None                    # the settings it was opened with
        self._mock = None                        # this page's simulated robot, while the page is open
        self._mock_port = None
        self._check_id = 0                       # only the latest command's check updates the chip
        self._awaiting = deque()                 # ANGLE commands waiting for a confirmation: (channel, angle, t)
        self.robot_said.connect(self._robot_said)
        self.wifi_done.connect(self._wifi_result)

        conn = Card("Connection", "How the app reaches the robot hand. The laptop must be on the same network as "
                                  "the robot.")
        self.form = ConfigForm(self.state, "robot", keys=["mode", "protocol", "host", "port"])
        self.form.changed.connect(self._settings_changed)
        conn.body.addWidget(self.form)
        adv = ConfigForm(self.state, "robot", keys=["heartbeat_s", "ack_timeout_s"])
        conn.body.addWidget(Collapsible("Reference firmware options", adv,
                                        tooltip="Only for the reference firmware (numbered messages with replies)."))
        save = button("Save", "primary", "Save the robot settings (and every other change) to config.json.")
        save.clicked.connect(self.state.save)
        conn.body.addLayout(row(save))

        test = Card("Test", "Check the connection, or send one command at a time to see what the hand does.")
        self.test_btn = button("Test connection", tooltip="Team firmware: sends RPS:PAPER once, so the hand should "
                                                          "open (it cannot reply). Reference firmware: sends a "
                                                          "message and waits for its reply.")
        self.test_btn.clicked.connect(lambda: self._send("test"))
        self.chip = Chip("Not tested", "off", "Result of the last test or command.")
        test.body.addLayout(row(self.test_btn, self.chip))
        self.hint = caption("")
        test.body.addWidget(self.hint)
        self.pose_btns = {}
        for pose in ("R", "P", "S", "N"):
            b = button(POSE_NAME[pose], tooltip="")
            b.clicked.connect(lambda _c=False, p=pose: self._send(p))
            self.pose_btns[pose] = b
        test.body.addLayout(row(*self.pose_btns.values()))
        self.log = LogView(300)
        test.body.addWidget(self.log)

        fingers = self._make_finger_card()

        timing = Card("Move times", "How long the hand takes to move between poses. The Play pages' Speed card "
                                    "and Evaluate use them.")
        timing.body.addWidget(ConfigForm(self.state, "latency", keys=["servo_transition_ms"]))
        timing.body.addWidget(caption("Measure them with a slow-motion phone video (240 fps) of the laptop screen and "
                                      "the hand: frames from the robot's move appearing on screen to the hand "
                                      "stopping, times 4.2 ms."))

        wifi = Card("Wi-Fi delay", "How long a command takes to reach the robot over Wi-Fi, measured with ping (the "
                                   "ESP32 answers pings by itself). The Play pages' Speed card shows it.")
        self.wifi_btn = button("Measure Wi-Fi delay", tooltip="Pings the robot 10 times (about 2 seconds) and keeps "
                                                              "half the typical round trip as the one-way delay.")
        self.wifi_btn.clicked.connect(self._measure_wifi)
        self.wifi_chip = Chip("Not measured", "off", "The last measurement.")
        wifi.body.addLayout(row(self.wifi_btn, self.wifi_chip))
        self.wifi_hint = caption("")
        self.wifi_hint.setWordWrap(True)
        wifi.body.addWidget(self.wifi_hint)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 6, 0)
        cols = QHBoxLayout()
        a, b = QVBoxLayout(), QVBoxLayout()
        a.addWidget(conn)
        a.addWidget(fingers)
        a.addWidget(timing)
        a.addWidget(wifi)
        a.addStretch(1)
        b.addWidget(test, 1)
        cols.addLayout(a, 1)
        cols.addLayout(b, 1)
        pl.addLayout(cols)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Bot tuning", "The robot hand: connection, commands, finger tuning and tests."))
        page.addWidget(scroll, 1)
        self._settings_changed()

    # ------------------------------------------------------------------ finger tuning
    def _make_finger_card(self) -> Card:
        card = Card("Finger tuning", "Moves one servo straight to an angle, 0 = extended to 180 = folded, with the "
                                     "team firmware's ANGLE:<channel>,<angle> command. The robot confirms each move. "
                                     "Use it to find each finger's straight and folded angles.")
        self.finger_card = card
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setVerticalSpacing(6)
        self.angle_sliders, self.angle_spins, self.angle_btns = {}, {}, {}
        for r, (channel, name) in enumerate(FINGER_CHANNELS.items()):
            slider = tip(QSlider(Qt.Orientation.Horizontal), f"Channel {channel} ({name}): 0 = extended, 180 = "
                                                             f"folded.")
            slider.setRange(ANGLE_MIN, ANGLE_MAX)
            spin = tip(QSpinBox(), slider.toolTip())
            spin.setRange(ANGLE_MIN, ANGLE_MAX)
            spin.setSuffix(" °")
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            slider.sliderReleased.connect(lambda c=channel: self._moved(c))
            spin.editingFinished.connect(lambda c=channel: self._moved(c))
            send = button("Send", tooltip=f"Send ANGLE:{channel},<angle> once.")
            send.clicked.connect(lambda _c=False, c=channel: self._send_angle(c, self.angle_spins[c].value()))
            grid.addWidget(label(f"{channel} · {name}", slider.toolTip()), r, 0)
            grid.addWidget(slider, r, 1)
            grid.addWidget(spin, r, 2)
            grid.addWidget(send, r, 3)
            self.angle_sliders[channel], self.angle_spins[channel], self.angle_btns[channel] = slider, spin, send
        card.body.addLayout(grid)
        self.extend_all = button("Extend all (0°)", tooltip="Send 0 to every channel: the open hand.")
        self.extend_all.clicked.connect(lambda: self._send_all(ANGLE_MIN))
        self.fold_all = button("Fold all (180°)", tooltip="Send 180 to every channel: a fist.")
        self.fold_all.clicked.connect(lambda: self._send_all(ANGLE_MAX))
        self.live = tip(QCheckBox("Send as I move"), "Send a channel's angle when you let go of its slider or finish "
                                                     "typing, without pressing Send.")
        card.body.addLayout(row(self.extend_all, self.fold_all, self.live))
        self.tune_chip = Chip("Nothing sent", "off", "The robot's confirmation of the last angle.")
        card.body.addLayout(row(self.tune_chip))
        self.tune_hint = caption("")
        card.body.addWidget(self.tune_hint)
        return card

    def _load_angles(self):
        angles = list(self.state.cfg.robot.finger_angles) + [0] * len(FINGER_CHANNELS)
        for channel, spin in self.angle_spins.items():
            spin.setValue(int(min(max(angles[channel], ANGLE_MIN), ANGLE_MAX)))

    def _moved(self, channel: int):
        if self.live.isChecked():
            self._send_angle(channel, self.angle_spins[channel].value())

    def _send_all(self, angle: int):
        for channel, spin in self.angle_spins.items():
            spin.setValue(angle)
            self._send_angle(channel, angle)

    def _send_angle(self, channel: int, angle: int):
        cfg = self.state.cfg.robot
        if cfg.mode == "off" or cfg.protocol != "rps_text":
            return
        link = self._ensure_link()
        if link is None:
            return
        self._remember_angle(channel, angle)
        self.log.log(f"{time.strftime('%H:%M:%S')}  ANGLE:{channel},{angle}  ->  {self._target()}")
        if not link.send_raw(encode_angle(channel, angle)):
            self.tune_chip.set("Could not send", "bad")
            self.tune_hint.setText(f"{link.send_error}. Check the robot address and that the laptop is on its "
                                   f"network.")
            self.log.log(f"    could not send: {link.send_error}")
            return
        self._awaiting.append((channel, angle, time.perf_counter()))
        self.tune_chip.set("Waiting for the robot...", "info")
        QTimer.singleShot(int(ANGLE_REPLY_S * 1000) + 50, self._expire_angles)

    def _robot_said(self, text: str):
        """Anything the robot sends back. The oldest ANGLE still waiting takes it as its confirmation."""
        self.log.log(f"    <- {text}")
        if self._awaiting:
            self._awaiting.popleft()
            who = "Simulated robot" if self._mock is not None else "Robot"
            self.tune_chip.set(f"{who} confirmed: {text}", "ok")
            self.tune_hint.setText("")

    def _expire_angles(self):
        now, missed = time.perf_counter(), None
        while self._awaiting and now - self._awaiting[0][2] >= ANGLE_REPLY_S - 1e-3:
            missed = self._awaiting.popleft()
            self.log.log(f"    no confirmation for ANGLE:{missed[0]},{missed[1]} within {ANGLE_REPLY_S:.1f} s")
        if missed is not None:
            self.tune_chip.set(f"No confirmation for channel {missed[0]}", "bad")
            self.tune_hint.setText("The command went out but nothing came back: check the address and port, and "
                                   "that Windows Firewall lets Python receive (the confirmation comes back to it).")

    # ------------------------------------------------------------------ Wi-Fi delay
    def _measure_wifi(self):
        cfg = self.state.cfg.robot
        if cfg.mode == "off" or not self.wifi_btn.isEnabled():
            return
        host = "127.0.0.1" if cfg.mode == "simulated" else cfg.host
        self.wifi_btn.setEnabled(False)
        self.wifi_chip.set("Measuring...", "info")
        threading.Thread(target=self._wifi_job, args=(host,), daemon=True).start()

    def _wifi_job(self, host: str):
        try:
            result = summarize(ping_ms(host, count=10))
        except Exception as e:                     # no ping command, bad address, timeout...
            result = f"Ping could not run: {e}"
        try:
            self.wifi_done.emit(result)
        except RuntimeError:                       # the window closed meanwhile
            pass

    def _wifi_result(self, result):
        self.wifi_btn.setEnabled(self.state.cfg.robot.mode != "off")
        simulated = self.state.cfg.robot.mode == "simulated"
        if isinstance(result, str):
            self.wifi_chip.set("Could not measure", "bad")
            self.wifi_hint.setText(result)
            return
        if not result["answered"]:
            self.wifi_chip.set(f"No answer (0 of {result['sent']})", "bad")
            self.wifi_hint.setText("The robot did not answer: check that it is on, its address, and that the laptop "
                                   "is on the same network.")
            return
        one_way = result["one_way_ms"]
        lost = result["sent"] - result["answered"]
        self.wifi_chip.set(f"{one_way:.1f} ms one way (round trip {result['median_ms']:.1f} ms, slowest "
                           f"{result['max_ms']:.0f} ms" + (f", {lost} lost" if lost else "") + ")",
                           "warn" if result["slow"] or lost else "ok")
        self.log.log(f"{time.strftime('%H:%M:%S')}  Wi-Fi delay: {one_way:.1f} ms one way (round trip median "
                     f"{result['median_ms']:.1f} ms, slowest {result['max_ms']:.0f} ms, {lost} lost)")
        if simulated:
            self.wifi_hint.setText("Simulated robot: this computer, so this is not a Wi-Fi delay. It is not kept.")
            return
        cfg = self.state.cfg.latency
        if round(one_way, 1) != cfg.network_ms:
            cfg.network_ms = round(one_way, 1)
            self.state.mark_dirty()
        self.wifi_hint.setText(
            "Slow or uneven for a local network. The usual cause is the ESP32's Wi-Fi power saving, which delays "
            "each command by up to ~100 ms: add WiFi.setSleep(false); to its setup(). Also keep the robot close "
            "to the router (or the laptop's hotspot)." if result["slow"] else
            "Kept for the Speed card on the Play pages (save the settings to remember it).")

    def _remember_angle(self, channel: int, angle: int):
        cfg = self.state.cfg.robot
        angles = (list(cfg.finger_angles) + [0] * len(FINGER_CHANNELS))[:len(FINGER_CHANNELS)]
        if angles[channel] != angle:
            angles[channel] = angle
            cfg.finger_angles = angles
            self.state.mark_dirty()

    # ------------------------------------------------------------------ page life and settings
    def on_activated(self):
        self.form.refresh()
        self._load_angles()
        self._settings_changed()
        saved = self.state.cfg.latency.network_ms
        if saved and self.wifi_chip.text() == "Not measured":
            self.wifi_chip.set(f"Last measured: {saved:.1f} ms one way", "ok")

    def on_robot_taken(self, computer: str):
        if self._link is not None and self._mock is None:
            self._close_link()
            self.chip.set(f"Robot in use by {computer}", "warn")
            self.hint.setText(f"{computer} started driving the robot. Any command from here takes it back.")

    def on_deactivated(self):
        self._close_link()               # the Play pages open their own connection (and simulated robot)

    def _settings_changed(self, *_):
        cfg = self.state.cfg.robot
        text = cfg.protocol == "rps_text"
        for pose, b in self.pose_btns.items():
            command = TEXT_COMMAND.get(pose) if text else f"P,<n>,{pose},<ms>"
            b.setVisible(not (text and pose == "N"))            # the team firmware has no ready position
            b.setToolTip(f"Send {command} once." if command else "")
        off = cfg.mode == "off"
        for b in [self.test_btn, *self.pose_btns.values(), self.wifi_btn]:
            b.setEnabled(not off)
        if off:
            self.chip.set("Robot off", "off")
        self.finger_card.setEnabled(text and not off)
        if not text:
            self.tune_hint.setText("Finger tuning uses the team firmware's ANGLE command: choose Team firmware "
                                   "under Robot commands.")
        elif off:
            self.tune_hint.setText("The robot is off.")
        elif self.tune_hint.text().startswith(("Finger tuning uses", "The robot is off")):
            self.tune_hint.setText("")
        if self._link is not None and self._link_key != self._key():
            self._close_link()           # the next command opens one with the new settings

    def _target(self) -> str:
        cfg = self.state.cfg.robot
        return "simulated robot" if cfg.mode == "simulated" else f"{cfg.host}:{cfg.port}"

    def _key(self):
        cfg = self.state.cfg.robot
        return cfg.mode, cfg.protocol, cfg.host, cfg.port, cfg.heartbeat_s, cfg.ack_timeout_s

    def _ensure_link(self):
        """This page's connection: opened on the first command, kept until the page is left or the
        connection settings change."""
        cfg = self.state.cfg.robot
        if cfg.mode == "off":
            return None
        if self._link is not None and self._link_key != self._key():
            self._close_link()
        if self._link is None:
            host = None
            if cfg.mode == "simulated":
                if self._simulated() is None:
                    return None
                host = "127.0.0.1"
            self._link = RobotLink.from_config(cfg, host=host, on_text=self._emit_robot_said).start()
            self._link_key = self._key()
            if cfg.mode == "real":
                self.main.announce_robot(self._link.addr[0])   # testing from here takes the robot
        return self._link

    def _emit_robot_said(self, text: str):
        try:
            self.robot_said.emit(text)
        except RuntimeError:                       # the window is closing
            pass

    def _close_link(self):
        if self._link is not None:
            self._link.stop(send_ready=False)
            self._link = None
        self._awaiting.clear()
        if self._mock is not None:
            self._mock.stop()
            self._mock = None

    def _simulated(self):
        """This page's simulated robot, started on first use and stopped with the connection."""
        if self._mock is None:
            port = self.state.cfg.robot.port
            try:
                self._mock = MockEsp(port=port, verbose=False).start()
                self._mock_port = port
            except OSError:
                self.chip.set("Simulated robot failed", "bad")
                self.hint.setText(f"Port {port} is already in use on this computer, probably by a running game or "
                                  f"another simulated robot. Stop it and try again.")
                return None
        return self._mock

    # ------------------------------------------------------------------ test and single commands
    def _send(self, what: str):
        """what: 'test' or a pose letter. Sends at once over the page's connection, then reports."""
        link = self._ensure_link()
        if link is None:
            return
        cfg = self.state.cfg.robot
        pose = ("N" if cfg.protocol == "ack" else "P") if what == "test" else what
        before = link.stats()
        received = self._mock.received if self._mock is not None else 0
        seq = link.send_pose(pose)
        sent = TEXT_COMMAND.get(pose, "(nothing: no ready command)") if cfg.protocol == "rps_text" \
            else f"P,{seq},{pose}"
        self.log.log(f"{time.strftime('%H:%M:%S')}  {sent}  ->  {self._target()}")
        self._check_id += 1
        check = (self._check_id, what, pose, before, received)
        self.chip.set("Sending...", "info")
        QTimer.singleShot(1500 if what == "test" else 300, lambda: self._result(*check))

    def _result(self, check_id, what, pose, before, received_before):
        if check_id != self._check_id or self._link is None:
            return                                # a newer command (or leaving the page) took over
        stats = self._link.stats()
        simulated = self._mock is not None
        name = POSE_NAME[pose]
        if stats["sent"] == before["sent"]:
            self.chip.set("Could not send", "bad")
            self.hint.setText(f"{self._link.send_error or 'Nothing was sent'}. Check that the laptop is on the same "
                              f"network as the robot, and the address.")
        elif not stats["replies"]:                 # team firmware: RPS:<GESTURE>, no replies
            if simulated:
                got = self._mock.received > received_before
                self.chip.set(f"Simulated robot received {TEXT_COMMAND.get(pose)}" if got else
                              "Simulated robot got nothing", "ok" if got else "bad")
                self.hint.setText("")
            else:
                self.chip.set(f"Sent {TEXT_COMMAND.get(pose)}", "info")
                self.hint.setText(f"This firmware does not reply, so delivery cannot be confirmed here: check that "
                                  f"the hand shows {name.lower()}. Every command sent is in the log.")
        elif stats["acked"] > before["acked"]:
            rtt = stats["rtt_median_ms"]
            robot = "Simulated robot" if simulated else "Robot"
            self.chip.set(f"{robot} replied" + (f" · {rtt:.1f} ms" if rtt else ""), "ok")
            self.hint.setText("The robot last restarted from a power dip: give the servos their own supply."
                              if stats["last_reset_reason"] == ESP_RST_BROWNOUT else "")
        else:
            self.chip.set("No reply", "bad")
            self.hint.setText("Check: laptop on the robot's Wi-Fi, address and port match the firmware, Windows "
                              "Firewall allows Python.")

    def shutdown(self):
        self._close_link()
