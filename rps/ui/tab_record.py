"""Record page: session form, recording controls with live motion preview, per-person progress."""

import threading
import time
from collections import Counter

import cv2
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from rps.dvs_emulator import PseudoDVS
from rps.recorder import INSTRUCTIONS, PROTOCOL, SessionRecorder, list_recordings
from rps.ui.base import Tab
from rps.ui.common import VideoView, to_pixmap
from rps.ui.style import BORDER, MUTED, OK, Card, button, caption, label, page_header, row, tip

TYPES = [("show", "Hold one gesture"), ("throws", "Countdown throws"), ("background", "No hand")]
TYPE_TIP = {
    "show": "Keep one gesture the whole time while moving, turning and changing distance.",
    "throws": "Play normally: pump 3 times, throw the chosen gesture, relax, repeat.",
    "background": "Keep the hand out of the green square; move your body and arm around it.",
}
TYPE_NAME = dict(TYPES)
DEFAULT_DURATION = {"show": 45.0, "throws": 45.0, "background": 45.0}


class RecordTab(Tab):
    title = "2  Record"
    uses_camera = True

    def __init__(self, main):
        super().__init__(main)
        self.view = VideoView(placeholder="Camera off")
        tip(self.view, "Live camera. Only the green square plus a small margin is saved, as lossless video.")
        self._pending = None          # recorder waiting for its countdown (set by UI, consumed by camera thread)
        self._countdown_until = 0.0
        self.countdown_s = 3.0
        self._active = None           # recorder currently recording (camera thread)
        self._stop_flag = False
        self._abort_flag = False
        self._finished = None         # (meta or None, error) handed back to the UI thread
        self._message_until = 0.0
        self._dvs = PseudoDVS(self.state.cfg.dvs)

        # --- session
        what = Card("Session", "Who is recording and what.")
        g = QGridLayout()
        g.setColumnStretch(1, 1)
        g.setVerticalSpacing(5)
        self.person = tip(QComboBox(), "Type a new name or pick an existing person. Each person should record all "
                                       "7 sessions in the checklist.")
        self.person.setEditable(True)
        self.person.currentTextChanged.connect(self._refresh_person)
        self.kind = QComboBox()
        for data, text in TYPES:
            self.kind.addItem(text, data)
            self.kind.setItemData(self.kind.count() - 1, TYPE_TIP[data], 3)   # per-item hover (ToolTipRole)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.label = tip(QComboBox(), "The gesture shown or thrown in this session.")
        for text in ("rock", "paper", "scissors"):
            self.label.addItem(text.capitalize(), text)
        self.label.currentIndexChanged.connect(self._update_instruction)
        self.duration = tip(QDoubleSpinBox(), "Recording length. 45 s is enough per session.")
        self.duration.setRange(3, 600)
        self.duration.setDecimals(0)
        self.duration.setSuffix(" s")
        self.throws = tip(QSpinBox(), "How many throws you plan (for the evaluation report).")
        self.throws.setRange(0, 100)
        self.throws.setValue(15)
        self.hand = tip(QComboBox(), "Which hand is used. Recording both hands across sessions helps.")
        for text in ("right", "left", "both"):
            self.hand.addItem(text.capitalize(), text)
        self.lighting = tip(QLineEdit(), "Optional note, e.g. lamp, daylight.")
        self.lighting.setPlaceholderText("optional")
        for i, (name, w) in enumerate((("Person", self.person), ("Type", self.kind), ("Gesture", self.label),
                                       ("Length", self.duration), ("Throws", self.throws), ("Hand", self.hand),
                                       ("Lighting", self.lighting))):
            g.addWidget(label(name, w.toolTip()), i, 0)
            g.addWidget(w, i, 1)
        what.body.addLayout(g)

        # --- recording
        rec = Card("Recording", "Read the instruction, press Record, follow the countdown.")
        self.instruction = QLabel("")
        self.instruction.setWordWrap(True)
        self.instruction.setStyleSheet("font-weight:600; color:#7fb0ff;")
        tip(self.instruction, "What to do during this recording.")
        rec.body.addWidget(self.instruction)
        self.start_btn = button("Record", "primary", "Start after a 3-second countdown.")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = button("Stop and save", tooltip="End now and keep what was recorded.")
        self.stop_btn.clicked.connect(lambda: setattr(self, "_stop_flag", True))
        self.abort_btn = button("Discard", tooltip="End now and delete this recording.")
        self.abort_btn.clicked.connect(lambda: setattr(self, "_abort_flag", True))
        rec.body.addLayout(row(self.start_btn, self.stop_btn, self.abort_btn))
        self.progress = tip(QProgressBar(), "Time recorded so far.")
        rec.body.addWidget(self.progress)
        self.status = caption("Ready", "Frames saved, frame rate, and frames lost if the computer was too busy.")
        rec.body.addWidget(self.status)
        pv = QHBoxLayout()
        self.preview = QLabel()
        self.preview.setFixedSize(96, 96)
        self.preview.setStyleSheet(f"background:#000; border:1px solid {BORDER};")
        tip(self.preview, "Motion image of the play zone. If it stays dark while you move, the light is too low.")
        pv.addWidget(self.preview)
        pv.addWidget(caption("Movement seen in the play zone"), 1)
        rec.body.addLayout(pv)

        # --- progress
        prog = Card("Progress", "Sessions recorded by this person.")
        self.checklist = QLabel()
        self.checklist.setWordWrap(True)
        tip(self.checklist, "The 7 recommended sessions per person. Green = recorded.")
        prog.body.addWidget(self.checklist)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Type", "Gesture", "Frames", "fps", "MB", "File"])
        for i, t in enumerate(("Session type", "Gesture", "Frames saved", "Measured frame rate", "Size on disk",
                               "File check: saved frames match the video")):
            self.table.horizontalHeaderItem(i).setToolTip(t)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(120)
        prog.body.addWidget(self.table)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 6, 0)
        for w in (what, rec, prog):
            pl.addWidget(w)
        pl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(500)
        body = QHBoxLayout()
        body.addWidget(self.view, 1)
        body.addWidget(scroll)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Record", "Labelled examples of each gesture, for training and testing."))
        page.addLayout(body, 1)
        self._kind_changed()
        self._reload_people()

    # ------------------------------------------------------------------ helpers
    def _reload_people(self):
        people = sorted({m.get("person", "") for m in list_recordings(self.state.recordings_root)} - {""})
        current = self.person.currentText()
        self.person.blockSignals(True)
        self.person.clear()
        self.person.addItems(people)
        self.person.setEditText(current)
        self.person.blockSignals(False)
        self._refresh_person()

    def _refresh_person(self, *_):
        person = self.person.currentText().strip()
        if not person:
            self.checklist.setText("Enter a name.")
            self.table.setRowCount(0)
            return
        recs = [m for m in list_recordings(self.state.recordings_root) if m.get("person") == person]
        done = Counter((m["type"], m["label"]) for m in recs)
        items = []
        for kind, lab in PROTOCOL:
            name = TYPE_NAME[kind] + ("" if kind == "background" else f": {lab}")
            n = done[(kind, lab)]
            items.append(f"<span style='color:{OK if n else MUTED}'>{name}{f' ({n})' if n else ''}</span>")
        minutes = sum(m.get("duration_s", 0) for m in recs) / 60
        mb = sum(m.get("size_mb", 0) for m in recs)
        complete = sum(1 for k in PROTOCOL if done[k])
        self.checklist.setText(f"{complete} of {len(PROTOCOL)} done · {minutes:.1f} min · {mb:.0f} MB<br>"
                               + "<br>".join(items))
        self.table.setRowCount(len(recs))
        for r, m in enumerate(recs):
            vals = [TYPE_NAME.get(m["type"], m["type"]), m["label"].capitalize() if m["type"] != "background" else "-",
                    m.get("frames", 0), f"{m.get('fps_measured', 0):.0f}", f"{m.get('size_mb', 0):.0f}",
                    "OK" if m.get("video_check_ok", True) else "Problem"]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))

    def _kind_changed(self, *_):
        kind = self.kind.currentData()
        self.kind.setToolTip(TYPE_TIP[kind])
        self.label.setEnabled(kind != "background")
        self.throws.setEnabled(kind == "throws")
        self.duration.setValue(DEFAULT_DURATION[kind])
        self._update_instruction()

    def _update_instruction(self, *_):
        kind = self.kind.currentData()
        lab = "background" if kind == "background" else self.label.currentData()
        self.instruction.setText(INSTRUCTIONS[kind].format(label=lab.upper()))

    def _set_recording_ui(self, recording: bool):
        self.start_btn.setEnabled(not recording)
        for w in (self.person, self.kind, self.label, self.duration, self.throws, self.hand):
            w.setEnabled(not recording)
        if not recording:
            self.label.setEnabled(self.kind.currentData() != "background")
            self.throws.setEnabled(self.kind.currentData() == "throws")
        self.main.lock_tabs(recording, self)

    # ------------------------------------------------------------------ UI actions
    def _start(self):
        try:
            rec = SessionRecorder(self.state.cfg, self.person.currentText(), self.kind.currentData(),
                                  self.label.currentData(), self.duration.value(), self.throws.value(),
                                  self.hand.currentData(), self.lighting.text(), "",
                                  self.state.recordings_root, dict(self.main.worker.info))
        except ValueError as e:
            QMessageBox.warning(self, "Cannot record", str(e))
            return
        if not self.main.worker.isRunning():
            self.main.start_camera()
        self._stop_flag = self._abort_flag = False
        self._countdown_until = time.perf_counter() + self.countdown_s
        self._dvs = PseudoDVS(self.state.cfg.dvs)
        self._pending = rec
        self._set_recording_ui(True)

    def _finish_async(self, rec):
        def work():
            try:
                self._finished = (rec.finish(), None)
            except Exception as e:  # disk full, codec failure...
                self._finished = (None, str(e))
        threading.Thread(target=work, name="finish", daemon=True).start()

    # ------------------------------------------------------------------ camera thread
    def processor(self, frame, src) -> dict:
        now = time.perf_counter()
        state, rec = "idle", None
        if self._abort_flag and (self._pending or self._active):
            (self._active or self._pending).abort()
            self._pending = self._active = None
            self._abort_flag = False
            self._finished = ("aborted", None)
        if self._pending is not None:
            rec, state = self._pending, "countdown"
            if now >= self._countdown_until:
                rec.start(frame, src.roi)
                self._active, self._pending = rec, None
        if self._active is not None:
            rec, state = self._active, "recording"
            if not rec.add(frame) or self._stop_flag:
                self._active = None
                self._stop_flag = False
                state = "saving"
                self._finish_async(rec)

        img = frame.bgr.copy()
        x, y, s = src.roi
        dvs_frame, stats = self._dvs.process(img[y:y + s, x:x + s], frame.t)
        cv2.rectangle(img, (x, y), (x + s, y + s), (0, 0, 255) if state == "recording" else (0, 220, 0), 2)
        if state == "recording":
            cv2.circle(img, (22, 22), 8, (0, 0, 255), -1)
            cv2.putText(img, "REC", (36, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif state == "countdown":
            cv2.putText(img, f"{max(0.0, self._countdown_until - now):.0f}", (x + s // 2 - 18, y + s // 2 + 18),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 3)
        out = {"display": img, "state": state, "events": stats.events,
               "dvs": dvs_frame.image if dvs_frame is not None else None}
        if rec is not None:
            out.update(elapsed=rec.elapsed, duration=rec.duration_s, frames=len(rec.rows),
                       dropped=rec.dropped, missed=rec.missed)
        return out

    # ------------------------------------------------------------------ UI thread
    def on_frame(self, p):
        self.view.show_image(p["display"])
        if p.get("dvs") is not None:
            self.preview.setPixmap(to_pixmap(cv2.applyColorMap(cv2.resize(p["dvs"], (96, 96),
                                   interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)))
        st = p.get("state", "idle")
        if st == "recording":
            self.progress.setValue(int(100 * p["elapsed"] / max(p["duration"], 1e-6)))
            lost = p["dropped"] + p["missed"]
            self.status.setText(f"Recording {p['elapsed']:.0f} / {p['duration']:.0f} s · {p['frames']} frames"
                                + (f" · {lost} lost" if lost else ""))
        elif st == "countdown":
            self.status.setText("Starting...")
            self.progress.setValue(0)
        elif st == "idle" and self._finished is None and self.start_btn.isEnabled() \
                and time.perf_counter() > self._message_until:
            text = f"Ready · camera {p.get('fps', 0):.0f} fps"
            if text != self.status.text():
                self.status.setText(text)
        if self._finished is not None:
            meta, err = self._finished
            self._finished = None
            self._message_until = time.perf_counter() + 8.0   # keep the result visible
            self._set_recording_ui(False)
            self.progress.setValue(100 if isinstance(meta, dict) else 0)
            if err:
                QMessageBox.critical(self, "Recording failed", err)
            elif meta == "aborted":
                self.status.setText("Discarded")
            elif meta is None:
                self.status.setText("Nothing recorded")
            else:
                check = "file check OK" if meta["video_check_ok"] else "file problem, record again"
                self.status.setText(f"Saved {meta['frames']} frames · {meta['fps_measured']:.0f} fps · "
                                    f"{meta['size_mb']} MB · {check}")
            self._reload_people()

    def on_activated(self):
        self._reload_people()
        if not self.main.worker.isRunning():
            self.main.start_camera()

    def shutdown(self):
        if self._active or self._pending:
            (self._active or self._pending).abort()
