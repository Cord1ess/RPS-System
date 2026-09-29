"""Dataset page: recordings, build training images, check the labels."""

import json
import os
import re
import shutil

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QProgressBar, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from rps.recorder import list_recordings
from rps.ui.base import Tab
from rps.ui.common import LogView, ProcessRunner, open_folder, to_pixmap
from rps.ui.style import Card, Collapsible, button, caption, label, page_header, row, tip

BUILD_PROGRESS = re.compile(r"\[build\] \((\d+)/(\d+)\)")
CLASS_TITLE = ["Rock", "Paper", "Scissors", "None"]
TYPE_NAME = {"show": "Hold gesture", "throws": "Throws", "background": "No hand"}


def review_grid(frames: np.ndarray, cols: int = 8, scale: int = 2) -> np.ndarray:
    tiles = []
    for f in frames:
        t = cv2.applyColorMap(cv2.resize(f, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST),
                              cv2.COLORMAP_INFERNO)
        tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60)))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def _table(headers, tooltips):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    for i, text in enumerate(tooltips):
        t.horizontalHeaderItem(i).setToolTip(text)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    return t


class DatasetTab(Tab):
    title = "4  Dataset"

    def __init__(self, main):
        super().__init__(main)
        self.runner = ProcessRunner(self)
        self.log = LogView()
        self.runner.line.connect(self._line)
        self.runner.finished.connect(self._built)
        self._recs = []

        # --- recordings
        recs = Card("Recordings", "Everything recorded on the Record page.")
        self.rec_table = _table(["Person", "Type", "Gesture", "Frames", "fps", "Seconds", "MB", "File"],
                                ["Who recorded it", "Session type", "Gesture", "Frames saved", "Measured frame rate",
                                 "Length", "Size on disk", "File check: saved frames match the video"])
        tip(self.rec_table, "Every recording. Select rows to open or delete them.")
        self.rec_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.rec_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.rec_table.setMinimumHeight(170)
        recs.body.addWidget(self.rec_table)
        self.rec_totals = caption("", "Totals across all recordings, and free space on this drive.")
        recs.body.addWidget(self.rec_totals)
        buttons = []
        for text, fn, t in (("Refresh", self._refresh_recordings, "Reload the list from disk."),
                            ("Delete", self._delete, "Permanently delete the selected recordings."),
                            ("Open folder", self._open_folder, "Open the selected recording's folder.")):
            b = button(text, tooltip=t)
            b.clicked.connect(fn)
            buttons.append(b)
        recs.body.addLayout(row(*buttons))

        # --- build
        build = Card("Build training images", "Convert every recording into the Dextra views that Dextra Tuned "
                                              "learns from (Train page). About a minute per person.")
        self.use_mp = tip(QCheckBox("Check labels with Mediapipe"),
                          "Drops images where the hand clearly shows a different gesture than the session label, "
                          "and labels each throw. Recommended.")
        self.use_mp.setChecked(True)
        self.slow_cam = tip(QCheckBox("Include a slower-camera copy"),
                            "Adds a copy made from every second frame, so the model copes with dropped frames.")
        self.slow_cam.setChecked(True)
        build.body.addWidget(self.use_mp)
        build.body.addWidget(self.slow_cam)
        adv = QWidget()
        ag = QGridLayout(adv)
        ag.setContentsMargins(0, 0, 0, 0)
        ag.setColumnStretch(1, 1)
        self.event_counts = tip(QComboBox(), "Movement collected per image. Each recording is built at every value "
                                             "listed (Dextra used 500-2000).")
        self.event_counts.setEditable(True)
        self.event_counts.addItems(["750,1500,3000", "1500", "1500,3000"])
        self.clean_conf = tip(QDoubleSpinBox(), "Drop an image only if Mediapipe is at least this sure it "
                                                "shows a different gesture (0-1).")
        self.clean_conf.setRange(0.3, 1.0)
        self.clean_conf.setSingleStep(0.05)
        self.clean_conf.setValue(0.75)
        self.lookahead = tip(QDoubleSpinBox(), "A blurred image during a throw is given the gesture seen up to "
                                               "this long afterwards.")
        self.lookahead.setRange(0.0, 2.0)
        self.lookahead.setSingleStep(0.05)
        self.lookahead.setValue(0.30)
        self.lookahead.setSuffix(" s")
        for i, (name, w) in enumerate((("Movement per image", self.event_counts), ("Mediapipe certainty", self.clean_conf),
                                       ("Throw look-ahead", self.lookahead))):
            ag.addWidget(label(name, w.toolTip()), i, 0)
            ag.addWidget(w, i, 1)
        build.body.addWidget(Collapsible("Build options", adv))
        self.build_btn = button("Build", "primary", "Build training images from all recordings.")
        self.build_btn.clicked.connect(self._build)
        cancel = button("Cancel", tooltip="Stop the build.")
        cancel.clicked.connect(self.runner.kill)
        build.body.addLayout(row(self.build_btn, cancel))
        self.progress = tip(QProgressBar(), "Recordings converted so far.")
        build.body.addWidget(self.progress)
        self.build_status = caption("")
        build.body.addWidget(self.build_status)

        # --- check
        check = Card("Check the labels", "Training images per person and gesture, and random samples to inspect.")
        self.frames_table = _table(["Person"] + CLASS_TITLE + ["Total"],
                                   ["Person"] + [f"Training images labelled {c.lower()}" for c in CLASS_TITLE]
                                   + ["All training images"])
        tip(self.frames_table, "Training images per person and gesture, after the label check.")
        self.frames_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.frames_table.setMaximumHeight(150)
        check.body.addWidget(self.frames_table)
        self.rev_person = tip(QComboBox(), "Person to sample from.")
        self.rev_class = tip(QComboBox(), "Gesture to sample.")
        for c in CLASS_TITLE:
            self.rev_class.addItem(c)
        show = button("Show 48 samples", tooltip="Random training images for this person and gesture. Each should "
                                                 "clearly show that gesture moving.")
        show.clicked.connect(self._review)
        check.body.addLayout(row(self.rev_person, self.rev_class, show))
        self.grid = QLabel("Build first, then pick a person and gesture.")
        self.grid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grid.setObjectName("help")
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setWidget(self.grid)
        check.body.addWidget(self.grid_scroll, 1)

        left = QVBoxLayout()
        left.addWidget(recs, 1)
        left.addWidget(build)
        self.output = Collapsible("Build output", self.log, tooltip="Full output of the last build.")
        left.addWidget(self.output)
        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addWidget(check, 1)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Dataset", "Turn recordings into training images and check them before "
                                              "training."))
        page.addLayout(body, 1)

    def on_activated(self):
        self._refresh_recordings()
        self._refresh_frames()

    # ------------------------------------------------------------------ recordings
    def _refresh_recordings(self):
        recs = list_recordings(self.state.recordings_root)
        self._recs = recs
        self.rec_table.setRowCount(len(recs))
        for r, m in enumerate(recs):
            gesture = "-" if m.get("type") == "background" else m.get("label", "").capitalize()
            vals = [m.get("person"), TYPE_NAME.get(m.get("type"), m.get("type")), gesture, m.get("frames", 0),
                    f"{m.get('fps_measured', 0):.0f}", f"{m.get('duration_s', 0):.0f}", f"{m.get('size_mb', 0):.0f}",
                    "OK" if m.get("video_check_ok", True) else "Problem"]
            for c, v in enumerate(vals):
                self.rec_table.setItem(r, c, QTableWidgetItem(str(v)))
        people = {m.get("person") for m in recs}
        minutes = sum(m.get("duration_s", 0) for m in recs) / 60
        gb = sum(m.get("size_mb", 0) for m in recs) / 1000
        root = self.state.recordings_root if os.path.isdir(self.state.recordings_root) else "."
        free = shutil.disk_usage(os.path.abspath(root)).free / 1e9
        self.rec_totals.setText(f"{len(recs)} sessions · {len(people)} people · {minutes:.1f} min · {gb:.2f} GB "
                                f"used · {free:.0f} GB free")

    def _selected_recs(self):
        rows = sorted({i.row() for i in self.rec_table.selectedIndexes()})
        return [self._recs[r] for r in rows]

    def _delete(self):
        recs = self._selected_recs()
        if not recs:
            return
        names = "\n".join(f"{m['person']}: {TYPE_NAME.get(m['type'], m['type'])} {m['label']}" for m in recs)
        if QMessageBox.question(self, "Delete recordings", f"Permanently delete {len(recs)} recording(s)?\n\n{names}") \
                != QMessageBox.StandardButton.Yes:
            return
        for m in recs:
            shutil.rmtree(m["dir"], ignore_errors=True)
        self._refresh_recordings()

    def _open_folder(self):
        recs = self._selected_recs()
        path = recs[0]["dir"] if recs else self.state.recordings_root
        open_folder(path)

    # ------------------------------------------------------------------ build
    def _line(self, line: str):
        self.log.log(line)
        m = BUILD_PROGRESS.search(line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            self.progress.setMaximum(n)
            self.progress.setValue(i)
            self.build_status.setText(f"Recording {i} of {n}")

    def _build(self):
        if self.runner.running():
            return
        counts = self.event_counts.currentText().replace(" ", "")
        args = ["build_dataset.py", "--recordings", self.state.recordings_root, "--out", self.state.frames_root,
                "--config", self.state.run_config(), "--event_counts", counts,
                "--frame_skips", "1,2" if self.slow_cam.isChecked() else "1",
                "--clean_conf", f"{self.clean_conf.value():.2f}", "--lookahead", f"{self.lookahead.value():.2f}"]
        if not self.use_mp.isChecked():
            args.append("--no_mp")
        self.progress.setValue(0)
        self.build_btn.setEnabled(False)
        self.build_status.setText("Starting")
        self.runner.start(args)

    def _built(self, code: int):
        self.build_btn.setEnabled(True)
        if self.runner.cancelled:
            self.build_status.setText("Cancelled")
        else:
            self.build_status.setText("Done" if code == 0 else "Failed: see Build output")
        if code != 0 and not self.runner.cancelled:
            self.output.toggle.setChecked(True)
        self._refresh_frames()

    # ------------------------------------------------------------------ check
    def _index(self):
        path = os.path.join(self.state.frames_root, "index.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _refresh_frames(self):
        index = self._index()
        self.frames_table.setRowCount(0)
        if index is None:
            return
        counts = {}
        for e in index["files"]:
            data = np.load(os.path.join(self.state.frames_root, e["path"]))
            per = counts.setdefault(e["person"], np.zeros(4, int))
            per += np.bincount(data["labels"].astype(int), minlength=4)[:4]
        self.frames_table.setRowCount(len(counts))
        for r, (person, c) in enumerate(sorted(counts.items())):
            for col, v in enumerate([person] + c.tolist() + [int(c.sum())]):
                self.frames_table.setItem(r, col, QTableWidgetItem(str(v)))
        self.rev_person.clear()
        self.rev_person.addItems(sorted(counts))

    def _review(self):
        index = self._index()
        if index is None or not self.rev_person.currentText():
            return
        lab = self.rev_class.currentIndex()
        frames = []
        for e in index["files"]:
            if e["person"] != self.rev_person.currentText():
                continue
            data = np.load(os.path.join(self.state.frames_root, e["path"]))
            frames.append(data["frames"][data["labels"] == lab])
        frames = np.concatenate(frames) if frames else np.zeros((0, 64, 64), np.uint8)
        if len(frames) == 0:
            self.grid.setText("No images for that person and gesture.")
            return
        pick = np.random.default_rng().choice(len(frames), min(48, len(frames)), replace=False)
        pix = to_pixmap(review_grid(frames[pick]))
        width = self.grid_scroll.viewport().width() - 8
        if pix.width() > width > 100:
            pix = pix.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
        self.grid.setPixmap(pix)

    def shutdown(self):
        self.runner.kill()
