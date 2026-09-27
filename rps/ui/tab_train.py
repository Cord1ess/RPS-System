"""Train page: training options, live accuracy curves and per-person results."""

import json
import os
import re

import pyqtgraph as pg
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QProgressBar, QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from rps.ui.base import Tab
from rps.ui.common import LogView, ProcessRunner
from rps.ui.style import PANEL, Card, Collapsible, button, caption, label, page_header, row, static_plot, tip

EPOCH = re.compile(r"Epoch \[(\d+)/(\d+)\] loss ([\d.]+) acc\s+([\d.]+)%(?: \|\| val balanced acc\s+([\d.]+)%)?")
FOLD = re.compile(r"\[train:holdout=([^\]]+)\]|\[train:all\]")
LOPO_ROW = re.compile(r"^\s{2}(.+?)\s+balanced acc\s+([\d.]+)%")
MEAN_ROW = re.compile(r"^\s{2}mean\s+([\d.]+)%\s+\+/-\s+([\d.]+)%")


class TrainTab(Tab):
    title = "4  Train"

    def __init__(self, main):
        super().__init__(main)
        self.runner = ProcessRunner(self)
        self.log = LogView()
        self.runner.line.connect(self._line)
        self.runner.finished.connect(self._done)

        what = Card("Training", "Train the motion model on the training images from the Dataset page. Not needed "
                                "to use Dextra's model as downloaded.")
        self.start_from = tip(QComboBox(), "Dextra's model: keep what it learned from about 20 people and adapt "
                                           "its last layers to this webcam (recommended: on one person's recordings "
                                           "this lifted scissors from 65% to 91%). Nothing: a new network; needs "
                                           "far more recordings.")
        self.start_from.addItem("Dextra's model: tune it on your recordings", "dextra")
        self.start_from.addItem("Nothing: train a new network from scratch", "scratch")
        self.start_from.currentIndexChanged.connect(self._start_from_changed)
        sf = QHBoxLayout()
        sf.addWidget(label("Start from", self.start_from.toolTip()))
        sf.addWidget(self.start_from, 1)
        what.body.addLayout(sf)
        self.mode_lopo = tip(QRadioButton("Accuracy check"), "Trains once per person, each time leaving that person "
                                                             "out, and reports accuracy on them. Saves no model.")
        self.mode_hold = tip(QRadioButton("Train, test on one person"), "Trains on everyone except the chosen "
                                                                        "person, reports accuracy on them, saves the "
                                                                        "model.")
        self.mode_all = tip(QRadioButton("Train on everyone"), "Uses all people for the final model. No accuracy "
                                                               "check. Do this last.")
        self.mode_hold.setChecked(True)
        group = QButtonGroup(self)
        for b in (self.mode_lopo, self.mode_hold, self.mode_all):
            group.addButton(b)
            what.body.addWidget(b)
        self.val_person = tip(QComboBox(), "The person left out and used to measure accuracy.")
        pr = QHBoxLayout()
        pr.addWidget(label("Test person", self.val_person.toolTip()))
        pr.addWidget(self.val_person, 1)
        what.body.addLayout(pr)
        adv = QWidget()
        ag = QGridLayout(adv)
        ag.setContentsMargins(0, 0, 0, 0)
        ag.setColumnStretch(1, 1)
        self.epochs = tip(QSpinBox(), "Passes over all training images. Defaults: 10 when tuning Dextra, 30 from "
                                      "scratch.")
        self.epochs.setRange(1, 500)
        self.batch = tip(QSpinBox(), "Images per training step.")
        self.batch.setRange(8, 2048)
        self.batch.setValue(128)
        self.lr = tip(QDoubleSpinBox(), "Step size of training. Defaults: 0.0003 when tuning Dextra, 0.001 from "
                                        "scratch.")
        self.lr.setDecimals(5)
        self.lr.setRange(1e-5, 1e-1)
        self.lr.setSingleStep(1e-4)
        self.event_counts = tip(QLineEdit(), "Use only images built with these movement-per-image values. Empty = "
                                             "all.")
        self.event_counts.setPlaceholderText("all")
        self.output = tip(QLineEdit(), "Where the trained model is saved. Play lists it by what it is.")
        for i, (name, w) in enumerate((("Passes", self.epochs), ("Batch size", self.batch), ("Step size", self.lr),
                                       ("Movement per image", self.event_counts), ("Save as", self.output))):
            ag.addWidget(label(name, w.toolTip()), i, 0)
            ag.addWidget(w, i, 1)
        what.body.addWidget(Collapsible("Options", adv))
        self._start_from_changed()
        self.start_btn = button("Train", "primary", "Start training.")
        self.start_btn.clicked.connect(self._start)
        cancel = button("Cancel", tooltip="Stop training.")
        cancel.clicked.connect(self.runner.kill)
        what.body.addLayout(row(self.start_btn, cancel))

        prog = Card("Progress", "Accuracy after each pass. The test-person line is the one that matters.")
        self.result = QLabel("")
        self.result.setStyleSheet("font-size:11pt; font-weight:600;")
        self.result.setWordWrap(True)
        tip(self.result, "Headline result of the last run.")
        prog.body.addWidget(self.result)
        self.progress = tip(QProgressBar(), "Passes completed in the current training run.")
        prog.body.addWidget(self.progress)
        self.fold_label = caption("Not started", "What is being trained right now.")
        prog.body.addWidget(self.fold_label)
        pg.setConfigOptions(antialias=True)
        self.acc_plot = static_plot(pg.PlotWidget())
        self.acc_plot.setBackground(PANEL)
        self.acc_plot.setLabel("left", "accuracy")
        self.acc_plot.setLabel("bottom", "pass")
        self.acc_plot.setYRange(0, 1)
        self.acc_plot.showGrid(y=True, alpha=0.15)
        self.acc_plot.addLegend(offset=(5, 5), labelTextSize="8pt")
        tip(self.acc_plot, "Training people: how well the model fits the people it learns from. Test person: "
                           "accuracy on the person it never saw.")
        self.train_curve = self.acc_plot.plot(pen=pg.mkPen("#e0524a", width=1.6), name="training people")
        self.val_curve = self.acc_plot.plot(pen=pg.mkPen("#2f6fed", width=2.4), name="test person")
        prog.body.addWidget(self.acc_plot, 1)
        self.folds = QTableWidget(0, 2)
        self.folds.setHorizontalHeaderLabels(["Person left out", "Accuracy"])
        self.folds.horizontalHeaderItem(0).setToolTip("The person the model was tested on and never trained on.")
        self.folds.horizontalHeaderItem(1).setToolTip("Accuracy on that person, each gesture counting equally.")
        tip(self.folds, "One row per person: accuracy when that person was left out of training.")
        self.folds.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.folds.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.folds.setMaximumHeight(140)
        prog.body.addWidget(self.folds)
        self.use_btn = button("Use in Play", tooltip="Make Play use the model just trained, and save that setting.")
        self.use_btn.clicked.connect(self._use_model)
        prog.body.addLayout(row(self.use_btn))
        self.output_box = Collapsible("Output", self.log, tooltip="Full training output.")
        prog.body.addWidget(self.output_box)
        self._reset_curves()

        left = QVBoxLayout()
        left.addWidget(what)
        left.addStretch(1)
        body = QHBoxLayout()
        body.addLayout(left, 2)
        body.addWidget(prog, 3)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Train", "Accuracy is always measured on a person the model did not learn from."))
        page.addLayout(body, 1)

    def on_activated(self):
        people = self._people()
        current = self.val_person.currentText()
        self.val_person.clear()
        self.val_person.addItems(people)
        if current in people:
            self.val_person.setCurrentText(current)
        self.use_btn.setEnabled(os.path.exists(self.output.text()))
        if not people:
            self.fold_label.setText("No training images yet: build them on the Dataset page.")

    def _start_from_changed(self, *_):
        dextra = self.start_from.currentData() == "dextra"
        self.epochs.setValue(10 if dextra else 30)
        self.lr.setValue(3e-4 if dextra else 1e-3)
        self.output.setText("models/dextra_tuned.pth" if dextra else "models/motion_cnn_v3.pth")

    def _people(self):
        path = os.path.join(self.state.frames_root, "index.json")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return sorted({e["person"] for e in json.load(f)["files"]})

    def _reset_curves(self):
        self.hist = {"train": [], "val": []}
        self.train_curve.setData([])
        self.val_curve.setData([])

    def _start(self):
        if self.runner.running():
            return
        if self.start_from.currentData() == "dextra" and not os.path.exists("models/dextra_roshambo.pth"):
            self.result.setText("Download Dextra's model first (Play page), or start from nothing.")
            return
        args = ["train.py", "--frames", self.state.frames_root, "--start_from", self.start_from.currentData(),
                "--config", self.state.run_config(), "--epochs", self.epochs.value(),
                "--batch_size", self.batch.value(), "--lr", self.lr.value(), "--output", self.output.text(),
                "--metrics_plot", os.path.splitext(self.output.text())[0] + "_metrics.png"]
        if self.event_counts.text().strip():
            args += ["--event_counts", self.event_counts.text().replace(" ", "")]
        if self.mode_lopo.isChecked():
            args.append("--lopo")
        elif self.mode_all.isChecked():
            args.append("--all")
        elif self.val_person.currentText():
            args += ["--val_person", self.val_person.currentText()]
        self.folds.setRowCount(0)
        self.result.setText("")
        self._reset_curves()
        self.start_btn.setEnabled(False)
        self.runner.start(args)

    def _line(self, line: str):
        self.log.log(line)
        m = FOLD.search(line)
        if m:
            self._reset_curves()
            self.fold_label.setText(f"Training without {m.group(1)}" if m.group(1) else "Training on everyone")
        m = EPOCH.search(line)
        if m:
            ep, total = int(m.group(1)), int(m.group(2))
            self.progress.setMaximum(total)
            self.progress.setValue(ep)
            self.hist["train"].append(float(m.group(4)) / 100)
            self.train_curve.setData(self.hist["train"])
            if m.group(5):
                self.hist["val"].append(float(m.group(5)) / 100)
                self.val_curve.setData(self.hist["val"])
        m = LOPO_ROW.match(line)
        if m:
            r = self.folds.rowCount()
            self.folds.insertRow(r)
            self.folds.setItem(r, 0, QTableWidgetItem(m.group(1)))
            self.folds.setItem(r, 1, QTableWidgetItem(f"{m.group(2)}%"))
        m = MEAN_ROW.match(line)
        if m:
            self.result.setText(f"Expected accuracy on a new person: {m.group(1)}% (± {m.group(2)}%)")

    def _done(self, code):
        self.start_btn.setEnabled(True)
        saved = code == 0 and os.path.exists(self.output.text()) and not self.mode_lopo.isChecked()
        self.use_btn.setEnabled(saved)
        if self.runner.cancelled:
            self.fold_label.setText("Cancelled")
        elif code != 0:
            self.fold_label.setText("Failed: see Output")
            self.output_box.toggle.setChecked(True)
        elif self.hist["val"] and not self.mode_lopo.isChecked():
            self.result.setText(f"Accuracy on {self.val_person.currentText()}, who the model never saw: "
                                f"{self.hist['val'][-1] * 100:.1f}%")
        elif saved:
            self.result.setText("Model saved")

    def _use_model(self):
        self.state.cfg.cnn.model_path = self.output.text()
        self.state.save()
        self.result.setText(f"Play now uses {os.path.basename(self.output.text())}")

    def shutdown(self):
        self.runner.kill()
