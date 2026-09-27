"""Evaluate page: Dextra model check (no training), compare recognition methods, results."""

import json
import os
import re

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from rps.recorder import list_recordings
from rps.ui.base import Tab
from rps.ui.common import LogView, ProcessRunner
from rps.ui.style import BAD, OK, WARN, Card, Collapsible, button, caption, label, page_header, row, tip

COLUMNS = [("setting", "Setting", "str", "Only for comparisons: the value being compared."),
           ("source", "Method", "str", "Which recognition method was replayed."),
           ("show_hold_acc", "Held correct", "pct", "Share of time the decision matched the gesture held."),
           ("show_switches_per_min", "Changes/min", "num", "Decision changes while one gesture was held. "
                                                            "Should be about 0."),
           ("show_cnn_frame_acc", "Motion model", "pct", "Motion model accuracy on single images."),
           ("show_mp_frame_acc", "Hand tracker", "pct", "Hand tracker accuracy on single frames."),
           ("throw_acc", "Throws correct", "pct", "Countdown throws where the decision was right."),
           ("throw_commits", "Throws decided", "str", "Throws the system decided."),
           ("throws_planned", "Throws recorded", "str", "Throws planned when recording."),
           ("commit_vs_throw_ms", "Decided vs throw (ms)", "num", "Decision time relative to the throw: when "
                                                                   "paper or scissors first shows, or when a rock "
                                                                   "throw stops moving down. Negative = before."),
           ("visible_ms", "Robot visible (ms)", "num", "Decision + camera delay + servo time. Target 200 or less."),
           ("bg_false_commits_per_min", "False moves/min", "num", "Decisions made with no hand present.")]
METHODS = [("mediapipe", "Hand tracker"), ("cnn", "Motion model"), ("fused", "Both")]


class EvaluateTab(Tab):
    title = "5  Evaluate"

    def __init__(self, main):
        super().__init__(main)
        self.runner = ProcessRunner(self)
        self.log = LogView()
        self.runner.line.connect(self._line)
        self.runner.finished.connect(self._done)
        self._job = ""

        pick = Card("Recordings", "Which recordings to test on. Use a person the model did not learn from.")
        self.target = tip(QComboBox(), "Recordings used for the test.")
        pick.body.addWidget(self.target)

        dextra = Card("Dextra model check", "Does Dextra's pretrained model recognise your gestures on this camera? "
                                            "No training.")
        self.transfer_btn = button("Run check", "primary", "Replays your 'Hold one gesture' and 'No hand' recordings "
                                                           "through Dextra's model under every rotation, flip, "
                                                           "movement-per-image and zoom, ranks the settings, and saves "
                                                           "pictures of your images next to Dextra's (a few minutes).")
        self.transfer_btn.clicked.connect(self._transfer)
        sheets = button("Open pictures", tooltip="Your motion images next to Dextra's, per gesture, for the best "
                                                 "setting.")
        sheets.clicked.connect(self._open_sheets)
        dextra.body.addLayout(row(self.transfer_btn, sheets))
        self.transfer_result = caption("", "Best setting found by the last check.")
        dextra.body.addWidget(self.transfer_result)
        self.apply_btn = button("Use this setting", tooltip="Apply the best turn, mirror, movement per image and "
                                                            "motion sensitivity to Play. Save on Settings to keep it.")
        self.apply_btn.clicked.connect(self._apply_best)
        self.apply_btn.setVisible(False)
        dextra.body.addLayout(row(self.apply_btn))
        self._best = None

        compare = Card("Compare recognition methods", "Replays recordings through exactly what Play runs and "
                                                       "scores each method.")
        self.src = {}
        boxes = []
        for data, text in METHODS:
            cb = tip(QCheckBox(text), f"Include '{text}' in the comparison.")
            cb.setChecked(True)
            self.src[data] = cb
            boxes.append(cb)
        compare.body.addLayout(row(*boxes))
        self.model_label = caption("", "Motion model file used for 'Motion model' and 'Both'.")
        compare.body.addWidget(self.model_label)
        adv = QWidget()
        ag = QGridLayout(adv)
        ag.setContentsMargins(0, 0, 0, 0)
        ag.setColumnStretch(1, 1)
        self.overrides = tip(QLineEdit(), "Temporary setting changes for this run only, e.g. vote.k=3; "
                                          "decision.still_frames=1")
        self.overrides.setPlaceholderText("vote.k=3; decision.still_frames=1")
        self.sweep = tip(QLineEdit(), "Run once per value and list the results side by side, e.g. vote.k=1,2,3")
        self.sweep.setPlaceholderText("vote.k=1,2,3")
        ag.addWidget(label("Temporary settings", self.overrides.toolTip()), 0, 0)
        ag.addWidget(self.overrides, 0, 1)
        ag.addWidget(label("Compare values", self.sweep.toolTip()), 1, 0)
        ag.addWidget(self.sweep, 1, 1)
        compare.body.addWidget(Collapsible("Settings to try", adv))
        self.run_btn = button("Run comparison", "primary", "Replay the recordings and score each selected method.")
        self.run_btn.clicked.connect(self._run)
        cancel = button("Cancel", tooltip="Stop the running check or comparison.")
        cancel.clicked.connect(self.runner.kill)
        compare.body.addLayout(row(self.run_btn, cancel))

        results = Card("Results", "Hover a column title for its meaning.")
        self.verdict = QLabel("Nothing run yet")
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet("font-size:11pt; font-weight:600;")
        tip(self.verdict, "Verdict: use both methods only if they beat the hand tracker alone on people the model "
                          "did not learn from.")
        results.body.addWidget(self.verdict)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        for i, c in enumerate(COLUMNS):
            self.table.horizontalHeaderItem(i).setToolTip(c[3])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        tip(self.table, "One row per method (and per value when comparing settings). Hover a column title for "
                        "its meaning.")
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results.body.addWidget(self.table, 1)
        self.output = Collapsible("Output", self.log, tooltip="Full output, including the complete ranking of the "
                                                              "Dextra check.")
        results.body.addWidget(self.output)

        left = QVBoxLayout()
        for w in (pick, dextra, compare):
            left.addWidget(w)
        left.addStretch(1)
        body = QHBoxLayout()
        body.addLayout(left, 2)
        body.addWidget(results, 3)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Evaluate", "Measure on recordings: does Dextra's model work here, and which "
                                               "method plays best."))
        page.addLayout(body, 1)

    def on_activated(self):
        people = sorted({m.get("person", "") for m in list_recordings(self.state.recordings_root)} - {""})
        current = self.target.currentData()
        self.target.clear()
        for p in people:
            self.target.addItem(p, os.path.join(self.state.recordings_root, p))
        self.target.addItem("All recordings", self.state.recordings_root)
        idx = self.target.findData(current)
        if idx >= 0:
            self.target.setCurrentIndex(idx)
        path = self.state.cfg.cnn.model_path
        self.model_label.setText(f"Motion model: {os.path.basename(path)}" if os.path.exists(path) else
                                 "No motion model file: 'Motion model' and 'Both' are skipped")
        if not people:
            self.verdict.setText("No recordings yet")

    def _transfer_dir(self) -> str:
        return os.path.join(self.state.analysis_root, "dextra_transfer")

    def _busy(self, job: str, text: str):
        self._job = job
        self.verdict.setText(text)
        self.verdict.setStyleSheet("font-size:11pt; font-weight:600;")
        self.run_btn.setEnabled(False)
        self.transfer_btn.setEnabled(False)

    def _transfer(self):
        if self.runner.running() or not self.target.currentData():
            return
        self.transfer_result.setText("")
        self.apply_btn.setVisible(False)
        self._busy("transfer", "Dextra model check running")
        self.runner.start(["tools/dextra_transfer.py", "--recordings", self.target.currentData(),
                           "--config", self.state.run_config(), "--out", self._transfer_dir()])

    def _open_sheets(self):
        path = self._transfer_dir()
        if os.path.isdir(path):
            os.startfile(os.path.abspath(path))

    def _run(self):
        if self.runner.running() or not self.target.currentData():
            return
        sources = [n for n, cb in self.src.items() if cb.isChecked()]
        if not sources:
            return
        args = ["replay_eval.py", "--recordings", self.target.currentData(), "--config", self.state.run_config(),
                "--sources", ",".join(sources)]
        for item in re.split(r"[;\n]+", self.overrides.text()):
            if item.strip():
                args += ["--set", item.strip()]
        if self.sweep.text().strip():
            args += ["--sweep", self.sweep.text().strip()]
        self.table.setRowCount(0)
        self._busy("compare", "Comparison running")
        self.runner.start(args)

    def _line(self, line: str):
        if line.startswith("@@RESULT "):
            self._add_result(json.loads(line[len("@@RESULT "):]))
            return
        self.log.log(line)
        if line.startswith("GO/NO-GO:"):
            text = line.split(":", 1)[1].strip()
            if "ship FUSED" in text:
                text, color = "Use both methods: they beat the hand tracker alone", OK
            elif "no verdict" in text:
                text, color = "No verdict: not enough recordings with decisions", WARN
            else:
                text, color = "Use the hand tracker alone: both methods together are not better", BAD
            self.verdict.setText(text)
            self.verdict.setStyleSheet(f"font-size:11pt; font-weight:600; color:{color};")
        if line.startswith("@@BEST "):
            self._show_best(json.loads(line[len("@@BEST "):]))

    def _show_best(self, best: dict):
        self._best = best
        per = ", ".join(f"{g} {v * 100:.0f}%" for g, v in best["per_gesture"].items())
        text = (f"Best: turned {best['rotate']} degrees, {'mirrored' if best['flip'] else 'not mirrored'}, "
                f"movement per image {best['event_count']}, motion sensitivity {best['contrast_threshold']:.2f}: "
                f"{best['balanced'] * 100:.0f}% of motion images correct ({per}).")
        if best.get("current_balanced") is not None:
            text += f" Current setting: {best['current_balanced'] * 100:.0f}%."
        if best["zoom"] != 1.0:
            text += f" Also shrink the play zone to about {best['zoom'] * 100:.0f}% around the hand on Setup."
        self.transfer_result.setText(text)
        self.apply_btn.setVisible(True)

    def _apply_best(self):
        b, cfg = self._best, self.state.cfg
        if not b:
            return
        cfg.cnn.rotate, cfg.cnn.flip = int(b["rotate"]), bool(b["flip"])
        cfg.dvs.event_count, cfg.dvs.contrast_threshold = int(b["event_count"]), float(b["contrast_threshold"])
        self.state.mark_dirty()
        self.apply_btn.setVisible(False)
        self.transfer_result.setText(self.transfer_result.text() + " Applied; save on Settings to keep it.")

    def _done(self, code: int):
        self.run_btn.setEnabled(True)
        self.transfer_btn.setEnabled(True)
        if self.runner.cancelled:
            self.verdict.setText("Cancelled")
            self.verdict.setStyleSheet("font-size:11pt; font-weight:600;")
        elif code != 0:
            self.verdict.setText("Stopped with an error: see Output")
            self.verdict.setStyleSheet(f"font-size:11pt; font-weight:600; color:{BAD};")
            self.output.toggle.setChecked(True)
        elif self._job == "transfer":
            self.verdict.setText("Dextra model check done: best setting shown on the left, full ranking in Output")
            self.output.toggle.setChecked(True)
        elif self.verdict.text() == "Comparison running":
            self.verdict.setText("Done. The verdict needs 'Hand tracker' and 'Both' selected.")

    def _add_result(self, res: dict):
        r = self.table.rowCount()
        self.table.insertRow(r)
        names = dict(METHODS)
        for c, (key, _name, kind, _tip) in enumerate(COLUMNS):
            v = res.get(key)
            if key == "source":
                text = names.get(v, v)
            elif v is None or v == "":
                text = "-"
            elif kind == "pct":
                text = f"{v * 100:.1f}%"
            elif kind == "num":
                text = f"{v:.1f}"
            else:
                text = str(v)
            self.table.setItem(r, c, QTableWidgetItem(text))

    def shutdown(self):
        self.runner.kill()
