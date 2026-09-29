"""Evaluate page: score any of the four recognizers on recordings, side by side."""

import json
import os

from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from rps.pipeline import RECOGNIZERS, missing_reason
from rps.recorder import list_recordings
from rps.ui.base import Tab
from rps.ui.common import LogView, ProcessRunner, open_folder
from rps.ui.style import BAD, OK, Card, Collapsible, button, caption, page_header, row, tip

# (result key, column title, format, hover)
COLUMNS = [("source", "Model", "name", "Which recognizer was replayed."),
           ("show_hold_acc", "Holding right", "pct", "'Hold one gesture' recordings: share of the time the decision "
                                                     "was the gesture held."),
           ("show_switches_per_min", "Changes/min", "num", "'Hold one gesture' recordings: decision changes per "
                                                          "minute while one gesture was held. Should be about 0."),
           ("throw_acc", "Throws right", "pct", "'Countdown throws' recordings: throws decided correctly."),
           ("throw_commits", "Throws decided", "int", "Throws the system decided (compare with the throws "
                                                      "recorded)."),
           ("throws_planned", "Throws recorded", "int", "Throws planned when recording."),
           ("commit_vs_throw_ms", "Decides after throw (ms)", "num", "How long after the throw showed the decision "
                                                                     "came. Lower is faster."),
           ("bg_false_commits_per_min", "No-hand moves/min", "num", "'No hand' recordings: decisions made with no "
                                                                   "hand present. Should be 0.")]


class EvaluateTab(Tab):
    title = "6  Evaluate"

    def __init__(self, main):
        super().__init__(main)
        self.runner = ProcessRunner(self)
        self.log = LogView()
        self.runner.line.connect(self._line)
        self.runner.finished.connect(self._done)
        self._job = ""
        self._best = None

        pick = Card("Test on", "Which recordings to replay. Use a person the models did not learn from.")
        self.target = tip(QComboBox(), "Recordings used for the test.")
        pick.body.addWidget(self.target)

        models = Card("Models", "Tick the models to compare. Each replays the recordings through exactly what Play "
                                "runs.")
        self.all_box = tip(QCheckBox("All"), "Tick or untick every available model.")
        self.all_box.toggled.connect(self._all_toggled)
        models.body.addWidget(self.all_box)
        self.boxes = {}
        for key, name in RECOGNIZERS.items():
            cb = tip(QCheckBox(name), f"Score {name}.")
            cb.toggled.connect(self._box_toggled)
            self.boxes[key] = cb
            models.body.addWidget(cb)
        self.run_btn = button("Run", "primary", "Replay the recordings and score each ticked model (the first run "
                                                "takes longer: Mediapipe's readings are stored for next time).")
        self.run_btn.clicked.connect(self._run)
        cancel = button("Cancel", tooltip="Stop the running job.")
        cancel.clicked.connect(self.runner.kill)
        models.body.addLayout(row(self.run_btn, cancel))

        check = Card("Dextra Raw view check", "How well Dextra Raw (as downloaded) reads your gestures on this camera, "
                                              "and which turn, mirror and movement setting suits it best.")
        self.transfer_btn = button("Run check", tooltip="Replays your 'Hold one gesture' and 'No hand' recordings "
                                                        "through Dextra Raw under every turn, mirror, movement per "
                                                        "image and zoom, and ranks them (a few minutes).")
        self.transfer_btn.clicked.connect(self._transfer)
        sheets = button("Open pictures", tooltip="Your Dextra views next to Dextra's own, per gesture, for the best "
                                                 "setting.")
        sheets.clicked.connect(self._open_sheets)
        check.body.addLayout(row(self.transfer_btn, sheets))
        self.transfer_result = caption("", "Best setting found by the last check.")
        check.body.addWidget(self.transfer_result)
        self.apply_btn = button("Use this setting", tooltip="Apply the best turn, mirror, movement per image and "
                                                            "movement sensitivity to Dextra Raw. Save on Settings to "
                                                            "keep it.")
        self.apply_btn.clicked.connect(self._apply_best)
        self.apply_btn.setVisible(False)
        check.body.addLayout(row(self.apply_btn))

        results = Card("Results", "One row per model. Hover a column title for its meaning.")
        self.verdict = QLabel("Nothing run yet")
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet("font-size:11pt; font-weight:600;")
        tip(self.verdict, "The best model on these recordings: most throws right, then most time holding right, "
                          "then fewest changes.")
        results.body.addWidget(self.verdict)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        for i, c in enumerate(COLUMNS):
            self.table.horizontalHeaderItem(i).setToolTip(c[3])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tip(self.table, "One row per model. Hover a column title for its meaning.")
        results.body.addWidget(self.table, 1)
        self.output = Collapsible("Output", self.log, tooltip="Full output of the last job.")
        results.body.addWidget(self.output)

        left = QVBoxLayout()
        for w in (pick, models):
            left.addWidget(w)
        left.addWidget(Collapsible("Dextra Raw view check", check, tooltip="Only for Dextra Raw: find the camera "
                                                                           "view setting that suits it."))
        left.addStretch(1)
        body = QHBoxLayout()
        body.addLayout(left, 2)
        body.addWidget(results, 3)
        page = QVBoxLayout(self)
        page.addWidget(page_header("Evaluate", "Which model reads your hand best, measured on recordings."))
        page.addLayout(body, 1)

    # ------------------------------------------------------------------ choices
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
        for key, cb in self.boxes.items():
            why = missing_reason(self.state.cfg, key)
            cb.setEnabled(why is None)
            cb.setToolTip(why or f"Score {RECOGNIZERS[key]}.")
            if why:
                cb.setChecked(False)
        if not any(cb.isChecked() for cb in self.boxes.values()):
            self.all_box.setChecked(True)
            self._all_toggled(True)
        if not people:
            self.verdict.setText("No recordings yet")

    def _all_toggled(self, on: bool):
        for cb in self.boxes.values():
            if cb.isEnabled():
                cb.blockSignals(True)
                cb.setChecked(on)
                cb.blockSignals(False)

    def _box_toggled(self, *_):
        available = [cb for cb in self.boxes.values() if cb.isEnabled()]
        self.all_box.blockSignals(True)
        self.all_box.setChecked(bool(available) and all(cb.isChecked() for cb in available))
        self.all_box.blockSignals(False)

    # ------------------------------------------------------------------ jobs
    def _busy(self, job: str, text: str):
        self._job = job
        self.verdict.setText(text)
        self.verdict.setStyleSheet("font-size:11pt; font-weight:600;")
        self.run_btn.setEnabled(False)
        self.transfer_btn.setEnabled(False)

    def _run(self):
        if self.runner.running() or not self.target.currentData():
            return
        sources = [k for k, cb in self.boxes.items() if cb.isChecked() and cb.isEnabled()]
        if not sources:
            self.verdict.setText("Tick at least one model")
            return
        self.table.setRowCount(0)
        self._best = None
        self._busy("compare", "Running")
        self.runner.start(["replay_eval.py", "--recordings", self.target.currentData(), "--config",
                           self.state.run_config(), "--sources", ",".join(sources)])

    def _transfer_dir(self) -> str:
        return os.path.join(self.state.analysis_root, "dextra_transfer")

    def _transfer(self):
        if self.runner.running() or not self.target.currentData():
            return
        self.transfer_result.setText("")
        self.apply_btn.setVisible(False)
        self._busy("transfer", "Dextra Raw view check running")
        self.runner.start(["tools/dextra_transfer.py", "--recordings", self.target.currentData(), "--model",
                           self.state.cfg.cnn.raw_model, "--config", self.state.run_config(),
                           "--out", self._transfer_dir()])

    def _open_sheets(self):
        path = self._transfer_dir()
        open_folder(path)

    def _line(self, line: str):
        if line.startswith("@@RESULT "):
            self._add_result(json.loads(line[len("@@RESULT "):]))
            return
        if line.startswith("@@SKIP "):
            info = json.loads(line[len("@@SKIP "):])
            self._add_row([RECOGNIZERS.get(info["source"], info["source"]), "not available"], info["reason"])
            return
        if line.startswith("@@BEST "):
            self._show_best(json.loads(line[len("@@BEST "):]))
            return
        self.log.log(line)
        if line.startswith("Best on these recordings:"):
            self._best = line.split(":", 1)[1].strip()

    def _done(self, code: int):
        self.run_btn.setEnabled(True)
        self.transfer_btn.setEnabled(True)
        style = "font-size:11pt; font-weight:600;"
        if self.runner.cancelled:
            self.verdict.setText("Cancelled")
            self.verdict.setStyleSheet(style)
        elif code != 0:
            self.verdict.setText("Stopped with an error: see Output")
            self.verdict.setStyleSheet(style + f" color:{BAD};")
            self.output.toggle.setChecked(True)
        elif self._job == "transfer":
            self.verdict.setText("Dextra Raw view check done: the best setting is on the left, the full ranking in "
                                 "Output")
            self.verdict.setStyleSheet(style)
        elif self._best:
            self.verdict.setText(f"Best on these recordings: {self._best}")
            self.verdict.setStyleSheet(style + f" color:{OK};")
        else:
            self.verdict.setText("Done: no model decided anything on these recordings")
            self.verdict.setStyleSheet(style)

    def _add_row(self, texts, tooltip: str = ""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, text in enumerate(texts):
            item = QTableWidgetItem(text)
            if tooltip:
                item.setToolTip(tooltip)
            self.table.setItem(r, c, item)

    def _add_result(self, res: dict):
        texts = []
        for key, _name, kind, _tip in COLUMNS:
            v = res.get(key)
            if kind == "name":
                texts.append(RECOGNIZERS.get(v, v))
            elif v is None or v == "":
                texts.append("-")
            elif kind == "pct":
                texts.append(f"{v * 100:.1f}%")
            elif kind == "num":
                texts.append(f"{v:.1f}")
            else:
                texts.append(str(v))
        self._add_row(texts)

    # ------------------------------------------------------------------ Dextra Raw view check result
    def _show_best(self, best: dict):
        self._best_setting = best
        per = ", ".join(f"{g} {v * 100:.0f}%" for g, v in best["per_gesture"].items())
        text = (f"Best: turned {best['rotate']} degrees, {'mirrored' if best['flip'] else 'not mirrored'}, "
                f"movement per image {best['event_count']}, movement sensitivity {best['contrast_threshold']:.2f}: "
                f"{best['balanced'] * 100:.0f}% of Dextra views correct ({per}).")
        if best.get("current_balanced") is not None:
            text += f" Current setting: {best['current_balanced'] * 100:.0f}%."
        if best["zoom"] != 1.0:
            text += f" Also shrink the play zone to about {best['zoom'] * 100:.0f}% around the hand on Setup."
        self.transfer_result.setText(text)
        self.apply_btn.setVisible(True)

    def _apply_best(self):
        b, cfg = getattr(self, "_best_setting", None), self.state.cfg
        if not b:
            return
        cfg.cnn.rotate, cfg.cnn.flip = int(b["rotate"]), bool(b["flip"])
        cfg.dvs.event_count, cfg.dvs.contrast_threshold = int(b["event_count"]), float(b["contrast_threshold"])
        self.state.mark_dirty()
        self.apply_btn.setVisible(False)
        self.transfer_result.setText(self.transfer_result.text() + " Applied; save on Settings to keep it.")

    def shutdown(self):
        self.runner.kill()
