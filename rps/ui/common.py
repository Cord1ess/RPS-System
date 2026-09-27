"""
Shared UI building blocks: app state, video view (with play-zone dragging), streaming process
runner for the CLI scripts, filtered log view, and a config form that shows readable names and
explanations (rps/ui/fields.py) instead of raw config keys.
"""

import json
import os
import re
import sys
from dataclasses import fields
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QSizePolicy, QSpinBox, QWidget)

from rps.config import Config, load_config, save_config
from rps.ui.fields import CHOICES, help_for, is_advanced, label_for
from rps.ui.style import BORDER

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOISE = re.compile(r"^(W0000|I0000|E0000|WARNING: All log|WARNING: Logging before InitGoogle|INFO: Created TensorFlow"
                   r"|.*inference_feedback_manager"
                   r"|.*landmark_projection_calculator|\[ WARN:)")


class AppState(QObject):
    """One Config shared by every tab, plus the data/model paths."""
    config_changed = Signal()

    def __init__(self, config_path: str = "config.json", data_root: str = "data"):
        super().__init__()
        self.config_path = config_path
        self.recordings_root = os.path.join(data_root, "recordings")
        self.frames_root = os.path.join(data_root, "frames")
        self.analysis_root = os.path.join(data_root, "analysis")
        self.cfg: Config = load_config(config_path)
        self.dirty = False

    def mark_dirty(self):
        self.dirty = True
        self.config_changed.emit()

    def save(self):
        save_config(self.cfg, self.config_path)
        self.dirty = False
        self.config_changed.emit()

    def reload(self):
        new = load_config(self.config_path)
        for f in fields(Config):             # update in place so every holder sees the change
            setattr(self.cfg, f.name, getattr(new, f.name))
        self.dirty = False
        self.config_changed.emit()

    def reset_defaults(self):
        new = Config()
        for f in fields(Config):
            setattr(self.cfg, f.name, getattr(new, f.name))
        self.mark_dirty()


def to_pixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


def select_data(combo: QComboBox, value) -> bool:
    """Selects the combo item whose data equals value."""
    idx = combo.findData(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    return idx >= 0


class VideoView(QLabel):
    """Aspect-correct image view. With roi_edit on, dragging selects a square play zone."""
    roi_selected = Signal(int, int, int)

    def __init__(self, min_w: int = 480, min_h: int = 360, placeholder: str = "Camera off"):
        super().__init__()
        self.setMinimumSize(min_w, min_h)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background:#0d1014; color:#6b7280; border:1px solid {BORDER}; border-radius:8px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.placeholder = placeholder
        self.setText(placeholder)
        self.roi_edit = False
        self._img_size = None
        self._scaled = None
        self._drag_start = None
        self._drag_now = None

    def clear_image(self, message: str = ""):
        self.clear()
        self._img_size = self._scaled = None
        self.setText(message or self.placeholder)

    def show_status(self, message: str):
        """Shows a message (e.g. 'Starting camera...') until the next image arrives."""
        self.clear_image(message)

    def show_image(self, img: np.ndarray):
        pix = to_pixmap(img)
        self._img_size = (pix.width(), pix.height())
        scaled = pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self._scaled = (scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def _offset(self):
        sw, sh = self._scaled
        return (self.width() - sw) / 2.0, (self.height() - sh) / 2.0

    def _to_image(self, pos) -> Optional[tuple]:
        if self._img_size is None or self._scaled is None:
            return None
        ox, oy = self._offset()
        scale = self._img_size[0] / self._scaled[0]
        x = int(np.clip((pos.x() - ox) * scale, 0, self._img_size[0] - 1))
        y = int(np.clip((pos.y() - oy) * scale, 0, self._img_size[1] - 1))
        return x, y

    def mousePressEvent(self, e):
        if self.roi_edit:
            self._drag_start = self._drag_now = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self.roi_edit and self._drag_start is not None:
            self._drag_now = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if not (self.roi_edit and self._drag_start is not None):
            return
        a, b = self._to_image(self._drag_start), self._to_image(e.position().toPoint())
        self._drag_start = self._drag_now = None
        self.update()
        if a is None or b is None:
            return
        size = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if size >= 32:
            cx, cy = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
            self.roi_selected.emit(cx - size // 2, cy - size // 2, size)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._drag_start is not None and self._drag_now is not None:
            p = QPainter(self)
            p.setPen(QPen(QColor(255, 220, 0), 2, Qt.PenStyle.DashLine))
            p.drawRect(QRect(self._drag_start, self._drag_now).normalized())
            p.end()


class LogView(QPlainTextEdit):
    def __init__(self, max_lines: int = 4000):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(9)
        self.setFont(f)

    def log(self, line: str):
        if line and not NOISE.match(line):
            self.appendPlainText(line.rstrip())


class ProcessRunner(QObject):
    """Runs one of the project's CLI scripts with the same interpreter, streaming output lines."""
    line = Signal(str)
    finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self._buf = ""

    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning

    def start(self, args: List[str]):
        if self.running():
            raise RuntimeError("a process is already running")
        self._buf = ""
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(PROJECT_ROOT)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._done)
        self.proc = proc
        self.line.emit(f"$ python {' '.join(str(a) for a in args)}")
        proc.start(sys.executable, [str(a) for a in args])

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buf += data.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self._buf:
            text, self._buf = self._buf.split("\n", 1)
            self.line.emit(text)

    def _done(self, code, _status=None):
        if self._buf:
            self.line.emit(self._buf)
            self._buf = ""
        self.line.emit(f"[finished, exit code {code}]")
        self.finished.emit(int(code))

    def kill(self):
        if self.running():
            self.proc.kill()


class ConfigForm(QWidget):
    """
    Editable form for one Config section. Labels and hover explanations come from
    rps/ui/fields.py. Edits apply to the shared Config immediately.
    """
    changed = Signal(str, str)

    def __init__(self, state: AppState, section: str, keys: Optional[List[str]] = None,
                 show_help: bool = False, show_advanced: bool = True):
        super().__init__()
        self.state, self.section = state, section
        self.keys = keys
        self.show_help = show_help
        self.widgets: Dict[str, QWidget] = {}
        self.rows: Dict[str, List[QWidget]] = {}
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(2)
        self.grid.setColumnStretch(1, 1)
        self._building = False
        self._build()
        self.set_show_advanced(show_advanced)
        state.config_changed.connect(self.refresh)

    def _sec(self):
        return getattr(self.state.cfg, self.section)

    def _build(self):
        self._building = True
        sec = self._sec()
        r = 0
        for f in fields(sec):
            if self.keys and f.name not in self.keys:
                continue
            value = getattr(sec, f.name)
            key = (self.section, f.name)
            if key in CHOICES:
                w = QComboBox()
                for data, text in CHOICES[key]:
                    w.addItem(text, data)
                w.currentIndexChanged.connect(lambda _i, n=f.name, w=w: self._set(n, w.currentData()))
            elif isinstance(value, bool):
                w = QCheckBox()
                w.toggled.connect(lambda v, n=f.name: self._set(n, bool(v)))
            elif isinstance(value, int):
                w = QSpinBox()
                w.setRange(-1_000_000, 1_000_000)
                w.valueChanged.connect(lambda v, n=f.name: self._set(n, int(v)))
            elif isinstance(value, float):
                w = QDoubleSpinBox()
                w.setRange(-1e6, 1e6)
                w.setDecimals(3)
                w.setSingleStep(0.01 if abs(value) < 1 else (0.1 if abs(value) < 10 else 1.0))
                w.valueChanged.connect(lambda v, n=f.name: self._set(n, float(v)))
            elif isinstance(value, dict):
                w = QLineEdit()
                w.editingFinished.connect(lambda n=f.name, w=w: self._set_json(n, w.text()))
            else:
                w = QLineEdit()
                w.editingFinished.connect(lambda n=f.name, w=w: self._set(n, w.text()))
            text = help_for(*key)
            tip = text or label_for(*key)
            label = QLabel(label_for(*key))
            label.setToolTip(tip)
            w.setToolTip(tip)
            self.grid.addWidget(label, r, 0)
            self.grid.addWidget(w, r, 1)
            row_widgets = [label, w]
            r += 1
            if self.show_help and text:
                h = QLabel(text)
                h.setObjectName("help")
                h.setWordWrap(True)
                self.grid.addWidget(h, r, 0, 1, 2)
                row_widgets.append(h)
                r += 1
            self.widgets[f.name] = w
            self.rows[f.name] = row_widgets
        self._building = False
        self.refresh()

    def set_show_advanced(self, show: bool):
        for name, ws in self.rows.items():
            visible = show or not is_advanced(self.section, name)
            for w in ws:
                w.setVisible(visible)

    def refresh(self):
        self._building = True
        sec = self._sec()
        for name, w in self.widgets.items():
            value = getattr(sec, name)
            if isinstance(w, QComboBox):
                select_data(w, value)
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setValue(value)
            elif isinstance(value, dict):
                w.setText(json.dumps(value))
            else:
                w.setText(str(value))
        self._building = False

    def _set(self, name: str, value):
        if self._building:
            return
        setattr(self._sec(), name, value)
        self.state.dirty = True
        self.changed.emit(self.section, name)
        self.state.config_changed.emit()

    def _set_json(self, name: str, text: str):
        try:
            self._set(name, json.loads(text))
        except ValueError:
            self.refresh()
