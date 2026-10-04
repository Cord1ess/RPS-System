"""
The Speed card's Qt side: DelayBar draws a ThrowDelay (rps.delay) as one stacked bar, with a line
where the UDP command left the laptop, and DelayCard adds the headline, the breakdown and this run's
summary. The delays themselves are measured and assembled in rps.delay, which the command line also
prints.
"""

from statistics import median
from typing import List

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHeaderView, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem, QWidget

from rps.decision import GESTURE_NAME
from rps.delay import (COLORS, COMPUTE, HARDWARE, HISTORY, POSE_GESTURE, WAIT, Segment, ThrowDelay,
                       robot_move_ms, throw_delay)
from rps.ui.style import BORDER, MUTED, OK, TEXT, Card, caption

__all__ = ["COLORS", "COMPUTE", "HARDWARE", "HISTORY", "POSE_GESTURE", "WAIT", "DelayBar", "DelayCard",
           "Segment", "ThrowDelay", "robot_move_ms", "throw_delay"]


class DelayBar(QWidget):
    """The throw's delays as one stacked bar; a line marks the moment the command left the laptop."""

    def __init__(self):
        super().__init__()
        self.segments: List[Segment] = []
        self.setMinimumHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_segments(self, segments: List[Segment]):
        self.segments = segments
        self.setToolTip("\n".join(f"{s.name}: {'?' if s.ms is None else f'{s.ms:.0f} ms'} ({s.source})"
                                  for s in segments))
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        known = [s for s in self.segments if s.ms is not None and s.ms > 0]
        top, height = 16.0, float(self.height() - 18)
        width = float(self.width() - 2)
        if not known:
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(0, top, width, height), Qt.AlignmentFlag.AlignCenter, "No throw yet")
            return
        total = sum(s.ms for s in known)
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1))
        p.setFont(small)
        x, sent_x, used = 1.0, None, {k: 0 for k in COLORS}
        for s in known:
            w = max(3.0, width * s.ms / total)
            if s.kind == HARDWARE and sent_x is None and x > 1.0 and any(
                    k.kind == COMPUTE for k in known[:known.index(s)]):
                sent_x = x                                   # the first hardware part after the laptop's work
            color = COLORS[s.kind][used[s.kind] % len(COLORS[s.kind])]
            used[s.kind] += 1
            rect = QRectF(x, top, w - 1, height)
            p.fillRect(rect, QColor(color))
            text = f"{s.name}  {s.ms:.0f} ms"
            if p.fontMetrics().horizontalAdvance(text) + 8 > w:
                text = f"{s.ms:.0f}"
            if p.fontMetrics().horizontalAdvance(text) + 4 <= w:
                p.setPen(QColor("#0d0f12"))
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
            x += w
        if sent_x is not None:                              # the UDP command leaves here
            p.setPen(QPen(QColor(TEXT), 2))
            p.drawLine(int(sent_x), 2, int(sent_x), int(top + height))
            label_w = p.fontMetrics().horizontalAdvance("command sent") + 2
            left = sent_x + 4 + label_w > width                # no room on the right: write it on the left
            p.drawText(QRectF(sent_x - 4 - label_w if left else sent_x + 4, 0, label_w, 14),
                       (Qt.AlignmentFlag.AlignRight if left else Qt.AlignmentFlag.AlignLeft)
                       | Qt.AlignmentFlag.AlignVCenter, "command sent")
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawRect(QRectF(0.5, top - 0.5, width, height + 1))


class DelayCard(Card):
    """The last throw's delays (headline, bar, breakdown) and this run's summary. With `table`,
    also `table_card`: the last throws one per row, for the page to place."""

    def __init__(self, title: str = "Speed", table: bool = False):
        super().__init__(title, "How fast each throw was read and its command sent to the robot, split into the "
                                "laptop's computing (green), waiting for camera images (amber) and the hardware "
                                "(grey: camera, Wi-Fi, the robot hand). Measured on every throw.")
        self.headline = QLabel("-")
        self.headline.setStyleSheet(f"font-size:18pt; font-weight:700; color:{OK};")
        self.headline.setToolTip("The laptop's own work for the last throw: the deciding camera image arriving -> "
                                 "the UDP command leaving. Then, the time from the first camera image showing the "
                                 "throw to the command.")
        self.body.addWidget(self.headline)
        self.bar = DelayBar()
        self.body.addWidget(self.bar)
        self.detail = caption("Play a throw to see where the time goes.")
        self.detail.setWordWrap(True)
        self.body.addWidget(self.detail)
        self.summary = caption("")
        self.summary.setWordWrap(True)
        self.body.addWidget(self.summary)
        self.history: List[ThrowDelay] = []
        self.table = self.table_card = None
        if table:
            self.table_card = Card("Last throws", "Each throw's timing, newest first (ms). Wait: camera images "
                                                  "needed to be sure. Compute: the laptop's work and the UDP send. "
                                                  "Total: first image showing the throw -> command sent.")
            self.table = QTableWidget(0, 6)
            self.table.setHorizontalHeaderLabels(["Throw", "Robot", "Read by", "Wait", "Compute", "Total"])
            self.table.verticalHeader().setVisible(False)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.setMinimumHeight(150)
            self.table.setMaximumHeight(260)
            self.table_card.body.addWidget(self.table)

    def reset(self):
        self.history = []
        self.headline.setText("-")
        self.bar.set_segments([])
        self.detail.setText("Play a throw to see where the time goes.")
        self.summary.setText("")
        if self.table is not None:
            self.table.setRowCount(0)

    def add(self, d: ThrowDelay):
        self.history.append(d)
        compute, total = d.compute_ms, d.to_command_ms
        wait = d.ms("Waiting for camera images")
        self.headline.setText(d.headline())
        self.bar.set_segments(d.segments)
        self.detail.setText(d.detail())
        done = [h for h in self.history if h.compute_ms is not None]
        if done:
            self.summary.setText(
                f"This run, {len(done)} throw{'s' if len(done) != 1 else ''}: computing median "
                f"{median(h.compute_ms for h in done):.0f} ms (slowest {max(h.compute_ms for h in done):.0f}), "
                f"throw -> command median {median(h.to_command_ms for h in done):.0f} ms, hardware "
                f"~{median(h.hardware_ms for h in self.history):.0f} ms.")
        if self.table is not None:
            self.table.insertRow(0)
            cells = [GESTURE_NAME[d.gesture], POSE_GESTURE.get(d.pose, "?"), d.reader, f"{wait:.0f}",
                     "-" if compute is None else f"{compute:.0f}", "-" if total is None else f"{total:.0f}"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(0, col, item)
            while self.table.rowCount() > HISTORY:
                self.table.removeRow(self.table.rowCount() - 1)
