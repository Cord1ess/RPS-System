"""
Visual language of the desktop app: one dark theme, compact flat controls, cards, status chips,
collapsible sections, and page headers. Every helper takes a tooltip so nothing is unexplained.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

BG, PANEL, RAISED, BORDER = "#16191d", "#1e2226", "#262b31", "#353c44"
TEXT, MUTED, ACCENT = "#e4e7eb", "#9aa3ad", "#2f6fed"
OK, WARN, BAD, OFF = "#3fb96b", "#e0a030", "#e0524a", "#6b7280"
GESTURE_COLOR = {"rock": "#e879b0", "paper": "#5ccf82", "scissors": "#5b9cf0", "none": "#8b949e",
                 "ready": "#8b949e"}
# Each reader has one colour, used for its card, its delay curve and its label on the video.
MOTION_COLOR, TRACKER_COLOR = "#a78bfa", "#2cc5c9"
CHECK_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "check.svg").replace("\\", "/")

QSS = f"""
QWidget {{ color:{TEXT}; font-size:9pt; }}
QToolTip {{ background:#0f1215; color:{TEXT}; border:1px solid {BORDER}; padding:5px 7px; font-size:9pt; }}
QFrame#card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:6px; }}
QLabel {{ background:transparent; }}
QLabel#cardTitle {{ font-size:9.5pt; font-weight:600; color:{TEXT}; }}
QLabel#help, QLabel#caption {{ color:{MUTED}; font-size:8.5pt; }}
QLabel#pageTitle {{ font-size:12pt; font-weight:600; }}
QLabel#metric {{ font-size:18pt; font-weight:600; }}
QLabel#metricLabel {{ color:{MUTED}; font-size:8pt; }}
QPushButton {{ background:{RAISED}; border:1px solid {BORDER}; border-radius:4px; padding:3px 10px;
    min-height:20px; }}
QPushButton:hover {{ background:#2e343b; border-color:#4a525c; }}
QPushButton:pressed {{ background:#23282d; }}
QPushButton:disabled {{ color:#5b636c; border-color:#2c3238; }}
QPushButton:checked {{ background:#24344f; border-color:{ACCENT}; }}
QPushButton#primary {{ background:{ACCENT}; border-color:{ACCENT}; color:white; }}
QPushButton#primary:hover {{ background:#3b7bf5; }}
QPushButton#primary:disabled {{ background:#26344f; border-color:#26344f; color:#7d8794; }}
QPushButton#danger {{ background:#a8322c; border-color:#a8322c; color:white; }}
QPushButton#danger:hover {{ background:#bd3b34; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background:{RAISED}; border:1px solid {BORDER};
    border-radius:4px; padding:2px 6px; min-height:20px; }}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color:#4a525c; }}
QComboBox QAbstractItemView {{ background:{RAISED}; selection-background-color:{ACCENT}; }}
QProgressBar {{ background:{RAISED}; border:1px solid {BORDER}; border-radius:3px; text-align:center;
    max-height:14px; font-size:8pt; }}
QProgressBar::chunk {{ background:{ACCENT}; border-radius:2px; }}
QTabWidget::pane {{ border:none; }}
QTabBar::tab {{ background:transparent; color:{MUTED}; padding:6px 14px; border:none;
    border-bottom:2px solid transparent; }}
QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color:{TEXT}; }}
QTabBar::tab:disabled {{ color:#4b5563; }}
QTableWidget {{ background:{PANEL}; gridline-color:{BORDER}; border:1px solid {BORDER}; border-radius:4px; }}
QHeaderView::section {{ background:{RAISED}; color:{MUTED}; border:none; padding:3px 6px; }}
QPlainTextEdit {{ background:#0f1215; border:1px solid {BORDER}; border-radius:4px; }}
QScrollArea {{ border:none; background:transparent; }}
QToolButton#collapse {{ background:transparent; border:none; color:{MUTED}; padding:2px 0; }}
QToolButton#collapse:hover {{ color:{TEXT}; }}
QCheckBox, QRadioButton {{ background:transparent; spacing:6px; }}
QCheckBox::indicator {{ width:14px; height:14px; border:1px solid #5b636c; border-radius:3px; background:{RAISED}; }}
QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; image:url({CHECK_ICON}); }}
QRadioButton::indicator {{ width:12px; height:12px; border:1px solid #5b636c; border-radius:7px; background:{RAISED}; }}
QRadioButton::indicator:checked {{ width:8px; height:8px; background:{ACCENT}; border:3px solid {RAISED}; }}   /* same 14 px outside as unchecked, so the label never moves */
QStatusBar {{ color:{MUTED}; }}
"""


def apply_theme(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.ColorRole.Window, BG), (QPalette.ColorRole.Base, RAISED),
                        (QPalette.ColorRole.AlternateBase, PANEL), (QPalette.ColorRole.Text, TEXT),
                        (QPalette.ColorRole.WindowText, TEXT), (QPalette.ColorRole.Button, RAISED),
                        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
                        (QPalette.ColorRole.ToolTipBase, "#0f1215"), (QPalette.ColorRole.ToolTipText, TEXT)):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(QSS)


def tip(widget, text: str):
    """Sets a hover explanation and returns the widget (for inline use)."""
    widget.setToolTip(text)
    return widget


def help_label(text: str, tooltip: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("help")
    lab.setWordWrap(True)
    if tooltip:
        lab.setToolTip(tooltip)
    return lab


def caption(text: str, tooltip: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("caption")
    lab.setWordWrap(True)
    if tooltip:
        lab.setToolTip(tooltip)
    return lab


def label(text: str, tooltip: str = "") -> QLabel:
    lab = QLabel(text)
    if tooltip:
        lab.setToolTip(tooltip)
    return lab


def button(text: str, kind: str = "", tooltip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if tooltip:
        b.setToolTip(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def set_kind(b: QPushButton, kind: str):
    """Switches a button between plain / primary / danger styles."""
    b.setObjectName(kind)
    b.style().unpolish(b)
    b.style().polish(b)


class Card(QFrame):
    """Titled panel; the title's hover text explains the card. Add content to self.body."""

    def __init__(self, title: str, tooltip: str = "", color: str = ""):
        super().__init__()
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(6)
        self.head = head = QLabel(title)
        head.setObjectName("cardTitle")
        if color:                                   # identifies a reader everywhere it appears
            head.setStyleSheet(f"color:{color};")
            self.setStyleSheet(f"QFrame#card {{ border-left: 3px solid {color}; }}")
        if tooltip:
            head.setToolTip(tooltip)
        outer.addWidget(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        outer.addLayout(self.body)


class Chip(QLabel):
    """Compact status label with a coloured edge. Restyles only when its state changes."""
    COLORS = {"ok": OK, "warn": WARN, "bad": BAD, "off": OFF, "info": ACCENT}

    def __init__(self, text: str = "", level: str = "off", tooltip: str = ""):
        super().__init__()
        self._level = None
        if tooltip:
            self.setToolTip(tooltip)
        self.set(text, level)

    def set(self, text: str, level: str = "off"):
        if text != self.text():
            self.setText(text)
        if level != self._level:
            self._level = level
            c = self.COLORS.get(level, OFF)
            self.setStyleSheet(f"QLabel {{ color:{TEXT if level != 'off' else MUTED}; background:{RAISED};"
                               f" border-left:3px solid {c}; border-radius:2px; padding:2px 8px; font-size:8.5pt; }}")


class Collapsible(QWidget):
    """A header that shows/hides its content (collapsed by default)."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False, tooltip: str = ""):
        super().__init__()
        self.title = title
        self.toggle = QToolButton()
        self.toggle.setObjectName("collapse")
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setToolTip(tooltip or f"Show or hide: {title.lower()}")
        self.toggle.toggled.connect(self._set)
        self.content = content
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle)
        lay.addWidget(content)
        self.toggle.setChecked(expanded)
        self._set(expanded)

    def _set(self, on: bool):
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.toggle.setText(self.title)
        self.content.setVisible(on)


def static_plot(plot):
    """A read-only graph: no drag, zoom, auto-scale button or right-click menu to disturb it."""
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    plot.setMenuEnabled(False)
    return plot


def row(*widgets):
    """Widgets side by side, left-aligned (buttons keep their natural width)."""
    from PySide6.QtWidgets import QHBoxLayout
    lay = QHBoxLayout()
    lay.setSpacing(6)
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch(1)
    return lay


def page_header(title: str, text: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(2, 0, 2, 4)
    lay.setSpacing(0)
    t = QLabel(title)
    t.setObjectName("pageTitle")
    lay.addWidget(t)
    lay.addWidget(help_label(text))
    return w
