"""
Settings page: every config.json field with a readable name and explanation; advanced ones hidden by
default. At the top, this computer's interface size (rps.ui.scale), which applies after a restart.
"""

from dataclasses import fields

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QMessageBox, QScrollArea, QVBoxLayout, QWidget

from rps.config import Config
from rps.ui import scale
from rps.ui.base import Tab
from rps.ui.common import ConfigForm, select_data
from rps.ui.fields import SECTION_INFO
from rps.ui.style import Card, button, caption, label, page_header, tip

ORDER = ["camera", "decision", "vote", "dvs", "cnn", "roi", "hand", "latency"]
# Chosen on their own pages, so not repeated here: the robot and its move times (Bot tuning), the match
# and the recognition and game choices (Play and Play Debug).
SKIP = {"decision": {"recognizer", "mode"}, "latency": {"servo_transition_ms"}}


class SettingsTab(Tab):
    title = "Settings"

    def __init__(self, main):
        super().__init__(main)
        top = QHBoxLayout()
        self.advanced = QCheckBox("Show advanced settings")
        self.advanced.setToolTip("Show every setting, including the fine-tuning ones.")
        self.advanced.toggled.connect(self._layout_cards)
        save = button("Save", "primary", "Write all settings to config.json.")
        save.clicked.connect(self.state.save)
        undo = button("Undo changes", tooltip="Reload the last saved settings.")
        undo.clicked.connect(lambda: self.state.reload())
        reset = button("Reset to defaults", tooltip="Set every setting to its default. Not saved until you "
                                                          "press Save.")
        reset.clicked.connect(self._reset)
        top.addWidget(self.advanced)
        top.addStretch(1)
        for b in (save, undo, reset):
            top.addWidget(b)

        size_row = QHBoxLayout()
        self.size = tip(QComboBox(), "How big the whole interface is on this computer: every font, button and panel. "
                                     "Auto fits the window to the screen it opens on. Kept for this computer only "
                                     "(the laptop and the Pi keep their own).")
        for value, text in scale.CHOICES:
            self.size.addItem(text, value)
        select_data(self.size, self._saved_size())
        self.size.currentIndexChanged.connect(self._size_changed)
        self.restart_btn = button("Restart to apply", "primary", "Close and reopen the app at the new size (asks to "
                                                                 "save changed settings first).")
        self.restart_btn.clicked.connect(self._restart)
        self.size_note = caption(f"Now: {scale.current() * 100:.0f}%")
        size_row.addWidget(label("Interface size", self.size.toolTip()))
        size_row.addWidget(self.size)
        size_row.addWidget(self.restart_btn)
        size_row.addWidget(self.size_note)
        size_row.addStretch(1)
        self.restart_btn.setVisible(False)

        self.cards = []
        names = {f.name for f in fields(Config)}
        for section in [s for s in ORDER if s in names]:
            title, text = SECTION_INFO.get(section, (section, ""))
            card = Card(title, text)
            keys = [f.name for f in fields(getattr(self.state.cfg, section)) if f.name not in SKIP.get(section, ())]
            form = ConfigForm(self.state, section, keys=keys, show_advanced=False)
            card.body.addWidget(form)
            self.cards.append((card, form))
        host = QWidget()
        cols = QHBoxLayout(host)
        cols.setContentsMargins(0, 0, 6, 0)
        self.columns = [QVBoxLayout(), QVBoxLayout()]
        for c in self.columns:
            cols.addLayout(c, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)

        page = QVBoxLayout(self)
        page.addWidget(page_header("Settings", "Saved in config.json and used by the whole app. The robot is on "
                                               "Bot tuning; the match on Play. Hover any setting for what it does."))
        page.addLayout(size_row)
        page.addLayout(top)
        page.addWidget(scroll, 1)
        self._layout_cards(False)

    def _layout_cards(self, show_advanced: bool):
        """Shows the cards that have visible fields, flowing them into two columns."""
        for col in self.columns:
            while col.count():
                item = col.takeAt(0)
                if item.widget() is not None:
                    item.widget().setParent(None)
        visible = []
        for card, form in self.cards:
            form.set_show_advanced(show_advanced)
            if show_advanced or any(not w.isHidden() for ws in form.rows.values() for w in ws):
                visible.append(card)
        heights = [0, 0]
        for card in visible:                    # put each card in the shorter column
            i = 0 if heights[0] <= heights[1] else 1
            self.columns[i].addWidget(card)
            card.show()
            heights[i] += card.sizeHint().height()
        for col in self.columns:
            col.addStretch(1)

    # ------------------------------------------------------------------ interface size
    def _saved_size(self) -> str:
        prefs = self.main.prefs
        value = prefs.value(scale.PREF_KEY, "auto") if prefs is not None else getattr(self, "_size_value", "auto")
        return str(value)

    def _size_changed(self, _i):
        value = self.size.currentData()
        if self.main.prefs is not None:
            self.main.prefs.setValue(scale.PREF_KEY, value)
        self._size_value = value
        self.restart_btn.setVisible(True)
        self.size_note.setText(f"Now: {scale.current() * 100:.0f}%. Restart to apply.")

    def _restart(self):
        if not self.main.restart():
            self.size_note.setText("Close and reopen the app to apply.")

    def _reset(self):
        if QMessageBox.question(self, "Reset settings", "Reset every setting to its default? (Nothing is saved until "
                                                        "you press Save.)") == QMessageBox.StandardButton.Yes:
            self.state.reset_defaults()
