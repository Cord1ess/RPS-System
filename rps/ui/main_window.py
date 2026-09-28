"""
Main window: page tabs, the shared camera worker and the status bar. Remembers its size and the last
page between runs, and asks before closing with unsaved settings.
"""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget

from rps.ui.camera_worker import CameraWorker
from rps.ui.common import AppState
from rps.ui.tab_bot import BotTab
from rps.ui.tab_dataset import DatasetTab
from rps.ui.tab_evaluate import EvaluateTab
from rps.ui.tab_game import GameTab
from rps.ui.tab_play import PlayTab
from rps.ui.tab_record import RecordTab
from rps.ui.tab_settings import SettingsTab
from rps.ui.tab_setup import SetupTab
from rps.ui.tab_train import TrainTab

TAB_TIPS = {
    "1  Setup": "Camera, light and play zone. Do this first.",
    "2  Bot tuning": "The robot hand: connection, commands and tests.",
    "3  Record": "Record labelled gesture sessions.",
    "4  Dataset": "Turn recordings into training images and check them.",
    "5  Train": "Tune Dextra on your recordings: makes Dextra Tuned.",
    "6  Evaluate": "Compare Dextra Raw, Dextra Tuned, Mediapipe and Both on recordings.",
    "7  Play": "The demo: a match against the robot, with the beat guide.",
    "Play Debug": "Every reading and game rule, for tuning.",
    "Settings": "Every other setting. Hover a setting for what it does.",
}


class MainWindow(QMainWindow):
    def __init__(self, state: AppState, start_kind: str = "camera", remember: bool = True):
        super().__init__()
        self.state = state
        self.confirm_close = True            # ask to save unsaved settings when closing
        self.prefs = QSettings("RPS-System", "app") if remember else None
        self.default_kind, self.default_video = start_kind, None
        self.camera_blocked = False          # a Setup tool (auto-configure, delay test) owns the webcam
        self.setWindowTitle("Rock-Paper-Scissors robot")
        self.resize(1440, 900)
        self.worker = CameraWorker(state)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.state_changed.connect(self._on_camera_state)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.pages = [SetupTab(self), BotTab(self), RecordTab(self), DatasetTab(self), TrainTab(self),
                      EvaluateTab(self), GameTab(self), PlayTab(self), SettingsTab(self)]
        for i, page in enumerate(self.pages):
            self.tabs.addTab(page, page.title)
            self.tabs.setTabToolTip(i, TAB_TIPS.get(page.title, ""))
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)

        self.cfg_status = QLabel()
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Save every setting to config.json; the app starts with them next time.")
        self.save_btn.clicked.connect(self.state.save)
        self.statusBar().addPermanentWidget(self.cfg_status)
        self.statusBar().addPermanentWidget(self.save_btn)
        state.config_changed.connect(self._update_cfg_status)
        self._update_cfg_status()
        self._current = self.pages[0]
        start = 0
        if self.prefs is not None:
            geometry = self.prefs.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
            start = int(self.prefs.value("page", 0) or 0)
            if not 0 <= start < len(self.pages):
                start = 0
        if start:
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(start)
            self.tabs.blockSignals(False)
        self._tab_changed(start)

    # ------------------------------------------------------------------ camera ownership
    def start_camera(self, kind: str = None, video: str = None):
        """Starts the camera. Without arguments, reopens the source last chosen on Setup."""
        if self.worker.isRunning():
            return
        if self.camera_blocked:
            self._show_on_views("The camera is in use by a Setup tool. It restarts when the tool finishes.")
            return
        if kind is not None:
            self.default_kind, self.default_video = kind, video
        self._show_on_views("Starting camera...")
        self.worker.begin(self.default_kind, self.default_video)

    def _show_on_views(self, message: str):
        for page in self.pages:
            view = getattr(page, "view", None)
            if view is not None:
                view.show_status(message)

    def stop_camera(self):
        if self.worker.isRunning():
            self.worker.stop()

    def current_page(self):
        return self._current

    def lock_tabs(self, locked: bool, owner):
        """Keeps the user on `owner` (e.g. while recording) so the camera cannot be pulled away."""
        for i, page in enumerate(self.pages):
            self.tabs.setTabEnabled(i, (not locked) or page is owner)

    # ------------------------------------------------------------------ routing
    def _tab_changed(self, index: int):
        new = self.pages[index]
        if new is not self._current:
            self._current.on_deactivated()
        self._current = new
        self.worker.set_processor(new.processor if new.uses_camera else None)
        new.on_activated()

    def _on_frame(self):
        payload = self.worker.take_latest()
        if payload is None:
            return
        if self._current.uses_camera:
            self._current.on_frame(payload)

    def _on_camera_state(self, s: str):
        if s == "stalled":
            self.statusBar().showMessage("The camera stopped sending images. Close other apps using it.", 8000)
        elif s == "stopped":
            self._show_on_views("Camera off")
        elif s.startswith("error"):
            self._show_on_views("Camera could not be opened. Close other apps using it, then try again.")
            self.statusBar().showMessage("Camera could not be opened: see the Setup page.", 8000)

    def _update_cfg_status(self):
        self.cfg_status.setText("Settings: unsaved changes" if self.state.dirty else "Settings: saved")
        self.save_btn.setVisible(self.state.dirty)

    def closeEvent(self, event):
        if self.confirm_close and self.state.dirty:
            answer = QMessageBox.question(
                self, "Save settings?", "Some settings changed since the last save. Save them so the app starts with "
                "them next time?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                self.state.save()
        if self.prefs is not None:
            self.prefs.setValue("geometry", self.saveGeometry())
            self.prefs.setValue("page", self.tabs.currentIndex())
        for page in self.pages:
            page.shutdown()
        self.worker.stop()
        self.state.cleanup()
        super().closeEvent(event)
