"""Main window: page tabs, the shared camera worker, and a two-item status bar."""

from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget

from rps.ui.camera_worker import CameraWorker
from rps.ui.common import AppState
from rps.ui.tab_dataset import DatasetTab
from rps.ui.tab_evaluate import EvaluateTab
from rps.ui.tab_play import PlayTab
from rps.ui.tab_record import RecordTab
from rps.ui.tab_settings import SettingsTab
from rps.ui.tab_setup import SetupTab
from rps.ui.tab_train import TrainTab

TAB_TIPS = {
    "1  Setup": "Camera, light, play zone and robot connection. Do this first.",
    "2  Record": "Record labelled gesture sessions.",
    "3  Dataset": "Turn recordings into training images and check them.",
    "4  Train": "Train the motion model on your recordings.",
    "5  Evaluate": "Measure how well Dextra's model and each recognition method work.",
    "6  Play": "Play against the robot hand.",
    "Settings": "Every setting. Hover a setting for what it does.",
}


class MainWindow(QMainWindow):
    def __init__(self, state: AppState, start_kind: str = "camera"):
        super().__init__()
        self.state = state
        self.default_kind, self.default_video = start_kind, None
        self.camera_blocked = False          # a Setup tool (auto-configure, delay test) owns the webcam
        self.setWindowTitle("Rock-Paper-Scissors robot")
        self.resize(1440, 900)
        self.worker = CameraWorker(state)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.state_changed.connect(self._on_camera_state)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.pages = [SetupTab(self), RecordTab(self), DatasetTab(self), TrainTab(self), EvaluateTab(self),
                      PlayTab(self), SettingsTab(self)]
        for i, page in enumerate(self.pages):
            self.tabs.addTab(page, page.title)
            self.tabs.setTabToolTip(i, TAB_TIPS.get(page.title, ""))
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)

        self.cfg_status = QLabel()
        self.statusBar().addPermanentWidget(self.cfg_status)
        state.config_changed.connect(self._update_cfg_status)
        self._update_cfg_status()
        self._current = self.pages[0]
        self._tab_changed(0)

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

    def closeEvent(self, event):
        for page in self.pages:
            page.shutdown()
        self.worker.stop()
        self.state.cleanup()
        super().closeEvent(event)
