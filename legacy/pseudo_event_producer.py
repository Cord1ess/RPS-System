"""
Pseudo-Event Producer: Webcam Motion Masking Pipeline.
Replaces DVS event camera stream with consecutive frame differencing + binary thresholding.
Produces 64x64 uint8 binary motion event masks for model input.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional


class PseudoEventProducer:
    """
    Captures live webcam stream, computes absolute difference between consecutive frames,
    applies binary thresholding to isolate sharp motion boundaries, and resizes to 64x64.
    """
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        threshold_value: int = 18,
        target_size: Tuple[int, int] = (64, 64),
        blur_kernel: int = 5,
        use_roi_crop: bool = True,
        mock_mode: bool = False,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.threshold_value = threshold_value
        self.target_size = target_size
        self.blur_kernel = blur_kernel
        self.use_roi_crop = use_roi_crop
        self.mock_mode = mock_mode

        self.cap = None
        self.prev_gray = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Synthetic mock generator state for testing environments without webcam
        self.mock_phase = 0.0

        if not self.mock_mode:
            self._init_camera()

    def _init_camera(self):
        """Attempts to open OpenCV VideoCapture with configured exposure and dimensions."""
        print(f"[PseudoEventProducer] Opening camera index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"[PseudoEventProducer] Warning: Failed to open camera {self.camera_index}. Falling back to mock generator.")
            self.mock_mode = True
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Attempt to configure manual exposure to mitigate camera auto-gain noise
        # 0.25 is manual mode in V4L2 backend on Linux, 1 or 3 depending on driver
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # typical value for low exposure/high speed
        except Exception as e:
            print(f"[PseudoEventProducer] Note: Auto-exposure adjustment not supported on this device ({e})")

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[PseudoEventProducer] Camera opened successfully at resolution {int(actual_w)}x{int(actual_h)}")

    def get_motion_mask(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reads next frame, computes temporal diff |F_t - F_{t-1}|, thresholds, and produces:
        - success: bool
        - raw_frame: full-resolution BGR camera frame (or mock frame)
        - motion_mask_64: 64x64 uint8 binary mask [0 or 255]
        """
        self.frame_count += 1
        now = time.time()
        self.last_frame_time = now

        if self.mock_mode or self.cap is None:
            return self._generate_mock_frame()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, None

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # Initialize previous frame on startup
        if self.prev_gray is None:
            self.prev_gray = gray
            motion_mask_64 = np.zeros(self.target_size, dtype=np.uint8)
            return True, frame, motion_mask_64

        # Compute absolute temporal frame difference: |Frame_t - Frame_t-1|
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        # Binary thresholding to produce sharp motion trail
        _, thresh = cv2.threshold(diff, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Extract Center ROI (square region where user's hand is displayed)
        h, w = thresh.shape
        min_dim = min(h, w)
        if self.use_roi_crop:
            # Center square crop
            sy = (h - min_dim) // 2
            sx = (w - min_dim) // 2
            roi_thresh = thresh[sy : sy + min_dim, sx : sx + min_dim]
        else:
            roi_thresh = thresh

        # Resize to 64x64 (matches model tensor shape)
        motion_mask_64 = cv2.resize(roi_thresh, self.target_size, interpolation=cv2.INTER_AREA)

        return True, frame, motion_mask_64

    def _generate_mock_frame(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Generates realistic synthetic motion masks when webcam is unavailable or in mock test."""
        self.mock_phase += 0.15
        h, w = self.height, self.width
        raw_frame = np.full((h, w, 3), 35, dtype=np.uint8)
        
        # Draw mock interface elements
        cv2.putText(raw_frame, "MOCK CAMERA STREAM (Simulated)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Create moving motion mask in 64x64
        mask = np.zeros((64, 64), dtype=np.uint8)
        cx = int(32 + 18 * np.sin(self.mock_phase))
        cy = int(32 + 18 * np.cos(self.mock_phase))
        cv2.circle(mask, (cx, cy), 10, 255, -1)
        cv2.circle(mask, (cx + 5, cy - 5), 4, 255, -1)

        # Draw ROI on raw frame
        min_dim = min(h, w)
        sy = (h - min_dim) // 2
        sx = (w - min_dim) // 2
        cv2.rectangle(raw_frame, (sx, sy), (sx + min_dim, sy + min_dim), (0, 255, 0), 2)
        cv2.putText(raw_frame, "Gesture ROI", (sx + 10, sy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        time.sleep(1.0 / self.fps)
        return True, raw_frame, mask

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[PseudoEventProducer] Camera released.")
