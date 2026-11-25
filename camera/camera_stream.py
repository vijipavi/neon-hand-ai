"""
Camera Stream Manager.
Encapsulates OpenCV VideoCapture with robust device initialization,
error handling, and frame pre-processing (mirroring, resizing).
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from configuration.settings import CameraConfig


class CameraStream:
    """Manages video capture lifecycle and frame retrieval."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False

    def start(self) -> bool:
        """
        Initializes the video capture device with target resolution and frame rate.
        Returns True if successful, False otherwise.
        """
        # cv2.CAP_DSHOW provides fast startup on Windows
        self._cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            # Fallback to default backend if DirectShow is unavailable
            self._cap = cv2.VideoCapture(self.config.device_index)

        if not self._cap.isOpened():
            self._is_opened = False
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        self._is_opened = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads and pre-processes the next frame from the camera stream.
        Applies horizontal flip for natural mirror interaction if configured.
        """
        if not self._is_opened or self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    def stop(self) -> None:
        """Releases the camera hardware resources cleanly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    @property
    def is_running(self) -> bool:
        """Checks if camera capture stream is active."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

