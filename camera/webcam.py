"""
Webcam Capture & Display Module for NEON HAND AI.
Provides a modular, reusable interface for accessing the default webcam,
configuring resolution, capturing continuous video frames, and handling
clean window display and ESC-key shutdown.
"""

import sys
from typing import Callable, Iterator, Optional, Tuple
import cv2
import numpy as np

from configuration.settings import CameraConfig, settings


class WebcamError(RuntimeError):
    """Raised when the webcam cannot be opened, configured, or read."""


class Webcam:
    """
    Modular OpenCV webcam controller with configurable resolution,
    error handling, and clean resource management.
    """

    ESC_KEY_CODE = 27

    def __init__(
        self,
        camera_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        flip_horizontal: Optional[bool] = None,
        window_name: Optional[str] = None,
        config: Optional[CameraConfig] = None,
    ) -> None:
        base_cfg = config or settings.camera

        self.camera_index: int = base_cfg.device_index if camera_index is None else camera_index
        self.width: int = base_cfg.frame_width if width is None else width
        self.height: int = base_cfg.frame_height if height is None else height
        self.fps: int = base_cfg.fps if fps is None else fps
        self.flip_horizontal: bool = (
            base_cfg.flip_horizontal if flip_horizontal is None else flip_horizontal
        )
        self.buffer_size: int = base_cfg.buffer_size
        self.window_name: str = window_name or settings.ui.window_name

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False

    def open(self) -> None:
        """
        Opens the webcam device and applies the configured resolution and FPS.
        Raises WebcamError if the device is unavailable.
        """
        if self._is_opened and self._cap is not None and self._cap.isOpened():
            return

        # Use DirectShow on Windows for faster camera initialization
        if sys.platform.startswith("win"):
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = cv2.VideoCapture(self.camera_index)
        else:
            self._cap = cv2.VideoCapture(self.camera_index)

        if self._cap is None or not self._cap.isOpened():
            self.release()
            raise WebcamError(
                f"Unable to access webcam at device index {self.camera_index}. "
                "Ensure a camera is connected, not in use by another app, and permissions are enabled."
            )

        # Apply configurable resolution and capture properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self._is_opened = True

    @property
    def is_running(self) -> bool:
        """Returns True if the webcam is currently open and ready to read."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        """Returns the (width, height) negotiated with the camera hardware."""
        if not self.is_running or self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def read_frame(self) -> np.ndarray:
        """
        Captures and returns a single BGR frame from the webcam.
        Raises WebcamError if the camera is not open or frame capture fails.
        """
        if not self.is_running or self._cap is None:
            raise WebcamError("Webcam is not opened. Call open() before reading frames.")

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            raise WebcamError(
                f"Failed to read frame from webcam at index {self.camera_index}. "
                "The camera may have been disconnected."
            )

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """
        Continuous frame generator for modular downstream processing.
        Ensures the webcam is opened and yields frames until stopped.
        """
        if not self.is_running:
            self.open()

        while self.is_running:
            yield self.read_frame()

    def display(
        self,
        frame_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """
        Runs a stable video capture and display loop in an OpenCV window.
        Shuts down cleanly when the ESC key is pressed or the window is closed.

        Args:
            frame_processor: Optional callback taking a BGR frame and returning a processed frame.
        """
        self.open()
        actual_w, actual_h = self.actual_resolution
        print(f"[✓] Webcam opened (index={self.camera_index}, resolution={actual_w}x{actual_h}).")
        print("[*] Press ESC in the video window to exit.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, actual_w or self.width, actual_h or self.height)

        try:
            for frame in self.frames():
                if frame_processor is not None:
                    frame = frame_processor(frame)

                cv2.imshow(self.window_name, frame)

                # Poll keyboard events (~1ms) to keep the UI responsive
                key = cv2.waitKey(1) & 0xFF
                if key == self.ESC_KEY_CODE:
                    print("[*] ESC key pressed. Shutting down webcam...")
                    break

                # Also exit cleanly if the user clicks the window's close [X] button
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[*] Window closed. Shutting down webcam...")
                    break
        except KeyboardInterrupt:
            print("\n[*] Keyboard interrupt received. Shutting down webcam...")
        finally:
            self.release()
            cv2.destroyAllWindows()
            print("[✓] Webcam resources released cleanly.")

    def release(self) -> None:
        """Releases OpenCV VideoCapture hardware resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def __enter__(self) -> "Webcam":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


if __name__ == "__main__":
    try:
        webcam = Webcam()
        webcam.display()
    except WebcamError as exc:
        print(f"[X] Webcam Error: {exc}", file=sys.stderr)
        sys.exit(1)

