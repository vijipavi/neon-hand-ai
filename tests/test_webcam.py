"""Unit tests for camera.webcam module."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from camera.webcam import Webcam, WebcamError


class TestWebcam(unittest.TestCase):
    """Verifies Webcam initialization, resolution config, error handling, and ESC shutdown."""

    @patch("camera.webcam.cv2.VideoCapture")
    def test_open_unavailable_webcam_raises_error(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        cam = Webcam(camera_index=0, width=640, height=480)
        with self.assertRaises(WebcamError):
            cam.open()

        self.assertFalse(cam.is_running)

    @patch("camera.webcam.cv2.VideoCapture")
    def test_configurable_resolution_and_read_frame(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else 480
        mock_cap_cls.return_value = mock_cap

        with Webcam(camera_index=0, width=640, height=480, flip_horizontal=False) as cam:
            self.assertTrue(cam.is_running)
            self.assertEqual(cam.actual_resolution, (640, 480))
            frame = cam.read_frame()
            self.assertEqual(frame.shape, (480, 640, 3))

        self.assertFalse(cam.is_running)
        mock_cap.release.assert_called()

    @patch("camera.webcam.cv2.destroyAllWindows")
    @patch("camera.webcam.cv2.getWindowProperty", return_value=1.0)
    @patch("camera.webcam.cv2.waitKey", return_value=27)
    @patch("camera.webcam.cv2.imshow")
    @patch("camera.webcam.cv2.resizeWindow")
    @patch("camera.webcam.cv2.namedWindow")
    @patch("camera.webcam.cv2.VideoCapture")
    def test_display_shuts_down_on_esc(
        self,
        mock_cap_cls,
        mock_named_win,
        mock_resize_win,
        mock_imshow,
        mock_wait_key,
        mock_win_prop,
        mock_destroy,
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: 1280 if prop == 3 else 720
        mock_cap_cls.return_value = mock_cap

        cam = Webcam(width=1280, height=720)
        cam.display()

        mock_imshow.assert_called_once()
        mock_destroy.assert_called_once()
        self.assertFalse(cam.is_running)


if __name__ == "__main__":
    unittest.main()

