"""Unit tests for the PySide6 UI (ui.main_window) with offscreen platform."""
import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from ui.main_window import NeonMainWindow
from ui.vision_worker import HandTelemetry, TelemetrySnapshot


class TestNeonMainWindow(unittest.TestCase):
    """Verifies UI controls, telemetry updates, and CV separation."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_window_controls_and_telemetry_update(self):
        mock_engine = MagicMock()
        window = NeonMainWindow(engine=mock_engine)

        # Verify initial controls
        self.assertTrue(window.start_btn.isEnabled())
        self.assertFalse(window.stop_btn.isEnabled())
        self.assertEqual(window.camera_combo.count(), 3)
        self.assertEqual(window.resolution_combo.count(), 3)

        # Glow intensity slider delegates to engine
        window.glow_slider.setValue(125)
        mock_engine.set_glow_intensity.assert_called_with(1.25)

        # Toggles delegate to engine
        window.particles_chk.setChecked(False)
        mock_engine.set_particles_enabled.assert_called_with(False)
        window.hologram_chk.setChecked(False)
        mock_engine.set_hologram_enabled.assert_called_with(False)

        # Theme selection delegates to engine
        window.theme_combo.setCurrentIndex(1)
        mock_engine.set_theme_index.assert_called_with(1)

        # Telemetry snapshot updates labels and confidence bars
        snap = TelemetrySnapshot(
            fps=30.0,
            hands_detected=2,
            left_hand=HandTelemetry(True, "PEACE_SIGN", 2, 0.92, 0.88),
            right_hand=HandTelemetry(True, "OPEN_PALM", 5, 0.96, 0.95),
            active_gesture="OPEN_PALM",
            theme_name="SYNTHWAVE PLASMA",
            zoom_scale=1.45,
            feedback_message="THEME: SYNTHWAVE PLASMA",
        )
        window.update_telemetry(snap)
        self.assertEqual(window.fps_badge.text(), "FPS: 30")
        self.assertIn("2 hand", window.detection_status_lbl.text())
        self.assertIn("PEACE_SIGN", window.left_info_lbl.text())
        self.assertIn("OPEN_PALM", window.right_info_lbl.text())
        self.assertEqual(window.left_conf_bar.value(), 92)
        self.assertEqual(window.right_conf_bar.value(), 96)
        self.assertIn("1.45x", window.zoom_lbl.text())

        window.close()


if __name__ == "__main__":
    unittest.main()

