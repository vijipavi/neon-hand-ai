"""Test environment and module structure sanity test."""
import unittest
import importlib


class TestProjectStructure(unittest.TestCase):
    """Verifies that all modular packages and settings can be loaded cleanly."""

    def test_imports(self):
        modules = [
            "configuration.settings",
            "settings",
            "camera.camera_stream",
            "tracking.hand_tracker",
            "gestures.gesture_detector",
            "effects.neon_renderer",
            "ui.hud_overlay",
        ]
        for mod in modules:
            with self.subTest(module=mod):
                imported = importlib.import_module(mod)
                self.assertIsNotNone(imported)


if __name__ == "__main__":
    unittest.main()

