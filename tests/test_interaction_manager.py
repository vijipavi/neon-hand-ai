"""Unit tests for gestures.interaction_manager and effects.interaction_renderer."""
import unittest
import numpy as np

from effects.interaction_renderer import InteractionRenderer
from gestures.gesture_recognition import GestureRecognizer
from gestures.interaction_manager import InteractionManager
from tests.test_gesture_recognition import (
    _make_closed_fist,
    _make_index_pointing,
    _make_peace_sign,
    _make_pinch,
    _make_thumbs_up,
)
from tests.test_landmark_processor import _make_open_hand
from tracking.landmark_processor import LandmarkProcessor


class TestInteractionSystem(unittest.TestCase):
    """Verifies all 6 gesture-to-AR interactions, edge triggering, and rendering."""

    def setUp(self):
        self.processor = LandmarkProcessor(smoothing_alpha=1.0)
        self.recognizer = GestureRecognizer(stabilization_window=1, min_confirmations=1)
        self.manager = InteractionManager(action_cooldown_sec=0.0)
        self.renderer = InteractionRenderer()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _step(self, raw_hand, dt: float = 0.016):
        ph = self.processor.process_hands([raw_hand], 640, 480)
        preds = self.recognizer.recognize_hands(ph, 640, 480)
        return self.manager.update(ph, preds, dt=dt)

    def test_peace_sign_cycles_theme_once_per_hold(self):
        initial_theme = self.manager.theme_index
        # Holding Peace Sign across multiple frames only triggers 1 theme change
        s1 = self._step(_make_peace_sign())
        s2 = self._step(_make_peace_sign())
        self.assertEqual(s1.theme_index, (initial_theme + 1) % len(self.manager.themes))
        self.assertEqual(s2.theme_index, s1.theme_index)

    def test_open_palm_fist_and_thumbs_up_interactions(self):
        s_fist = self._step(_make_closed_fist())
        self.assertTrue(s_fist.special_effect_active)
        self.assertIsNotNone(s_fist.shockwave_progress)

        s_palm = self._step(_make_open_hand())
        self.assertTrue(s_palm.hologram_active)
        self.assertFalse(s_palm.special_effect_active)

        s_thumb = self._step(_make_thumbs_up())
        self.assertIsNotNone(s_thumb.confirmation_progress)

    def test_index_pointing_smooth_cursor_and_pinch_zoom(self):
        s_cursor = self._step(_make_index_pointing())
        self.assertTrue(s_cursor.cursor_active)
        self.assertIsNotNone(s_cursor.cursor_pos)

        initial_zoom = self.manager.zoom_scale
        s_pinch = self._step(_make_pinch())
        self.assertTrue(s_pinch.zoom_active)
        self.assertNotEqual(s_pinch.zoom_scale, initial_zoom)

        out = self.renderer.render(self.frame.copy(), s_pinch, phase=0.5)
        self.assertEqual(out.shape, self.frame.shape)
        self.assertGreater(int(out.sum()), 0)


if __name__ == "__main__":
    unittest.main()

