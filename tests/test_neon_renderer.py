"""Unit tests for effects.neon_renderer module."""
import unittest
import numpy as np

from configuration.settings import NeonColors, NeonEffectConfig
from effects.neon_renderer import NeonPalette, NeonRenderer
from tests.test_landmark_processor import _make_open_hand
from tracking.landmark_processor import LandmarkProcessor


class TestNeonRenderer(unittest.TestCase):
    """Verifies neon skeleton, fingertip halos, palm animation, two-hand rendering, and config."""

    def setUp(self):
        self.processor = LandmarkProcessor(smoothing_alpha=1.0)
        self.left = self.processor.process_hand(_make_open_hand("Left", -100), 640, 480)
        self.right = self.processor.process_hand(_make_open_hand("Right", 100), 640, 480)
        self.frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    def test_render_empty_hands_dims_backdrop_cleanly(self):
        renderer = NeonRenderer(backdrop_dim=0.5)
        out = renderer.render_hands(self.frame, [])
        self.assertEqual(out.shape, self.frame.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out == 20))

    def test_two_hand_render_and_animation_phase(self):
        renderer = NeonRenderer()
        out_t0 = renderer.render_hands(self.frame, [self.left, self.right], phase=0.0)
        out_t1 = renderer.render_hands(self.frame, [self.left, self.right], phase=1.2)

        self.assertEqual(out_t0.shape, self.frame.shape)
        self.assertGreater(int(out_t0.sum()), int(self.frame.sum()))
        # Animated palm reticle produces distinct frames at different phases
        self.assertFalse(np.array_equal(out_t0, out_t1))

    def test_configurable_intensity_and_palette(self):
        cfg = NeonEffectConfig(glow_intensity=0.2)
        palette = NeonPalette(
            skeleton_color=NeonColors.ELECTRIC_BLUE,
            joint_color=NeonColors.CYAN,
            accent_color=NeonColors.ELECTRIC_PURPLE,
        )
        renderer = NeonRenderer(config=cfg, palette=palette)
        low_glow = renderer.render_hand(self.frame, self.right, phase=0.5)

        renderer.set_glow_intensity(1.5)
        high_glow = renderer.render_hand(self.frame, self.right, phase=0.5)
        self.assertGreater(int(high_glow.sum()), int(low_glow.sum()))


if __name__ == "__main__":
    unittest.main()

