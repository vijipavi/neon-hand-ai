"""Unit tests for effects.particle_effects and effects.hologram."""
import unittest
import numpy as np

from effects.hologram import HologramConfig, HologramRenderer
from effects.particle_effects import ParticleConfig, ParticleSystem
from tests.test_landmark_processor import _make_open_hand
from tracking.landmark_processor import LandmarkProcessor


class TestAdvancedEffects(unittest.TestCase):
    """Verifies particle caps, lifetime expiration, hologram response, and multi-hand support."""

    def setUp(self):
        self.processor = LandmarkProcessor(smoothing_alpha=1.0)
        self.left = self.processor.process_hand(_make_open_hand("Left", -100), 640, 480)
        self.right = self.processor.process_hand(_make_open_hand("Right", 100), 640, 480)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_particle_cap_and_lifetime_expiration(self):
        cfg = ParticleConfig(max_particles=25, spawn_per_tip=5, lifetime_sec=0.2)
        ps = ParticleSystem(config=cfg)

        for _ in range(10):
            ps.update([self.left, self.right], dt=0.016)
        self.assertLessEqual(ps.particle_count, 25)
        self.assertGreater(ps.particle_count, 0)

        # Advance time with no hands so all particles expire cleanly
        for _ in range(20):
            ps.update([], dt=0.05)
        self.assertEqual(ps.particle_count, 0)

    def test_particle_render_two_hands(self):
        ps = ParticleSystem()
        out = ps.render(self.frame.copy(), [self.left, self.right], dt=0.016)
        self.assertEqual(out.shape, self.frame.shape)
        self.assertGreater(int(out.sum()), 0)

    def test_hologram_render_and_motion_response(self):
        holo = HologramRenderer(HologramConfig(show_telemetry=True))
        out1 = holo.render(self.frame.copy(), [self.left, self.right], phase=0.0)
        out2 = holo.render(self.frame.copy(), [self.left, self.right], phase=0.8)

        self.assertEqual(out1.shape, self.frame.shape)
        self.assertGreater(int(out1.sum()), 0)
        self.assertFalse(np.array_equal(out1, out2))


if __name__ == "__main__":
    unittest.main()

