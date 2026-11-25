"""Unit tests for tracking.landmark_processor module."""
import unittest

from tracking.hand_detector import DetectedHand
from tracking.landmark_processor import (
    LandmarkProcessor,
    LandmarkSmoother,
    calculate_angle,
    euclidean_distance,
    normalize_point,
)


def _make_open_hand(handedness: str = "Right", offset_x: int = 0) -> DetectedHand:
    """Creates a synthetic 21-landmark open hand in pixel coordinates."""
    pts = [(0, 0)] * 21
    # Wrist & Palm anchors
    pts[0] = (300 + offset_x, 400)
    pts[1] = (260 + offset_x, 370)
    pts[2] = (230 + offset_x, 340)
    pts[3] = (205 + offset_x, 310)
    pts[4] = (180 + offset_x, 280)  # Thumb tip extended outward

    # Index, Middle, Ring, Pinky extended straight up
    pts[5], pts[6], pts[7], pts[8] = (270 + offset_x, 310), (270 + offset_x, 250), (270 + offset_x, 210), (270 + offset_x, 170)
    pts[9], pts[10], pts[11], pts[12] = (300 + offset_x, 305), (300 + offset_x, 240), (300 + offset_x, 195), (300 + offset_x, 155)
    pts[13], pts[14], pts[15], pts[16] = (330 + offset_x, 310), (330 + offset_x, 250), (330 + offset_x, 210), (330 + offset_x, 175)
    pts[17], pts[18], pts[19], pts[20] = (360 + offset_x, 325), (360 + offset_x, 275), (360 + offset_x, 240), (360 + offset_x, 205)

    norm = [(x / 640.0, y / 480.0, 0.0) for x, y in pts]
    return DetectedHand(
        handedness=handedness,
        score=0.98,
        landmarks_norm=norm,
        landmarks_px=pts,
    )


class TestLandmarkProcessor(unittest.TestCase):
    """Verifies math helpers, smoothing, finger counting, and multi-hand isolation."""

    def test_math_helpers(self):
        self.assertAlmostEqual(euclidean_distance((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(calculate_angle((0, 1), (0, 0), (1, 0)), 90.0)
        self.assertAlmostEqual(calculate_angle((0, 0), (1, 0), (2, 0)), 180.0)
        nx, ny = normalize_point((320, 240), 640, 480)
        self.assertAlmostEqual(nx, 0.5)
        self.assertAlmostEqual(ny, 0.5)

    def test_open_hand_extraction_and_finger_count(self):
        processor = LandmarkProcessor(smoothing_alpha=1.0)
        hand = _make_open_hand("Right")
        ph = processor.process_hand(hand, 640, 480)

        self.assertEqual(ph.wrist, (300, 400))
        self.assertEqual(len(ph.fingertips), 5)
        self.assertEqual(ph.fingertips["index"], (270, 170))
        self.assertEqual(ph.finger_count, 5)
        self.assertTrue(all(ph.extended_fingers.values()))
        self.assertGreater(ph.thumb_index_distance_px, 0.0)

    def test_curled_fingers_reduce_count(self):
        processor = LandmarkProcessor(smoothing_alpha=1.0)
        hand = _make_open_hand("Right")
        # Curl ring (16) and pinky (20) down toward the wrist
        hand.landmarks_px[16] = (330, 390)
        hand.landmarks_px[20] = (360, 395)

        ph = processor.process_hand(hand, 640, 480)
        self.assertFalse(ph.extended_fingers["ring"])
        self.assertFalse(ph.extended_fingers["pinky"])
        self.assertEqual(ph.finger_count, 3)

    def test_ema_smoothing_and_multi_hand_independence(self):
        smoother = LandmarkSmoother(alpha=0.5)
        first = smoother.smooth("Right_0", [(100, 100)])
        second = smoother.smooth("Right_0", [(110, 120)])
        self.assertEqual(first, [(100, 100)])
        self.assertEqual(second, [(105, 110)])

        processor = LandmarkProcessor(smoothing_alpha=0.5)
        left_hand = _make_open_hand("Left", offset_x=-100)
        right_hand = _make_open_hand("Right", offset_x=100)
        results = processor.process_hands([left_hand, right_hand], 640, 480)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].handedness, "Left")
        self.assertEqual(results[1].handedness, "Right")
        self.assertNotEqual(results[0].palm_center, results[1].palm_center)


if __name__ == "__main__":
    unittest.main()

