"""Unit tests for tracking.hand_detector module."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np

from configuration.settings import TrackingConfig
from tracking.hand_detector import DetectedHand, HandDetector


class TestHandDetector(unittest.TestCase):
    """Verifies MediaPipe HandDetector initialization, extraction, drawing, and empty handling."""

    def test_no_hands_detected_on_blank_frame(self):
        cfg = TrackingConfig(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            model_path="nonexistent_model.task",
        )
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        with HandDetector(config=cfg) as detector:
            hands = detector.detect(blank)
            self.assertEqual(hands, [])
            rendered = detector.draw_landmarks(blank.copy(), hands)
            self.assertEqual(rendered.shape, blank.shape)

    def test_extracts_21_landmarks_world_landmarks_and_handedness(self):
        cfg = TrackingConfig(max_num_hands=2, model_path="nonexistent_model.task")
        detector = HandDetector(config=cfg)

        # Build mock MediaPipe Solutions output for 1 hand with 21 landmarks
        mock_lms = [SimpleNamespace(x=i * 0.04, y=0.5, z=-0.01 * i) for i in range(21)]
        mock_world_lms = [SimpleNamespace(x=0.01 * i, y=-0.02, z=0.005) for i in range(21)]
        mock_result = SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=mock_lms)],
            multi_hand_world_landmarks=[SimpleNamespace(landmark=mock_world_lms)],
            multi_handedness=[
                SimpleNamespace(classification=[SimpleNamespace(label="Right", score=0.97)])
            ],
        )
        detector._hands = MagicMock()
        detector._hands.process.return_value = mock_result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = detector.detect(frame)

        self.assertEqual(len(hands), 1)
        hand: DetectedHand = hands[0]
        self.assertEqual(hand.handedness, "Right")
        self.assertAlmostEqual(hand.score, 0.97)
        self.assertEqual(len(hand.landmarks_norm), 21)
        self.assertEqual(len(hand.landmarks_px), 21)
        self.assertIsNotNone(hand.world_landmarks)
        self.assertEqual(len(hand.world_landmarks), 21)

        drawn = detector.draw_landmarks(frame.copy(), hands)
        self.assertEqual(drawn.shape, frame.shape)
        self.assertGreater(int(drawn.sum()), 0)
        detector.close()


if __name__ == "__main__":
    unittest.main()

