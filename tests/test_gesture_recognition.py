"""Unit tests for gestures.gesture_recognition module."""
import unittest

from gestures.gesture_recognition import (
    BaseGestureClassifier,
    GestureLabel,
    GestureRecognizer,
)
from tests.test_landmark_processor import _make_open_hand
from tracking.hand_detector import DetectedHand


def _make_closed_fist(handedness: str = "Right", offset_x: int = 0) -> DetectedHand:
    hand = _make_open_hand(handedness, offset_x)
    pts = hand.landmarks_px
    # Tuck thumb inward & curl all 4 fingers down toward palm
    pts[3] = (275 + offset_x, 340)
    pts[4] = (295 + offset_x, 345)
    pts[6], pts[7], pts[8] = (270 + offset_x, 325), (270 + offset_x, 350), (270 + offset_x, 365)
    pts[10], pts[11], pts[12] = (300 + offset_x, 320), (300 + offset_x, 348), (300 + offset_x, 365)
    pts[14], pts[15], pts[16] = (330 + offset_x, 325), (330 + offset_x, 350), (330 + offset_x, 365)
    pts[18], pts[19], pts[20] = (355 + offset_x, 335), (355 + offset_x, 355), (350 + offset_x, 370)
    return hand


def _make_thumbs_up(handedness: str = "Right") -> DetectedHand:
    hand = _make_closed_fist(handedness)
    pts = hand.landmarks_px
    # Extend thumb straight upward above wrist
    pts[1] = (275, 360)
    pts[2] = (275, 315)
    pts[3] = (275, 270)
    pts[4] = (275, 225)
    return hand


def _make_index_pointing(handedness: str = "Right") -> DetectedHand:
    hand = _make_closed_fist(handedness)
    pts = hand.landmarks_px
    pts[6], pts[7], pts[8] = (270, 250), (270, 210), (270, 170)
    return hand


def _make_peace_sign(handedness: str = "Right") -> DetectedHand:
    hand = _make_closed_fist(handedness)
    pts = hand.landmarks_px
    # Spread index left and middle right in a V shape
    pts[6], pts[7], pts[8] = (260, 250), (250, 210), (240, 170)
    pts[10], pts[11], pts[12] = (310, 240), (320, 195), (330, 155)
    return hand


def _make_pinch(handedness: str = "Right") -> DetectedHand:
    hand = _make_open_hand(handedness)
    pts = hand.landmarks_px
    # Bring thumb tip (4) and index tip (8) together above palm
    pts[3] = (255, 260)
    pts[4] = (268, 225)
    pts[6], pts[7], pts[8] = (270, 260), (272, 240), (270, 227)
    return hand


class TestGestureRecognition(unittest.TestCase):
    """Verifies all 6 gestures, stabilization, unknown handling, and two-hand independence."""

    def setUp(self):
        # Set min_confirmations=1 for direct single-frame pose checks
        self.recognizer = GestureRecognizer(stabilization_window=1, min_confirmations=1)

    def test_all_six_gestures(self):
        cases = [
            (_make_open_hand(), GestureLabel.OPEN_PALM),
            (_make_closed_fist(), GestureLabel.CLOSED_FIST),
            (_make_thumbs_up(), GestureLabel.THUMBS_UP),
            (_make_index_pointing(), GestureLabel.INDEX_POINTING),
            (_make_peace_sign(), GestureLabel.PEACE_SIGN),
            (_make_pinch(), GestureLabel.PINCH),
        ]
        for hand, expected in cases:
            with self.subTest(expected=expected):
                pred = self.recognizer.recognize_hand(hand, 640, 480)
                self.assertEqual(pred.gesture, expected)
                self.assertGreaterEqual(pred.confidence, 0.60)

    def test_unknown_and_malformed_hands_handled_safely(self):
        empty_hand = DetectedHand("Right", 0.5, [], [])
        pred_empty = self.recognizer.recognize_hand(empty_hand, 640, 480)
        self.assertEqual(pred_empty.gesture, GestureLabel.UNKNOWN)
        self.assertEqual(pred_empty.confidence, 0.0)

        # Ambiguous pose: only pinky extended
        ambiguous = _make_closed_fist()
        ambiguous.landmarks_px[18] = (360, 275)
        ambiguous.landmarks_px[19] = (360, 240)
        ambiguous.landmarks_px[20] = (360, 205)
        pred_ambig = self.recognizer.recognize_hand(ambiguous, 640, 480)
        self.assertEqual(pred_ambig.gesture, GestureLabel.UNKNOWN)

    def test_temporal_stabilization_prevents_flicker(self):
        rec = GestureRecognizer(stabilization_window=4, min_confirmations=3)
        open_hand = _make_open_hand()
        fist_hand = _make_closed_fist()

        # Establish OPEN_PALM
        for _ in range(3):
            res = rec.recognize_hand(open_hand, 640, 480)
        self.assertEqual(res.gesture, GestureLabel.OPEN_PALM)
        self.assertTrue(res.is_stable)

        # Single-frame glitch to CLOSED_FIST should NOT switch the stable gesture
        glitch = rec.recognize_hand(fist_hand, 640, 480)
        self.assertEqual(glitch.raw_gesture, GestureLabel.CLOSED_FIST)
        self.assertEqual(glitch.gesture, GestureLabel.OPEN_PALM)

    def test_two_hands_independent_and_pluggable_classifier(self):
        left = _make_peace_sign("Left")
        right = _make_thumbs_up("Right")
        preds = self.recognizer.recognize_hands([left, right], 640, 480)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0].gesture, GestureLabel.PEACE_SIGN)
        self.assertEqual(preds[1].gesture, GestureLabel.THUMBS_UP)

        class DummyMLClassifier(BaseGestureClassifier):
            def predict(self, hand):
                return GestureLabel.PINCH, 0.99, {"ml_model": 1.0}

        ml_rec = GestureRecognizer(
            classifier=DummyMLClassifier(),
            stabilization_window=1,
            min_confirmations=1,
        )
        ml_pred = ml_rec.recognize_hand(_make_open_hand(), 640, 480)
        self.assertEqual(ml_pred.gesture, GestureLabel.PINCH)
        self.assertAlmostEqual(ml_pred.confidence, 0.99)


if __name__ == "__main__":
    unittest.main()

