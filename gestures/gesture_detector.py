"""
Gesture Detector.
Analyzes hand landmark spatial relationships, joint angles, and fingertip distances
to classify dynamic and static hand gestures (Pinch, Fist, Open Palm, Pointing, etc.).
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from tracking.hand_tracker import HandData


class GestureType(str, Enum):
    """Supported gesture classifications."""
    NONE = "NONE"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    PINCH = "PINCH"
    POINTING = "POINTING"
    PEACE = "PEACE"
    THUMBS_UP = "THUMBS_UP"


@dataclass
class GestureResult:
    """Detection output for a single hand gesture."""
    gesture: GestureType
    confidence: float
    fingers_extended: List[bool]


class GestureDetector:
    """Evaluates spatial landmark geometries to detect interaction gestures."""

    def __init__(self) -> None:
        # Landmark indices for fingertips in MediaPipe Hand Model
        self.FINGERTIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.PIP_IDS = [2, 6, 10, 14, 18]        # Proximal Interphalangeal joints

    def detect(self, hand: HandData) -> GestureResult:
        """
        Calculates finger state and identifies the dominant gesture.
        Initial baseline implementation for architectural setup.
        """
        if not hand.landmarks_px or len(hand.landmarks_px) < 21:
            return GestureResult(gesture=GestureType.NONE, confidence=0.0, fingers_extended=[])

        # Initial simple finger extension calculation (y-coordinate comparison)
        fingers_up: List[bool] = []

        # Thumb (uses x-distance relative to IP joint depending on handedness)
        if hand.handedness == "Right":
            thumb_up = hand.landmarks_px[4][0] < hand.landmarks_px[3][0]
        else:
            thumb_up = hand.landmarks_px[4][0] > hand.landmarks_px[3][0]
        fingers_up.append(thumb_up)

        # Other 4 fingers (Tip higher than PIP joint in inverted screen coordinates)
        for tip_id, pip_id in zip(self.FINGERTIP_IDS[1:], self.PIP_IDS[1:]):
            fingers_up.append(hand.landmarks_px[tip_id][1] < hand.landmarks_px[pip_id][1])

        extended_count = sum(fingers_up)

        if extended_count == 5 :
            gesture = GestureType.OPEN_PALM
        elif extended_count == 0:
            gesture = GestureType.FIST
        elif fingers_up[1] and extended_count == 1:
            gesture = GestureType.POINTING
        elif fingers_up[1] and fingers_up[2] and extended_count == 2:
            gesture = GestureType.PEACE
        else:
            gesture = GestureType.NONE

        return GestureResult(
            gesture=gesture,
            confidence=0.85 if gesture != GestureType.NONE else 0.0,
            fingers_extended=fingers_up,
        )
