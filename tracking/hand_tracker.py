"""
MediaPipe Hand Tracking Engine.
Extracts 21 3D hand landmarks per hand, normalized coordinates,
and pixel space conversions for downstream AR rendering.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp

from configuration.settings import TrackingConfig


@dataclass
class HandData:
    """Structured representation of tracked hand features."""
    handedness: str  # 'Left' or 'Right'
    score: float
    landmarks_norm: List[Tuple[float, float, float]]  # (x, y, z) in normalized [0, 1] space
    landmarks_px: List[Tuple[int, int]]  # (x, y) mapped to image pixels


class HandTracker:
    """Wrapper around MediaPipe Hands solution with modular data extraction."""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or TrackingConfig()
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        """
        Processes a BGR image frame through MediaPipe and returns structured HandData.
        """
        h, w, _ = frame_bgr.shape
        # MediaPipe requires RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Performance optimization

        results = self._hands.process(frame_rgb)
        hands_detected: List[HandData] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, classification in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                norm_coords: List[Tuple[float, float, float]] = []
                pixel_coords: List[Tuple[int, int]] = []

                for lm in hand_lms.landmark:
                    norm_coords.append((lm.x, lm.y, lm.z))
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    pixel_coords.append((px, py))

                label = classification.classification[0].label
                confidence = classification.classification[0].score

                hands_detected.append(
                    HandData(
                        handedness=label,
                        score=confidence,
                        landmarks_norm=norm_coords,
                        landmarks_px=pixel_coords,
                    )
                )

        return hands_detected

    def close(self) -> None:
        """Releases underlying MediaPipe graph."""
        if self._hands is not None:
            self._hands.close()

