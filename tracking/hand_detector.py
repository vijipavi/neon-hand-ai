"""
Real-Time Hand Detector using MediaPipe Hand Landmarker.
Supports MediaPipe Tasks API (HandLandmarker in VIDEO mode) when a .task model
bundle is present, with automatic fallback to MediaPipe Solutions Hands API
for zero-config real-time execution on mediapipe==0.10.14.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from configuration.settings import NeonColors, TrackingConfig


@dataclass
class DetectedHand:
    """Structured representation of a detected hand with 21 landmarks."""
    handedness: str  # "Left" or "Right"
    score: float
    landmarks_norm: List[Tuple[float, float, float]]  # (x, y, z) normalized
    landmarks_px: List[Tuple[int, int]]  # (x, y) pixel coordinates
    world_landmarks: Optional[List[Tuple[float, float, float]]] = None  # 3D meters


class HandDetector:
    """
    Real-time MediaPipe Hand Landmarker wrapper.
    Extracts up to 2 hands, handedness classification, 21 normalized landmarks,
    pixel coordinates, and 3D world landmarks when available.
    """

    # Standard 21-landmark MediaPipe hand skeleton topology
    HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # Index
        (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17),                                # Palm base
    )

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or TrackingConfig()
        self._last_timestamp_ms: int = 0
        self._use_tasks_api: bool = False
        self._landmarker = None
        self._hands = None

        model_file = Path(self.config.model_path)
        if model_file.is_file():
            self._init_tasks_landmarker(str(model_file))
        else:
            self._init_solutions_hands()

    def _init_tasks_landmarker(self, model_path: str) -> None:
        """Initializes MediaPipe Tasks HandLandmarker in VIDEO mode."""
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.max_num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._use_tasks_api = True

    def _init_solutions_hands(self) -> None:
        """Initializes bundled MediaPipe Solutions Hands pipeline."""
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._use_tasks_api = False

    def _next_timestamp_ms(self, timestamp_ms: Optional[int]) -> int:
        """Ensures strictly monotonically increasing timestamps for VIDEO mode."""
        ts = timestamp_ms if timestamp_ms is not None else int(time.monotonic() * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> List[DetectedHand]:
        """
        Detects up to `max_num_hands` hands in a BGR frame.
        Returns an empty list cleanly when no hands are detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        if self._use_tasks_api and self._landmarker is not None:
            return self._detect_tasks(frame_rgb, w, h, timestamp_ms)
        if self._hands is not None:
            return self._detect_solutions(frame_rgb, w, h)
        return []

    def _detect_tasks(
        self,
        frame_rgb: np.ndarray,
        width: int,
        height: int,
        timestamp_ms: Optional[int],
    ) -> List[DetectedHand]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = self._next_timestamp_ms(timestamp_ms)
        result = self._landmarker.detect_for_video(mp_image, ts)

        if not result.hand_landmarks:
            return []

        detected: List[DetectedHand] = []
        world_list = result.hand_world_landmarks or []
        handedness_list = result.handedness or []

        for idx, hand_lms in enumerate(result.hand_landmarks):
            norm_coords = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            pixel_coords = [(int(lm.x * width), int(lm.y * height)) for lm in hand_lms]

            world_coords: Optional[List[Tuple[float, float, float]]] = None
            if idx < len(world_list) and world_list[idx]:
                world_coords = [(wlm.x, wlm.y, wlm.z) for wlm in world_list[idx]]

            label = "Unknown"
            score = 0.0
            if idx < len(handedness_list) and handedness_list[idx]:
                cat = handedness_list[idx][0]
                label = cat.category_name or cat.display_name or "Unknown"
                score = float(cat.score)

            detected.append(
                DetectedHand(
                    handedness=label,
                    score=score,
                    landmarks_norm=norm_coords,
                    landmarks_px=pixel_coords,
                    world_landmarks=world_coords,
                )
            )

        return detected

    def _detect_solutions(
        self,
        frame_rgb: np.ndarray,
        width: int,
        height: int,
    ) -> List[DetectedHand]:
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []

        detected: List[DetectedHand] = []
        world_list = results.multi_hand_world_landmarks or []
        handedness_list = results.multi_handedness or []

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            norm_coords = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            pixel_coords = [
                (int(lm.x * width), int(lm.y * height)) for lm in hand_lms.landmark
            ]

            world_coords: Optional[List[Tuple[float, float, float]]] = None
            if idx < len(world_list) and world_list[idx] is not None:
                world_coords = [
                    (wlm.x, wlm.y, wlm.z) for wlm in world_list[idx].landmark
                ]

            label = "Unknown"
            score = 0.0
            if idx < len(handedness_list) and handedness_list[idx].classification:
                cls = handedness_list[idx].classification[0]
                label = cls.label
                score = float(cls.score)

            detected.append(
                DetectedHand(
                    handedness=label,
                    score=score,
                    landmarks_norm=norm_coords,
                    landmarks_px=pixel_coords,
                    world_landmarks=world_coords,
                )
            )

        return detected

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hands: List[DetectedHand],
    ) -> np.ndarray:
        """
        Draws the 21 hand landmarks, skeletal connections, and handedness labels
        onto the webcam frame. Safe to call when `hands` is empty.
        """
        if not hands:
            return frame

        for hand in hands:
            color = (
                NeonColors.CYAN
                if hand.handedness.lower() == "right"
                else NeonColors.MAGENTA
            )

            # Draw skeletal connections
            for start_idx, end_idx in self.HAND_CONNECTIONS:
                if start_idx < len(hand.landmarks_px) and end_idx < len(hand.landmarks_px):
                    cv2.line(
                        frame,
                        hand.landmarks_px[start_idx],
                        hand.landmarks_px[end_idx],
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # Draw all 21 landmark keypoints
            for lm_idx, pt in enumerate(hand.landmarks_px):
                radius = 5 if lm_idx in (0, 4, 8, 12, 16, 20) else 3
                cv2.circle(frame, pt, radius, NeonColors.WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, radius + 1, color, 1, cv2.LINE_AA)

            # Draw handedness classification badge near wrist (landmark 0)
            if hand.landmarks_px:
                wx, wy = hand.landmarks_px[0]
                label = f"{hand.handedness} ({int(hand.score * 100)}%)"
                cv2.putText(
                    frame,
                    label,
                    (max(10, wx - 40), max(25, wy + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return frame

    def close(self) -> None:
        """Releases MediaPipe resources cleanly."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

