"""
Modular Gesture Recognition System for NEON HAND AI.
Uses geometric landmark analysis (normalized distances, joint angles, and
orientation vectors) with a pluggable BaseGestureClassifier interface so a
Machine Learning classifier can be swapped in without changing callers.
"""

from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

from tracking.hand_detector import DetectedHand
from tracking.landmark_processor import (
    FINGERTIP_INDICES,
    LandmarkProcessor,
    ProcessedHand,
    euclidean_distance,
)

HandInput = Union[ProcessedHand, DetectedHand]


class GestureLabel(str, Enum):
    """Supported gesture classifications."""
    OPEN_PALM = "OPEN_PALM"
    CLOSED_FIST = "CLOSED_FIST"
    THUMBS_UP = "THUMBS_UP"
    INDEX_POINTING = "INDEX_POINTING"
    PEACE_SIGN = "PEACE_SIGN"
    PINCH = "PINCH"
    UNKNOWN = "UNKNOWN"


@dataclass
class GesturePrediction:
    """Structured gesture recognition output for a single hand."""
    gesture: GestureLabel
    raw_gesture: GestureLabel
    confidence: float
    is_stable: bool
    handedness: str
    extended_fingers: Dict[str, bool]
    finger_angles: Dict[str, float]
    pinch_ratio: float
    metadata: Dict[str, float] = field(default_factory=dict)


class BaseGestureClassifier(ABC):
    """
    Abstract classifier interface.
    Implement this interface to plug in an ML/DL model (e.g. sklearn, PyTorch, TFLite)
    while keeping GestureRecognizer and temporal stabilization unchanged.
    """

    @abstractmethod
    def predict(
        self,
        hand: ProcessedHand,
    ) -> Tuple[GestureLabel, float, Dict[str, float]]:
        """
        Classifies a ProcessedHand into (GestureLabel, confidence, metadata_dict).
        Confidence must be in [0.0, 1.0].
        """


class GeometricGestureClassifier(BaseGestureClassifier):
    """
    Rule-based mathematical gesture classifier using scale-normalized
    landmark distances, joint angles, and directional vectors.
    """

    def __init__(
        self,
        pinch_ratio_threshold: float = 0.38,
        peace_spread_threshold: float = 0.25,
        min_confidence: float = 0.60,
    ) -> None:
        self.pinch_ratio_threshold = pinch_ratio_threshold
        self.peace_spread_threshold = peace_spread_threshold
        self.min_confidence = min_confidence

    @staticmethod
    def _palm_size(hand: ProcessedHand) -> float:
        """Scale reference: distance from wrist (0) to middle MCP (9)."""
        size = euclidean_distance(hand.landmarks_px[0], hand.landmarks_px[9])
        return max(1.0, size)

    def predict(
        self,
        hand: ProcessedHand,
    ) -> Tuple[GestureLabel, float, Dict[str, float]]:
        pts = hand.landmarks_px
        ext = hand.extended_fingers
        angles = hand.finger_angles
        palm_sz = self._palm_size(hand)

        pinch_dist = hand.thumb_index_distance_px
        pinch_ratio = pinch_dist / palm_sz
        index_reach = euclidean_distance(pts[0], pts[FINGERTIP_INDICES["index"]]) / palm_sz
        peace_spread = (
            euclidean_distance(
                pts[FINGERTIP_INDICES["index"]],
                pts[FINGERTIP_INDICES["middle"]],
            )
            / palm_sz
        )

        # Thumb upward alignment vector (MCP 2 -> TIP 4; negative Y is up in screen space)
        dx_thumb = pts[4][0] - pts[2][0]
        dy_thumb = pts[4][1] - pts[2][1]
        thumb_mag = max(1.0, math.hypot(dx_thumb, dy_thumb))
        thumb_upward_score = -dy_thumb / thumb_mag  # +1.0 = pointing straight up

        meta = {
            "palm_size_px": palm_sz,
            "pinch_ratio": pinch_ratio,
            "peace_spread_ratio": peace_spread,
            "thumb_upward_score": thumb_upward_score,
        }

        # 1. PINCH: thumb tip & index tip close together, index not folded into wrist
        if pinch_ratio < self.pinch_ratio_threshold and index_reach > 0.70:
            conf = min(1.0, max(0.65, 1.0 - (pinch_ratio / self.pinch_ratio_threshold) * 0.35))
            return GestureLabel.PINCH, conf, meta

        # 2. OPEN PALM: all 5 fingers extended
        if all(ext.values()):
            avg_angle = sum(angles.values()) / 5.0
            conf = min(1.0, max(0.65, avg_angle / 180.0))
            return (
                (GestureLabel.OPEN_PALM, conf, meta)
                if conf >= self.min_confidence
                else (GestureLabel.UNKNOWN, conf, meta)
            )

        # 3. CLOSED FIST: 0 fingers extended
        if not any(ext.values()):
            non_thumb_avg = sum(angles[f] for f in ("index", "middle", "ring", "pinky")) / 4.0
            curl_score = max(0.0, min(1.0, (165.0 - non_thumb_avg) / 75.0))
            conf = 0.65 + 0.35 * curl_score
            return (
                (GestureLabel.CLOSED_FIST, conf, meta)
                if conf >= self.min_confidence
                else (GestureLabel.UNKNOWN, conf, meta)
            )

        # 4. THUMBS UP: only thumb extended AND pointing upward
        if (
            ext["thumb"]
            and not ext["index"]
            and not ext["middle"]
            and not ext["ring"]
            and not ext["pinky"]
            and thumb_upward_score > 0.55
            and pts[4][1] < pts[0][1]
        ):
            conf = min(1.0, 0.60 + 0.40 * thumb_upward_score)
            return GestureLabel.THUMBS_UP, conf, meta

        # 5. PEACE SIGN: index & middle extended with V spread; ring & pinky curled
        if (
            ext["index"]
            and ext["middle"]
            and not ext["ring"]
            and not ext["pinky"]
            and peace_spread >= self.peace_spread_threshold
        ):
            spread_bonus = min(1.0, peace_spread / 0.65)
            conf = 0.70 + 0.30 * spread_bonus
            return GestureLabel.PEACE_SIGN, conf, meta

        # 6. INDEX POINTING: index extended; middle, ring, pinky curled
        if (
            ext["index"]
            and not ext["middle"]
            and not ext["ring"]
            and not ext["pinky"]
        ):
            idx_score = min(1.0, angles["index"] / 175.0)
            conf = 0.65 + 0.35 * idx_score
            return GestureLabel.INDEX_POINTING, conf, meta

        # 7. Fallback for ambiguous poses
        return GestureLabel.UNKNOWN, 0.0, meta


class GestureStabilizer:
    """
    Temporal debounce / majority-vote filter per hand to prevent rapid
    frame-to-frame flickering between gestures.
    """

    def __init__(self, window_size: int = 5, min_confirmations: int = 3) -> None:
        if window_size < 1 or min_confirmations < 1 or min_confirmations > window_size:
            raise ValueError("Require 1 <= min_confirmations <= window_size.")
        self.window_size = window_size
        self.min_confirmations = min_confirmations
        self._histories: Dict[str, Deque[GestureLabel]] = {}
        self._stable_labels: Dict[str, GestureLabel] = {}

    def update(self, hand_key: str, raw_gesture: GestureLabel) -> Tuple[GestureLabel, bool]:
        """Updates history for `hand_key` and returns (stable_gesture, is_stable)."""
        if hand_key not in self._histories:
            self._histories[hand_key] = deque(maxlen=self.window_size)
            self._stable_labels[hand_key] = GestureLabel.UNKNOWN

        history = self._histories[hand_key]
        history.append(raw_gesture)

        counts = Counter(history)
        top_gesture, top_count = counts.most_common(1)[0]

        if top_count >= self.min_confirmations:
            self._stable_labels[hand_key] = top_gesture
            return top_gesture, True

        return self._stable_labels[hand_key], False

    def prune(self, active_keys: Sequence[str]) -> None:
        """Removes temporal state for hands no longer visible."""
        active = set(active_keys)
        for stale in [k for k in self._histories if k not in active]:
            del self._histories[stale]
            self._stable_labels.pop(stale, None)


class GestureRecognizer:
    """
    High-level gesture recognition coordinator.
    Supports both ProcessedHand and DetectedHand inputs, independent two-hand
    tracking, temporal stabilization, and pluggable classifiers.
    """

    def __init__(
        self,
        classifier: Optional[BaseGestureClassifier] = None,
        stabilization_window: int = 5,
        min_confirmations: int = 3,
        landmark_processor: Optional[LandmarkProcessor] = None,
    ) -> None:
        self.classifier: BaseGestureClassifier = classifier or GeometricGestureClassifier()
        self.stabilizer = GestureStabilizer(
            window_size=stabilization_window,
            min_confirmations=min_confirmations,
        )
        self._processor = landmark_processor or LandmarkProcessor(smoothing_alpha=1.0)

    def recognize_hand(
        self,
        hand: HandInput,
        frame_width: int = 1280,
        frame_height: int = 720,
        hand_key: Optional[str] = None,
    ) -> GesturePrediction:
        """Recognizes the gesture for a single hand safely."""
        if not getattr(hand, "landmarks_px", None) or len(hand.landmarks_px) < 21:
            return GesturePrediction(
                gesture=GestureLabel.UNKNOWN,
                raw_gesture=GestureLabel.UNKNOWN,
                confidence=0.0,
                is_stable=False,
                handedness=getattr(hand, "handedness", "Unknown"),
                extended_fingers={},
                finger_angles={},
                pinch_ratio=0.0,
            )

        processed = (
            hand
            if isinstance(hand, ProcessedHand)
            else self._processor.process_hand(hand, frame_width, frame_height, hand_key=hand_key)
        )

        key = hand_key or processed.handedness
        raw_gesture, confidence, meta = self.classifier.predict(processed)
        stable_gesture, is_stable = self.stabilizer.update(key, raw_gesture)

        return GesturePrediction(
            gesture=stable_gesture,
            raw_gesture=raw_gesture,
            confidence=confidence,
            is_stable=is_stable,
            handedness=processed.handedness,
            extended_fingers=processed.extended_fingers,
            finger_angles=processed.finger_angles,
            pinch_ratio=meta.get("pinch_ratio", 0.0),
            metadata=meta,
        )

    def recognize_hands(
        self,
        hands: Sequence[HandInput],
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> List[GesturePrediction]:
        """Recognizes gestures for multiple hands independently."""
        if not hands:
            self.stabilizer.prune([])
            return []

        predictions: List[GesturePrediction] = []
        seen_counts: Dict[str, int] = {}
        active_keys: List[str] = []

        for hand in hands:
            label = getattr(hand, "handedness", "Right")
            idx = seen_counts.get(label, 0)
            seen_counts[label] = idx + 1
            hand_key = f"{label}_{idx}"
            active_keys.append(hand_key)
            predictions.append(
                self.recognize_hand(
                    hand,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    hand_key=hand_key,
                )
            )

        self.stabilizer.prune(active_keys)
        return predictions

