"""
Landmark Processing Module for NEON HAND AI.
Extracts wrist, palm center, and 5 fingertips; normalizes coordinates to webcam
dimensions; applies per-hand EMA smoothing; computes inter-landmark distances,
joint angles, finger extension states, and basic finger counts.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

from tracking.hand_detector import DetectedHand

FINGER_NAMES: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")

WRIST_INDEX: int = 0
PALM_INDICES: Tuple[int, ...] = (0, 1, 5, 9, 13, 17)

FINGERTIP_INDICES: Dict[str, int] = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

# Triplets (proximal, vertex, distal) for calculating each finger's bend angle
FINGER_ANGLE_TRIPLETS: Dict[str, Tuple[int, int, int]] = {
    "thumb": (2, 3, 4),    # MCP -> IP -> TIP
    "index": (5, 6, 8),    # MCP -> PIP -> TIP
    "middle": (9, 10, 12), # MCP -> PIP -> TIP
    "ring": (13, 14, 16),  # MCP -> PIP -> TIP
    "pinky": (17, 18, 20), # MCP -> PIP -> TIP
}


def normalize_point(
    point_px: Tuple[int, int],
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float]:
    """Normalizes pixel coordinates (x, y) to [0.0, 1.0] based on frame dimensions."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive integers.")
    nx = max(0.0, min(1.0, point_px[0] / float(frame_width)))
    ny = max(0.0, min(1.0, point_px[1] / float(frame_height)))
    return (nx, ny)


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Computes Euclidean distance between two 2D or 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calculate_angle(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> float:
    """
    Calculates the interior angle (in degrees, 0..180) at vertex `b`
    formed by segments `ba` and `bc`.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba == 0.0 or mag_bc == 0.0:
        return 0.0

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_theta))


class LandmarkSmoother:
    """
    Exponential Moving Average (EMA) smoother for 21 hand landmarks.
    Maintains independent smoothing state for multiple hands.
    """

    def __init__(self, alpha: float = 0.45) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0.0, 1.0].")
        self.alpha = alpha
        self._state: Dict[str, List[Tuple[float, float]]] = {}

    def smooth(
        self,
        hand_key: str,
        landmarks_px: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Applies EMA smoothing to a hand's pixel landmarks."""
        if hand_key not in self._state or len(self._state[hand_key]) != len(landmarks_px):
            self._state[hand_key] = [(float(x), float(y)) for x, y in landmarks_px]
            return [(int(x), int(y)) for x, y in landmarks_px]

        prev = self._state[hand_key]
        smoothed_float: List[Tuple[float, float]] = []
        smoothed_int: List[Tuple[int, int]] = []
        a = self.alpha

        for (px, py), (sx, sy) in zip(landmarks_px, prev):
            nx = a * px + (1.0 - a) * sx
            ny = a * py + (1.0 - a) * sy
            smoothed_float.append((nx, ny))
            smoothed_int.append((int(round(nx)), int(round(ny))))

        self._state[hand_key] = smoothed_float
        return smoothed_int

    def prune(self, active_keys: Sequence[str]) -> None:
        """Removes smoothing history for hands no longer visible."""
        active_set = set(active_keys)
        stale = [k for k in self._state if k not in active_set]
        for k in stale:
            del self._state[k]

    def reset(self) -> None:
        """Clears all stored smoothing states."""
        self._state.clear()


@dataclass
class ProcessedHand:
    """Processed geometric features for a single detected hand."""
    handedness: str
    score: float
    landmarks_px: List[Tuple[int, int]]
    landmarks_norm: List[Tuple[float, float]]
    wrist: Tuple[int, int]
    wrist_norm: Tuple[float, float]
    palm_center: Tuple[int, int]
    palm_center_norm: Tuple[float, float]
    fingertips: Dict[str, Tuple[int, int]]
    fingertips_norm: Dict[str, Tuple[float, float]]
    finger_angles: Dict[str, float]
    extended_fingers: Dict[str, bool]
    finger_count: int
    thumb_index_distance_px: float
    thumb_index_distance_norm: float


class LandmarkProcessor:
    """
    Processes raw MediaPipe DetectedHand objects into smoothed coordinates,
    palm/fingertip positions, distances, joint angles, and finger counts.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.45,
        angle_threshold_deg: float = 150.0,
        thumb_angle_threshold_deg: float = 140.0,
    ) -> None:
        self.smoother = LandmarkSmoother(alpha=smoothing_alpha)
        self.angle_threshold_deg = angle_threshold_deg
        self.thumb_angle_threshold_deg = thumb_angle_threshold_deg

    @staticmethod
    def distance_between(
        landmarks: Sequence[Sequence[float]],
        idx_a: int,
        idx_b: int,
    ) -> float:
        """Returns the Euclidean distance between two landmark indices."""
        return euclidean_distance(landmarks[idx_a], landmarks[idx_b])

    @staticmethod
    def _compute_palm_center(landmarks_px: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
        """Computes the centroid of the palm anchor landmarks (0, 1, 5, 9, 13, 17)."""
        xs = [landmarks_px[i][0] for i in PALM_INDICES]
        ys = [landmarks_px[i][1] for i in PALM_INDICES]
        count = len(PALM_INDICES)
        return (int(round(sum(xs) / count)), int(round(sum(ys) / count)))

    def _compute_finger_angles(
        self,
        landmarks_px: Sequence[Tuple[int, int]],
    ) -> Dict[str, float]:
        """Calculates the bend angle (degrees) for each of the 5 fingers."""
        angles: Dict[str, float] = {}
        for name, (a_idx, b_idx, c_idx) in FINGER_ANGLE_TRIPLETS.items():
            angles[name] = calculate_angle(
                landmarks_px[a_idx],
                landmarks_px[b_idx],
                landmarks_px[c_idx],
            )
        return angles

    def _determine_extended_fingers(
        self,
        landmarks_px: Sequence[Tuple[int, int]],
        finger_angles: Dict[str, float],
    ) -> Dict[str, bool]:
        """
        Determines which of the 5 fingers are extended using both joint angles
        and wrist-to-joint radial distances (orientation-independent).
        """
        extended: Dict[str, bool] = {}

        # Thumb: extended if tip (4) is farther from pinky MCP (17) than IP (3)
        # and the thumb IP joint angle is relatively straight.
        dist_pinky_to_tip = euclidean_distance(landmarks_px[17], landmarks_px[4])
        dist_pinky_to_ip = euclidean_distance(landmarks_px[17], landmarks_px[3])
        extended["thumb"] = (
            dist_pinky_to_tip > dist_pinky_to_ip * 1.12
            and finger_angles["thumb"] >= self.thumb_angle_threshold_deg
        )

        # Index, Middle, Ring, Pinky: extended if tip is farther from wrist (0)
        # than PIP joint AND PIP angle exceeds threshold.
        wrist = landmarks_px[WRIST_INDEX]
        for name in FINGER_NAMES[1:]:
            _, pip_idx, tip_idx = FINGER_ANGLE_TRIPLETS[name]
            dist_wrist_tip = euclidean_distance(wrist, landmarks_px[tip_idx])
            dist_wrist_pip = euclidean_distance(wrist, landmarks_px[pip_idx])
            extended[name] = (
                dist_wrist_tip > dist_wrist_pip * 1.1
                and finger_angles[name] >= self.angle_threshold_deg
            )

        return extended

    def process_hand(
        self,
        hand: DetectedHand,
        frame_width: int,
        frame_height: int,
        hand_key: Optional[str] = None,
    ) -> ProcessedHand:
        """Processes a single DetectedHand into a ProcessedHand."""
        if len(hand.landmarks_px) < 21:
            raise ValueError("Expected 21 landmarks for hand processing.")

        key = hand_key or hand.handedness
        smoothed_px = self.smoother.smooth(key, hand.landmarks_px)
        norm_coords = [
            normalize_point(pt, frame_width, frame_height) for pt in smoothed_px
        ]

        wrist = smoothed_px[WRIST_INDEX]
        wrist_norm = norm_coords[WRIST_INDEX]

        palm_center = self._compute_palm_center(smoothed_px)
        palm_center_norm = normalize_point(palm_center, frame_width, frame_height)

        fingertips = {
            name: smoothed_px[idx] for name, idx in FINGERTIP_INDICES.items()
        }
        fingertips_norm = {
            name: norm_coords[idx] for name, idx in FINGERTIP_INDICES.items()
        }

        finger_angles = self._compute_finger_angles(smoothed_px)
        extended_fingers = self._determine_extended_fingers(smoothed_px, finger_angles)
        finger_count = sum(1 for is_up in extended_fingers.values() if is_up)

        thumb_idx_px = self.distance_between(
            smoothed_px, FINGERTIP_INDICES["thumb"], FINGERTIP_INDICES["index"]
        )
        thumb_idx_norm = self.distance_between(
            norm_coords, FINGERTIP_INDICES["thumb"], FINGERTIP_INDICES["index"]
        )

        return ProcessedHand(
            handedness=hand.handedness,
            score=hand.score,
            landmarks_px=smoothed_px,
            landmarks_norm=norm_coords,
            wrist=wrist,
            wrist_norm=wrist_norm,
            palm_center=palm_center,
            palm_center_norm=palm_center_norm,
            fingertips=fingertips,
            fingertips_norm=fingertips_norm,
            finger_angles=finger_angles,
            extended_fingers=extended_fingers,
            finger_count=finger_count,
            thumb_index_distance_px=thumb_idx_px,
            thumb_index_distance_norm=thumb_idx_norm,
        )

    def process_hands(
        self,
        hands: List[DetectedHand],
        frame_width: int,
        frame_height: int,
    ) -> List[ProcessedHand]:
        """
        Processes multiple detected hands independently and prunes smoothing
        state for any hands that left the frame.
        """
        if not hands:
            self.smoother.prune([])
            return []

        processed: List[ProcessedHand] = []
        seen_counts: Dict[str, int] = {}
        active_keys: List[str] = []

        for hand in hands:
            count = seen_counts.get(hand.handedness, 0)
            seen_counts[hand.handedness] = count + 1
            hand_key = f"{hand.handedness}_{count}"
            active_keys.append(hand_key)
            processed.append(
                self.process_hand(hand, frame_width, frame_height, hand_key=hand_key)
            )

        self.smoother.prune(active_keys)
        return processed

