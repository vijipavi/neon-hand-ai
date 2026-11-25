"""
Futuristic Hologram Interface Renderer for NEON HAND AI.
Renders an animated circular AR hologram around each detected palm with
rotating segmented rings, sweeping radar scan arcs, oscillating scan waves,
and velocity-responsive scaling.
"""

from dataclasses import dataclass
import math
import time
from typing import Dict, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from configuration.settings import NeonColors
from tracking.hand_detector import DetectedHand
from tracking.landmark_processor import PALM_INDICES, ProcessedHand

HandLike = Union[ProcessedHand, DetectedHand]


@dataclass
class HologramConfig:
    """Configurable visual parameters for the palm hologram interface."""
    enabled: bool = True
    primary_color: Tuple[int, int, int] = NeonColors.CYAN
    secondary_color: Tuple[int, int, int] = NeonColors.ELECTRIC_BLUE
    accent_color: Tuple[int, int, int] = NeonColors.ELECTRIC_PURPLE
    core_color: Tuple[int, int, int] = NeonColors.WHITE
    rotation_speed_deg: float = 65.0
    scan_speed: float = 2.4
    pulse_speed: float = 3.2
    glow_intensity: float = 0.85
    blur_downscale: float = 0.5
    show_telemetry: bool = True


def draw_segmented_ring(
    canvas: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    angle_offset: float,
    segments: int = 3,
    arc_span: float = 75.0,
    thickness: int = 2,
) -> None:
    """Reusable function to draw a rotating segmented holographic ring."""
    if radius <= 1 or segments <= 0:
        return
    step = 360.0 / segments
    for i in range(segments):
        start_a = (angle_offset + i * step) % 360.0
        end_a = start_a + arc_span
        cv2.ellipse(
            canvas,
            center,
            (radius, radius),
            0.0,
            start_a,
            end_a,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_scanning_sweep(
    glow_canvas: np.ndarray,
    core_canvas: np.ndarray,
    center: Tuple[int, int],
    max_radius: int,
    phase: float,
    primary_color: Tuple[int, int, int],
    accent_color: Tuple[int, int, int],
    scan_speed: float = 2.4,
) -> None:
    """
    Reusable function to draw an oscillating radial scan wave and sweeping
    radar beam inside the palm hologram.
    """
    if max_radius <= 4:
        return

    # 1. Expanding radial scan ring (0 -> max_radius loop)
    scan_t = (phase * scan_speed * 0.45) % 1.0
    scan_r = max(4, int(max_radius * scan_t))
    fade = 1.0 - scan_t
    ring_color = (
        int(primary_color[0] * fade),
        int(primary_color[1] * fade),
        int(primary_color[2] * fade),
    )
    cv2.circle(glow_canvas, center, scan_r, ring_color, 2, cv2.LINE_AA)

    # 2. Rotating radar sweep line + trailing arc
    sweep_deg = (phase * 140.0) % 360.0
    rad = math.radians(sweep_deg)
    end_pt = (
        int(center[0] + max_radius * math.cos(rad)),
        int(center[1] + max_radius * math.sin(rad)),
    )
    cv2.line(glow_canvas, center, end_pt, accent_color, 2, cv2.LINE_AA)
    cv2.line(core_canvas, center, end_pt, NeonColors.WHITE, 1, cv2.LINE_AA)
    cv2.ellipse(
        glow_canvas,
        center,
        (int(max_radius * 0.85), int(max_radius * 0.85)),
        0.0,
        sweep_deg - 35.0,
        sweep_deg,
        accent_color,
        2,
        cv2.LINE_AA,
    )


class HologramRenderer:
    """
    Renders a multi-ring holographic AR interface around each palm that
    dynamically responds to palm movement speed and hand openness.
    """

    def __init__(self, config: Optional[HologramConfig] = None) -> None:
        self.config = config or HologramConfig()
        self._prev_palms: Dict[str, Tuple[int, int]] = {}
        self._smoothed_speed: Dict[str, float] = {}
        self._start_time: float = time.perf_counter()
        self._glow_buf: Optional[np.ndarray] = None
        self._core_buf: Optional[np.ndarray] = None

    def _ensure_buffers(self, shape: Tuple[int, int, int]) -> None:
        if self._glow_buf is None or self._glow_buf.shape != shape:
            self._glow_buf = np.zeros(shape, dtype=np.uint8)
            self._core_buf = np.zeros(shape, dtype=np.uint8)
        else:
            self._glow_buf.fill(0)
            self._core_buf.fill(0)

    @staticmethod
    def _palm_geometry(hand: HandLike) -> Tuple[Tuple[int, int], int]:
        pts = hand.landmarks_px
        if isinstance(hand, ProcessedHand):
            cx, cy = hand.palm_center
        else:
            xs = [pts[i][0] for i in PALM_INDICES]
            ys = [pts[i][1] for i in PALM_INDICES]
            cx = int(round(sum(xs) / len(PALM_INDICES)))
            cy = int(round(sum(ys) / len(PALM_INDICES)))

        wx, wy = pts[0]
        mx, my = pts[9]
        span = math.hypot(mx - wx, my - wy)
        radius = max(24, min(85, int(span * 0.52)))
        return (cx, cy), radius

    def _update_palm_motion(self, hand_key: str, center: Tuple[int, int]) -> float:
        """Computes smoothed palm movement speed to drive reactive hologram animation."""
        prev = self._prev_palms.get(hand_key, center)
        inst_speed = math.hypot(center[0] - prev[0], center[1] - prev[1])
        self._prev_palms[hand_key] = center

        prev_smooth = self._smoothed_speed.get(hand_key, 0.0)
        smoothed = 0.3 * inst_speed + 0.7 * prev_smooth
        self._smoothed_speed[hand_key] = smoothed
        return smoothed

    def render_to_buffers(
        self,
        glow_canvas: np.ndarray,
        core_canvas: np.ndarray,
        hands: Sequence[HandLike],
        phase: Optional[float] = None,
    ) -> None:
        """Draws reactive circular holograms for all detected hands onto buffers."""
        if not self.config.enabled or not hands:
            return

        anim_phase = phase if phase is not None else (time.perf_counter() - self._start_time)
        seen_counts: Dict[str, int] = {}
        active_keys = set()

        for hand in hands:
            if len(hand.landmarks_px) < 21:
                continue

            label = getattr(hand, "handedness", "Right")
            idx = seen_counts.get(label, 0)
            seen_counts[label] = idx + 1
            hand_key = f"{label}_{idx}"
            active_keys.add(hand_key)

            center, base_r = self._palm_geometry(hand)
            speed = self._update_palm_motion(hand_key, center)

            # Respond to palm movement: speed expands ring slightly and boosts rotation
            motion_scale = 1.0 + min(0.25, speed * 0.008)
            pulse = 1.0 + 0.07 * math.sin(anim_phase * self.config.pulse_speed)
            outer_r = int(base_r * motion_scale * pulse)
            mid_r = int(outer_r * 0.74)
            inner_r = int(outer_r * 0.45)

            rot_cw = (anim_phase * (self.config.rotation_speed_deg + speed * 2.5)) % 360.0
            rot_ccw = (-anim_phase * (self.config.rotation_speed_deg * 0.7)) % 360.0

            is_left = label.lower() == "left"
            primary = self.config.secondary_color if is_left else self.config.primary_color
            accent = self.config.accent_color

            # Outer segmented orbital ring + thin white core
            draw_segmented_ring(glow_canvas, center, outer_r, primary, rot_cw, segments=4, arc_span=55.0, thickness=3)
            draw_segmented_ring(core_canvas, center, outer_r, self.config.core_color, rot_cw, segments=4, arc_span=55.0, thickness=1)

            # Middle counter-rotating ring
            draw_segmented_ring(glow_canvas, center, mid_r, accent, rot_ccw, segments=3, arc_span=85.0, thickness=2)

            # Finger-count power arc on inner ring
            finger_count = getattr(hand, "finger_count", 5)
            arc_span = max(30.0, (finger_count / 5.0) * 330.0)
            cv2.ellipse(
                glow_canvas,
                center,
                (inner_r, inner_r),
                -90.0,
                0.0,
                arc_span,
                primary,
                2,
                cv2.LINE_AA,
            )

            # Scanning sweep + radial wave
            draw_scanning_sweep(
                glow_canvas,
                core_canvas,
                center,
                mid_r,
                anim_phase,
                primary,
                accent,
                scan_speed=self.config.scan_speed,
            )

            # Outer orbital tick marks
            for tick in range(8):
                rad = math.radians(rot_cw + tick * 45.0)
                t_in = outer_r + 3
                t_out = outer_r + (9 if tick % 2 == 0 else 5)
                p1 = (int(center[0] + t_in * math.cos(rad)), int(center[1] + t_in * math.sin(rad)))
                p2 = (int(center[0] + t_out * math.cos(rad)), int(center[1] + t_out * math.sin(rad)))
                cv2.line(glow_canvas, p1, p2, accent, 2, cv2.LINE_AA)

            # Optional compact telemetry tag next to the hologram
            if self.config.show_telemetry:
                tag_pt = (center[0] + outer_r + 12, center[1] - 6)
                cv2.putText(
                    core_canvas,
                    f"{label.upper()} // {finger_count}F",
                    tag_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    primary,
                    1,
                    cv2.LINE_AA,
                )

        # Prune stale palm motion states
        for stale in [k for k in self._prev_palms if k not in active_keys]:
            del self._prev_palms[stale]
            self._smoothed_speed.pop(stale, None)

    def render(
        self,
        frame: np.ndarray,
        hands: Sequence[HandLike],
        phase: Optional[float] = None,
    ) -> np.ndarray:
        """Renders palm holograms with Gaussian bloom onto the webcam frame."""
        if not self.config.enabled or not hands or frame is None or frame.size == 0:
            return frame

        self._ensure_buffers(frame.shape)
        assert self._glow_buf is not None and self._core_buf is not None

        self.render_to_buffers(self._glow_buf, self._core_buf, hands, phase=phase)

        h, w = frame.shape[:2]
        if self.config.blur_downscale < 1.0:
            sw = max(1, int(w * self.config.blur_downscale))
            sh = max(1, int(h * self.config.blur_downscale))
            small = cv2.resize(self._glow_buf, (sw, sh), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.GaussianBlur(small, (7, 7), 0)
            bloom = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            bloom = cv2.GaussianBlur(self._glow_buf, (15, 15), 0)

        out = cv2.addWeighted(frame, 1.0, bloom, self.config.glow_intensity, 0)
        cv2.add(out, self._glow_buf, dst=out)
        cv2.add(out, self._core_buf, dst=out)
        return out

