"""
Neon AR Visual Renderer for NEON HAND AI.
Renders futuristic cyberpunk hand skeletons, glowing fingertip halos,
animated palm energy reticles, and Gaussian-blurred bloom composited
onto live webcam frames in real time.
"""

from dataclasses import dataclass
import math
import time
from typing import Dict, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from configuration.settings import NeonColors, NeonEffectConfig
from tracking.hand_detector import DetectedHand
from tracking.landmark_processor import FINGERTIP_INDICES, PALM_INDICES, ProcessedHand

HandLike = Union[ProcessedHand, DetectedHand]


@dataclass
class NeonPalette:
    """Configurable BGR color palette for the neon AR renderer."""
    skeleton_color: Tuple[int, int, int] = NeonColors.CYAN
    joint_color: Tuple[int, int, int] = NeonColors.ELECTRIC_BLUE
    accent_color: Tuple[int, int, int] = NeonColors.ELECTRIC_PURPLE
    core_color: Tuple[int, int, int] = NeonColors.WHITE


class NeonRenderer:
    """
    High-performance Neon AR rendering engine.
    Keeps all visual rendering decoupled from hand detection/tracking.
    """

    HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # Index
        (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17),                                # Palm closure
    )

    FINGERTIP_SET = frozenset(FINGERTIP_INDICES.values())

    def __init__(
        self,
        config: Optional[NeonEffectConfig] = None,
        colors: Optional[NeonColors] = None,
        palette: Optional[NeonPalette] = None,
        backdrop_dim: float = 0.85,
        blur_downscale: float = 0.5,
    ) -> None:
        self.config = config or NeonEffectConfig()
        self.colors = colors or NeonColors()
        self.palette = palette or NeonPalette(
            skeleton_color=self.colors.CYAN,
            joint_color=self.colors.ELECTRIC_BLUE,
            accent_color=self.colors.ELECTRIC_PURPLE,
            core_color=self.colors.WHITE,
        )
        self.backdrop_dim = max(0.2, min(1.0, backdrop_dim))
        self.blur_downscale = max(0.25, min(1.0, blur_downscale))

        # Reusable off-screen buffers to avoid per-frame allocations
        self._glow_buffer: Optional[np.ndarray] = None
        self._core_buffer: Optional[np.ndarray] = None
        self._start_time: float = time.perf_counter()

    def set_glow_intensity(self, intensity: float) -> None:
        """Dynamically updates glow intensity (>= 0.0)."""
        self.config.glow_intensity = max(0.0, float(intensity))

    def set_palette(self, palette: NeonPalette) -> None:
        """Dynamically updates the active neon color palette."""
        self.palette = palette

    def _ensure_buffers(self, shape: Tuple[int, int, int]) -> None:
        """Allocates or resets reusable glow and core canvas buffers."""
        if self._glow_buffer is None or self._glow_buffer.shape != shape:
            self._glow_buffer = np.zeros(shape, dtype=np.uint8)
            self._core_buffer = np.zeros(shape, dtype=np.uint8)
        else:
            self._glow_buffer.fill(0)
            self._core_buffer.fill(0)

    @staticmethod
    def _extract_palm_center_and_radius(hand: HandLike) -> Tuple[Tuple[int, int], int]:
        """Extracts smoothed palm center and scale-aware radius from hand landmarks."""
        pts = hand.landmarks_px
        if isinstance(hand, ProcessedHand):
            cx, cy = hand.palm_center
        else:
            xs = [pts[i][0] for i in PALM_INDICES]
            ys = [pts[i][1] for i in PALM_INDICES]
            cx = int(round(sum(xs) / len(PALM_INDICES)))
            cy = int(round(sum(ys) / len(PALM_INDICES)))

        # Scale palm reticle relative to wrist-to-middle-MCP distance
        wx, wy = pts[0]
        mx, my = pts[9]
        palm_span = math.hypot(mx - wx, my - wy)
        radius = max(18, min(65, int(palm_span * 0.42)))
        return (cx, cy), radius

    def _get_hand_colors(
        self,
        hand: HandLike,
        override_color: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        """Returns (primary_skeleton, secondary_joint, accent_ring) BGR colors."""
        if override_color is not None:
            return override_color, self.palette.joint_color, self.palette.accent_color

        if getattr(hand, "handedness", "Right").lower() == "left":
            return (
                self.palette.joint_color,     # Electric Blue primary
                self.palette.accent_color,    # Purple joints
                self.palette.skeleton_color,  # Cyan accents
            )
        return (
            self.palette.skeleton_color,      # Cyan primary
            self.palette.joint_color,         # Electric Blue joints
            self.palette.accent_color,        # Purple accents
        )

    def _draw_palm_reticle(
        self,
        glow_canvas: np.ndarray,
        core_canvas: np.ndarray,
        center: Tuple[int, int],
        base_radius: int,
        phase: float,
        primary_color: Tuple[int, int, int],
        accent_color: Tuple[int, int, int],
    ) -> None:
        """Draws an animated cyberpunk rotating reticle and pulsing energy ring at the palm."""
        pulse = 1.0 + 0.12 * math.sin(phase * 3.5)
        outer_r = max(12, int(base_radius * pulse))
        inner_r = max(6, int(base_radius * 0.55))

        rot_deg = (phase * 70.0) % 360.0
        rev_deg = (-phase * 45.0) % 360.0

        # Rotating segmented outer arcs
        for offset in (0.0, 120.0, 240.0):
            start_a = rot_deg + offset
            end_a = start_a + 70.0
            cv2.ellipse(
                glow_canvas,
                center,
                (outer_r, outer_r),
                0.0,
                start_a,
                end_a,
                accent_color,
                3,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                core_canvas,
                center,
                (outer_r, outer_r),
                0.0,
                start_a,
                end_a,
                self.palette.core_color,
                1,
                cv2.LINE_AA,
            )

        # Counter-rotating inner arcs
        for offset in (0.0, 180.0):
            start_a = rev_deg + offset
            end_a = start_a + 110.0
            cv2.ellipse(
                glow_canvas,
                center,
                (inner_r, inner_r),
                0.0,
                start_a,
                end_a,
                primary_color,
                2,
                cv2.LINE_AA,
            )

        # Radial energy ticks
        tick_inner = outer_r + 3
        tick_outer = outer_r + 8
        for idx in range(4):
            rad = math.radians(rot_deg + idx * 90.0)
            p1 = (
                int(center[0] + tick_inner * math.cos(rad)),
                int(center[1] + tick_inner * math.sin(rad)),
            )
            p2 = (
                int(center[0] + tick_outer * math.cos(rad)),
                int(center[1] + tick_outer * math.sin(rad)),
            )
            cv2.line(glow_canvas, p1, p2, primary_color, 2, cv2.LINE_AA)

        # Glowing palm core dot
        cv2.circle(glow_canvas, center, 4, accent_color, -1, cv2.LINE_AA)
        cv2.circle(core_canvas, center, 2, self.palette.core_color, -1, cv2.LINE_AA)

    def _draw_hand_to_buffers(
        self,
        glow_canvas: np.ndarray,
        core_canvas: np.ndarray,
        hand: HandLike,
        phase: float,
        override_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Draws a single hand's neon skeleton, fingertips, and palm animation onto shared buffers."""
        pts = hand.landmarks_px
        if len(pts) < 21:
            return

        primary, secondary, accent = self._get_hand_colors(hand, override_color)
        bone_glow_thickness = max(4, self.config.connection_thickness + 4)
        core_thickness = max(1, self.config.connection_thickness - 1)

        # 1. Neon skeletal connections
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            pt1, pt2 = pts[start_idx], pts[end_idx]
            cv2.line(glow_canvas, pt1, pt2, primary, bone_glow_thickness, cv2.LINE_AA)
            cv2.line(core_canvas, pt1, pt2, self.palette.core_color, core_thickness, cv2.LINE_AA)

        # 2. Animated palm reticle
        palm_center, palm_radius = self._extract_palm_center_and_radius(hand)
        self._draw_palm_reticle(
            glow_canvas, core_canvas, palm_center, palm_radius, phase, primary, accent
        )

        # 3. Landmarks & glowing fingertip halos
        tip_pulse = 1.0 + 0.18 * math.sin(phase * 5.0)
        extended_map: Dict[str, bool] = getattr(hand, "extended_fingers", {})
        tip_index_to_name = {idx: name for name, idx in FINGERTIP_INDICES.items()}

        for idx, pt in enumerate(pts):
            if idx in self.FINGERTIP_SET:
                finger_name = tip_index_to_name[idx]
                is_extended = extended_map.get(finger_name, True)
                ring_color = accent if is_extended else secondary
                halo_r = int((self.config.landmark_radius + 7) * (tip_pulse if is_extended else 0.9))

                cv2.circle(glow_canvas, pt, halo_r, ring_color, 2, cv2.LINE_AA)
                cv2.circle(glow_canvas, pt, self.config.landmark_radius + 2, primary, -1, cv2.LINE_AA)
                cv2.circle(core_canvas, pt, max(2, self.config.landmark_radius - 1), self.palette.core_color, -1, cv2.LINE_AA)
            else:
                cv2.circle(glow_canvas, pt, self.config.landmark_radius + 1, secondary, -1, cv2.LINE_AA)
                cv2.circle(core_canvas, pt, max(1, self.config.landmark_radius - 2), self.palette.core_color, -1, cv2.LINE_AA)

    def _apply_gaussian_bloom(self, glow_canvas: np.ndarray) -> np.ndarray:
        """
        Generates a smooth neon bloom using downscaled Gaussian blur for
        real-time webcam performance.
        """
        h, w = glow_canvas.shape[:2]
        kx, ky = self.config.blur_kernel_size
        # Ensure odd positive kernel sizes
        kx = max(3, kx | 1)
        ky = max(3, ky | 1)

        if self.blur_downscale < 1.0:
            small_w = max(1, int(w * self.blur_downscale))
            small_h = max(1, int(h * self.blur_downscale))
            small = cv2.resize(glow_canvas, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            skx = max(3, int(kx * self.blur_downscale) | 1)
            sky = max(3, int(ky * self.blur_downscale) | 1)
            blurred_small = cv2.GaussianBlur(small, (skx, sky), 0)
            return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

        return cv2.GaussianBlur(glow_canvas, (kx, ky), 0)

    def render_hands(
        self,
        frame: np.ndarray,
        hands: Sequence[HandLike],
        phase: Optional[float] = None,
    ) -> np.ndarray:
        """
        Renders neon AR effects for zero, one, or two+ hands in a single
        optimized pass and composites them over the webcam frame.
        """
        if frame is None or frame.size == 0:
            return frame

        # Subtle dark cyberpunk contrast adjustment
        if self.backdrop_dim < 1.0:
            base_frame = cv2.convertScaleAbs(frame, alpha=self.backdrop_dim, beta=0)
        else:
            base_frame = frame

        if not hands:
            return base_frame

        self._ensure_buffers(frame.shape)
        assert self._glow_buffer is not None and self._core_buffer is not None

        anim_phase = phase if phase is not None else (time.perf_counter() - self._start_time)

        for hand in hands:
            self._draw_hand_to_buffers(self._glow_buffer, self._core_buffer, hand, anim_phase)

        bloom = self._apply_gaussian_bloom(self._glow_buffer)

        # Composite: dimmed_frame + glow_intensity * bloom + crisp neon core
        blended = cv2.addWeighted(base_frame, 1.0, bloom, self.config.glow_intensity, 0)
        cv2.add(blended, self._glow_buffer, dst=blended)
        cv2.add(blended, self._core_buffer, dst=blended)
        return blended

    def render_hand(
        self,
        frame: np.ndarray,
        hand: HandLike,
        neon_color: Optional[Tuple[int, int, int]] = None,
        phase: Optional[float] = None,
    ) -> np.ndarray:
        """Convenience method to render a single hand with optional color override."""
        if frame is None or frame.size == 0:
            return frame

        base_frame = (
            cv2.convertScaleAbs(frame, alpha=self.backdrop_dim, beta=0)
            if self.backdrop_dim < 1.0
            else frame
        )
        self._ensure_buffers(frame.shape)
        assert self._glow_buffer is not None and self._core_buffer is not None

        anim_phase = phase if phase is not None else (time.perf_counter() - self._start_time)
        self._draw_hand_to_buffers(
            self._glow_buffer, self._core_buffer, hand, anim_phase, override_color=neon_color
        )
        bloom = self._apply_gaussian_bloom(self._glow_buffer)

        blended = cv2.addWeighted(base_frame, 1.0, bloom, self.config.glow_intensity, 0)
        cv2.add(blended, self._glow_buffer, dst=blended)
        cv2.add(blended, self._core_buffer, dst=blended)
        return blended
