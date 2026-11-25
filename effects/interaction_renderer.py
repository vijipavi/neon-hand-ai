"""
Interactive AR Visual Renderer for NEON HAND AI.
Renders virtual cursor, zoomable 3D wireframe AR object, fist shockwave
surge, thumbs-up confirmation animation, and HUD interaction feedback
purely from InteractionState (zero gesture logic inside).
"""

import math
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from configuration.settings import NeonColors
from gestures.interaction_manager import InteractionState


class InteractionRenderer:
    """Renders HCI visual elements driven by InteractionState."""

    # 3D Octahedron vertices and edges for the zoomable virtual AR object
    _OCT_VERTICES: Tuple[Tuple[float, float, float], ...] = (
        (0.0, -1.2, 0.0),   # Top
        (1.0, 0.0, 0.0),    # Right
        (0.0, 0.0, 1.0),    # Front
        (-1.0, 0.0, 0.0),   # Left
        (0.0, 0.0, -1.0),   # Back
        (0.0, 1.2, 0.0),    # Bottom
    )
    _OCT_EDGES: Tuple[Tuple[int, int], ...] = (
        (0, 1), (0, 2), (0, 3), (0, 4),
        (5, 1), (5, 2), (5, 3), (5, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
    )

    def __init__(self) -> None:
        self._start_time: float = time.perf_counter()

    def _draw_virtual_cursor(
        self,
        frame: np.ndarray,
        pos: Tuple[int, int],
        color: Tuple[int, int, int],
        accent: Tuple[int, int, int],
        phase: float,
    ) -> None:
        """Draws a futuristic animated crosshair cursor at `pos`."""
        cx, cy = pos
        r_outer = int(18 + 3 * math.sin(phase * 6.0))
        rot = (phase * 120.0) % 360.0

        cv2.circle(frame, (cx, cy), r_outer, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, NeonColors.WHITE, -1, cv2.LINE_AA)

        for i in range(4):
            a1 = rot + i * 90.0 - 15.0
            a2 = rot + i * 90.0 + 15.0
            cv2.ellipse(frame, (cx, cy), (r_outer + 6, r_outer + 6), 0.0, a1, a2, accent, 2, cv2.LINE_AA)

        # Crosshair ticks
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p1 = (cx + dx * 8, cy + dy * 8)
            p2 = (cx + dx * (r_outer + 4), cy + dy * (r_outer + 4))
            cv2.line(frame, p1, p2, NeonColors.WHITE, 1, cv2.LINE_AA)

    def _draw_zoomable_object(
        self,
        frame: np.ndarray,
        state: InteractionState,
        phase: float,
    ) -> None:
        """Draws a rotating 3D wireframe crystal whose scale tracks pinch zoom."""
        h, w = frame.shape[:2]
        anchor_x = w - 150
        anchor_y = 140
        scale = 34.0 * state.zoom_scale

        angle_y = phase * 1.3
        angle_x = 0.45 + 0.2 * math.sin(phase * 0.8)
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)

        projected: List[Tuple[int, int]] = []
        for vx, vy, vz in self._OCT_VERTICES:
            # Rotate around Y then X
            rx = vx * cos_y + vz * sin_y
            rz = -vx * sin_y + vz * cos_y
            ry = vy * cos_x - rz * sin_x
            projected.append((int(anchor_x + rx * scale), int(anchor_y + ry * scale)))

        color = state.theme.accent if state.zoom_active else state.theme.primary
        thickness = 2 if state.zoom_active else 1

        for i, j in self._OCT_EDGES:
            cv2.line(frame, projected[i], projected[j], color, thickness, cv2.LINE_AA)

        for pt in projected:
            cv2.circle(frame, pt, 3, NeonColors.WHITE, -1, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"AR CORE // {state.zoom_scale:.2f}x",
            (anchor_x - 62, anchor_y + int(scale * 1.45) + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_shockwave(
        frame: np.ndarray,
        origin: Tuple[int, int],
        progress: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Draws an expanding cybernetic shockwave ring for Closed Fist."""
        radius = int(25 + progress * 190)
        fade = max(0.0, 1.0 - progress)
        ring_color = (int(color[0] * fade), int(color[1] * fade), int(color[2] * fade))
        cv2.circle(frame, origin, radius, ring_color, 4, cv2.LINE_AA)
        cv2.circle(frame, origin, max(5, int(radius * 0.65)), NeonColors.WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_confirmation(
        frame: np.ndarray,
        origin: Tuple[int, int],
        progress: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Draws a glowing confirmation checkmark animation for Thumbs Up."""
        fade = max(0.0, 1.0 - max(0.0, (progress - 0.5) * 2.0))
        c = (int(color[0] * fade), int(color[1] * fade), int(color[2] * fade))
        radius = int(32 + math.sin(min(1.0, progress * 2.0) * math.pi * 0.5) * 14)

        cv2.circle(frame, origin, radius, c, 3, cv2.LINE_AA)
        # Checkmark polyline
        ox, oy = origin
        pts = np.array(
            [[ox - 14, oy + 2], [ox - 3, oy + 13], [ox + 18, oy - 12]],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts], False, NeonColors.WHITE, 3, cv2.LINE_AA)
        cv2.putText(
            frame,
            "CONFIRMED",
            (ox - 46, oy - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            c,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_feedback_banner(frame: np.ndarray, state: InteractionState) -> None:
        """Draws active interaction feedback and theme indicator at the bottom-right."""
        h, w = frame.shape[:2]
        if state.feedback_alpha > 0.05 and state.feedback_message:
            c = (
                int(state.theme.primary[0] * state.feedback_alpha),
                int(state.theme.primary[1] * state.feedback_alpha),
                int(state.theme.primary[2] * state.feedback_alpha),
            )
            cv2.putText(
                frame,
                f">> {state.feedback_message}",
                (20, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                c,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"THEME: {state.theme.name}",
            (w - 230, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            state.theme.accent,
            1,
            cv2.LINE_AA,
        )

    def render(
        self,
        frame: np.ndarray,
        state: InteractionState,
        phase: Optional[float] = None,
    ) -> np.ndarray:
        """Renders all active AR interaction visuals onto `frame`."""
        if frame is None or frame.size == 0:
            return frame

        anim_phase = phase if phase is not None else (time.perf_counter() - self._start_time)

        self._draw_zoomable_object(frame, state, anim_phase)

        if state.shockwave_progress is not None and state.shockwave_origin is not None:
            self._draw_shockwave(frame, state.shockwave_origin, state.shockwave_progress, state.theme.accent)

        if state.confirmation_progress is not None and state.confirmation_origin is not None:
            self._draw_confirmation(frame, state.confirmation_origin, state.confirmation_progress, NeonColors.NEON_GREEN)

        if state.cursor_active and state.cursor_pos is not None:
            self._draw_virtual_cursor(frame, state.cursor_pos, state.theme.primary, state.theme.accent, anim_phase)

        self._draw_feedback_banner(frame, state)
        return frame

