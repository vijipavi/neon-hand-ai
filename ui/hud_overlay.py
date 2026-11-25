"""
HUD (Heads-Up Display) Overlay.
Renders cyberpunk status telemetry, real-time FPS counter,
tracking metrics, and detected gesture badges .
"""

from typing import List, Optional
import time
import cv2
import numpy as np

from configuration.settings import NeonColors, UIConfig
from gestures.gesture_detector import GestureResult


class HUDOverlay:
    """Renders cybernetic HUD elements on top of the AR frame."""

    def __init__(
        self,
        config: Optional[UIConfig] = None,
        colors: Optional[NeonColors] = None,
    ) -> None:
        self.config = config or UIConfig()
        self.colors = colors or NeonColors()
        self._prev_time: float = time.time()
        self._fps: float = 0.0

    def update_fps(self) -> float:
        """Calculates rolling FPS based on frame-to-frame delta time."""
        curr_time = time.time()
        delta = curr_time - self._prev_time
        self._prev_time = curr_time
        if delta > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / delta)
        return self._fps

    def draw(
        self,
        frame: np.ndarray,
        gestures: Optional[List[GestureResult]] = None,
    ) -> np.ndarray:
        """Draws telemetry panels and gesture labels."""
        if not self.config.show_hud:
            return frame

        fps = self.update_fps()

        # HUD Top Banner / Title
        cv2.putText(
            frame,
            "// NEON HAND AI : ONLINE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors.CYAN,
            2,
            cv2.LINE_AA,
        )

        # FPS Telemetry
        if self.config.show_fps:
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(
                frame,
                fps_text,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.colors.NEON_GREEN,
                1,
                cv2.LINE_AA,
            )

        # Active Gestures Display
        if gestures and self.config.show_gesture_badge:
            for idx, g in enumerate(gestures):
                y_offset = 100 + (idx * 30)
                gesture_text = f"HAND [{idx + 1}]: {g.gesture.value} ({int(g.confidence * 100)}%)"
                cv2.putText(
                    frame,
                    gesture_text,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    self.colors.MAGENTA,
                    1,
                    cv2.LINE_AA,
                )

        # Bottom control guide
        h, _, _ = frame.shape
        cv2.putText(
            frame,
            "[ESC / Q] Exit System   [H] Toggle HUD",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.colors.WHITE,
            1,
            cv2.LINE_AA,
        )

        return frame

