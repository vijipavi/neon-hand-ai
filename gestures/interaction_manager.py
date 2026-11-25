"""
HCI Interaction & State Management Module for NEON HAND AI.
Maps stabilized hand gestures to AR application states using edge-triggered
action dispatching (preventing repeated triggers while a gesture is held),
smooth cursor interpolation, and smooth pinch-zoom scaling.

Strictly decoupled from OpenCV rendering logic.
"""

from dataclasses import dataclass
import time
from typing import Dict, Optional, Sequence, Tuple

from configuration.settings import NeonColors
from gestures.gesture_recognition import GestureLabel, GesturePrediction
from tracking.landmark_processor import FINGERTIP_INDICES, ProcessedHand


@dataclass(frozen=True)
class NeonThemePreset:
    """Named neon color theme preset in BGR space."""
    name: str
    primary: Tuple[int, int, int]
    secondary: Tuple[int, int, int]
    accent: Tuple[int, int, int]


DEFAULT_THEMES: Tuple[NeonThemePreset, ...] = (
    NeonThemePreset(
        name="CYBER CYAN",
        primary=NeonColors.CYAN,
        secondary=NeonColors.ELECTRIC_BLUE,
        accent=NeonColors.ELECTRIC_PURPLE,
    ),
    NeonThemePreset(
        name="SYNTHWAVE PLASMA",
        primary=NeonColors.MAGENTA,
        secondary=NeonColors.ELECTRIC_PURPLE,
        accent=NeonColors.CYAN,
    ),
    NeonThemePreset(
        name="MATRIX EMERALD",
        primary=NeonColors.NEON_GREEN,
        secondary=NeonColors.CYAN,
        accent=NeonColors.NEON_YELLOW,
    ),
    NeonThemePreset(
        name="SOLAR OVERDRIVE",
        primary=NeonColors.NEON_YELLOW,
        secondary=NeonColors.HOT_PINK,
        accent=NeonColors.ELECTRIC_BLUE,
    ),
)


@dataclass
class InteractionState:
    """Immutable-style snapshot of current AR interaction state for renderers."""
    active_gesture: GestureLabel
    active_handedness: str
    confidence: float
    hologram_active: bool
    special_effect_active: bool
    shockwave_progress: Optional[float]
    shockwave_origin: Optional[Tuple[int, int]]
    confirmation_progress: Optional[float]
    confirmation_origin: Optional[Tuple[int, int]]
    cursor_active: bool
    cursor_pos: Optional[Tuple[int, int]]
    theme_index: int
    theme: NeonThemePreset
    zoom_scale: float
    zoom_active: bool
    feedback_message: str
    feedback_alpha: float


class InteractionManager:
    """
    Stateful HCI coordinator connecting gesture predictions to AR system states.
    Uses rising-edge detection + cooldowns so discrete actions fire once per pose.
    """

    def __init__(
        self,
        themes: Sequence[NeonThemePreset] = DEFAULT_THEMES,
        cursor_smoothing: float = 0.35,
        zoom_smoothing: float = 0.25,
        action_cooldown_sec: float = 0.45,
        confirmation_duration_sec: float = 1.1,
        shockwave_duration_sec: float = 0.85,
        feedback_duration_sec: float = 1.8,
    ) -> None:
        self.themes = tuple(themes)
        self.cursor_smoothing = cursor_smoothing
        self.zoom_smoothing = zoom_smoothing
        self.action_cooldown_sec = action_cooldown_sec
        self.confirmation_duration_sec = confirmation_duration_sec
        self.shockwave_duration_sec = shockwave_duration_sec
        self.feedback_duration_sec = feedback_duration_sec

        # Persistent interactive states
        self.hologram_active: bool = True
        self.special_effect_active: bool = False
        self.theme_index: int = 0
        self.zoom_scale: float = 1.0

        # Smooth continuous tracking state
        self._cursor_pos: Optional[Tuple[float, float]] = None

        # Edge-trigger & cooldown state per hand
        self._prev_gestures: Dict[str, GestureLabel] = {}
        self._last_trigger_time: Dict[str, float] = {}

        # Timed animation states
        self._confirm_timer: float = 0.0
        self._confirm_origin: Optional[Tuple[int, int]] = None
        self._shockwave_timer: float = 0.0
        self._shockwave_origin: Optional[Tuple[int, int]] = None
        self._feedback_message: str = "SYSTEM READY"
        self._feedback_timer: float = 1.2
        self._last_time: float = time.perf_counter()

    def _set_feedback(self, message: str) -> None:
        self._feedback_message = message
        self._feedback_timer = self.feedback_duration_sec

    def _can_trigger_edge(
        self,
        hand_key: str,
        gesture: GestureLabel,
        now: float,
    ) -> bool:
        """
        Returns True only on a rising edge (gesture just entered) and after
        the action cooldown has elapsed, preventing repeated triggers while held.
        """
        prev = self._prev_gestures.get(hand_key, GestureLabel.UNKNOWN)
        if prev == gesture:
            return False

        action_key = f"{hand_key}:{gesture.value}"
        last_t = self._last_trigger_time.get(action_key, -999.0)
        if (now - last_t) < self.action_cooldown_sec:
            return False

        self._last_trigger_time[action_key] = now
        return True

    def update(
        self,
        hands: Sequence[ProcessedHand],
        predictions: Sequence[GesturePrediction],
        dt: Optional[float] = None,
    ) -> InteractionState:
        """
        Updates interaction states from the latest processed hands and gesture
        predictions, returning an InteractionState snapshot for rendering.
        """
        now = time.perf_counter()
        step_dt = dt if dt is not None else min(0.1, max(0.001, now - self._last_time))
        self._last_time = now

        # Advance active timers
        if self._confirm_timer > 0.0:
            self._confirm_timer = max(0.0, self._confirm_timer - step_dt)
        if self._shockwave_timer > 0.0:
            self._shockwave_timer = max(0.0, self._shockwave_timer - step_dt)
        if self._feedback_timer > 0.0:
            self._feedback_timer = max(0.0, self._feedback_timer - step_dt)

        cursor_active = False
        zoom_active = False
        primary_gesture = GestureLabel.UNKNOWN
        primary_handedness = "None"
        primary_conf = 0.0
        active_keys = set()

        seen_counts: Dict[str, int] = {}
        for hand, pred in zip(hands, predictions):
            label = hand.handedness
            idx = seen_counts.get(label, 0)
            seen_counts[label] = idx + 1
            hand_key = f"{label}_{idx}"
            active_keys.add(hand_key)

            gesture = pred.gesture
            if primary_gesture == GestureLabel.UNKNOWN and gesture != GestureLabel.UNKNOWN:
                primary_gesture = gesture
                primary_handedness = hand.handedness
                primary_conf = pred.confidence

            is_rising = self._can_trigger_edge(hand_key, gesture, now)

            # 1. OPEN PALM -> Activate holographic interface
            if gesture == GestureLabel.OPEN_PALM and is_rising:
                self.hologram_active = True
                self.special_effect_active = False
                self._set_feedback("HOLOGRAM INTERFACE ACTIVATED")

            # 2. CLOSED FIST -> Activate special energy shockwave effect
            elif gesture == GestureLabel.CLOSED_FIST and is_rising:
                self.special_effect_active = True
                self._shockwave_timer = self.shockwave_duration_sec
                self._shockwave_origin = hand.palm_center
                self._set_feedback("ENERGY SURGE ACTIVATED")

            # 3. THUMBS UP -> Display confirmation animation
            elif gesture == GestureLabel.THUMBS_UP and is_rising:
                self._confirm_timer = self.confirmation_duration_sec
                self._confirm_origin = hand.palm_center
                self._set_feedback("ACTION CONFIRMED")

            # 4. PEACE SIGN -> Cycle neon visual theme
            elif gesture == GestureLabel.PEACE_SIGN and is_rising:
                self.theme_index = (self.theme_index + 1) % len(self.themes)
                theme_name = self.themes[self.theme_index].name
                self._set_feedback(f"THEME: {theme_name}")

            # 5. INDEX POINTING -> Smooth virtual cursor control
            elif gesture == GestureLabel.INDEX_POINTING:
                cursor_active = True
                tip_x, tip_y = hand.landmarks_px[FINGERTIP_INDICES["index"]]
                if self._cursor_pos is None:
                    self._cursor_pos = (float(tip_x), float(tip_y))
                else:
                    a = self.cursor_smoothing
                    self._cursor_pos = (
                        a * tip_x + (1.0 - a) * self._cursor_pos[0],
                        a * tip_y + (1.0 - a) * self._cursor_pos[1],
                    )
                if is_rising:
                    self._set_feedback("VIRTUAL CURSOR ACTIVE")

            # 6. PINCH -> Smoothly zoom virtual object in/out
            elif gesture == GestureLabel.PINCH:
                zoom_active = True
                # Map pinch_ratio [0.05 .. 0.38] -> zoom [0.5 .. 2.5]
                clamped_r = max(0.05, min(0.38, pred.pinch_ratio))
                norm_t = (clamped_r - 0.05) / (0.38 - 0.05)
                target_zoom = 0.5 + norm_t * 2.0
                self.zoom_scale = (
                    self.zoom_smoothing * target_zoom
                    + (1.0 - self.zoom_smoothing) * self.zoom_scale
                )
                if is_rising:
                    self._set_feedback("PINCH ZOOM ACTIVE")

            self._prev_gestures[hand_key] = gesture

        # Reset gesture edge history for hands that left the frame
        for stale in [k for k in self._prev_gestures if k not in active_keys]:
            del self._prev_gestures[stale]

        if not cursor_active:
            self._cursor_pos = None

        shockwave_progress = (
            1.0 - (self._shockwave_timer / self.shockwave_duration_sec)
            if self._shockwave_timer > 0.0
            else None
        )
        confirm_progress = (
            1.0 - (self._confirm_timer / self.confirmation_duration_sec)
            if self._confirm_timer > 0.0
            else None
        )
        feedback_alpha = min(1.0, self._feedback_timer / 0.5) if self._feedback_timer > 0.0 else 0.0

        cursor_int = (
            (int(round(self._cursor_pos[0])), int(round(self._cursor_pos[1])))
            if self._cursor_pos is not None
            else None
        )

        return InteractionState(
            active_gesture=primary_gesture,
            active_handedness=primary_handedness,
            confidence=primary_conf,
            hologram_active=self.hologram_active,
            special_effect_active=self.special_effect_active,
            shockwave_progress=shockwave_progress,
            shockwave_origin=self._shockwave_origin,
            confirmation_progress=confirm_progress,
            confirmation_origin=self._confirm_origin,
            cursor_active=cursor_active,
            cursor_pos=cursor_int,
            theme_index=self.theme_index,
            theme=self.themes[self.theme_index],
            zoom_scale=self.zoom_scale,
            zoom_active=zoom_active,
            feedback_message=self._feedback_message,
            feedback_alpha=feedback_alpha,
        )

