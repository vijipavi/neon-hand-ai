"""
Computer Vision Engine Bridge & Background Worker for PySide6 UI.
Runs webcam capture, MediaPipe hand detection, gesture recognition, and
AR rendering on a dedicated QThread, keeping the UI decoupled and responsive.
"""

from dataclasses import dataclass
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from camera.webcam import Webcam, WebcamError
from configuration.settings import settings
from effects.hologram import HologramRenderer
from effects.interaction_renderer import InteractionRenderer
from effects.neon_renderer import NeonPalette, NeonRenderer
from effects.particle_effects import ParticleSystem
from gestures.gesture_recognition import GestureLabel, GestureRecognizer
from gestures.interaction_manager import DEFAULT_THEMES, InteractionManager
from tracking.hand_detector import HandDetector
from tracking.landmark_processor import LandmarkProcessor


@dataclass
class HandTelemetry:
    """Telemetry summary for a single hand (Left or Right)."""
    present: bool = False
    gesture: str = "NONE"
    finger_count: int = 0
    confidence: float = 0.0
    gesture_confidence: float = 0.0


@dataclass
class TelemetrySnapshot:
    """Thread-safe snapshot emitted to the UI every frame."""
    fps: float
    hands_detected: int
    left_hand: HandTelemetry
    right_hand: HandTelemetry
    active_gesture: str
    theme_name: str
    zoom_scale: float
    feedback_message: str


class VisionEngine:
    """
    Pure Computer Vision & AR pipeline wrapper.
    Encapsulates all OpenCV/MediaPipe/Effects objects so the UI layer
    never imports CV libraries directly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.camera_index: int = settings.camera.device_index
        self.width: int = settings.camera.frame_width
        self.height: int = settings.camera.frame_height

        self.hand_detector = HandDetector(config=settings.tracking)
        self.landmark_processor = LandmarkProcessor()
        self.gesture_recognizer = GestureRecognizer()
        self.interaction_manager = InteractionManager()
        self.neon_renderer = NeonRenderer(config=settings.effects, colors=settings.colors)
        self.hologram_renderer = HologramRenderer()
        self.particle_system = ParticleSystem()
        self.interaction_renderer = InteractionRenderer()

        self._prev_time: float = time.perf_counter()
        self._fps: float = 0.0

    def set_glow_intensity(self, intensity: float) -> None:
        with self._lock:
            self.neon_renderer.set_glow_intensity(intensity)
            self.hologram_renderer.config.glow_intensity = intensity
            self.particle_system.config.glow_intensity = intensity

    def set_theme_index(self, index: int) -> None:
        with self._lock:
            if 0 <= index < len(DEFAULT_THEMES):
                self.interaction_manager.theme_index = index

    def set_particles_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.particle_system.config.enabled = enabled

    def set_hologram_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.hologram_renderer.config.enabled = enabled
            self.interaction_manager.hologram_active = enabled

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, TelemetrySnapshot]:
        """Processes one BGR frame through detection, gestures, and AR effects."""
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self._fps = 0.85 * self._fps + 0.15 * (1.0 / dt)

        with self._lock:
            h, w = frame_bgr.shape[:2]
            hands = self.hand_detector.detect(frame_bgr)
            processed_hands = self.landmark_processor.process_hands(hands, w, h)
            gestures = self.gesture_recognizer.recognize_hands(processed_hands, w, h)
            state = self.interaction_manager.update(processed_hands, gestures)

            # Sync theme colors across renderers
            self.neon_renderer.set_palette(
                NeonPalette(
                    skeleton_color=state.theme.primary,
                    joint_color=state.theme.secondary,
                    accent_color=state.theme.accent,
                )
            )
            self.hologram_renderer.config.primary_color = state.theme.primary
            self.hologram_renderer.config.secondary_color = state.theme.secondary
            self.hologram_renderer.config.accent_color = state.theme.accent

            out_frame = self.neon_renderer.render_hands(frame_bgr, processed_hands)
            if state.hologram_active and self.hologram_renderer.config.enabled:
                out_frame = self.hologram_renderer.render(out_frame, processed_hands)
            out_frame = self.particle_system.render(out_frame, processed_hands)
            out_frame = self.interaction_renderer.render(out_frame, state)

            left_t = HandTelemetry()
            right_t = HandTelemetry()
            for ph, pred in zip(processed_hands, gestures):
                info = HandTelemetry(
                    present=True,
                    gesture=pred.gesture.value,
                    finger_count=ph.finger_count,
                    confidence=ph.score,
                    gesture_confidence=pred.confidence,
                )
                if ph.handedness.lower() == "left":
                    left_t = info
                else:
                    right_t = info

            snapshot = TelemetrySnapshot(
                fps=self._fps,
                hands_detected=len(processed_hands),
                left_hand=left_t,
                right_hand=right_t,
                active_gesture=(
                    state.active_gesture.value
                    if state.active_gesture != GestureLabel.UNKNOWN
                    else "NONE"
                ),
                theme_name=state.theme.name,
                zoom_scale=state.zoom_scale,
                feedback_message=state.feedback_message,
            )
            return out_frame, snapshot

    def close(self) -> None:
        self.hand_detector.close()


class VisionWorker(QThread):
    """Background QThread running webcam capture and VisionEngine."""

    frame_ready = Signal(QImage)
    telemetry_updated = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        engine: Optional[VisionEngine] = None,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.engine = engine or VisionEngine()
        self._running = False

    def run(self) -> None:
        self._running = True
        webcam = Webcam(
            camera_index=self.camera_index,
            width=self.width,
            height=self.height,
        )
        try:
            webcam.open()
            aw, ah = webcam.actual_resolution
            self.status_changed.emit(f"ONLINE ({aw}x{ah})")

            while self._running and webcam.is_running:
                frame_bgr = webcam.read_frame()
                rendered_bgr, telemetry = self.engine.process_frame(frame_bgr)

                rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(
                    rgb.data, w, h, ch * w, QImage.Format.Format_RGB888
                ).copy()

                self.frame_ready.emit(qimg)
                self.telemetry_updated.emit(telemetry)
        except WebcamError as exc:
            self.error_occurred.emit(str(exc))
            self.status_changed.emit("CAMERA ERROR")
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Unexpected vision error: {exc}")
            self.status_changed.emit("ERROR")
        finally:
            webcam.release()
            self._running = False
            self.status_changed.emit("STANDBY")

    def stop(self) -> None:
        """Signals the capture loop to exit and waits for clean thread shutdown."""
        self._running = False
        self.wait(3000)

