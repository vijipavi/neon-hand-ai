"""
Global Application Configuration for NEON HAND AI.
Defines camera parameters, MediaPipe tracking thresholds,
neon color palettes, and UI display settings.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class NeonColors:
    """
    BGR Color definitions for the Neon Cyberpunk aesthetic.
    OpenCV operates natively in BGR color space.
    """
    CYAN: Tuple[int, int, int] = (255, 255, 0)
    ELECTRIC_BLUE: Tuple[int, int, int] = (255, 128, 0)
    MAGENTA: Tuple[int, int, int] = (255, 0, 255)
    HOT_PINK: Tuple[int, int, int] = (180, 105, 255)
    NEON_GREEN: Tuple[int, int, int] = (0, 255, 128)
    LIME: Tuple[int, int, int] = (50, 255, 50)
    ELECTRIC_PURPLE: Tuple[int, int, int] = (255, 0, 128)
    NEON_YELLOW: Tuple[int, int, int] = (0, 230, 255)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    DARK_BG: Tuple[int, int, int] = (15, 15, 20)


@dataclass
class CameraConfig:
    """Webcam capture specifications."""
    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    flip_horizontal: bool = True  # Mirror view for natural interaction
    buffer_size: int = 1


@dataclass
class TrackingConfig:
    """MediaPipe Hand Tracking hyper-parameters."""
    max_num_hands: int = 2
    model_complexity: int = 1  # 0: Lite, 1: Full (used by Solutions fallback)
    min_detection_confidence: float = 0.7
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_path: str = "assets/models/hand_landmarker.task"


@dataclass
class NeonEffectConfig:
    """Parameters controlling AR glow intensity and rendering."""
    glow_intensity: float = 0.8
    blur_kernel_size: Tuple[int, int] = (15, 15)
    landmark_radius: int = 4
    connection_thickness: int = 2
    glow_layers: int = 3
    particle_trail_enabled: bool = True
    max_trail_points: int = 20


@dataclass
class UIConfig:
    """Heads-Up Display (HUD) and window configurations."""
    window_name: str = "NEON HAND AI - Interactive AR System"
    show_fps: bool = True
    show_hud: bool = True
    show_landmarks: bool = True
    show_gesture_badge: bool = True
    font_scale: float = 0.6
    font_thickness: int = 1


@dataclass
class Settings:
    """Master Application Settings container."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    effects: NeonEffectConfig = field(default_factory=NeonEffectConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    colors: NeonColors = field(default_factory=NeonColors)


# Default singleton instance for easy import
settings = Settings()

