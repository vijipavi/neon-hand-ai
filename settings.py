"""
Root level re-export of application settings for convenience.
Allows both `import settings` and `from configuration.settings import settings`.
"""

from configuration.settings import (
    Settings,
    CameraConfig,
    TrackingConfig,
    NeonEffectConfig,
    UIConfig,
    NeonColors,
    settings,
)

__all__ = [
    "Settings",
    "CameraConfig",
    "TrackingConfig",
    "NeonEffectConfig",
    "UIConfig",
    "NeonColors",
    "settings",
]

