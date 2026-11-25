"""Camera acquisition module for NEON HAND AI."""
from camera.camera_stream import CameraStream
from camera.webcam import Webcam, WebcamError

__all__ = ["CameraStream", "Webcam", "WebcamError"]

