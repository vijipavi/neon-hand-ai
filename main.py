"""
==============================================================================
NEON HAND AI - AI-Powered Hand Tracking and Interactive AR System
Entry Point & Modular Pipeline Orchestrator
==============================================================================
"""

import argparse
import sys
import time
from typing import Optional
import numpy as np

# Verify critical packages with informative diagnostic reporting
def check_dependencies() -> bool:
    """Verifies that opencv, mediapipe, and numpy are correctly installed and compatible."""
    missing = []
    versions = {}

    try:
        import numpy as np
        versions["numpy"] = np.__version__
        # Check for NumPy 2.x incompatibility with MediaPipe
        np_major = int(np.__version__.split(".")[0])
        if np_major >= 2:
            print(f"[!] Warning: NumPy version {np.__version__} detected.")
            print("    MediaPipe 0.10.x requires NumPy < 2.0.0. Please install numpy<2.0.0.")
    except ImportError:
        missing.append("numpy")

    try:
        import cv2
        versions["opencv"] = cv2.__version__
    except ImportError:
        missing.append("opencv-python")

    try:
        import mediapipe as mp
        versions["mediapipe"] = mp.__version__
    except ImportError:
        missing.append("mediapipe")

    print("=" * 60)
    print("NEON HAND AI - Environment & Dependency Diagnostic")
    print("=" * 60)
    for pkg, ver in versions.items():
        print(f" [+] {pkg:<15}: {ver}")

    if missing:
        print("\n [X] Missing dependencies:")
        for pkg in missing:
            print(f"     - {pkg}")
        print("\n Please activate your virtual environment and run:")
        print("     pip install -r requirements.txt\n")
        return False

    print(" [✓] Core dependencies verified successfully.\n")
    return True


def render_standby_frame(width: int, height: int, message: str) -> np.ndarray:
    """Generates a stylish cyberpunk diagnostic canvas when camera is unavailable."""
    import cv2
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 15, 20)  # Dark cybernetic background

    # Border glow
    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (255, 255, 0), 2)
    cv2.rectangle(frame, (14, 14), (width - 14, height - 14), (255, 0, 255), 1)

    # Header
    cv2.putText(frame, "NEON HAND AI : STANDBY MODE", (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

    # Status Message
    cv2.putText(frame, f"STATUS: {message}", (40, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1, cv2.LINE_AA)

    # Info
    info_lines = [
        "Architecture initialized in modular setup mode.",
        "Module Check:",
        "  [+] camera.camera_stream       : READY",
        "  [+] tracking.hand_tracker      : READY",
        "  [+] gestures.gesture_detector  : READY",
        "  [+] effects.neon_renderer      : READY",
        "  [+] ui.hud_overlay             : READY",
        "",
        "Press 'Q' or 'ESC' to exit standby."
    ]

    for idx, line in enumerate(info_lines):
        color = (255, 0, 180) if line.startswith("  [+]") else (200, 200, 200)
        cv2.putText(frame, line, (40, 180 + (idx * 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="NEON HAND AI - AR Hand Tracking System")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Target frame width")
    parser.add_argument("--height", type=int, default=720, help="Target frame height")
    parser.add_argument("--cli", action="store_true", help="Launch OpenCV CLI window instead of PySide6 GUI")
    parser.add_argument("--check-env", action="store_true", help="Run environment check and exit")
    return parser.parse_args()


def launch_gui() -> None:
    """Launches the PySide6 futuristic dark-themed desktop application."""
    from PySide6.QtWidgets import QApplication
    from ui.main_window import NeonMainWindow

    app = QApplication(sys.argv)
    window = NeonMainWindow()
    window.show()
    sys.exit(app.exec())


def main():
    args = parse_args()

    # Step 1: Check environment dependencies
    if not check_dependencies():
        sys.exit(1)

    if args.check_env:
        print("[✓] Environment check passed. Exiting (--check-env).")
        return

    if not args.cli:
        launch_gui()
        return

    # Step 2: Import project modules (safe after dependency verification)
    import cv2
    from configuration.settings import settings
    from camera.webcam import Webcam, WebcamError
    from effects.hologram import HologramRenderer
    from effects.interaction_renderer import InteractionRenderer
    from effects.neon_renderer import NeonPalette, NeonRenderer
    from effects.particle_effects import ParticleSystem
    from gestures.gesture_recognition import GestureRecognizer
    from gestures.interaction_manager import InteractionManager
    from tracking.hand_detector import HandDetector
    from tracking.landmark_processor import LandmarkProcessor
    from ui.hud_overlay import HUDOverlay

    # Override settings with CLI arguments if provided
    settings.camera.device_index = args.camera_id
    settings.camera.frame_width = args.width
    settings.camera.frame_height = args.height

    print("[*] Initializing NEON HAND AI interactive AR pipeline...")
    webcam = Webcam(config=settings.camera, window_name=settings.ui.window_name)
    hand_detector = HandDetector(config=settings.tracking)
    landmark_processor = LandmarkProcessor()
    gesture_recognizer = GestureRecognizer()
    interaction_manager = InteractionManager()
    neon_renderer = NeonRenderer(config=settings.effects, colors=settings.colors)
    hologram_renderer = HologramRenderer()
    particle_system = ParticleSystem()
    interaction_renderer = InteractionRenderer()
    hud = HUDOverlay(config=settings.ui, colors=settings.colors)

    def process_frame(frame: np.ndarray) -> np.ndarray:
        """Full AR pipeline: Detect -> Process -> Recognize -> Interact -> Render."""
        h, w, _ = frame.shape
        hands = hand_detector.detect(frame)
        processed_hands = landmark_processor.process_hands(hands, w, h)
        gestures = gesture_recognizer.recognize_hands(processed_hands, w, h)
        state = interaction_manager.update(processed_hands, gestures)

        # Apply active theme to neon & hologram renderers
        neon_renderer.set_palette(
            NeonPalette(
                skeleton_color=state.theme.primary,
                joint_color=state.theme.secondary,
                accent_color=state.theme.accent,
            )
        )
        hologram_renderer.config.primary_color = state.theme.primary
        hologram_renderer.config.secondary_color = state.theme.secondary
        hologram_renderer.config.accent_color = state.theme.accent

        frame = neon_renderer.render_hands(frame, processed_hands)
        if state.hologram_active:
            frame = hologram_renderer.render(frame, processed_hands)
        frame = particle_system.render(frame, processed_hands)
        frame = interaction_renderer.render(frame, state)

        return hud.draw(frame, gestures=gestures)

    try:
        webcam.display(frame_processor=process_frame)
    except WebcamError as exc:
        print(f"[!] Warning: {exc}")
        print("    Displaying diagnostic standby window. Connect a webcam or check permissions.")

        cv2.namedWindow(settings.ui.window_name, cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                standby = render_standby_frame(
                    settings.camera.frame_width,
                    settings.camera.frame_height,
                    "Camera offline - Press 'ESC' to exit",
                )
                cv2.imshow(settings.ui.window_name, standby)
                key = cv2.waitKey(30) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:
                    print("[*] Exit signal received. Terminating gracefully...")
                    break
        except KeyboardInterrupt:
            print("\n[*] Interrupted by user.")
        finally:
            cv2.destroyAllWindows()
            print("[✓] All resources released cleanly. Goodbye.")
    finally:
        hand_detector.close()



if __name__ == "__main__":
    main()

