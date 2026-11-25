"""
PySide6 Main Application Window for NEON HAND AI.
Provides a futuristic dark-themed GUI with live webcam preview, hand/gesture
telemetry, real-time FPS, theme & glow controls, effect toggles, camera
configuration, and start/stop tracking controls.

Strictly separated from OpenCV/MediaPipe CV logic.
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gestures.interaction_manager import DEFAULT_THEMES
from ui.styles import CYBERPUNK_QSS
from ui.vision_worker import HandTelemetry, TelemetrySnapshot, VisionEngine, VisionWorker


class NeonMainWindow(QMainWindow):
    """Main futuristic dark-themed AR application window."""

    RESOLUTIONS = (
        ("1280 x 720 (HD)", 1280, 720),
        ("640 x 480 (VGA)", 640, 480),
        ("1920 x 1080 (FHD)", 1920, 1080),
    )

    def __init__(self, engine: Optional[VisionEngine] = None) -> None:
        super().__init__()
        self.setWindowTitle("NEON HAND AI — Interactive AR System")
        self.resize(1360, 820)
        self.setMinimumSize(960, 620)
        self.setStyleSheet(CYBERPUNK_QSS)

        self.engine = engine or VisionEngine()
        self.worker: Optional[VisionWorker] = None
        self._last_pixmap: Optional[QPixmap] = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Header Bar
        header = QFrame()
        header.setObjectName("HeaderBar")
        header.setFixedHeight(64)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 8, 20, 8)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_lbl = QLabel("NEON HAND AI")
        title_lbl.setObjectName("AppTitle")
        sub_lbl = QLabel("AI-POWERED HAND TRACKING & INTERACTIVE AR SYSTEM")
        sub_lbl.setObjectName("AppSubtitle")
        title_col.addWidget(title_lbl)
        title_col.addWidget(sub_lbl)

        self.status_pill = QLabel("STANDBY")
        self.status_pill.setObjectName("StatusPill")
        self.fps_badge = QLabel("FPS: 00")
        self.fps_badge.setObjectName("FpsBadge")

        header_layout.addLayout(title_col)
        header_layout.addStretch()
        header_layout.addWidget(self.status_pill)
        header_layout.addSpacing(10)
        header_layout.addWidget(self.fps_badge)
        root_layout.addWidget(header)

        # 2. Main Split Content (Live Video + Right Sidebar)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)

        self.video_label = QLabel("CAMERA STANDBY\n\nClick 'Start Tracking' to initialize webcam")
        self.video_label.setObjectName("VideoSurface")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_label.setMinimumSize(560, 400)
        content_layout.addWidget(self.video_label, stretch=3)

        # Scrollable Right Control & Telemetry Panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(370)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 6, 0)
        sidebar_layout.setSpacing(12)

        sidebar_layout.addWidget(self._build_controls_group())
        sidebar_layout.addWidget(self._build_telemetry_group())
        sidebar_layout.addWidget(self._build_effects_group())
        sidebar_layout.addStretch()

        scroll.setWidget(sidebar)
        content_layout.addWidget(scroll, stretch=1)
        root_layout.addLayout(content_layout)

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("CAMERA & TRACKING CONTROLS")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.setObjectName("StartBtn")
        self.stop_btn = QPushButton("Stop Tracking")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setEnabled(False)

        layout.addWidget(self.start_btn, 0, 0)
        layout.addWidget(self.stop_btn, 0, 1)

        layout.addWidget(QLabel("Camera Device:"), 1, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0 (Default)", "Camera 1 (External)", "Camera 2"])
        layout.addWidget(self.camera_combo, 1, 1)

        layout.addWidget(QLabel("Resolution:"), 2, 0)
        self.resolution_combo = QComboBox()
        for label, _, _ in self.RESOLUTIONS:
            self.resolution_combo.addItem(label)
        layout.addWidget(self.resolution_combo, 2, 1)

        return group

    def _build_telemetry_group(self) -> QGroupBox:
        group = QGroupBox("HAND & GESTURE TELEMETRY")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.detection_status_lbl = QLabel("Detection: NO HANDS")
        self.active_gesture_lbl = QLabel("Active Gesture: NONE")
        self.zoom_lbl = QLabel("AR Core Zoom: 1.00x")
        self.feedback_lbl = QLabel("Status: SYSTEM READY")
        self.feedback_lbl.setStyleSheet("color: #00F5FF;")

        layout.addWidget(self.detection_status_lbl)
        layout.addWidget(self.active_gesture_lbl)
        layout.addWidget(self.zoom_lbl)
        layout.addWidget(self.feedback_lbl)

        # Left Hand Telemetry
        self.left_info_lbl = QLabel("Left Hand: Not detected")
        self.left_conf_bar = QProgressBar()
        self.left_conf_bar.setRange(0, 100)
        self.left_conf_bar.setFormat("Left Confidence: %p%")
        layout.addWidget(self.left_info_lbl)
        layout.addWidget(self.left_conf_bar)

        # Right Hand Telemetry
        self.right_info_lbl = QLabel("Right Hand: Not detected")
        self.right_conf_bar = QProgressBar()
        self.right_conf_bar.setRange(0, 100)
        self.right_conf_bar.setFormat("Right Confidence: %p%")
        layout.addWidget(self.right_info_lbl)
        layout.addWidget(self.right_conf_bar)

        return group

    def _build_effects_group(self) -> QGroupBox:
        group = QGroupBox("NEON AR EFFECTS & THEME")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Neon Theme:"))
        self.theme_combo = QComboBox()
        for preset in DEFAULT_THEMES:
            self.theme_combo.addItem(preset.name)
        theme_row.addWidget(self.theme_combo)
        layout.addLayout(theme_row)

        glow_row = QHBoxLayout()
        self.glow_val_lbl = QLabel("Glow Intensity: 0.80")
        glow_row.addWidget(self.glow_val_lbl)
        layout.addLayout(glow_row)

        self.glow_slider = QSlider(Qt.Orientation.Horizontal)
        self.glow_slider.setRange(0, 200)
        self.glow_slider.setValue(80)
        layout.addWidget(self.glow_slider)

        self.particles_chk = QCheckBox("Enable Particle Trails")
        self.particles_chk.setChecked(True)
        self.hologram_chk = QCheckBox("Enable Holographic Palm UI")
        self.hologram_chk.setChecked(True)

        layout.addWidget(self.particles_chk)
        layout.addWidget(self.hologram_chk)
        return group

    def _connect_signals(self) -> None:
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.glow_slider.valueChanged.connect(self._on_glow_changed)
        self.particles_chk.toggled.connect(self.engine.set_particles_enabled)
        self.hologram_chk.toggled.connect(self.engine.set_hologram_enabled)

    def start_tracking(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        cam_idx = self.camera_combo.currentIndex()
        _, width, height = self.RESOLUTIONS[self.resolution_combo.currentIndex()]

        self.worker = VisionWorker(
            camera_index=cam_idx,
            width=width,
            height=height,
            engine=self.engine,
        )
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.telemetry_updated.connect(self.update_telemetry)
        self.worker.status_changed.connect(self.status_pill.setText)
        self.worker.error_occurred.connect(self._on_worker_error)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.resolution_combo.setEnabled(False)
        self.status_pill.setText("STARTING...")
        self.worker.start()

    def stop_tracking(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.resolution_combo.setEnabled(True)
        self.status_pill.setText("STANDBY")
        self.fps_badge.setText("FPS: 00")
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("CAMERA STANDBY\n\nClick 'Start Tracking' to initialize webcam")

    def _on_theme_changed(self, index: int) -> None:
        self.engine.set_theme_index(index)

    def _on_glow_changed(self, value: int) -> None:
        intensity = value / 100.0
        self.glow_val_lbl.setText(f"Glow Intensity: {intensity:.2f}")
        self.engine.set_glow_intensity(intensity)

    def _on_worker_error(self, message: str) -> None:
        self.feedback_lbl.setText(f"Error: {message}")
        self.stop_tracking()
        self.status_pill.setText("CAMERA ERROR")

    def update_video_frame(self, image: QImage) -> None:
        self._last_pixmap = QPixmap.fromImage(image)
        scaled = self._last_pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    @staticmethod
    def _format_hand_text(name: str, info: HandTelemetry) -> str:
        if not info.present:
            return f"{name} Hand: Not detected"
        return f"{name} Hand: {info.gesture} ({info.finger_count} fingers)"

    def update_telemetry(self, snapshot: TelemetrySnapshot) -> None:
        self.fps_badge.setText(f"FPS: {int(snapshot.fps):02d}")
        self.detection_status_lbl.setText(
            f"Detection: {'ACTIVE (' + str(snapshot.hands_detected) + ' hand)' if snapshot.hands_detected else 'NO HANDS'}"
        )
        self.active_gesture_lbl.setText(f"Active Gesture: {snapshot.active_gesture}")
        self.zoom_lbl.setText(f"AR Core Zoom: {snapshot.zoom_scale:.2f}x")
        self.feedback_lbl.setText(f"Status: {snapshot.feedback_message}")

        self.left_info_lbl.setText(self._format_hand_text("Left", snapshot.left_hand))
        self.left_conf_bar.setValue(int(snapshot.left_hand.confidence * 100))

        self.right_info_lbl.setText(self._format_hand_text("Right", snapshot.right_hand))
        self.right_conf_bar.setValue(int(snapshot.right_hand.confidence * 100))

        # Keep theme dropdown synced if user switched theme via Peace Sign gesture
        for idx, preset in enumerate(DEFAULT_THEMES):
            if preset.name == snapshot.theme_name and self.theme_combo.currentIndex() != idx:
                self.theme_combo.blockSignals(True)
                self.theme_combo.setCurrentIndex(idx)
                self.theme_combo.blockSignals(False)
                break

    def closeEvent(self, event) -> None:
        self.stop_tracking()
        self.engine.close()
        super().closeEvent(event)

