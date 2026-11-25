"""Futuristic Dark Cyberpunk QSS Stylesheet for NEON HAND AI."""

CYBERPUNK_QSS = """
QMainWindow, QWidget {
    background-color: #090C12;
    color: #E2E8F0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QFrame#HeaderBar {
    background-color: #0E131F;
    border-bottom: 1px solid #1E293B;
}

QLabel#AppTitle {
    color: #00F5FF;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 2px;
}

QLabel#AppSubtitle {
    color: #7C8DB5;
    font-size: 11px;
    letter-spacing: 1px;
}

QLabel#StatusPill {
    background-color: #111927;
    border: 1px solid #00F5FF;
    border-radius: 10px;
    color: #00F5FF;
    padding: 4px 12px;
    font-weight: 600;
}

QLabel#FpsBadge {
    background-color: #111927;
    border: 1px solid #00FF80;
    border-radius: 10px;
    color: #00FF80;
    padding: 4px 12px;
    font-family: "Consolas", monospace;
    font-weight: 700;
}

QLabel#VideoSurface {
    background-color: #05070B;
    border: 1px solid #1E293B;
    border-radius: 10px;
    color: #64748B;
    font-size: 15px;
}

QGroupBox {
    background-color: #0E131F;
    border: 1px solid #1E293B;
    border-radius: 10px;
    margin-top: 12px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
    color: #00F5FF;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    background-color: #090C12;
}

QPushButton#StartBtn {
    background-color: #003B46;
    border: 1px solid #00F5FF;
    border-radius: 6px;
    color: #00F5FF;
    padding: 8px 14px;
    font-weight: 700;
}
QPushButton#StartBtn:hover {
    background-color: #005666;
}
QPushButton#StartBtn:disabled {
    background-color: #111726;
    border-color: #334155;
    color: #475569;
}

QPushButton#StopBtn {
    background-color: #3B0D26;
    border: 1px solid #FF2A85;
    border-radius: 6px;
    color: #FF2A85;
    padding: 8px 14px;
    font-weight: 700;
}
QPushButton#StopBtn:hover {
    background-color: #59143A;
}
QPushButton#StopBtn:disabled {
    background-color: #111726;
    border-color: #334155;
    color: #475569;
}

QComboBox {
    background-color: #111726;
    border: 1px solid #293548;
    border-radius: 6px;
    padding: 5px 10px;
    color: #E2E8F0;
}
QComboBox:hover {
    border-color: #00F5FF;
}
QComboBox QAbstractItemView {
    background-color: #111726;
    border: 1px solid #00F5FF;
    selection-background-color: #1E3A5F;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #1E293B;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #00F5FF;
    border: 2px solid #090C12;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2979FF, stop:1 #00F5FF);
    border-radius: 3px;
}

QCheckBox {
    spacing: 8px;
    color: #CBD5E1;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #334155;
    background-color: #111726;
}
QCheckBox::indicator:checked {
    background-color: #00F5FF;
    border-color: #00F5FF;
}

QProgressBar {
    background-color: #111726;
    border: 1px solid #1E293B;
    border-radius: 4px;
    text-align: center;
    height: 14px;
    font-size: 10px;
    color: #E2E8F0;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2979FF, stop:1 #B026FF);
    border-radius: 3px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}
"""

