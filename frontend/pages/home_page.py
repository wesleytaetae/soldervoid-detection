from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class HomePage(QWidget):
    def __init__(self, navigate=None):
        super().__init__()
        self._navigate = navigate
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(16)

        title = QLabel("Solder Inspection Pipeline")
        title.setFont(QFont("", 22, QFont.Weight.Bold))
        layout.addWidget(title)

        subtitle = QLabel("X-ray defect segmentation — end-to-end workflow")
        subtitle.setFont(QFont("", 11))
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        steps = [
            ("1 · Resize",        "Center-crop raw X-rays to 1200×1200 and resize to 1024×1024."),
            ("2 · Prepare",       "Apply CLAHE contrast enhancement so voids are visible in LabelMe."),
            ("3 · Compile Masks", "Convert LabelMe JSON annotations into integer segmentation PNGs."),
            ("4 · Train",         "Fine-tune a U-Net (ResNet-34 encoder) with mixed-precision on GPU."),
            ("5 · Inference",     "Load a trained model and inspect a single X-ray for void percentage."),
        ]
        for step, desc in steps:
            row = QLabel(f"<b>{step}</b> — {desc}")
            row.setWordWrap(True)
            row.setStyleSheet("padding: 6px 0;")
            layout.addWidget(row)

        layout.addSpacing(24)

        start_btn = QPushButton("Start →")
        start_btn.setFixedWidth(140)
        start_btn.setFixedHeight(38)
        start_btn.clicked.connect(lambda: self._navigate(1) if self._navigate else None)
        layout.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addStretch()
