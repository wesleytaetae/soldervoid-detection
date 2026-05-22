from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QCheckBox,
)


class InferencePage(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._last_visual_bytes: bytes | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(20)

        # ── Left panel: controls ─────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        title = QLabel("Step 5 — Inference")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        left_layout.addWidget(title)

        desc = QLabel("Load a trained model and run it on a single X-ray to measure void percentage.")
        desc.setWordWrap(True)
        left_layout.addWidget(desc)

        # Model picker
        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)
        model_row = QHBoxLayout()
        self.model_edit = QLineEdit("best_unet_model.pth")
        browse_model = QPushButton("Browse…")
        browse_model.setFixedWidth(80)
        browse_model.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_edit)
        model_row.addWidget(browse_model)
        model_form.addRow("Weights (.pth):", model_row)
        left_layout.addWidget(model_group)

        # Image picker
        image_group = QGroupBox("Image")
        image_form = QFormLayout(image_group)
        image_row = QHBoxLayout()
        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText("Select X-ray image…")
        browse_image = QPushButton("Browse…")
        browse_image.setFixedWidth(80)
        browse_image.clicked.connect(self._browse_image)
        image_row.addWidget(self.image_edit)
        image_row.addWidget(browse_image)
        image_form.addRow("X-ray image:", image_row)
        left_layout.addWidget(image_group)

        display_group = QGroupBox("Render Options")
        display_form = QFormLayout(display_group)
        self.solder_outline_check = QCheckBox("Show solder outline overlay")
        self.solder_outline_check.setChecked(True)
        display_form.addRow("", self.solder_outline_check)
        left_layout.addWidget(display_group)

        left_layout.addSpacing(8)

        # Result
        self.result_label = QLabel("—")
        result_font = QFont("", 14)
        result_font.setBold(True)
        self.result_label.setFont(result_font)
        self.result_label.setWordWrap(True)
        left_layout.addWidget(self.result_label)

        # Buttons
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.setFixedHeight(36)
        self.run_btn.clicked.connect(self._on_run)
        left_layout.addWidget(self.run_btn)

        self.save_btn = QPushButton("Save Report…")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        left_layout.addWidget(self.save_btn)

        left_layout.addStretch()
        layout.addWidget(left)

        # ── Right panel: image display ───────────────────────────────────
        self.image_display = QLabel("No result yet")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setMinimumSize(512, 512)
        self.image_display.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #444; color: #555;"
        )
        layout.addWidget(self.image_display, stretch=1)

    # ── File dialogs ──────────────────────────────────────────────────────

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weights", self.model_edit.text(), "PyTorch models (*.pth)"
        )
        if path:
            self.model_edit.setText(path)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select X-ray Image", "", "Images (*.jpg *.jpeg *.png *.webp)"
        )
        if path:
            self.image_edit.setText(path)

    # ── Run ───────────────────────────────────────────────────────────────

    def _on_run(self):
        from frontend.workers.inference_worker import InferenceWorker

        model_path = self.model_edit.text().strip()
        image_path = self.image_edit.text().strip()
        if not model_path or not image_path:
            QMessageBox.warning(self, "Missing Input", "Please select both a model file and an image.")
            return

        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.result_label.setText("Running…")
        self.image_display.setText("Running…")

        self._worker = InferenceWorker(
            model_path,
            image_path,
            show_solder_outline=self.solder_outline_check.isChecked(),
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.completed.connect(self._on_completed)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_result(self, void_ratio: float, png_bytes: bytes):
        self._last_visual_bytes = png_bytes
        self.result_label.setText(f"Solder Void Area: {void_ratio:.2f}%")

        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes)
        w = self.image_display.width()
        h = self.image_display.height()
        self.image_display.setPixmap(
            pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )

    def _on_done(self):
        self.run_btn.setEnabled(True)
        self._worker = None

    def _on_completed(self):
        self.save_btn.setEnabled(True)

    def _on_error(self, message: str):
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.result_label.setText("Error!")
        self.image_display.setText("Error")
        QMessageBox.critical(self, "Inference Error", message)

    def _on_save(self):
        if not self._last_visual_bytes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "inspection_report.png", "PNG images (*.png)"
        )
        if path:
            with open(path, "wb") as f:
                f.write(self._last_visual_bytes)
