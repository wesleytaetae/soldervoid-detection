from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QProgressBar, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
)


class PreparePage(QWidget):
    def __init__(self, navigate=None):
        super().__init__()
        self._navigate = navigate
        self._worker = None
        self._preview_worker = None
        self._setup_ui()

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(20)

        # ══ LEFT PANEL: controls ══════════════════════════════════════════
        left = QWidget()
        left.setFixedWidth(400)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        title = QLabel("Step 2 — Prepare Images")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        left_layout.addWidget(title)

        desc = QLabel("Applies median blur + CLAHE contrast enhancement so voids stand out in LabelMe.")
        desc.setWordWrap(True)
        left_layout.addWidget(desc)

        # Directories
        dir_group = QGroupBox("Directories")
        form = QFormLayout(dir_group)
        form.setSpacing(10)

        self.input_edit = QLineEdit("2.resize")
        browse_in = QPushButton("Browse…")
        browse_in.setFixedWidth(80)
        browse_in.clicked.connect(lambda: self._browse_dir(self.input_edit, "Select Input Folder"))
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit)
        in_row.addWidget(browse_in)
        form.addRow("Input folder:", in_row)

        self.output_edit = QLineEdit("3.output")
        browse_out = QPushButton("Browse…")
        browse_out.setFixedWidth(80)
        browse_out.clicked.connect(lambda: self._browse_dir(self.output_edit, "Select Output Folder"))
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse_out)
        form.addRow("Output folder:", out_row)

        left_layout.addWidget(dir_group)

        # Settings
        settings_group = QGroupBox("Settings")
        grid = QGridLayout(settings_group)
        grid.setSpacing(8)
        grid.setColumnMinimumWidth(0, 160)

        grid.addWidget(QLabel("Blur kernel (odd):"), 0, 0)
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(1, 15)
        self.blur_spin.setValue(3)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setToolTip("Must be odd. Even values are rounded up automatically.")
        grid.addWidget(self.blur_spin, 0, 1)

        grid.addWidget(QLabel("CLAHE clip limit:"), 1, 0)
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setRange(0.5, 80.0)
        self.clip_spin.setValue(12.0)
        self.clip_spin.setDecimals(1)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.setToolTip("Higher = more aggressive contrast boost.")
        grid.addWidget(self.clip_spin, 1, 1)

        grid.addWidget(QLabel("CLAHE tile width:"), 2, 0)
        self.tile_w_spin = QSpinBox()
        self.tile_w_spin.setRange(2, 256)
        self.tile_w_spin.setValue(32)
        self.tile_w_spin.setSingleStep(8)
        grid.addWidget(self.tile_w_spin, 2, 1)

        grid.addWidget(QLabel("CLAHE tile height:"), 3, 0)
        self.tile_h_spin = QSpinBox()
        self.tile_h_spin.setRange(2, 256)
        self.tile_h_spin.setValue(32)
        self.tile_h_spin.setSingleStep(8)
        grid.addWidget(self.tile_h_spin, 3, 1)

        left_layout.addWidget(settings_group)

        # Status + progress
        self.status_label = QLabel("Ready.")
        left_layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Prepare")
        self.run_btn.setFixedHeight(36)
        self.run_btn.clicked.connect(self._on_run)

        self.next_btn = QPushButton("Next: Annotate →")
        self.next_btn.setFixedHeight(36)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self._navigate(3) if self._navigate else None)

        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.next_btn)
        left_layout.addLayout(btn_row)

        left_layout.addStretch()
        root.addWidget(left)

        # ══ RIGHT PANEL: preview ══════════════════════════════════════════
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        preview_group = QGroupBox("Preview")
        preview_inner = QVBoxLayout(preview_group)
        preview_inner.setSpacing(10)

        # Sample image picker
        sample_row = QHBoxLayout()
        self.sample_edit = QLineEdit()
        self.sample_edit.setPlaceholderText("Pick a sample image to preview…")
        browse_sample = QPushButton("Browse…")
        browse_sample.setFixedWidth(80)
        browse_sample.clicked.connect(self._browse_sample)
        sample_row.addWidget(self.sample_edit)
        sample_row.addWidget(browse_sample)
        preview_inner.addLayout(sample_row)

        self.preview_btn = QPushButton("Update Preview")
        self.preview_btn.setFixedHeight(32)
        self.preview_btn.clicked.connect(self._on_preview)
        preview_inner.addWidget(self.preview_btn)

        # Side-by-side image labels
        images_row = QHBoxLayout()
        images_row.setSpacing(10)

        orig_col = QVBoxLayout()
        orig_col.addWidget(QLabel("Original", alignment=Qt.AlignmentFlag.AlignCenter))
        self.orig_label = QLabel("No preview")
        self.orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.orig_label.setMinimumSize(260, 260)
        self.orig_label.setStyleSheet("background:#1a1a1a; border:1px solid #444; color:#555;")
        orig_col.addWidget(self.orig_label)
        images_row.addLayout(orig_col)

        enh_col = QVBoxLayout()
        enh_col.addWidget(QLabel("Enhanced", alignment=Qt.AlignmentFlag.AlignCenter))
        self.enh_label = QLabel("No preview")
        self.enh_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.enh_label.setMinimumSize(260, 260)
        self.enh_label.setStyleSheet("background:#1a1a1a; border:1px solid #444; color:#555;")
        enh_col.addWidget(self.enh_label)
        images_row.addLayout(enh_col)

        preview_inner.addLayout(images_row)

        right_layout.addWidget(preview_group)
        right_layout.addStretch()
        root.addWidget(right, stretch=1)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _browse_dir(self, edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title, edit.text())
        if path:
            edit.setText(path)

    def _browse_sample(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Sample Image", self.input_edit.text(),
            "Images (*.jpg *.jpeg *.png)"
        )
        if path:
            self.sample_edit.setText(path)

    def _show_pixmap(self, label: QLabel, png_bytes: bytes):
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes)
        w, h = label.width(), label.height()
        label.setPixmap(
            pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )

    # ── Preview ───────────────────────────────────────────────────────────

    def _on_preview(self):
        path = self.sample_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No Image", "Browse to a sample image first.")
            return

        from frontend.workers.preview_worker import PreparePreviewWorker
        self.preview_btn.setEnabled(False)
        self.orig_label.setText("Loading…")
        self.enh_label.setText("Loading…")

        self._preview_worker = PreparePreviewWorker(
            path,
            blur_kernel=self.blur_spin.value(),
            clahe_clip=self.clip_spin.value(),
            clahe_tile_w=self.tile_w_spin.value(),
            clahe_tile_h=self.tile_h_spin.value(),
        )
        self._preview_worker.preview_ready.connect(self._on_preview_ready)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.finished.connect(lambda: self.preview_btn.setEnabled(True))
        self._preview_worker.finished.connect(self._preview_worker.deleteLater)
        self._preview_worker.start()

    def _on_preview_ready(self, orig: bytes, enhanced: bytes):
        self._show_pixmap(self.orig_label, orig)
        self._show_pixmap(self.enh_label, enhanced)

    def _on_preview_error(self, message: str):
        self.orig_label.setText("Error")
        self.enh_label.setText("Error")
        QMessageBox.critical(self, "Preview Error", message)

    # ── Batch run ─────────────────────────────────────────────────────────

    def _on_run(self):
        from frontend.workers.prepare_worker import PrepareWorker
        self.run_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Processing…")

        self._worker = PrepareWorker(
            self.input_edit.text(),
            self.output_edit.text(),
            blur_kernel=self.blur_spin.value(),
            clahe_clip=self.clip_spin.value(),
            clahe_tile_w=self.tile_w_spin.value(),
            clahe_tile_h=self.tile_h_spin.value(),
        )
        self._worker.file_processed.connect(lambda f: self.status_label.setText(f"Processed: {f}"))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_done(self):
        self.run_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.status_label.setText("Done.")

    def _on_error(self, message: str):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("Error — see details.")
        QMessageBox.critical(self, "Prepare Error", message)
