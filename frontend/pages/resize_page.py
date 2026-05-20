from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QProgressBar, QFileDialog, QMessageBox, QSpinBox,
)


class ResizePage(QWidget):
    def __init__(self, navigate=None):
        super().__init__()
        self._navigate = navigate
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        title = QLabel("Step 1 — Resize Images")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Automatically center-crops each image to a square "
            "(removes equal margins from the longer axis), then resizes to the target dimensions."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # ── Directories ──────────────────────────────────────────────────
        dir_group = QGroupBox("Directories")
        form = QFormLayout(dir_group)
        form.setSpacing(10)

        self.input_edit = QLineEdit("1.input")
        browse_in = QPushButton("Browse…")
        browse_in.setFixedWidth(80)
        browse_in.clicked.connect(lambda: self._browse_dir(self.input_edit, "Select Input Folder"))
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit)
        in_row.addWidget(browse_in)
        form.addRow("Input folder:", in_row)

        self.output_edit = QLineEdit("2.resize")
        browse_out = QPushButton("Browse…")
        browse_out.setFixedWidth(80)
        browse_out.clicked.connect(lambda: self._browse_dir(self.output_edit, "Select Output Folder"))
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse_out)
        form.addRow("Output folder:", out_row)

        layout.addWidget(dir_group)

        # ── Settings ─────────────────────────────────────────────────────
        settings_group = QGroupBox("Settings")
        grid = QGridLayout(settings_group)
        grid.setSpacing(8)
        grid.setColumnMinimumWidth(0, 110)

        grid.addWidget(QLabel("Output width:"), 0, 0)
        self.out_width = QSpinBox()
        self.out_width.setRange(64, 8192)
        self.out_width.setValue(1024)
        self.out_width.setSingleStep(64)
        grid.addWidget(self.out_width, 0, 1)

        grid.addWidget(QLabel("Output height:"), 1, 0)
        self.out_height = QSpinBox()
        self.out_height.setRange(64, 8192)
        self.out_height.setValue(1024)
        self.out_height.setSingleStep(64)
        grid.addWidget(self.out_height, 1, 1)

        layout.addWidget(settings_group)

        # ── Status + progress ────────────────────────────────────────────
        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ── Action buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Resize")
        self.run_btn.setFixedHeight(36)
        self.run_btn.clicked.connect(self._on_run)

        self.next_btn = QPushButton("Next: Prepare →")
        self.next_btn.setFixedHeight(36)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self._navigate(2) if self._navigate else None)

        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.next_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

    def _browse_dir(self, edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title, edit.text())
        if path:
            edit.setText(path)

    def _on_run(self):
        from frontend.workers.resize_worker import ResizeWorker
        self.run_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Processing…")

        self._worker = ResizeWorker(
            self.input_edit.text(),
            self.output_edit.text(),
            target_size=(self.out_width.value(), self.out_height.value()),
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
        QMessageBox.critical(self, "Resize Error", message)
