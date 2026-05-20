from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QProgressBar, QFileDialog, QMessageBox,
)


class MaskPage(QWidget):
    def __init__(self, navigate=None):
        super().__init__()
        self._navigate = navigate
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        title = QLabel("Step 3 — Compile Masks")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Rasterises LabelMe JSON polygon annotations into integer PNG masks "
            "(0 = Background, 1 = Solder, 2 = Void).\n"
            "Place your exported LabelMe JSON files in 4.json_labels before running."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        dir_group = QGroupBox("Directories")
        form = QFormLayout(dir_group)
        form.setSpacing(10)

        self.json_edit = QLineEdit("4.json_labels")
        browse_json = QPushButton("Browse…")
        browse_json.setFixedWidth(80)
        browse_json.clicked.connect(lambda: self._browse_dir(self.json_edit, "Select JSON Labels Folder"))
        json_row = QHBoxLayout()
        json_row.addWidget(self.json_edit)
        json_row.addWidget(browse_json)
        form.addRow("JSON labels folder:", json_row)

        self.output_edit = QLineEdit("5.compiled_masks")
        browse_out = QPushButton("Browse…")
        browse_out.setFixedWidth(80)
        browse_out.clicked.connect(lambda: self._browse_dir(self.output_edit, "Select Mask Output Folder"))
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse_out)
        form.addRow("Mask output folder:", out_row)

        layout.addWidget(dir_group)

        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Compile Masks")
        self.run_btn.setFixedHeight(36)
        self.run_btn.clicked.connect(self._on_run)

        self.next_btn = QPushButton("Next: Train →")
        self.next_btn.setFixedHeight(36)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self._navigate(5) if self._navigate else None)

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
        from frontend.workers.mask_worker import MaskWorker
        self.run_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Processing…")

        self._worker = MaskWorker(self.json_edit.text(), self.output_edit.text())
        self._worker.file_processed.connect(lambda f: self.status_label.setText(f"Compiled: {f}"))
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
        QMessageBox.critical(self, "Mask Compile Error", message)
