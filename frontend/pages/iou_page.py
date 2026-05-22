from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class IoUPage(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(20)

        left = QWidget()
        left.setMaximumWidth(460)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        title = QLabel("Step 7 - IoU Evaluation")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        left_layout.addWidget(title)

        desc = QLabel(
            "Compare a model prediction against a ground-truth mask to measure Intersection over Union."
        )
        desc.setWordWrap(True)
        left_layout.addWidget(desc)

        files_group = QGroupBox("Inputs")
        form = QFormLayout(files_group)

        self.model_edit = QLineEdit("best_unet_model.pth")
        form.addRow("Weights (.pth):", self._file_row(self.model_edit, self._browse_model))

        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText("Select X-ray image...")
        form.addRow("X-ray image:", self._file_row(self.image_edit, self._browse_image))

        self.mask_edit = QLineEdit()
        self.mask_edit.setPlaceholderText("Select compiled mask PNG...")
        form.addRow("Ground truth mask:", self._file_row(self.mask_edit, self._browse_mask))

        left_layout.addWidget(files_group)

        self.run_btn = QPushButton("Calculate IoU")
        self.run_btn.setFixedHeight(36)
        self.run_btn.clicked.connect(self._on_run)
        left_layout.addWidget(self.run_btn)

        left_layout.addStretch()
        layout.addWidget(left)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.summary_label = QLabel("-")
        self.summary_label.setFont(QFont("", 14, QFont.Weight.Bold))
        self.summary_label.setWordWrap(True)
        results_layout.addWidget(self.summary_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier New", 10))
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group, stretch=1)

    def _file_row(self, edit: QLineEdit, callback):
        row = QHBoxLayout()
        button = QPushButton("Browse...")
        button.setFixedWidth(80)
        button.clicked.connect(callback)
        row.addWidget(edit)
        row.addWidget(button)
        return row

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

    def _browse_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Ground-Truth Mask", "", "Mask images (*.png)"
        )
        if path:
            self.mask_edit.setText(path)

    def _on_run(self):
        from frontend.workers.iou_worker import IoUWorker

        model_path = self.model_edit.text().strip()
        image_path = self.image_edit.text().strip()
        mask_path = self.mask_edit.text().strip()
        if not model_path or not image_path or not mask_path:
            QMessageBox.warning(self, "Missing Input", "Please select model, image, and ground-truth mask.")
            return

        self.run_btn.setEnabled(False)
        self.summary_label.setText("Running...")
        self.results_text.setPlainText("")

        self._worker = IoUWorker(model_path, image_path, mask_path)
        self._worker.result_ready.connect(self._on_result)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_result(self, metrics: dict):
        mean_iou = metrics.get("mean_iou")
        if mean_iou is None:
            self.summary_label.setText("Mean IoU: N/A")
        else:
            self.summary_label.setText(f"Mean IoU: {mean_iou:.4f}")

        lines = []
        for row in metrics.get("per_class", []):
            iou = row["iou"]
            iou_text = "N/A" if iou is None else f"{iou:.4f}"
            lines.append(
                f"{row['class_name']:<12} IoU={iou_text:<8} "
                f"Intersection={row['intersection']:<8} Union={row['union']}"
            )

        self.results_text.setPlainText("\n".join(lines))

    def _on_done(self):
        self.run_btn.setEnabled(True)
        self._worker = None

    def _on_error(self, message: str):
        self.run_btn.setEnabled(True)
        self.summary_label.setText("Error!")
        self.results_text.setPlainText("")
        QMessageBox.critical(self, "IoU Error", message)
