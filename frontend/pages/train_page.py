from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QProgressBar, QFileDialog, QMessageBox,
    QTextEdit, QSplitter, QSpinBox, QDoubleSpinBox,
)


class TrainPage(QWidget):
    def __init__(self, navigate=None):
        super().__init__()
        self._navigate = navigate
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 28)
        outer.setSpacing(12)

        title = QLabel("Step 4 — Train Model")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        outer.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(splitter)

        # ── Config section ──────────────────────────────────────────────
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)

        config_group = QGroupBox("Training Configuration")
        grid = QGridLayout(config_group)
        grid.setSpacing(8)
        grid.setColumnMinimumWidth(0, 130)

        def add_dir_row(row, label, default):
            grid.addWidget(QLabel(label), row, 0)
            edit = QLineEdit(default)
            btn = QPushButton("Browse…")
            btn.setFixedWidth(80)
            btn.clicked.connect(lambda _, e=edit: self._browse_dir(e))
            h = QHBoxLayout()
            h.addWidget(edit)
            h.addWidget(btn)
            grid.addLayout(h, row, 1)
            return edit

        self.image_dir_edit   = add_dir_row(0, "Image dir:", "3.output")
        self.mask_dir_edit    = add_dir_row(1, "Mask dir:", "5.compiled_masks")

        grid.addWidget(QLabel("Save weights to:"), 2, 0)
        weights_row = QHBoxLayout()
        self.weights_edit = QLineEdit("best_unet_model.pth")
        browse_wts = QPushButton("Browse…")
        browse_wts.setFixedWidth(80)
        browse_wts.clicked.connect(self._browse_weights)
        weights_row.addWidget(self.weights_edit)
        weights_row.addWidget(browse_wts)
        grid.addLayout(weights_row, 2, 1)

        # Numeric params
        grid.addWidget(QLabel("Batch size:"), 3, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        grid.addWidget(self.batch_spin, 3, 1)

        grid.addWidget(QLabel("Epochs:"), 4, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(500)
        grid.addWidget(self.epochs_spin, 4, 1)

        grid.addWidget(QLabel("Patience:"), 5, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(15)
        grid.addWidget(self.patience_spin, 5, 1)

        grid.addWidget(QLabel("Learning rate:"), 6, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setSingleStep(1e-5)
        grid.addWidget(self.lr_spin, 6, 1)

        grid.addWidget(QLabel("Val split:"), 7, 0)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setDecimals(2)
        self.val_spin.setRange(0.05, 0.5)
        self.val_spin.setValue(0.20)
        self.val_spin.setSingleStep(0.05)
        grid.addWidget(self.val_spin, 7, 1)

        config_layout.addWidget(config_group)
        splitter.addWidget(config_widget)

        # ── Log section ──────────────────────────────────────────────────
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_group = QGroupBox("Training Log")
        log_inner = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        log_inner.addWidget(self.log_text)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        log_inner.addWidget(self.train_progress)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setFixedHeight(36)
        self.start_btn.clicked.connect(self._on_start)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        log_inner.addLayout(btn_row)

        log_layout.addWidget(log_group)
        splitter.addWidget(log_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", edit.text())
        if path:
            edit.setText(path)

    def _browse_weights(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Weights As", self.weights_edit.text(), "PyTorch models (*.pth)"
        )
        if path:
            self.weights_edit.setText(path)

    def _build_config(self) -> dict:
        return {
            "image_dir":         self.image_dir_edit.text(),
            "mask_dir":          self.mask_dir_edit.text(),
            "weights_save_path": self.weights_edit.text(),
            "batch_size":        self.batch_spin.value(),
            "epochs":            self.epochs_spin.value(),
            "patience":          self.patience_spin.value(),
            "learning_rate":     self.lr_spin.value(),
            "num_classes":       3,
            "val_split":         self.val_spin.value(),
        }

    def _append(self, html: str):
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html + "<br>")
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_start(self):
        from frontend.workers.train_worker import TrainWorker
        config = self._build_config()

        self.log_text.clear()
        self.train_progress.setRange(0, config["epochs"])
        self.train_progress.setValue(0)
        self.train_progress.setVisible(True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._worker = TrainWorker(config)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.no_improve.connect(self._on_no_improve)
        self._worker.early_stop.connect(self._on_early_stop)
        self._worker.log_message.connect(self._on_log)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.requestInterruption()
            self.stop_btn.setEnabled(False)

    def _on_epoch(self, epoch: int, train_loss: float, val_loss: float):
        self.train_progress.setValue(epoch)
        self._append(
            f"Epoch <b>{epoch:03d}</b> &nbsp;|&nbsp; "
            f"Train: {train_loss:.4f} &nbsp;|&nbsp; Val: {val_loss:.4f}"
        )

    def _on_no_improve(self, count: int):
        self._append(
            f'<span style="color:#FFD700;">  → No improvement for {count} epoch(s)</span>'
        )

    def _on_early_stop(self, epoch: int):
        self._append(
            f'<span style="color:#FF6B6B;"><b>[EARLY STOP]</b> Triggered at epoch {epoch}</span>'
        )

    def _on_log(self, msg: str):
        self._append(f'<span style="color:#90EE90;">{msg}</span>')

    def _on_done(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append('<span style="color:#90EE90;"><b>Training complete.</b></span>')

    def _on_error(self, message: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append(f'<span style="color:#FF6B6B;"><b>Error:</b> {message}</span>')
        QMessageBox.critical(self, "Training Error", message)
