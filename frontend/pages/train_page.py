from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TrainPage(QWidget):
    def __init__(self, navigate=None):
        super().__init__()
        self._navigate = navigate
        self._worker = None
        self.profile_groups = []
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 28)
        outer.setSpacing(12)

        title = QLabel("Step 4 - Train Model")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        outer.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(splitter)

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)

        config_group = QGroupBox("Training Configuration")
        grid = QGridLayout(config_group)
        grid.setSpacing(8)
        grid.setColumnMinimumWidth(0, 150)

        def add_dir_row(row, label, default):
            grid.addWidget(QLabel(label), row, 0)
            edit = QLineEdit(default)
            btn = QPushButton("Browse...")
            btn.setFixedWidth(80)
            btn.clicked.connect(lambda _, e=edit: self._browse_dir(e))
            layout = QHBoxLayout()
            layout.addWidget(edit)
            layout.addWidget(btn)
            grid.addLayout(layout, row, 1)
            return edit

        self.image_dir_edit = add_dir_row(0, "Image dir:", "3.output")
        self.mask_dir_edit = add_dir_row(1, "Mask dir:", "5.compiled_masks")

        grid.addWidget(QLabel("Save weights to:"), 2, 0)
        weights_row = QHBoxLayout()
        self.weights_edit = QLineEdit("best_unet_model.pth")
        browse_wts = QPushButton("Browse...")
        browse_wts.setFixedWidth(80)
        browse_wts.clicked.connect(self._browse_weights)
        weights_row.addWidget(self.weights_edit)
        weights_row.addWidget(browse_wts)
        grid.addLayout(weights_row, 2, 1)

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

        preprocess_group = QGroupBox("Optional Raw-Image Preprocessing")
        preprocess_layout = QVBoxLayout(preprocess_group)
        preprocess_layout.setSpacing(10)

        self.auto_preprocess_check = QCheckBox(
            "Expand training with preprocessing presets"
        )
        self.auto_preprocess_check.toggled.connect(self._update_preprocess_ui)
        preprocess_layout.addWidget(self.auto_preprocess_check)

        hint = QLabel(
            "Enable this when Image dir already contains correctly sized training images. "
            "Each enabled preset adds another training view of the same labeled sample."
        )
        hint.setWordWrap(True)
        preprocess_layout.addWidget(hint)

        self.include_original_check = QCheckBox(
            "Include the original image in addition to enabled presets"
        )
        self.include_original_check.setChecked(True)
        preprocess_layout.addWidget(self.include_original_check)

        presets_box = QGroupBox("Presets")
        presets_grid = QGridLayout(presets_box)
        presets_grid.setHorizontalSpacing(12)
        presets_grid.setVerticalSpacing(6)

        headers = ["Use", "Preset", "Blur", "Clip", "Tile W", "Tile H"]
        for col, text in enumerate(headers):
            header = QLabel(text)
            header.setFont(QFont("", 9, QFont.Weight.Bold))
            presets_grid.addWidget(header, 0, col)

        defaults_list = [
            {"enabled": True, "blur_kernel": 3, "clahe_clip": 12.0, "clahe_tile_w": 32, "clahe_tile_h": 32},
            {"enabled": False, "blur_kernel": 5, "clahe_clip": 8.0, "clahe_tile_w": 24, "clahe_tile_h": 24},
            {"enabled": False, "blur_kernel": 3, "clahe_clip": 16.0, "clahe_tile_w": 48, "clahe_tile_h": 48},
        ]
        for idx, defaults in enumerate(defaults_list, start=1):
            self._add_profile_row(presets_grid, idx, defaults)

        preprocess_layout.addWidget(presets_box)

        config_layout.addWidget(preprocess_group)
        splitter.addWidget(config_widget)

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
        self._update_preprocess_ui(self.auto_preprocess_check.isChecked())

    def _add_profile_row(self, grid: QGridLayout, index: int, defaults: dict):
        enabled = QCheckBox()
        enabled.setChecked(defaults["enabled"])
        grid.addWidget(enabled, index, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        name_label = QLabel(f"Preset {index}")
        grid.addWidget(name_label, index, 1)

        blur_spin = QSpinBox()
        blur_spin.setRange(1, 15)
        blur_spin.setSingleStep(2)
        blur_spin.setValue(defaults["blur_kernel"])
        blur_spin.setMaximumWidth(70)
        grid.addWidget(blur_spin, index, 2)

        clip_spin = QDoubleSpinBox()
        clip_spin.setRange(0.5, 80.0)
        clip_spin.setDecimals(1)
        clip_spin.setSingleStep(0.5)
        clip_spin.setValue(defaults["clahe_clip"])
        clip_spin.setMaximumWidth(80)
        grid.addWidget(clip_spin, index, 3)

        tile_w_spin = QSpinBox()
        tile_w_spin.setRange(2, 256)
        tile_w_spin.setSingleStep(8)
        tile_w_spin.setValue(defaults["clahe_tile_w"])
        tile_w_spin.setMaximumWidth(80)
        grid.addWidget(tile_w_spin, index, 4)

        tile_h_spin = QSpinBox()
        tile_h_spin.setRange(2, 256)
        tile_h_spin.setSingleStep(8)
        tile_h_spin.setValue(defaults["clahe_tile_h"])
        tile_h_spin.setMaximumWidth(80)
        grid.addWidget(tile_h_spin, index, 5)

        self.profile_groups.append(
            {
                "name_label": name_label,
                "enabled": enabled,
                "blur_spin": blur_spin,
                "clip_spin": clip_spin,
                "tile_w_spin": tile_w_spin,
                "tile_h_spin": tile_h_spin,
            }
        )

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

    def _update_preprocess_ui(self, enabled: bool):
        self.include_original_check.setEnabled(enabled)
        for profile in self.profile_groups:
            profile["name_label"].setEnabled(enabled)
            profile["enabled"].setEnabled(enabled)
            profile["blur_spin"].setEnabled(enabled)
            profile["clip_spin"].setEnabled(enabled)
            profile["tile_w_spin"].setEnabled(enabled)
            profile["tile_h_spin"].setEnabled(enabled)

    def _build_config(self) -> dict:
        preprocess_profiles = []
        for idx, profile in enumerate(self.profile_groups, start=1):
            preprocess_profiles.append(
                {
                    "name": f"Preset {idx}",
                    "enabled": profile["enabled"].isChecked(),
                    "blur_kernel": profile["blur_spin"].value(),
                    "clahe_clip": profile["clip_spin"].value(),
                    "clahe_tile_w": profile["tile_w_spin"].value(),
                    "clahe_tile_h": profile["tile_h_spin"].value(),
                }
            )

        return {
            "image_dir": self.image_dir_edit.text(),
            "mask_dir": self.mask_dir_edit.text(),
            "weights_save_path": self.weights_edit.text(),
            "batch_size": self.batch_spin.value(),
            "epochs": self.epochs_spin.value(),
            "patience": self.patience_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "num_classes": 3,
            "val_split": self.val_spin.value(),
            "auto_preprocess": self.auto_preprocess_check.isChecked(),
            "include_original_variant": self.include_original_check.isChecked(),
            "preprocess_profiles": preprocess_profiles,
        }

    def _append(self, html: str):
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html + "<br>")
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def _on_start(self):
        from frontend.workers.train_worker import TrainWorker

        config = self._build_config()
        if config["auto_preprocess"]:
            has_enabled_profile = any(profile["enabled"] for profile in config["preprocess_profiles"])
            if not has_enabled_profile and not config["include_original_variant"]:
                QMessageBox.warning(
                    self,
                    "No Training Variants",
                    "Enable at least one preprocessing preset, or include the original image.",
                )
                return

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
            f'<span style="color:#FFD700;">  - No improvement for {count} epoch(s)</span>'
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
