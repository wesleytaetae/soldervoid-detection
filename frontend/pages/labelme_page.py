import shutil

from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox,
)


class LabelMePage(QWidget):
    def __init__(self, navigate=None):
        super().__init__()
        self._navigate = navigate
        self._process: QProcess | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        title = QLabel("Step 3 — Annotate with LabelMe")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Open LabelMe to draw polygon annotations for <b>Solder</b> and <b>Solder Void</b> regions. "
            "JSON files are saved automatically to the output folder you specify."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(4)

        # ── Folder pickers ────────────────────────────────────────────────
        folder_group = QGroupBox("Folders")
        form = QFormLayout(folder_group)
        form.setSpacing(10)

        self.image_edit = QLineEdit("3.output")
        browse_img = QPushButton("Browse…")
        browse_img.setFixedWidth(80)
        browse_img.clicked.connect(
            lambda: self._browse_dir(self.image_edit, "Select Image Folder to Annotate")
        )
        img_row = QHBoxLayout()
        img_row.addWidget(self.image_edit)
        img_row.addWidget(browse_img)
        form.addRow("Image folder:", img_row)

        self.json_edit = QLineEdit("4.json_labels")
        browse_json = QPushButton("Browse…")
        browse_json.setFixedWidth(80)
        browse_json.clicked.connect(
            lambda: self._browse_dir(self.json_edit, "Select JSON Output Folder")
        )
        json_row = QHBoxLayout()
        json_row.addWidget(self.json_edit)
        json_row.addWidget(browse_json)
        form.addRow("JSON output folder:", json_row)

        layout.addWidget(folder_group)

        # ── Status ────────────────────────────────────────────────────────
        self.status_label = QLabel("Not started.")
        layout.addWidget(self.status_label)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.launch_btn = QPushButton("Launch LabelMe")
        self.launch_btn.setFixedHeight(36)
        self.launch_btn.clicked.connect(self._on_launch)

        self.next_btn = QPushButton("Next: Compile Masks →")
        self.next_btn.setFixedHeight(36)
        self.next_btn.clicked.connect(lambda: self._navigate(4) if self._navigate else None)

        btn_row.addWidget(self.launch_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.next_btn)
        layout.addLayout(btn_row)

        # ── Hint ──────────────────────────────────────────────────────────
        hint = QLabel(
            "<i>Hint: label solder areas as <b>Solder</b> and defects as <b>Solder Void</b>. "
            "Close LabelMe when done — JSON files will be ready for the next step.</i>"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; margin-top: 8px;")
        layout.addWidget(hint)

        layout.addStretch()

    def _browse_dir(self, edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title, edit.text())
        if path:
            edit.setText(path)

    def _on_launch(self):
        uv_path = shutil.which("uv")
        if not uv_path:
            QMessageBox.critical(
                self, "uv not found",
                "'uv' was not found in PATH.\n\n"
                "Make sure uv is installed and available, then try again."
            )
            return

        img_folder = self.image_edit.text().strip()
        json_folder = self.json_edit.text().strip()

        if not img_folder:
            QMessageBox.warning(self, "No folder", "Please select an image folder.")
            return

        self._process = QProcess(self)
        self._process.finished.connect(self._on_labelme_finished)

        args = ["tool", "run", "labelme", img_folder]
        if json_folder:
            args += ["--output", json_folder]

        self._process.start(uv_path, args)

        if not self._process.waitForStarted(3000):
            self.status_label.setText("Failed to start LabelMe.")
            QMessageBox.critical(self, "Launch Failed",
                                 "Could not start LabelMe. Is it installed?\n"
                                 "Run:  uv tool install labelme")
            return

        self.launch_btn.setEnabled(False)
        self.status_label.setText("LabelMe is running… close it when you are done annotating.")

    def _on_labelme_finished(self):
        self.launch_btn.setEnabled(True)
        self.status_label.setText("LabelMe closed. JSON files are ready in the output folder.")
