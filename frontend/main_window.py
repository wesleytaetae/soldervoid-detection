from PySide6.QtWidgets import QHBoxLayout, QListWidget, QMainWindow, QStackedWidget, QWidget

from frontend.pages.home_page import HomePage
from frontend.pages.inference_page import InferencePage
from frontend.pages.iou_page import IoUPage
from frontend.pages.labelme_page import LabelMePage
from frontend.pages.mask_page import MaskPage
from frontend.pages.prepare_page import PreparePage
from frontend.pages.resize_page import ResizePage
from frontend.pages.train_page import TrainPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solder Inspection Pipeline")

        self.nav = QListWidget()
        self.nav.setFixedWidth(185)
        self.nav.addItems(
            [
                "  Home",
                "  1 · Resize",
                "  2 · Prepare",
                "  3 · Annotate",
                "  4 · Compile Masks",
                "  5 · Train",
                "  6 · Inference",
                "  7 · IoU",
            ]
        )
        self.nav.setStyleSheet(
            """
            QListWidget {
                background: #252525;
                border: none;
                border-right: 1px solid #3a3a3a;
                font-size: 13px;
                padding: 6px 0;
            }
            QListWidget::item {
                padding: 11px 14px;
                color: #bbb;
            }
            QListWidget::item:selected {
                background: #2a82da;
                color: #fff;
            }
            QListWidget::item:hover:!selected {
                background: #333;
            }
        """
        )

        self.stack = QStackedWidget()
        pages = [
            HomePage(navigate=self.navigate_to),
            ResizePage(navigate=self.navigate_to),
            PreparePage(navigate=self.navigate_to),
            LabelMePage(navigate=self.navigate_to),
            MaskPage(navigate=self.navigate_to),
            TrainPage(navigate=self.navigate_to),
            InferencePage(),
            IoUPage(),
        ]
        for page in pages:
            self.stack.addWidget(page)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.nav)
        layout.addWidget(self.stack)
        self.setCentralWidget(central)

    def navigate_to(self, index: int):
        self.nav.setCurrentRow(index)
