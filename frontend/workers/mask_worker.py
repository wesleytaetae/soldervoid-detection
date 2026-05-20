from PySide6.QtCore import QThread, Signal


class MaskWorker(QThread):
    file_processed = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, json_dir: str, output_dir: str):
        super().__init__()
        self._json_dir = json_dir
        self._output_dir = output_dir

    def run(self):
        try:
            from backend.mask_ops import compile_dataset_masks
            compile_dataset_masks(
                self._json_dir,
                self._output_dir,
                on_file_done=lambda f: self.file_processed.emit(f),
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
