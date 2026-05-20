from PySide6.QtCore import QThread, Signal


class ResizeWorker(QThread):
    file_processed = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_size: tuple[int, int] = (1024, 1024),
    ):
        super().__init__()
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._target_size = target_size

    def run(self):
        try:
            from backend.resize_ops import process_image_directory
            process_image_directory(
                self._input_dir,
                self._output_dir,
                target_size=self._target_size,
                on_file_done=lambda f: self.file_processed.emit(f),
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
