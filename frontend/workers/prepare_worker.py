from PySide6.QtCore import QThread, Signal


class PrepareWorker(QThread):
    file_processed = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        blur_kernel: int = 3,
        clahe_clip: float = 12.0,
        clahe_tile_w: int = 32,
        clahe_tile_h: int = 32,
    ):
        super().__init__()
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._blur_kernel = blur_kernel
        self._clahe_clip = clahe_clip
        self._clahe_tile_w = clahe_tile_w
        self._clahe_tile_h = clahe_tile_h

    def run(self):
        try:
            from backend.prepare_ops import prepare_images_for_labelme
            prepare_images_for_labelme(
                self._input_dir,
                self._output_dir,
                blur_kernel=self._blur_kernel,
                clahe_clip=self._clahe_clip,
                clahe_tile_w=self._clahe_tile_w,
                clahe_tile_h=self._clahe_tile_h,
                on_file_done=lambda f: self.file_processed.emit(f),
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
