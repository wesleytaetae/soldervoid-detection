from PySide6.QtCore import QThread, Signal


class PreparePreviewWorker(QThread):
    preview_ready = Signal(bytes, bytes)   # original_png, enhanced_png
    error = Signal(str)

    def __init__(
        self,
        image_path: str,
        blur_kernel: int,
        clahe_clip: float,
        clahe_tile_w: int,
        clahe_tile_h: int,
    ):
        super().__init__()
        self._image_path = image_path
        self._blur_kernel = blur_kernel
        self._clahe_clip = clahe_clip
        self._clahe_tile_w = clahe_tile_w
        self._clahe_tile_h = clahe_tile_h

    def run(self):
        try:
            from backend.prepare_ops import preview_single_image
            orig, enhanced = preview_single_image(
                self._image_path,
                blur_kernel=self._blur_kernel,
                clahe_clip=self._clahe_clip,
                clahe_tile_w=self._clahe_tile_w,
                clahe_tile_h=self._clahe_tile_h,
            )
            self.preview_ready.emit(orig, enhanced)
        except Exception as e:
            self.error.emit(str(e))
