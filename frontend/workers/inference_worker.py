from PySide6.QtCore import QThread, Signal


class InferenceWorker(QThread):
    result_ready = Signal(float, bytes)
    finished     = Signal()
    error        = Signal(str)

    def __init__(self, model_path: str, image_path: str):
        super().__init__()
        self._model_path = model_path
        self._image_path = image_path

    def run(self):
        try:
            import cv2
            import torch
            from backend.inference_ops import load_production_model, inspect_xray

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_production_model(self._model_path, device)
            void_ratio, _, visual_img = inspect_xray(model, self._image_path, device)

            ok, buf = cv2.imencode(".png", visual_img)
            if not ok:
                raise RuntimeError("Failed to encode result image.")

            self.result_ready.emit(void_ratio, bytes(buf))
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
