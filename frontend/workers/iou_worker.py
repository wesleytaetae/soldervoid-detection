from PySide6.QtCore import QThread, Signal


class IoUWorker(QThread):
    result_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, model_path: str, image_path: str, mask_path: str):
        super().__init__()
        self._model_path = model_path
        self._image_path = image_path
        self._mask_path = mask_path

    def run(self):
        try:
            import torch
            from backend.inference_ops import load_production_model, inspect_xray
            from backend.metrics_ops import compute_iou_metrics, load_mask

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_production_model(self._model_path, device)
            _, predicted_mask, _, _ = inspect_xray(model, self._image_path, device, show_solder_outline=False)
            true_mask = load_mask(self._mask_path)
            metrics = compute_iou_metrics(predicted_mask, true_mask)
            self.result_ready.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))
