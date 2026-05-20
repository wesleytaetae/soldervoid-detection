from PySide6.QtCore import QThread, Signal


class TrainWorker(QThread):
    epoch_done   = Signal(int, float, float)
    no_improve   = Signal(int)
    early_stop   = Signal(int)
    log_message  = Signal(str)
    finished     = Signal()
    error        = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self._config = config

    def run(self):
        try:
            from backend.train_ops import train_model, TrainCallbacks
            callbacks = TrainCallbacks(
                on_epoch_end  = lambda e, tl, vl: self.epoch_done.emit(e, tl, vl),
                on_no_improve = lambda n: self.no_improve.emit(n),
                on_early_stop = lambda e: self.early_stop.emit(e),
                on_log        = lambda msg: self.log_message.emit(msg),
                should_stop   = self.isInterruptionRequested,
            )
            train_model(self._config, callbacks)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
