from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import json
from pathlib import Path

class Callback(ABC):
    def on_train_begin(self, trainer: Any):
        pass

    def on_train_end(self, trainer: Any):
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int):
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict):
        pass

    def on_step_begin(self, trainer: Any, step: int):
        pass

    def on_step_end(self, trainer: Any, step: int, loss: float):
        pass

    def on_log(self, trainer: Any, logs: Dict):
        pass

class CheckpointCallback(Callback):
    def __init__(
        self,
        output_dir: str = "./checkpoints",
        save_steps: int = 500,
        save_total_limit: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.checkpoints: List[str] = []

    def on_step_end(self, trainer: Any, step: int, loss: float):
        if step > 0 and step % self.save_steps == 0:
            self._save_checkpoint(trainer, step)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict):
        self._save_checkpoint(trainer, f"epoch-{epoch}")

    def _save_checkpoint(self, trainer: Any, name):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.output_dir / str(name)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        trainer.model.save(str(checkpoint_path))

        safe_metrics = {}
        for k, v in trainer.metrics.items():
            try:
                safe_metrics[k] = float(v)
            except (TypeError, ValueError):
                safe_metrics[k] = str(v)
        with open(checkpoint_path / "trainer_state.json", "w") as f:
            json.dump({
                "global_step": trainer.global_step,
                "epoch": trainer.epoch,
                "metrics": safe_metrics,
            }, f)

        self.checkpoints.append(str(checkpoint_path))

        if len(self.checkpoints) > self.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                import shutil
                shutil.rmtree(old_checkpoint)

class LoggingCallback(Callback):
    def __init__(
        self,
        log_dir: str = "./logs",
        log_steps: int = 10,
    ):
        self.log_dir = Path(log_dir)
        self.log_steps = log_steps
        self.logs: List[Dict] = []

    def on_step_end(self, trainer: Any, step: int, loss: float):
        if step > 0 and step % self.log_steps == 0:
            log_entry = {
                "step": step,
                "epoch": trainer.epoch,
                "loss": loss,
                "learning_rate": trainer.learning_rate,
            }
            self.logs.append(log_entry)

            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / "logs.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        metric: str = "loss",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_value: Optional[float] = None
        self.wait = 0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict):
        current_value = metrics.get(self.metric)

        if current_value is None:
            return

        if self.best_value is None:
            self.best_value = current_value
            return

        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True