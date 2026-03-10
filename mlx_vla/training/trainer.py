import os
import json
from pathlib import Path
from typing import Optional, List, Any, Dict
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from mlx_vla.core import VLATrainingArguments
from mlx_vla.utils.config import DEFAULT_CONFIG
from mlx_vla.data.dataset import VLADataset
from mlx_vla.data.collator import VLAModuleDataCollator
from mlx_vla.data.dataloader import VLADataloader
from mlx_vla.training.optimizers import create_optimizer, create_scheduler
from mlx_vla.training.callbacks import Callback, CheckpointCallback, LoggingCallback

class VLATrainer:
    def __init__(
        self,
        model: nn.Module,
        args: VLATrainingArguments,
        train_dataset: VLADataset,
        eval_dataset: Optional[VLADataset] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[VLAModuleDataCollator] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.data_collator = data_collator or VLAModuleDataCollator(
            image_size=DEFAULT_CONFIG["data"]["image_size"],
            action_normalization=DEFAULT_CONFIG["data"]["action_normalization"],
            tokenizer=tokenizer,
        )

        self.train_dataloader = VLADataloader(
            dataset=train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=False,
        )

        num_training_steps = len(self.train_dataloader) * args.num_train_epochs
        self.optimizer = create_optimizer(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            num_training_steps,
            warmup_ratio=args.warmup_ratio,
            scheduler_type=args.lr_scheduler_type,
        )

        self.global_step = 0
        self.epoch = 0
        self.learning_rate = args.learning_rate
        self.should_stop = False
        self.metrics: Dict = {}
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.accumulated_grads = None

        self.callbacks: List[Callback] = [
            CheckpointCallback(
                output_dir=args.output_dir,
                save_steps=args.save_steps,
                save_total_limit=args.save_total_limit,
            ),
            LoggingCallback(
                log_dir=args.logging_dir or f"{args.output_dir}/logs",
                log_steps=args.logging_steps,
            ),
        ]

        if args.resume_from_checkpoint:
            self._resume_from_checkpoint(args.resume_from_checkpoint)

        for callback in self.callbacks:
            callback.on_train_begin(self)

    def train(self):
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            self._train_epoch()

            if self.eval_dataset and self.args.eval_strategy == "epoch":
                eval_metrics = self.evaluate()
                self.metrics.update(eval_metrics)

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, self.metrics)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end(self)

    def _train_epoch(self):
        self.model.train()
        epoch_losses = []
        self.accumulated_grads = None

        for step, batch in enumerate(self.train_dataloader):
            for callback in self.callbacks:
                callback.on_step_begin(self, step)

            loss = self._train_step(batch)

            epoch_losses.append(float(loss))

            self.metrics["loss"] = np.mean(epoch_losses)
            self.metrics["learning_rate"] = self.learning_rate
            self.metrics["epoch"] = self.epoch

            for callback in self.callbacks:
                callback.on_step_end(self, step, float(loss))

            if self.eval_dataset and self.args.eval_strategy == "steps":
                if step > 0 and step % self.args.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.metrics.update(eval_metrics)

            self.global_step += 1

    def _train_step(self, batch: Dict):
        def loss_fn(model, batch):
            return self._compute_loss(model, batch)

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn(self.model, batch)

        if self.gradient_accumulation_steps > 1:
            if self.accumulated_grads is None:
                self.accumulated_grads = grads
            else:
                self.accumulated_grads = mlx.utils.tree_map(
                    lambda a, b: a + b, self.accumulated_grads, grads
                )

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                scale = 1.0 / self.gradient_accumulation_steps
                grads = mlx.utils.tree_map(lambda g: g * scale, self.accumulated_grads)
                self.accumulated_grads = None
            else:
                return loss

        if self.args.max_grad_norm > 0:
            grads, grad_norm = optim.clip_grad_norm(
                grads,
                self.args.max_grad_norm,
            )
            mx.eval(grad_norm)
            self.metrics["grad_norm"] = float(grad_norm)

        if self.scheduler:
            self.learning_rate = float(self.args.learning_rate * self.scheduler(self.global_step))

        self.optimizer.update(self.model, grads)

        return loss

    def _compute_loss(self, model: nn.Module, batch: Dict) -> mx.array:
        pixel_values = batch.get("pixel_values")
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        actions = batch.get("action")

        if pixel_values is None or len(pixel_values) == 0:
            return mx.array(0.0)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if model.action_type == "discrete":
            action_tokens = model.action_head.action_to_tokens(actions)
            logits = outputs["logits"]
            B, L, A, C = logits.shape
            last_logits = logits[:, -1, :, :]
            losses = []
            for a in range(A):
                dim_loss = nn.losses.cross_entropy(
                    last_logits[:, a, :],
                    action_tokens[:, a],
                )
                losses.append(dim_loss)
            loss = mx.mean(mx.stack(losses))
        elif model.action_type == "diffusion":
            predicted_actions = outputs["action"]
            if predicted_actions.ndim == 3 and actions.ndim == 2:
                pred_first = predicted_actions[:, 0, :]
                loss = mx.mean((pred_first - actions) ** 2)
            else:
                loss = mx.mean((predicted_actions - actions) ** 2)
        else:
            pred_actions = outputs["action"] if "action" in outputs else outputs["logits"]
            if pred_actions.ndim == 3 and actions.ndim == 2:
                pred_actions = pred_actions[:, 0, :]
            loss = mx.mean((pred_actions - actions) ** 2)

        return loss

    def _resume_from_checkpoint(self, checkpoint_path: str):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            self.model = self.model.load(str(checkpoint_path))
            state_file = checkpoint_path / "trainer_state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.global_step = state.get("global_step", 0)
                    self.epoch = state.get("epoch", 0)

    def evaluate(self) -> Dict:
        if self.eval_dataset is None:
            return {}

        eval_dataloader = VLADataloader(
            dataset=self.eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=False,
        )

        self.model.eval()
        eval_losses = []

        for batch in eval_dataloader:
            loss = self._compute_loss(self.model, batch)
            eval_losses.append(float(loss))

        self.model.train()

        return {"eval_loss": np.mean(eval_losses)}