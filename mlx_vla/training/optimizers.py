import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Optional, List, Dict, Any

def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
    momentum: float = 0.9,
) -> optim.Optimizer:
    if optimizer_type == "adamw":
        return optim.AdamW(learning_rate=learning_rate, betas=betas, eps=eps)
    elif optimizer_type == "adam":
        return optim.Adam(learning_rate=learning_rate, betas=betas, eps=eps)
    elif optimizer_type == "sgd":
        return optim.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(learning_rate=learning_rate)
    elif optimizer_type == "lion":
        return optim.Lion(learning_rate=learning_rate)
    else:
        return optim.AdamW(learning_rate=learning_rate)

def create_scheduler(
    optimizer: optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "cosine",
) -> Optional[Any]:
    warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type == "cosine":
        def cosine_schedule(step: int):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_schedule
    elif scheduler_type == "linear":
        def linear_schedule(step: int):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, (num_training_steps - step) / max(1, num_training_steps - warmup_steps))
        return linear_schedule
    elif scheduler_type == "constant":
        return lambda step: 1.0 if step >= warmup_steps else step / max(1, warmup_steps)
    else:
        return None