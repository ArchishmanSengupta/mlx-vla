from mlx_vla.utils.config import (
    VLAConfigManager,
    ModelConfig,
    LoRAConfig,
    DataConfig,
    TrainingConfig,
    CheckpointingConfig,
    LoggingConfig,
    load_config,
    DEFAULT_CONFIG,
)

__version__ = "0.1.0"

__all__ = [
    "VLAConfigManager",
    "ModelConfig",
    "LoRAConfig",
    "DataConfig",
    "TrainingConfig",
    "CheckpointingConfig",
    "LoggingConfig",
    "load_config",
    "DEFAULT_CONFIG",
]

class VLATrainingArguments:
    def __init__(
        self,
        output_dir: str = None,
        num_train_epochs: int = None,
        per_device_train_batch_size: int = None,
        gradient_accumulation_steps: int = None,
        learning_rate: float = None,
        warmup_ratio: float = None,
        weight_decay: float = None,
        max_grad_norm: float = None,
        lr_scheduler_type: str = None,
        save_strategy: str = None,
        save_steps: int = None,
        save_total_limit: int = None,
        resume_from_checkpoint: str = None,
        logging_steps: int = None,
        logging_dir: str = None,
        eval_strategy: str = None,
        eval_steps: int = None,
        use_compile: bool = False,
        max_memory_mb: int = None,
        report_to: list = None,
    ):
        cfg = load_config()

        self.output_dir = output_dir or cfg.checkpointing.output_dir
        self.num_train_epochs = num_train_epochs or cfg.training.epochs
        self.per_device_train_batch_size = per_device_train_batch_size or cfg.data.batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps or cfg.training.gradient_accumulation_steps
        self.learning_rate = learning_rate or cfg.training.learning_rate
        self.warmup_ratio = warmup_ratio or cfg.training.warmup_ratio
        self.weight_decay = weight_decay or cfg.training.weight_decay
        self.max_grad_norm = max_grad_norm or cfg.training.max_grad_norm
        self.lr_scheduler_type = lr_scheduler_type or cfg.training.lr_scheduler_type
        self.save_strategy = save_strategy or "epoch"
        self.save_steps = save_steps or cfg.checkpointing.save_steps
        self.save_total_limit = save_total_limit or cfg.checkpointing.save_total_limit
        self.resume_from_checkpoint = resume_from_checkpoint or cfg.checkpointing.resume_from
        self.logging_steps = logging_steps or cfg.logging.logging_steps
        self.logging_dir = logging_dir or cfg.logging.log_dir
        self.eval_strategy = eval_strategy or "no"
        self.eval_steps = eval_steps or 500
        self.use_compile = use_compile
        self.max_memory_mb = max_memory_mb
        self.report_to = report_to or ["tensorboard"]