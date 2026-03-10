from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import yaml

DEFAULT_CONFIG = {
    "model": {
        "name": "openvla-7b",
        "vision_backbone": "clip",
        "vision_hidden_dim": 768,
        "language_hidden_dim": 4096,
        "fusion_type": "cross_attention",
        "action_type": "discrete",
        "action_dim": 7,
        "action_horizon": 1,
        "num_action_bins": 256,
    },
    "lora": {
        "enabled": True,
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["query_proj", "key_proj", "value_proj", "out_proj"],
    },
    "data": {
        "dataset_name": "bridge_v2",
        "image_size": 224,
        "batch_size": 1,
        "num_workers": 0,
        "action_normalization": "clip_minus_one_to_one",
    },
    "training": {
        "epochs": 3,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 8,
        "lr_scheduler_type": "cosine",
        "eval_strategy": "no",
        "eval_steps": 500,
    },
    "checkpointing": {
        "output_dir": "./vla_output",
        "save_steps": 500,
        "save_total_limit": 3,
        "save_strategy": "epoch",
        "resume_from": None,
    },
    "logging": {
        "logging_steps": 10,
        "log_dir": None,
        "report_to": ["tensorboard"],
    },
}

@dataclass
class ModelConfig:
    name: str = "openvla-7b"
    vision_backbone: str = "clip"
    vision_hidden_dim: int = 768
    language_hidden_dim: int = 4096
    fusion_type: str = "cross_attention"
    action_type: str = "discrete"
    action_dim: int = 7
    action_horizon: int = 1
    num_action_bins: int = 256

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["query_proj", "key_proj", "value_proj", "out_proj"])

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LoRAConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class DataConfig:
    dataset_name: str = "bridge_v2"
    image_size: int = 224
    batch_size: int = 1
    num_workers: int = 0
    action_normalization: str = "clip_minus_one_to_one"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "DataConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8
    lr_scheduler_type: str = "cosine"
    eval_strategy: str = "no"
    eval_steps: int = 500

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class CheckpointingConfig:
    output_dir: str = "./vla_output"
    save_steps: int = 500
    save_total_limit: int = 3
    save_strategy: str = "epoch"
    resume_from: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class LoggingConfig:
    logging_steps: int = 10
    log_dir: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LoggingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

class VLAConfigManager:
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        lora: Optional[LoRAConfig] = None,
        data: Optional[DataConfig] = None,
        training: Optional[TrainingConfig] = None,
        checkpointing: Optional[CheckpointingConfig] = None,
        logging: Optional[LoggingConfig] = None,
    ):
        self.model = model or ModelConfig()
        self.lora = lora or LoRAConfig()
        self.data = data or DataConfig()
        self.training = training or TrainingConfig()
        self.checkpointing = checkpointing or CheckpointingConfig()
        self.logging = logging or LoggingConfig()

    def to_dict(self) -> Dict:
        return {
            "model": self.model.to_dict(),
            "lora": self.lora.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "checkpointing": self.checkpointing.to_dict(),
            "logging": self.logging.to_dict(),
        }

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            if path.suffix == ".yaml" or path.suffix == ".yml":
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VLAConfigManager":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            if path.suffix == ".yaml" or path.suffix == ".yml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(
            model=ModelConfig.from_dict(data.get("model", {})),
            lora=LoRAConfig.from_dict(data.get("lora", {})),
            data=DataConfig.from_dict(data.get("data", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            checkpointing=CheckpointingConfig.from_dict(data.get("checkpointing", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
        )

    @classmethod
    def from_default(cls) -> "VLAConfigManager":
        return cls(
            model=ModelConfig(**DEFAULT_CONFIG["model"]),
            lora=LoRAConfig(**DEFAULT_CONFIG["lora"]),
            data=DataConfig(**DEFAULT_CONFIG["data"]),
            training=TrainingConfig(**DEFAULT_CONFIG["training"]),
            checkpointing=CheckpointingConfig(**DEFAULT_CONFIG["checkpointing"]),
            logging=LoggingConfig(**DEFAULT_CONFIG["logging"]),
        )

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                config = getattr(self, key)
                if isinstance(value, dict):
                    for k, v in value.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
                else:
                    setattr(self, key, value)
        return self

_global_config = None

def load_config(path: str = None, **overrides) -> VLAConfigManager:
    global _global_config
    if path:
        config = VLAConfigManager.load(path)
    else:
        config = VLAConfigManager.from_default()

    if overrides:
        config.update(**overrides)

    _global_config = config
    return config

def get_global_config() -> VLAConfigManager:
    global _global_config
    if _global_config is None:
        _global_config = VLAConfigManager.from_default()
    return _global_config

def set_global_config(config: VLAConfigManager):
    global _global_config
    _global_config = config