"""High-level training API for VLAs."""

from typing import Optional, Union, Any
import warnings

from mlx_vla.utils.config import load_config
from mlx_vla.utils.pretrained import get_model_config, get_default_config
from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.data.dataset import VLADataset, RLDSDataset, BridgeDataset, EpisodeDataset
from mlx_vla.data.collator import VLAModuleDataCollator
from mlx_vla.data.dataloader import VLADataloader
from mlx_vla.data.normalizer import ActionNormalizer
from mlx_vla.training.trainer import VLATrainer
from mlx_vla.training.lora import apply_lora
from mlx_vla.core import VLATrainingArguments
from mlx_vla.training.optimizers import create_optimizer, create_scheduler


def train_vla(
    model: str = "openvla-7b",
    dataset: str = "bridge_v2",
    use_lora: bool = True,
    output_dir: str = "./vla_output",
    num_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    **kwargs,
) -> VLATrainer:
    """High-level API to train a Vision-Language-Action model.

    This is a convenience function that sets up training with sensible defaults.

    Args:
        model: Model name (e.g., "openvla-7b", "openvla-3b", "octo-base")
        dataset: Dataset name or path. Supported:
            - "bridge_v2", "oxe/bridge_v2" (RLDS format)
            - Path to local dataset directory
        use_lora: Whether to use LoRA fine-tuning (recommended for fine-tuning)
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        **kwargs: Additional training arguments

    Returns:
        VLATrainer: The configured trainer instance

    Example:
        >>> trainer = train_vla(
        ...     model="openvla-7b",
        ...     dataset="bridge_v2",
        ...     use_lora=True,
        ...     output_dir="./output",
        ...     num_epochs=3,
        ... )
        >>> trainer.train()

    Note:
        - For RLDS datasets (bridge_v2, oxe/*), you need tensorflow_datasets installed
        - For local datasets, use EpisodeDataset directly for more control
        - The model will be initialized randomly unless pretrained weights are loaded
    """
    # Get model configuration
    model_config = get_model_config(model)

    # Create the VLA model
    vla_model = VLAForAction.from_pretrained(
        model_name_or_path=model,
        action_type=model_config.get("action_type", "discrete"),
        action_dim=model_config.get("action_dim", 7),
        **model_config,
    )

    # Apply LoRA if requested
    if use_lora:
        warnings.warn(
            "LoRA is requested. Note: apply_lora() only works on Linear layers "
            "and won't modify vision/language encoders unless they have Linear layers.",
            UserWarning,
        )
        lora_rank = kwargs.pop("lora_rank", 8)
        lora_alpha = kwargs.pop("lora_alpha", 16)
        lora_dropout = kwargs.pop("lora_dropout", 0.05)
        target_modules = kwargs.pop("lora_target_modules", ["q_proj", "v_proj"])

        vla_model = apply_lora(
            vla_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
        )

    # Load dataset
    if dataset.startswith("oxe/") or dataset in RLDSDataset.SUPPORTED_DATASETS:
        train_dataset = RLDSDataset(
            dataset_name=dataset,
            split="train",
            image_size=model_config.get("image_size", 224),
        )
    elif dataset == "bridge_v2" or "/" not in dataset:
        # Try as RLDS first, then as local path
        try:
            train_dataset = RLDSDataset(
                dataset_name=dataset,
                split="train",
                image_size=model_config.get("image_size", 224),
            )
        except Exception:
            # Try as local dataset
            train_dataset = EpisodeDataset(
                data_path=dataset,
                split="train",
                image_size=model_config.get("image_size", 224),
            )
    else:
        # Local dataset
        train_dataset = EpisodeDataset(
            data_path=dataset,
            split="train",
            image_size=model_config.get("image_size", 224),
        )

    # Create data collator
    data_collator = VLAModuleDataCollator(
        image_size=model_config.get("image_size", 224),
        action_normalization=kwargs.pop("action_normalization", "clip_minus_one_to_one"),
        action_dim=model_config.get("action_dim", 7),
    )

    # Create training arguments
    training_args = VLATrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs,
    )

    # Create trainer
    trainer = VLATrainer(
        model=vla_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    return trainer


__all__ = ["train_vla"]
