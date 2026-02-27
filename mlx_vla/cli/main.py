import argparse
import sys
from pathlib import Path

from mlx_vla.utils.config import load_config, VLAConfigManager
from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.data.dataset import RLDSDataset, EpisodeDataset
from mlx_vla.training.trainer import VLATrainer
from mlx_vla.core.config import VLATrainingArguments
from mlx_vla.training.lora import apply_lora
from mlx_vla.inference.pipeline import VLAPipeline

def train_command(args):
    cfg = load_config(args.config) if args.config else VLAConfigManager.from_default()

    if args.model:
        cfg.update(model={"name": args.model})
    if args.dataset:
        cfg.update(data={"dataset_name": args.dataset})
    if args.use_lora is not None:
        cfg.update(lora={"enabled": args.use_lora})
    if args.lora_rank:
        cfg.update(lora={"rank": args.lora_rank})
    if args.action_type:
        cfg.update(model={"action_type": args.action_type})
    if args.action_dim:
        cfg.update(model={"action_dim": args.action_dim})
    if args.epochs:
        cfg.update(training={"epochs": args.epochs})
    if args.batch_size:
        cfg.update(data={"batch_size": args.batch_size})
    if args.learning_rate:
        cfg.update(training={"learning_rate": args.learning_rate})
    if args.output_dir:
        cfg.update(checkpointing={"output_dir": args.output_dir})

    print(f"Model: {cfg.model.name}")
    print(f"Dataset: {cfg.data.dataset_name}")
    print(f"LoRA: {cfg.lora.enabled} (rank={cfg.lora.rank})")

    vla_model = VLAForAction.from_pretrained(
        cfg.model.name,
        action_type=cfg.model.action_type,
        action_dim=cfg.model.action_dim,
        vision_backbone=cfg.model.vision_backbone,
        vision_hidden_dim=cfg.model.vision_hidden_dim,
        language_hidden_dim=cfg.model.language_hidden_dim,
        fusion_type=cfg.model.fusion_type,
    )

    if cfg.lora.enabled:
        vla_model = apply_lora(
            vla_model,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
        )

    if cfg.data.dataset_name.startswith("oxe/") or cfg.data.dataset_name in ["bridge_v2", "aloha"]:
        train_dataset = RLDSDataset(cfg.data.dataset_name, split="train")
    else:
        train_dataset = EpisodeDataset(cfg.data.dataset_name, split="train")

    train_args = VLATrainingArguments(
        output_dir=cfg.checkpointing.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.data.batch_size,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        save_steps=cfg.checkpointing.save_steps,
        save_total_limit=cfg.checkpointing.save_total_limit,
        resume_from_checkpoint=cfg.checkpointing.resume_from,
        logging_steps=cfg.logging.logging_steps,
        logging_dir=cfg.logging.log_dir,
    )

    print("Starting training...")
    trainer = VLATrainer(
        model=vla_model,
        args=train_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    cfg.save(f"{cfg.checkpointing.output_dir}/config.yaml")
    print(f"Training complete! Config saved to {cfg.checkpointing.output_dir}/config.yaml")

def infer_command(args):
    pipeline = VLAPipeline(model=args.model)
    action = pipeline.predict(
        image=args.image,
        language=args.instruction,
        unnorm_key=args.unnorm_key,
    )
    print(f"Predicted action: {action}")

def export_command(args):
    model = VLAForAction.load(args.model)
    model.save(args.output)
    print(f"Exported to: {args.output}")

def create_config_command(args):
    cfg = VLAConfigManager.from_default()
    cfg.update(
        model={"name": args.model or "openvla-7b"},
        data={"dataset_name": args.dataset or "bridge_v2"},
    )
    cfg.save(args.output)
    print(f"Config saved to: {args.output}")

def main():
    parser = argparse.ArgumentParser(prog="mlx-vla")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a VLA model")
    train_parser.add_argument("--config", type=str, help="YAML config file")
    train_parser.add_argument("--model", type=str, help="Model name")
    train_parser.add_argument("--dataset", type=str, help="Dataset name or path")
    train_parser.add_argument("--use-lora", type=bool, help="Enable LoRA")
    train_parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    train_parser.add_argument("--action-type", type=str, help="Action type")
    train_parser.add_argument("--action-dim", type=int, help="Action dimension")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--output-dir", type=str, help="Output directory")

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model", type=str, required=True, help="Model path")
    infer_parser.add_argument("--image", type=str, required=True, help="Image path")
    infer_parser.add_argument("--instruction", type=str, required=True, help="Instruction")
    infer_parser.add_argument("--unnorm-key", type=str, default="bridge_orig", help="Robot type")

    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--model", type=str, required=True, help="Model path")
    export_parser.add_argument("--output", type=str, required=True, help="Output path")

    config_parser = subparsers.add_parser("create-config", help="Create config file")
    config_parser.add_argument("--output", type=str, default="config.yaml", help="Output path")
    config_parser.add_argument("--model", type=str, help="Model name")
    config_parser.add_argument("--dataset", type=str, help="Dataset name")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "export":
        export_command(args)
    elif args.command == "create-config":
        create_config_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()