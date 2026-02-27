"""
MLX-VLA Usage Examples
=======================

This file demonstrates how different users can leverage the MLX-VLA framework
for various Vision-Language-Action tasks on Apple Silicon.
"""

# =============================================================================
# Example 1: Robotics Researcher - Training a VLA from Scratch
# =============================================================================

def example_1_robotics_researcher():
    """
    A robotics researcher wants to train a VLA model on their own
    robot demonstration dataset for pick-and-place tasks.
    """
    from mlx_vla import VLA, VLATrainer, VLADataset, VLAModuleDataCollator, VLATrainingArguments
    from mlx_vla.utils.config import set_global_config, VLAConfigManager

    # Configure training
    config = VLAConfigManager()
    config.training.learning_rate = 1e-4
    config.training.num_train_epochs = 100
    config.training.per_device_train_batch_size = 8
    config.checkpointing.output_dir = "./my_robot_vla"

    # Create dataset from RLDS format data
    train_dataset = VLADataset(
        data_path="./data/my_robot_demos",
        dataset_format="rlds",
    )

    eval_dataset = VLADataset(
        data_path="./data/my_robot_demos_val",
        dataset_format="rlds",
    )

    # Create VLA model with CLIP vision encoder
    model = VLA(
        vision_backbone="clip",
        vision_hidden_dim=768,
        action_type="continuous",
        action_dim=7,
        action_horizon=8,
    )

    # Configure training arguments
    training_args = VLATrainingArguments(
        output_dir="./my_robot_vla",
        num_train_epochs=100,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        save_steps=500,
        eval_strategy="epoch",
    )

    # Initialize trainer
    trainer = VLATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train!
    trainer.train()


# =============================================================================
# Example 2: ML Engineer - Fine-tuning OpenVLA with LoRA
# =============================================================================

def example_2_ml_engineer_lora():
    """
    An ML engineer wants to fine-tune a pretrained VLA model using
    LoRA for efficient adaptation to a new task.
    """
    from mlx_vla import VLA, VLATrainer, VLADataset, VLATrainingArguments
    from mlx_vla.training.lora import LoRAConfig, apply_lora

    # Load pretrained VLA
    model = VLA.from_pretrained("openvla-7b")

    # Configure LoRA
    lora_config = LoRAConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )

    # Apply LoRA to model
    model = apply_lora(model, lora_config)

    # Load domain-specific dataset
    dataset = VLADataset(
        data_path="./data/kitchen_tasks",
        dataset_format="rlds",
    )

    # Training arguments for LoRA
    args = VLATrainingArguments(
        output_dir="./lora_finetuned_vla",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        gradient_accumulation_steps=4,
    )

    trainer = VLATrainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save fine-tuned model
    model.save("./lora_finetuned_vla/final")


# =============================================================================
# Example 3: Hobbyist - Using Pretrained VLA for Inference
# =============================================================================

def example_3_hobbyist_inference():
    """
    A hobbyist wants to use a pretrained VLA model for running
    inference on their local robot.
    """
    from mlx_vla import VLA
    from mlx_vla.inference import VLAInferenceEngine
    import mlx.core as mx
    from PIL import Image
    import numpy as np

    # Load pretrained model
    model = VLA.from_pretrained("bridge_vla")

    # Create inference engine
    inference_engine = VLAInferenceEngine(model)

    # Load observation image
    image = Image.open("robot_view.jpg").resize((224, 224))

    # Get robot's current observation
    # In real usage, this would come from camera

    # Predict action
    action = inference_engine.predict(
        image=image,
        language_instruction="pick up the cup",
        temperature=0.7,
    )

    print(f"Predicted action: {action}")
    # Output: [x, y, z, roll, pitch, yaw, gripper]


# =============================================================================
# Example 4: Data Scientist - Training with Diffusion Policy
# =============================================================================

def example_4_data_scientist_diffusion():
    """
    A data scientist wants to train a VLA with diffusion action head
    for smoother action prediction in contact-rich tasks.
    """
    from mlx_vla import VLA, VLATrainer, VLADataset, VLATrainingArguments
    from mlx_vla.data.normalizer import ActionNormalizer

    # Create dataset with normalization for diffusion
    normalizer = ActionNormalizer(robot="franka", action_dim=7)

    dataset = VLADataset(
        data_path="./data/surgical_demos",
        dataset_format="rlds",
        action_normalizer=normalizer,
    )

    # Use diffusion action head for continuous control
    model = VLA(
        vision_backbone="dinov2",
        vision_hidden_dim=768,
        action_type="diffusion",
        action_dim=7,
        action_horizon=16,  # Longer horizon for smooth motions
    )

    args = VLATrainingArguments(
        output_dir="./diffusion_vla",
        num_train_epochs=50,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
    )

    trainer = VLATrainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()


# =============================================================================
# Example 5: Research Scientist - Experimenting with Different Fusion
# =============================================================================

def example_5_researcher_fusion():
    """
    A research scientist wants to compare different vision-language
    fusion strategies for VLA models.
    """
    from mlx_vla import VLA, VLATrainer, VLADataset, VLATrainingArguments
    from mlx_vla.training.evaluation import evaluate_fusion_strategies

    dataset = VLADataset(
        data_path="./data/manipulation",
        dataset_format="rlds",
    )

    # Compare different fusion types
    fusion_types = ["cross_attention", "concat", "gated", "qkv_fusion"]

    results = {}
    for fusion_type in fusion_types:
        print(f"Training with {fusion_type} fusion...")

        model = VLA(
            vision_backbone="siglip",
            vision_hidden_dim=768,
            action_type="continuous",
            fusion_type=fusion_type,
        )

        args = VLATrainingArguments(
            output_dir=f"./fusion_{fusion_type}",
            num_train_epochs=20,
            per_device_train_batch_size=8,
        )

        trainer = VLATrainer(
            model=model,
            args=args,
            train_dataset=dataset,
        )

        trainer.train()

        # Evaluate
        eval_metrics = trainer.evaluate()
        results[fusion_type] = eval_metrics

    print("Results:")
    for fusion_type, metrics in results.items():
        print(f"  {fusion_type}: {metrics}")


# =============================================================================
# Example 6: Startup - Building Production Pipeline
# =============================================================================

def example_6_startup_pipeline():
    """
    A startup wants to build a complete production pipeline with
    data augmentation, validation, and model versioning.
    """
    from mlx_vla import VLA, VLADataset, VLATrainingArguments
    from mlx_vla.data.augmentation import ImageAugmentation, ActionNoiseInjection
    from mlx_vla.training.callbacks import ModelVersioningCallback
    from mlx_vla.training.trainer import VLATrainer

    # Custom dataset with augmentation
    class AugmentedVLADataset(VLADataset):
        def __init__(self, *args, image_aug=None, action_aug=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.image_aug = image_aug or ImageAugmentation()
            self.action_aug = action_aug or ActionNoiseInjection()

        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item = self.image_aug(item)
            item = self.action_aug(item)
            return item

    # Create augmented dataset
    train_dataset = AugmentedVLADataset(
        data_path="./data/production_demos",
        dataset_format="rlds",
        image_aug=ImageAugmentation(
            random_crop=True,
            random_flip=True,
            color_jitter=True,
        ),
        action_aug=ActionNoiseInjection(noise_std=0.01),
    )

    val_dataset = VLADataset(
        data_path="./data/production_val",
        dataset_format="rlds",
    )

    # Production model
    model = VLA(
        vision_backbone="clip",
        vision_hidden_dim=1024,  # Larger for production
        action_type="discrete",
        action_dim=7,
        num_action_bins=256,
    )

    # Production training args
    args = VLATrainingArguments(
        output_dir="./production_vla",
        num_train_epochs=200,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_steps=1000,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        report_to=["tensorboard", "wandb"],
    )

    # Trainer with versioning
    trainer = VLATrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # Export for deployment
    model.export_for_deployment("./production_vla/deployment")


# =============================================================================
# Example 7: Student - Learning VLA Training
# =============================================================================

def example_7_student_learning():
    """
    A student wants to learn about VLAs by training a small model
    on a tiny dataset for quick experimentation.
    """
    from mlx_vla import VLA, VLADataset, VLATrainingArguments
    from mlx_vla.training.trainer import VLATrainer
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    # Create a tiny dummy dataset for learning
    class TinyDemoDataset(VLADataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random "observation"
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            action = np.random.randn(7).astype(np.float32)
            return {
                "image": image,
                "action": action,
            }

    # Small model for quick learning
    model = VLA(
        vision_backbone="sam",  # Smaller encoder
        vision_hidden_dim=256,  # Smaller hidden dim
        action_type="continuous",
        action_dim=7,
    )

    # Quick training config
    args = VLATrainingArguments(
        output_dir="./learning_vla",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=1e-3,  # Higher LR for quick learning
        max_steps=100,  # Just 100 steps for demo
    )

    dataset = TinyDemoDataset()

    trainer = VLATrainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    # Train and watch the loss go down!
    trainer.train()

    print("Training complete! Try running inference now.")


# =============================================================================
# Example 8: Enterprise - Multi-GPU Training
# =============================================================================

def example_8_enterprise_multigpu():
    """
    An enterprise needs to train on multiple GPUs for large-scale
    VLA training on massive datasets.
    """
    from mlx_vla import VLA, VLADataset, VLATrainingArguments
    from mlx_vla.training.distributed import DistributedVLATrainer

    # Large dataset
    dataset = VLADataset(
        data_path="s3://enterprise-robotics-dataset/full",
        dataset_format="rlds",
        streaming=True,  # Stream from cloud storage
    )

    # Large model
    model = VLA(
        vision_backbone="dinov2",
        vision_hidden_dim=1536,
        language_model="llama3-8b",  # Full LLM
        action_type="diffusion",
        action_dim=14,  # 7 joints + 7 velocities
        action_horizon=32,
    )

    # Multi-GPU training config
    args = VLATrainingArguments(
        output_dir="./enterprise_vla",
        num_train_epochs=50,
        per_device_train_batch_size=64,
        learning_rate=5e-5,
        # Distributed training
        distributed_strategy="fsdp",
        fsdp_backbone="transformer",
        # Performance
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        # Checkpointing
        save_steps=1000,
        save_total_limit=10,
    )

    # Distributed trainer
    trainer = DistributedVLATrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        num_nodes=4,  # 4 nodes
        num_gpus=8,   # 8 GPUs per node
    )

    trainer.train()


# =============================================================================
# Main - Run Examples
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python examples.py <example_number>")
        print("Available examples:")
        print("  1 - Robotics Researcher (training from scratch)")
        print("  2 - ML Engineer (LoRA fine-tuning)")
        print("  3 - Hobbyist (inference)")
        print("  4 - Data Scientist (diffusion policy)")
        print("  5 - Researcher (fusion comparison)")
        print("  6 - Startup (production pipeline)")
        print("  7 - Student (learning)")
        print("  8 - Enterprise (multi-GPU)")
        sys.exit(1)

    example_num = sys.argv[1]

    examples = {
        "1": example_1_robotics_researcher,
        "2": example_2_ml_engineer_lora,
        "3": example_3_hobbyist_inference,
        "4": example_4_data_scientist_diffusion,
        "5": example_5_researcher_fusion,
        "6": example_6_startup_pipeline,
        "7": example_7_student_learning,
        "8": example_8_enterprise_multigpu,
    }

    if example_num not in examples:
        print(f"Unknown example: {example_num}")
        sys.exit(1)

    print(f"Running Example {example_num}...")
    examples[example_num]()
