# =============================================================================
# MLX-VLA Command Line Interface Examples
# =============================================================================

# Train a VLA model
mlx-vla train \
    --data_path ./data/my_robot_demos \
    --dataset_format rlds \
    --vision_backbone clip \
    --action_type continuous \
    --output_dir ./output/my_vla \
    --num_epochs 100

# Fine-tune with LoRA
mlx-vla finetune \
    --base_model openvla-7b \
    --data_path ./data/task_specific \
    --lora_rank 16 \
    --output_dir ./lora_output

# Run inference
mlx-vla infer \
    --model_path ./output/my_vla \
    --image robot_view.jpg \
    --instruction "pick up the cup"

# Evaluate a trained model
mlx-vla evaluate \
    --model_path ./output/my_vla \
    --data_path ./data/test_set

# Export for deployment
mlx-vla export \
    --model_path ./output/my_vla \
    --output_path ./deployment/vla

# List available pretrained models
mlx-vla list_models
