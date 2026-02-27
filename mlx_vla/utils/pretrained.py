from typing import Optional, Dict, Any
import json
from pathlib import Path

def get_model_config(model_name: str) -> Dict[str, Any]:
    model_configs = {
        "openvla-7b": {
            "vision_backbone": "clip",
            "vision_hidden_dim": 1024,
            "language_hidden_dim": 4096,
            "fusion_type": "cross_attention",
            "action_type": "discrete",
            "action_dim": 7,
            "num_action_bins": 256,
        },
        "openvla-3b": {
            "vision_backbone": "clip",
            "vision_hidden_dim": 768,
            "language_hidden_dim": 4096,
            "fusion_type": "cross_attention",
            "action_type": "discrete",
            "action_dim": 7,
            "num_action_bins": 256,
        },
        "llava-1.5-7b": {
            "vision_backbone": "clip",
            "vision_hidden_dim": 1024,
            "language_hidden_dim": 4096,
            "fusion_type": "concat",
            "action_type": "continuous",
            "action_dim": 7,
        },
        "llava-1.5-13b": {
            "vision_backbone": "clip",
            "vision_hidden_dim": 1024,
            "language_hidden_dim": 5120,
            "fusion_type": "concat",
            "action_type": "continuous",
            "action_dim": 7,
        },
        "octo-small": {
            "vision_backbone": "dinov2",
            "vision_hidden_dim": 384,
            "language_hidden_dim": 512,
            "fusion_type": "concat",
            "action_type": "diffusion",
            "action_dim": 7,
            "action_horizon": 4,
        },
        "octo-base": {
            "vision_backbone": "dinov2",
            "vision_hidden_dim": 768,
            "language_hidden_dim": 512,
            "fusion_type": "concat",
            "action_type": "diffusion",
            "action_dim": 7,
            "action_horizon": 4,
        },
    }

    for key, config in model_configs.items():
        if key in model_name.lower():
            return config

    return {
        "vision_backbone": "clip",
        "vision_hidden_dim": 768,
        "language_hidden_dim": 4096,
        "fusion_type": "cross_attention",
        "action_type": "discrete",
        "action_dim": 7,
    }

def get_default_config(model_path: str) -> Dict[str, Any]:
    model_name = model_path.split("/")[-1].lower()

    config = get_model_config(model_name)

    model_dir = Path(model_path)
    if model_dir.exists():
        config_file = model_dir / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                saved_config = json.load(f)
                config.update(saved_config)

    return config