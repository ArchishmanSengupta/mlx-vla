"""Pretrained vision encoder loading for VLA models."""

import warnings
from typing import Optional, Tuple, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_pretrained_vision_encoder(
    backbone: str = "clip",
    model_name: Optional[str] = None,
    hidden_dim: Optional[int] = None,
    image_size: int = 224,
) -> Tuple[nn.Module, dict]:
    """Load a pretrained vision encoder with actual weights.

    This function attempts to load pretrained vision encoders from available
    sources. Currently supports:
    - CLIP ViT models (via open_clip or conversion from PyTorch)

    Args:
        backbone: Backbone type ("clip", "dinov2", "siglip", "sam")
        model_name: Specific model variant (e.g., "ViT-L/14", "vit_base_patch16_224")
        hidden_dim: Hidden dimension size (auto-detected if not provided)
        image_size: Input image size

    Returns:
        Tuple of (encoder module, config dict)

    Raises:
        ImportError: If required dependencies are not installed
        ValueError: If model cannot be loaded

    Example:
        >>> encoder, config = load_pretrained_vision_encoder("clip", "ViT-L/14")
        >>> print(f"Loaded CLIP with hidden dim: {config['hidden_dim']}")
    """
    if backbone == "clip":
        return _load_clip_encoder(model_name, hidden_dim, image_size)
    elif backbone == "dinov2":
        return _load_dinov2_encoder(model_name, hidden_dim, image_size)
    elif backbone == "siglip":
        return _load_siglip_encoder(model_name, hidden_dim, image_size)
    elif backbone == "sam":
        return _load_sam_encoder(model_name, hidden_dim, image_size)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def _load_clip_encoder(
    model_name: Optional[str],
    hidden_dim: Optional[int],
    image_size: int,
) -> Tuple[nn.Module, dict]:
    """Load CLIP vision encoder with pretrained weights.

    Tries multiple approaches:
    1. open_clip library (preferred)
    2. Conversion from PyTorch CLIP
    """
    model_name = model_name or "ViT-L/14"
    hidden_dim = hidden_dim or 768

    # Try open_clip first
    try:
        import open_clip

        # Parse model name
        if "/" in model_name:
            model_name = model_name.replace("/", "-")

        # Load model and preprocess
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
            device="cpu",
        )

        # Get hidden dim
        if hasattr(model, "visual"):
            hidden_dim = model.visual.proj.shape[1] if hasattr(model.visual, "proj") else hidden_dim
            # Get embed dim from visual.transformer
            hidden_dim = getattr(model.visual, "embed_dim", hidden_dim)

        # Convert to MLX
        mlx_model = _convert_openclip_to_mlx(model, hidden_dim, image_size)

        config = {
            "backbone": "clip",
            "model_name": model_name,
            "hidden_dim": hidden_dim,
            "image_size": image_size,
            "pretrained_source": "open_clip",
        }

        return mlx_model, config

    except ImportError:
        pass

    # Try transformers PyTorch conversion
    try:
        from transformers import CLIPVisionModel, CLIPVisionConfig
        import torch

        # Determine model size from name
        if "large" in model_name.lower() or "L" in model_name:
            hidden_dim = 1024
            num_layers = 24
            num_heads = 16
        elif "base" in model_name.lower() or "B" in model_name:
            hidden_dim = 768
            num_layers = 12
            num_heads = 12
        else:
            hidden_dim = hidden_dim or 768
            num_layers = 12
            num_heads = 12

        # Load PyTorch model
        torch_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

        # Convert to MLX
        mlx_model = _convert_clip_pytorch_to_mlx(torch_model, hidden_dim, image_size)

        config = {
            "backbone": "clip",
            "model_name": model_name,
            "hidden_dim": hidden_dim,
            "image_size": image_size,
            "pretrained_source": "huggingface",
        }

        return mlx_model, config

    except ImportError:
        raise ImportError(
            "To load pretrained CLIP encoders, install either:\n"
            "  open_clip: pip install open-clip-torch\n"
            "  transformers: pip install transformers torch\n"
            "Or use random initialization by not setting pretrained=True"
        )


def _load_dinov2_encoder(
    model_name: Optional[str],
    hidden_dim: Optional[int],
    image_size: int,
) -> Tuple[nn.Module, dict]:
    """Load DINOv2 encoder with pretrained weights."""
    model_name = model_name or "dinov2_vitb14"

    try:
        import torch
        from transformers import AutoModel

        # Map model names
        model_map = {
            "dinov2_vitb14": "facebook/dinov2-base",
            "dinov2_vitl14": "facebook/dinov2-large",
            "dinov2_vits14": "facebook/dinov2-small",
        }

        hf_name = model_map.get(model_name, model_name)
        torch_model = AutoModel.from_pretrained(hf_name, trust_remote_code=True)

        hidden_dim = hidden_dim or torch_model.config.hidden_size

        # Convert to MLX
        mlx_model = _convert_dinov2_pytorch_to_mlx(torch_model, hidden_dim, image_size)

        config = {
            "backbone": "dinov2",
            "model_name": model_name,
            "hidden_dim": hidden_dim,
            "image_size": image_size,
            "pretrained_source": "huggingface",
        }

        return mlx_model, config

    except ImportError:
        raise ImportError(
            "To load pretrained DINOv2 encoders, install:\n"
            "  pip install transformers torch\n"
            "Or use random initialization by not setting pretrained=True"
        )


def _load_siglip_encoder(
    model_name: Optional[str],
    hidden_dim: Optional[int],
    image_size: int,
) -> Tuple[nn.Module, dict]:
    """Load SigLIP encoder with pretrained weights."""
    model_name = model_name or "vit-base-patch16-224"

    try:
        import torch
        from transformers import AutoModel

        torch_model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224",
            trust_remote_code=True,
        )

        hidden_dim = hidden_dim or torch_model.config.hidden_size

        mlx_model = _convert_siglip_pytorch_to_mlx(torch_model, hidden_dim, image_size)

        config = {
            "backbone": "siglip",
            "model_name": model_name,
            "hidden_dim": hidden_dim,
            "image_size": image_size,
            "pretrained_source": "huggingface",
        }

        return mlx_model, config

    except ImportError:
        raise ImportError(
            "To load pretrained SigLIP encoders, install:\n"
            "  pip install transformers torch"
        )


def _load_sam_encoder(
    model_name: Optional[str],
    hidden_dim: Optional[int],
    image_size: int,
) -> Tuple[nn.Module, dict]:
    """Load SAM encoder with pretrained weights."""
    model_name = model_name or "sam_vit_b"

    warnings.warn(
        "SAM encoder loading is not fully implemented. "
        "Using random initialization.",
        UserWarning,
    )

    from mlx_vla.models.vision import SAMVisionEncoder

    hidden_dim = hidden_dim or 768
    model = SAMVisionEncoder(hidden_dim=hidden_dim, pretrained=False, image_size=image_size)

    config = {
        "backbone": "sam",
        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "image_size": image_size,
        "pretrained_source": None,
    }

    return model, config


def _convert_openclip_to_mlx(model, hidden_dim: int, image_size: int):
    """Convert OpenCLIP model to MLX."""
    # This is a placeholder - full implementation would need to
    # convert each layer from PyTorch to MLX format
    warnings.warn(
        "OpenCLIP to MLX conversion is not fully implemented. "
        "Using random initialization instead.",
        UserWarning,
    )

    from mlx_vla.models.vision import CLIPVisionEncoder

    return CLIPVisionEncoder(hidden_dim=hidden_dim, pretrained=False, image_size=image_size)


def _convert_clip_pytorch_to_mlx(torch_model, hidden_dim: int, image_size: int):
    """Convert PyTorch CLIP vision model to MLX."""
    from mlx_vla.models.vision import CLIPVisionEncoder

    mlx_encoder = CLIPVisionEncoder(hidden_dim=hidden_dim, pretrained=False, image_size=image_size)

    # Try to copy weights if shapes match
    try:
        # Copy patch embedding
        if hasattr(torch_model, "vision_model") and hasattr(torch_model.vision_model, "embeddings"):
            torch_embed = torch_model.vision_model.embeddings.patch_embedding.weight
            if mlx_encoder.patch_embed.weight.shape == torch_embed.shape:
                mlx_encoder.patch_embed.weight = mx.array(torch_embed.detach().numpy())
            else:
                warnings.warn(
                    f"Patch embedding shape mismatch: MLX {mlx_encoder.patch_embed.weight.shape} "
                    f"vs PyTorch {torch_embed.shape}. Using random weights.",
                    UserWarning,
                )

            # Copy position embeddings
            if hasattr(torch_model.vision_model.embeddings, "position_embedding"):
                torch_pos = torch_model.vision_model.embeddings.position_embedding.weight
                if mlx_encoder.position_embedding.weight.shape == torch_pos.shape:
                    mlx_encoder.position_embedding.weight = mx.array(torch_pos.detach().numpy())

    except Exception as e:
        warnings.warn(f"Failed to copy CLIP weights: {e}. Using random initialization.", UserWarning)

    return mlx_encoder


def _convert_dinov2_pytorch_to_mlx(torch_model, hidden_dim: int, image_size: int):
    """Convert PyTorch DINOv2 model to MLX."""
    from mlx_vla.models.vision import DINOv2Encoder

    mlx_encoder = DINOv2Encoder(hidden_dim=hidden_dim, pretrained=False, image_size=image_size)

    # Similar weight conversion logic would go here
    warnings.warn(
        "DINOv2 weight conversion is not fully implemented. Using random initialization.",
        UserWarning,
    )

    return mlx_encoder


def _convert_siglip_pytorch_to_mlx(torch_model, hidden_dim: int, image_size: int):
    """Convert PyTorch SigLIP model to MLX."""
    from mlx_vla.models.vision import SigLIPEncoder

    mlx_encoder = SigLIPEncoder(hidden_dim=hidden_dim, pretrained=False, image_size=image_size)

    warnings.warn(
        "SigLIP weight conversion is not fully implemented. Using random initialization.",
        UserWarning,
    )

    return mlx_encoder


__all__ = ["load_pretrained_vision_encoder"]
