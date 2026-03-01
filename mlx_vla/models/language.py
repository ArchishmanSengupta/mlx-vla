"""Language model integration for VLA models."""

import warnings
from typing import Optional, Tuple
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class LanguageModelWrapper(nn.Module):
    """Wrapper for language models in VLA architecture.

    This class provides a unified interface for different language models
    (embedding-only, small transformer, or full LLM).
    """

    def __init__(
        self,
        language_model: nn.Module,
        hidden_dim: int,
        use_full_model: bool = False,
    ):
        """Initialize language model wrapper.

        Args:
            language_model: The underlying language model (embedding or full LM)
            hidden_dim: Hidden dimension size
            use_full_model: Whether this is a full transformer model
        """
        super().__init__()
        self.language_model = language_model
        self.hidden_dim = hidden_dim
        self.use_full_model = use_full_model

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass through language model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Hidden states of shape (batch, seq_len, hidden_dim)
        """
        if self.use_full_model:
            # Full transformer model
            outputs = self.language_model(input_ids)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs
        else:
            # Just embedding lookup
            return self.language_model(input_ids)


def load_language_model(
    model_name: str = "llama-7b",
    hidden_dim: Optional[int] = None,
    max_seq_length: int = 512,
    **kwargs,
) -> Tuple[nn.Module, dict]:
    """Load a language model for VLA.

    Currently supports:
    - Embedding-only (fallback)
    - LLaMA via MLX-LM
    - Small transformer language model

    Args:
        model_name: Name of the language model
        hidden_dim: Hidden dimension (auto-detected if not provided)
        max_seq_length: Maximum sequence length
        **kwargs: Additional arguments

    Returns:
        Tuple of (language model, config dict)

    Example:
        >>> lm, config = load_language_model("llama-7b", hidden_dim=4096)
        >>> print(f"Loaded {config['model_name']} with dim {config['hidden_dim']}")
    """
    model_name_lower = model_name.lower()

    # Try loading with MLX-LM (for LLaMA, Mistral, etc.)
    if any(name in model_name_lower for name in ["llama", "mistral", "qwen", "gemma"]):
        try:
            return _load_mlx_lm(model_name, hidden_dim, max_seq_length)
        except ImportError:
            pass

    # Try loading with transformers
    try:
        return _load_transformers_lm(model_name, hidden_dim, max_seq_length)
    except ImportError:
        pass

    # Fallback to embedding-only
    warnings.warn(
        f"Could not load full language model '{model_name}'. "
        f"Using simple embedding layer instead. "
        f"Install mlx-lm or transformers for full LLM support.",
        UserWarning,
    )
    return _create_embedding_model(hidden_dim or 4096)


def _load_mlx_lm(
    model_name: str,
    hidden_dim: Optional[int],
    max_seq_length: int,
) -> Tuple[nn.Module, dict]:
    """Load language model using MLX-LM."""
    try:
        from mlx_lm import load as load_mlx_model

        model, tokenizer = load_mlx_model(model_name, trust_remote_code=True)

        # Get hidden dim from model
        if hidden_dim is None:
            if hasattr(model, "config"):
                hidden_dim = getattr(model.config, "hidden_size", 4096)
            elif hasattr(model, "embed_dim"):
                hidden_dim = model.embed_dim
            else:
                hidden_dim = 4096

        wrapper = LanguageModelWrapper(model, hidden_dim, use_full_model=True)

        config = {
            "model_name": model_name,
            "hidden_dim": hidden_dim,
            "max_seq_length": max_seq_length,
            "framework": "mlx-lm",
        }

        return wrapper, config

    except ImportError:
        raise ImportError("mlx-lm is required for loading LLaMA/Mistral models")


def _load_transformers_lm(
    model_name: str,
    hidden_dim: Optional[int],
    max_seq_length: int,
) -> Tuple[nn.Module, dict]:
    """Load language model using transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        raise ImportError("transformers and torch are required for this feature")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    )

    # Get hidden dim
    if hidden_dim is None:
        hidden_dim = getattr(config, "hidden_size", 4096)

    # Convert to MLX (placeholder - full conversion not implemented)
    warnings.warn(
        "Full transformer conversion from PyTorch to MLX is not implemented. "
        "Using embedding layer as fallback.",
        UserWarning,
    )

    mlx_model = _create_embedding_model(
        hidden_dim,
        vocab_size=len(torch_model.model.embed_tokens),
    )

    config_dict = {
        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "max_seq_length": max_seq_length,
        "framework": "pytorch_conversion_fallback",
    }

    return mlx_model, config_dict


def _create_embedding_model(
    hidden_dim: int,
    vocab_size: int = 32000,
) -> Tuple[nn.Module, dict]:
    """Create a simple embedding model as fallback."""
    embedding = nn.Embedding(vocab_size, hidden_dim)

    config = {
        "model_name": "embedding_only",
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "framework": "mlx",
    }

    wrapper = LanguageModelWrapper(embedding, hidden_dim, use_full_model=False)
    return wrapper, config


class VLALanguageEncoder(nn.Module):
    """Transformer-based language encoder for VLAs.

    This is a small language encoder that can be used when full LLM
    is not available. It's not as powerful as LLaMA but works for
    basic VLA tasks.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_length: int = 512,
    ):
        """Initialize language encoder.

        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=hidden_dim,
            num_heads=num_heads,
            mlp_dims=hidden_dim * 4,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Hidden states of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Position embeddings
        positions = mx.arange(seq_len, dtype=mx.int32)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb

        # Transformer - pass mask=None explicitly for MLX
        x = self.transformer(x, mask=None)
        x = self.norm(x)

        return x


def create_small_language_encoder(
    hidden_dim: int = 768,
    vocab_size: int = 32000,
    num_layers: int = 6,
) -> Tuple[VLALanguageEncoder, dict]:
    """Create a small transformer language encoder.

    Args:
        hidden_dim: Hidden dimension
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers

    Returns:
        Tuple of (encoder, config)
    """
    model = VLALanguageEncoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    config = {
        "model_name": "small_transformer",
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "framework": "mlx",
    }

    return model, config


__all__ = [
    "LanguageModelWrapper",
    "load_language_model",
    "VLALanguageEncoder",
    "create_small_language_encoder",
]
