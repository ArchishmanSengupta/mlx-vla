"""Tokenizers for VLA models."""

from typing import Optional, List, Union, Any
from pathlib import Path
import numpy as np


class VLATokenizer:
    """Tokenizer for VLA models that wraps HuggingFace tokenizers.

    This class provides a unified interface for tokenizing language instructions
    for use with VLA models.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 128,
        add_special_tokens: bool = True,
    ):
        """Initialize VLA tokenizer.

        Args:
            tokenizer: A HuggingFace tokenizer instance
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens (CLS, SEP, etc.)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Union[str, bool] = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Tokenize text(s).

        Args:
            text: Text or list of texts to tokenize
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return format ("np", "pt", "tf", "mlx")

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        encoded = self.tokenizer(
            text,
            padding=padding,
            max_length=self.max_length,
            truncation=truncation,
            add_special_tokens=self.add_special_tokens,
            return_tensors=return_tensors if return_tensors != "mlx" else "np",
            **kwargs,
        )

        # Convert to MLX format if requested
        if return_tensors == "mlx":
            import mlx.core as mx
            encoded = {
                "input_ids": mx.array(encoded["input_ids"]),
                "attention_mask": mx.array(encoded["attention_mask"]),
            }

        return encoded

    def decode(self, token_ids: Union[np.ndarray, List[int]], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids: Union[np.ndarray, List[List[int]]], **kwargs) -> List[str]:
        """Decode a batch of token IDs.

        Args:
            token_ids: Batch of token IDs
            **kwargs: Additional arguments for decode

        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        """Get beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_length: int = 128,
        **kwargs,
    ) -> "VLATokenizer":
        """Load a tokenizer from pretrained model.

        Args:
            pretrained_model_name_or_path: Model name or path (e.g., "llama-7b", "vicuna-7b")
            max_length: Maximum sequence length
            **kwargs: Additional arguments for AutoTokenizer

        Returns:
            VLATokenizer instance
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for loading pretrained tokenizers. "
                "Install with: pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(tokenizer, max_length=max_length)

    @classmethod
    def from_config(cls, config: dict) -> "VLATokenizer":
        """Create tokenizer from config dictionary.

        Args:
            config: Configuration dictionary with tokenizer settings

        Returns:
            VLATokenizer instance
        """
        tokenizer_name = config.get("tokenizer_name", "gpt2")
        max_length = config.get("max_length", 128)

        return cls.from_pretrained(tokenizer_name, max_length=max_length)


def create_tokenizer(
    model_name: str,
    max_length: int = 128,
) -> VLATokenizer:
    """Create a tokenizer for the given model.

    Args:
        model_name: Model name (e.g., "openvla-7b", "llava-1.5-7b")
        max_length: Maximum sequence length

    Returns:
        VLATokenizer instance
    """
    # Map model names to tokenizer names
    tokenizer_map = {
        "openvla-7b": "llama-7b",
        "openvla-3b": "llama-7b",
        "llava-1.5-7b": "llama-7b",
        "llava-1.5-13b": "llama2-13b",
    }

    tokenizer_name = tokenizer_map.get(model_name.lower(), "gpt2")

    try:
        return VLATokenizer.from_pretrained(tokenizer_name, max_length=max_length)
    except Exception:
        # Fall back to GPT-2 if model-specific tokenizer fails
        return VLATokenizer.from_pretrained("gpt2", max_length=max_length)


__all__ = ["VLATokenizer", "create_tokenizer"]
