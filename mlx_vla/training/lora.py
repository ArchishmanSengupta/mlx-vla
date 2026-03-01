import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Dict

class LoRALayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha

        # Handle rank=0 (no LoRA, just use base layer)
        if rank <= 0:
            self.scaling = 0.0
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()
            return

        self.scaling = alpha / rank

        # Note: MLX doesn't support trainable flag directly
        # The base layer parameters are not updated when training LoRA

        in_dim = base_layer.weight.shape[1]
        out_dim = base_layer.weight.shape[0]

        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        self.lora_A.weight = mx.zeros_like(self.lora_A.weight)
        self.lora_B.weight = mx.zeros_like(self.lora_B.weight)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        base_output = self.base_layer(x)

        # Handle rank=0 (no LoRA)
        if self.rank <= 0 or self.lora_A is None or self.lora_B is None:
            return base_output

        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return base_output + lora_output * self.scaling

def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    # If target_modules is "all" or None, apply to all Linear layers
    apply_to_all = target_modules is None or target_modules == "all"

    def replace_layer(name: str, module: nn.Module) -> nn.Module:
        if not isinstance(module, nn.Linear):
            return module

        # Apply to all Linear layers if target_modules is "all" or None
        if apply_to_all:
            return LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)

        # Otherwise, check if any target module name is in the layer name
        if any(tm in name for tm in target_modules):
            return LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
        return module

    def traverse_and_replace(model: nn.Module, prefix: str = "") -> nn.Module:

        # Handle Sequential which has layers as a list
        if hasattr(model, 'layers') and isinstance(model.layers, list):
            for i, module in enumerate(model.layers):
                full_name = f"{prefix}.layers.{i}" if prefix else f"layers.{i}"

                if isinstance(module, nn.Linear):
                    new_module = replace_layer(full_name, module)
                    if new_module is not module:
                        model.layers[i] = new_module
                    # Don't recurse into LoRA layers
                elif hasattr(module, 'children'):
                    traverse_and_replace(module, full_name)
        else:
            # For other modules, iterate through children
            for module in model.children():
                name = None
                for attr_name, attr_val in vars(model).items():
                    if attr_val is module:
                        name = attr_name
                        break
                if name is None:
                    continue

                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(module, nn.Module):
                    new_module = replace_layer(full_name, module)
                    if new_module is not module:
                        setattr(model, name, new_module)
                    traverse_and_replace(new_module, full_name)

        return model

    return traverse_and_replace(model)

def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA weights back into the base layer."""
    # Collect LoRA modules with their names first
    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_modules.append((name, module))

    for name, module in lora_modules:
        if module.lora_A is None or module.lora_B is None:
            continue
        # Compute merged weights: W + B @ A * scaling
        # lora_A: (in_dim, rank), lora_B: (out_dim, rank)
        # lora_B.weight: (out_dim, rank), lora_A.weight: (in_dim, rank)
        # B @ A: (out_dim, rank) @ (rank, in_dim) = (out_dim, in_dim)
        lora_A_weight = module.lora_A.weight  # (in_dim, rank)
        lora_B_weight = module.lora_B.weight  # (out_dim, rank)
        # Need to transpose A for proper matrix multiplication
        lora_contribution = mx.matmul(mx.transpose(lora_A_weight), mx.transpose(lora_B_weight)) * module.scaling
        merged_weight = module.base_layer.weight + lora_contribution

        # Update base layer weights
        module.base_layer.weight = merged_weight
        # Remove LoRA layers
        module.lora_A = None
        module.lora_B = None

    return model