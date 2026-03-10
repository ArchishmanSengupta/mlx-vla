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
    apply_to_all = target_modules is None or target_modules == "all"

    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if apply_to_all or any(tm in name for tm in target_modules):
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if isinstance(parent, list):
                parent = parent[int(part)]
            elif part.isdigit() and hasattr(parent, 'layers') and isinstance(getattr(parent, 'layers'), list):
                parent = parent.layers[int(part)]
            elif hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)

        final_key = parts[-1]
        lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)

        if isinstance(parent, list):
            parent[int(final_key)] = lora_layer
        elif final_key.isdigit() and hasattr(parent, 'layers') and isinstance(getattr(parent, 'layers'), list):
            parent.layers[int(final_key)] = lora_layer
        else:
            setattr(parent, final_key, lora_layer)

    return model

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
        lora_A_weight = module.lora_A.weight  # (rank, in_dim)
        lora_B_weight = module.lora_B.weight  # (out_dim, rank)
        lora_contribution = mx.matmul(lora_B_weight, lora_A_weight) * module.scaling
        merged_weight = module.base_layer.weight + lora_contribution

        module.base_layer.weight = merged_weight
        module.lora_A = None
        module.lora_B = None

    return model