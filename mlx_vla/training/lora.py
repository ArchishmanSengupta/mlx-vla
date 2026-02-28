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
        self.scaling = alpha / rank

        # Freeze the base layer parameters
        for p in base_layer.parameters():
            p.trainable = False

        in_dim = base_layer.weight.shape[1]
        out_dim = base_layer.weight.shape[0]

        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        self.lora_A.weight = mx.zeros_like(self.lora_A.weight)
        self.lora_B.weight = mx.zeros_like(self.lora_B.weight)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        base_output = self.base_layer(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return base_output + lora_output * self.scaling

def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    def replace_layer(name: str, module: nn.Module) -> nn.Module:
        if not isinstance(module, nn.Linear):
            return module

        if any(tm in name for tm in target_modules):
            return LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
        return module

    def traverse_and_replace(model: nn.Module, prefix: str = "") -> nn.Module:

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
        lora_A = module.lora_A.weight
        lora_B = module.lora_B.weight
        merged_weight = module.base_layer.weight + module.lora_B(module.lora_A.weight) * module.scaling

        # Unfreeze base layer and update weights
        for p in module.base_layer.parameters():
            p.trainable = True
        module.base_layer.weight = merged_weight
        if module.base_layer.bias is not None:
            module.base_layer.bias = module.base_layer.bias

    return model