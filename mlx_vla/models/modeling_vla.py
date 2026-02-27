import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass

from mlx_vla.models.vision import VisionEncoder
from mlx_vla.models.fusion import VLAMixer
from mlx_vla.models.action_heads import (
    DiscreteActionHead,
    DiffusionActionHead,
    ContinuousActionHead,
    ActionChunkingHead,
)

class VLAForAction(nn.Module):
    def __init__(
        self,
        vision_backbone: str = "clip",
        language_model: Optional[nn.Module] = None,
        vision_hidden_dim: int = 768,
        language_hidden_dim: int = 4096,
        fusion_type: str = "cross_attention",
        action_type: str = "discrete",
        action_dim: int = 7,
        action_horizon: int = 1,
        num_action_bins: int = 256,
        image_size: int = 224,
    ):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.action_type = action_type
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.image_size = image_size

        self.vision_encoder = VisionEncoder(
            backbone=vision_backbone,
            image_size=image_size,
            hidden_dim=vision_hidden_dim,
        )

        if language_model is not None:
            self.language_model = language_model
            self.language_hidden_dim = language_hidden_dim
        else:
            self.language_model = nn.Embedding(32000, language_hidden_dim)
            self.language_hidden_dim = language_hidden_dim

        hidden_dim = max(vision_hidden_dim, language_hidden_dim)
        self.fusion = VLAMixer(
            vision_dim=vision_hidden_dim,
            language_dim=language_hidden_dim,
            hidden_dim=hidden_dim,
            fusion_type=fusion_type,
        )

        if action_type == "discrete":
            self.action_head = DiscreteActionHead(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                num_bins=num_action_bins,
            )
        elif action_type == "diffusion":
            self.action_head = DiffusionActionHead(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                action_horizon=action_horizon,
            )
        elif action_type == "continuous":
            self.action_head = ContinuousActionHead(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                action_horizon=action_horizon,
            )
        elif action_type == "chunking":
            self.action_head = ActionChunkingHead(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                chunk_size=action_horizon,
            )
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        vision_embeds = self.vision_encoder(pixel_values)

        if input_ids is not None:
            language_embeds = self.language_model(input_ids)
        else:
            B = pixel_values.shape[0]
            language_embeds = mx.zeros((B, 1, self.language_hidden_dim))

        fused_embeds = self.fusion(vision_embeds, language_embeds)

        if self.action_type == "diffusion":
            actions = self.action_head.forward(fused_embeds)
            return {"action": actions, "hidden_states": fused_embeds}
        else:
            action_logits = self.action_head.forward(fused_embeds)
            return {"logits": action_logits, "hidden_states": fused_embeds}

    def predict_action(
        self,
        pixel_values: mx.array,
        input_ids: Optional[mx.array] = None,
        temperature: float = 1.0,
    ) -> mx.array:
        outputs = self(pixel_values, input_ids)

        if self.action_type == "discrete":
            logits = outputs["logits"]
            probs = mx.softmax(logits[:, -1, :, :] / temperature, axis=-1)
            bins = mx.argmax(probs, axis=-1)
            actions = self.action_head.tokens_to_action(bins)
        elif self.action_type == "diffusion":
            actions = self.action_head.denoise(outputs["hidden_states"])
        else:
            actions = outputs["action"]

        return actions.squeeze(1)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        action_type: str = "discrete",
        action_dim: int = 7,
        **kwargs,
    ) -> "VLAForAction":
        config = {
            "vision_backbone": kwargs.get("vision_backbone", "clip"),
            "vision_hidden_dim": kwargs.get("vision_hidden_dim", 768),
            "language_hidden_dim": kwargs.get("language_hidden_dim", 4096),
            "fusion_type": kwargs.get("fusion_type", "cross_attention"),
            "action_type": action_type,
            "action_dim": action_dim,
            "action_horizon": kwargs.get("action_horizon", 1),
            "num_action_bins": kwargs.get("num_action_bins", 256),
            "image_size": kwargs.get("image_size", 224),
        }

        return cls(**config)

    def save(self, path: str):
        import json
        path = path.rstrip("/")
        mx.savez(f"{path}/model.npz", **self.state_dict())

        config = {
            "vision_backbone": self.vision_backbone,
            "action_type": self.action_type,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "image_size": self.image_size,
        }
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str) -> "VLAForAction":
        import json
        path = path.rstrip("/")

        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_weights(f"{path}/model.npz")
        return model

VLA = VLAForAction