import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import numpy as np
import warnings

class VisionEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "clip",
        image_size: int = 224,
        hidden_dim: int = 768,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.pretrained = pretrained

        if backbone == "clip":
            self.encoder = CLIPVisionEncoder(hidden_dim, pretrained, image_size)
        elif backbone == "dinov2":
            self.encoder = DINOv2Encoder(hidden_dim, pretrained, image_size)
        elif backbone == "siglip":
            self.encoder = SigLIPEncoder(hidden_dim, pretrained, image_size)
        elif backbone == "sam":
            self.encoder = SAMVisionEncoder(hidden_dim, pretrained, image_size)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if pretrained:
            warnings.warn(
                f"VisionEncoder: pretrained={pretrained} is set, but actual pretrained "
                f"weights are not loaded. The model is initialized randomly. "
                f"To load pretrained weights, use mlx_vla.models.load_pretrained_vision_encoder().",
                UserWarning,
            )

    def __call__(self, images: mx.array) -> mx.array:
        return self.encoder(images)

class CLIPVisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, pretrained: bool, image_size: int = 224,
                 patch_size: int = 14, num_layers: int = 12, num_heads: int = 12,
                 mlp_ratio: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        while hidden_dim % num_heads != 0:
            num_heads -= 1

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches + 1, hidden_dim)
        self.cls_token = nn.Embedding(1, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=hidden_dim,
            num_heads=num_heads,
            mlp_dims=hidden_dim * mlp_ratio,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, images: mx.array) -> mx.array:

        x = mx.transpose(images, (0, 2, 3, 1))
        x = self.patch_embed(x)

        x = mx.transpose(x, (0, 3, 1, 2))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x.transpose(0, 2, 1)

        cls_token = self.cls_token(mx.zeros((B, 1), dtype=mx.int32))
        x = mx.concatenate([cls_token, x], axis=1)

        num_positions = x.shape[1]
        positions = mx.arange(num_positions, dtype=mx.int32)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb

        x = self.transformer(x, mask=None)
        x = self.norm(x)
        return x

class DINOv2Encoder(nn.Module):
    def __init__(self, hidden_dim: int, pretrained: bool, image_size: int = 224,
                 patch_size: int = 14, num_layers: int = 24, num_heads: int = 16,
                 mlp_ratio: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        while hidden_dim % num_heads != 0:
            num_heads -= 1

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches + 1, hidden_dim)
        self.cls_token = nn.Embedding(1, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=hidden_dim,
            num_heads=num_heads,
            mlp_dims=hidden_dim * mlp_ratio,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, images: mx.array) -> mx.array:

        x = mx.transpose(images, (0, 2, 3, 1))
        x = self.patch_embed(x)

        x = mx.transpose(x, (0, 3, 1, 2))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x.transpose(0, 2, 1)

        cls_token = self.cls_token(mx.zeros((B, 1), dtype=mx.int32))
        x = mx.concatenate([cls_token, x], axis=1)

        num_positions = x.shape[1]
        positions = mx.arange(num_positions, dtype=mx.int32)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb

        x = self.transformer(x, mask=None)
        x = self.norm(x)
        return x

class SigLIPEncoder(nn.Module):
    def __init__(self, hidden_dim: int, pretrained: bool, image_size: int = 224,
                 patch_size: int = 14, num_layers: int = 24, num_heads: int = 16,
                 mlp_ratio: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        while hidden_dim % num_heads != 0:
            num_heads -= 1

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches + 1, hidden_dim)
        self.cls_token = nn.Embedding(1, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=hidden_dim,
            num_heads=num_heads,
            mlp_dims=hidden_dim * mlp_ratio,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, images: mx.array) -> mx.array:

        x = mx.transpose(images, (0, 2, 3, 1))
        x = self.patch_embed(x)

        x = mx.transpose(x, (0, 3, 1, 2))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x.transpose(0, 2, 1)

        cls_token = self.cls_token(mx.zeros((B, 1), dtype=mx.int32))
        x = mx.concatenate([cls_token, x], axis=1)

        num_positions = x.shape[1]
        positions = mx.arange(num_positions, dtype=mx.int32)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb

        x = self.transformer(x, mask=None)
        x = self.norm(x)
        return x

class SAMVisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, pretrained: bool, image_size: int = 224,
                 patch_size: int = 16, num_layers: int = 12, num_heads: int = 12,
                 mlp_ratio: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        while hidden_dim % num_heads != 0:
            num_heads -= 1

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches + 1, hidden_dim)
        self.cls_token = nn.Embedding(1, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=hidden_dim,
            num_heads=num_heads,
            mlp_dims=hidden_dim * mlp_ratio,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, images: mx.array) -> mx.array:

        x = mx.transpose(images, (0, 2, 3, 1))
        x = self.patch_embed(x)

        x = mx.transpose(x, (0, 3, 1, 2))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x.transpose(0, 2, 1)

        cls_token = self.cls_token(mx.zeros((B, 1), dtype=mx.int32))
        x = mx.concatenate([cls_token, x], axis=1)

        num_positions = x.shape[1]
        positions = mx.arange(num_positions, dtype=mx.int32)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb

        x = self.transformer(x, mask=None)
        x = self.norm(x)
        return x