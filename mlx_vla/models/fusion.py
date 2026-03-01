import mlx.core as mx
import mlx.nn as nn
from typing import Optional

class VLAMixer(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        fusion_type: str = "cross_attention",
    ):
        super().__init__()
        self.fusion_type = fusion_type

        if vision_dim != language_dim:
            self.vision_projection = nn.Linear(vision_dim, hidden_dim)
            self.language_projection = nn.Linear(language_dim, hidden_dim)
            self.vision_dim = hidden_dim
            self.language_dim = hidden_dim
        else:
            self.vision_projection = nn.Identity()
            self.language_projection = nn.Identity()
            self.vision_dim = vision_dim
            self.language_dim = language_dim

        if fusion_type == "cross_attention":
            self.fusion_layers = [
                CrossAttentionFusion(self.vision_dim, self.language_dim, num_heads)
                for _ in range(num_layers)
            ]
        elif fusion_type == "concat":
            self.concat_fusion = nn.Linear(self.vision_dim + self.language_dim, hidden_dim)
        elif fusion_type == "gated":
            self.gate = GatedFusion(self.vision_dim, self.language_dim, hidden_dim)
        elif fusion_type == "qkv_fusion":
            self.qkv_fusion = QKVFusion(self.vision_dim, self.language_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def __call__(
        self,
        vision_embeds: mx.array,
        language_embeds: mx.array,
    ) -> mx.array:
        vision_proj = self.vision_projection(vision_embeds)
        language_proj = self.language_projection(language_embeds)

        if self.fusion_type == "cross_attention":
            for layer in self.fusion_layers:
                vision_proj = layer(vision_proj, language_proj)
            return vision_proj
        elif self.fusion_type == "concat":
            # Handle different sequence lengths by using the minimum
            min_seq_len = min(vision_proj.shape[1], language_proj.shape[1])
            vision_proj = vision_proj[:, :min_seq_len, :]
            language_proj = language_proj[:, :min_seq_len, :]
            fused = mx.concatenate([vision_proj, language_proj], axis=-1)
            return self.concat_fusion(fused)
        elif self.fusion_type == "gated":
            return self.gate(vision_proj, language_proj)
        elif self.fusion_type == "qkv_fusion":
            return self.qkv_fusion(vision_proj, language_proj)
        return vision_proj

class CrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim: int, language_dim: int, num_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiHeadAttention(
            dims=vision_dim,
            num_heads=num_heads,
        )
        self.norm1 = nn.LayerNorm(vision_dim)
        self.norm2 = nn.LayerNorm(vision_dim)
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 4),
            nn.GELU(),
            nn.Linear(vision_dim * 4, vision_dim),
        )

    def __call__(self, vision: mx.array, language: mx.array) -> mx.array:
        attn_out = self.cross_attn(vision, language, language)
        vision = self.norm1(vision + attn_out)
        mlp_out = self.mlp(vision)
        vision = self.norm2(vision + mlp_out)
        return vision

class GatedFusion(nn.Module):
    def __init__(self, vision_dim: int, language_dim: int, hidden_dim: int):
        super().__init__()
        self.vision_gate = nn.Linear(language_dim, hidden_dim)
        self.language_gate = nn.Linear(vision_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

    def __call__(self, vision: mx.array, language: mx.array) -> mx.array:
        # Handle different sequence lengths
        min_seq_len = min(vision.shape[1], language.shape[1])
        vision = vision[:, :min_seq_len, :]
        language = language[:, :min_seq_len, :]

        gate_v = mx.sigmoid(self.vision_gate(language))
        gate_l = mx.sigmoid(self.language_gate(vision))
        vision_proj = self.vision_proj(vision)
        language_proj = self.language_proj(language)
        return vision_proj * gate_v + language_proj * gate_l

class QKVFusion(nn.Module):
    def __init__(self, vision_dim: int, language_dim: int, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(vision_dim, hidden_dim)
        self.k_proj = nn.Linear(language_dim, hidden_dim)
        self.v_proj = nn.Linear(language_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, vision: mx.array, language: mx.array) -> mx.array:
        # Handle different sequence lengths
        min_seq_len = min(vision.shape[1], language.shape[1])
        vision = vision[:, :min_seq_len, :]
        language = language[:, :min_seq_len, :]

        q = self.q_proj(vision)
        k = self.k_proj(language)
        v = self.v_proj(language)

        # Simple attention: (batch, seq, dim) x (batch, dim, seq) -> (batch, seq, seq)
        k_t = mx.transpose(k, (0, 2, 1))
        scores = mx.matmul(q, k_t) / (q.shape[-1] ** 0.5)
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)

        return self.norm(self.out_proj(out) + vision)