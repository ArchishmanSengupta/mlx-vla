import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import numpy as np

class DiscreteActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 7,
        num_bins: int = 256,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins

        self.action_head = nn.Linear(hidden_dim, num_bins * action_dim)
        self.vocab_size = vocab_size

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.forward(hidden_states)

    def forward(self, hidden_states: mx.array) -> mx.array:
        logits = self.action_head(hidden_states)
        return logits.reshape(hidden_states.shape[0], hidden_states.shape[1], self.action_dim, self.num_bins)

    def action_to_tokens(self, actions: mx.array) -> mx.array:
        bin_indices = ((actions + 1) / 2 * (self.num_bins - 1)).astype(mx.int32)
        return bin_indices

    def tokens_to_action(self, tokens: mx.array) -> mx.array:
        actions = tokens.astype(mx.float32) / (self.num_bins - 1) * 2 - 1
        return actions

class DiffusionActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 7,
        action_horizon: int = 4,
        num_diffusion_steps: int = 100,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_diffusion_steps = num_diffusion_steps

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim * action_horizon, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, action_dim * action_horizon),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        noisy_actions: Optional[mx.array] = None,
        timesteps: Optional[mx.array] = None,
    ) -> mx.array:
        return self.forward(hidden_states, noisy_actions, timesteps)

    def forward(
        self,
        hidden_states: mx.array,
        noisy_actions: Optional[mx.array] = None,
        timesteps: Optional[mx.array] = None,
    ) -> mx.array:
        if noisy_actions is None:
            noisy_actions = mx.random.normal((hidden_states.shape[0], self.action_horizon, self.action_dim))
        if timesteps is None:
            timesteps = mx.zeros((hidden_states.shape[0],))

        if len(hidden_states.shape) == 3:
            hidden = hidden_states[:, 0:1, :]  
        else:
            hidden = hidden_states

        t_emb = self.time_mlp(timesteps[:, None, None])
        hidden = hidden + t_emb

        hidden_flat = hidden.reshape(hidden.shape[0], -1)
        action_input = noisy_actions.reshape(hidden.shape[0], -1)
        combined = mx.concatenate([hidden_flat, action_input], axis=-1)

        return self.action_net(combined).reshape(-1, self.action_horizon, self.action_dim)

    def denoise(
        self,
        hidden_states: mx.array,
        num_steps: int = 10,
        sigma_min: float = 0.002,
    ) -> mx.array:
        """Denoise actions using DDPM-style sampling.

        This implements the reverse diffusion process to generate actions.
        """
        # Start from random noise
        actions = mx.random.normal((hidden_states.shape[0], self.action_horizon, self.action_dim))
        sigma_max = 1.0

        # Create timestep schedule
        step_schedule = mx.linspace(sigma_max, sigma_min, num_steps)

        for i, sigma in enumerate(step_schedule):
            t = mx.ones((hidden_states.shape[0],)) * sigma

            # Predict noise
            noise_pred = self.forward(hidden_states, actions, t)

            # Update actions using Euler integration
            if i < len(step_schedule) - 1:
                next_sigma = step_schedule[i + 1]
                # Euler step: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * noise_pred
                actions = actions + (next_sigma - sigma) * noise_pred
            else:
                # On final step, the noise prediction IS the denoised action
                actions = noise_pred

        return actions

class ContinuousActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 7,
        action_horizon: int = 1,
        num_layers: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        layers = []
        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = action_dim * action_horizon if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.forward(hidden_states)

    def forward(self, hidden_states: mx.array) -> mx.array:
        pooled = hidden_states[:, 0, :]
        return self.net(pooled).reshape(-1, self.action_horizon, self.action_dim)

class ActionChunkingHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 7,
        chunk_size: int = 100,
        num_layers: int = 4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        self.encoder = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=hidden_dim,
            num_heads=8,
            mlp_dims=hidden_dim * 4,
            dropout=0.0,
        )

        self.action_predictor = nn.Linear(hidden_dim, action_dim)

    def forward(self, hidden_states: mx.array) -> mx.array:
        # MLX TransformerEncoder requires mask argument
        encoded = self.encoder(hidden_states, mask=None)
        actions = self.action_predictor(encoded)
        return actions[:, :self.chunk_size, :]