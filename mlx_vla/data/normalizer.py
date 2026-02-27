import numpy as np
from typing import Dict, Optional

class ActionNormalizer:
    DEFAULT_CONFIG = {
        "action_min": np.array([-150, -150, -150, -3.14, -3.14, -3.14, 0]),
        "action_max": np.array([150, 150, 150, 3.14, 3.14, 3.14, 1]),
    }

    ROBOT_CONFIGS = {
        "bridge_orig": {
            "action_min": np.array([-150, -150, -150, -3.14, -3.14, -3.14, 0]),
            "action_max": np.array([150, 150, 150, 3.14, 3.14, 3.14, 1]),
        },
        "widowx_250": {
            "action_min": np.array([-150, -150, -150, -3.14, -3.14, -3.14, 0]),
            "action_max": np.array([150, 150, 150, 3.14, 3.14, 3.14, 1]),
        },
        "franka": {
            "action_min": np.array([-0.5, -0.5, -0.5, -3.14, -3.14, -3.14, 0]),
            "action_max": np.array([0.5, 0.5, 0.5, 3.14, 3.14, 3.14, 1]),
        },
        "panda": {
            "action_min": np.array([-0.5, -0.5, -0.5, -3.14, -3.14, -3.14, 0]),
            "action_max": np.array([0.5, 0.5, 0.5, 3.14, 3.14, 3.14, 1]),
        },
        "kuka": {
            "action_min": np.array([-0.5, -0.5, -0.5, -3.14, -3.14, -3.14, -1]),
            "action_max": np.array([0.5, 0.5, 0.5, 3.14, 3.14, 3.14, 1]),
        },
    }

    def __init__(self, robot: str = "bridge_orig", action_dim: int = 7):
        self.robot = robot
        self.action_dim = action_dim
        self.config = self.ROBOT_CONFIGS.get(robot, self.DEFAULT_CONFIG)

        min_vals = self.config["action_min"]
        max_vals = self.config["action_max"]

        if len(min_vals) < action_dim:

            pad_len = action_dim - len(min_vals)
            min_vals = np.pad(min_vals, (0, pad_len), mode='edge')
            max_vals = np.pad(max_vals, (0, pad_len), mode='edge')
        elif len(min_vals) > action_dim:

            min_vals = min_vals[:action_dim]
            max_vals = max_vals[:action_dim]

        self.action_min = min_vals
        self.action_max = max_vals

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        actions = np.array(actions, dtype=np.float32)

        if actions.ndim == 0:
            actions = actions.reshape(1)

        if len(actions) != self.action_dim:

            if len(actions) < self.action_dim:

                padded = np.zeros(self.action_dim)
                padded[:len(actions)] = actions
                actions = padded
            else:

                actions = actions[:self.action_dim]

        range_vals = self.action_max - self.action_min
        range_vals = np.where(range_vals == 0, 1, range_vals)  

        normalized = (actions - self.action_min) / range_vals
        return np.clip(normalized, -1, 1)

    def unnormalize(self, normalized: np.ndarray) -> np.ndarray:
        normalized = np.array(normalized, dtype=np.float32)

        if normalized.ndim == 0:
            normalized = normalized.reshape(1)

        if len(normalized) != self.action_dim:
            if len(normalized) < self.action_dim:
                padded = np.zeros(self.action_dim)
                padded[:len(normalized)] = normalized
                normalized = padded
            else:
                normalized = normalized[:self.action_dim]

        range_vals = self.action_max - self.action_min
        range_vals = np.where(range_vals == 0, 1, range_vals)

        unnormalized = normalized * range_vals + self.action_min
        return np.clip(unnormalized, self.action_min, self.action_max)

    @staticmethod
    def from_model(model_name: str, action_dim: int = 7) -> "ActionNormalizer":
        model_robot_map = {
            "openvla": "bridge_orig",
            "bridge": "widowx_250",
            "octo": "bridge_orig",
            "rt-1": "panda",
            "rt-2": "panda",
        }

        for key, robot in model_robot_map.items():
            if key in model_name.lower():
                return ActionNormalizer(robot, action_dim)

        return ActionNormalizer("bridge_orig", action_dim)