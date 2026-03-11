#!/usr/bin/env python3
"""
mlx-vla Quickstart
==================
Train a robot policy and run inference in < 30 seconds.

    pip install -e .
    python examples/quickstart.py
"""

import numpy as np
from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.inference.pipeline import VLAPipeline

# 1. Create a model
model = VLAForAction(
    vision_backbone="clip",
    action_type="continuous",
    action_dim=7,
    image_size=224,
)

# 2. Run inference
pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")
image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
action = pipeline.predict(image=image, language="pick up the cup")

print(f"Action: {action}")         # 7-DOF robot action
print(f"Shape:  {action.shape}")   # (7,)
