# mlx-vla

Vision-Language-Action training framework for Apple Silicon using MLX.

[![GitHub](https://img.shields.io/github/stars/ArchishmanSengupta/mlx-vla)](https://github.com/ArchishmanSengupta/mlx-vla)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

Train and deploy Vision-Language-Action models natively on Apple Silicon. mlx-vla provides a config-driven, modular architecture with support for multiple vision encoders, fusion strategies, action head types, and robot platforms.

## Installation

### From Source

```bash
git clone https://github.com/ArchishmanSengupta/mlx-vla.git
cd mlx-vla
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[all]"        # dev + training + data extras
pip install -e ".[training]"   # mlx-lm for full LLM support
pip install -e ".[data]"       # h5py for HDF5 datasets
pip install -e ".[dev]"        # pytest, black, ruff, mypy
```

### PyPI

```bash
pip install mlx-vla
```

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX >= 0.18.0

## Quick Start

### One-Line Training (Python API)

```python
from mlx_vla.train_vla import train_vla

trainer = train_vla(
    model="openvla-7b",
    dataset="bridge_v2",
    use_lora=True,
    output_dir="./output",
    num_epochs=3,
    batch_size=1,
    learning_rate=1e-4,
)
trainer.train()
```

### One-Line Inference

```python
from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.inference.pipeline import VLAPipeline

model = VLAForAction(
    vision_backbone="clip",
    action_type="continuous",
    action_dim=7,
    image_size=224,
)

pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")
action = pipeline.predict(image="camera_frame.jpg", language="pick up the red block")
# action shape: (7,) -- unnormalized robot action
```

### CLI

```bash
# Train with a YAML config
mlx-vla train --config examples/train_lora.yaml

# Train with CLI args
mlx-vla train --model openvla-7b --dataset bridge_v2 --use-lora --epochs 5

# Run inference
mlx-vla infer --model ./my_model --image img.jpg --instruction "pick up the cup"

# Generate a default config file
mlx-vla create-config --output my_config.yaml

# Export a model
mlx-vla export --model ./my_model --output ./exported_model
```

## Architecture

```
mlx_vla/
  models/
    vision.py          # Vision encoders (CLIP, DINOv2, SigLIP, SAM)
    language.py         # Language model wrappers (MLX-LM, embedding fallback)
    fusion.py           # Vision-language fusion (cross_attention, concat, gated, qkv)
    action_heads.py     # Action heads (discrete, continuous, diffusion, chunking)
    modeling_vla.py     # VLAForAction -- main model class
  training/
    trainer.py          # VLATrainer with gradient accumulation, eval, callbacks
    lora.py             # LoRA apply / merge
    optimizers.py       # AdamW, Adam, SGD, RMSprop, Lion + cosine/linear schedulers
    callbacks.py        # Checkpoint, logging, early stopping callbacks
  data/
    dataset.py          # VLADataset, RLDSDataset, BridgeDataset, EpisodeDataset
    collator.py         # Batch collation with image preprocessing + action normalization
    dataloader.py       # VLADataloader with shuffling and batching
    normalizer.py       # ActionNormalizer for different robot platforms
    download.py         # Dataset download utilities
  inference/
    pipeline.py         # VLAPipeline -- image-in, action-out
  utils/
    config.py           # Config system (YAML/JSON, dataclasses, defaults)
    pretrained.py       # Known model configs (openvla-7b, octo-base, etc.)
  cli/
    main.py             # CLI entry point (train, infer, export, create-config)
  core/
    __init__.py         # VLATrainingArguments
  train_vla.py          # High-level train_vla() convenience function
```

## Configuration

Everything is config-driven. All defaults live in `mlx_vla/utils/config.py` via `DEFAULT_CONFIG` and typed dataclasses (`ModelConfig`, `TrainingConfig`, `LoRAConfig`, etc.).

### YAML Config

```yaml
model:
  name: openvla-7b
  vision_backbone: clip       # clip | dinov2 | siglip | sam
  vision_hidden_dim: 768
  language_hidden_dim: 4096
  fusion_type: cross_attention # cross_attention | concat | gated | qkv_fusion
  action_type: discrete        # discrete | continuous | diffusion | chunking
  action_dim: 7
  action_horizon: 1
  num_action_bins: 256

lora:
  enabled: true
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - query_proj
    - key_proj
    - value_proj
    - out_proj

data:
  dataset_name: bridge_v2
  image_size: 224
  batch_size: 1
  action_normalization: clip_minus_one_to_one

training:
  epochs: 3
  learning_rate: 1.0e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  gradient_accumulation_steps: 8
  lr_scheduler_type: cosine    # cosine | linear | constant
  eval_strategy: "no"          # no | epoch | steps
  eval_steps: 500

checkpointing:
  output_dir: ./vla_output
  save_steps: 500
  save_total_limit: 3
  save_strategy: epoch

logging:
  logging_steps: 10
  report_to:
    - tensorboard
```

Use with CLI:
```bash
mlx-vla train --config my_config.yaml
```

Or load in Python:
```python
from mlx_vla.utils.config import load_config

cfg = load_config("my_config.yaml")
print(cfg.model.vision_backbone)  # "clip"
```

### CLI Overrides

CLI args override config values:

```bash
mlx-vla train --config examples/train_lora.yaml \
    --epochs 10 \
    --learning-rate 5e-5 \
    --batch-size 2 \
    --use-lora \
    --action-type continuous \
    --output-dir ./my_output
```

## Model Components

### Vision Encoders

| Encoder | `vision_backbone` | Description |
|---------|-------------------|-------------|
| CLIP    | `clip`            | OpenAI CLIP ViT |
| DINOv2  | `dinov2`          | Meta DINOv2 ViT |
| SigLIP  | `siglip`          | Google SigLIP ViT |
| SAM     | `sam`             | Meta SAM ViT |

All encoders are parameterized (`patch_size`, `num_layers`, `num_heads`, `mlp_ratio`) and accept NCHW input.

### Fusion Types

| Fusion | `fusion_type` | Description |
|--------|---------------|-------------|
| Cross-Attention | `cross_attention` | Vision attends to language via multi-head cross-attention |
| Concat | `concat` | Concatenate + linear projection |
| Gated | `gated` | Learned gating between vision and language |
| QKV | `qkv_fusion` | Vision as query, language as key/value |

### Action Heads

| Head | `action_type` | Output | Use Case |
|------|---------------|--------|----------|
| Discrete | `discrete` | Binned action tokens | OpenVLA-style, discretized actions |
| Continuous | `continuous` | Direct regression | MSE-based continuous control |
| Diffusion | `diffusion` | Denoised action trajectory | Diffusion policy, multi-step |
| Chunking | `chunking` | Transformer-decoded action chunk | ACT-style action chunking |

`diffusion` and `chunking` support `action_horizon > 1` for multi-step prediction.

### Known Model Configs

| Model | Vision | Fusion | Action | Notes |
|-------|--------|--------|--------|-------|
| `openvla-7b` | CLIP (1024) | cross_attention | discrete | LLaMA-7B backbone |
| `openvla-3b` | CLIP (768) | cross_attention | discrete | Smaller variant |
| `llava-1.5-7b` | CLIP (1024) | concat | continuous | LLaVA-based |
| `llava-1.5-13b` | CLIP (1024) | concat | continuous | LLaVA 13B |
| `octo-small` | DINOv2 (384) | concat | diffusion | 4-step horizon |
| `octo-base` | DINOv2 (768) | concat | diffusion | 4-step horizon |

## Usage Examples

### Training on Local Data

Prepare a directory with JSON episode files:

```
my_data/
  episode_0.json
  episode_1.json
  img_0.png
  img_1.png
```

Each JSON file is a list of steps:
```json
[
  {"image": "img_0.png", "action": [0.1, -0.2, 0.0, 0.5, 0.0, 0.0, 1.0], "language": "pick up the cup"},
  {"image": "img_1.png", "action": [0.2, -0.1, 0.1, 0.4, 0.0, 0.0, 1.0], "language": "pick up the cup"}
]
```

```python
from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.data.dataset import EpisodeDataset
from mlx_vla.training.trainer import VLATrainer
from mlx_vla.core import VLATrainingArguments

model = VLAForAction(
    vision_backbone="clip",
    action_type="continuous",
    action_dim=7,
    image_size=224,
)

dataset = EpisodeDataset("./my_data", split="train")

args = VLATrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    save_steps=100,
)

trainer = VLATrainer(model=model, args=args, train_dataset=dataset)
trainer.train()
model.save("./my_trained_model")
```

### LoRA Fine-Tuning

```python
from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.training.lora import apply_lora, merge_lora

model = VLAForAction(action_type="discrete", action_dim=7)

# Apply LoRA adapters to attention layers
model = apply_lora(
    model,
    rank=8,
    alpha=16,
    dropout=0.05,
    target_modules=["query_proj", "value_proj"],
)

# ... train the model ...

# Merge LoRA weights back for deployment (no overhead at inference)
merged = merge_lora(model)
merged.save("./deployed_model")
```

### Diffusion Policy

```python
model = VLAForAction(
    vision_backbone="dinov2",
    action_type="diffusion",
    action_dim=7,
    action_horizon=4,   # predict 4 future timesteps
    image_size=224,
)

# Predict via denoising
actions = model.predict_action(pixel_values, input_ids)
# shape: (batch, 4, 7) -- 4-step action trajectory
```

Or via YAML:
```bash
mlx-vla train --config examples/train_diffusion.yaml
```

### Streaming Actions (Robot Control Loop)

```python
from mlx_vla.inference.pipeline import VLAPipeline

pipeline = VLAPipeline(model="./my_model", unnorm_key="franka")

def camera_stream():
    # yield RGB numpy arrays from your camera
    ...

for action in pipeline.stream_actions(camera_stream(), language="stack the cups"):
    robot.send(action)  # 7-DOF action
```

### Save and Load

```python
model.save("./my_model")                    # saves model.npz + config.json
loaded = VLAForAction.load("./my_model")     # restores architecture + weights
```

### Robot Platforms

The pipeline unnormalizes predicted actions to your robot's workspace:

```python
pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")  # WidowX/Bridge
pipeline = VLAPipeline(model=model, unnorm_key="franka")       # Franka Emika
pipeline = VLAPipeline(model=model, unnorm_key="panda")        # Panda
pipeline = VLAPipeline(model=model, unnorm_key="kuka")         # Kuka
```

| Robot | `unnorm_key` | Workspace (XYZ) | Gripper |
|-------|-------------|-----------------|---------|
| Bridge/WidowX | `bridge_orig`, `widowx_250` | +/-150mm | 0-1 |
| Franka | `franka` | +/-0.5m | 0-1 |
| Panda | `panda` | +/-0.5m | 0-1 |
| Kuka | `kuka` | +/-0.5m | -1 to 1 |

### Datasets

**RLDS datasets** (require `tensorflow_datasets`):
- `bridge_v2`
- `oxe/bridge_v2`
- `oxe/rx1`
- `oxe/franka_kitchen`
- `oxe/taco`
- `aloha`

**Local formats**:
- Directory of JSON episode files
- Single JSON file
- HDF5 files (require `h5py`)

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

258 tests covering all model types, action heads, vision encoders, fusion types, LoRA, save/load, config, pipeline, data loading, training, callbacks, and schedulers.

## Example Configs

- [`examples/train_lora.yaml`](examples/train_lora.yaml) -- LoRA fine-tuning with discrete actions
- [`examples/train_diffusion.yaml`](examples/train_diffusion.yaml) -- Diffusion policy with DINOv2

## License

MIT License - see [LICENSE](LICENSE) for details.
