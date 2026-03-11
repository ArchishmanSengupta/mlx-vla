# mlx-vla

Train and deploy Vision-Language-Action models natively on Apple Silicon.

[![GitHub Stars](https://img.shields.io/github/stars/ArchishmanSengupta/mlx-vla?style=flat-square)](https://github.com/ArchishmanSengupta/mlx-vla)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-%E2%89%A50.18.0-orange.svg?style=flat-square)](https://github.com/ml-explore/mlx)
[![Tests](https://img.shields.io/badge/tests-281%20passing-brightgreen.svg?style=flat-square)](#testing)

> **6 lines of Python. Image in, robot action out. Runs entirely on your Mac.**

```python
from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.inference.pipeline import VLAPipeline

model = VLAForAction(vision_backbone="clip", action_type="continuous", action_dim=7)
pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")
action = pipeline.predict(image="camera.jpg", language="pick up the cup")
# action: (7,) -- [x, y, z, rx, ry, rz, gripper]
```

## Performance (Apple M4, 16GB)

| | Inference | Training | Throughput |
|---|---|---|---|
| CLIP + Continuous | **1.9 ms** | 17.5 ms/step | 847 img/s |
| DINOv2 + Diffusion | 2.7 ms | 24.9 ms/step | 626 img/s |
| SigLIP + Discrete | 2.2 ms | 25.0 ms/step | 746 img/s |
| SAM + Chunking | **1.5 ms** | 17.5 ms/step | 1530 img/s |

Run `python examples/benchmark.py` to reproduce on your hardware.

## Installation

```bash
git clone https://github.com/ArchishmanSengupta/mlx-vla.git
cd mlx-vla
pip install -e .
```

Optional extras:

```bash
pip install -e ".[all]"        # everything (dev + training + data)
pip install -e ".[training]"   # mlx-lm for full LLM backbones
pip install -e ".[data]"       # h5py for HDF5 datasets
pip install -e ".[dev]"        # pytest, black, ruff, mypy
```

Requirements: Python 3.10+, Apple Silicon Mac (M1/M2/M3/M4), MLX >= 0.18.0

## Quick Start

### Quickstart Script

```bash
python examples/quickstart.py
```

6 lines -- creates a model, runs inference, prints a 7-DOF action. Under 1 second.

### Train a Robot Policy

```python
from mlx_vla.train_vla import train_vla

trainer = train_vla(
    model="openvla-7b",
    dataset="bridge_v2",
    use_lora=True,
    output_dir="./output",
    num_epochs=3,
)
trainer.train()
```

### CLI

```bash
# Train with YAML config
mlx-vla train --config examples/train_lora.yaml

# Train with CLI flags
mlx-vla train --model openvla-7b --dataset bridge_v2 --use-lora --epochs 5

# Inference
mlx-vla infer --model ./my_model --image img.jpg --instruction "pick up the cup"

# Generate a default config
mlx-vla create-config --output config.yaml
```

### Full Demo

```bash
python examples/demo.py
```

End-to-end: creates dataset, trains model, saves, loads, runs inference, streams actions. Takes ~2 seconds.

## What Can You Configure?

| Component | Options | Default |
|---|---|---|
| Vision encoder | `clip`, `dinov2`, `siglip`, `sam` | `clip` |
| Fusion | `cross_attention`, `concat`, `gated`, `qkv_fusion` | `cross_attention` |
| Action head | `discrete`, `continuous`, `diffusion`, `chunking` | `discrete` |
| Action dim | any int | `7` |
| Action horizon | any int (multi-step for diffusion/chunking) | `1` |
| Robot platform | `bridge_orig`, `widowx_250`, `franka`, `panda`, `kuka` | `bridge_orig` |

Everything is config-driven via YAML or Python dataclasses. See [`examples/train_lora.yaml`](examples/train_lora.yaml) for a full example.

## Usage Examples

### Train on Your Own Data

Prepare a directory with JSON episode files:

```
my_data/
  episode_0.json    # [{"image": "img_0.png", "action": [0.1, ...], "language": "pick up cup"}, ...]
  img_0.png
```

```python
from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.data.dataset import EpisodeDataset
from mlx_vla.training.trainer import VLATrainer
from mlx_vla.core import VLATrainingArguments

model = VLAForAction(vision_backbone="clip", action_type="continuous", action_dim=7, image_size=224)
dataset = EpisodeDataset("./my_data", split="train")
args = VLATrainingArguments(output_dir="./output", num_train_epochs=10, learning_rate=1e-4)

trainer = VLATrainer(model=model, args=args, train_dataset=dataset)
trainer.train()
model.save("./my_model")
```

### LoRA Fine-Tuning

```python
from mlx_vla.training.lora import apply_lora, merge_lora

model = VLAForAction(action_type="discrete", action_dim=7)
model = apply_lora(model, rank=8, alpha=16, target_modules=["query_proj", "value_proj"])

# ... train ...

merged = merge_lora(model)  # merge weights for zero-overhead inference
merged.save("./deployed_model")
```

### Diffusion Policy (Multi-Step)

```python
model = VLAForAction(
    vision_backbone="dinov2",
    action_type="diffusion",
    action_dim=7,
    action_horizon=4,   # predict 4 future timesteps
)

actions = model.predict_action(pixel_values, input_ids)
# shape: (batch, 4, 7) -- 4-step action trajectory
```

### Streaming Robot Control

```python
from mlx_vla.inference.pipeline import VLAPipeline

pipeline = VLAPipeline(model="./my_model", unnorm_key="franka")

for action in pipeline.stream_actions(camera_stream(), language="stack the cups"):
    robot.send(action)  # 7-DOF action at each frame
```

### Save and Load

```python
model.save("./my_model")                 # saves config.json + model.npz
loaded = VLAForAction.load("./my_model")  # restores architecture + weights
```

## Architecture

```
mlx_vla/
  models/
    vision.py          # CLIP, DINOv2, SigLIP, SAM encoders
    language.py         # Language model wrappers (MLX-LM, embedding fallback)
    fusion.py           # Vision-language fusion (4 strategies)
    action_heads.py     # Action heads (discrete, continuous, diffusion, chunking)
    modeling_vla.py     # VLAForAction -- the main model
  training/
    trainer.py          # VLATrainer (gradient accumulation, eval, callbacks)
    lora.py             # LoRA apply / merge
    optimizers.py       # AdamW, Adam, SGD, RMSprop, Lion + schedulers
    callbacks.py        # Checkpoint, logging, early stopping
  data/
    dataset.py          # VLADataset, RLDSDataset, BridgeDataset, EpisodeDataset
    collator.py         # Batch collation, image preprocessing, action normalization
    dataloader.py       # Batching and shuffling
    normalizer.py       # Per-robot action unnormalization
  inference/
    pipeline.py         # VLAPipeline -- image in, action out
  utils/
    config.py           # YAML/JSON config system with typed dataclasses
  cli/
    main.py             # CLI: train, infer, export, create-config
```

### Known Model Configs

| Model | Vision | Fusion | Action | Notes |
|-------|--------|--------|--------|-------|
| `openvla-7b` | CLIP (1024) | cross_attention | discrete | LLaMA-7B backbone |
| `openvla-3b` | CLIP (768) | cross_attention | discrete | Smaller variant |
| `llava-1.5-7b` | CLIP (1024) | concat | continuous | LLaVA-based |
| `octo-small` | DINOv2 (384) | concat | diffusion | 4-step horizon |
| `octo-base` | DINOv2 (768) | concat | diffusion | 4-step horizon |

### Supported Datasets

**RLDS** (require `tensorflow_datasets`): `bridge_v2`, `oxe/bridge_v2`, `oxe/rx1`, `oxe/franka_kitchen`, `oxe/taco`, `aloha`

**Local**: directory of JSON episodes, single JSON file, HDF5

### Robot Platforms

| Robot | `unnorm_key` | Workspace | Gripper |
|-------|-------------|-----------|---------|
| Bridge/WidowX | `bridge_orig`, `widowx_250` | +/-150mm | 0-1 |
| Franka | `franka` | +/-0.5m | 0-1 |
| Panda | `panda` | +/-0.5m | 0-1 |
| Kuka | `kuka` | +/-0.5m | -1 to 1 |

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

281 tests covering models, training, inference, LoRA, data loading, config, edge cases, and all component combinations.

## Examples

| File | Description |
|------|-------------|
| [`examples/quickstart.py`](examples/quickstart.py) | 6-line inference demo |
| [`examples/demo.py`](examples/demo.py) | Full end-to-end demo (dataset + train + inference + streaming) |
| [`examples/benchmark.py`](examples/benchmark.py) | Performance benchmarks across all configurations |
| [`examples/train_lora.yaml`](examples/train_lora.yaml) | YAML config for LoRA fine-tuning |
| [`examples/train_diffusion.yaml`](examples/train_diffusion.yaml) | YAML config for diffusion policy |

## License

MIT License - see [LICENSE](LICENSE) for details.
