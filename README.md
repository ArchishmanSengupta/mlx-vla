# mlx-vla

Vision-Language-Action training framework for Apple Silicon using MLX.

[![GitHub](https://img.shields.io/github/stars/ArchishmanSengupta/mlx-vla)](https://github.com/ArchishmanSengupta/mlx-vla)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

## Installation

### PyPI

```bash
pip install mlx-vla
```

### Homebrew

```bash
brew tap ArchishmanSengupta/mlx-vla
brew install mlx-vla
```

### From Source

```bash
git clone https://github.com/ArchishmanSengupta/mlx-vla.git
cd mlx-vla
pip install -e .
```

## Quick Start

```python
from mlx_vla import train_vla

train_vla(
    model="openvla-7b",
    dataset="bridge_v2",
    use_lora=True,
)
```

## CLI Usage

```bash
mlx-vla train --model openvla-7b --dataset bridge_v2 --use-lora
mlx-vla infer --model ./checkpoints/vla --image img.jpg --instruction "pick up cup"
```

## Features

- Vision-Language-Action model training on Apple Silicon
- LoRA/QLoRA fine-tuning for memory efficiency
- Multiple action types: discrete, diffusion, continuous, chunking
- Support for BridgeData V2, Open X-Embodiment, ALOHA datasets
- Action normalization for different robot platforms
- Clean, modular architecture

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX 0.18.0+

## Documentation

Full documentation available at [docs.mlx-vla.org](https://docs.mlx-vla.org)

## License

MIT License - see [LICENSE](LICENSE) for details
