#!/usr/bin/env python3
"""
mlx-vla Benchmark
=================
Measures training throughput, inference latency, and memory usage
across all model configurations on Apple Silicon.

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --full    # extended benchmark with larger models
"""

import time
import sys
import os
import platform
import argparse
import numpy as np

import mlx.core as mx

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def get_system_info():
    chip = "Apple Silicon"
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            chip = result.stdout.strip()
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        ram_gb = int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        ram_gb = 0

    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "device": str(mx.default_device()),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "mlx": mx.__version__,
    }


def count_params(tree):
    total = 0
    if isinstance(tree, dict):
        for v in tree.values():
            total += count_params(v)
    elif isinstance(tree, list):
        for v in tree:
            total += count_params(v)
    elif isinstance(tree, mx.array):
        total += tree.size
    return total


def model_size_mb(model):
    total_bytes = 0
    params = model.parameters()
    def walk(tree):
        nonlocal total_bytes
        if isinstance(tree, dict):
            for v in tree.values():
                walk(v)
        elif isinstance(tree, list):
            for v in tree:
                walk(v)
        elif isinstance(tree, mx.array):
            total_bytes += tree.size * tree.dtype.size
    walk(params)
    return total_bytes / (1024 * 1024)


def benchmark_forward(model, image_size, batch_size, n_warmup=3, n_iter=20):
    images = mx.random.normal((batch_size, 3, image_size, image_size))
    ids = mx.zeros((batch_size, 8), dtype=mx.int32)

    for _ in range(n_warmup):
        out = model(images, ids)
        mx.eval(out["hidden_states"])

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = model(images, ids)
        mx.eval(out["hidden_states"])
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "throughput": batch_size / np.mean(times),
    }


def benchmark_inference(model, image_size, n_warmup=3, n_iter=50):
    from mlx_vla.inference.pipeline import VLAPipeline

    pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")
    img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    for _ in range(n_warmup):
        pipeline.predict(image=img, language="test")

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        action = pipeline.predict(image=img, language="pick up the block")
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "p99_ms": np.percentile(times, 99) * 1000,
    }


def benchmark_training(model, image_size, batch_size, n_steps=20):
    import mlx.nn as nn
    import mlx.optimizers as optim

    optimizer = optim.AdamW(learning_rate=1e-4)

    images = mx.random.normal((batch_size, 3, image_size, image_size))
    ids = mx.zeros((batch_size, 8), dtype=mx.int32)
    targets = mx.random.normal((batch_size, 7))

    def loss_fn(model, images, ids, targets):
        out = model(images, ids)
        pred = out.get("action", out.get("logits"))
        if pred.ndim == 4:
            pred = pred[:, -1, :, 0]
        if pred.ndim == 3:
            pred = pred[:, 0, :]
        min_dim = min(pred.shape[-1], targets.shape[-1])
        return mx.mean((pred[..., :min_dim] - targets[..., :min_dim]) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # warmup
    for _ in range(3):
        loss, grads = loss_and_grad(model, images, ids, targets)
        optimizer.update(model, grads)
        mx.eval(loss)

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        loss, grads = loss_and_grad(model, images, ids, targets)
        optimizer.update(model, grads)
        mx.eval(loss)
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "samples_per_sec": batch_size / np.mean(times),
        "final_loss": float(loss),
    }


def main():
    parser = argparse.ArgumentParser(description="mlx-vla benchmark")
    parser.add_argument("--full", action="store_true", help="Run extended benchmarks with larger models")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    from mlx_vla.models.modeling_vla import VLAForAction

    sys_info = get_system_info()

    print(f"\n{BOLD}{CYAN}mlx-vla Benchmark{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"  Chip:     {BOLD}{sys_info['chip']}{RESET}")
    print(f"  RAM:      {sys_info['ram_gb']:.0f} GB unified memory")
    print(f"  Device:   {sys_info['device']}")
    print(f"  MLX:      {sys_info['mlx']}")
    print(f"  Python:   {sys_info['python']}")
    print()

    configs = [
        {"name": "Small (CLIP+Continuous)", "vision_backbone": "clip", "vision_hidden_dim": 128,
         "language_hidden_dim": 128, "fusion_type": "cross_attention", "action_type": "continuous",
         "action_dim": 7, "image_size": 64},
        {"name": "Small (DINOv2+Diffusion)", "vision_backbone": "dinov2", "vision_hidden_dim": 128,
         "language_hidden_dim": 128, "fusion_type": "concat", "action_type": "diffusion",
         "action_dim": 7, "action_horizon": 4, "image_size": 64},
        {"name": "Small (SigLIP+Discrete)", "vision_backbone": "siglip", "vision_hidden_dim": 128,
         "language_hidden_dim": 128, "fusion_type": "gated", "action_type": "discrete",
         "action_dim": 7, "num_action_bins": 64, "image_size": 64},
        {"name": "Small (SAM+Chunking)", "vision_backbone": "sam", "vision_hidden_dim": 128,
         "language_hidden_dim": 128, "fusion_type": "qkv_fusion", "action_type": "chunking",
         "action_dim": 7, "action_horizon": 4, "image_size": 64},
    ]

    if args.full:
        configs.extend([
            {"name": "Medium (CLIP+Continuous)", "vision_backbone": "clip", "vision_hidden_dim": 384,
             "language_hidden_dim": 384, "fusion_type": "cross_attention", "action_type": "continuous",
             "action_dim": 7, "image_size": 112},
            {"name": "Medium (DINOv2+Diffusion)", "vision_backbone": "dinov2", "vision_hidden_dim": 384,
             "language_hidden_dim": 384, "fusion_type": "concat", "action_type": "diffusion",
             "action_dim": 7, "action_horizon": 4, "image_size": 112},
            {"name": "Large (CLIP+Continuous)", "vision_backbone": "clip", "vision_hidden_dim": 768,
             "language_hidden_dim": 768, "fusion_type": "cross_attention", "action_type": "continuous",
             "action_dim": 7, "image_size": 224},
        ])

    # -- Forward pass benchmark --
    print(f"{BOLD}Forward Pass (batch=4){RESET}")
    print(f"{'Model':<32} {'Params':>10} {'Size':>8} {'Latency':>12} {'Throughput':>14}")
    print(f"{'-'*32} {'-'*10} {'-'*8} {'-'*12} {'-'*14}")

    for cfg in configs:
        name = cfg.pop("name")
        model = VLAForAction(**cfg)
        n_p = count_params(model.parameters())
        sz = model_size_mb(model)
        res = benchmark_forward(model, cfg["image_size"], batch_size=4, n_warmup=3, n_iter=20)
        print(f"{name:<32} {n_p:>10,} {sz:>6.1f}MB {res['mean_ms']:>8.1f}ms {res['throughput']:>10.0f} img/s")
        cfg["name"] = name
        del model

    # -- Inference (single image) --
    print(f"\n{BOLD}Inference Latency (single image){RESET}")
    print(f"{'Model':<32} {'Mean':>10} {'Min':>10} {'p99':>10}")
    print(f"{'-'*32} {'-'*10} {'-'*10} {'-'*10}")

    for cfg in configs:
        name = cfg.pop("name")
        model = VLAForAction(**cfg)
        res = benchmark_inference(model, cfg["image_size"], n_warmup=3, n_iter=30)
        print(f"{name:<32} {res['mean_ms']:>7.1f} ms {res['min_ms']:>7.1f} ms {res['p99_ms']:>7.1f} ms")
        cfg["name"] = name
        del model

    # -- Training step --
    print(f"\n{BOLD}Training Step (batch=4){RESET}")
    print(f"{'Model':<32} {'Step Time':>12} {'Throughput':>14}")
    print(f"{'-'*32} {'-'*12} {'-'*14}")

    for cfg in configs:
        name = cfg.pop("name")
        model = VLAForAction(**cfg)
        res = benchmark_training(model, cfg["image_size"], batch_size=4, n_steps=15)
        print(f"{name:<32} {res['mean_ms']:>8.1f} ms {res['samples_per_sec']:>10.0f} samp/s")
        cfg["name"] = name
        del model

    print(f"\n{CYAN}{'=' * 60}{RESET}")
    print(f"{DIM}All benchmarks on {sys_info['chip']} with {sys_info['ram_gb']:.0f}GB unified memory{RESET}")
    print(f"{DIM}MLX {sys_info['mlx']} | Python {sys_info['python']}{RESET}")
    print()


if __name__ == "__main__":
    main()
