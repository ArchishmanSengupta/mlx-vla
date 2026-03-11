#!/usr/bin/env python3
"""
mlx-vla Demo Script
===================
A polished end-to-end demo for terminal recording.
Shows: dataset creation -> training -> save -> load -> inference.

Usage:
    python examples/demo.py

Record with asciinema:
    asciinema rec demo.cast -c "python examples/demo.py"
    agg demo.cast demo.gif --theme monokai
"""

import time
import sys
import numpy as np
from PIL import Image

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
CHECK = f"{GREEN}\u2713{RESET}"

def header(text):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")

def step(num, text):
    print(f"  {BOLD}{YELLOW}[{num}]{RESET} {text}")

def done(text):
    print(f"  {CHECK}  {text}")

def info(text):
    print(f"      {DIM}{text}{RESET}")


def main():
    header("mlx-vla: Vision-Language-Action on Apple Silicon")

    import platform
    import mlx.core as mx
    from mlx_vla._version import __version__

    print(f"  {DIM}Version:  {__version__}{RESET}")
    print(f"  {DIM}Device:   {mx.default_device()}{RESET}")
    print(f"  {DIM}Platform: {platform.processor()} ({platform.machine()}){RESET}")
    print()

    # ----------------------------------------------------------------
    # Step 1: Create synthetic robot dataset
    # ----------------------------------------------------------------
    step(1, "Creating synthetic robot dataset...")
    import tempfile, os, json
    tmpdir = tempfile.mkdtemp(prefix="mlx_vla_demo_")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir)

    np.random.seed(42)
    n_episodes = 12
    steps_per_ep = 6
    for ep in range(n_episodes):
        steps = []
        for s in range(steps_per_ep):
            img = np.random.randint(40, 180, (64, 64, 3), dtype=np.uint8)
            color = [220, 60, 60] if ep < n_episodes // 2 else [60, 60, 220]
            y = 15 + s * 5
            img[y:y+12, 20:44] = color
            img_name = f"ep{ep}_s{s}.png"
            Image.fromarray(img).save(os.path.join(data_dir, img_name))

            direction = 1.0 if ep < n_episodes // 2 else -1.0
            action = [
                0.08 * (s + 1) * direction,
                -0.03 * s,
                0.015 * s,
                0.0, 0.0, 0.0,
                1.0 if s == steps_per_ep - 1 else 0.0,
            ]
            task = "pick up the red block" if ep < n_episodes // 2 else "pick up the blue block"
            steps.append({"image": img_name, "action": action, "language": task})

        with open(os.path.join(data_dir, f"episode_{ep:03d}.json"), "w") as f:
            json.dump(steps, f)

    total_frames = n_episodes * steps_per_ep
    done(f"{n_episodes} episodes, {total_frames} frames, 64x64 RGB")
    info(f"Tasks: 'pick up the red block', 'pick up the blue block'")

    # ----------------------------------------------------------------
    # Step 2: Build model
    # ----------------------------------------------------------------
    step(2, "Building VLA model...")
    from mlx_vla.models.modeling_vla import VLAForAction

    model = VLAForAction(
        vision_backbone="clip",
        vision_hidden_dim=128,
        language_hidden_dim=128,
        fusion_type="cross_attention",
        action_type="continuous",
        action_dim=7,
        action_horizon=1,
        image_size=64,
    )

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

    n_params = count_params(model.parameters())
    done(f"VLAForAction  |  {n_params:,} parameters")
    info("CLIP vision + cross-attention fusion + continuous action head")

    # ----------------------------------------------------------------
    # Step 3: Train
    # ----------------------------------------------------------------
    step(3, "Training...")
    from mlx_vla.data.dataset import EpisodeDataset
    from mlx_vla.data.collator import VLAModuleDataCollator
    from mlx_vla.training.trainer import VLATrainer
    from mlx_vla.core import VLATrainingArguments

    dataset = EpisodeDataset(data_dir, split="train", image_size=64)
    collator = VLAModuleDataCollator(image_size=64, action_dim=7, max_length=16)

    out_dir = os.path.join(tmpdir, "output")
    args = VLATrainingArguments(
        output_dir=out_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        learning_rate=2e-3,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        save_steps=99999,
        logging_steps=99999,
    )

    trainer = VLATrainer(model=model, args=args, train_dataset=dataset, data_collator=collator)

    start = time.time()
    losses = []
    for epoch in range(args.num_train_epochs):
        trainer.epoch = epoch
        trainer.model.train()
        trainer.accumulated_grads = None
        ep_losses = []
        for s, batch in enumerate(trainer.train_dataloader):
            loss = trainer._train_step(batch)
            mx.eval(loss)
            ep_losses.append(float(loss))
            trainer.global_step += 1
        avg = np.mean(ep_losses)
        losses.append(avg)
        bar_len = int(30 * (epoch + 1) / args.num_train_epochs)
        bar = f"[{'=' * bar_len}{' ' * (30 - bar_len)}]"
        sys.stdout.write(f"\r      {bar}  epoch {epoch+1:2d}/{args.num_train_epochs}  loss={avg:.5f}")
        sys.stdout.flush()

    duration = time.time() - start
    print()
    reduction = (1 - losses[-1] / losses[0]) * 100
    done(f"Loss: {losses[0]:.5f} -> {losses[-1]:.5f}  ({reduction:.0f}% reduction)")
    info(f"{args.num_train_epochs} epochs in {duration:.1f}s  |  {total_frames * args.num_train_epochs / duration:.0f} samples/sec")

    # ----------------------------------------------------------------
    # Step 4: Save
    # ----------------------------------------------------------------
    step(4, "Saving model...")
    model_dir = os.path.join(tmpdir, "trained_model")
    model.save(model_dir)
    weights_size = os.path.getsize(os.path.join(model_dir, "model.npz"))
    done(f"Saved to disk  |  {weights_size / 1024 / 1024:.1f} MB")

    # ----------------------------------------------------------------
    # Step 5: Load + Inference
    # ----------------------------------------------------------------
    step(5, "Loading model & running inference...")
    loaded = VLAForAction.load(model_dir)
    from mlx_vla.inference.pipeline import VLAPipeline

    pipeline = VLAPipeline(model=loaded, unnorm_key="bridge_orig")

    print()
    tasks = [
        ("pick up the red block", os.path.join(data_dir, "ep0_s0.png"), "Red block image"),
        ("pick up the blue block", os.path.join(data_dir, "ep6_s0.png"), "Blue block image"),
        ("move to the left", None, "Random noise image"),
    ]

    for instruction, img_path, label in tasks:
        if img_path:
            img = Image.open(img_path)
        else:
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        t0 = time.time()
        action = pipeline.predict(image=img, language=instruction)
        latency = (time.time() - t0) * 1000

        action_str = ", ".join(f"{a:+.3f}" for a in action[:3])
        grip = action[-1]
        print(f"      {MAGENTA}\"{instruction}\"{RESET}")
        print(f"      {DIM}{label} -> [{action_str}, ...] gripper={grip:.2f}  ({latency:.0f}ms){RESET}")
        print()

    done("Inference complete")

    # ----------------------------------------------------------------
    # Step 6: Streaming demo
    # ----------------------------------------------------------------
    step(6, "Streaming robot control loop...")
    frames = [Image.open(os.path.join(data_dir, f"ep0_s{s}.png")) for s in range(steps_per_ep)]
    print(f"      {DIM}Instruction: \"pick up the red block\"{RESET}")
    print(f"      {DIM}Frames: {len(frames)} sequential observations{RESET}")
    print()

    for i, action in enumerate(pipeline.stream_actions(iter(frames), language="pick up the red block")):
        xyz = f"({action[0]:+.1f}, {action[1]:+.1f}, {action[2]:+.1f})"
        grip = "CLOSE" if action[6] > 0.5 else "OPEN"
        print(f"      t={i}  pos={xyz}  gripper={grip}")

    print()
    done("Streaming complete")

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    import shutil
    shutil.rmtree(tmpdir)

    header("Done! mlx-vla is ready for robot learning on your Mac.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
