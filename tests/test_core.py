import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path

import numpy as np


class TestConfig:
    def test_default_config(self):
        from mlx_vla.utils.config import VLAConfigManager, DEFAULT_CONFIG

        config = VLAConfigManager.from_default()

        assert config.model.name == DEFAULT_CONFIG["model"]["name"]
        assert config.lora.rank == DEFAULT_CONFIG["lora"]["rank"]
        assert config.data.batch_size == DEFAULT_CONFIG["data"]["batch_size"]

    def test_config_update(self):
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()
        config.update(
            model={"name": "custom-model"},
            lora={"rank": 32},
            training={"epochs": 10},
        )

        assert config.model.name == "custom-model"
        assert config.lora.rank == 32
        assert config.training.epochs == 10

    def test_config_save_json(self):
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()
        config.model.name = "test-model"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)

            with open(temp_path, "r") as f:
                loaded = json.load(f)

            assert loaded["model"]["name"] == "test-model"
        finally:
            os.unlink(temp_path)

    def test_config_save_yaml(self):
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()
        config.model.name = "test-model-yaml"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)

            with open(temp_path, "r") as f:
                loaded = yaml.safe_load(f)

            assert loaded["model"]["name"] == "test-model-yaml"
        finally:
            os.unlink(temp_path)

    def test_config_load(self):
        from mlx_vla.utils.config import VLAConfigManager

        config_data = {
            "model": {"name": "loaded-model", "action_dim": 14},
            "lora": {"enabled": False},
            "data": {"batch_size": 4},
            "training": {"epochs": 5},
            "checkpointing": {"output_dir": "./test"},
            "logging": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = VLAConfigManager.load(temp_path)

            assert config.model.name == "loaded-model"
            assert config.model.action_dim == 14
            assert config.lora.enabled is False
            assert config.data.batch_size == 4
            assert config.training.epochs == 5
        finally:
            os.unlink(temp_path)

    def test_global_config(self):
        from mlx_vla.utils.config import set_global_config, get_global_config, VLAConfigManager

        config = VLAConfigManager.from_default()
        config.model.name = "global-test"

        set_global_config(config)

        retrieved = get_global_config()
        assert retrieved.model.name == "global-test"


class TestDataCollator:
    def test_collator_init(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(
            image_size=224,
            action_dim=7,
            action_normalization="clip_minus_one_to_one",
        )

        assert collator.image_size == 224
        assert collator.action_dim == 7
        assert collator.action_normalization == "clip_minus_one_to_one"

    def test_collator_with_custom_action_dim(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(action_dim=14)

        action = np.zeros(14)
        result = collator._normalize_action(action)

        assert len(result) == 14

    def test_collator_normalize_action(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(action_normalization="clip_minus_one_to_one")

        action = np.array([0.0, 0.5, -0.5, 1.5, -1.5, 0.0, 1.0])
        result = collator._normalize_action(action)

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_collator_preprocess_image_none(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=224)

        result = collator._preprocess_image(None)

        assert result.shape == (3, 224, 224)

    def test_collator_batch(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=224, action_dim=7)

        batch = [
            {"image": np.zeros((224, 224, 3), dtype=np.uint8), "action": np.zeros(7)},
            {"image": np.zeros((224, 224, 3), dtype=np.uint8), "action": np.zeros(7)},
        ]

        result = collator(batch)

        assert "pixel_values" in result
        assert "action" in result


class TestDataLoader:
    def test_dataloader_init(self):
        from mlx_vla.data.dataloader import VLADataloader
        from mlx_vla.data.dataset import VLADataset

        class DummyDataset(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0]*7}]}]

        dataset = DummyDataset("dummy")
        loader = VLADataloader(dataset, batch_size=2, shuffle=False)

        assert loader.batch_size == 2

    def test_dataloader_iterate(self):
        from mlx_vla.data.dataloader import VLADataloader
        from mlx_vla.data.dataset import VLADataset

        class DummyDataset(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0]*7}]} for _ in range(4)]

        dataset = DummyDataset("dummy")
        loader = VLADataloader(dataset, batch_size=2, shuffle=False)

        batches = list(loader)

        assert len(batches) == 2


class TestActionNormalizer:
    def test_normalizer_init(self):
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("bridge_orig")

        assert norm.robot == "bridge_orig"

    def test_normalizer_normalize(self):
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("bridge_orig")

        action = np.array([0, 0, 0, 0, 0, 0, 0])
        result = norm.normalize(action)

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_normalizer_unnormalize(self):
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("bridge_orig")

        action = np.array([0, 0, 0, 0, 0, 0, 0])
        normalized = norm.normalize(action)
        result = norm.unnormalize(normalized)

        assert result.shape == action.shape

    def test_normalizer_from_model(self):
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer.from_model("openvla")

        assert norm.robot == "bridge_orig"


class TestModels:
    def test_vision_encoder_init(self):
        from mlx_vla.models.vision import VisionEncoder

        encoder = VisionEncoder(backbone="clip", hidden_dim=768, image_size=224)

        assert encoder.backbone == "clip"
        assert encoder.hidden_dim == 768

    def test_vision_encoder_forward_shape(self):
        import mlx.core as mx
        from mlx_vla.models.vision import VisionEncoder

        encoder = VisionEncoder(backbone="clip", hidden_dim=256, image_size=224)

        images = mx.random.normal((1, 3, 224, 224))
        output = encoder(images)

        assert output.ndim == 3
        assert output.shape[-1] == 256

    def test_action_heads_discrete(self):
        import mlx.core as mx
        from mlx_vla.models.action_heads import DiscreteActionHead

        head = DiscreteActionHead(hidden_dim=256, action_dim=7, num_bins=256)

        hidden = mx.random.normal((1, 10, 256))
        output = head(hidden)

        assert output.shape == (1, 10, 7, 256)

    def test_action_heads_continuous(self):
        import mlx.core as mx
        from mlx_vla.models.action_heads import ContinuousActionHead

        head = ContinuousActionHead(hidden_dim=256, action_dim=7, action_horizon=1)

        hidden = mx.random.normal((1, 10, 256))
        output = head(hidden)

        assert output.shape[0] == 1
        assert output.shape[-1] == 7

    def test_action_heads_diffusion(self):
        import mlx.core as mx
        from mlx_vla.models.action_heads import DiffusionActionHead

        head = DiffusionActionHead(hidden_dim=256, action_dim=7, action_horizon=4)

        hidden = mx.random.normal((1, 10, 256))
        output = head(hidden)

        assert output.shape == (1, 4, 7)


class TestLoRA:
    def test_lora_layer_init(self):
        import mlx.nn as nn
        from mlx_vla.training.lora import LoRALayer

        linear = nn.Linear(256, 256)
        lora = LoRALayer(linear, rank=8, alpha=16)

        assert lora.rank == 8
        assert lora.alpha == 16


class TestVLATrainerArguments:
    def test_args_from_default(self):
        from mlx_vla.core import VLATrainingArguments

        args = VLATrainingArguments()

        assert args.output_dir is not None
        assert args.learning_rate > 0
        assert args.num_train_epochs > 0

    def test_args_with_overrides(self):
        from mlx_vla.core import VLATrainingArguments

        args = VLATrainingArguments(
            learning_rate=5e-5,
            num_train_epochs=10,
            per_device_train_batch_size=2,
        )

        assert args.learning_rate == 5e-5
        assert args.num_train_epochs == 10
        assert args.per_device_train_batch_size == 2


class TestDataset:
    def test_episode_dataset_custom(self):
        from mlx_vla.data.dataset import EpisodeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_file = Path(tmpdir) / "episode_0.json"
            episode_file.write_text(json.dumps({
                "steps": [
                    {"image": "img.jpg", "action": [0, 0, 0, 0, 0, 0, 0], "language": "test"}
                ]
            }))

            dataset = EpisodeDataset(tmpdir)

            assert len(dataset) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
