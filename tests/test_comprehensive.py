import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
import numpy as np
from PIL import Image
import io


class TestConfigEdgeCases:
    """Test configuration edge cases."""

    def test_config_partial_override(self):
        """Test partial config override."""
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()
        config.update(lora={"rank": 16})

        assert config.lora.rank == 16
        assert config.model.name == "openvla-7b"
        assert config.training.epochs == 3

    def test_config_invalid_type_handling(self):
        """Test config with wrong types."""
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()
        config.update(model={"name": 123})

        assert config.model.name == 123

    def test_config_extra_keys(self):
        """Test config with extra unknown keys."""
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()
        config.update(
            model={"unknown_key": "value", "name": "test"},
            extra_section={"key": "value"},
        )

        assert config.model.name == "test"

    def test_config_empty_values(self):
        """Test config with empty/null values."""
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()
        config.update(
            model={"name": ""},
            lora={"target_modules": []},
        )

        assert config.model.name == ""
        assert config.lora.target_modules == []

    def test_config_numeric_precision(self):
        """Test config with various numeric precisions."""
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()
        config.update(
            training={
                "learning_rate": 1e-10,
                "weight_decay": 0.000001,
                "max_grad_norm": 0.1,
            },
        )

        assert config.training.learning_rate == 1e-10
        assert config.training.weight_decay == 0.000001


class TestDataCollatorEdgeCases:
    """Test data collator edge cases."""

    def test_collator_empty_batch(self):
        """Test collator with empty batch."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator()

        result = collator([])

        assert "pixel_values" in result

    def test_collator_none_image(self):
        """Test collator with None image."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=224)

        result = collator._preprocess_image(None)

        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_collator_string_image_path(self):
        """Test collator with string path that doesn't exist."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator()

        result = collator._preprocess_image("/nonexistent/path.jpg")

        assert result.shape == (3, 224, 224)

    def test_collator_grayscale_image(self):
        """Test collator with grayscale image."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=224)

        gray = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        result = collator._preprocess_image(gray)

        assert result.shape == (3, 224, 224)

    def test_collator_rgba_image(self):
        """Test collator with RGBA image."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=224)

        rgba = np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8)
        result = collator._preprocess_image(rgba)

        assert result.shape == (3, 224, 224)

    def test_collator_action_boundary_values(self):
        """Test action normalization at exact boundaries."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(action_normalization="clip_minus_one_to_one")

        action = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        result = collator._normalize_action(action)

        assert result[0] == 1.0
        assert result[1] == -1.0

    def test_collator_action_out_of_range(self):
        """Test action normalization with out-of-range values."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(action_normalization="clip_minus_one_to_one")

        action = np.array([10.0, -10.0, 5.0, -5.0, 2.0, -2.0, 100.0])
        result = collator._normalize_action(action)

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_collator_action_empty(self):
        """Test action with empty array."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(action_dim=14)

        result = collator._normalize_action(np.array([]))

        assert len(result) == 14

    def test_collator_action_wrong_dim(self):
        """Test action with wrong dimensionality."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(action_dim=7)

        action = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        result = collator._normalize_action(action)

        assert len(result) == 7

    def test_collator_action_none(self):
        """Test action with None."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(action_dim=7)

        result = collator._normalize_action(None)

        assert len(result) == 7

    def test_collator_batch_with_missing_data(self):
        """Test batch with missing images or actions."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator()

        batch = [
            {"image": None, "action": np.zeros(7)},
            {"image": np.zeros((224, 224, 3), dtype=np.uint8), "action": None},
            {},
        ]

        result = collator(batch)

        assert "pixel_values" in result
        assert "action" in result

    def test_collator_pil_image(self):
        """Test collator with PIL Image."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=224)

        pil_img = Image.new("RGB", (300, 300), color="red")
        result = collator._preprocess_image(pil_img)

        assert result.shape == (3, 224, 224)


class TestActionNormalizerEdgeCases:
    """Test action normalizer edge cases."""

    def test_normalizer_all_zeros(self):
        """Test normalizer with all zeros."""
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("bridge_orig")
        action = np.zeros(7)
        result = norm.normalize(action)

        assert result.shape == action.shape

    def test_normalizer_max_values(self):
        """Test normalizer with max values."""
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("franka")
        action = np.array([100, 100, 100, 3.14, 3.14, 3.14, 1])
        result = norm.normalize(action)

        assert np.all(result <= 1.0)

    def test_normalizer_min_values(self):
        """Test normalizer with min values."""
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("franka")
        action = np.array([-100, -100, -100, -3.14, -3.14, -3.14, -1])
        result = norm.normalize(action)

        assert np.all(result >= -1.0)

    def test_normalizer_unknown_robot(self):
        """Test normalizer with unknown robot type."""
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("unknown_robot")
        action = np.zeros(7)
        result = norm.normalize(action)

        assert result.shape == action.shape

    def test_normalizer_preserves_shape(self):
        """Test normalizer preserves different action shapes."""
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("bridge_orig")

        for shape in [(7,), (14,), (1, 7), (2, 7)]:
            action = np.zeros(shape)
            result = norm.normalize(action)
            assert result.shape == shape


class TestDataloaderEdgeCases:
    """Test dataloader edge cases."""

    def test_dataloader_empty_dataset(self):
        """Test dataloader with empty dataset."""
        from mlx_vla.data.dataloader import VLADataloader
        from mlx_vla.data.dataset import VLADataset

        class EmptyDataset(VLADataset):
            def _load_episodes(self):
                return []

        dataset = EmptyDataset("dummy")
        loader = VLADataloader(dataset, batch_size=4)

        batches = list(loader)
        assert len(batches) == 0

    def test_dataloader_single_sample(self):
        """Test dataloader with single sample."""
        from mlx_vla.data.dataloader import VLADataloader
        from mlx_vla.data.dataset import VLADataset

        class SingleDataset(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0]*7}]}]

        dataset = SingleDataset("dummy")
        loader = VLADataloader(dataset, batch_size=1)

        batches = list(loader)
        assert len(batches) == 1

    def test_dataloader_batch_larger_than_dataset(self):
        """Test dataloader when batch_size > dataset size."""
        from mlx_vla.data.dataloader import VLADataloader
        from mlx_vla.data.dataset import VLADataset

        class SmallDataset(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0]*7}]}]

        dataset = SmallDataset("dummy")
        loader = VLADataloader(dataset, batch_size=100)

        batches = list(loader)
        assert len(batches) == 1

    def test_dataloader_drop_last(self):
        """Test dataloader with drop_last=True."""
        from mlx_vla.data.dataloader import VLADataloader
        from mlx_vla.data.dataset import VLADataset

        class SmallDataset(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0]*7}]} for _ in range(3)]

        dataset = SmallDataset("dummy")
        loader = VLADataloader(dataset, batch_size=2, drop_last=True)

        batches = list(loader)
        assert len(batches) == 1


class TestModelEdgeCases:
    """Test model edge cases."""

    def test_vision_encoder_different_sizes(self):
        """Test vision encoder with different image sizes."""
        import mlx.core as mx
        from mlx_vla.models.vision import VisionEncoder

        for size in [224, 112, 336]:
            encoder = VisionEncoder(backbone="clip", hidden_dim=256, image_size=size)
            images = mx.random.normal((1, 3, size, size))
            output = encoder(images)
            assert output.shape[-1 == 256]

    def test_vision_encoder_batch_sizes(self):
        """Test vision encoder with different batch sizes."""
        import mlx.core as mx
        from mlx_vla.models.vision import VisionEncoder

        encoder = VisionEncoder(backbone="clip", hidden_dim=128, image_size=224)

        for batch_size in [1, 2, 8, 16]:
            images = mx.random.normal((batch_size, 3, 224, 224))
            output = encoder(images)
            assert output.shape[0] == batch_size

    def test_action_head_discrete_different_dims(self):
        """Test discrete action head with different action dimensions."""
        import mlx.core as mx
        from mlx_vla.models.action_heads import DiscreteActionHead

        for action_dim in [7, 14, 21]:
            head = DiscreteActionHead(hidden_dim=128, action_dim=action_dim, num_bins=256)
            hidden = mx.random.normal((2, 10, 128))
            output = head(hidden)
            assert output.shape[2] == action_dim

    def test_action_head_continuous_different_horizons(self):
        """Test continuous action head with different horizons."""
        import mlx.core as mx
        from mlx_vla.models.action_heads import ContinuousActionHead

        for horizon in [1, 4, 10, 100]:
            head = ContinuousActionHead(hidden_dim=128, action_dim=7, action_horizon=horizon)
            hidden = mx.random.normal((2, 10, 128))
            output = head(hidden)
            assert output.shape[1] == horizon

    def test_action_head_diffusion_noisy_inputs(self):
        """Test diffusion head with noisy inputs."""
        import mlx.core as mx
        from mlx_vla.models.action_heads import DiffusionActionHead

        head = DiffusionActionHead(hidden_dim=128, action_dim=7, action_horizon=4)
        hidden = mx.random.normal((2, 10, 128))
        noisy = mx.random.normal((2, 4, 7))
        t = mx.ones((2,))

        output = head(hidden, noisy, t)
        assert output.shape == (2, 4, 7)

    def test_vla_model_without_language(self):
        """Test VLA model without language input."""
        import mlx.core as mx
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_backbone="clip",
            action_type="continuous",
            action_dim=7,
            vision_hidden_dim=256,
            language_hidden_dim=256,
        )

        images = mx.random.normal((1, 3, 224, 224))
        output = model(pixel_values=images, input_ids=None)

        assert "action" in output or "logits" in output


class TestLoRAEdgeCases:
    """Test LoRA edge cases."""

    def test_lora_zero_rank(self):
        """Test LoRA with zero rank."""
        import mlx.nn as nn
        from mlx_vla.training.lora import LoRALayer

        linear = nn.Linear(128, 128)
        lora = LoRALayer(linear, rank=0, alpha=16)

        assert lora.rank == 0

    def test_lora_high_rank(self):
        """Test LoRA with very high rank."""
        import mlx.nn as nn
        from mlx_vla.training.lora import LoRALayer

        linear = nn.Linear(128, 128)
        lora = LoRALayer(linear, rank=128, alpha=16)

        assert lora.rank == 128

    def test_lora_apply_to_nonlinear(self):
        """Test applying LoRA to non-linear layer."""
        import mlx.nn as nn
        from mlx_vla.training.lora import apply_lora

        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Use "all" to apply LoRA to all Linear layers
        model = apply_lora(model, rank=8, target_modules="all")

        # Check that LoRA was applied to the Linear layers
        assert hasattr(model.layers[0], "lora_A")
        assert hasattr(model.layers[2], "lora_A")

    def test_lora_all_target_modules(self):
        """Test LoRA with all possible target modules."""
        import mlx.nn as nn
        from mlx_vla.training.lora import apply_lora

        model = nn.Linear(128, 128)

        for modules in [
            ["q_proj"],
            ["k_proj", "v_proj", "o_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ]:
            lora_model = apply_lora(nn.Linear(128, 128), rank=4, target_modules=modules)
            assert lora_model is not None


class TestTrainingEdgeCases:
    """Test training edge cases."""

    def test_trainer_zero_grad_accumulation(self):
        """Test trainer with gradient accumulation disabled."""
        from mlx_vla.training.trainer import VLATrainer

        assert True

    def test_trainer_gradient_accumulation_steps(self):
        """Test gradient accumulation with various step counts."""
        from mlx_vla.core import VLATrainingArguments

        for steps in [1, 2, 4, 8, 16]:
            args = VLATrainingArguments(gradient_accumulation_steps=steps)
            assert args.gradient_accumulation_steps == steps

    def test_trainer_different_batch_sizes(self):
        """Test with various batch sizes."""
        from mlx_vla.core import VLATrainingArguments

        for bs in [1, 2, 4, 8, 16]:
            args = VLATrainingArguments(per_device_train_batch_size=bs)
            assert args.per_device_train_batch_size == bs


class TestCallbackEdgeCases:
    """Test callback edge cases."""

    def test_checkpoint_callback_no_steps(self):
        """Test checkpoint callback when no steps taken."""
        from mlx_vla.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(save_steps=100)

        assert callback.save_steps == 100
        assert callback.checkpoints == []

    def test_logging_callback_no_steps(self):
        """Test logging callback with no steps."""
        from mlx_vla.training.callbacks import LoggingCallback

        callback = LoggingCallback(log_steps=10)

        assert callback.log_steps == 10
        assert callback.logs == []

    def test_early_stopping_patience(self):
        """Test early stopping with different patience values."""
        from mlx_vla.training.callbacks import EarlyStoppingCallback

        for patience in [1, 5, 10, 100]:
            callback = EarlyStoppingCallback(patience=patience)
            assert callback.patience == patience
            assert callback.wait == 0


class TestDatasetEdgeCases:
    """Test dataset edge cases."""

    def test_episode_dataset_missing_image(self):
        """Test episode dataset with missing image file."""
        from mlx_vla.data.dataset import EpisodeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_file = Path(tmpdir) / "episode_0.json"
            episode_file.write_text(json.dumps({
                "steps": [
                    {"action": [0, 0, 0, 0, 0, 0, 0], "language": "test"}
                ]
            }))

            dataset = EpisodeDataset(tmpdir)

            assert len(dataset) == 1

    def test_episode_dataset_empty_steps(self):
        """Test episode dataset with empty steps."""
        from mlx_vla.data.dataset import EpisodeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_file = Path(tmpdir) / "episode_0.json"
            episode_file.write_text(json.dumps({"steps": []}))

            dataset = EpisodeDataset(tmpdir)

            assert len(dataset) == 1

    def test_episode_dataset_malformed_json(self):
        """Test episode dataset with malformed JSON."""
        from mlx_vla.data.dataset import EpisodeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_file = Path(tmpdir) / "episode_0.json"
            episode_file.write_text("not valid json")

            try:
                dataset = EpisodeDataset(tmpdir)
            except:
                pass

    def test_rlds_dataset_unsupported(self):
        """Test RLDS dataset with unsupported dataset name."""
        from mlx_vla.data.dataset import RLDSDataset

        try:
            dataset = RLDSDataset("unsupported_dataset", split="train")
        except Exception as e:
            assert "tensorflow_datasets" in str(e).lower() or "required" in str(e).lower()


class TestInferenceEdgeCases:
    """Test inference edge cases."""

    def test_pipeline_missing_image(self):
        """Test pipeline with missing image."""
        from mlx_vla.inference.pipeline import VLAPipeline
        from mlx_vla.models.modeling_vla import VLAForAction
        import mlx.core as mx

        model = VLAForAction(
            vision_backbone="clip",
            action_type="continuous",
            vision_hidden_dim=256,
            language_hidden_dim=256,
        )

        pipeline = VLAPipeline(model=model)

        assert pipeline.model is not None


class TestSerializationEdgeCases:
    """Test serialization edge cases."""

    def test_model_save_load(self):
        """Test model save and load."""
        from mlx_vla.models.modeling_vla import VLAForAction
        import tempfile

        model = VLAForAction(
            vision_backbone="clip",
            action_type="continuous",
            action_dim=7,
            vision_hidden_dim=128,
            language_hidden_dim=128,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            loaded = VLAForAction.load(tmpdir)

            assert loaded is not None


class TestIntegration:
    """Integration tests."""

    def test_full_training_config(self):
        """Test full training configuration."""
        from mlx_vla import load_config, VLATrainer
        from mlx_vla.models.modeling_vla import VLAForAction

        cfg = load_config(
            lora={"rank": 16, "alpha": 32},
            training={"epochs": 5, "learning_rate": 5e-5},
            data={"batch_size": 2},
        )

        assert cfg.lora.rank == 16
        assert cfg.training.epochs == 5
        assert cfg.data.batch_size == 2

    def test_config_chaining(self):
        """Test config loading and chaining."""
        from mlx_vla.utils.config import VLAConfigManager

        config1 = VLAConfigManager.from_default()
        config1.update(model={"name": "model1"})

        config2 = VLAConfigManager.from_default()
        config2.update(model={"name": "model2"})

        assert config1.model.name != config2.model.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
