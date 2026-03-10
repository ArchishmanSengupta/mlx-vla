import pytest
import os
import tempfile
import json
import numpy as np
from pathlib import Path
from PIL import Image

import mlx.core as mx
import mlx.nn as nn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TestVLAForActionAllTypes:
    @pytest.fixture(params=["discrete", "continuous", "diffusion", "chunking"])
    def action_type(self, request):
        return request.param

    @pytest.fixture(params=["cross_attention", "concat", "gated", "qkv_fusion"])
    def fusion_type(self, request):
        return request.param

    def test_forward_all_action_types(self, action_type):
        from mlx_vla.models.modeling_vla import VLAForAction

        kwargs = dict(
            vision_backbone="clip",
            vision_hidden_dim=128,
            language_hidden_dim=128,
            fusion_type="cross_attention",
            action_type=action_type,
            action_dim=7,
            action_horizon=4 if action_type in ("diffusion", "chunking") else 1,
            num_action_bins=64,
            image_size=112,
        )
        model = VLAForAction(**kwargs)
        images = mx.random.normal((2, 3, 112, 112))
        ids = mx.zeros((2, 5), dtype=mx.int32)
        out = model(images, ids)
        mx.eval(out["hidden_states"])
        assert out["hidden_states"].shape[0] == 2

        if action_type == "discrete":
            assert "logits" in out
            assert out["logits"].shape[2] == 7
            assert out["logits"].shape[3] == 64
        elif action_type == "diffusion":
            assert "action" in out
            assert out["action"].shape == (2, 4, 7)
        elif action_type == "continuous":
            assert "action" in out
        elif action_type == "chunking":
            assert "action" in out

    def test_predict_action_all_types(self, action_type):
        from mlx_vla.models.modeling_vla import VLAForAction

        kwargs = dict(
            vision_backbone="clip",
            vision_hidden_dim=128,
            language_hidden_dim=128,
            fusion_type="cross_attention",
            action_type=action_type,
            action_dim=7,
            action_horizon=4 if action_type in ("diffusion", "chunking") else 1,
            num_action_bins=64,
            image_size=112,
        )
        model = VLAForAction(**kwargs)
        images = mx.random.normal((1, 3, 112, 112))
        action = model.predict_action(images)
        mx.eval(action)
        assert action.shape[0] == 1
        assert action.shape[-1] == 7

    def test_forward_all_fusion_types(self, fusion_type):
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_backbone="clip",
            vision_hidden_dim=128,
            language_hidden_dim=128,
            fusion_type=fusion_type,
            action_type="continuous",
            action_dim=7,
            image_size=112,
        )
        images = mx.random.normal((1, 3, 112, 112))
        ids = mx.zeros((1, 5), dtype=mx.int32)
        out = model(images, ids)
        mx.eval(out["action"])
        assert "action" in out

    def test_forward_without_language(self):
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_backbone="clip",
            vision_hidden_dim=128,
            language_hidden_dim=128,
            action_type="discrete",
            action_dim=7,
            image_size=112,
        )
        images = mx.random.normal((1, 3, 112, 112))
        out = model(images, input_ids=None)
        mx.eval(out["logits"])
        assert "logits" in out

    def test_unknown_action_type_raises(self):
        from mlx_vla.models.modeling_vla import VLAForAction

        with pytest.raises(ValueError, match="Unknown action type"):
            VLAForAction(action_type="nonexistent", vision_hidden_dim=128, language_hidden_dim=128)

    def test_unknown_backbone_raises(self):
        from mlx_vla.models.vision import VisionEncoder

        with pytest.raises(ValueError, match="Unknown backbone"):
            VisionEncoder(backbone="nonexistent", hidden_dim=128)


class TestVLASaveLoad:
    @pytest.fixture(params=["discrete", "continuous", "diffusion"])
    def action_type(self, request):
        return request.param

    def test_save_load_roundtrip(self, action_type):
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_backbone="clip",
            vision_hidden_dim=128,
            language_hidden_dim=128,
            fusion_type="cross_attention",
            action_type=action_type,
            action_dim=7,
            action_horizon=4 if action_type == "diffusion" else 1,
            num_action_bins=64,
            image_size=112,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model"
            model.save(path)

            assert os.path.exists(f"{path}/config.json")
            assert os.path.exists(f"{path}/model.npz")

            with open(f"{path}/config.json") as f:
                cfg = json.load(f)
            assert cfg["vision_hidden_dim"] == 128
            assert cfg["language_hidden_dim"] == 128
            assert cfg["action_type"] == action_type
            assert cfg["fusion_type"] == "cross_attention"

            loaded = VLAForAction.load(path)
            images = mx.random.normal((1, 3, 112, 112))
            out_orig = model(images)
            out_loaded = loaded(images)

            for key in out_orig:
                mx.eval(out_orig[key])
                mx.eval(out_loaded[key])
                assert out_orig[key].shape == out_loaded[key].shape

    def test_from_pretrained(self):
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction.from_pretrained(
            "test-model",
            action_type="discrete",
            action_dim=7,
            vision_hidden_dim=128,
            language_hidden_dim=128,
            image_size=112,
        )
        images = mx.random.normal((1, 3, 112, 112))
        out = model(images)
        mx.eval(out["logits"])
        assert out["logits"].shape[-1] == 256

    def test_vla_alias(self):
        from mlx_vla.models.modeling_vla import VLA, VLAForAction

        assert VLA is VLAForAction


class TestLoRAComplete:
    def test_apply_lora_all_modules(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        from mlx_vla.training.lora import apply_lora, LoRALayer

        model = VLAForAction(
            vision_hidden_dim=128,
            language_hidden_dim=128,
            action_type="discrete",
            image_size=112,
        )
        model = apply_lora(model, rank=4, alpha=8, target_modules=None)

        count = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALayer))
        assert count > 0

        images = mx.random.normal((1, 3, 112, 112))
        out = model(images)
        mx.eval(out["logits"])

    def test_apply_lora_targeted(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        from mlx_vla.training.lora import apply_lora, LoRALayer

        model = VLAForAction(
            vision_hidden_dim=128,
            language_hidden_dim=128,
            action_type="discrete",
            image_size=112,
        )
        model = apply_lora(model, rank=4, alpha=8, target_modules=["query_proj", "value_proj"])

        count = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALayer))
        assert count > 0

        images = mx.random.normal((1, 3, 112, 112))
        out = model(images)
        mx.eval(out["logits"])

    def test_lora_layer_forward(self):
        from mlx_vla.training.lora import LoRALayer

        base = nn.Linear(64, 64)
        lora = LoRALayer(base, rank=4, alpha=8, dropout=0.0)

        x = mx.random.normal((2, 10, 64))
        out = lora(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_lora_layer_zero_rank(self):
        from mlx_vla.training.lora import LoRALayer

        base = nn.Linear(64, 64)
        lora = LoRALayer(base, rank=0, alpha=8)

        x = mx.random.normal((2, 10, 64))
        out = lora(x)
        base_out = base(x)
        mx.eval(out)
        mx.eval(base_out)
        np.testing.assert_allclose(
            np.array(out), np.array(base_out), rtol=1e-5
        )

    def test_lora_layer_with_dropout(self):
        from mlx_vla.training.lora import LoRALayer

        base = nn.Linear(64, 64)
        lora = LoRALayer(base, rank=4, alpha=8, dropout=0.1)

        x = mx.random.normal((2, 10, 64))
        lora.train()
        out_train = lora(x)
        mx.eval(out_train)

        lora.eval()
        out_eval = lora(x)
        mx.eval(out_eval)

        assert out_train.shape == out_eval.shape

    def test_apply_lora_to_sequential(self):
        from mlx_vla.training.lora import apply_lora, LoRALayer

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        model = apply_lora(model, rank=4, target_modules=None)

        count = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALayer))
        assert count == 2

        x = mx.random.normal((2, 32))
        out = model(x)
        mx.eval(out)
        assert out.shape == (2, 32)


class TestVisionEncoderForwardPasses:
    @pytest.fixture(params=["clip", "dinov2", "siglip", "sam"])
    def backbone(self, request):
        return request.param

    def test_forward_shape(self, backbone):
        from mlx_vla.models.vision import VisionEncoder

        encoder = VisionEncoder(backbone=backbone, hidden_dim=256, image_size=224, pretrained=False)
        images = mx.random.normal((2, 3, 224, 224))
        out = encoder(images)
        mx.eval(out)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 256

    def test_forward_small_image(self, backbone):
        from mlx_vla.models.vision import VisionEncoder

        encoder = VisionEncoder(backbone=backbone, hidden_dim=128, image_size=112, pretrained=False)
        images = mx.random.normal((1, 3, 112, 112))
        out = encoder(images)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[2] == 128


class TestActionHeadComplete:
    def test_discrete_action_to_tokens(self):
        from mlx_vla.models.action_heads import DiscreteActionHead

        head = DiscreteActionHead(hidden_dim=64, action_dim=7, num_bins=256)
        actions = mx.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5]])
        tokens = head.action_to_tokens(actions)
        mx.eval(tokens)
        assert tokens.shape == (1, 7)

    def test_discrete_tokens_to_action(self):
        from mlx_vla.models.action_heads import DiscreteActionHead

        head = DiscreteActionHead(hidden_dim=64, action_dim=7, num_bins=256)
        tokens = mx.array([[0, 64, 128, 192, 255, 0, 128]], dtype=mx.int32)
        actions = head.tokens_to_action(tokens)
        mx.eval(actions)
        assert actions.shape == (1, 7)
        assert float(actions[0, 0]) == pytest.approx(-1.0, abs=0.01)
        assert float(actions[0, 4]) == pytest.approx(1.0, abs=0.01)

    def test_discrete_roundtrip(self):
        from mlx_vla.models.action_heads import DiscreteActionHead

        head = DiscreteActionHead(hidden_dim=64, action_dim=7, num_bins=256)
        original = mx.array([[0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]])
        tokens = head.action_to_tokens(original)
        recovered = head.tokens_to_action(tokens)
        mx.eval(recovered)
        np.testing.assert_allclose(
            np.array(original), np.array(recovered), atol=0.01
        )

    def test_diffusion_denoise(self):
        from mlx_vla.models.action_heads import DiffusionActionHead

        head = DiffusionActionHead(hidden_dim=64, action_dim=7, action_horizon=4, num_diffusion_steps=100)
        hidden = mx.random.normal((1, 5, 64))
        denoised = head.denoise(hidden, num_steps=5)
        mx.eval(denoised)
        assert denoised.shape == (1, 4, 7)

    def test_action_chunking_head(self):
        from mlx_vla.models.action_heads import ActionChunkingHead

        head = ActionChunkingHead(hidden_dim=128, action_dim=7, chunk_size=10, num_layers=2)
        hidden = mx.random.normal((2, 20, 128))
        out = head.forward(hidden)
        mx.eval(out)
        assert out.shape == (2, 10, 7)

    def test_continuous_head_different_layers(self):
        from mlx_vla.models.action_heads import ContinuousActionHead

        for n_layers in [1, 2, 3, 5]:
            head = ContinuousActionHead(hidden_dim=64, action_dim=7, num_layers=n_layers)
            hidden = mx.random.normal((1, 5, 64))
            out = head(hidden)
            mx.eval(out)
            assert out.shape[-1] == 7


class TestFusionComplete:
    def test_cross_attention_different_seq_lens(self):
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=128, language_dim=128, hidden_dim=128,
            fusion_type="cross_attention",
        )
        v = mx.random.normal((1, 20, 128))
        l = mx.random.normal((1, 5, 128))
        out = mixer(v, l)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[1] == 20

    def test_concat_different_seq_lens(self):
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=128, language_dim=128, hidden_dim=128,
            fusion_type="concat",
        )
        v = mx.random.normal((1, 20, 128))
        l = mx.random.normal((1, 5, 128))
        out = mixer(v, l)
        mx.eval(out)
        assert out.shape[1] == 5

    def test_gated_different_dims(self):
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=256, language_dim=512, hidden_dim=256,
            fusion_type="gated",
        )
        v = mx.random.normal((2, 10, 256))
        l = mx.random.normal((2, 10, 512))
        out = mixer(v, l)
        mx.eval(out)
        assert out.shape == (2, 10, 256)

    def test_qkv_fusion_output(self):
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=128, language_dim=128, hidden_dim=128,
            fusion_type="qkv_fusion",
        )
        v = mx.random.normal((2, 10, 128))
        l = mx.random.normal((2, 10, 128))
        out = mixer(v, l)
        mx.eval(out)
        assert out.shape == (2, 10, 128)

    def test_unknown_fusion_type(self):
        from mlx_vla.models.fusion import VLAMixer

        with pytest.raises(ValueError, match="Unknown fusion type"):
            VLAMixer(vision_dim=128, language_dim=128, hidden_dim=128, fusion_type="invalid")


class TestConfigComplete:
    def test_dataclass_to_dict(self):
        from mlx_vla.utils.config import ModelConfig, LoRAConfig, DataConfig

        mc = ModelConfig(name="test", action_dim=14)
        d = mc.to_dict()
        assert d["name"] == "test"
        assert d["action_dim"] == 14

        lc = LoRAConfig(rank=32)
        d2 = lc.to_dict()
        assert d2["rank"] == 32

        dc = DataConfig(batch_size=8)
        d3 = dc.to_dict()
        assert d3["batch_size"] == 8

    def test_dataclass_from_dict_ignores_extra(self):
        from mlx_vla.utils.config import ModelConfig

        mc = ModelConfig.from_dict({"name": "test", "extra_field": 123})
        assert mc.name == "test"

    def test_config_manager_to_dict(self):
        from mlx_vla.utils.config import VLAConfigManager

        cfg = VLAConfigManager.from_default()
        d = cfg.to_dict()
        assert "model" in d
        assert "lora" in d
        assert "data" in d
        assert "training" in d
        assert "checkpointing" in d
        assert "logging" in d

    def test_config_save_load_yaml_roundtrip(self):
        from mlx_vla.utils.config import VLAConfigManager

        cfg = VLAConfigManager.from_default()
        cfg.model.name = "roundtrip-test"
        cfg.lora.rank = 32
        cfg.training.epochs = 10

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            cfg.save(path)
            loaded = VLAConfigManager.load(path)
            assert loaded.model.name == "roundtrip-test"
            assert loaded.lora.rank == 32
            assert loaded.training.epochs == 10
        finally:
            os.unlink(path)

    def test_config_save_load_json_roundtrip(self):
        from mlx_vla.utils.config import VLAConfigManager

        cfg = VLAConfigManager.from_default()
        cfg.data.batch_size = 16
        cfg.checkpointing.save_steps = 1000

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            cfg.save(path)
            loaded = VLAConfigManager.load(path)
            assert loaded.data.batch_size == 16
            assert loaded.checkpointing.save_steps == 1000
        finally:
            os.unlink(path)

    def test_config_load_nonexistent_raises(self):
        from mlx_vla.utils.config import VLAConfigManager

        with pytest.raises(FileNotFoundError):
            VLAConfigManager.load("/nonexistent/path.yaml")

    def test_load_config_function(self):
        from mlx_vla.utils.config import load_config

        cfg = load_config()
        assert cfg.model.name == "openvla-7b"

        cfg2 = load_config(model={"name": "custom"})
        assert cfg2.model.name == "custom"

    def test_global_config_lifecycle(self):
        from mlx_vla.utils.config import get_global_config, set_global_config, VLAConfigManager

        cfg = VLAConfigManager.from_default()
        cfg.model.name = "lifecycle-test"
        set_global_config(cfg)

        got = get_global_config()
        assert got.model.name == "lifecycle-test"


class TestPretrainedConfig:
    def test_get_model_config_known_models(self):
        from mlx_vla.utils.pretrained import get_model_config

        for name in ["openvla-7b", "openvla-3b", "llava-1.5-7b", "octo-small", "octo-base"]:
            cfg = get_model_config(name)
            assert "vision_backbone" in cfg
            assert "action_dim" in cfg

    def test_get_model_config_unknown_model(self):
        from mlx_vla.utils.pretrained import get_model_config

        cfg = get_model_config("totally-unknown-model")
        assert cfg["vision_backbone"] == "clip"
        assert cfg["action_dim"] == 7

    def test_get_default_config(self):
        from mlx_vla.utils.pretrained import get_default_config

        cfg = get_default_config("some/openvla-7b")
        assert cfg["vision_backbone"] == "clip"


class TestSchedulerBehavior:
    def test_cosine_schedule_warmup(self):
        from mlx_vla.training.optimizers import create_scheduler
        import mlx.optimizers as optim

        opt = optim.AdamW(learning_rate=1e-4)
        sched = create_scheduler(opt, 1000, warmup_ratio=0.1, scheduler_type="cosine")

        assert float(sched(0)) == 0.0
        assert float(sched(50)) == pytest.approx(0.5, abs=0.01)
        assert float(sched(100)) == pytest.approx(1.0, abs=0.01)
        assert float(sched(1000)) == pytest.approx(0.0, abs=0.01)

    def test_linear_schedule_warmup(self):
        from mlx_vla.training.optimizers import create_scheduler
        import mlx.optimizers as optim

        opt = optim.AdamW(learning_rate=1e-4)
        sched = create_scheduler(opt, 1000, warmup_ratio=0.1, scheduler_type="linear")

        assert sched(0) == 0.0
        assert sched(100) == pytest.approx(1.0, abs=0.01)
        assert sched(1000) == pytest.approx(0.0, abs=0.01)

    def test_constant_schedule_after_warmup(self):
        from mlx_vla.training.optimizers import create_scheduler
        import mlx.optimizers as optim

        opt = optim.AdamW(learning_rate=1e-4)
        sched = create_scheduler(opt, 1000, warmup_ratio=0.1, scheduler_type="constant")

        assert sched(0) == 0.0
        assert sched(100) == 1.0
        assert sched(500) == 1.0
        assert sched(999) == 1.0


class TestCallbackComplete:
    def test_checkpoint_callback_save(self):
        from mlx_vla.training.callbacks import CheckpointCallback
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="continuous", image_size=112,
        )

        class FakeTrainer:
            def __init__(self, m):
                self.model = m
                self.global_step = 500
                self.epoch = 0
                self.metrics = {"loss": 0.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(output_dir=tmpdir, save_steps=500, save_total_limit=2)
            trainer = FakeTrainer(model)
            cb.on_step_end(trainer, 500, 0.5)
            assert len(cb.checkpoints) == 1
            assert os.path.exists(cb.checkpoints[0])

    def test_logging_callback_write(self):
        from mlx_vla.training.callbacks import LoggingCallback

        class FakeTrainer:
            epoch = 0
            learning_rate = 1e-4

        with tempfile.TemporaryDirectory() as tmpdir:
            cb = LoggingCallback(log_dir=tmpdir, log_steps=10)
            trainer = FakeTrainer()
            cb.on_step_end(trainer, 10, 0.5)
            assert len(cb.logs) == 1
            assert os.path.exists(os.path.join(tmpdir, "logs.jsonl"))

    def test_early_stopping_triggers(self):
        from mlx_vla.training.callbacks import EarlyStoppingCallback

        class FakeTrainer:
            should_stop = False

        cb = EarlyStoppingCallback(patience=3, min_delta=0.01)
        trainer = FakeTrainer()

        cb.on_epoch_end(trainer, 0, {"loss": 1.0})
        assert not trainer.should_stop
        cb.on_epoch_end(trainer, 1, {"loss": 1.0})
        assert not trainer.should_stop
        cb.on_epoch_end(trainer, 2, {"loss": 1.0})
        assert not trainer.should_stop
        cb.on_epoch_end(trainer, 3, {"loss": 1.0})
        assert trainer.should_stop

    def test_early_stopping_resets_on_improvement(self):
        from mlx_vla.training.callbacks import EarlyStoppingCallback

        class FakeTrainer:
            should_stop = False

        cb = EarlyStoppingCallback(patience=2, min_delta=0.01)
        trainer = FakeTrainer()

        cb.on_epoch_end(trainer, 0, {"loss": 1.0})
        cb.on_epoch_end(trainer, 1, {"loss": 1.0})
        cb.on_epoch_end(trainer, 2, {"loss": 0.5})
        assert not trainer.should_stop
        assert cb.wait == 0


class TestNormalizerComplete:
    def test_all_robot_configs(self):
        from mlx_vla.data.normalizer import ActionNormalizer

        for robot in ["bridge_orig", "widowx_250", "franka", "panda", "kuka"]:
            norm = ActionNormalizer(robot)
            action = np.zeros(7)
            n = norm.normalize(action)
            u = norm.unnormalize(n)
            assert n.shape == (7,)
            assert u.shape == (7,)

    def test_normalizer_large_action_dim(self):
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("bridge_orig", action_dim=14)
        assert len(norm.action_min) == 14
        assert len(norm.action_max) == 14

    def test_normalizer_small_action_dim(self):
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("bridge_orig", action_dim=3)
        assert len(norm.action_min) == 3
        assert len(norm.action_max) == 3

    def test_from_model_mapping(self):
        from mlx_vla.data.normalizer import ActionNormalizer

        assert ActionNormalizer.from_model("openvla-7b").robot == "bridge_orig"
        assert ActionNormalizer.from_model("bridge_v2").robot == "widowx_250"
        assert ActionNormalizer.from_model("rt-1").robot == "panda"
        assert ActionNormalizer.from_model("unknown").robot == "bridge_orig"


class TestDataloaderComplete:
    def _make_dataset(self, n):
        from mlx_vla.data.dataset import VLADataset

        class D(VLADataset):
            def _load_episodes(self_inner):
                return [{"steps": [{"image": None, "action": [0] * 7}]} for _ in range(n)]

        return D("dummy")

    def test_len_drop_last(self):
        from mlx_vla.data.dataloader import VLADataloader

        ds = self._make_dataset(5)
        loader = VLADataloader(ds, batch_size=2, drop_last=True)
        assert len(loader) == 2

    def test_len_no_drop_last(self):
        from mlx_vla.data.dataloader import VLADataloader

        ds = self._make_dataset(5)
        loader = VLADataloader(ds, batch_size=2, drop_last=False)
        assert len(loader) == 3

    def test_no_shuffle_deterministic(self):
        from mlx_vla.data.dataloader import VLADataloader

        ds = self._make_dataset(4)
        loader = VLADataloader(ds, batch_size=2, shuffle=False)
        batches1 = list(loader)
        batches2 = list(loader)
        assert len(batches1) == len(batches2) == 2

    def test_with_collate_fn(self):
        from mlx_vla.data.dataloader import VLADataloader
        from mlx_vla.data.collator import VLAModuleDataCollator

        ds = self._make_dataset(4)
        collator = VLAModuleDataCollator(image_size=112, action_dim=7)
        loader = VLADataloader(ds, batch_size=2, collate_fn=collator, shuffle=False)
        batches = list(loader)
        assert len(batches) == 2
        assert "pixel_values" in batches[0]
        assert "action" in batches[0]


class TestCollatorComplete:
    def test_preprocess_float_image(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=112)
        img = np.random.rand(112, 112, 3).astype(np.float32)
        result = collator._preprocess_image(img)
        assert result.shape == (3, 112, 112)

    def test_preprocess_uint8_image(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=112)
        img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = collator._preprocess_image(img)
        assert result.shape == (3, 112, 112)
        assert result.dtype == np.float32

    def test_batch_output_types(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(image_size=112, action_dim=7)
        batch = [
            {"image": np.zeros((112, 112, 3), dtype=np.uint8), "action": np.zeros(7)},
        ]
        result = collator(batch)
        assert isinstance(result["pixel_values"], mx.array)
        assert isinstance(result["input_ids"], mx.array)
        assert isinstance(result["action"], mx.array)

    def test_normalization_0_to_1(self):
        from mlx_vla.data.collator import VLAModuleDataCollator

        collator = VLAModuleDataCollator(action_normalization="0_to_1")
        action = np.array([0.0] * 7)
        result = collator._normalize_action(action)
        assert result.dtype == np.float32


class TestDatasetComplete:
    def test_vla_dataset_iter(self):
        from mlx_vla.data.dataset import VLADataset

        class D(VLADataset):
            def _load_episodes(self):
                return [{"id": i} for i in range(3)]

        ds = D("dummy")
        items = list(ds)
        assert len(items) == 3

    def test_episode_dataset_from_json_file(self):
        from mlx_vla.data.dataset import EpisodeDataset

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{"steps": [{"action": [0] * 7}]}], f)
            path = f.name

        try:
            ds = EpisodeDataset(path)
            assert len(ds) == 1
        finally:
            os.unlink(path)

    def test_episode_dataset_directory_no_json(self):
        from mlx_vla.data.dataset import EpisodeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = EpisodeDataset(tmpdir)
            assert len(ds) == 0

    def test_episode_dataset_nonexistent_path_raises(self):
        from mlx_vla.data.dataset import EpisodeDataset

        with pytest.raises(FileNotFoundError):
            EpisodeDataset("/nonexistent/path")

    def test_episode_dataset_unsupported_format_raises(self):
        from mlx_vla.data.dataset import EpisodeDataset

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported data format"):
                EpisodeDataset(path)
        finally:
            os.unlink(path)


class TestPipelineComplete:
    def test_pipeline_with_model_object(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="continuous", image_size=112,
        )
        pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")
        assert pipeline.model is model

    def test_pipeline_predict_numpy(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="continuous", image_size=112,
        )
        pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")
        img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        action = pipeline.predict(img, "pick up the cup")
        assert isinstance(action, np.ndarray)

    def test_pipeline_predict_pil(self):
        from PIL import Image
        from mlx_vla.inference.pipeline import VLAPipeline
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="continuous", image_size=112,
        )
        pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")
        img = Image.new("RGB", (200, 200), color="blue")
        action = pipeline.predict(img, "pick up the block")
        assert isinstance(action, np.ndarray)

    def test_pipeline_stream(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        from mlx_vla.models.modeling_vla import VLAForAction

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="continuous", image_size=112,
        )
        pipeline = VLAPipeline(model=model, unnorm_key="bridge_orig")
        images = [np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(3)]
        actions = list(pipeline.stream_actions(iter(images), "pick up"))
        assert len(actions) == 3


class TestCLIImports:
    def test_cli_import(self):
        from mlx_vla.cli.main import main
        assert callable(main)

    def test_cli_subcommands(self):
        from mlx_vla.cli.main import train_command, infer_command, export_command, create_config_command
        assert callable(train_command)
        assert callable(infer_command)
        assert callable(export_command)
        assert callable(create_config_command)


class TestTopLevelImports:
    def test_all_public_imports(self):
        from mlx_vla import (
            VLAConfigManager, ModelConfig, LoRAConfig, DataConfig,
            TrainingConfig, CheckpointingConfig, LoggingConfig,
            load_config, get_global_config, set_global_config, DEFAULT_CONFIG,
            get_model_config, get_default_config,
            VLATrainer, VLADataset, VLAModuleDataCollator,
            VLADataloader, ActionNormalizer, VLA,
            VisionEncoder, CLIPVisionEncoder, DINOv2Encoder, SigLIPEncoder, SAMVisionEncoder,
            VLATrainingArguments, train_vla,
        )
        assert VLA is not None

    def test_models_init_imports(self):
        from mlx_vla.models import (
            VLAForAction, VLA, VisionEncoder, CLIPVisionEncoder,
            DINOv2Encoder, SigLIPEncoder, SAMVisionEncoder,
            VLAMixer, DiscreteActionHead, DiffusionActionHead, ContinuousActionHead,
            load_pretrained_vision_encoder,
            LanguageModelWrapper, load_language_model,
            VLALanguageEncoder, create_small_language_encoder,
        )
        assert VLAForAction is not None

    def test_data_init_imports(self):
        from mlx_vla.data import (
            VLADataset, RLDSDataset, BridgeDataset, EpisodeDataset,
            VLAModuleDataCollator, ActionNormalizer,
            VLADataloader, DatasetSampler,
            VLATokenizer, create_tokenizer,
            download_dataset, get_available_datasets,
            list_downloaded_datasets, AVAILABLE_DATASETS,
        )
        assert VLADataset is not None

    def test_training_init_imports(self):
        from mlx_vla.training import (
            VLATrainer, apply_lora, merge_lora, LoRALayer,
            create_optimizer, create_scheduler,
            Callback, CheckpointCallback, LoggingCallback,
        )
        assert VLATrainer is not None

    def test_inference_init_imports(self):
        from mlx_vla.inference import VLAPipeline
        assert VLAPipeline is not None


class TestDownloadModule:
    def test_available_datasets(self):
        from mlx_vla.data.download import get_available_datasets, AVAILABLE_DATASETS

        datasets = get_available_datasets()
        assert "bridge_v2" in datasets
        assert "aloha" in datasets
        assert len(datasets) == len(AVAILABLE_DATASETS)

    def test_list_downloaded_nonexistent_dir(self):
        from mlx_vla.data.download import list_downloaded_datasets

        result = list_downloaded_datasets("/nonexistent/dir/12345")
        assert result == []


class TestLanguageModelLoading:
    def test_load_language_model_fallback(self):
        from mlx_vla.models.language import load_language_model

        model, config = load_language_model("unknown-model", hidden_dim=256)
        assert config["framework"] == "mlx"
        assert config["hidden_dim"] == 256

    def test_create_small_encoder_config(self):
        from mlx_vla.models.language import create_small_language_encoder

        model, config = create_small_language_encoder(hidden_dim=512, vocab_size=10000, num_layers=4)
        assert config["hidden_dim"] == 512
        assert config["vocab_size"] == 10000
        assert config["num_layers"] == 4

        input_ids = mx.zeros((1, 5), dtype=mx.int32)
        out = model(input_ids)
        mx.eval(out)
        assert out.shape == (1, 5, 512)


class TestTrainerComputeLoss:
    def test_compute_loss_discrete(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        from mlx_vla.training.trainer import VLATrainer
        from mlx_vla.core import VLATrainingArguments
        from mlx_vla.data.dataset import VLADataset

        class DummyDS(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0] * 7}]}]

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="discrete", action_dim=7, image_size=112,
        )
        args = VLATrainingArguments(output_dir="/tmp/test_trainer", num_train_epochs=1)
        ds = DummyDS("dummy")
        trainer = VLATrainer(model=model, args=args, train_dataset=ds)

        batch = {
            "pixel_values": mx.random.normal((1, 3, 112, 112)),
            "input_ids": mx.zeros((1, 5), dtype=mx.int32),
            "attention_mask": mx.zeros((1, 5), dtype=mx.int32),
            "action": mx.zeros((1, 7)),
        }
        loss = trainer._compute_loss(model, batch)
        mx.eval(loss)
        assert loss.ndim == 0 or loss.size == 1

    def test_compute_loss_continuous(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        from mlx_vla.training.trainer import VLATrainer
        from mlx_vla.core import VLATrainingArguments
        from mlx_vla.data.dataset import VLADataset

        class DummyDS(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0] * 7}]}]

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="continuous", action_dim=7, image_size=112,
        )
        args = VLATrainingArguments(output_dir="/tmp/test_trainer", num_train_epochs=1)
        ds = DummyDS("dummy")
        trainer = VLATrainer(model=model, args=args, train_dataset=ds)

        batch = {
            "pixel_values": mx.random.normal((1, 3, 112, 112)),
            "input_ids": mx.zeros((1, 5), dtype=mx.int32),
            "attention_mask": mx.zeros((1, 5), dtype=mx.int32),
            "action": mx.zeros((1, 7)),
        }
        loss = trainer._compute_loss(model, batch)
        mx.eval(loss)
        assert float(loss) >= 0

    def test_compute_loss_diffusion(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        from mlx_vla.training.trainer import VLATrainer
        from mlx_vla.core import VLATrainingArguments
        from mlx_vla.data.dataset import VLADataset

        class DummyDS(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0] * 7}]}]

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="diffusion", action_dim=7, action_horizon=4, image_size=112,
        )
        args = VLATrainingArguments(output_dir="/tmp/test_trainer", num_train_epochs=1)
        ds = DummyDS("dummy")
        trainer = VLATrainer(model=model, args=args, train_dataset=ds)

        batch = {
            "pixel_values": mx.random.normal((1, 3, 112, 112)),
            "input_ids": mx.zeros((1, 5), dtype=mx.int32),
            "attention_mask": mx.zeros((1, 5), dtype=mx.int32),
            "action": mx.zeros((1, 7)),
        }
        loss = trainer._compute_loss(model, batch)
        mx.eval(loss)
        assert float(loss) >= 0

    def test_compute_loss_empty_batch(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        from mlx_vla.training.trainer import VLATrainer
        from mlx_vla.core import VLATrainingArguments
        from mlx_vla.data.dataset import VLADataset

        class DummyDS(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0] * 7}]}]

        model = VLAForAction(
            vision_hidden_dim=64, language_hidden_dim=64,
            action_type="discrete", image_size=112,
        )
        args = VLATrainingArguments(output_dir="/tmp/test_trainer", num_train_epochs=1)
        ds = DummyDS("dummy")
        trainer = VLATrainer(model=model, args=args, train_dataset=ds)

        batch = {}
        loss = trainer._compute_loss(model, batch)
        mx.eval(loss)
        assert float(loss) == 0.0


class TestGradClipping:
    def test_clip_grad_norm_returns_tuple(self):
        import mlx.optimizers as optim

        model = nn.Linear(10, 5)

        def loss_fn(model, x):
            return mx.mean(model(x))

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        x = mx.random.normal((2, 10))
        loss, grads = loss_and_grad(model, x)

        clipped, norm = optim.clip_grad_norm(grads, 1.0)
        mx.eval(norm)
        assert isinstance(norm, mx.array)
        assert isinstance(clipped, dict)


class TestDatasetSampler:
    def test_sampler_basic(self):
        from mlx_vla.data.dataloader import DatasetSampler
        from mlx_vla.data.dataset import VLADataset

        class D(VLADataset):
            def _load_episodes(self):
                return [
                    {"steps": [{"image": None, "action": [0] * 7, "language": "test"}]}
                    for _ in range(4)
                ]

        ds = D("dummy")
        sampler = DatasetSampler(ds, batch_size=2, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 2
        assert "pixel_values" in batches[0]
        assert "actions" in batches[0]

    def test_sampler_len(self):
        from mlx_vla.data.dataloader import DatasetSampler
        from mlx_vla.data.dataset import VLADataset

        class D(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0] * 7}]} for _ in range(5)]

        ds = D("dummy")
        sampler = DatasetSampler(ds, batch_size=2)
        assert len(sampler) == 3


class TestPhase3HardcodedValueElimination:
    def test_version_from_version_module(self):
        from mlx_vla._version import __version__
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_version_consistent_across_modules(self):
        from mlx_vla._version import __version__ as v1
        from mlx_vla import __version__ as v2
        from mlx_vla.core import __version__ as v3
        assert v1 == v2 == v3

    def test_constants_module_exists(self):
        from mlx_vla.models._constants import DEFAULT_VOCAB_SIZE, DEFAULT_LM_HIDDEN_DIM
        assert DEFAULT_VOCAB_SIZE == 32000
        assert DEFAULT_LM_HIDDEN_DIM == 4096

    def test_discrete_action_head_uses_constant_vocab(self):
        from mlx_vla.models.action_heads import DiscreteActionHead
        from mlx_vla.models._constants import DEFAULT_VOCAB_SIZE
        head = DiscreteActionHead(hidden_dim=64)
        assert head.vocab_size == DEFAULT_VOCAB_SIZE

    def test_discrete_action_head_custom_vocab(self):
        from mlx_vla.models.action_heads import DiscreteActionHead
        head = DiscreteActionHead(hidden_dim=64, vocab_size=50000)
        assert head.vocab_size == 50000

    def test_action_chunking_head_parameterized_num_heads(self):
        from mlx_vla.models.action_heads import ActionChunkingHead
        head = ActionChunkingHead(hidden_dim=128, num_heads=4)
        x = mx.random.normal((2, 10, 128))
        out = head.forward(x)
        mx.eval(out)
        assert out.shape[2] == 7

    def test_action_chunking_head_parameterized_dropout(self):
        from mlx_vla.models.action_heads import ActionChunkingHead
        head = ActionChunkingHead(hidden_dim=128, dropout=0.1)
        x = mx.random.normal((2, 10, 128))
        out = head.forward(x)
        mx.eval(out)
        assert out.shape[2] == 7

    def test_action_chunking_head_default_params(self):
        from mlx_vla.models.action_heads import ActionChunkingHead
        import inspect
        sig = inspect.signature(ActionChunkingHead.__init__)
        assert sig.parameters["num_heads"].default == 8
        assert sig.parameters["dropout"].default == 0.0

    def test_diffusion_action_head_parameterized_sigma_max(self):
        from mlx_vla.models.action_heads import DiffusionActionHead
        head = DiffusionActionHead(hidden_dim=64, sigma_max=2.0)
        assert head.sigma_max == 2.0

    def test_diffusion_action_head_default_sigma_max(self):
        from mlx_vla.models.action_heads import DiffusionActionHead
        head = DiffusionActionHead(hidden_dim=64)
        assert head.sigma_max == 1.0

    def test_diffusion_denoise_uses_instance_sigma_max(self):
        from mlx_vla.models.action_heads import DiffusionActionHead
        head = DiffusionActionHead(hidden_dim=64, action_dim=3, action_horizon=2, sigma_max=5.0)
        hidden = mx.random.normal((1, 1, 64))
        actions = head.denoise(hidden, num_steps=3)
        mx.eval(actions)
        assert actions.shape == (1, 2, 3)

    def test_create_optimizer_sgd_custom_momentum(self):
        from mlx_vla.training.optimizers import create_optimizer
        model = nn.Linear(10, 5)
        opt = create_optimizer(model, optimizer_type="sgd", momentum=0.95)
        assert opt is not None

    def test_create_optimizer_sgd_default_momentum(self):
        from mlx_vla.training.optimizers import create_optimizer
        import inspect
        sig = inspect.signature(create_optimizer)
        assert sig.parameters["momentum"].default == 0.9

    def test_dataset_action_dim_configurable(self):
        from mlx_vla.data.dataset import EpisodeDataset, DEFAULT_ACTION_DIM
        assert DEFAULT_ACTION_DIM == 7

    def test_episode_dataset_custom_action_dim(self):
        from mlx_vla.data.dataset import EpisodeDataset
        ds = EpisodeDataset.__new__(EpisodeDataset)
        ds.data_path = Path("/nonexistent")
        ds.split = "train"
        ds.image_size = 224
        ds.normalize_actions = True
        ds.action_normalization = "clip_minus_one_to_one"
        ds.action_dim = 10
        ds.episodes = []
        assert ds.action_dim == 10

    def test_episode_dataset_default_action_in_load(self):
        from mlx_vla.data.dataset import EpisodeDataset
        ds = EpisodeDataset.__new__(EpisodeDataset)
        ds.data_path = Path(".")
        ds.split = "train"
        ds.image_size = 64
        ds.normalize_actions = True
        ds.action_normalization = "clip_minus_one_to_one"
        ds.action_dim = 5
        ds.episodes = []
        with tempfile.TemporaryDirectory() as tmpdir:
            ep_file = Path(tmpdir) / "ep0.json"
            with open(ep_file, "w") as f:
                json.dump([{"image": "", "language": "test"}], f)
            ds.data_path = Path(tmpdir)
            episodes = ds._load_from_directory(Path(tmpdir))
            assert len(episodes) == 1
            action = episodes[0]["steps"][0]["action"]
            assert len(action) == 5
            assert all(a == 0 for a in action)

    def test_rlds_dataset_supported_used_in_cli(self):
        from mlx_vla.data.dataset import RLDSDataset
        assert "bridge_v2" in RLDSDataset.SUPPORTED_DATASETS
        assert "aloha" in RLDSDataset.SUPPORTED_DATASETS

    def test_config_has_eval_strategy(self):
        from mlx_vla.utils.config import TrainingConfig
        tc = TrainingConfig()
        assert tc.eval_strategy == "no"
        assert tc.eval_steps == 500

    def test_config_has_save_strategy(self):
        from mlx_vla.utils.config import CheckpointingConfig
        cc = CheckpointingConfig()
        assert cc.save_strategy == "epoch"

    def test_config_has_report_to(self):
        from mlx_vla.utils.config import LoggingConfig
        lc = LoggingConfig()
        assert lc.report_to == ["tensorboard"]

    def test_default_config_roundtrip_with_new_fields(self):
        from mlx_vla.utils.config import VLAConfigManager
        cfg = VLAConfigManager.from_default()
        d = cfg.to_dict()
        assert d["training"]["eval_strategy"] == "no"
        assert d["training"]["eval_steps"] == 500
        assert d["checkpointing"]["save_strategy"] == "epoch"
        assert d["logging"]["report_to"] == ["tensorboard"]

    def test_training_args_uses_config_for_eval_strategy(self):
        from mlx_vla.core import VLATrainingArguments
        args = VLATrainingArguments()
        assert args.eval_strategy == "no"
        assert args.eval_steps == 500
        assert args.save_strategy == "epoch"
        assert args.report_to == ["tensorboard"]

    def test_training_args_override_eval_strategy(self):
        from mlx_vla.core import VLATrainingArguments
        args = VLATrainingArguments(eval_strategy="steps", eval_steps=100)
        assert args.eval_strategy == "steps"
        assert args.eval_steps == 100

    def test_language_model_uses_constants(self):
        from mlx_vla.models.language import DEFAULT_LM_HIDDEN_DIM
        from mlx_vla.models._constants import DEFAULT_LM_HIDDEN_DIM as CONST
        assert DEFAULT_LM_HIDDEN_DIM == CONST

    def test_language_model_fallback_uses_constant_dim(self):
        from mlx_vla.models.language import load_language_model
        wrapper, config = load_language_model("nonexistent-model-xyz")
        assert config["hidden_dim"] == 4096

    def test_vla_language_encoder_default_vocab(self):
        from mlx_vla.models.language import VLALanguageEncoder
        from mlx_vla.models._constants import DEFAULT_VOCAB_SIZE
        import inspect
        sig = inspect.signature(VLALanguageEncoder.__init__)
        assert sig.parameters["vocab_size"].default == DEFAULT_VOCAB_SIZE

    def test_create_small_encoder_default_vocab(self):
        from mlx_vla.models.language import create_small_language_encoder
        from mlx_vla.models._constants import DEFAULT_VOCAB_SIZE
        import inspect
        sig = inspect.signature(create_small_language_encoder)
        assert sig.parameters["vocab_size"].default == DEFAULT_VOCAB_SIZE

    def test_trainer_collator_uses_config_defaults(self):
        from mlx_vla.utils.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["data"]["image_size"] == 224
        assert DEFAULT_CONFIG["data"]["action_normalization"] == "clip_minus_one_to_one"

    def test_modeling_vla_imports_vocab_from_constants(self):
        from mlx_vla.models.modeling_vla import DEFAULT_VOCAB_SIZE
        from mlx_vla.models._constants import DEFAULT_VOCAB_SIZE as CONST
        assert DEFAULT_VOCAB_SIZE == CONST

    def test_config_save_load_new_fields_yaml(self):
        from mlx_vla.utils.config import VLAConfigManager
        cfg = VLAConfigManager.from_default()
        cfg.training.eval_strategy = "steps"
        cfg.training.eval_steps = 200
        cfg.checkpointing.save_strategy = "steps"
        cfg.logging.report_to = ["wandb"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.yaml")
            cfg.save(path)
            loaded = VLAConfigManager.load(path)
            assert loaded.training.eval_strategy == "steps"
            assert loaded.training.eval_steps == 200
            assert loaded.checkpointing.save_strategy == "steps"
            assert loaded.logging.report_to == ["wandb"]

    def test_no_hardcoded_32000_in_action_heads(self):
        import inspect
        from mlx_vla.models import action_heads
        source = inspect.getsource(action_heads)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "32000" in line and "DEFAULT_VOCAB_SIZE" not in line and "import" not in line:
                pytest.fail(f"Hardcoded 32000 found at line {i+1}: {line.strip()}")

    def test_no_hardcoded_4096_in_language(self):
        import inspect
        from mlx_vla.models import language
        source = inspect.getsource(language)
        lines = source.split("\n")
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                count = stripped.count('"""') + stripped.count("'''")
                if count % 2 == 1:
                    in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if "4096" in line and "DEFAULT_LM_HIDDEN_DIM" not in line and "import" not in line and "#" not in line:
                pytest.fail(f"Hardcoded 4096 found at line {i+1}: {line.strip()}")


class TestEdgeCaseRegressions:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        self.model = VLAForAction(
            vision_backbone="clip", vision_hidden_dim=64, language_hidden_dim=64,
            action_type="continuous", action_dim=7, image_size=64,
        )

    def test_pipeline_rgba_pil(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8), mode="RGBA")
        a = pipe.predict(image=img, language="test")
        assert a.shape == (7,)

    def test_pipeline_rgba_numpy(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        a = pipe.predict(image=img, language="test")
        assert a.shape == (7,)

    def test_pipeline_cmyk_pil(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).convert("CMYK")
        a = pipe.predict(image=img, language="test")
        assert a.shape == (7,)

    def test_pipeline_nonexistent_path(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        a = pipe.predict(image="/nonexistent/path.jpg", language="test")
        assert a.shape == (7,)

    def test_pipeline_grayscale_pil(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L")
        a = pipe.predict(image=img, language="test")
        assert a.shape == (7,)

    def test_pipeline_palette_pil(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L").convert("P")
        a = pipe.predict(image=img, language="test")
        assert a.shape == (7,)

    def test_pipeline_1x1_image(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        a = pipe.predict(image=np.array([[[128, 128, 128]]], dtype=np.uint8), language="test")
        assert a.shape == (7,)

    def test_pipeline_large_image(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        a = pipe.predict(image=np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8), language="test")
        assert a.shape == (7,)

    def test_pipeline_float64_numpy(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        a = pipe.predict(image=np.random.rand(64, 64, 3).astype(np.float64), language="test")
        assert a.shape == (7,)

    def test_pipeline_grayscale_numpy_2d(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        a = pipe.predict(image=np.random.randint(0, 255, (64, 64), dtype=np.uint8), language="test")
        assert a.shape == (7,)

    def test_pipeline_single_channel_numpy(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        a = pipe.predict(image=np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8), language="test")
        assert a.shape == (7,)

    def test_pipeline_empty_language(self):
        from mlx_vla.inference.pipeline import VLAPipeline
        pipe = VLAPipeline(model=self.model, unnorm_key="bridge_orig")
        a = pipe.predict(image=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), language="")
        assert a.shape == (7,)

    def test_collator_nonsquare_image(self):
        from mlx_vla.data.collator import VLAModuleDataCollator
        c = VLAModuleDataCollator(image_size=64, action_dim=7, max_length=16)
        img = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        out = c([{"image": img, "action": [0] * 7}])
        assert out["pixel_values"].shape == (1, 3, 64, 64)

    def test_collator_string_action(self):
        from mlx_vla.data.collator import VLAModuleDataCollator
        c = VLAModuleDataCollator(image_size=64, action_dim=7, max_length=16)
        out = c([{"image": None, "action": "not_an_action"}])
        assert out["action"].shape == (1, 7)
        assert out["raw_action"].shape == (1, 7)

    def test_collator_none_action(self):
        from mlx_vla.data.collator import VLAModuleDataCollator
        c = VLAModuleDataCollator(image_size=64, action_dim=7, max_length=16)
        out = c([{"image": None, "action": None}])
        assert out["action"].shape == (1, 7)

    def test_collator_empty_action_list(self):
        from mlx_vla.data.collator import VLAModuleDataCollator
        c = VLAModuleDataCollator(image_size=64, action_dim=7, max_length=16)
        out = c([{"image": None, "action": []}])
        assert out["action"].shape == (1, 7)

    def test_collator_short_action(self):
        from mlx_vla.data.collator import VLAModuleDataCollator
        c = VLAModuleDataCollator(image_size=64, action_dim=7, max_length=16)
        out = c([{"image": None, "action": [0.1, 0.2]}])
        assert out["action"].shape == (1, 7)

    def test_collator_long_action(self):
        from mlx_vla.data.collator import VLAModuleDataCollator
        c = VLAModuleDataCollator(image_size=64, action_dim=7, max_length=16)
        out = c([{"image": None, "action": list(range(20))}])
        assert out["action"].shape == (1, 7)

    def test_collator_rgba_image(self):
        from mlx_vla.data.collator import VLAModuleDataCollator
        c = VLAModuleDataCollator(image_size=64, action_dim=7, max_length=16)
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        out = c([{"image": img, "action": [0] * 7}])
        assert out["pixel_values"].shape == (1, 3, 64, 64)

    def test_dataset_malformed_json_skipped(self):
        from mlx_vla.data.dataset import EpisodeDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bad.json"), "w") as f:
                f.write("{invalid json!!!}")
            with open(os.path.join(tmpdir, "good.json"), "w") as f:
                json.dump([{"image": "", "action": [0] * 7, "language": "ok"}], f)
            ds = EpisodeDataset(tmpdir, split="train", image_size=64)
            assert len(ds) == 1

    def test_load_missing_path_error_message(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        with pytest.raises(FileNotFoundError, match="config.json"):
            VLAForAction.load("/nonexistent/path/model")

    def test_load_corrupted_config_error_message(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        with tempfile.TemporaryDirectory() as tmpdir:
            m = VLAForAction(vision_backbone="clip", vision_hidden_dim=64,
                             language_hidden_dim=64, action_type="continuous",
                             action_dim=7, image_size=64)
            m.save(tmpdir)
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                f.write("{corrupted!")
            with pytest.raises(ValueError, match="[Cc]orrupted"):
                VLAForAction.load(tmpdir)

    def test_load_missing_weights_still_works(self):
        from mlx_vla.models.modeling_vla import VLAForAction
        with tempfile.TemporaryDirectory() as tmpdir:
            m = VLAForAction(vision_backbone="clip", vision_hidden_dim=64,
                             language_hidden_dim=64, action_type="continuous",
                             action_dim=7, image_size=64)
            m.save(tmpdir)
            os.remove(os.path.join(tmpdir, "model.npz"))
            m2 = VLAForAction.load(tmpdir)
            out = m2(mx.random.normal((1, 3, 64, 64)))
            mx.eval(out["action"])
            assert out["action"].shape[-1] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
