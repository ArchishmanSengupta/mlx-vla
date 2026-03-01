"""Additional comprehensive tests for mlx_vla modules."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path


def check_transformers_available():
    """Check if transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False


transformers_available = check_transformers_available()


class TestTokenizer:
    """Test VLATokenizer functionality."""

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_vocab_size(self):
        """Test tokenizer vocab size property."""
        from transformers import AutoTokenizer
        from mlx_vla.data.tokenizer import VLATokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vla_tokenizer = VLATokenizer(tokenizer, max_length=128)
        assert vla_tokenizer.vocab_size > 0

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_eos_token_id(self):
        """Test tokenizer eos token ID property."""
        from transformers import AutoTokenizer
        from mlx_vla.data.tokenizer import VLATokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vla_tokenizer = VLATokenizer(tokenizer)
        assert vla_tokenizer.eos_token_id is not None

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_bos_token_id(self):
        """Test tokenizer bos token ID property."""
        from transformers import AutoTokenizer
        from mlx_vla.data.tokenizer import VLATokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vla_tokenizer = VLATokenizer(tokenizer)
        # GPT-2 doesn't have BOS, so it should be None or use EOS
        assert vla_tokenizer.bos_token_id is None or isinstance(vla_tokenizer.bos_token_id, int)

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_decode(self):
        """Test tokenizer decode method."""
        from transformers import AutoTokenizer
        from mlx_vla.data.tokenizer import VLATokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vla_tokenizer = VLATokenizer(tokenizer)

        # Tokenize then decode
        text = "hello world"
        encoded = vla_tokenizer(text, return_tensors="np")
        decoded = vla_tokenizer.decode(encoded["input_ids"][0])

        assert isinstance(decoded, str)

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_batch_decode(self):
        """Test tokenizer batch_decode method."""
        from transformers import AutoTokenizer
        from mlx_vla.data.tokenizer import VLATokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vla_tokenizer = VLATokenizer(tokenizer)

        # Batch tokenize then decode
        texts = ["hello", "world"]
        encoded = vla_tokenizer(texts, return_tensors="np")
        decoded = vla_tokenizer.batch_decode(encoded["input_ids"])

        assert len(decoded) == 2

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_call_single_string(self):
        """Test tokenizer with single string input."""
        from transformers import AutoTokenizer
        from mlx_vla.data.tokenizer import VLATokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vla_tokenizer = VLATokenizer(tokenizer, max_length=64)

        result = vla_tokenizer("test input")

        assert "input_ids" in result
        assert "attention_mask" in result
        # Handle both list and array returns - check input_ids is valid
        assert result["input_ids"] is not None

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_call_list_of_strings(self):
        """Test tokenizer with list of strings."""
        from transformers import AutoTokenizer
        from mlx_vla.data.tokenizer import VLATokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vla_tokenizer = VLATokenizer(tokenizer, max_length=64)

        result = vla_tokenizer(["hello", "world"])

        assert "input_ids" in result
        # Check input_ids is valid
        assert result["input_ids"] is not None

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_mlx_tensors(self):
        """Test tokenizer returns MLX tensors."""
        from transformers import AutoTokenizer
        import mlx.core as mx
        from mlx_vla.data.tokenizer import VLATokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vla_tokenizer = VLATokenizer(tokenizer)

        result = vla_tokenizer("test", return_tensors="mlx")

        assert isinstance(result["input_ids"], mx.array)
        assert isinstance(result["attention_mask"], mx.array)

    @pytest.mark.skipif(not transformers_available, reason="transformers not installed")
    def test_tokenizer_from_config(self):
        """Test tokenizer creation from config."""
        from mlx_vla.data.tokenizer import VLATokenizer

        config = {"tokenizer_name": "gpt2", "max_length": 64}

        # This will try to load gpt2 tokenizer
        vla_tokenizer = VLATokenizer.from_config(config)

        assert vla_tokenizer.max_length == 64


class TestFusionModule:
    """Test vision-language fusion modules."""

    def test_vla_mixer_init_cross_attention(self):
        """Test VLAMixer with cross attention fusion."""
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=256,
            language_dim=256,
            hidden_dim=256,
            num_heads=8,
            num_layers=2,
            fusion_type="cross_attention",
        )

        assert mixer.fusion_type == "cross_attention"
        assert len(mixer.fusion_layers) == 2

    def test_vla_mixer_init_concat(self):
        """Test VLAMixer with concat fusion."""
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=256,
            language_dim=256,
            hidden_dim=256,
            fusion_type="concat",
        )

        assert mixer.fusion_type == "concat"

    def test_vla_mixer_init_gated(self):
        """Test VLAMixer with gated fusion."""
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=256,
            language_dim=256,
            hidden_dim=256,
            fusion_type="gated",
        )

        assert mixer.fusion_type == "gated"

    def test_vla_mixer_init_qkv(self):
        """Test VLAMixer with QKV fusion."""
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=256,
            language_dim=256,
            hidden_dim=256,
            fusion_type="qkv_fusion",
        )

        assert mixer.fusion_type == "qkv_fusion"

    def test_vla_mixer_init_unknown_type(self):
        """Test VLAMixer with unknown fusion type."""
        from mlx_vla.models.fusion import VLAMixer

        with pytest.raises(ValueError):
            VLAMixer(
                vision_dim=256,
                language_dim=256,
                hidden_dim=256,
                fusion_type="unknown",
            )

    def test_vla_mixer_forward_cross_attention(self):
        """Test VLAMixer forward pass with cross attention."""
        import mlx.core as mx
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=256,
            language_dim=256,
            hidden_dim=256,
            num_heads=8,
            fusion_type="cross_attention",
        )

        vision_embeds = mx.random.normal((2, 10, 256))
        language_embeds = mx.random.normal((2, 10, 256))  # Same seq length for cross attention

        output = mixer(vision_embeds, language_embeds)

        assert output.shape[0] == 2
        assert output.shape[-1] == 256

    def test_vla_mixer_projection_different_dims(self):
        """Test VLAMixer with different vision and language dimensions."""
        import mlx.core as mx
        from mlx_vla.models.fusion import VLAMixer

        mixer = VLAMixer(
            vision_dim=512,
            language_dim=256,
            hidden_dim=256,
            fusion_type="cross_attention",
        )

        vision_embeds = mx.random.normal((2, 10, 512))
        language_embeds = mx.random.normal((2, 10, 256))

        output = mixer(vision_embeds, language_embeds)

        assert output.shape[-1] == 256

    def test_cross_attention_fusion(self):
        """Test CrossAttentionFusion module."""
        import mlx.core as mx
        from mlx_vla.models.fusion import CrossAttentionFusion

        fusion = CrossAttentionFusion(vision_dim=128, language_dim=128, num_heads=4)

        vision = mx.random.normal((2, 10, 128))
        language = mx.random.normal((2, 10, 128))  # Same seq length

        output = fusion(vision, language)

        assert output.shape == vision.shape


class TestLanguageEncoder:
    """Test language encoder module."""

    def test_vla_language_encoder_init(self):
        """Test VLALanguageEncoder initialization with valid hidden_dim."""
        from mlx_vla.models.language import VLALanguageEncoder

        # Hidden dim must be divisible by num_heads
        encoder = VLALanguageEncoder(
            vocab_size=1000,
            hidden_dim=128,  # 128 % 4 = 0
            num_layers=2,
            num_heads=4,  # 4 divides 128
        )

        # Check that the encoder was created successfully
        assert encoder is not None
        assert hasattr(encoder, 'token_embedding')
        assert hasattr(encoder, 'transformer')
        assert hasattr(encoder, 'norm')

    def test_vla_language_encoder_forward(self):
        """Test VLALanguageEncoder forward pass."""
        import mlx.core as mx
        from mlx_vla.models.language import VLALanguageEncoder

        encoder = VLALanguageEncoder(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = encoder(input_ids)

        assert output.shape == (1, 5, 128)

    def test_vla_language_encoder_different_batch_sizes(self):
        """Test VLALanguageEncoder with different batch sizes."""
        import mlx.core as mx
        from mlx_vla.models.language import VLALanguageEncoder

        encoder = VLALanguageEncoder(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        for batch_size in [1, 2, 4, 8]:
            input_ids = mx.zeros((batch_size, 10), dtype=mx.int32)
            output = encoder(input_ids)
            assert output.shape[0] == batch_size

    def test_vla_language_encoder_different_seq_lengths(self):
        """Test VLALanguageEncoder with different sequence lengths."""
        import mlx.core as mx
        from mlx_vla.models.language import VLALanguageEncoder

        encoder = VLALanguageEncoder(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        for seq_len in [1, 5, 10, 50]:
            input_ids = mx.zeros((2, seq_len), dtype=mx.int32)
            output = encoder(input_ids)
            assert output.shape[1] == seq_len

    def test_create_small_language_encoder(self):
        """Test create_small_language_encoder helper function."""
        from mlx_vla.models.language import create_small_language_encoder

        encoder, config = create_small_language_encoder(
            hidden_dim=256,
            vocab_size=5000,
            num_layers=4,
        )

        assert config["hidden_dim"] == 256
        assert config["vocab_size"] == 5000
        assert config["num_layers"] == 4

    def test_language_model_wrapper_embedding(self):
        """Test LanguageModelWrapper with embedding only."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vla.models.language import LanguageModelWrapper

        embedding = nn.Embedding(1000, 256)
        wrapper = LanguageModelWrapper(embedding, hidden_dim=256, use_full_model=False)

        input_ids = mx.array([[1, 2, 3]])
        output = wrapper(input_ids)

        assert output.shape == (1, 3, 256)


class TestOptimizers:
    """Test optimizer creation and scheduling."""

    def test_create_optimizer_adamw(self):
        """Test create_optimizer with AdamW."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            optimizer_type="adamw",
        )

        assert optimizer is not None

    def test_create_optimizer_adam(self):
        """Test create_optimizer with Adam."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            optimizer_type="adam",
        )

        assert optimizer is not None

    def test_create_optimizer_sgd(self):
        """Test create_optimizer with SGD."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(
            model,
            learning_rate=1e-3,
            optimizer_type="sgd",
        )

        assert optimizer is not None

    def test_create_optimizer_rmsprop(self):
        """Test create_optimizer with RMSprop."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            optimizer_type="rmsprop",
        )

        assert optimizer is not None

    def test_create_optimizer_lion(self):
        """Test create_optimizer with Lion."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            optimizer_type="lion",
        )

        assert optimizer is not None

    def test_create_optimizer_unknown_type(self):
        """Test create_optimizer with unknown type defaults to AdamW."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            optimizer_type="unknown_optimizer",
        )

        assert optimizer is not None

    def test_create_scheduler_cosine(self):
        """Test cosine learning rate scheduler."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer, create_scheduler

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(model)

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_ratio=0.1,
            scheduler_type="cosine",
        )

        assert scheduler is not None

        # Test warmup phase
        assert scheduler(0) == 0.0
        assert scheduler(50) > 0.0

    def test_create_scheduler_linear(self):
        """Test linear learning rate scheduler."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer, create_scheduler

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(model)

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_ratio=0.1,
            scheduler_type="linear",
        )

        assert scheduler is not None

    def test_create_scheduler_constant(self):
        """Test constant learning rate scheduler."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer, create_scheduler

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(model)

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_ratio=0.1,
            scheduler_type="constant",
        )

        assert scheduler is not None

    def test_create_scheduler_unknown_type(self):
        """Test unknown scheduler type returns None."""
        import mlx.nn as nn
        from mlx_vla.training.optimizers import create_optimizer, create_scheduler

        model = nn.Linear(128, 128)
        optimizer = create_optimizer(model)

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            scheduler_type="unknown",
        )

        assert scheduler is None


class TestVisionEncoders:
    """Additional tests for vision encoders."""

    def test_clip_vision_encoder_valid_hidden_dims(self):
        """Test CLIP encoder with hidden dims divisible by num_heads."""
        import mlx.core as mx
        from mlx_vla.models.vision import CLIPVisionEncoder

        # Use hidden dims divisible by 12 (default num_heads)
        for hidden_dim in [384, 768, 936]:
            encoder = CLIPVisionEncoder(hidden_dim=hidden_dim, pretrained=False, image_size=224)
            images = mx.random.normal((1, 3, 224, 224))
            output = encoder(images)
            assert output.shape[-1] == hidden_dim

    def test_dinov2_encoder_valid_hidden_dims(self):
        """Test DINOv2 encoder with valid hidden dimensions."""
        import mlx.core as mx
        from mlx_vla.models.vision import DINOv2Encoder

        # Use hidden dims divisible by 16 (default num_heads)
        for hidden_dim in [512, 768, 1024]:
            encoder = DINOv2Encoder(hidden_dim=hidden_dim, pretrained=False, image_size=224)
            images = mx.random.normal((1, 3, 224, 224))
            output = encoder(images)
            assert output.shape[-1] == hidden_dim

    def test_siglip_encoder_valid_hidden_dims(self):
        """Test SigLIP encoder with valid hidden dimensions."""
        import mlx.core as mx
        from mlx_vla.models.vision import SigLIPEncoder

        # Use hidden dims divisible by 16 (default num_heads)
        for hidden_dim in [512, 768, 1024]:
            encoder = SigLIPEncoder(hidden_dim=hidden_dim, pretrained=False, image_size=224)
            images = mx.random.normal((1, 3, 224, 224))
            output = encoder(images)
            assert output.shape[-1] == hidden_dim

    def test_sam_encoder_valid_hidden_dims(self):
        """Test SAM encoder with valid hidden dimensions."""
        import mlx.core as mx
        from mlx_vla.models.vision import SAMVisionEncoder

        # Use hidden dims divisible by 12 (default num_heads)
        for hidden_dim in [384, 768, 936]:
            encoder = SAMVisionEncoder(hidden_dim=hidden_dim, pretrained=False, image_size=224)
            images = mx.random.normal((1, 3, 224, 224))
            output = encoder(images)
            assert output.shape[-1] == hidden_dim

    def test_vision_encoder_factory_all_backbones(self):
        """Test VisionEncoder factory with all backbones."""
        import mlx.core as mx
        from mlx_vla.models.vision import VisionEncoder

        # Use hidden dims divisible by default num_heads
        for backbone in ["clip", "dinov2", "siglip", "sam"]:
            encoder = VisionEncoder(
                backbone=backbone,
                hidden_dim=768,
                image_size=224,
                pretrained=False,
            )
            images = mx.random.normal((1, 3, 224, 224))
            output = encoder(images)
            assert output.shape[-1] == 768


class TestActionHeads:
    """Additional tests for action heads."""

    def test_discrete_action_head_logits_shape(self):
        """Test discrete action head output logits shape."""
        import mlx.core as mx
        from mlx_vla.models.action_heads import DiscreteActionHead

        head = DiscreteActionHead(hidden_dim=128, action_dim=7, num_bins=256)
        hidden = mx.random.normal((2, 5, 128))

        output = head.forward(hidden)

        assert output.shape == (2, 5, 7, 256)

    def test_continuous_action_head_output_shape(self):
        """Test continuous action head output shape."""
        import mlx.core as mx
        from mlx_vla.models.action_heads import ContinuousActionHead

        head = ContinuousActionHead(hidden_dim=128, action_dim=7, action_horizon=4)
        hidden = mx.random.normal((2, 5, 128))

        output = head.forward(hidden)

        assert output.shape == (2, 4, 7)

    def test_diffusion_action_head_output_shape(self):
        """Test diffusion action head output shape."""
        import mlx.core as mx
        from mlx_vla.models.action_heads import DiffusionActionHead

        head = DiffusionActionHead(hidden_dim=128, action_dim=7, action_horizon=4)
        hidden = mx.random.normal((2, 5, 128))
        noisy_actions = mx.random.normal((2, 4, 7))

        output = head.forward(hidden, noisy_actions)

        assert output.shape == (2, 4, 7)


class TestDataCollatorAdvanced:
    """Advanced data collator tests."""

    def test_collator_different_image_sizes(self):
        """Test collator with different image sizes."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        for size in [112, 224, 336, 448]:
            collator = VLAModuleDataCollator(image_size=size)
            result = collator._preprocess_image(None)
            assert result.shape == (3, size, size)

    def test_collator_different_action_normalizations(self):
        """Test collator with all action normalization types."""
        from mlx_vla.data.collator import VLAModuleDataCollator

        for norm_type in [
            "clip_minus_one_to_one",
            "zero_to_one",
            "none",
        ]:
            collator = VLAModuleDataCollator(action_normalization=norm_type)
            action = np.array([0.5, 0.0, -0.5, 1.0, -1.0, 0.25, -0.25])
            result = collator._normalize_action(action)
            assert result.shape == action.shape


class TestDataloaderAdvanced:
    """Advanced dataloader tests."""

    def test_dataloader_with_collate_fn(self):
        """Test dataloader with custom collate function."""
        from mlx_vla.data.dataloader import VLADataloader
        from mlx_vla.data.dataset import VLADataset
        from mlx_vla.data.collator import VLAModuleDataCollator

        class DummyDataset(VLADataset):
            def _load_episodes(self):
                return [{"steps": [{"image": None, "action": [0]*7}]} for _ in range(4)]

        dataset = DummyDataset("dummy")
        collator = VLAModuleDataCollator()

        loader = VLADataloader(
            dataset,
            batch_size=2,
            collate_fn=collator,
            shuffle=False,
        )

        batches = list(loader)
        assert len(batches) == 2


class TestNormalizerAdvanced:
    """Advanced normalizer tests."""

    def test_normalizer_all_robot_types(self):
        """Test normalizer with all supported robot types."""
        from mlx_vla.data.normalizer import ActionNormalizer

        robot_types = ["bridge_orig", "franka", "kuka", "sawyer", "panda"]

        for robot in robot_types:
            norm = ActionNormalizer(robot)
            action = np.zeros(7)
            result = norm.normalize(action)
            assert result.shape == action.shape

    def test_normalizer_inverse_consistency(self):
        """Test normalizer unnormalize is inverse of normalize."""
        from mlx_vla.data.normalizer import ActionNormalizer

        norm = ActionNormalizer("bridge_orig")
        original = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        normalized = norm.normalize(original)
        recovered = norm.unnormalize(normalized)

        np.testing.assert_array_almost_equal(original, recovered, decimal=5)


class TestConfigAdvanced:
    """Advanced config tests."""

    def test_config_nested_update(self):
        """Test config with deeply nested updates."""
        from mlx_vla.utils.config import VLAConfigManager

        config = VLAConfigManager.from_default()

        config.update(
            model={"name": "test", "vision_hidden_dim": 512},
            training={"learning_rate": 1e-5, "warmup_ratio": 0.2},
            lora={"alpha": 64, "dropout": 0.1},
        )

        assert config.model.vision_hidden_dim == 512
        assert config.training.learning_rate == 1e-5
        assert config.lora.alpha == 64


class TestPretrainedModels:
    """Tests for pretrained model loading."""

    def test_load_pretrained_unknown_backbone(self):
        """Test loading pretrained with unknown backbone raises error."""
        from mlx_vla.models.pretrained import load_pretrained_vision_encoder

        with pytest.raises(ValueError):
            load_pretrained_vision_encoder(backbone="unknown_backbone")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
