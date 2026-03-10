from typing import Union, Iterator, Optional, Any
from pathlib import Path
import numpy as np
from PIL import Image

import mlx.core as mx

from mlx_vla.models.modeling_vla import VLAForAction
from mlx_vla.data.normalizer import ActionNormalizer
from mlx_vla.data.collator import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


class VLAPipeline:
    def __init__(
        self,
        model: Union[str, VLAForAction],
        tokenizer: Optional[Any] = None,
        device: str = "gpu",
        unnorm_key: str = "bridge_orig",
        prompt_template: Optional[str] = None,
        image_mean: Optional[np.ndarray] = None,
        image_std: Optional[np.ndarray] = None,
    ):
        if isinstance(model, str):
            self.model = VLAForAction.load(model)
        else:
            self.model = model

        self.tokenizer = tokenizer
        self.device = device
        # Use provided unnorm_key, or derive from model if available
        # The vision_backbone is not the right key - use dataset/robot config
        self.unnorm_key = unnorm_key
        self.normalizer = ActionNormalizer(unnorm_key)
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.image_mean = np.array(image_mean) if image_mean is not None else IMAGENET_MEAN
        self.image_std = np.array(image_std) if image_std is not None else IMAGENET_STD

    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        language: str,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Predict robot action from image and language instruction.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            language: Language instruction (e.g., "pick up the cup")
            temperature: Sampling temperature for action prediction

        Returns:
            Unnormalized action array
        """
        pixel_values = self._preprocess_image(image)

        if self.tokenizer:
            encoded = self.tokenizer(
                self.prompt_template.format(instruction=language),
                return_tensors="mlx",
            )
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
        else:
            input_ids = None
            attention_mask = None

        action = self.model.predict_action(
            pixel_values=pixel_values,
            input_ids=input_ids,
            temperature=temperature,
        )

        action_np = np.array(action)
        if action_np.ndim > 1 and action_np.shape[0] == 1:
            action_np = action_np.squeeze(0)
        action_unnorm = self.normalizer.unnormalize(action_np)

        return action_unnorm

    def stream_actions(
        self,
        images: Iterator,
        language: str,
        **kwargs,
    ) -> Iterator[np.ndarray]:
        for image in images:
            action = self.predict(image, language, **kwargs)
            yield action

    def _preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> mx.array:
        target = self.model.image_size

        if isinstance(image, str):
            try:
                image = Image.open(image)
            except (FileNotFoundError, OSError):
                return mx.zeros((1, 3, target, target))

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = image.resize((target, target))
            image = np.array(image)

        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        if image.shape[0] != target or image.shape[1] != target:
            pil_img = Image.fromarray((image * 255).clip(0, 255).astype(np.uint8))
            pil_img = pil_img.resize((target, target))
            image = np.array(pil_img).astype(np.float32) / 255.0

        image = (image - self.image_mean) / self.image_std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return mx.array(image)