import io
import os
import numpy as np
import mlx.core as mx
from typing import Dict, List, Any, Optional
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class VLAModuleDataCollator:
    IMAGE_MEAN = IMAGENET_MEAN
    IMAGE_STD = IMAGENET_STD

    def __init__(
        self,
        image_size: int = 224,
        normalize_images: bool = True,
        action_normalization: str = "clip_minus_one_to_one",
        tokenizer: Optional[Any] = None,
        action_dim: int = 7,
        max_length: int = 128,
        image_mean: Optional[np.ndarray] = None,
        image_std: Optional[np.ndarray] = None,
    ):
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.action_normalization = action_normalization
        self.tokenizer = tokenizer
        self.action_dim = action_dim
        self.max_length = max_length
        if image_mean is not None:
            self.IMAGE_MEAN = np.array(image_mean)
        if image_std is not None:
            self.IMAGE_STD = np.array(image_std)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        pixel_values = []
        input_ids = []
        attention_mask = []
        actions = []
        raw_actions = []

        for item in batch:
            if "steps" in item:
                steps = item["steps"]
            else:
                steps = [item]

            for step in steps:
                pixel_values.append(self._preprocess_image(step.get("image")))

                # Handle action - use zeros if None
                action = step.get("action")
                if action is None:
                    action = np.zeros(self.action_dim)
                normalized = self._normalize_action(action)
                actions.append(normalized)
                try:
                    raw = np.array(action, dtype=np.float32)
                    if raw.ndim == 0 or raw.size == 0:
                        raw = np.zeros(self.action_dim, dtype=np.float32)
                except (ValueError, TypeError):
                    raw = np.zeros(self.action_dim, dtype=np.float32)
                raw_actions.append(raw)

                if self.tokenizer and step.get("language"):
                    encoded = self.tokenizer(
                        step.get("language", ""),
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                        return_tensors="np",
                    )
                    input_ids.append(encoded.input_ids[0])
                    attention_mask.append(encoded.attention_mask[0])
                else:
                    input_ids.append(np.zeros(self.max_length, dtype=np.int64))
                    attention_mask.append(np.zeros(self.max_length, dtype=np.int64))

        return {
            "pixel_values": mx.array(np.array(pixel_values)),
            "input_ids": mx.array(np.array(input_ids)),
            "attention_mask": mx.array(np.array(attention_mask)),
            "action": mx.array(np.array(actions)),
            "raw_action": mx.array(np.array(raw_actions)),
        }

    def _preprocess_image(self, image: Any) -> np.ndarray:
        if image is None:
            return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        if isinstance(image, str):
            # Handle string paths - try to open, fall back to default if not found
            try:
                if os.path.exists(image):
                    image = Image.open(image)
                else:
                    # File doesn't exist, return default image
                    return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            except Exception:
                # Any error loading image, return default
                return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image)

        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image)

        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)

        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            pil_img = Image.fromarray((image * 255).clip(0, 255).astype(np.uint8))
            pil_img = pil_img.resize((self.image_size, self.image_size))
            image = np.array(pil_img).astype(np.float32) / 255.0

        if self.normalize_images:
            if image.shape[-1] == 3:
                image = ((image - self.IMAGE_MEAN) / self.IMAGE_STD).astype(np.float32)

        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))

        return image

    def _normalize_action(self, action: Any) -> np.ndarray:
        if action is None:
            return np.zeros(self.action_dim, dtype=np.float32)

        try:
            action = np.array(action, dtype=np.float32)
        except (ValueError, TypeError):
            return np.zeros(self.action_dim, dtype=np.float32)

        if action.ndim == 0 or action.size == 0:
            return np.zeros(self.action_dim, dtype=np.float32)

        action = np.pad(action, (0, max(0, self.action_dim - len(action))), mode="constant")
        action = action[:self.action_dim]

        if self.action_normalization == "clip_minus_one_to_one":
            return np.clip(action, -1, 1).astype(np.float32)
        elif self.action_normalization == "0_to_1":
            return ((action + 1) / 2).astype(np.float32)
        return action.astype(np.float32)