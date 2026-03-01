import os
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from PIL import Image
import mlx.core as mx

class VLADataset:
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        image_size: int = 224,
        normalize_actions: bool = True,
        action_normalization: str = "clip_minus_one_to_one",
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.normalize_actions = normalize_actions
        self.action_normalization = action_normalization

        self.episodes = self._load_episodes()

    def _load_episodes(self) -> List[Dict]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.episodes[idx]

    def __iter__(self) -> Iterator[Dict]:
        return iter(self.episodes)

class RLDSDataset(VLADataset):
    SUPPORTED_DATASETS = [
        "bridge_v2",
        "oxe/bridge_v2",
        "oxe/rx1",
        "oxe/franka_kitchen",
        "oxe/taco",
        "aloha",
    ]

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_size: int = 224,
        normalize_actions: bool = True,
    ):
        self.dataset_name = dataset_name
        super().__init__(
            data_path=dataset_name,
            split=split,
            image_size=image_size,
            normalize_actions=normalize_actions,
        )

    def _load_episodes(self) -> List[Dict]:
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow_datasets is required for RLDS datasets. "
                "Install with: pip install tensorflow tensorflow-datasets"
            )

        try:
            dataset = tfds.load(self.dataset_name, split=self.split)
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset '{self.dataset_name}': {e}. "
                f"Supported datasets: {self.SUPPORTED_DATASETS}"
            )

        episodes = []
        for episode in dataset:
            steps = []
            for step in episode["steps"]:
                image = step["observation"]["image"].numpy()
                action = step["action"].numpy()
                language = step["observation"]["language_instruction"].numpy()
                language = language.decode() if isinstance(language, bytes) else language

                steps.append({
                    "image": image,
                    "action": action,
                    "language": language,
                })
            episodes.append({"steps": steps})

        return episodes

class BridgeDataset(VLADataset):
    def __init__(
        self,
        data_path: str = "bridge_v2",
        split: str = "train",
        image_size: int = 224,
    ):
        super().__init__(data_path, split, image_size)

    def _load_episodes(self) -> List[Dict]:
        data_root = Path(self.data_path)

        if not data_root.exists():
            raise FileNotFoundError(f"Dataset not found at {data_root}")

        split_file = data_root / f"{self.split}.hdf5"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        episodes = []
        with h5py.File(split_file, "r") as f:
            for episode_id in f.keys():
                episode = f[episode_id]
                steps = []

                for i in range(len(episode["actions"])):
                    image = episode["observations"]["image_0"][i]
                    action = episode["actions"][i]
                    language = episode.get("language", [b""])[i] if i == 0 else b""

                    steps.append({
                        "image": image,
                        "action": action,
                        "language": language.decode() if isinstance(language, bytes) else language,
                    })

                episodes.append({"steps": steps})

        return episodes

class EpisodeDataset(VLADataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        image_size: int = 224,
    ):
        super().__init__(data_path, split, image_size)

    def _load_episodes(self) -> List[Dict]:
        data_path = Path(self.data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if data_path.is_dir():
            return self._load_from_directory(data_path)
        elif data_path.suffix == ".json":
            return self._load_from_json(data_path)
        elif data_path.suffix == ".hdf5":
            return self._load_from_hdf5(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")

    def _load_from_directory(self, data_path: Path) -> List[Dict]:
        episodes = []
        for episode_file in sorted(data_path.glob("*.json")):
            with open(episode_file, "r") as f:
                episode_data = json.load(f)

            steps = []
            for step in episode_data.get("steps", []):
                img_path_str = step.get("image")
                if img_path_str is None or img_path_str == "":
                    image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                else:
                    img_path = data_path / img_path_str
                    if img_path.exists():
                        image = np.array(Image.open(img_path).resize((self.image_size, self.image_size)))
                    else:
                        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

                steps.append({
                    "image": image,
                    "action": np.array(step.get("action", [0] * 7)),
                    "language": step.get("language", ""),
                })

            episodes.append({"steps": steps})

        return episodes

    def _load_from_json(self, data_path: Path) -> List[Dict]:
        with open(data_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        return [data]

    def _load_from_hdf5(self, data_path: Path) -> List[Dict]:
        episodes = []
        with h5py.File(data_path, "r") as f:
            for episode_id in f.keys():
                episode = f[episode_id]
                steps = []

                for i in range(len(episode["actions"])):
                    steps.append({
                        "image": episode["observations"]["image_0"][i],
                        "action": episode["actions"][i],
                        "language": "",
                    })

                episodes.append({"steps": steps})

        return episodes