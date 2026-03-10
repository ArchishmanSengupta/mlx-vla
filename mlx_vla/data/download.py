"""Dataset download utilities for VLA training."""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
import warnings


DEFAULT_CACHE_DIR = os.path.expanduser("~/.mlx_vla/datasets")


@dataclass
class DatasetInfo:
    """Information about a downloadable dataset."""

    name: str
    source: str  # "rlds", "huggingface", "url"
    path: str  # RLDS name or HF path or URL
    requires_auth: bool = False
    size_gb: Optional[float] = None
    description: str = ""


# Registry of available datasets
AVAILABLE_DATASETS: Dict[str, DatasetInfo] = {
    "bridge_v2": DatasetInfo(
        name="BridgeData V2",
        source="rlds",
        path="bridge_v2",
        size_gb=2.5,
        description="Robotic manipulation dataset with diverse tasks",
    ),
    "oxe/bridge_v2": DatasetInfo(
        name="BridgeData V2 (OXE)",
        source="rlds",
        path="oxe/bridge_v2",
        size_gb=2.5,
        description="BridgeData V2 in Open X-Embodiment format",
    ),
    "oxe/rx1": DatasetInfo(
        name="RX1",
        source="rlds",
        path="oxe/rx1",
        size_gb=5.0,
        description="RX1 robot manipulation dataset",
    ),
    "oxe/franka_kitchen": DatasetInfo(
        name="Franka Kitchen",
        source="rlds",
        path="oxe/franka_kitchen",
        size_gb=1.0,
        description="Franka Kitchen manipulation tasks",
    ),
    "oxe/taco": DatasetInfo(
        name="TACO",
        source="rlds",
        path="oxe/taco",
        size_gb=3.0,
        description="TACO robot dataset",
    ),
    "aloha": DatasetInfo(
        name="ALOHA",
        source="rlds",
        path="aloha",
        size_gb=10.0,
        description="ALOHA bimanual manipulation dataset",
    ),
}


def get_available_datasets() -> Dict[str, DatasetInfo]:
    """Get all available datasets.

    Returns:
        Dictionary mapping dataset names to their info
    """
    return AVAILABLE_DATASETS.copy()


def download_rlds_dataset(
    dataset_name: str,
    split: str = "train",
    data_dir: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Download an RLDS-format dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "bridge_v2")
        split: Dataset split ("train", "test", "val")
        data_dir: Directory to store dataset (default: ~/.mlx_vla/datasets/)
        force_download: Whether to re-download if exists

    Returns:
        Path to downloaded dataset

    Raises:
        ImportError: If tensorflow_datasets is not installed
        ValueError: If dataset is not found
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        raise ImportError(
            "tensorflow_datasets is required for downloading RLDS datasets. "
            "Install with: pip install tensorflow tensorflow-datasets\n"
            "Note: You may need to install tfds-nightly for some datasets."
        )

    # Determine save directory
    if data_dir is None:
        data_dir = DEFAULT_CACHE_DIR

    save_dir = Path(data_dir) / dataset_name.replace("/", "_")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if save_dir.exists() and not force_download:
        # Check if split files exist
        split_file = save_dir / f"{split}.tsv"
        if split_file.exists():
            print(f"Dataset already exists at {save_dir}. Use force_download=True to re-download.")
            return save_dir

    print(f"Downloading {dataset_name} (split: {split})...")

    try:
        # Download dataset
        ds = tfds.load(
            dataset_name,
            split=split,
            data_dir=str(save_dir / "tfds_data"),
            download=True,
            try_gcs=True,
        )

        # Convert to simpler format (this is a simplified version)
        # In practice, you'd want to save in a more efficient format
        print(f"Converting to simpler format...")

        # Save as numpy files for faster loading
        import numpy as np

        images = []
        actions = []
        language_instructions = []

        for i, example in enumerate(tfds.as_numpy(ds)):
            if "observation" in example and "image" in example["observation"]:
                img = example["observation"]["image"]
                # Resize if needed
                if img.shape[-1] == 3:
                    img = img.astype(np.uint8)
                images.append(img)

            if "action" in example:
                actions.append(example["action"])

            if "observation" in example and "language_instruction" in example["observation"]:
                lang = example["observation"]["language_instruction"]
                if isinstance(lang, bytes):
                    lang = lang.decode()
                language_instructions.append(lang)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} examples...")

        # Save as .npz
        npz_path = save_dir / f"{split}.npz"
        np.savez_compressed(
            npz_path,
            images=np.array(images, dtype=object),
            actions=np.array(actions, dtype=object),
            language_instructions=np.array(language_instructions, dtype=object),
        )

        print(f"Dataset saved to {npz_path}")
        print(f"  - Images: {len(images)}")
        print(f"  - Actions: {len(actions)}")
        print(f"  - Language instructions: {len(language_instructions)}")

        return save_dir

    except Exception as e:
        raise RuntimeError(f"Failed to download dataset {dataset_name}: {e}")


def download_huggingface_dataset(
    dataset_name: str,
    data_dir: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Download a dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "robotics/bridge_v2")
        data_dir: Directory to store dataset
        force_download: Whether to re-download if exists

    Returns:
        Path to downloaded dataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required for downloading HuggingFace datasets. "
            "Install with: pip install datasets"
        )

    if data_dir is None:
        data_dir = DEFAULT_CACHE_DIR

    save_dir = Path(data_dir) / dataset_name.replace("/", "_")

    if save_dir.exists() and not force_download:
        print(f"Dataset already exists at {save_dir}. Use force_download=True to re-download.")
        return save_dir

    print(f"Downloading {dataset_name} from HuggingFace...")

    try:
        ds = load_dataset(dataset_name, split="train")
        ds.save_to_disk(str(save_dir))
        print(f"Dataset saved to {save_dir}")
        return save_dir

    except Exception as e:
        raise RuntimeError(f"Failed to download HuggingFace dataset {dataset_name}: {e}")


def download_dataset(
    dataset_name: str,
    split: str = "train",
    data_dir: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Download a dataset by name.

    Automatically detects the source (RLDS or HuggingFace) and downloads.

    Args:
        dataset_name: Name of the dataset
        split: Dataset split
        data_dir: Directory to store dataset
        force_download: Whether to re-download

    Returns:
        Path to downloaded dataset
    """
    # Check if it's in our registry
    if dataset_name in AVAILABLE_DATASETS:
        info = AVAILABLE_DATASETS[dataset_name]

        if info.source == "rlds":
            return download_rlds_dataset(
                dataset_name=info.path,
                split=split,
                data_dir=data_dir,
                force_download=force_download,
            )
        elif info.source == "huggingface":
            return download_huggingface_dataset(
                dataset_name=info.path,
                data_dir=data_dir,
                force_download=force_download,
            )

    # Try to guess source
    if "/" in dataset_name:
        # Likely HuggingFace
        return download_huggingface_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            force_download=force_download,
        )
    else:
        # Try as RLDS
        return download_rlds_dataset(
            dataset_name=dataset_name,
            split=split,
            data_dir=data_dir,
            force_download=force_download,
        )


def list_downloaded_datasets(data_dir: Optional[str] = None) -> List[str]:
    """List downloaded datasets.

    Args:
        data_dir: Data directory to check

    Returns:
        List of downloaded dataset names
    """
    if data_dir is None:
        data_dir = DEFAULT_CACHE_DIR

    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    datasets = []
    for item in data_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            datasets.append(item.name)

    return sorted(datasets)


__all__ = [
    "DatasetInfo",
    "AVAILABLE_DATASETS",
    "get_available_datasets",
    "download_dataset",
    "download_rlds_dataset",
    "download_huggingface_dataset",
    "list_downloaded_datasets",
]
