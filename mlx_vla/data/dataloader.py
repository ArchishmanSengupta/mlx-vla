import numpy as np
from typing import Iterator, List, Dict, Any, Optional
from pathlib import Path

class VLADataloader:
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        collate_fn = None,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.indices = None

    def __iter__(self) -> Iterator[List]:
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        batch = []
        for idx in indices:
            batch.append(self.dataset[int(idx)])

            if len(batch) == self.batch_size:
                if self.collate_fn:
                    yield self.collate_fn(batch)
                else:
                    yield batch
                batch = []

        if batch and not self.drop_last:
            if self.collate_fn:
                yield self.collate_fn(batch)
            else:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class DatasetSampler:
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        batch = []
        for idx in indices:
            batch.append(self.dataset[int(idx)])

            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []

        if batch:
            yield self._collate(batch)

    def _collate(self, batch: List[Dict]) -> Dict[str, Any]:
        pixel_values = []
        input_ids = []
        attention_mask = []
        actions = []
        raw_actions = []

        for episode in batch:
            steps = episode.get("steps", [])
            for step in steps:
                if step.get("image") is not None:
                    pixel_values.append(step["image"])
                if step.get("action") is not None:
                    actions.append(step["action"])
                    raw_actions.append(step["action"])
                if step.get("language"):
                    input_ids.append(step.get("language", ""))
                else:
                    input_ids.append("")

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "actions": actions,
            "raw_actions": raw_actions,
        }

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size