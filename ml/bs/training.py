"""Training primitives shared by BS training and ablation entry points."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ml._shared import IGNORE_INDEX
from ml.bs.augmentation import augment_bs
from ml.bs.baseline import BsMetrics, evaluate_bs
from ml.bs.data import BurnSeverityDataset
from ml.bs.preprocessing import BurnSeverityPreprocessor


FEATURE_MODES = {
    "s2": tuple(range(18)),
    "s2_aux": tuple(range(18)) + (22, 23) + tuple(range(63, 74)),
    "s2_s1_aux": tuple(range(18)) + tuple(range(18, 24)) + tuple(range(63, 74)),
    "full": tuple(range(74)),
}


def feature_indices(mode: str) -> tuple[int, ...]:
    try:
        return FEATURE_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"unknown BS feature mode {mode!r}; choose from {', '.join(FEATURE_MODES)}") from exc


class TorchBurnSeverityDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """On-the-fly preprocessing prevents stale, duplicate feature rasters."""

    def __init__(self, dataset: BurnSeverityDataset, indices: Sequence[int], preprocessor: BurnSeverityPreprocessor, feature_mode: str, augment: bool = False, seed: int = 42):
        self.dataset = dataset
        self.indices = list(indices)
        self.preprocessor = preprocessor
        self.indices_to_keep = np.asarray(feature_indices(feature_mode))
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset.load(self.indices[item], preprocessor=self.preprocessor)
        image = np.asarray(sample["image"], dtype=np.float32)
        target = np.asarray(sample["target"], dtype=np.int64)
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, item)))
            image, target = augment_bs(image, target, rng)
        return torch.from_numpy(image[self.indices_to_keep].copy()), torch.from_numpy(target.copy())


def class_weights(dataset: BurnSeverityDataset, indices: Sequence[int]) -> torch.Tensor:
    """Use restrained inverse-frequency weights; class 3 is capped at 4x."""
    counts = np.zeros(4, dtype=np.float64)
    for index in indices:
        target = dataset.load(index)["target"]
        valid = target[target != IGNORE_INDEX]
        counts += np.bincount(valid, minlength=4)[:4]
    if np.any(counts == 0):
        raise ValueError(f"at least one BS class is absent from train split: {counts.tolist()}")
    weights = np.sqrt(counts[0] / counts)
    weights[0] = 1.0
    weights[1:] = np.minimum(weights[1:], 4.0)
    return torch.tensor(weights, dtype=torch.float32)


def prediction_metrics(predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> BsMetrics:
    return evaluate_bs((item.detach().cpu().numpy() for item in predictions), (item.detach().cpu().numpy() for item in targets))


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: BsMetrics
