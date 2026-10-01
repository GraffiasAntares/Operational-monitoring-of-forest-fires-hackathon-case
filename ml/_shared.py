"""Small primitives shared by the two independent competition tracks."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import rasterio


Kind = Literal["af", "bs"]
IGNORE_INDEX = 255


@dataclass(frozen=True)
class ChipRecord:
    chip_id: str
    kind: Kind
    metadata: dict[str, str]


def read_metadata(path: Path, kind: Kind) -> list[ChipRecord]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and "kind" in rows[0]:
        rows = [row for row in rows if row.get("kind") == kind]
    if not rows:
        raise ValueError(f"metadata has no {kind!r} records: {path}")
    records = [ChipRecord(row["chip_id"], kind, row) for row in rows]
    if len({record.chip_id for record in records}) != len(records):
        raise ValueError(f"duplicate chip_id in {path}")
    return records


def read_aligned_rasters(paths: dict[str, Path], chip_id: str) -> dict[str, np.ndarray]:
    """Read a chip and reject layers with a different georeferencing."""
    arrays: dict[str, np.ndarray] = {}
    reference_transform = None
    reference_crs = None
    for name, path in paths.items():
        with rasterio.open(path) as dataset:
            array, transform, crs = dataset.read(), dataset.transform, dataset.crs
        if reference_transform is None:
            reference_transform, reference_crs = transform, crs
        elif transform != reference_transform or crs != reference_crs:
            raise ValueError(f"unaligned raster layers for {chip_id}: {path.name}")
        arrays[name] = array
    return arrays


def grouped_split(
    records: list[ChipRecord],
    group_key: Callable[[ChipRecord], str],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create a deterministic validation split while keeping groups intact."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record.chip_id)
    if len(groups) < 2:
        raise ValueError("at least two groups are required for a grouped split")
    rng = np.random.default_rng(seed)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    validation_groups = set(group_names[:max(1, round(len(group_names) * validation_fraction))])
    return {
        "train": sorted(chip_id for group, ids in groups.items() if group not in validation_groups for chip_id in ids),
        "validation": sorted(chip_id for group, ids in groups.items() if group in validation_groups for chip_id in ids),
    }


def valid_labelled_pixels(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")
    mask = target != IGNORE_INDEX
    return prediction[mask], target[mask]
