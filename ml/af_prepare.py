"""Feature preparation and train-only normalization for the AF task."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ml.data import FireChipDataset, grouped_split


FEATURE_NAMES = (
    "I1", "I2", "I3", "I4", "I5", "solar_zenith", "sensor_zenith",
    "valid", "landcover", "dem", "t2m", "rh2m", "wind_speed",
    "I4_minus_I5", "I4_local_anomaly_21",
)
# valid is binary and landcover is categorical. They must not be z-scored.
CONTINUOUS_FEATURES = tuple(index for index, name in enumerate(FEATURE_NAMES) if name not in {"valid", "landcover"})


def _box_mean(values: np.ndarray, valid: np.ndarray, size: int = 21) -> np.ndarray:
    """Masked local mean with edge padding, without a SciPy dependency."""
    pad = size // 2
    padded_values = np.pad(values * valid, pad, mode="edge")
    padded_valid = np.pad(valid.astype(np.float32), pad, mode="edge")

    def window_sum(array: np.ndarray) -> np.ndarray:
        integral = np.pad(array, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]

    return window_sum(padded_values) / np.maximum(window_sum(padded_valid), 1.0)


def build_af_features(sample: dict[str, object]) -> np.ndarray:
    """Return raw AF features plus two explicitly tested thermal features."""
    image = np.asarray(sample["image"], dtype=np.float32)
    valid = np.asarray(sample["valid_mask"], dtype=bool)
    # Derived thermal features remain finite even when one raw auxiliary value
    # is absent. Raw continuous NaNs are imputed by ``normalize_af_features``.
    i4 = np.nan_to_num(image[3], nan=0.0, posinf=0.0, neginf=0.0)
    i5 = np.nan_to_num(image[4], nan=0.0, posinf=0.0, neginf=0.0)
    thermal_delta = i4 - i5
    local_anomaly = i4 - _box_mean(i4, valid)
    return np.concatenate((image, thermal_delta[None], local_anomaly[None]))


def normalize_af_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Mean-impute continuous NaNs and z-score continuous AF features."""
    result = features.astype(np.float32, copy=True)
    for channel in CONTINUOUS_FEATURES:
        values = result[channel]
        values[~np.isfinite(values)] = mean[channel]
        result[channel] = (values - mean[channel]) / std[channel]
    # valid and landcover are kept as their supplied binary/categorical values.
    result[7] = np.nan_to_num(result[7], nan=0.0)
    result[8] = np.nan_to_num(result[8], nan=0.0)
    return result


def strict_spatiotemporal_split(dataset: FireChipDataset) -> dict[str, list[str]]:
    """Hold out an autumn regional block to mimic the private-test shift.

    The reserved rectangle covers two adjacent 192 km UTM-37N blocks. Train
    excludes every chip touching that rectangle and every Sep-Oct chip. Only
    their intersection becomes validation; the remaining cross-combinations are
    deliberately unused, preventing spatial or seasonal leakage.
    """
    def touches_reserved_region(metadata: dict[str, str]) -> bool:
        if metadata.get("epsg") != "32637.0":
            return False
        return (
            float(metadata["x_min"]) < 576_000 and float(metadata["x_max"]) > 384_000
            and float(metadata["y_min"]) < 5_568_000 and float(metadata["y_max"]) > 5_184_000
        )

    def is_autumn(metadata: dict[str, str]) -> bool:
        return datetime.fromisoformat(metadata["acq_datetime"]).month in {9, 10}

    split = {"train": [], "validation": [], "excluded": []}
    for record in dataset.records:
        spatial, autumn = touches_reserved_region(record.metadata), is_autumn(record.metadata)
        if spatial and autumn:
            split["validation"].append(record.chip_id)
        elif not spatial and not autumn:
            split["train"].append(record.chip_id)
        else:
            split["excluded"].append(record.chip_id)
    if not split["train"] or not split["validation"]:
        raise ValueError("strict split unexpectedly produced an empty train or validation part")
    return split


def prepare_af_data(data_dir: str | Path, output_dir: str | Path, validation_fraction: float = .2, seed: int = 42, strict: bool = False) -> dict[str, object]:
    """Create a compact manifest and normalization JSON without copying rasters."""
    dataset = FireChipDataset(data_dir, "af")
    split = strict_spatiotemporal_split(dataset) if strict else grouped_split(dataset.records, validation_fraction=validation_fraction, seed=seed)
    train_ids = set(split["train"])
    count = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    squared_total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    positive_pixels = {name: 0 for name in split}
    valid_pixels = {name: 0 for name in split}

    for index, record in enumerate(dataset.records):
        sample = dataset.load(index)
        split_name = next(name for name, ids in split.items() if record.chip_id in ids)
        target = np.asarray(sample["target"])
        valid = np.asarray(sample["valid_mask"], dtype=bool)
        positive_pixels[split_name] += int(np.count_nonzero((target == 1) & valid))
        valid_pixels[split_name] += int(np.count_nonzero(valid))
        if split_name != "train":
            continue
        features = build_af_features(sample)
        for channel in CONTINUOUS_FEATURES:
            values = features[channel][valid].astype(np.float64)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            count[channel] += values.size
            total[channel] += values.sum()
            squared_total[channel] += np.square(values).sum()

    means = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    stds = np.ones(len(FEATURE_NAMES), dtype=np.float64)
    for channel in CONTINUOUS_FEATURES:
        if count[channel] == 0:
            raise ValueError(f"no finite train values for feature {FEATURE_NAMES[channel]}")
        means[channel] = total[channel] / count[channel]
        variance = max(squared_total[channel] / count[channel] - means[channel] ** 2, 1e-12)
        stds[channel] = np.sqrt(variance)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("chip_id", "split", "acq_datetime", "satellite", "epsg"))
        writer.writeheader()
        for record in dataset.records:
            writer.writerow({
                "chip_id": record.chip_id,
                "split": next(name for name, ids in split.items() if record.chip_id in ids),
                "acq_datetime": record.metadata.get("acq_datetime", ""),
                "satellite": record.metadata.get("satellite", ""),
                "epsg": record.metadata.get("epsg", ""),
            })
    payload = {
        "version": "af-v1-thermal",
        "data_dir": str(Path(data_dir).resolve()),
        "seed": seed,
        "split_strategy": "strict_spatiotemporal" if strict else "overpass_grouped",
        "validation_fraction": validation_fraction,
        "feature_names": FEATURE_NAMES,
        "continuous_feature_indexes": CONTINUOUS_FEATURES,
        "mean": means.tolist(),
        "std": stds.tolist(),
        "nan_fill": means.tolist(),
        "train_chip_count": len(split["train"]),
        "validation_chip_count": len(split["validation"]),
        "excluded_chip_count": len(split.get("excluded", [])),
        "positive_pixels": positive_pixels,
        "valid_pixels": valid_pixels,
        "positive_fraction": {name: positive_pixels[name] / valid_pixels[name] for name in valid_pixels},
    }
    (output / "normalization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
