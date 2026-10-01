"""Thermal-candidate features and residual targets for AF correction models."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ml.af_prepare import _box_mean, strict_spatiotemporal_split
from ml.data import FireChipDataset


LOW_CANDIDATE_THRESHOLD = 310.0
BASELINE_THRESHOLD = 330.0
FEATURE_NAMES = (
    "I1", "I2", "I3", "I4", "I5", "solar_zenith", "sensor_zenith", "valid", "landcover", "dem", "t2m", "rh2m", "wind_speed",
    "I4_minus_I5", "I3_minus_I4", "I3_minus_I5", "I4_div_I5",
    "I4_anomaly_5", "I4_anomaly_11", "I4_anomaly_21", "I4_anomaly_41", "I4_local_std_21",
    "candidate_I4_gt_300", "candidate_I4_gt_310", "candidate_I4_gt_320", "baseline_I4_gt_330",
)
BINARY_FEATURES = frozenset({"valid", "landcover", "candidate_I4_gt_300", "candidate_I4_gt_310", "candidate_I4_gt_320", "baseline_I4_gt_330"})
CONTINUOUS_INDEXES = tuple(index for index, name in enumerate(FEATURE_NAMES) if name not in BINARY_FEATURES)
LOW_CANDIDATE_INDEX = FEATURE_NAMES.index("candidate_I4_gt_310")
BASELINE_INDEX = FEATURE_NAMES.index("baseline_I4_gt_330")


def build_residual_features(sample: dict[str, object]) -> np.ndarray:
    """Build physical thermal context plus two candidate masks for AF residual learning."""
    image = np.asarray(sample["image"], dtype=np.float32)
    valid = np.asarray(sample["valid_mask"], dtype=bool)
    i3 = np.nan_to_num(image[2], nan=0.0)
    i4 = np.nan_to_num(image[3], nan=0.0)
    i5 = np.nan_to_num(image[4], nan=0.0)
    local_mean_21 = _box_mean(i4, valid, 21)
    local_variance = np.maximum(_box_mean(np.square(i4), valid, 21) - np.square(local_mean_21), 0.0)
    candidates = tuple(((i4 > threshold) & valid).astype(np.float32)[None] for threshold in (300., LOW_CANDIDATE_THRESHOLD, 320., BASELINE_THRESHOLD))
    derived = (
        (i4 - i5)[None], (i3 - i4)[None], (i3 - i5)[None], (i4 / np.maximum(i5, 1e-3))[None],
        (i4 - _box_mean(i4, valid, 5))[None], (i4 - _box_mean(i4, valid, 11))[None],
        (i4 - local_mean_21)[None], (i4 - _box_mean(i4, valid, 41))[None], np.sqrt(local_variance)[None],
    )
    return np.concatenate((image, *derived, *candidates)).astype(np.float32, copy=False)


def normalize_residual_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    result = np.asarray(features, dtype=np.float32).copy()
    for channel in CONTINUOUS_INDEXES:
        values = result[channel]
        values[~np.isfinite(values)] = mean[channel]
        result[channel] = (values - mean[channel]) / std[channel]
    result[7] = np.nan_to_num(result[7], nan=0.0)
    result[8] = np.nan_to_num(result[8], nan=0.0)
    return result


def residual_targets(target: np.ndarray, valid: np.ndarray, baseline: np.ndarray, low_candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return add/remove labels and their candidate-restricted supervision masks."""
    fire = np.asarray(target) == 1
    baseline = np.asarray(baseline, dtype=bool)
    low_candidate = np.asarray(low_candidate, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    add = fire & ~baseline
    remove = baseline & ~fire
    # Additions cannot affect a non-candidate pixel at inference. Restricting
    # the corresponding loss creates hard thermal negatives rather than an
    # overwhelming all-background objective.
    return add, remove, valid & low_candidate, valid & baseline


def prepare_residual_data(data_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Create strict AF manifest and train-only normalization for the residual experiment."""
    dataset = FireChipDataset(data_dir, "af")
    split = strict_spatiotemporal_split(dataset)
    train_ids = set(split["train"])
    count = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    squared = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    correction_pixels = {name: {"add": 0, "remove": 0} for name in split}
    for index, record in enumerate(dataset.records):
        sample = dataset.load(index)
        split_name = next(name for name, ids in split.items() if record.chip_id in ids)
        features = build_residual_features(sample)
        valid = np.asarray(sample["valid_mask"], dtype=bool)
        baseline = features[BASELINE_INDEX] > 0
        low_candidate = features[LOW_CANDIDATE_INDEX] > 0
        add, remove, _, _ = residual_targets(np.asarray(sample["target"]), valid, baseline, low_candidate)
        correction_pixels[split_name]["add"] += int(add.sum())
        correction_pixels[split_name]["remove"] += int(remove.sum())
        if record.chip_id not in train_ids:
            continue
        for channel in CONTINUOUS_INDEXES:
            values = features[channel][valid].astype(np.float64)
            values = values[np.isfinite(values)]
            count[channel] += len(values)
            total[channel] += values.sum()
            squared[channel] += np.square(values).sum()
    mean, std = np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES))
    for channel in CONTINUOUS_INDEXES:
        if count[channel] == 0:
            raise ValueError(f"no finite train values for {FEATURE_NAMES[channel]}")
        mean[channel] = total[channel] / count[channel]
        std[channel] = np.sqrt(max(squared[channel] / count[channel] - mean[channel] ** 2, 1e-12))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("chip_id", "split"))
        writer.writeheader()
        for name, ids in split.items():
            writer.writerows({"chip_id": chip_id, "split": name} for chip_id in ids)
    payload = {
        "version": "af-residual-v1", "feature_names": FEATURE_NAMES, "continuous_indexes": CONTINUOUS_INDEXES,
        "low_candidate_threshold": LOW_CANDIDATE_THRESHOLD, "baseline_threshold": BASELINE_THRESHOLD,
        "low_candidate_index": LOW_CANDIDATE_INDEX, "baseline_index": BASELINE_INDEX,
        "mean": mean.tolist(), "std": std.tolist(), "split_strategy": "strict_spatiotemporal",
        "chip_counts": {name: len(ids) for name, ids in split.items()}, "correction_pixels": correction_pixels,
    }
    (output / "normalization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
