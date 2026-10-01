"""Leakage-safe dNBR baseline for Burn Severity."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ml._shared import IGNORE_INDEX
from ml.bs.metrics import burn_iou, class_iou, severity_miou
from ml.bs.preprocessing import EPSILON


LANDCOVER_GROUPS = {
    "forest": frozenset((10, 20, 95, 100)),
    "steppe": frozenset((30,)),
    "cropland": frozenset((40,)),
    "floodplain": frozenset((80, 90)),
}
DEFAULT_GROUP = "other"


def landcover_groups(landcover: np.ndarray) -> np.ndarray:
    """Map WorldCover codes to the broad groups used by adaptive thresholds."""
    result = np.full(landcover.shape, DEFAULT_GROUP, dtype=object)
    for name, codes in LANDCOVER_GROUPS.items():
        result[np.isin(landcover, tuple(codes))] = name
    return result


def compute_dnbr(layers: Mapping[str, np.ndarray]) -> np.ndarray:
    """Calculate dNBR from the B8A and B12 surface-reflectance bands."""
    pre = np.asarray(layers["s2_pre"], dtype=np.float32)
    post = np.asarray(layers["s2_post"], dtype=np.float32)
    if pre.shape[0] < 9 or post.shape != pre.shape:
        raise ValueError("matching Sentinel-2 layers with B8A and B12 are required")
    nbr_pre = (pre[6] - pre[8]) / (pre[6] + pre[8] + EPSILON)
    nbr_post = (post[6] - post[8]) / (post[6] + post[8] + EPSILON)
    return nbr_pre - nbr_post


@dataclass(frozen=True)
class BsMetrics:
    burn_iou: float
    burn_precision: float
    burn_recall: float
    burn_f1: float
    iou_severity_1: float
    iou_severity_2: float
    iou_severity_3: float
    miou_severity: float

    @property
    def local_score(self) -> float:
        """BS-only proxy aligned with the competition's two BS components."""
        return 0.35 * self.burn_iou + 0.30 * self.miou_severity


def evaluate_bs(predictions: Iterable[np.ndarray], targets: Iterable[np.ndarray]) -> BsMetrics:
    """Micro-average metrics across all supplied chips."""
    predicted = np.concatenate([np.asarray(item).reshape(-1) for item in predictions])
    target = np.concatenate([np.asarray(item).reshape(-1) for item in targets])
    valid = target != IGNORE_INDEX
    predicted_burn = predicted[valid] >= 1
    target_burn = target[valid] >= 1
    true_positive = int(np.count_nonzero(predicted_burn & target_burn))
    false_positive = int(np.count_nonzero(predicted_burn & ~target_burn))
    false_negative = int(np.count_nonzero(~predicted_burn & target_burn))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = 1.0 if precision_denominator == 0 and recall_denominator == 0 else true_positive / max(precision_denominator, 1)
    recall = 1.0 if recall_denominator == 0 and precision_denominator == 0 else true_positive / max(recall_denominator, 1)
    f1_denominator = precision + recall
    return BsMetrics(
        burn_iou=burn_iou(predicted, target),
        burn_precision=precision,
        burn_recall=recall,
        burn_f1=1.0 if f1_denominator == 0 and recall_denominator == 0 else 2 * precision * recall / max(f1_denominator, EPSILON),
        iou_severity_1=class_iou(predicted, target, 1),
        iou_severity_2=class_iou(predicted, target, 2),
        iou_severity_3=class_iou(predicted, target, 3),
        miou_severity=severity_miou(predicted, target),
    )


class AdaptiveDnbrBaseline:
    """Classify dNBR with thresholds calibrated per broad land-cover group."""

    def __init__(self, thresholds: Mapping[str, tuple[float, float, float]] | None = None):
        self.thresholds = dict(thresholds or {})

    @staticmethod
    def _best_threshold(values: np.ndarray, targets: np.ndarray, severity: int) -> float:
        """Maximize IoU for ``target >= severity`` on training pixels only."""
        candidates = np.quantile(values, np.linspace(0.01, 0.99, 101))
        positive = targets >= severity
        best_threshold, best_iou = float(candidates[0]), -1.0
        for threshold in np.unique(candidates):
            prediction = values >= threshold
            union = np.count_nonzero(prediction | positive)
            score = 1.0 if union == 0 else np.count_nonzero(prediction & positive) / union
            if score > best_iou:
                best_threshold, best_iou = float(threshold), float(score)
        return best_threshold

    def fit(self, samples: Iterable[tuple[Mapping[str, np.ndarray], np.ndarray]], max_pixels_per_group: int = 250_000) -> "AdaptiveDnbrBaseline":
        """Fit thresholds using labelled train chips; validation data never enters here."""
        collected: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        counts: dict[str, int] = defaultdict(int)
        for layers, target in samples:
            dnbr = compute_dnbr(layers)
            groups = landcover_groups(np.asarray(layers["aux"])[2])
            valid = np.asarray(target) != IGNORE_INDEX
            for group in (*LANDCOVER_GROUPS, DEFAULT_GROUP):
                mask = valid & (groups == group)
                if not np.any(mask) or counts[group] >= max_pixels_per_group:
                    continue
                values, labels = dnbr[mask], np.asarray(target)[mask]
                remaining = max_pixels_per_group - counts[group]
                stride = max(1, int(np.ceil(values.size / remaining)))
                collected[group].append((values[::stride], labels[::stride]))
                counts[group] += values[::stride].size
        if not collected:
            raise ValueError("no labelled pixels available for dNBR baseline")
        self.thresholds = {}
        for group, pieces in collected.items():
            values = np.concatenate([item[0] for item in pieces])
            labels = np.concatenate([item[1] for item in pieces])
            thresholds = [self._best_threshold(values, labels, severity) for severity in (1, 2, 3)]
            self.thresholds[group] = tuple(np.maximum.accumulate(thresholds).tolist())
        # Sparse land-cover groups back off to the broad default calibration.
        default = self.thresholds.get(DEFAULT_GROUP) or next(iter(self.thresholds.values()))
        for group in (*LANDCOVER_GROUPS, DEFAULT_GROUP):
            self.thresholds.setdefault(group, default)
        return self

    def predict(self, layers: Mapping[str, np.ndarray]) -> np.ndarray:
        if not self.thresholds:
            raise RuntimeError("fit or load dNBR thresholds before prediction")
        dnbr = compute_dnbr(layers)
        groups = landcover_groups(np.asarray(layers["aux"])[2])
        result = np.zeros(dnbr.shape, dtype=np.uint8)
        for group in (*LANDCOVER_GROUPS, DEFAULT_GROUP):
            first, second, third = self.thresholds.get(group, self.thresholds[DEFAULT_GROUP])
            mask = groups == group
            result[mask] = (dnbr[mask] >= first).astype(np.uint8)
            result[mask & (dnbr >= second)] = 2
            result[mask & (dnbr >= third)] = 3
        return result

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({name: list(values) for name, values in self.thresholds.items()}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveDnbrBaseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({name: tuple(float(value) for value in values) for name, values in payload.items()})


def metrics_dict(metrics: BsMetrics) -> dict[str, float]:
    return {**asdict(metrics), "local_score": metrics.local_score}
