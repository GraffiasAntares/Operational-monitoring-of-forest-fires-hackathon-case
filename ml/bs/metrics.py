"""Metrics specific to the four-class Burn Severity task."""
from __future__ import annotations

import numpy as np

from ml._shared import valid_labelled_pixels


def class_iou(prediction: np.ndarray, target: np.ndarray, class_id: int) -> float:
    prediction, target = valid_labelled_pixels(prediction, target)
    predicted_class = prediction == class_id
    target_class = target == class_id
    union = int(np.count_nonzero(predicted_class | target_class))
    return 1.0 if union == 0 else int(np.count_nonzero(predicted_class & target_class)) / union


def burn_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = valid_labelled_pixels(prediction, target)
    predicted_burn = prediction >= 1
    target_burn = target >= 1
    union = int(np.count_nonzero(predicted_burn | target_burn))
    return 1.0 if union == 0 else int(np.count_nonzero(predicted_burn & target_burn)) / union


def severity_miou(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean([class_iou(prediction, target, class_id) for class_id in (1, 2, 3)]))
