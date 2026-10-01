"""Metrics specific to the binary Active Fire task."""
from __future__ import annotations

import numpy as np

from ml._shared import valid_labelled_pixels


def binary_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = valid_labelled_pixels(prediction, target)
    predicted_positive = prediction == 1
    target_positive = target == 1
    tp = int(np.count_nonzero(predicted_positive & target_positive))
    fp = int(np.count_nonzero(predicted_positive & ~target_positive))
    fn = int(np.count_nonzero(~predicted_positive & target_positive))
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2 * tp / denominator
