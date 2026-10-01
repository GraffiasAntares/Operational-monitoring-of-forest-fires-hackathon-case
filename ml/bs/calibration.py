"""Small, explicit post-training calibration helpers for BS logits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def calibrated_argmax(logits: np.ndarray, biases: np.ndarray | list[float] | tuple[float, ...] | None = None) -> np.ndarray:
    """Return class labels after adding one fixed bias per class.

    ``logits`` may be channels-last ``(..., 4)`` or channels-first
    ``(4, height, width)``.  Biases are intentionally applied before argmax,
    never to an already discrete prediction.
    """
    values = np.asarray(logits, dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32) if biases is None else np.asarray(biases, dtype=np.float32)
    if bias.shape != (4,):
        raise ValueError("BS logit biases must contain exactly four values")
    if values.ndim >= 1 and values.shape[-1] == 4:
        return np.argmax(values + bias, axis=-1).astype(np.uint8)
    if values.ndim >= 1 and values.shape[0] == 4:
        return np.argmax(values + bias[:, None, None], axis=0).astype(np.uint8)
    raise ValueError("expected a logits array with a class dimension of size four")


def load_biases(path: str | Path | None) -> np.ndarray:
    """Load calibration JSON, or return neutral biases when no path is given."""
    if path is None:
        return np.zeros(4, dtype=np.float32)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(payload["biases"], dtype=np.float32)
