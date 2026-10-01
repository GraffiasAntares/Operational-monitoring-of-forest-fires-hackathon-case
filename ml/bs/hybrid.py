"""Confidence-gated fusion of BS U-Net logits with a dNBR severity prior."""
from __future__ import annotations

import numpy as np

from ml.bs.calibration import calibrated_argmax


def _adjusted_logits(logits: np.ndarray, biases: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    bias = np.asarray(biases, dtype=np.float32)
    if bias.shape != (4,) or values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("expected flattened BS logits shaped (pixels, 4) and four biases")
    return values + bias


def _confidence(adjusted: np.ndarray) -> np.ndarray:
    stable = adjusted - adjusted.max(axis=1, keepdims=True)
    probabilities = np.exp(stable)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.max(axis=1)


def hybrid_from_flat_logits(
    logits: np.ndarray,
    dnbr_prediction: np.ndarray,
    biases: np.ndarray | list[float] | tuple[float, ...],
    mode: str,
    confidence_threshold: float,
) -> np.ndarray:
    """Fuse dNBR labels only where the calibrated U-Net is uncertain."""
    adjusted = _adjusted_logits(logits, biases)
    prediction = np.argmax(adjusted, axis=1).astype(np.uint8)
    dnbr = np.asarray(dnbr_prediction, dtype=np.uint8).reshape(-1)
    if dnbr.shape != prediction.shape or not 0 <= confidence_threshold <= 1:
        raise ValueError("dNBR labels must match logits and confidence must be in [0, 1]")
    uncertain = _confidence(adjusted) < confidence_threshold
    if mode == "replace":
        replace = uncertain
    elif mode == "promote":
        replace = uncertain & (dnbr > prediction)
    elif mode == "burn_promote":
        replace = uncertain & (prediction == 0) & (dnbr > 0)
    elif mode == "severity_only":
        replace = uncertain & (prediction > 0) & (dnbr > 0)
    else:
        raise ValueError(f"unknown hybrid mode {mode!r}")
    result = prediction.copy()
    result[replace] = dnbr[replace]
    return result


def hybrid_prediction(
    logits: np.ndarray,
    dnbr_prediction: np.ndarray,
    biases: np.ndarray | list[float] | tuple[float, ...],
    config: dict[str, object] | None,
) -> np.ndarray:
    """Apply an enabled hybrid config to channels-first logits, or calibrated argmax."""
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("expected channels-first logits shaped (4, height, width)")
    if not config or not bool(config.get("enabled", False)):
        return calibrated_argmax(values, biases)
    flattened = np.moveaxis(values, 0, -1).reshape(-1, 4)
    prediction = hybrid_from_flat_logits(
        flattened,
        np.asarray(dnbr_prediction).reshape(-1),
        biases,
        str(config["mode"]),
        float(config["confidence_threshold"]),
    )
    return prediction.reshape(values.shape[1:])
