"""Spatially consistent BS augmentations for full 512px competition chips."""
from __future__ import annotations

import numpy as np


S2_PRE_CONTINUOUS_CHANNELS = slice(0, 9)
S2_POST_CONTINUOUS_CHANNELS = slice(9, 18)
OPTICAL_DERIVED_CHANNELS = slice(24, 37)
S1_CONTINUOUS_CHANNELS = slice(18, 22)


def augment_bs(image: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply dihedral geometry plus restrained radar and optical jitter.

    No crop is used: a full chip is the evaluation unit and can contain a small,
    rare burned area near its edge.
    """
    turns = int(rng.integers(4))
    augmented_image = np.rot90(image, turns, axes=(1, 2)).copy()
    augmented_target = np.rot90(target, turns).copy()
    if rng.random() < 0.5:
        augmented_image = augmented_image[:, :, ::-1].copy()
        augmented_target = augmented_target[:, ::-1].copy()
    if rng.random() < 0.5:
        augmented_image = augmented_image[:, ::-1, :].copy()
        augmented_target = augmented_target[::-1, :].copy()
    # Features are train-standardized, so use very small perturbations. Separate
    # pre/post gains mimic differences in illumination and atmospheric correction.
    augmented_image[S1_CONTINUOUS_CHANNELS] += rng.normal(0.0, 0.03, size=augmented_image[S1_CONTINUOUS_CHANNELS].shape).astype(np.float32)
    augmented_image[S2_PRE_CONTINUOUS_CHANNELS] *= np.float32(rng.uniform(0.98, 1.02))
    augmented_image[S2_POST_CONTINUOUS_CHANNELS] *= np.float32(rng.uniform(0.98, 1.02))
    augmented_image[OPTICAL_DERIVED_CHANNELS] += rng.normal(0.0, 0.015, size=augmented_image[OPTICAL_DERIVED_CHANNELS].shape).astype(np.float32)
    return augmented_image, augmented_target
