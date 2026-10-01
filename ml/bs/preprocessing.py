"""Feature preparation for the Burn Severity segmentation task.

The competition supplies aligned Sentinel-2, Sentinel-1 and auxiliary rasters.
This module converts their storage units into model features without discarding
cloudy pixels: clouds are part of the evaluated data and therefore become input
features rather than a filtering rule.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


S2_REFLECTANCE_BANDS = 9
SCL_CLASS_COUNT = 12
# ESA WorldCover v200 classes. Unknown values deliberately produce an all-zero
# vector so inference remains robust to a future unseen category.
LANDCOVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
EPSILON = 1e-6


def _one_hot(values: np.ndarray, classes: tuple[int, ...] | int) -> np.ndarray:
    """Return channels-first one-hot features without assuming valid categories."""
    if isinstance(classes, int):
        codes = np.arange(classes, dtype=values.dtype)
    else:
        codes = np.asarray(classes, dtype=values.dtype)
    return (values[None, :, :] == codes[:, None, None]).astype(np.float32)


@dataclass(frozen=True)
class FeatureStatistics:
    """Per-channel training-set statistics for continuous BS features."""

    mean: np.ndarray
    std: np.ndarray

    def save(self, path: str | Path) -> None:
        """Persist fitted statistics so training and inference use identical scaling."""
        np.savez_compressed(Path(path), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStatistics":
        with np.load(Path(path)) as data:
            return cls(data["mean"].astype(np.float32), data["std"].astype(np.float32))


class BurnSeverityPreprocessor:
    """Build normalized, cloud-aware features from one aligned BS chip.

    The 74 output channels consist of 39 continuous features, the Scene
    Classification Layer for both optical dates encoded as 12 classes each,
    and 11 ESA WorldCover classes. Continuous features are standardized with
    statistics fitted on *training chips only*.
    """

    continuous_feature_names = (
        *(f"s2_pre_{band}" for band in ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")),
        *(f"s2_post_{band}" for band in ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")),
        "s1_pre_vv",
        "s1_pre_vh",
        "s1_post_vv",
        "s1_post_vh",
        "dem",
        "slope",
        "nbr_pre",
        "nbr_post",
        "dnbr",
        "rdnbr",
        *(f"delta_{band}" for band in ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")),
        "delta_vv",
        "delta_vh",
    )
    categorical_feature_names = (
        *(f"scl_pre_{value}" for value in range(SCL_CLASS_COUNT)),
        *(f"scl_post_{value}" for value in range(SCL_CLASS_COUNT)),
        *(f"landcover_{value}" for value in LANDCOVER_CODES),
    )
    feature_names = continuous_feature_names + categorical_feature_names

    def __init__(self, statistics: FeatureStatistics | None = None, clip_zscore: float = 8.0):
        if clip_zscore <= 0:
            raise ValueError("clip_zscore must be positive")
        self.statistics = statistics
        self.clip_zscore = float(clip_zscore)

    @property
    def is_fitted(self) -> bool:
        return self.statistics is not None

    @classmethod
    def _continuous_features(cls, layers: Mapping[str, np.ndarray]) -> np.ndarray:
        required = {"s2_pre", "s2_post", "s1_pre", "s1_post", "aux"}
        missing = required.difference(layers)
        if missing:
            raise ValueError(f"BS layers are missing: {', '.join(sorted(missing))}")

        s2_pre = np.asarray(layers["s2_pre"])
        s2_post = np.asarray(layers["s2_post"])
        s1_pre = np.asarray(layers["s1_pre"])
        s1_post = np.asarray(layers["s1_post"])
        aux = np.asarray(layers["aux"])
        if s2_pre.shape[0] != 10 or s2_post.shape != s2_pre.shape:
            raise ValueError("Sentinel-2 layers must have ten matching bands (B2-B12 and SCL)")
        if s1_pre.shape[0] != 2 or s1_post.shape != s1_pre.shape:
            raise ValueError("Sentinel-1 layers must have matching VV and VH bands")
        if aux.shape[0] != 3:
            raise ValueError("auxiliary layer must contain DEM, slope and landcover")
        spatial_shape = s2_pre.shape[1:]
        if any(array.shape[1:] != spatial_shape for array in (s1_pre, s1_post, aux)):
            raise ValueError("all BS layers must share one pixel grid")

        # Sentinel-2 is L2A surface reflectance scaled by 10,000. Sentinel-1
        # stores dB values multiplied by 100. Keep their physical ranges before
        # estimating standardization statistics.
        optical_pre = np.clip(s2_pre[:S2_REFLECTANCE_BANDS].astype(np.float32) / 10_000.0, 0.0, 1.5)
        optical_post = np.clip(s2_post[:S2_REFLECTANCE_BANDS].astype(np.float32) / 10_000.0, 0.0, 1.5)
        sar_pre = s1_pre.astype(np.float32) / 100.0
        sar_post = s1_post.astype(np.float32) / 100.0
        terrain = aux[:2].astype(np.float32)

        nbr_pre = (optical_pre[6] - optical_pre[8]) / (optical_pre[6] + optical_pre[8] + EPSILON)
        nbr_post = (optical_post[6] - optical_post[8]) / (optical_post[6] + optical_post[8] + EPSILON)
        dnbr = nbr_pre - nbr_post
        rdnbr = dnbr / np.sqrt(np.abs(nbr_pre) + EPSILON)

        return np.concatenate(
            (
                optical_pre,
                optical_post,
                sar_pre,
                sar_post,
                terrain,
                nbr_pre[None],
                nbr_post[None],
                dnbr[None],
                rdnbr[None],
                optical_pre - optical_post,
                sar_pre - sar_post,
            )
        ).astype(np.float32, copy=False)

    @classmethod
    def _categorical_features(cls, layers: Mapping[str, np.ndarray]) -> np.ndarray:
        s2_pre = np.asarray(layers["s2_pre"])
        s2_post = np.asarray(layers["s2_post"])
        aux = np.asarray(layers["aux"])
        return np.concatenate(
            (
                _one_hot(s2_pre[9], SCL_CLASS_COUNT),
                _one_hot(s2_post[9], SCL_CLASS_COUNT),
                _one_hot(aux[2], LANDCOVER_CODES),
            )
        )

    def fit(self, training_layers: Iterable[Mapping[str, np.ndarray]]) -> "BurnSeverityPreprocessor":
        """Fit standardization on an iterable containing only training-split chips."""
        total: np.ndarray | None = None
        squared_total: np.ndarray | None = None
        pixel_count = 0
        for layers in training_layers:
            continuous = self._continuous_features(layers).astype(np.float64)
            if not np.isfinite(continuous).all():
                raise ValueError("non-finite continuous value found while fitting BS preprocessing")
            flattened = continuous.reshape(continuous.shape[0], -1)
            if total is None:
                total = np.zeros(flattened.shape[0], dtype=np.float64)
                squared_total = np.zeros_like(total)
            total += flattened.sum(axis=1)
            squared_total += np.square(flattened).sum(axis=1)
            pixel_count += flattened.shape[1]
        if total is None or squared_total is None or pixel_count == 0:
            raise ValueError("at least one training chip is required to fit BS preprocessing")
        mean = total / pixel_count
        variance = np.maximum(squared_total / pixel_count - np.square(mean), EPSILON)
        self.statistics = FeatureStatistics(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))
        return self

    def transform(self, layers: Mapping[str, np.ndarray]) -> np.ndarray:
        """Convert one raw BS chip into standardized channels-first model input."""
        if self.statistics is None:
            raise RuntimeError("fit the preprocessor on training chips before transform")
        continuous = self._continuous_features(layers)
        if continuous.shape[0] != len(self.continuous_feature_names):
            raise AssertionError("continuous feature definition and names are out of sync")
        if self.statistics.mean.shape != (continuous.shape[0],) or self.statistics.std.shape != (continuous.shape[0],):
            raise ValueError("statistics do not match the current BS feature schema")
        normalized = (continuous - self.statistics.mean[:, None, None]) / self.statistics.std[:, None, None]
        normalized = np.clip(np.nan_to_num(normalized, nan=0.0, posinf=self.clip_zscore, neginf=-self.clip_zscore), -self.clip_zscore, self.clip_zscore)
        features = np.concatenate((normalized.astype(np.float32), self._categorical_features(layers)))
        if features.shape[0] != len(self.feature_names):
            raise AssertionError("BS feature definition and names are out of sync")
        return features.astype(np.float32, copy=False)
