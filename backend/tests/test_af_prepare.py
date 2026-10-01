import numpy as np

from ml.af_prepare import _box_mean, build_af_features, normalize_af_features, strict_spatiotemporal_split
from ml.data import ChipRecord


def test_box_mean_preserves_constant_valid_field() -> None:
    values = np.full((5, 5), 12.0, dtype=np.float32)
    assert np.allclose(_box_mean(values, np.ones((5, 5), dtype=bool), size=3), values)


def test_af_features_append_thermal_channels() -> None:
    image = np.zeros((13, 3, 3), dtype=np.float32)
    image[3] = 350
    image[4] = 300
    features = build_af_features({"image": image, "valid_mask": np.ones((3, 3), dtype=bool)})
    assert features.shape == (15, 3, 3)
    assert np.all(features[13] == 50)
    assert np.allclose(features[14], 0)


def test_normalization_imputes_missing_continuous_values() -> None:
    features = np.zeros((15, 2, 2), dtype=np.float32)
    features[0, 0, 0] = np.nan
    result = normalize_af_features(features, np.ones(15), np.ones(15))
    assert np.isfinite(result).all()
    assert result[0, 0, 0] == 0


def test_strict_split_excludes_crossed_spatial_or_seasonal_examples() -> None:
    class Source:
        records = [
            ChipRecord("validation", "af", {"epsg": "32637.0", "x_min": "400000", "x_max": "496000", "y_min": "5200000", "y_max": "5296000", "acq_datetime": "2024-09-01T00:00:00+00:00"}),
            ChipRecord("train", "af", {"epsg": "32638.0", "x_min": "400000", "x_max": "496000", "y_min": "5200000", "y_max": "5296000", "acq_datetime": "2024-04-01T00:00:00+00:00"}),
            ChipRecord("excluded_space", "af", {"epsg": "32637.0", "x_min": "400000", "x_max": "496000", "y_min": "5200000", "y_max": "5296000", "acq_datetime": "2024-04-01T00:00:00+00:00"}),
            ChipRecord("excluded_time", "af", {"epsg": "32638.0", "x_min": "400000", "x_max": "496000", "y_min": "5200000", "y_max": "5296000", "acq_datetime": "2024-09-01T00:00:00+00:00"}),
        ]
    assert strict_spatiotemporal_split(Source()) == {"train": ["train"], "validation": ["validation"], "excluded": ["excluded_space", "excluded_time"]}
