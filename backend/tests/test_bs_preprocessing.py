import numpy as np
import pytest

from ml.bs.preprocessing import BurnSeverityPreprocessor, LANDCOVER_CODES


def _layers() -> dict[str, np.ndarray]:
    s2_pre = np.zeros((10, 2, 2), dtype=np.uint16)
    s2_post = np.zeros((10, 2, 2), dtype=np.uint16)
    s2_pre[6], s2_pre[8] = 8000, 2000
    s2_post[6], s2_post[8] = 2000, 8000
    s2_pre[9] = np.array([[4, 8], [3, 10]], dtype=np.uint16)
    s2_post[9] = np.array([[5, 9], [6, 7]], dtype=np.uint16)
    return {
        "s2_pre": s2_pre,
        "s2_post": s2_post,
        "s1_pre": np.full((2, 2, 2), -1200, dtype=np.int16),
        "s1_post": np.full((2, 2, 2), -1400, dtype=np.int16),
        "aux": np.stack((np.full((2, 2), 100, dtype=np.int16), np.full((2, 2), 3, dtype=np.int16), np.array([[10, 20], [90, 100]], dtype=np.int16))),
    }


def test_preprocessor_derives_normalized_cloud_and_landcover_features() -> None:
    layers = _layers()
    preprocessor = BurnSeverityPreprocessor().fit([layers])
    image = preprocessor.transform(layers)

    assert image.shape == (74, 2, 2)
    # With one constant fitting chip, standardized values are zero, but the
    # unscaled feature builder exposes the physically meaningful burn indices.
    continuous = preprocessor._continuous_features(layers)
    assert continuous[24, 0, 0] == pytest.approx(0.6, abs=1e-5)  # NBR before fire
    assert continuous[25, 0, 0] == pytest.approx(-0.6, abs=1e-5)  # NBR after fire
    assert continuous[26, 0, 0] == pytest.approx(1.2, abs=1e-5)  # dNBR
    assert continuous[27, 0, 0] == pytest.approx(1.2 / np.sqrt(0.6), abs=1e-5)
    # One-hot channels begin after 39 continuous channels.
    assert image[39 + 4, 0, 0] == 1.0
    assert image[39 + 12 + 5, 0, 0] == 1.0
    landcover_start = 39 + 24
    assert image[landcover_start + LANDCOVER_CODES.index(10), 0, 0] == 1.0
    assert image[landcover_start + LANDCOVER_CODES.index(100), 1, 1] == 1.0


def test_preprocessor_rejects_transform_before_fitting() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        BurnSeverityPreprocessor().transform(_layers())
