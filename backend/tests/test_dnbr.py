from datetime import date
import numpy as np

from backend.app.services.dnbr import vectorize_dnbr
from backend.app.services.worldcover import _tile_url


def test_dnbr_assigns_severity_classes() -> None:
    dnbr = np.array([[0.12, 0.35, 0.8, 0.02]], dtype=np.float32)
    features = vectorize_dnbr(dnbr, np.ones_like(dnbr, dtype=bool), (90.0, 55.0, 90.04, 55.01), date(2025, 8, 1), date(2025, 8, 20))
    assert {item["properties"]["severity"] for item in features} == {"low", "moderate", "high"}


def test_worldcover_uses_public_cog_tile_convention() -> None:
    assert _tile_url(33, -120).endswith("ESA_WorldCover_10m_2021_v200_N33W120_Map.tif")
