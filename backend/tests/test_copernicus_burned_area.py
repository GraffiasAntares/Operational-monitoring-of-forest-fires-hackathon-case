import asyncio
from datetime import date
from io import BytesIO
import json

import httpx
import numpy as np
import tifffile

from backend.app.core.config import Settings
from backend.app.services.copernicus_burned_area import (
    COPERNICUS_BURNT_AREA_V4_COLLECTION_ID,
    CopernicusBurnedAreaClient,
    vectorize_burned_area_tiff,
)


def _tiff_bytes(values: np.ndarray) -> bytes:
    stream = BytesIO()
    tifffile.imwrite(stream, values)
    return stream.getvalue()


def test_vectorizes_components_by_day_of_burn() -> None:
    raster = np.array(
        [
            [[1, 600, 800], [1, 800, 900], [0, 0, 0]],
            [[0, 0, 0], [2, 500, 700], [2, 500, 700]],
        ],
        dtype=np.int16,
    )

    features = vectorize_burned_area_tiff(
        _tiff_bytes(raster), (90.0, 55.0, 90.03, 55.02), date(2026, 1, 2)
    )

    assert len(features) == 2
    first, second = features
    assert first["properties"]["day_of_burn"] == 1
    assert first["properties"]["pixel_count"] == 2
    assert first["properties"]["burned_fraction"] == 0.7
    assert first["properties"]["confidence"] == 0.85
    assert first["properties"]["area_ha"] > 0
    assert second["properties"]["day_of_burn"] == 2
    assert second["properties"]["pixel_count"] == 2


def test_copernicus_client_requests_token_and_processes_tiff() -> None:
    received_payload: dict[str, object] = {}
    raster = np.array([[[3, 1000, 950]]], dtype=np.int16)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "identity.test":
            assert request.method == "POST"
            assert "client_credentials" in request.content.decode()
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 300})
        assert request.url.host == "process.test"
        assert request.headers["Authorization"] == "Bearer test-token"
        received_payload.update(json.loads(request.content))
        return httpx.Response(200, content=_tiff_bytes(raster), headers={"Content-Type": "image/tiff"})

    settings = Settings(
        firms_map_key=None,
        copernicus_client_id="client-id",
        copernicus_client_secret="client-secret",
        copernicus_token_url="https://identity.test/token",
        copernicus_process_url="https://process.test/api/v1/process",
    )
    client = CopernicusBurnedAreaClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    features = asyncio.run(client.fetch_burned_areas((90.0, 55.0, 90.01, 55.01), date(2026, 1, 3)))

    assert len(features) == 1
    assert features[0]["properties"]["day_of_burn"] == 3
    input_data = received_payload["input"]["data"][0]
    assert input_data["type"] == f"byoc-{COPERNICUS_BURNT_AREA_V4_COLLECTION_ID}"
    assert received_payload["output"]["responses"][0]["format"]["type"] == "image/tiff"
