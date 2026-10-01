import asyncio
from datetime import date
import json

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.services.copernicus_burned_area import CopernicusUpstreamError
from backend.app.services.scene_preview import SentinelPreviewClient, select_scene_date


BBOX = (-119.0, 33.8, -118.4, 34.4)


def feature(day, cloud):
    return {"properties": {"datetime": f"{day}T10:00:00Z", "eo:cloud_cover": cloud}}


def test_catalog_uses_cql2_and_preserves_zero_cloud_cover():
    def handler(request):
        payload = json.loads(request.content)
        assert "query" not in payload
        assert payload["filter"] == "eo:cloud_cover <= 30"
        assert payload["filter-lang"] == "cql2-text"
        assert payload["bbox"] == list(BBOX)
        # Equal distance: the cloud-free observation must win, not be treated as 100%.
        return httpx.Response(200, json={"features": [
            feature("2025-01-09", 10), feature("2025-01-11", 0),
            feature("2025-01-10", None), feature("2025-01-10", 50),
        ]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await select_scene_date(client, "https://catalog.test/search", "token", BBOX, date(2025, 1, 10), 7)

    assert asyncio.run(run()) == date(2025, 1, 11)


@pytest.mark.parametrize("position, expected", [("start", "2025-01-02"), ("end", "2025-01-12")])
def test_preview_downloads_selected_distinct_endpoint(position, expected):
    def handler(request):
        if request.url.host == "identity.test":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 300})
        payload = json.loads(request.content)
        if request.url.host == "catalog.test":
            assert "query" not in payload
            return httpx.Response(200, json={"features": [
                feature("2025-01-02", 0), feature("2025-01-12", 5),
            ]})
        assert payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"] == expected + "T00:00:00Z"
        return httpx.Response(200, content=b"test-png", headers={"Content-Type": "image/png"})

    settings = Settings(
        firms_map_key=None, copernicus_client_id="id", copernicus_client_secret="secret",
        copernicus_token_url="https://identity.test/token",
        copernicus_catalog_url="https://catalog.test/search",
        copernicus_process_url="https://process.test/process",
    )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await SentinelPreviewClient(settings, client).fetch_preview(
                BBOX, date(2025, 1, 2), end=date(2025, 1, 12), position=position,
            )

    image, actual = asyncio.run(run())
    assert image == b"test-png"
    assert actual.isoformat() == expected


def test_missing_distinct_scene_is_not_silently_replaced():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"features": [feature("2025-01-02", 0)]})
        )) as client:
            return await select_scene_date(client, "https://catalog.test/search", "token", BBOX, date(2025, 1, 2), 7, {date(2025, 1, 2)})

    with pytest.raises(CopernicusUpstreamError, match="No distinct"):
        asyncio.run(run())
