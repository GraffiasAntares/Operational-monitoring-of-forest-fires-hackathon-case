"""Small Sentinel-2 true-colour previews for visual before/after inspection."""
from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any

import httpx

from backend.app.services.copernicus_burned_area import CopernicusUpstreamError, CopernicusBurnedAreaClient


MAX_SIDE_PX = 900
EVALSCRIPT = """//VERSION=3
function setup() {
  return { input: [{ bands: ["B02", "B03", "B04"] }], output: { bands: 3, sampleType: "AUTO" } };
}
function evaluatePixel(s) {
  return [2.5 * s.B04, 2.5 * s.B03, 2.5 * s.B02];
}
"""


def _dimensions(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    west, south, east, north = bbox
    width_m = (east - west) * 111_320 * max(math.cos(math.radians((south + north) / 2)), 0.01)
    height_m = (north - south) * 110_574
    if width_m >= height_m:
        return MAX_SIDE_PX, max(1, round(MAX_SIDE_PX * height_m / width_m))
    return max(1, round(MAX_SIDE_PX * width_m / height_m)), MAX_SIDE_PX


def _payload(bbox: tuple[float, float, float, float], center: date, window_days: int) -> dict[str, Any]:
    width, height = _dimensions(bbox)
    return {
        "input": {
            "bounds": {"bbox": list(bbox), "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{center - timedelta(days=window_days)}T00:00:00Z",
                        "to": f"{center + timedelta(days=window_days + 1)}T00:00:00Z",
                    },
                    "maxCloudCoverage": 30,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": EVALSCRIPT,
    }


async def select_scene_date(
    client: httpx.AsyncClient,
    catalog_url: str,
    token: str,
    bbox: tuple[float, float, float, float],
    center: date,
    window_days: int,
    excluded: set[date] | None = None,
) -> date:
    response = await client.post(catalog_url, json={
        "bbox": list(bbox),
        "datetime": f"{center - timedelta(days=window_days)}T00:00:00Z/{center + timedelta(days=window_days)}T23:59:59Z",
        "collections": ["sentinel-2-l2a"], "limit": 100,
        "filter": "eo:cloud_cover <= 30",
        "filter-lang": "cql2-text",
        "fields": {"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
    }, headers={"Authorization": f"Bearer {token}"})
    if response.is_error:
        raise CopernicusUpstreamError("Copernicus Sentinel-2 catalog request failed")
    candidates: list[tuple[int, float, date]] = []
    for feature in response.json().get("features", []):
        properties = feature.get("properties", {})
        raw_datetime = properties.get("datetime")
        if not raw_datetime:
            continue
        candidate = date.fromisoformat(str(raw_datetime)[:10])
        if excluded and candidate in excluded:
            continue
        raw_cloud = properties.get("eo:cloud_cover")
        cloud = float(raw_cloud) if raw_cloud is not None else 100.0
        if not math.isfinite(cloud) or not 0 <= cloud <= 30:
            continue
        candidates.append((abs((candidate - center).days), cloud, candidate))
    if not candidates:
        raise CopernicusUpstreamError("No distinct low-cloud Sentinel-2 scene was found near the requested date")
    return min(candidates)[2]


async def select_period_scene_dates(client: httpx.AsyncClient, settings, bbox, start: date, end: date, window_days: int) -> tuple[date, date]:
    token = await CopernicusBurnedAreaClient(settings, client)._access_token_for(client)
    before = await select_scene_date(client, settings.copernicus_catalog_url, token, bbox, start, window_days)
    after = await select_scene_date(client, settings.copernicus_catalog_url, token, bbox, end, window_days, {before})
    return before, after


class SentinelPreviewClient(CopernicusBurnedAreaClient):
    async def fetch_preview(self, bbox: tuple[float, float, float, float], center: date, window_days: int = 7, *, end: date | None = None, position: str = "start") -> tuple[bytes, date]:
        self._check_configuration()
        owns_client = self.http_client is None
        client = self.http_client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            selected = center
            if end is not None:
                before, after = await select_period_scene_dates(client, self.settings, bbox, center, end, window_days)
                selected = before if position == "start" else after
            response = await client.post(
                self.settings.copernicus_process_url,
                json=_payload(bbox, selected, 0),
                headers={"Authorization": f"Bearer {await self._access_token_for(client)}"},
            )
            if response.is_error:
                raise CopernicusUpstreamError("Copernicus Sentinel-2 preview request failed")
            return response.content, selected
        except httpx.HTTPError as error:
            raise CopernicusUpstreamError("Copernicus is temporarily unavailable") from error
        finally:
            if owns_client:
                await client.aclose()
