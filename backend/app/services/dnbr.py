"""CPU-only dNBR severity stage backed by Copernicus Sentinel-2 L2A."""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import math
import asyncio
from typing import Any

import httpx
import numpy as np
import tifffile
from rasterio.features import geometry_mask, shapes
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape
from shapely.ops import transform

from backend.app.core.config import Settings
from backend.app.services.cache import AsyncTtlCache
from backend.app.services.copernicus_burned_area import (
    CopernicusBurnedAreaClient,
    CopernicusUpstreamError,
    EQUAL_AREA_TRANSFORMER,
)

PIXEL_SIZE_M = 20
MAX_OUTPUT_PIXELS = 250_000
EVALSCRIPT = """//VERSION=3
function setup() { return {input:[{bands:[\"B08\",\"B12\",\"SCL\",\"dataMask\"]}], output:{bands:4,sampleType:\"FLOAT32\"}}; }
function evaluatePixel(s) { return [s.B08, s.B12, s.SCL, s.dataMask]; }
"""


def _dimensions(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    west, south, east, north = bbox
    width = max(1, math.ceil((east - west) * 111_320 * max(math.cos(math.radians((south + north) / 2)), .01) / PIXEL_SIZE_M))
    height = max(1, math.ceil((north - south) * 110_574 / PIXEL_SIZE_M))
    if width * height > MAX_OUTPUT_PIXELS:
        raise ValueError("bbox is too large for a synchronous 20 m dNBR request")
    return width, height


def _payload(bbox: tuple[float, float, float, float], center: date, window_days: int) -> dict[str, Any]:
    width, height = _dimensions(bbox)
    return {
        "input": {"bounds": {"bbox": list(bbox), "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}, "data": [{
            "type": "sentinel-2-l2a",
            "dataFilter": {"timeRange": {"from": (center - timedelta(days=window_days)).isoformat() + "T00:00:00Z", "to": (center + timedelta(days=window_days + 1)).isoformat() + "T00:00:00Z"}, "maxCloudCoverage": 30, "mosaickingOrder": "leastCC"},
        }]},
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }


def _bands(content: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(tifffile.imread(BytesIO(content)))
    if raw.ndim != 3:
        raise CopernicusUpstreamError("Sentinel-2 response has an unexpected TIFF layout")
    if raw.shape[-1] == 4:
        return tuple(np.moveaxis(raw, -1, 0))  # type: ignore[return-value]
    if raw.shape[0] == 4:
        return tuple(raw)  # type: ignore[return-value]
    raise CopernicusUpstreamError("Sentinel-2 response has an unexpected band count")


class DnbrClient(CopernicusBurnedAreaClient):
    async def _scene(self, client: httpx.AsyncClient, bbox: tuple[float, float, float, float], center: date, window_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        response = await client.post(self.settings.copernicus_process_url, json=_payload(bbox, center, window_days), headers={"Authorization": f"Bearer {await self._access_token_for(client)}"})
        if response.is_error:
            raise CopernicusUpstreamError("Copernicus Sentinel-2 Process API request failed")
        return _bands(response.content)

    async def calculate(self, bbox: tuple[float, float, float, float], before: date, after: date, window_days: int = 7) -> list[dict[str, Any]]:
        self._check_configuration()
        if before >= after:
            raise ValueError("before_date must be earlier than after_date")
        if window_days < 0 or window_days > 30:
            raise ValueError("window_days must be between 0 and 30")
        return await self._cache.get_or_set(
            ("dnbr", bbox, before, after, window_days), 3600,
            lambda: self._calculate_uncached(bbox, before, after, window_days),
        )

    async def _calculate_uncached(self, bbox: tuple[float, float, float, float], before: date, after: date, window_days: int) -> list[dict[str, Any]]:
        owns = self.http_client is None
        client = self.http_client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            before_scene, after_scene = await asyncio.gather(
                self._scene(client, bbox, before, window_days),
                self._scene(client, bbox, after, window_days),
            )
            before_nir, before_swir, before_scl, before_mask = before_scene
            after_nir, after_swir, after_scl, after_mask = after_scene
        except httpx.HTTPError as error:
            raise CopernicusUpstreamError("Copernicus is temporarily unavailable") from error
        finally:
            if owns:
                await client.aclose()
        before_nbr = (before_nir - before_swir) / np.where(before_nir + before_swir == 0, np.nan, before_nir + before_swir)
        after_nbr = (after_nir - after_swir) / np.where(after_nir + after_swir == 0, np.nan, after_nir + after_swir)
        dnbr = before_nbr - after_nbr
        # SCL excludes cloud, cirrus, cloud-shadow, snow and no-data pixels.
        usable_scl = np.isin(before_scl, [4, 5, 7]) & np.isin(after_scl, [4, 5, 7])
        valid = (before_mask > 0) & (after_mask > 0) & usable_scl & np.isfinite(dnbr)
        return await asyncio.to_thread(vectorize_dnbr, dnbr, valid, bbox, before, after)


def vectorize_dnbr(dnbr: np.ndarray, valid: np.ndarray, bbox: tuple[float, float, float, float], before: date, after: date) -> list[dict[str, Any]]:
    rows, columns = dnbr.shape
    west, south, east, north = bbox
    classes = (("high", 0.66, float("inf")), ("moderate", 0.27, 0.66), ("low", 0.10, 0.27))
    features: list[dict[str, Any]] = []
    for label, lower, upper in classes:
        mask = valid & (dnbr >= lower) & (dnbr < upper)
        transform_ = from_origin(west, north, (east - west) / columns, (north - south) / rows)
        for index, (geometry_mapping, _) in enumerate(shapes(mask.astype("uint8"), mask=mask, transform=transform_), 1):
            geometry = shape(geometry_mapping)
            area = transform(EQUAL_AREA_TRANSFORMER.transform, geometry).area / 10_000
            component_mask = geometry_mask([geometry_mapping], out_shape=mask.shape, transform=transform_, invert=True) & mask
            values = dnbr[component_mask]
            features.append({"type":"Feature", "id":f"dnbr-{after:%Y%m%d}-{label}-{index}", "geometry":mapping(geometry), "properties":{"method":"dNBR", "severity":label, "before_date":before.isoformat(), "after_date":after.isoformat(), "mean_dnbr":round(float(values.mean()), 3), "area_ha":round(area,2), "pixel_count":int(component_mask.sum()), "resolution_m":PIXEL_SIZE_M}})
    return features
