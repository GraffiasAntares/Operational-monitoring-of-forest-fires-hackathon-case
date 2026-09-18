"""Copernicus Global Burnt Area V4 adapter and raster-to-GeoJSON conversion."""

from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import math
import time
from typing import Any

import httpx
import numpy as np
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform, unary_union
import tifffile

from backend.app.core.config import Settings


COPERNICUS_BURNT_AREA_V4_COLLECTION_ID = "162ee729-86a7-45bc-9cfe-c01f718e3216"
COPERNICUS_BURNT_AREA_V4_START = date(2025, 1, 1)
PIXEL_SIZE_M = 300
MAX_OUTPUT_PIXELS = 250_000

EQUAL_AREA_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: [\"DOB\", \"BF\", \"CP\"] }],
    output: { bands: 3, sampleType: \"INT16\" }
  };
}
function evaluatePixel(sample) {
  return [sample.DOB, sample.BF, sample.CP];
}
"""


class CopernicusConfigurationError(RuntimeError):
    pass


class CopernicusUpstreamError(RuntimeError):
    pass


class BurnedAreaRequestTooLarge(ValueError):
    pass


def output_dimensions(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    """Choose a roughly 300 m grid and protect the synchronous API from huge AOIs."""
    west, south, east, north = bbox
    mean_latitude = math.radians((south + north) / 2)
    width_m = (east - west) * 111_320 * max(math.cos(mean_latitude), 0.01)
    height_m = (north - south) * 110_574
    width = max(1, math.ceil(width_m / PIXEL_SIZE_M))
    height = max(1, math.ceil(height_m / PIXEL_SIZE_M))
    if width * height > MAX_OUTPUT_PIXELS:
        raise BurnedAreaRequestTooLarge(
            "bbox is too large for the synchronous 300 m burned-area request"
        )
    return width, height


def build_process_payload(
    bbox: tuple[float, float, float, float], observation_date: date
) -> dict[str, Any]:
    width, height = output_dimensions(bbox)
    start = observation_date.isoformat()
    end = (observation_date + timedelta(days=1)).isoformat()
    return {
        "input": {
            "bounds": {"bbox": list(bbox), "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [
                {
                    "type": f"byoc-{COPERNICUS_BURNT_AREA_V4_COLLECTION_ID}",
                    "dataFilter": {"timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T00:00:00Z"}},
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }


def _as_bands(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise TIFF band placement emitted by TIFF encoders to (DOB, BF, CP)."""
    if raw.ndim != 3:
        raise CopernicusUpstreamError("Copernicus returned a TIFF without the expected three bands")
    if raw.shape[-1] == 3:
        dob, burned_fraction, confidence = np.moveaxis(raw, -1, 0)
    elif raw.shape[0] == 3:
        dob, burned_fraction, confidence = raw
    else:
        raise CopernicusUpstreamError("Copernicus returned a TIFF without the expected three bands")
    return dob, burned_fraction, confidence


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return four-connected pixel components without pulling in a heavy GIS stack."""
    rows, columns = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if visited[row, column]:
            continue
        component: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([(row, column)])
        visited[row, column] = True
        while queue:
            current_row, current_column = queue.popleft()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < rows
                    and 0 <= next_column < columns
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
        result.append(component)
    return result


def vectorize_burned_area_tiff(
    tiff_bytes: bytes, bbox: tuple[float, float, float, float], observation_date: date
) -> list[dict[str, Any]]:
    """Turn DOB/BF/CP raster values into one GeoJSON feature per connected burn patch."""
    try:
        raw = tifffile.imread(BytesIO(tiff_bytes))
        dob, burned_fraction, confidence = _as_bands(np.asarray(raw))
    except (tifffile.TiffFileError, ValueError) as error:
        raise CopernicusUpstreamError("Copernicus returned an unreadable TIFF") from error

    rows, columns = dob.shape
    west, south, east, north = bbox
    pixel_width = (east - west) / columns
    pixel_height = (north - south) / rows
    features: list[dict[str, Any]] = []

    for day_of_burn in sorted(int(value) for value in np.unique(dob) if value > 0):
        for index, component in enumerate(_components(dob == day_of_burn), start=1):
            cells = [
                box(
                    west + column * pixel_width,
                    north - (row + 1) * pixel_height,
                    west + (column + 1) * pixel_width,
                    north - row * pixel_height,
                )
                for row, column in component
            ]
            geometry = unary_union(cells)
            area_ha = transform(EQUAL_AREA_TRANSFORMER.transform, geometry).area / 10_000
            values = np.array([burned_fraction[row, column] for row, column in component])
            probabilities = np.array([confidence[row, column] for row, column in component])
            feature_id = f"burn-{observation_date:%Y%m%d}-{day_of_burn}-{index}"
            features.append(
                {
                    "type": "Feature",
                    "id": feature_id,
                    "geometry": mapping(geometry),
                    "properties": {
                        "source": "copernicus_global_burnt_area_v4",
                        "observed_date": observation_date.isoformat(),
                        "day_of_burn": day_of_burn,
                        "burned_fraction": round(float(values.mean()) / 1000, 3),
                        "confidence": round(float(probabilities.mean()) / 1000, 3),
                        "area_ha": round(area_ha, 2),
                        "pixel_count": len(component),
                        "resolution_m": PIXEL_SIZE_M,
                        "precision_note": "approximate_300m_product",
                    },
                }
            )
    return features


class CopernicusBurnedAreaClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.http_client = http_client
        self._access_token: str | None = None
        self._token_expires_at = 0.0

    def _check_configuration(self) -> None:
        if not self.settings.copernicus_client_id or not self.settings.copernicus_client_secret:
            raise CopernicusConfigurationError("COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET are required")

    async def _access_token_for(self, client: httpx.AsyncClient) -> str:
        if self._access_token and time.monotonic() < self._token_expires_at:
            return self._access_token
        response = await client.post(
            self.settings.copernicus_token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.settings.copernicus_client_id,
                "client_secret": self.settings.copernicus_client_secret,
            },
        )
        if response.is_error:
            raise CopernicusUpstreamError("Copernicus OAuth token request failed")
        body = response.json()
        token = body.get("access_token")
        if not token:
            raise CopernicusUpstreamError("Copernicus OAuth response contains no access token")
        self._access_token = str(token)
        self._token_expires_at = time.monotonic() + max(int(body.get("expires_in", 300)) - 30, 1)
        return self._access_token

    async def fetch_burned_areas(
        self, bbox: tuple[float, float, float, float], observation_date: date
    ) -> list[dict[str, Any]]:
        self._check_configuration()
        if observation_date < COPERNICUS_BURNT_AREA_V4_START:
            raise ValueError("Burnt Area V4 is available from 2025-01-01")
        if observation_date > datetime.now(timezone.utc).date():
            raise ValueError("date cannot be in the future")
        payload = build_process_payload(bbox, observation_date)
        owns_client = self.http_client is None
        client = self.http_client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            token = await self._access_token_for(client)
            response = await client.post(
                self.settings.copernicus_process_url,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.is_error:
                raise CopernicusUpstreamError("Copernicus Process API request failed")
            return vectorize_burned_area_tiff(response.content, bbox, observation_date)
        except httpx.HTTPError as error:
            raise CopernicusUpstreamError("Copernicus is temporarily unavailable") from error
        finally:
            if owns_client:
                await client.aclose()
