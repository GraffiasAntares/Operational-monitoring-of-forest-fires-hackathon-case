"""Serve the trained BS U-Net on a pair of cloud-filtered Sentinel-2 scenes.

The competition checkpoint expects additional SAR/terrain channels.  For the
interactive preview these are explicitly neutral-filled until the matching
operational providers are connected; optical channels and SCL are real scenes.
This keeps the UI on the trained BS model rather than silently reverting to a
dNBR threshold layer.
"""
from __future__ import annotations

import asyncio
from datetime import date
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
import math

import numpy as np
import tifffile

from backend.app.services.copernicus_burned_area import CopernicusUpstreamError
from backend.app.services.dnbr import DnbrClient, PIXEL_SIZE_M, MAX_OUTPUT_PIXELS, vectorize_dnbr
from backend.app.services.scene_preview import select_period_scene_dates

EVALSCRIPT = """//VERSION=3
function setup(){return {input:[{bands:["B02","B03","B04","B05","B06","B07","B8A","B11","B12","SCL","dataMask"]}],output:{bands:11,sampleType:"FLOAT32"}};}
function evaluatePixel(s){return [s.B02,s.B03,s.B04,s.B05,s.B06,s.B07,s.B8A,s.B11,s.B12,s.SCL,s.dataMask];}"""


def _adaptive_dimensions(bbox: tuple[float, float, float, float]) -> tuple[int, int, float]:
    """Keep the synchronous U-Net request bounded by lowering resolution."""
    west, south, east, north = bbox
    width_m = (east - west) * 111_320 * max(math.cos(math.radians((south + north) / 2)), .01)
    height_m = (north - south) * 110_574
    base_width, base_height = math.ceil(width_m / PIXEL_SIZE_M), math.ceil(height_m / PIXEL_SIZE_M)
    scale = max(1.0, math.sqrt((base_width * base_height) / MAX_OUTPUT_PIXELS))
    width, height = max(1, math.ceil(base_width / scale)), max(1, math.ceil(base_height / scale))
    # Ceiling can put us a handful of pixels above the exact cap.
    if width * height > MAX_OUTPUT_PIXELS:
        scale *= math.sqrt((width * height) / MAX_OUTPUT_PIXELS)
        width, height = max(1, math.floor(base_width / scale)), max(1, math.floor(base_height / scale))
    return width, height, PIXEL_SIZE_M * scale


def _payload(bbox: tuple[float, float, float, float], center: date, window_days: int) -> dict[str, Any]:
    width, height, _ = _adaptive_dimensions(bbox)
    from datetime import timedelta
    return {
        "input": {"bounds": {"bbox": list(bbox), "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}, "data": [{
            "type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": f"{center - timedelta(days=window_days)}T00:00:00Z", "to": f"{center + timedelta(days=window_days + 1)}T00:00:00Z"}, "maxCloudCoverage": 30, "mosaickingOrder": "leastCC"},
        }]},
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]}, "evalscript": EVALSCRIPT,
    }


def _scene(content: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(tifffile.imread(BytesIO(content)))
    bands = np.moveaxis(raw, -1, 0) if raw.ndim == 3 and raw.shape[-1] == 11 else raw
    if bands.ndim != 3 or bands.shape[0] != 11:
        raise CopernicusUpstreamError("Sentinel-2 response has an unexpected BS band layout")
    return bands[:9], bands[9], bands[10]


@lru_cache(maxsize=1)
def _model_assets(checkpoint_path: str, preprocessing_path: str):
    try:
        import torch
        from ml.bs import BurnSeverityPreprocessor, FeatureStatistics
        from ml.bs.model import CompactUNet
        from ml.bs.training import feature_indices
    except ImportError as error:
        raise RuntimeError("BS inference requires the optional ML dependencies") from error
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = CompactUNet(config["input_channels"], config["base_channels"], config["dropout"])
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return torch, model, BurnSeverityPreprocessor(FeatureStatistics.load(preprocessing_path)), np.asarray(feature_indices(config["feature_mode"]))


class BsModelClient(DnbrClient):
    def __init__(self, settings, checkpoint_path: str, preprocessing_path: str, http_client=None):
        super().__init__(settings, http_client); self.checkpoint_path = checkpoint_path; self.preprocessing_path = preprocessing_path

    async def calculate(self, bbox: tuple[float, float, float, float], before: date, after: date, window_days: int = 7) -> list[dict[str, Any]]:
        self._check_configuration()
        if before >= after: raise ValueError("before_date must be earlier than after_date")
        if not Path(self.checkpoint_path).is_file() or not Path(self.preprocessing_path).is_file(): raise RuntimeError("BS model assets are not configured")
        return await self._cache.get_or_set(("bs-model", bbox, before, after, window_days), 3600, lambda: self._calculate(bbox, before, after, window_days))

    async def _calculate(self, bbox, before, after, window_days):
        import httpx
        client = self.http_client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds); owns = self.http_client is None
        try:
            selected_before, selected_after = await select_period_scene_dates(client, self.settings, bbox, before, after, window_days)
            headers = {"Authorization": f"Bearer {await self._access_token_for(client)}"}
            first, second = await asyncio.gather(client.post(self.settings.copernicus_process_url, json=_payload(bbox, selected_before, 0), headers=headers), client.post(self.settings.copernicus_process_url, json=_payload(bbox, selected_after, 0), headers=headers))
            if first.is_error or second.is_error: raise CopernicusUpstreamError("Copernicus Sentinel-2 Process API request failed")
            before_s2, before_scl, before_valid = _scene(first.content); after_s2, after_scl, after_valid = _scene(second.content)
        except httpx.HTTPError as error: raise CopernicusUpstreamError("Copernicus is temporarily unavailable") from error
        finally:
            if owns: await client.aclose()
        # The U-Net schema is fixed.  SAR and terrain become neutral channels
        # instead of triggering a dNBR fallback while operational adapters land.
        rows, columns = before_scl.shape
        layers = {"s2_pre": np.concatenate((before_s2 * 10_000, before_scl[None])).astype(np.float32), "s2_post": np.concatenate((after_s2 * 10_000, after_scl[None])).astype(np.float32), "s1_pre": np.zeros((2, rows, columns), dtype=np.float32), "s1_post": np.zeros((2, rows, columns), dtype=np.float32), "aux": np.zeros((3, rows, columns), dtype=np.float32)}
        torch, model, preprocessor, indices = _model_assets(self.checkpoint_path, self.preprocessing_path)
        prediction = await asyncio.to_thread(lambda: model(torch.from_numpy(preprocessor.transform(layers)[indices].copy()).unsqueeze(0)).argmax(dim=1).squeeze().detach().numpy().astype(np.uint8))
        valid = (before_valid > 0) & (after_valid > 0) & np.isin(before_scl, [4, 5, 7]) & np.isin(after_scl, [4, 5, 7])
        # Vectorizer is class-threshold based, so use encoded class values only.
        severity_values = np.choose(prediction, [0.0, 0.12, 0.35, 0.8]).astype(np.float32)
        features = vectorize_dnbr(severity_values, valid & (prediction > 0), bbox, selected_before, selected_after)
        _, _, resolution_m = _adaptive_dimensions(bbox)
        for feature in features:
            feature["id"] = feature["id"].replace("dnbr", "bs")
            feature["properties"]["method"] = "BS U-Net"
            feature["properties"]["resolution_m"] = round(resolution_m, 1)
            feature["properties"]["model_input_note"] = "Sentinel-2 + neutral SAR/terrain preview channels"
        return features
