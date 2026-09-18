"""Read ESA WorldCover 2021 COG windows and quantify Tree cover inside burn polygons."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import shape

TREE_COVER_CLASS = 10
MAX_WORLDCOVER_PIXELS = 1_500_000
BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

def _tile_name(latitude: int, longitude: int) -> str:
    return f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}"

def _tile_url(latitude: int, longitude: int) -> str:
    return f"{BASE_URL}/ESA_WorldCover_10m_2021_v200_{_tile_name(latitude, longitude)}_Map.tif"

def _tile_origins(bbox: tuple[float, float, float, float]):
    west, south, east, north = bbox
    for latitude in range(math.floor(south / 3) * 3, math.ceil(north / 3) * 3, 3):
        for longitude in range(math.floor(west / 3) * 3, math.ceil(east / 3) * 3, 3):
            yield latitude, longitude

def forest_impact(features: list[dict[str, Any]], bbox: tuple[float, float, float, float]) -> tuple[list[dict[str, Any]], float]:
    geometries = [shape(feature["geometry"]) for feature in features]
    forest_ha = [0.0] * len(features)
    for latitude, longitude in _tile_origins(bbox):
        with rasterio.Env(CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            with rasterio.open(_tile_url(latitude, longitude)) as dataset:
                left, bottom, right, top = dataset.bounds
                west, south, east, north = bbox
                bounds = (max(west, left), max(south, bottom), min(east, right), min(north, top))
                if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                    continue
                window = from_bounds(*bounds, transform=dataset.transform).round_offsets().round_lengths()
                if window.width * window.height > MAX_WORLDCOVER_PIXELS:
                    raise ValueError("bbox is too large for 10 m WorldCover analysis; use a smaller event bbox")
                labels = dataset.read(1, window=window)
                affine = dataset.window_transform(window)
                tree = labels == TREE_COVER_CLASS
                if not tree.any():
                    continue
                row_area_m2 = 111_320 * 110_574 * abs(affine.a * affine.e) * np.cos(np.radians(affine.f + (np.arange(labels.shape[0]) + .5) * affine.e))
                for index, geometry in enumerate(geometries):
                    inside = geometry_mask([geometry], out_shape=labels.shape, transform=affine, invert=True)
                    forest_ha[index] += float((tree & inside).sum(axis=1).dot(row_area_m2) / 10_000)
    annotated = [{**feature, "properties": {**feature["properties"], "worldcover_year": 2021, "worldcover_class": "Tree cover", "forest_burned_ha": round(area, 2)}} for feature, area in zip(features, forest_ha, strict=True)]
    return annotated, round(sum(forest_ha), 2)
