from datetime import date
from functools import lru_cache

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.core.config import Settings, get_settings
from backend.app.models.hotspot import HotspotFeatureCollection
from backend.app.services.firms import FirmsClient, FirmsConfigurationError, FirmsUpstreamError


router = APIRouter(prefix="/api/v1", tags=["hotspots"])


def parse_bbox(bbox: str = Query(description="west,south,east,north")) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(value) for value in bbox.split(","))
    except ValueError as error:
        raise HTTPException(status_code=422, detail="bbox must be west,south,east,north") from error
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise HTTPException(status_code=422, detail="bbox coordinates are invalid")
    return west, south, east, north


@lru_cache(maxsize=1)
def _get_cached_firms_client(settings: Settings) -> FirmsClient:
    return FirmsClient(settings, httpx.AsyncClient(timeout=settings.request_timeout_seconds))


def get_firms_client(settings: Settings = Depends(get_settings)) -> FirmsClient:
    return _get_cached_firms_client(settings)


@router.get("/hotspots", response_model=HotspotFeatureCollection)
async def get_hotspots(
    bbox: tuple[float, float, float, float] = Depends(parse_bbox),
    days: int = Query(default=1, ge=1, le=5),
    start_date: date | None = Query(default=None),
    firms: FirmsClient = Depends(get_firms_client),
) -> HotspotFeatureCollection:
    try:
        hotspots = await firms.fetch_hotspots(bbox, days, start_date)
    except FirmsConfigurationError as error:
        raise HTTPException(status_code=503, detail="FIRMS integration is not configured") from error
    except FirmsUpstreamError as error:
        raise HTTPException(status_code=502, detail="FIRMS is temporarily unavailable") from error

    return HotspotFeatureCollection(
        features=[hotspot.as_geojson_feature() for hotspot in hotspots],
        metadata={
            "bbox": list(bbox),
            "sources": ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]
            if start_date
            else ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"],
            "days": days,
            "start_date": start_date.isoformat() if start_date else None,
            "count": len(hotspots),
        },
    )
