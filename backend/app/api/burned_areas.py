from datetime import date
from functools import lru_cache

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.hotspots import parse_bbox
from backend.app.core.config import Settings, get_settings
from backend.app.models.burned_area import BurnedAreaFeatureCollection
from backend.app.services.copernicus_burned_area import (
    BurnedAreaRequestTooLarge,
    CopernicusBurnedAreaClient,
    CopernicusConfigurationError,
    CopernicusUpstreamError,
)


router = APIRouter(prefix="/api/v1", tags=["burned areas"])


@lru_cache(maxsize=1)
def _get_cached_client(settings: Settings) -> CopernicusBurnedAreaClient:
    return CopernicusBurnedAreaClient(
        settings, httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    )


def get_copernicus_client(settings: Settings = Depends(get_settings)) -> CopernicusBurnedAreaClient:
    return _get_cached_client(settings)


@router.get("/burned-areas", response_model=BurnedAreaFeatureCollection)
async def get_burned_areas(
    bbox: tuple[float, float, float, float] = Depends(parse_bbox),
    observation_date: date = Query(alias="date"),
    copernicus: CopernicusBurnedAreaClient = Depends(get_copernicus_client),
) -> BurnedAreaFeatureCollection:
    try:
        features = await copernicus.fetch_burned_areas(bbox, observation_date)
    except CopernicusConfigurationError as error:
        raise HTTPException(status_code=503, detail="Copernicus integration is not configured") from error
    except BurnedAreaRequestTooLarge as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except CopernicusUpstreamError as error:
        raise HTTPException(status_code=502, detail="Copernicus is temporarily unavailable") from error

    return BurnedAreaFeatureCollection(
        features=features,
        metadata={
            "bbox": list(bbox),
            "date": observation_date.isoformat(),
            "source": "Copernicus Global Burnt Area V4",
            "resolution_m": 300,
            "feature_count": len(features),
            "total_area_ha": round(sum(feature["properties"]["area_ha"] for feature in features), 2),
        },
    )
