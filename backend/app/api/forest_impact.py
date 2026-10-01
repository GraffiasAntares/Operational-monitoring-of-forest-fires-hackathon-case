from datetime import date
import asyncio
import rasterio
from fastapi import APIRouter, Depends, HTTPException, Query
from backend.app.api.hotspots import parse_bbox
from backend.app.api.burned_areas import get_copernicus_client
from backend.app.api.dnbr import get_dnbr_client
from backend.app.models.burned_area import BurnedAreaFeatureCollection
from backend.app.services.dnbr import DnbrClient
from backend.app.services.cache import AsyncTtlCache
from backend.app.services.copernicus_burned_area import CopernicusConfigurationError, CopernicusUpstreamError
from backend.app.services.worldcover import forest_impact

router = APIRouter(prefix="/api/v1", tags=["forest impact"])
forest_cache = AsyncTtlCache()

@router.get("/forest-impact", response_model=BurnedAreaFeatureCollection)
async def get_forest_impact(
    bbox: tuple[float, float, float, float] = Depends(parse_bbox),
    observation_date: date | None = Query(default=None, alias="date"),
    before_date: date | None = Query(default=None),
    after_date: date | None = Query(default=None),
    client=Depends(get_copernicus_client),
    dnbr_client: DnbrClient = Depends(get_dnbr_client),
):
    if (before_date is None) != (after_date is None):
        raise HTTPException(422, "before_date and after_date must be specified together")
    try:
        if before_date and after_date:
            burned = await dnbr_client.calculate(bbox, before_date, after_date)
            metadata = {
                "bbox": list(bbox),
                "before_date": before_date.isoformat(),
                "after_date": after_date.isoformat(),
                "burn_source": "dNBR Sentinel-2 severity mask",
            }
        elif observation_date:
            burned = await client.fetch_burned_areas(bbox, observation_date)
            metadata = {
                "bbox": list(bbox),
                "date": observation_date.isoformat(),
                "burn_source": "Copernicus Global Burnt Area V4",
            }
        else:
            raise HTTPException(422, "provide date or before_date and after_date")
    except CopernicusConfigurationError as error:
        raise HTTPException(503, "Copernicus integration is not configured") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except CopernicusUpstreamError as error:
        raise HTTPException(502, "An upstream geospatial service is temporarily unavailable") from error
    try:
        features, forest_area = await forest_cache.get_or_set(
            (bbox, observation_date, before_date, after_date),
            3600,
            lambda: asyncio.to_thread(forest_impact, burned, bbox),
        )
    except CopernicusConfigurationError as error:
        raise HTTPException(503, "Copernicus integration is not configured") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except (CopernicusUpstreamError, rasterio.errors.RasterioIOError) as error:
        raise HTTPException(502, "An upstream geospatial service is temporarily unavailable") from error
    return BurnedAreaFeatureCollection(features=features, metadata={**metadata, "landcover_source":"ESA WorldCover 2021 v200", "resolution_m":10, "feature_count":len(features), "forest_burned_ha":forest_area})
