from datetime import date
from functools import lru_cache
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from backend.app.api.hotspots import parse_bbox
from backend.app.api.burned_areas import get_copernicus_client
from backend.app.models.burned_area import BurnedAreaFeatureCollection
from backend.app.services.bs_model import BsModelClient
from backend.app.services.copernicus_burned_area import CopernicusConfigurationError, CopernicusUpstreamError

router = APIRouter(prefix="/api/v1", tags=["burn severity"])

@lru_cache(maxsize=1)
def _get_cached_dnbr_client(settings):
    return BsModelClient(settings, settings.bs_model_path or "", settings.bs_preprocessing_path or "", httpx.AsyncClient(timeout=settings.request_timeout_seconds))

def get_dnbr_client(client=Depends(get_copernicus_client)) -> BsModelClient:
    return _get_cached_dnbr_client(client.settings)

@router.get("/severity/dnbr", response_model=BurnedAreaFeatureCollection)
@router.get("/severity/bs", response_model=BurnedAreaFeatureCollection)
async def get_dnbr(bbox: tuple[float,float,float,float]=Depends(parse_bbox), before_date: date=Query(), after_date: date=Query(), window_days: int=Query(default=7,ge=0,le=30), client: BsModelClient=Depends(get_dnbr_client)):
    try:
        features = await client.calculate(bbox, before_date, after_date, window_days)
    except CopernicusConfigurationError as error:
        raise HTTPException(503, "Copernicus integration is not configured") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except CopernicusUpstreamError as error:
        raise HTTPException(502, "Copernicus is temporarily unavailable") from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    resolution_m = max((float(item["properties"].get("resolution_m", 20)) for item in features), default=20)
    return BurnedAreaFeatureCollection(features=features, metadata={"method":"BS U-Net", "bbox":list(bbox), "before_date":before_date.isoformat(), "after_date":after_date.isoformat(), "resolution_m":resolution_m, "feature_count":len(features), "total_area_ha":round(sum(x['properties']['area_ha'] for x in features),2)})
