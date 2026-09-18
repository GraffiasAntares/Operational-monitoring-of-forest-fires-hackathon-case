from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query
from backend.app.api.hotspots import parse_bbox
from backend.app.api.burned_areas import get_copernicus_client
from backend.app.models.burned_area import BurnedAreaFeatureCollection
from backend.app.services.dnbr import DnbrClient
from backend.app.services.copernicus_burned_area import CopernicusConfigurationError, CopernicusUpstreamError

router = APIRouter(prefix="/api/v1", tags=["burn severity"])

@router.get("/severity/dnbr", response_model=BurnedAreaFeatureCollection)
async def get_dnbr(bbox: tuple[float,float,float,float]=Depends(parse_bbox), before_date: date=Query(), after_date: date=Query(), window_days: int=Query(default=7,ge=0,le=30), client=Depends(get_copernicus_client)):
    try:
        features = await DnbrClient(client.settings).calculate(bbox, before_date, after_date, window_days)
    except CopernicusConfigurationError as error:
        raise HTTPException(503, "Copernicus integration is not configured") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except CopernicusUpstreamError as error:
        raise HTTPException(502, "Copernicus is temporarily unavailable") from error
    return BurnedAreaFeatureCollection(features=features, metadata={"method":"dNBR", "bbox":list(bbox), "before_date":before_date.isoformat(), "after_date":after_date.isoformat(), "resolution_m":20, "feature_count":len(features), "total_area_ha":round(sum(x['properties']['area_ha'] for x in features),2)})
