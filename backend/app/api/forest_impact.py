from datetime import date
import rasterio
from fastapi import APIRouter, Depends, HTTPException, Query
from backend.app.api.hotspots import parse_bbox
from backend.app.api.burned_areas import get_copernicus_client
from backend.app.models.burned_area import BurnedAreaFeatureCollection
from backend.app.services.copernicus_burned_area import CopernicusConfigurationError, CopernicusUpstreamError
from backend.app.services.worldcover import forest_impact

router = APIRouter(prefix="/api/v1", tags=["forest impact"])

@router.get("/forest-impact", response_model=BurnedAreaFeatureCollection)
async def get_forest_impact(bbox: tuple[float,float,float,float]=Depends(parse_bbox), observation_date: date=Query(alias="date"), client=Depends(get_copernicus_client)):
    try:
        burned = await client.fetch_burned_areas(bbox, observation_date)
        features, forest_area = forest_impact(burned, bbox)
    except CopernicusConfigurationError as error:
        raise HTTPException(503, "Copernicus integration is not configured") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except (CopernicusUpstreamError, rasterio.errors.RasterioIOError) as error:
        raise HTTPException(502, "An upstream geospatial service is temporarily unavailable") from error
    return BurnedAreaFeatureCollection(features=features, metadata={"bbox":list(bbox), "date":observation_date.isoformat(), "burn_source":"Copernicus Global Burnt Area V4", "landcover_source":"ESA WorldCover 2021 v200", "resolution_m":10, "feature_count":len(features), "forest_burned_ha":forest_area})
