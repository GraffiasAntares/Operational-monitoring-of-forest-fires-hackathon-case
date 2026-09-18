from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.hotspots import get_firms_client, parse_bbox
from backend.app.models.fire_event import FireEventFeatureCollection
from backend.app.services.events import FireEventClusterer
from backend.app.services.firms import FirmsClient, FirmsConfigurationError, FirmsUpstreamError


router = APIRouter(prefix="/api/v1", tags=["fire events"])


@router.get("/fire-events", response_model=FireEventFeatureCollection)
async def get_fire_events(
    bbox: tuple[float, float, float, float] = Depends(parse_bbox),
    days: int = Query(default=1, ge=1, le=5),
    start_date: date | None = Query(default=None),
    min_cluster_size: int = Query(default=3, ge=2, le=50),
    firms: FirmsClient = Depends(get_firms_client),
) -> FireEventFeatureCollection:
    try:
        hotspots = await firms.fetch_hotspots(bbox, days, start_date)
    except FirmsConfigurationError as error:
        raise HTTPException(status_code=503, detail="FIRMS integration is not configured") from error
    except FirmsUpstreamError as error:
        raise HTTPException(status_code=502, detail="FIRMS is temporarily unavailable") from error

    clusterer = FireEventClusterer(min_cluster_size=min_cluster_size)
    events, noise_count = clusterer.cluster(hotspots)
    return FireEventFeatureCollection(
        features=[event.as_geojson_feature() for event in events],
        metadata={
            "bbox": list(bbox),
            "hotspot_count": len(hotspots),
            "event_count": len(events),
            "noise_hotspot_count": noise_count,
            "min_cluster_size": min_cluster_size,
            "time_scale_m_per_hour": clusterer.time_scale_m_per_hour,
        },
    )
