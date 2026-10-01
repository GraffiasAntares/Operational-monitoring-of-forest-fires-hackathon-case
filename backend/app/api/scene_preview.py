from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from backend.app.api.burned_areas import get_copernicus_client
from backend.app.api.hotspots import parse_bbox
from backend.app.services.copernicus_burned_area import CopernicusConfigurationError, CopernicusUpstreamError
from backend.app.services.scene_preview import SentinelPreviewClient


router = APIRouter(prefix="/api/v1", tags=["scene previews"])


@router.get("/scene-preview", response_class=Response)
async def get_scene_preview(
    bbox: tuple[float, float, float, float] = Depends(parse_bbox),
    date: date = Query(),
    window_days: int = Query(default=7, ge=0, le=30),
    end_date: date | None = Query(default=None),
    position: str = Query(default="start", pattern="^(start|end)$"),
    client=Depends(get_copernicus_client),
) -> Response:
    try:
        image, selected_date = await SentinelPreviewClient(client.settings).fetch_preview(bbox, date, window_days, end=end_date, position=position)
    except CopernicusConfigurationError as error:
        raise HTTPException(503, "Copernicus integration is not configured") from error
    except CopernicusUpstreamError as error:
        raise HTTPException(502, "Copernicus Sentinel-2 preview is temporarily unavailable") from error
    # The Process API chooses the least-cloudy daytime observation in the requested
    # search window. Keep the requested date in the response so a client never
    # presents an old image as a scene captured exactly on that date.
    return Response(content=image, media_type="image/png", headers={
        "Cache-Control": "private, max-age=300", "X-Requested-Scene-Date": date.isoformat(), "X-Actual-Scene-Date": selected_date.isoformat(),
        "X-Scene-Search-Window-Days": str(window_days),
    })
