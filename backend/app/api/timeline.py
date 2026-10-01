"""Period-oriented payload used by the monitoring timeline."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.hotspots import get_firms_client, parse_bbox
from backend.app.services.firms import FirmsClient, FirmsConfigurationError, FirmsUpstreamError


router = APIRouter(prefix="/api/v1", tags=["observation timeline"])
INTERVAL_COUNT = 10


def split_period(start: date, end: date) -> list[tuple[date, date]]:
    """Return ten contiguous, inclusive date ranges without gaps."""
    if start >= end:
        raise ValueError("end_date must be later than start_date")
    total_days = (end - start).days + 1
    if total_days < INTERVAL_COUNT:
        raise ValueError("the observation period must be at least 10 days")
    ranges: list[tuple[date, date]] = []
    for index in range(INTERVAL_COUNT):
        first_offset = index * total_days // INTERVAL_COUNT
        next_offset = (index + 1) * total_days // INTERVAL_COUNT
        interval_start = start + timedelta(days=first_offset)
        interval_end = start + timedelta(days=next_offset - 1)
        ranges.append((interval_start, interval_end))
    return ranges


@router.get("/observation-timeline")
async def get_observation_timeline(
    bbox: tuple[float, float, float, float] = Depends(parse_bbox),
    start_date: date = Query(),
    end_date: date = Query(),
    firms: FirmsClient = Depends(get_firms_client),
) -> dict:
    try:
        intervals = split_period(start_date, end_date)
        # FIRMS accepts a start date and a day range for historical sources.
        batches = await asyncio.gather(
            *(firms.fetch_hotspots(bbox, (interval_end - interval_start).days + 1, interval_start)
              for interval_start, interval_end in intervals)
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FirmsConfigurationError as error:
        raise HTTPException(status_code=503, detail="FIRMS integration is not configured") from error
    except FirmsUpstreamError as error:
        raise HTTPException(status_code=502, detail="FIRMS is temporarily unavailable") from error

    return {
        "bbox": list(bbox), "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "intervals": [
            {
                "index": index,
                "start_date": interval_start.isoformat(),
                "end_date": interval_end.isoformat(),
                "hotspots": {
                    "type": "FeatureCollection",
                    "features": [hotspot.as_geojson_feature() for hotspot in hotspots],
                    "metadata": {"count": len(hotspots), "start_date": interval_start.isoformat(), "end_date": interval_end.isoformat()},
                },
            }
            for index, ((interval_start, interval_end), hotspots) in enumerate(zip(intervals, batches, strict=True))
        ],
    }
