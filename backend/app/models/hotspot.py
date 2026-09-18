from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class Hotspot(BaseModel):
    id: str
    source: str
    latitude: float
    longitude: float
    acquired_date: date
    acquired_time: str = Field(pattern=r"^\d{4}$")
    confidence: str | None = None
    frp: float | None = None
    brightness: float | None = None
    daynight: str | None = None

    def as_geojson_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "id": self.id,
            "geometry": {"type": "Point", "coordinates": [self.longitude, self.latitude]},
            "properties": {
                "source": self.source,
                "acquired_date": self.acquired_date.isoformat(),
                "acquired_time": self.acquired_time,
                "confidence": self.confidence,
                "frp": self.frp,
                "brightness": self.brightness,
                "daynight": self.daynight,
            },
        }


class HotspotFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[dict[str, Any]]
    metadata: dict[str, Any]
