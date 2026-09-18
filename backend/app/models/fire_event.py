from datetime import datetime
from typing import Any

from pydantic import BaseModel


class FireEvent(BaseModel):
    id: str
    hotspot_count: int
    first_seen: datetime
    last_seen: datetime
    centroid_longitude: float
    centroid_latitude: float
    bbox: tuple[float, float, float, float]
    max_frp: float | None
    mean_confidence: float | None
    source_counts: dict[str, int]

    def as_geojson_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "id": self.id,
            "geometry": {
                "type": "Point",
                "coordinates": [self.centroid_longitude, self.centroid_latitude],
            },
            "properties": {
                "hotspot_count": self.hotspot_count,
                "first_seen": self.first_seen.isoformat(),
                "last_seen": self.last_seen.isoformat(),
                "bbox": list(self.bbox),
                "max_frp": self.max_frp,
                "mean_confidence": self.mean_confidence,
                "source_counts": self.source_counts,
            },
        }


class FireEventFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[dict[str, Any]]
    metadata: dict[str, Any]
