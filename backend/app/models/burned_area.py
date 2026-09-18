from datetime import date
from typing import Any

from pydantic import BaseModel


class BurnedAreaFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[dict[str, Any]]
    metadata: dict[str, Any]


class BurnedAreaRequest(BaseModel):
    bbox: tuple[float, float, float, float]
    date: date
