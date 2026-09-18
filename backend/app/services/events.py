from collections import Counter
from datetime import datetime, time, timezone
from math import cos, pi

import hdbscan
import numpy as np

from backend.app.models.fire_event import FireEvent
from backend.app.models.hotspot import Hotspot


class FireEventClusterer:
    """Groups nearby FIRMS detections into spatiotemporal fire events."""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_epsilon_m: float = 1_500,
        time_scale_m_per_hour: float = 300,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon_m = cluster_selection_epsilon_m
        self.time_scale_m_per_hour = time_scale_m_per_hour

    def cluster(self, hotspots: list[Hotspot]) -> tuple[list[FireEvent], int]:
        if len(hotspots) < self.min_cluster_size:
            return [], len(hotspots)

        features = self._features(hotspots)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon_m,
            metric="euclidean",
        ).fit_predict(features)

        events: list[FireEvent] = []
        for label in sorted(set(labels)):
            if label == -1:
                continue
            members = [hotspot for hotspot, member_label in zip(hotspots, labels, strict=True) if member_label == label]
            events.append(self._build_event(members, int(label)))
        events.sort(key=lambda event: event.first_seen)
        return events, int(np.count_nonzero(labels == -1))

    def _features(self, hotspots: list[Hotspot]) -> np.ndarray:
        timestamps = [self._timestamp(hotspot) for hotspot in hotspots]
        reference_latitude = sum(hotspot.latitude for hotspot in hotspots) / len(hotspots)
        radius_m = 6_371_008.8
        latitude_factor = pi / 180
        start_time = min(timestamps)
        rows: list[list[float]] = []
        for hotspot, timestamp in zip(hotspots, timestamps, strict=True):
            x = radius_m * hotspot.longitude * latitude_factor * cos(reference_latitude * latitude_factor)
            y = radius_m * hotspot.latitude * latitude_factor
            elapsed_hours = (timestamp - start_time).total_seconds() / 3600
            rows.append([x, y, elapsed_hours * self.time_scale_m_per_hour])
        return np.asarray(rows)

    @staticmethod
    def _timestamp(hotspot: Hotspot) -> datetime:
        hour = int(hotspot.acquired_time[:2])
        minute = int(hotspot.acquired_time[2:])
        return datetime.combine(hotspot.acquired_date, time(hour, minute), tzinfo=timezone.utc)

    def _build_event(self, hotspots: list[Hotspot], label: int) -> FireEvent:
        timestamps = [self._timestamp(hotspot) for hotspot in hotspots]
        latitudes = [hotspot.latitude for hotspot in hotspots]
        longitudes = [hotspot.longitude for hotspot in hotspots]
        frps = [hotspot.frp for hotspot in hotspots if hotspot.frp is not None]
        confidence_values = [self._confidence_score(hotspot.confidence) for hotspot in hotspots]
        first_seen = min(timestamps)
        centroid_latitude = sum(latitudes) / len(latitudes)
        centroid_longitude = sum(longitudes) / len(longitudes)
        event_id = f"fire-{first_seen:%Y%m%d%H%M}-{centroid_latitude:.4f}-{centroid_longitude:.4f}-{label}"
        return FireEvent(
            id=event_id,
            hotspot_count=len(hotspots),
            first_seen=first_seen,
            last_seen=max(timestamps),
            centroid_longitude=centroid_longitude,
            centroid_latitude=centroid_latitude,
            bbox=(min(longitudes), min(latitudes), max(longitudes), max(latitudes)),
            max_frp=max(frps) if frps else None,
            mean_confidence=sum(confidence_values) / len(confidence_values),
            source_counts=dict(Counter(hotspot.source for hotspot in hotspots)),
        )

    @staticmethod
    def _confidence_score(value: str | None) -> float:
        mapping = {"low": 0.33, "nominal": 0.66, "high": 1.0, "l": 0.33, "n": 0.66, "h": 1.0}
        return mapping.get((value or "").lower(), 0.0)
