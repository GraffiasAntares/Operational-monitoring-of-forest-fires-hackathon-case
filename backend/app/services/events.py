from collections import Counter, defaultdict
from datetime import datetime, time, timezone
from typing import Literal

import hdbscan
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, mapping
from shapely.ops import transform, unary_union

from backend.app.models.fire_event import FireEvent
from backend.app.models.hotspot import Hotspot


EVENT_OBSERVATION_RADIUS_M = 375
TO_EQUAL_AREA = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
TO_WGS84 = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True).transform
Sensitivity = Literal["strict", "normal", "aggressive"]
PROFILES: dict[Sensitivity, tuple[float, float]] = {
    "strict": (900, 350),
    "normal": (1_500, 700),
    "aggressive": (2_500, 1_200),
}


class FireEventClusterer:
    """Cluster dense detections locally and link plausible fire growth over time."""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        sensitivity: Sensitivity = "normal",
        window_hours: int = 12,
        max_link_gap_hours: int = 24,
        cluster_selection_epsilon_m: float | None = None,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.sensitivity = sensitivity
        self.window_hours = window_hours
        self.max_link_gap_hours = max_link_gap_hours
        profile_radius, self.spread_rate_m_per_hour = PROFILES[sensitivity]
        self.cluster_selection_epsilon_m = cluster_selection_epsilon_m or profile_radius
        self.time_scale_m_per_hour = self.spread_rate_m_per_hour

    def cluster(self, hotspots: list[Hotspot]) -> tuple[list[FireEvent], int]:
        if not hotspots:
            return [], 0
        groups = self._local_groups(hotspots)
        linked_groups = self._link_groups(groups)
        events = [self._build_event(members, index) for index, members in enumerate(linked_groups, 1)]
        events.sort(key=lambda event: (event.first_seen, event.centroid_longitude, event.centroid_latitude))
        candidate_count = sum(event.hotspot_count for event in events if event.classification == "candidate")
        return events, candidate_count

    def _local_groups(self, hotspots: list[Hotspot]) -> list[tuple[int, list[Hotspot]]]:
        start = min(self._timestamp(hotspot) for hotspot in hotspots)
        windows: dict[int, list[Hotspot]] = defaultdict(list)
        for hotspot in hotspots:
            elapsed_hours = (self._timestamp(hotspot) - start).total_seconds() / 3600
            windows[int(elapsed_hours // self.window_hours)].append(hotspot)

        groups: list[tuple[int, list[Hotspot]]] = []
        for window, members in windows.items():
            if len(members) < self.min_cluster_size:
                groups.extend((window, [member]) for member in members)
                continue
            points = np.asarray([TO_EQUAL_AREA(member.longitude, member.latitude) for member in members])
            labels = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon_m,
                metric="euclidean",
            ).fit_predict(points)
            for label in sorted(set(labels)):
                labelled = [member for member, member_label in zip(members, labels, strict=True) if member_label == label]
                if label == -1:
                    groups.extend((window, [member]) for member in labelled)
                else:
                    groups.append((window, labelled))
        return groups

    def _link_groups(self, groups: list[tuple[int, list[Hotspot]]]) -> list[list[Hotspot]]:
        parents = list(range(len(groups)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def join(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left, (left_window, left_members) in enumerate(groups):
            for right in range(left + 1, len(groups)):
                right_window, right_members = groups[right]
                if left_window == right_window:
                    continue
                if abs(right_window - left_window) * self.window_hours > self.max_link_gap_hours:
                    continue
                if self._can_link(left_members, right_members):
                    join(left, right)

        linked: dict[int, list[Hotspot]] = defaultdict(list)
        for index, (_, members) in enumerate(groups):
            linked[find(index)].extend(members)
        return list(linked.values())

    def _can_link(self, left: list[Hotspot], right: list[Hotspot]) -> bool:
        for first in left:
            first_time = self._timestamp(first)
            first_x, first_y = TO_EQUAL_AREA(first.longitude, first.latitude)
            for second in right:
                elapsed_hours = abs((self._timestamp(second) - first_time).total_seconds()) / 3600
                if elapsed_hours > self.max_link_gap_hours:
                    continue
                second_x, second_y = TO_EQUAL_AREA(second.longitude, second.latitude)
                distance = float(np.hypot(second_x - first_x, second_y - first_y))
                if distance <= self.cluster_selection_epsilon_m + elapsed_hours * self.spread_rate_m_per_hour:
                    return True
        return False

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
        zone_in_meters = unary_union([
            transform(TO_EQUAL_AREA, Point(hotspot.longitude, hotspot.latitude)).buffer(EVENT_OBSERVATION_RADIUS_M)
            for hotspot in hotspots
        ])
        observation_zone = transform(TO_WGS84, zone_in_meters)
        quality_score = sum(confidence_values) / len(confidence_values)
        classification = "confirmed" if len(hotspots) >= self.min_cluster_size and quality_score >= 0.5 else "candidate"
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
            mean_confidence=quality_score,
            source_counts=dict(Counter(hotspot.source for hotspot in hotspots)),
            hotspot_ids=[hotspot.id for hotspot in hotspots],
            observation_zone=mapping(observation_zone),
            observation_zone_area_ha=round(zone_in_meters.area / 10_000, 2),
            classification=classification,
            quality_score=round(quality_score, 2),
        )

    @staticmethod
    def _confidence_score(value: str | None) -> float:
        mapping = {"low": 0.33, "nominal": 0.66, "high": 1.0, "l": 0.33, "n": 0.66, "h": 1.0}
        return mapping.get((value or "").lower(), 0.0)
