from datetime import date

from backend.app.models.hotspot import Hotspot
from backend.app.services.events import FireEventClusterer


def hotspot(identifier: str, latitude: float, longitude: float, acquired_time: str) -> Hotspot:
    return Hotspot(
        id=identifier,
        source="VIIRS_SNPP_NRT",
        latitude=latitude,
        longitude=longitude,
        acquired_date=date(2026, 9, 17),
        acquired_time=acquired_time,
        confidence="high",
        frp=5.0,
    )


def test_hdbscan_creates_two_events_and_keeps_singleton_as_candidate():
    detections = [
        hotspot("a1", 56.1200, 92.8400, "0600"),
        hotspot("a2", 56.1210, 92.8410, "0630"),
        hotspot("a3", 56.1190, 92.8390, "0700"),
        hotspot("b1", 56.1900, 92.9200, "0600"),
        hotspot("b2", 56.1910, 92.9210, "0630"),
        hotspot("b3", 56.1890, 92.9190, "0700"),
        hotspot("noise", 57.0000, 94.0000, "0600"),
    ]

    events, noise_count = FireEventClusterer(cluster_selection_epsilon_m=1_500).cluster(detections)

    assert len(events) == 3
    assert [event.hotspot_count for event in events] == [3, 3, 1]
    assert noise_count == 1
    assert events[0].max_frp == 5.0
    assert events[0].observation_zone["type"] in {"Polygon", "MultiPolygon"}
    assert events[0].observation_zone_area_ha > 0
    assert len(events[0].hotspot_ids) == 3
    assert events[-1].classification == "candidate"


def test_links_fire_growth_between_time_windows():
    detections = [
        hotspot("a1", 56.1200, 92.8400, "0600"),
        hotspot("a2", 56.1210, 92.8410, "0630"),
        hotspot("a3", 56.1190, 92.8390, "0700"),
        hotspot("b1", 56.1300, 92.8500, "1800"),
        hotspot("b2", 56.1310, 92.8510, "1830"),
        hotspot("b3", 56.1290, 92.8490, "1900"),
    ]

    events, candidate_count = FireEventClusterer().cluster(detections)

    assert len(events) == 1
    assert events[0].hotspot_count == 6
    assert events[0].classification == "confirmed"
    assert candidate_count == 0
