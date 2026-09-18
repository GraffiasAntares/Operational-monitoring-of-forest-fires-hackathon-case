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


def test_hdbscan_creates_two_events_and_marks_singleton_as_noise():
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

    assert len(events) == 2
    assert [event.hotspot_count for event in events] == [3, 3]
    assert noise_count == 1
    assert events[0].max_frp == 5.0
