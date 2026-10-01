from fastapi.testclient import TestClient

from backend.app.api.burned_areas import get_copernicus_client
from backend.app.api.hotspots import get_firms_client
from backend.app.core.config import Settings
from backend.app.main import app
from backend.app.models.hotspot import Hotspot
from backend.app.services.copernicus_burned_area import CopernicusBurnedAreaClient


class StubFirmsClient:
    async def fetch_hotspots(self, bbox, days, start_date):
        return [
            Hotspot(
                id="test-hotspot",
                source="VIIRS_SNPP_NRT",
                latitude=56.12,
                longitude=92.84,
                acquired_date="2026-09-17",
                acquired_time="0621",
                confidence="high",
                frp=8.5,
            )
        ]


class EventStubFirmsClient:
    async def fetch_hotspots(self, bbox, days, start_date):
        return [
            Hotspot(
                id=f"event-{index}",
                source="VIIRS_SNPP_NRT",
                latitude=56.12 + latitude_offset,
                longitude=92.84 + longitude_offset,
                acquired_date="2026-09-17",
                acquired_time=acquired_time,
                confidence="high",
                frp=8.5,
            )
            for index, latitude_offset, longitude_offset, acquired_time in [
                (1, 0.0, 0.0, "0600"),
                (2, 0.001, 0.001, "0630"),
                (3, -0.001, -0.001, "0700"),
                (4, 0.07, 0.08, "0600"),
                (5, 0.071, 0.081, "0630"),
                (6, 0.069, 0.079, "0700"),
                (7, 0.88, 0.88, "0600"),
            ]
        ]


def test_hotspots_endpoint_returns_geojson():
    app.dependency_overrides[get_firms_client] = lambda: StubFirmsClient()
    try:
        response = TestClient(app).get("/api/v1/hotspots", params={"bbox": "92.5,55.8,93.2,56.4"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert payload["metadata"]["count"] == 1
    assert payload["features"][0]["geometry"]["coordinates"] == [92.84, 56.12]


def test_hotspots_endpoint_rejects_invalid_bbox():
    response = TestClient(app).get("/api/v1/hotspots", params={"bbox": "92.5,55.8"})
    assert response.status_code == 422


def test_observation_timeline_returns_ten_contiguous_firms_intervals():
    app.dependency_overrides[get_firms_client] = lambda: StubFirmsClient()
    try:
        response = TestClient(app).get("/api/v1/observation-timeline", params={
            "bbox": "92.5,55.8,93.2,56.4", "start_date": "2026-08-01", "end_date": "2026-08-30",
        })
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    intervals = response.json()["intervals"]
    assert len(intervals) == 10
    assert intervals[0]["start_date"] == "2026-08-01"
    assert intervals[-1]["end_date"] == "2026-08-30"
    assert all(len(item["hotspots"]["features"]) == 1 for item in intervals)


def test_fire_events_endpoint_returns_clustered_geojson():
    app.dependency_overrides[get_firms_client] = lambda: EventStubFirmsClient()
    try:
        response = TestClient(app).get("/api/v1/fire-events", params={"bbox": "92.5,55.8,93.2,56.4"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert payload["metadata"]["event_count"] == 3
    assert payload["features"][0]["properties"]["hotspot_count"] == 3
    assert payload["metadata"]["candidate_hotspot_count"] == 1


def test_burned_areas_endpoint_returns_geojson():
    class StubCopernicusClient:
        async def fetch_burned_areas(self, bbox, observation_date):
            assert bbox == (92.5, 55.8, 93.2, 56.4)
            assert observation_date.isoformat() == "2026-09-17"
            return [
                {
                    "type": "Feature",
                    "id": "burn-1",
                    "geometry": {"type": "Polygon", "coordinates": []},
                    "properties": {"area_ha": 12.5},
                }
            ]

    app.dependency_overrides[get_copernicus_client] = lambda: StubCopernicusClient()
    try:
        response = TestClient(app).get(
            "/api/v1/burned-areas?bbox=92.5,55.8,93.2,56.4&date=2026-09-17"
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["metadata"]["total_area_ha"] == 12.5


def test_burned_areas_endpoint_reports_missing_configuration():
    app.dependency_overrides[get_copernicus_client] = lambda: CopernicusBurnedAreaClient(
        Settings(firms_map_key=None)
    )
    try:
        response = TestClient(app).get(
            "/api/v1/burned-areas?bbox=92.5,55.8,93.2,56.4&date=2026-09-17"
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
