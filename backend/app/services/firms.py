import csv
from datetime import date
from io import StringIO

import httpx

from backend.app.core.config import Settings
from backend.app.models.hotspot import Hotspot


FIRMS_NRT_SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT")
FIRMS_STANDARD_SOURCES = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")


class FirmsConfigurationError(RuntimeError):
    pass


class FirmsUpstreamError(RuntimeError):
    pass


class FirmsClient:
    """Small adapter around the FIRMS Area CSV API."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.http_client = http_client

    async def fetch_hotspots(
        self, bbox: tuple[float, float, float, float], days: int, start_date: date | None = None
    ) -> list[Hotspot]:
        if not self.settings.firms_map_key:
            raise FirmsConfigurationError("FIRMS_MAP_KEY is not configured")

        sources = FIRMS_STANDARD_SOURCES if start_date else FIRMS_NRT_SOURCES
        own_client = self.http_client is None
        client = self.http_client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            batches = [
                await self._fetch_source(client, source, bbox, days, start_date)
                for source in sources
            ]
        finally:
            if own_client:
                await client.aclose()

        return self._deduplicate([hotspot for batch in batches for hotspot in batch])

    async def _fetch_source(
        self,
        client: httpx.AsyncClient,
        source: str,
        bbox: tuple[float, float, float, float],
        days: int,
        start_date: date | None,
    ) -> list[Hotspot]:
        bbox_string = ",".join(format(value, ".6f") for value in bbox)
        path_parts = [self.settings.firms_base_url, self.settings.firms_map_key or "", source, bbox_string, str(days)]
        if start_date:
            path_parts.append(start_date.isoformat())
        url = "/".join(path_parts)

        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise FirmsUpstreamError(f"FIRMS request failed for {source}") from error

        return self._parse_csv(response.text, source)

    @staticmethod
    def _parse_csv(payload: str, source: str) -> list[Hotspot]:
        rows = csv.DictReader(StringIO(payload))
        hotspots: list[Hotspot] = []
        for row in rows:
            if not row.get("latitude") or not row.get("longitude"):
                continue
            acquired_time = str(row.get("acq_time", "")).strip().zfill(4)
            acquired_date = date.fromisoformat(row["acq_date"])
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            hotspot_id = f"{source}:{acquired_date.isoformat()}:{acquired_time}:{latitude:.5f}:{longitude:.5f}"
            hotspots.append(
                Hotspot(
                    id=hotspot_id,
                    source=source,
                    latitude=latitude,
                    longitude=longitude,
                    acquired_date=acquired_date,
                    acquired_time=acquired_time,
                    confidence=row.get("confidence") or None,
                    frp=float(row["frp"]) if row.get("frp") else None,
                    brightness=float(row.get("bright_ti4") or row.get("brightness"))
                    if row.get("bright_ti4") or row.get("brightness")
                    else None,
                    daynight=row.get("daynight") or None,
                )
            )
        return hotspots

    @staticmethod
    def _deduplicate(hotspots: list[Hotspot]) -> list[Hotspot]:
        unique: dict[str, Hotspot] = {}
        for hotspot in hotspots:
            unique[hotspot.id] = hotspot
        return sorted(unique.values(), key=lambda item: (item.acquired_date, item.acquired_time, item.id))
