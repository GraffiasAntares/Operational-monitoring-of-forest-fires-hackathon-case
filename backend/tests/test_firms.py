import asyncio
from datetime import date

import httpx
from backend.app.core.config import Settings
from backend.app.services.firms import FirmsClient


CSV = """latitude,longitude,bright_ti4,acq_date,acq_time,satellite,confidence,frp,daynight
56.1200,92.8400,327.4,2026-09-17,621,N,high,8.5,D
56.1200,92.8400,327.4,2026-09-17,621,N,high,8.5,D
"""


def test_fetches_both_viirs_sources_and_deduplicates_each_source():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(200, text=CSV)

    transport = httpx.MockTransport(handler)
    async def fetch():
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = FirmsClient(Settings(firms_map_key="test-key"), http_client)
            return await client.fetch_hotspots((92.5, 55.8, 93.2, 56.4), days=2)

    hotspots = asyncio.run(fetch())

    assert len(requested_urls) == 2
    assert any("VIIRS_SNPP_NRT" in url for url in requested_urls)
    assert any("VIIRS_NOAA20_NRT" in url for url in requested_urls)
    assert len(hotspots) == 2
    assert hotspots[0].acquired_time == "0621"
    assert hotspots[0].frp == 8.5


def test_historical_request_adds_start_date_to_firms_url():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(200, text=CSV)

    transport = httpx.MockTransport(handler)
    async def fetch():
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = FirmsClient(Settings(firms_map_key="test-key"), http_client)
            await client.fetch_hotspots((92.5, 55.8, 93.2, 56.4), days=1, start_date=date(2026, 9, 17))

    asyncio.run(fetch())

    assert all("VIIRS_" in url and "_SP/" in url for url in requested_urls)
    assert all(url.endswith("/1/2026-09-17") for url in requested_urls)
