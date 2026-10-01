"""Download four small NOAA-20 VIIRS NRT samples and record their HDF5 layout."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import re
from typing import Any

import h5py
import requests


SAMPLES = {
    "russia_khabarovsk": (134.85, 48.25, 135.05, 48.45),
    "russia_yakutsk": (129.55, 62.00, 129.75, 62.20),
    "australia_darwin": (130.75, -12.55, 130.95, -12.35),
    "australia_perth": (115.75, -32.05, 115.95, -31.85),
}
API = "https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/details"


def token_from_env(path: Path | None) -> str:
    """Prefer the LAADS download token; generic EDL OAuth tokens may not stream files."""
    token = os.environ.get("NASA_LAADS_TOKEN", "").strip() or os.environ.get("NASA_EARTHDATA_TOKEN", "").strip()
    if not token and path and path.is_file():
        contents = path.read_text(encoding="utf-8")
        for variable in ("NASA_LAADS_TOKEN", "NASA_EARTHDATA_TOKEN"):
            match = re.search(rf"^\s*{variable}\s*=\s*['\"]?([^'\"\r\n#]+)", contents, re.MULTILINE)
            if match:
                token = match.group(1).strip()
                break
    if not token:
        raise RuntimeError("NASA_EARTHDATA_TOKEN is missing; use --env-file or the environment")
    return token


def records(session: requests.Session, product: str, bbox: tuple[float, float, float, float], days: int) -> list[dict[str, Any]]:
    west, south, east, north = bbox
    now = datetime.now(UTC).date()
    params = {"products": product, "archiveSets": "5201", "temporalRanges": f"{now - timedelta(days=days)}..{now}", "regions": f"[BBOX]W{west} N{north} E{east} S{south}", "formats": "json"}
    result: list[dict[str, Any]] = []
    for page in range(1, 10):
        response = session.get(API, params={**params, "page": page}, timeout=60)
        response.raise_for_status()
        payload = response.json(); result.extend(payload.get("content", []))
        if page >= int(payload.get("n_pages", 1)): break
    return result


def matching_geo(session: requests.Session, image: dict[str, Any], bbox: tuple[float, float, float, float], days: int) -> dict[str, Any]:
    stamp = image["name"].split(".")[1:3]
    candidates = records(session, "VJ103IMG_NRT", bbox, days)
    for item in candidates:
        if item["name"].split(".")[1:3] == stamp:
            return item
    raise RuntimeError(f"matching VJ103IMG_NRT not found for {image['name']}")


def download(session: requests.Session, item: dict[str, Any], output: Path) -> Path:
    destination = output / item["name"]
    if destination.is_file() and destination.stat().st_size == int(item["size"]): return destination
    with session.get(item["downloadsLink"], stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                if block: stream.write(block)
    with destination.open("rb") as stream:
        prefix = stream.read(32).lstrip()
    if prefix.startswith(b"<!DOCTYPE") or prefix.startswith(b"<html"):
        destination.unlink(missing_ok=True)
        raise RuntimeError("LANCE returned an Earthdata login page, not NetCDF. Add a LAADS download token as NASA_LAADS_TOKEN to .env.")
    if destination.stat().st_size != int(item["size"]):
        raise RuntimeError(f"incomplete download: {destination.name}")
    return destination


def layout(path: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    with h5py.File(path, "r") as source:
        def visit(name: str, item: h5py.Dataset | h5py.Group) -> None:
            if isinstance(item, h5py.Dataset):
                datasets[name] = {"shape": list(item.shape), "dtype": str(item.dtype), "attributes": {str(key): str(value)[:160] for key, value in item.attrs.items()}}
        source.visititems(visit)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--locations", nargs="*", choices=tuple(SAMPLES), help="Optional subset; use one location per run to keep transfers bounded.")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.headers["Authorization"] = f"Bearer {token_from_env(Path(args.env_file))}"
    report: dict[str, Any] = {"product": "NOAA-20 VJ102IMG_NRT + VJ103IMG_NRT", "requested_at_utc": datetime.now(UTC).isoformat(), "samples": {}}
    selected = args.locations or tuple(SAMPLES)
    for label in selected:
        bbox = SAMPLES[label]
        images = records(session, "VJ102IMG_NRT", bbox, args.lookback_days)
        if not images: raise RuntimeError(f"no recent image granule for {label}")
        image = max(images, key=lambda item: item["start"])
        geo = matching_geo(session, image, bbox, args.lookback_days)
        image_path, geo_path = download(session, image, output), download(session, geo, output)
        report["samples"][label] = {"bbox_wgs84": bbox, "image": {key: image[key] for key in ("name", "start", "size", "downloadsLink")}, "geo": {key: geo[key] for key in ("name", "start", "size", "downloadsLink")}, "image_layout": layout(image_path), "geo_layout": layout(geo_path)}
        print(f"{label}: {image['start']}Z; {image['name']}; {image['size'] / 1024 / 1024:.1f} MiB")
    (output / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
