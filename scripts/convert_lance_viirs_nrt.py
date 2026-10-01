"""Convert downloaded NOAA-20 VIIRS NRT granules to the AF inference layout.

The script intentionally keeps absent auxiliary sources explicit.  VIIRS supplies
the eight ``viirs`` bands (I1--I5, two zenith angles and validity); landcover,
DEM and weather are written as checkpoint-normalization means so they become
neutral after the AF model normalization.  This is useful for an integration
probe, but not a substitute for the real auxiliary-data providers in production.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform
from scipy.spatial import cKDTree


# Centers of the four downloaded probe areas.  The output footprint is always
# 256 x 256 at 375 m, matching the competition AF chips.
SAMPLES = {
    "russia_khabarovsk": (134.95, 48.35, "VJ102IMG_NRT.A2026262.0418.021.nc"),
    # The catalogue's nominal bounding box for the originally selected Yakutsk
    # granule did not contain Yakutsk on inspection.  Blagoveshchensk is in the
    # same fresh Russian overpass and has actual I-band coverage.
    "russia_blagoveshchensk": (127.53, 50.29, "VJ102IMG_NRT.A2026262.0418.021.nc"),
    "australia_darwin": (130.85, -12.45, "VJ102IMG_NRT.A2026262.0542.021.nc"),
    "australia_perth": (115.60, -32.10, "VJ102IMG_NRT.A2026262.0718.021.nc"),
}
PIXELS = 256
GSD_METERS = 375.0


def scalar_attribute(dataset: h5py.Dataset, name: str) -> float:
    return float(np.asarray(dataset.attrs[name]).reshape(-1)[0])


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def target_grid(lon: float, lat: float) -> tuple[np.ndarray, np.ndarray, CRS, object, tuple[float, float, float, float]]:
    """Return target lon/lat grids and UTM GeoTIFF metadata."""
    epsg = utm_epsg(lon, lat)
    center_x, center_y = transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    x_min = center_x[0] - PIXELS * GSD_METERS / 2
    y_max = center_y[0] + PIXELS * GSD_METERS / 2
    affine = from_origin(x_min, y_max, GSD_METERS, GSD_METERS)
    cols, rows = np.meshgrid(np.arange(PIXELS, dtype=np.float64), np.arange(PIXELS, dtype=np.float64))
    xs = x_min + (cols + .5) * GSD_METERS
    ys = y_max - (rows + .5) * GSD_METERS
    target_lon, target_lat = transform(f"EPSG:{epsg}", "EPSG:4326", xs.ravel(), ys.ravel())
    target_lon = np.asarray(target_lon, dtype=np.float32).reshape(PIXELS, PIXELS)
    target_lat = np.asarray(target_lat, dtype=np.float32).reshape(PIXELS, PIXELS)
    return target_lon, target_lat, CRS.from_epsg(epsg), affine, (float(target_lon.min()), float(target_lat.min()), float(target_lon.max()), float(target_lat.max()))


def nearest_indices(source_lon: np.ndarray, source_lat: np.ndarray, target_lon: np.ndarray, target_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_points = np.column_stack((source_lon.ravel(), source_lat.ravel()))
    tree = cKDTree(source_points)
    distance, indices = tree.query(np.column_stack((target_lon.ravel(), target_lat.ravel())), k=1)
    # 0.015 degree is deliberately wider than a VIIRS I-band pixel, but rejects
    # off-swath pixels instead of inventing edge values.
    valid = distance <= .015
    return indices.astype(np.int64), valid.reshape(PIXELS, PIXELS)


def read_resampled(image_path: Path, geo_path: Path, lon: float, lat: float) -> tuple[np.ndarray, np.ndarray, CRS, object, dict[str, object]]:
    target_lon, target_lat, crs, affine, bounds = target_grid(lon, lat)
    with h5py.File(geo_path, "r") as geo_file:
        geo = geo_file["geolocation_data"]
        source_lat_all = geo["latitude"][:]
        source_lon_all = geo["longitude"][:]
        margin = .12
        in_window = (
            (source_lat_all >= bounds[1] - margin) & (source_lat_all <= bounds[3] + margin) &
            (source_lon_all >= bounds[0] - margin) & (source_lon_all <= bounds[2] + margin)
        )
        locations = np.where(in_window)
        if not locations[0].size:
            raise ValueError(f"the granule {geo_path.name} does not cover the requested 96 km chip")
        row_slice = slice(int(locations[0].min()), int(locations[0].max()) + 1)
        col_slice = slice(int(locations[1].min()), int(locations[1].max()) + 1)
        source_lat = source_lat_all[row_slice, col_slice]
        source_lon = source_lon_all[row_slice, col_slice]
        solar = geo["solar_zenith"][row_slice, col_slice].astype(np.float32)
        sensor = geo["sensor_zenith"][row_slice, col_slice].astype(np.float32)
        solar = solar * scalar_attribute(geo["solar_zenith"], "scale_factor") + scalar_attribute(geo["solar_zenith"], "add_offset")
        sensor = sensor * scalar_attribute(geo["sensor_zenith"], "scale_factor") + scalar_attribute(geo["sensor_zenith"], "add_offset")
    finite_source = np.isfinite(source_lat) & np.isfinite(source_lon) & (np.abs(source_lat) <= 90) & (np.abs(source_lon) <= 180)
    flat_indices, nearest_valid = nearest_indices(source_lon[finite_source], source_lat[finite_source], target_lon, target_lat)
    with h5py.File(image_path, "r") as image_file:
        observations = image_file["observation_data"]
        bands: list[np.ndarray] = []
        source_valid = finite_source.copy()
        for number in range(1, 6):
            name = f"I{number:02d}"
            raw = observations[name][row_slice, col_slice]
            good = raw <= scalar_attribute(observations[name], "valid_max")
            source_valid &= good
            if number <= 3:
                values = raw.astype(np.float32) * scalar_attribute(observations[name], "scale_factor") + scalar_attribute(observations[name], "add_offset")
            else:
                lut = observations[f"{name}_brightness_temperature_lut"][:]
                values = lut[raw].astype(np.float32)
            bands.append(values)
    source_valid &= np.isfinite(solar) & np.isfinite(sensor)
    valid_values = source_valid[finite_source].ravel()[flat_indices].reshape(PIXELS, PIXELS) & nearest_valid
    result = np.empty((8, PIXELS, PIXELS), dtype=np.float32)
    for index, source_band in enumerate(bands):
        values = source_band[finite_source].ravel()[flat_indices].reshape(PIXELS, PIXELS)
        result[index] = np.where(valid_values, values, np.nan)
    for index, source_angle in ((5, solar), (6, sensor)):
        values = source_angle[finite_source].ravel()[flat_indices].reshape(PIXELS, PIXELS)
        result[index] = np.where(valid_values, values, np.nan)
    result[7] = valid_values.astype(np.float32)
    debug = {"source_window_shape": [int(source_lat.shape[0]), int(source_lat.shape[1])], "valid_fraction": float(valid_values.mean()), "epsg": crs.to_epsg()}
    return result, valid_values, crs, affine, debug


def observation_time(filename: str) -> datetime:
    match = re.search(r"A(\d{4})(\d{3})\.(\d{2})(\d{2})", filename)
    if not match:
        raise ValueError(f"cannot parse acquisition time from {filename}")
    return datetime.strptime("".join(match.groups()), "%Y%j%H%M").replace(tzinfo=timezone.utc)


def write_raster(path: Path, data: np.ndarray, crs: CRS, affine: object, descriptions: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=PIXELS, height=PIXELS, count=data.shape[0], dtype="float32", crs=crs, transform=affine, compress="deflate", nodata=np.nan) as destination:
        destination.write(data)
        destination.descriptions = descriptions


def auxiliary_placeholders(checkpoint_path: Path | None) -> tuple[np.ndarray, str]:
    if checkpoint_path is None:
        return np.zeros(5, dtype=np.float32), "zero_placeholders_no_checkpoint"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalization = checkpoint["normalization"]
    means = np.asarray(normalization["mean"], dtype=np.float32)
    # Channels 8--12 of the raw AF image: landcover, DEM, t2m, rh2m, wind.
    values = means[8:13].copy()
    values[0] = 0.0  # categorical landcover has no meaningful z-score mean.
    return values, "checkpoint_continuous_means_landcover_unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing VJ102IMG_NRT/VJ103IMG_NRT pairs")
    parser.add_argument("--output-dir", required=True, help="Destination compatible test/af layout")
    parser.add_argument("--af-checkpoint", help="Optional direct-AF checkpoint, used only for neutral auxiliary placeholders")
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    placeholders, aux_status = auxiliary_placeholders(Path(args.af_checkpoint) if args.af_checkpoint else None)
    rows: list[dict[str, object]] = []
    report: list[dict[str, object]] = []
    for name, (lon, lat, image_name) in SAMPLES.items():
        image_path = input_dir / image_name
        geo_path = input_dir / image_name.replace("VJ102IMG", "VJ103IMG")
        if not image_path.is_file() or not geo_path.is_file():
            raise FileNotFoundError(f"missing NRT pair for {name}: {image_path.name} / {geo_path.name}")
        viirs, valid, crs, affine, debug = read_resampled(image_path, geo_path, lon, lat)
        aux = np.broadcast_to(placeholders[:, None, None], (5, PIXELS, PIXELS)).copy()
        chip_id = f"AF_nrt_{name}"
        write_raster(output_dir / "test" / "af" / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif", viirs, crs, affine, ("I1", "I2", "I3", "I4_kelvin", "I5_kelvin", "solar_zenith_deg", "sensor_zenith_deg", "valid"))
        write_raster(output_dir / "test" / "af" / "aux" / f"{chip_id}_AUX.tif", aux, crs, affine, ("landcover_placeholder", "dem_placeholder", "t2m_placeholder", "rh2m_placeholder", "wind_speed_placeholder"))
        time = observation_time(image_name)
        bounds = rasterio.transform.array_bounds(PIXELS, PIXELS, affine)
        rows.append({"chip_id": chip_id, "kind": "af", "fire_event_id": "", "region": name, "epsg": float(crs.to_epsg()), "x_min": bounds[0], "y_min": bounds[1], "x_max": bounds[2], "y_max": bounds[3], "width": PIXELS, "height": PIXELS, "gsd": GSD_METERS, "acq_datetime": time.isoformat(), "satellite": "NOAA-20", "date_pre": "", "date_post": "", "s1_date_pre": "", "s1_date_post": "", "valid_frac": float(valid.mean()), "cloud_frac": "", "landcover_top": "", "n_fire_px": "", "burn_area_ha": "", "sev1_px": "", "sev2_px": "", "sev3_px": ""})
        report.append({"chip_id": chip_id, "region": name, "source_image": image_name, "source_geo": geo_path.name, "acq_datetime_utc": time.isoformat(), "center_lon_lat": [lon, lat], "auxiliary_status": aux_status, **debug})
    header = tuple(rows[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test" / "meta.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "chips": len(report), "auxiliary_status": aux_status, "report": str(output_dir / "conversion_report.json"), "valid_fractions": {row["chip_id"]: row["valid_frac"] for row in rows}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
