"""Run the I4>threshold Active Fire baseline on an unlabeled test directory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.submission import encode_rle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Test root containing test/meta.csv and test/af/viirs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=330.0, help="I4 brightness-temperature threshold in K")
    args = parser.parse_args()

    root = Path(args.data_dir) / "test"
    output = Path(args.output_dir)
    mask_dir = output / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with (root / "meta.csv").open(encoding="utf-8", newline="") as stream:
        chip_ids = [row["chip_id"] for row in csv.DictReader(stream) if row["kind"] == "af"]
    if not chip_ids:
        raise ValueError("no AF chips in test/meta.csv")

    rows: list[dict[str, str | int]] = []
    positive_chips = 0
    positive_pixels = 0
    valid_pixels = 0
    for chip_id in chip_ids:
        path = root / "af" / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif"
        with rasterio.open(path) as source:
            # I1..I5 are bands 1..5; the supplied validity mask is band 8.
            i4, valid = source.read((4, 8))
        mask = ((i4 > args.threshold) & (valid > 0)).astype(np.uint8)
        np.save(mask_dir / f"{chip_id}.npy", mask)
        count = int(mask.sum())
        positive_chips += int(count > 0)
        positive_pixels += count
        valid_pixels += int(np.count_nonzero(valid))
        rows.append({"chip_id": chip_id, "class_id": 1, "rle": encode_rle(mask, 1)})

    predictions_path = output / "af_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("chip_id", "class_id", "rle"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "I4 > threshold AND supplied_valid > 0",
        "threshold_kelvin": args.threshold,
        "af_chips": len(chip_ids),
        "chips_with_predicted_fire": positive_chips,
        "predicted_fire_pixels": positive_pixels,
        "valid_pixels": valid_pixels,
        "predicted_fire_fraction": positive_pixels / valid_pixels if valid_pixels else 0.0,
        "predictions": str(predictions_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
