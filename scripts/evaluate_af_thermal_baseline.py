"""Evaluate the simple VIIRS thermal AF baseline on a prepared split.

The visual composite is R=I4, G=I5, B=I2.  Its binary baseline mask is
I4 > threshold (Kelvin); RGB is intentionally not used in the decision rule.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data import FireChipDataset, IGNORE_INDEX


def score(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    prediction, target = prediction[valid], target[valid]
    tp = int(np.count_nonzero(prediction & (target == 1)))
    fp = int(np.count_nonzero(prediction & (target == 0)))
    fn = int(np.count_nonzero(~prediction & (target == 1)))
    denominator = 2 * tp + fp + fn
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if denominator == 0 else 2 * tp / denominator
    iou = 0.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def split_ids(manifest: Path, split: str) -> set[str]:
    if split == "all":
        return set()
    with manifest.open(encoding="utf-8", newline="") as stream:
        return {row["chip_id"] for row in csv.DictReader(stream) if row["split"] == split}


def territory_id(dataset: FireChipDataset, chip_id: str) -> str:
    """A territory is one 96 km VIIRS chip footprint, shared across dates."""
    with rasterio.open(dataset._paths(chip_id)["viirs"]) as raster:
        bounds = raster.bounds
    x = round((bounds.left + bounds.right) / 2 / 1000)
    y = round((bounds.bottom + bounds.top) / 2 / 1000)
    return f"UTM37N_x{x:03d}km_y{y:04d}km"


def write_territory_report(path: Path, parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["territory", "chips", "valid_pixels", "fire_pixels", "f1", "iou", "precision", "recall", "tp", "fp", "fn"])
        writer.writeheader()
        for territory, entries in sorted(parts.items()):
            i4 = np.concatenate([item[0].ravel() for item in entries])
            target = np.concatenate([item[1].ravel() for item in entries])
            valid = np.concatenate([item[2].ravel() for item in entries]) & (target != IGNORE_INDEX)
            result = score(i4 > threshold, target, valid)
            writer.writerow({
                "territory": territory, "chips": len(entries), "valid_pixels": int(valid.sum()),
                "fire_pixels": int(np.count_nonzero((target == 1) & valid)), **result,
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prepared-dir", default="ml/artifacts/af_strict_v1")
    parser.add_argument("--split", choices=("train", "validation", "excluded", "all"), default="validation")
    parser.add_argument("--threshold", type=float, default=330.0)
    parser.add_argument("--search", action="store_true", help="Also find the best I4 threshold on validation.")
    parser.add_argument("--min-threshold", type=float, default=280.0)
    parser.add_argument("--max-threshold", type=float, default=400.0)
    parser.add_argument("--step", type=float, default=.25)
    parser.add_argument("--territory-report", help="Write one metric row per 96 km spatial footprint.")
    args = parser.parse_args()

    dataset = FireChipDataset(args.data_dir, "af")
    ids = split_ids(Path(args.prepared_dir) / "manifest.csv", args.split)
    if args.split != "all" and not ids:
        raise ValueError(f"no {args.split} chips in manifest")

    i4_parts, target_parts, valid_parts = [], [], []
    territories: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for index, record in enumerate(dataset.records):
        if args.split != "all" and record.chip_id not in ids:
            continue
        sample = dataset.load(index)
        # Stack order starts I1, I2, I3, I4, I5: zero-based index 3 is I4.
        i4 = np.asarray(sample["image"])[3]
        target = np.asarray(sample["target"], dtype=np.uint8)
        valid = np.asarray(sample["valid_mask"], dtype=bool)
        i4_parts.append(i4)
        target_parts.append(target)
        valid_parts.append(valid)
        if args.territory_report:
            territories.setdefault(territory_id(dataset, record.chip_id), []).append((i4, target, valid))

    i4 = np.concatenate([part.ravel() for part in i4_parts])
    target = np.concatenate([part.ravel() for part in target_parts])
    valid = np.concatenate([part.ravel() for part in valid_parts]) & (target != IGNORE_INDEX)

    def report(threshold: float, label: str) -> dict[str, float | int]:
        result = score(i4 > threshold, target, valid)
        print(
            f"{label}: I4>{threshold:.2f} F1={result['f1']:.4f} IoU={result['iou']:.4f} "
            f"precision={result['precision']:.4f} recall={result['recall']:.4f} "
            f"TP={result['tp']} FP={result['fp']} FN={result['fn']}"
        )
        return result

    report(args.threshold, "requested baseline")
    if args.search:
        thresholds = np.arange(args.min_threshold, args.max_threshold + args.step / 2, args.step)
        best_threshold = max(thresholds, key=lambda item: float(score(i4 > item, target, valid)["f1"]))
        report(float(best_threshold), "best validation threshold")
    if args.territory_report:
        report_path = Path(args.territory_report)
        write_territory_report(report_path, territories, args.threshold)
        print(f"territory report: {report_path} ({len(territories)} territories)")


if __name__ == "__main__":
    main()
