"""Tune and evaluate a four-class dNBR threshold baseline for BS.

The three ordered thresholds map dNBR to classes 0, 1, 2, 3.  Thresholds are
tuned only on the train fire events in a split JSON, then evaluated once on
the held-out fire events.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.metrics import burn_iou, class_iou, severity_miou


def dnbr(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Compute NBR(pre) - NBR(post); inputs contain B8A then B12."""
    pre = pre.astype(np.float32, copy=False)
    post = post.astype(np.float32, copy=False)
    nbr_pre = (pre[0] - pre[1]) / (pre[0] + pre[1] + 1e-6)
    nbr_post = (post[0] - post[1]) / (post[0] + post[1] + 1e-6)
    return nbr_pre - nbr_post


def predict(values: np.ndarray, thresholds: tuple[float, float, float]) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.uint8)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    burn = burn_iou(prediction, target)
    miou = severity_miou(prediction, target)
    # This is the BS contribution to the competition score. AF is intentionally
    # not included because this script evaluates BS only.
    return {
        "iou_burn": burn,
        "iou_class_1": class_iou(prediction, target, 1),
        "iou_class_2": class_iou(prediction, target, 2),
        "iou_class_3": class_iou(prediction, target, 3),
        "miou_sev": miou,
        "bs_weighted_score": .35 * burn + .30 * miou,
    }


def read_chip(data_dir: Path, chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    root = data_dir / "train" / "bs"
    with rasterio.open(root / "sentinel2_pre" / f"{chip_id}_Sentinel-2_pre.tif") as source:
        pre = source.read((7, 9))  # B8A, B12
    with rasterio.open(root / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif") as source:
        post = source.read((7, 9))
    with rasterio.open(root / "masks" / f"{chip_id}_MASK.tif") as source:
        target = source.read(1)
    return dnbr(pre, post), target


def load_values(data_dir: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values, targets = [], []
    for chip_id in ids:
        value, target = read_chip(data_dir, chip_id)
        valid = target != 255
        values.append(value[valid].ravel())
        targets.append(target[valid].ravel())
    return np.concatenate(values), np.concatenate(targets)


def tune(values: np.ndarray, target: np.ndarray, candidates: np.ndarray, top_k: int) -> list[dict[str, object]]:
    """Exhaustively search ordered threshold triples on train pixels."""
    # Re-evaluating every pixel for every triple would be prohibitively slow
    # (millions of pixels times thousands of triples).  For fixed candidate
    # thresholds, cumulative counts per true class are sufficient to recreate
    # the 4x4 confusion matrix exactly.
    sorted_by_class = [np.sort(values[target == class_id]) for class_id in range(4)]
    cumulative = np.array([
        np.searchsorted(class_values, candidates, side="left")
        for class_values in sorted_by_class
    ])
    totals = np.array([len(class_values) for class_values in sorted_by_class], dtype=np.int64)

    def fast_metrics(first: int, second: int, third: int) -> dict[str, float]:
        confusion = np.empty((4, 4), dtype=np.int64)
        confusion[:, 0] = cumulative[:, first]
        confusion[:, 1] = cumulative[:, second] - cumulative[:, first]
        confusion[:, 2] = cumulative[:, third] - cumulative[:, second]
        confusion[:, 3] = totals - cumulative[:, third]
        diagonals = np.diag(confusion)
        predicted = confusion.sum(axis=0)
        class_iou = diagonals / np.maximum(totals + predicted - diagonals, 1)
        tp_burn = int(confusion[1:, 1:].sum())
        actual_burn = int(totals[1:].sum())
        predicted_burn = int(confusion[:, 1:].sum())
        burn = tp_burn / max(actual_burn + predicted_burn - tp_burn, 1)
        miou = float(class_iou[1:].mean())
        return {"iou_burn": float(burn), "miou_sev": miou, "bs_weighted_score": .35 * float(burn) + .30 * miou}

    results: list[dict[str, object]] = []
    for first, second, third in combinations(range(len(candidates)), 3):
        result = fast_metrics(first, second, third)
        results.append({"thresholds": [round(float(candidates[item]), 5) for item in (first, second, third)], **result})
    results.sort(key=lambda item: (float(item["bs_weighted_score"]), float(item["miou_sev"])), reverse=True)
    return results[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True, help="JSON from scripts/build_ml_split.py for kind=bs")
    parser.add_argument("--min-threshold", type=float, default=-.10)
    parser.add_argument("--max-threshold", type=float, default=.70)
    parser.add_argument("--step", type=float, default=.02)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", required=True, help="JSON report path")
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    if split.get("kind") != "bs":
        raise ValueError("split must be for kind=bs")
    candidates = np.arange(args.min_threshold, args.max_threshold + args.step / 2, args.step)
    if len(candidates) < 3:
        raise ValueError("threshold range must contain at least three values")

    data_dir = Path(args.data_dir)
    train_values, train_target = load_values(data_dir, split["train"])
    validation_values, validation_target = load_values(data_dir, split["validation"])
    ranked = tune(train_values, train_target, candidates, args.top_k)
    for item in ranked:
        item["validation"] = metrics(predict(validation_values, tuple(item["thresholds"])), validation_target)

    report = {
        "method": "dNBR = NBR_pre - NBR_post; NBR = (B8A - B12) / (B8A + B12)",
        "split": {"path": str(args.split), "train_chips": len(split["train"]), "validation_chips": len(split["validation"]), "seed": split.get("seed")},
        "search": {"min_threshold": args.min_threshold, "max_threshold": args.max_threshold, "step": args.step, "candidate_count": len(candidates), "tested_triples": int(len(candidates) * (len(candidates) - 1) * (len(candidates) - 2) / 6)},
        "best_train": ranked[0],
        "top_train_configurations": ranked,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    best = ranked[0]
    print(f"best train thresholds={best['thresholds']} train={metrics(predict(train_values, tuple(best['thresholds'])), train_target)}")
    print(f"held-out validation={best['validation']}")
    print(f"wrote {output}; tested {report['search']['tested_triples']} ordered triples")


if __name__ == "__main__":
    main()
