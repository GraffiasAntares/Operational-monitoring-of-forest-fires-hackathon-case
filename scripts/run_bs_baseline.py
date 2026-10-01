"""Fit and evaluate the adaptive dNBR BS baseline; render validation chips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.bs import BurnSeverityDataset
from ml.bs.baseline import AdaptiveDnbrBaseline, compute_dnbr, evaluate_bs, metrics_dict


def _read_split(path: str | Path) -> tuple[set[str], set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "bs":
        raise ValueError("baseline requires a BS split")
    return set(payload["train"]), set(payload["validation"])


def _render(path: Path, layers: dict[str, np.ndarray], target: np.ndarray, prediction: np.ndarray) -> None:
    pre, post = layers["s2_pre"], layers["s2_post"]

    def rgb(image: np.ndarray) -> np.ndarray:
        return np.clip(np.moveaxis(image[[6, 3, 2]].astype(np.float32) / 3000.0, 0, -1), 0, 1)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = (
        (rgb(pre), "S2 before", None, None, None),
        (rgb(post), "S2 after", None, None, None),
        (compute_dnbr(layers), "dNBR", "RdYlGn_r", -0.4, 1.0),
        (np.hstack((target, prediction)), "Target | baseline", "viridis", 0, 3),
    )
    for axis, (image, title, cmap, lower, upper) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_axis_off()
    axes[3].axvline(target.shape[1] - 0.5, color="white", linewidth=2)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visualizations", type=int, default=12)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_ids, validation_ids = _read_split(args.split)
    dataset = BurnSeverityDataset(args.data_dir)
    train_indices = [i for i, record in enumerate(dataset.records) if record.chip_id in train_ids]
    validation_indices = [i for i, record in enumerate(dataset.records) if record.chip_id in validation_ids]
    train_samples = ((dataset.load_layers(index), dataset.load(index)["target"]) for index in train_indices)
    baseline = AdaptiveDnbrBaseline().fit(train_samples)
    baseline.save(output / "thresholds.json")
    predictions: dict[str, np.ndarray] = {}
    targets: list[np.ndarray] = []
    for position, index in enumerate(validation_indices):
        layers = dataset.load_layers(index)
        target = dataset.load(index)["target"]
        prediction = baseline.predict(layers)
        chip_id = dataset.records[index].chip_id
        predictions[chip_id] = prediction
        targets.append(target)
        if position < args.visualizations:
            _render(output / f"{chip_id}.png", layers, target, prediction)
    metrics = evaluate_bs(predictions.values(), targets)
    (output / "metrics.json").write_text(json.dumps(metrics_dict(metrics), indent=2), encoding="utf-8")
    np.savez_compressed(output / "validation_predictions.npz", **predictions)
    print(json.dumps(metrics_dict(metrics), indent=2))


if __name__ == "__main__":
    main()
