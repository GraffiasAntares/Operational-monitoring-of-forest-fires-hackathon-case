"""Evaluate a saved BS U-Net checkpoint on the fixed grouped validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ml.bs import BurnSeverityDataset, BurnSeverityPreprocessor, FeatureStatistics
from ml.bs.baseline import evaluate_bs, metrics_dict
from ml.bs.model import CompactUNet
from ml.bs.training import feature_indices


def _validation_ids(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "bs":
        raise ValueError("evaluation requires a BS split")
    return set(payload["validation"])


def _render(path: Path, layers: dict[str, np.ndarray], target: np.ndarray, prediction: np.ndarray) -> None:
    def rgb(image: np.ndarray) -> np.ndarray:
        return np.clip(np.moveaxis(image[[6, 3, 2]].astype(np.float32) / 3000.0, 0, -1), 0, 1)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = ((rgb(layers["s2_pre"]), "S2 before", None), (target, "Target", "viridis"), (prediction, "U-Net prediction", "viridis"))
    for axis, (image, title, cmap) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=3 if cmap else None)
        axis.set_title(title)
        axis.set_axis_off()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visualizations", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CompactUNet(config["input_channels"], config["base_channels"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preprocessor = BurnSeverityPreprocessor(FeatureStatistics.load(args.preprocessing))
    selected = np.asarray(feature_indices(config["feature_mode"]))
    dataset = BurnSeverityDataset(args.data_dir)
    validation_indices = [index for index, record in enumerate(dataset.records) if record.chip_id in _validation_ids(args.split)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {}
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for position, index in enumerate(validation_indices):
            layers = dataset.load_layers(index)
            sample = dataset.load(index, preprocessor)
            target = np.asarray(sample["target"])
            image = torch.from_numpy(np.asarray(sample["image"])[selected].copy()).unsqueeze(0).to(device)
            prediction = model(image).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            chip_id = dataset.records[index].chip_id
            predictions[chip_id] = prediction
            targets.append(target)
            if position < args.visualizations:
                _render(output / f"{chip_id}.png", layers, target, prediction)
    metrics = metrics_dict(evaluate_bs(predictions.values(), targets))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output / "validation_predictions.npz", **predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
