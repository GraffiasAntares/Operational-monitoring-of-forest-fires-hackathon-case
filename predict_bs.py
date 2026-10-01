"""Predict four-class BS masks from a saved compact U-Net checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from ml.bs import BurnSeverityInferenceDataset, BurnSeverityPreprocessor, FeatureStatistics
from ml.bs.model import CompactUNet
from ml.bs.training import feature_indices
from ml.submission import write_submission


def _rgb(image: np.ndarray) -> np.ndarray:
    return np.clip(np.moveaxis(image[[6, 3, 2]].astype(np.float32) / 3000.0, 0, -1), 0, 1)


def _render_preview(path: Path, layers: dict[str, np.ndarray], prediction: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (
        (_rgb(layers["s2_pre"]), "Sentinel-2 before", None),
        (_rgb(layers["s2_post"]), "Sentinel-2 after", None),
        (prediction, "U-Net prediction", "viridis"),
    )
    for axis, (image, title, cmap) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=3 if cmap else None)
        axis.set_title(title)
        axis.set_axis_off()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--output", required=True, help="Output .npz file containing chip_id -> uint8 mask")
    parser.add_argument("--template", help="Optional BS-only submission template; a combined AF+BS template needs AF predictions too")
    parser.add_argument("--submission", help="Destination CSV; requires --template")
    parser.add_argument("--preview-dir", help="Directory for visual previews of the most extensively predicted burns")
    parser.add_argument("--preview-limit", type=int, default=12, help="Number of preview PNGs to save; default: 12")
    parser.add_argument("--log", help="Optional JSON summary of the test inference run")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if bool(args.template) != bool(args.submission):
        raise ValueError("--template and --submission must be provided together")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CompactUNet(config["input_channels"], config["base_channels"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preprocessor = BurnSeverityPreprocessor(FeatureStatistics.load(args.preprocessing))
    indices = np.asarray(feature_indices(config["feature_mode"]))
    dataset = BurnSeverityInferenceDataset(args.data_dir)
    predictions: dict[str, np.ndarray] = {}
    started = perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            image = dataset.load(index, preprocessor)["image"]
            tensor = torch.from_numpy(np.asarray(image)[indices].copy()).unsqueeze(0).to(device)
            predictions[record.chip_id] = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **predictions)
    flat_predictions = np.concatenate([prediction.reshape(-1) for prediction in predictions.values()])
    class_counts = np.bincount(flat_predictions, minlength=4)
    preview_ids: list[str] = []
    if args.preview_dir:
        if args.preview_limit < 1:
            raise ValueError("--preview-limit must be at least 1")
        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        burn_fraction = {chip_id: float(np.mean(prediction > 0)) for chip_id, prediction in predictions.items()}
        preview_ids = sorted(burn_fraction, key=burn_fraction.get, reverse=True)[: args.preview_limit]
        record_index = {record.chip_id: index for index, record in enumerate(dataset.records)}
        for chip_id in preview_ids:
            _render_preview(preview_dir / f"{chip_id}.png", dataset.load_layers(record_index[chip_id]), predictions[chip_id])
    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "preprocessing": str(Path(args.preprocessing)),
        "device": str(device),
        "feature_mode": config["feature_mode"],
        "chips": len(predictions),
        "shape": list(next(iter(predictions.values())).shape),
        "class_pixel_counts": {str(label): int(count) for label, count in enumerate(class_counts)},
        "class_pixel_shares": {str(label): float(count / flat_predictions.size) for label, count in enumerate(class_counts)},
        "preview_chip_ids": preview_ids,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.template:
        write_submission(args.template, predictions, args.submission)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
