"""Fit train-only dNBR priors and validate confidence-gated U-Net fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.bs import BurnSeverityDataset, BurnSeverityPreprocessor, FeatureStatistics
from ml.bs.baseline import AdaptiveDnbrBaseline, evaluate_bs, metrics_dict
from ml.bs.calibration import calibrated_argmax, load_biases
from ml.bs.hybrid import hybrid_from_flat_logits
from ml.bs.model import CompactUNet
from ml.bs.training import feature_indices


def read_split(path: str | Path) -> tuple[set[str], set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "bs":
        raise ValueError("split must describe BS")
    return set(payload["train"]), set(payload["validation"])


def score(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics_dict(evaluate_bs([prediction], [target]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", default="ml/artifacts/bs_unet_full_tuned_seed42/best.pt")
    parser.add_argument("--preprocessing", default="ml/artifacts/bs_preprocessing_seed42.npz")
    parser.add_argument("--calibration", default="ml/artifacts/bs_unet_full_tuned_seed42/logit_calibration.json")
    parser.add_argument("--output", required=True, help="Hybrid JSON consumed by inference.py")
    parser.add_argument("--min-confidence", type=float, default=.35)
    parser.add_argument("--max-confidence", type=float, default=.90)
    parser.add_argument("--confidence-step", type=float, default=.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= args.max_confidence <= 1 or args.confidence_step <= 0:
        raise ValueError("confidence range must be within [0, 1] and step must be positive")

    train_ids, validation_ids = read_split(args.split)
    dataset = BurnSeverityDataset(args.data_dir)
    index_by_id = {record.chip_id: index for index, record in enumerate(dataset.records)}
    train_indices = [index_by_id[chip_id] for chip_id in train_ids]
    validation_indices = [index_by_id[chip_id] for chip_id in validation_ids]

    def train_samples():
        for index in train_indices:
            layers = dataset.load_layers(index)
            target = layers.pop("target")[0].astype(np.uint8, copy=False)
            yield layers, target

    dnbr = AdaptiveDnbrBaseline().fit(train_samples())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CompactUNet(config["input_channels"], config["base_channels"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preprocessor = BurnSeverityPreprocessor(FeatureStatistics.load(args.preprocessing))
    selected = np.asarray(feature_indices(config["feature_mode"]))
    biases = load_biases(args.calibration)

    logits_parts, target_parts, dnbr_parts = [], [], []
    with torch.inference_mode():
        for index in validation_indices:
            layers = dataset.load_layers(index)
            target = layers.pop("target")[0].astype(np.uint8, copy=False)
            image = preprocessor.transform(layers)
            tensor = torch.from_numpy(np.asarray(image)[selected].copy()).unsqueeze(0).to(device)
            logits = model(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
            dnbr_labels = dnbr.predict(layers)
            valid = target != 255
            logits_parts.append(logits[valid])
            target_parts.append(target[valid])
            dnbr_parts.append(dnbr_labels[valid])
    logits = np.concatenate(logits_parts)
    target = np.concatenate(target_parts)
    dnbr_labels = np.concatenate(dnbr_parts)
    baseline = score(calibrated_argmax(logits, biases), target)

    thresholds = np.arange(args.min_confidence, args.max_confidence + args.confidence_step / 2, args.confidence_step).round(6)
    candidates: list[dict[str, object]] = []
    for mode in ("replace", "promote", "burn_promote", "severity_only"):
        for threshold in thresholds:
            result = score(hybrid_from_flat_logits(logits, dnbr_labels, biases, mode, float(threshold)), target)
            candidates.append({"mode": mode, "confidence_threshold": float(threshold), "metrics": result})
    best = max(candidates, key=lambda item: (float(item["metrics"]["local_score"]), float(item["metrics"]["miou_severity"])))
    enabled = float(best["metrics"]["local_score"]) > baseline["local_score"]
    report = {
        "method": "train-only adaptive dNBR labels replace calibrated U-Net labels only at low confidence",
        "checkpoint": str(args.checkpoint), "preprocessing": str(args.preprocessing), "calibration": str(args.calibration), "split": str(args.split),
        "train_chips": len(train_ids), "validation_chips": len(validation_ids), "valid_pixels": int(len(target)),
        "dnbr_thresholds": {name: list(values) for name, values in dnbr.thresholds.items()},
        "baseline_metrics": baseline,
        "enabled": enabled,
        "mode": best["mode"],
        "confidence_threshold": best["confidence_threshold"],
        "hybrid_metrics": best["metrics"],
        "tested_configurations": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "enabled": enabled, "best": best}, ensure_ascii=False))


if __name__ == "__main__":
    main()
