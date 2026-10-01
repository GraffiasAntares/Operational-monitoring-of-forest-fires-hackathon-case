"""Tune fixed BS logit biases on held-out fire events, without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.bs import BurnSeverityDataset, BurnSeverityPreprocessor, FeatureStatistics
from ml.bs.baseline import evaluate_bs, metrics_dict
from ml.bs.calibration import calibrated_argmax
from ml.bs.model import CompactUNet
from ml.bs.training import feature_indices


def read_validation_ids(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "bs" or not isinstance(payload.get("validation"), list):
        raise ValueError("split must be a BS JSON file containing validation IDs")
    return set(payload["validation"])


def score(logits: np.ndarray, target: np.ndarray, biases: np.ndarray) -> dict[str, float]:
    return metrics_dict(evaluate_bs([calibrated_argmax(logits, biases)], [target]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", default="ml/artifacts/bs_unet_full_tuned_seed42/best.pt")
    parser.add_argument("--preprocessing", default="ml/artifacts/bs_preprocessing_seed42.npz")
    parser.add_argument("--output", required=True, help="Calibration JSON with the selected four biases")
    parser.add_argument("--min-bias", type=float, default=-.30)
    parser.add_argument("--max-bias", type=float, default=.30)
    parser.add_argument("--step", type=float, default=.05)
    parser.add_argument("--passes", type=int, default=2, help="Coordinate-descent passes over classes 1, 2, 3")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.min_bias <= 0 <= args.max_bias or args.step <= 0 or args.passes < 1:
        raise ValueError("range must include zero, step must be positive, and passes must be at least one")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CompactUNet(config["input_channels"], config["base_channels"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preprocessor = BurnSeverityPreprocessor(FeatureStatistics.load(args.preprocessing))
    selected = np.asarray(feature_indices(config["feature_mode"]))
    dataset = BurnSeverityDataset(args.data_dir)
    validation_ids = read_validation_ids(args.split)

    logits_parts, target_parts = [], []
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            if record.chip_id not in validation_ids:
                continue
            sample = dataset.load(index, preprocessor)
            image = torch.from_numpy(np.asarray(sample["image"])[selected].copy()).unsqueeze(0).to(device)
            logits = model(image).squeeze(0).permute(1, 2, 0).cpu().numpy()
            target = np.asarray(sample["target"], dtype=np.uint8)
            valid = target != 255
            logits_parts.append(logits[valid])
            target_parts.append(target[valid])
    logits = np.concatenate(logits_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    candidates = np.arange(args.min_bias, args.max_bias + args.step / 2, args.step).round(6)

    biases = np.zeros(4, dtype=np.float32)
    baseline = score(logits, target, biases)
    history: list[dict[str, object]] = []
    for pass_index in range(args.passes):
        for class_id in (1, 2, 3):
            options: list[tuple[float, dict[str, float]]] = []
            for candidate in candidates:
                trial = biases.copy()
                trial[class_id] = candidate
                options.append((float(candidate), score(logits, target, trial)))
            value, result = max(options, key=lambda item: (item[1]["local_score"], item[1]["miou_severity"]))
            biases[class_id] = value
            history.append({"pass": pass_index + 1, "class_id": class_id, "chosen_bias": value, "metrics": result})

    final = score(logits, target, biases)
    report = {
        "method": "coordinate descent over additive class logits; class-0 bias fixed at zero",
        "checkpoint": str(args.checkpoint),
        "preprocessing": str(args.preprocessing),
        "split": str(args.split),
        "validation_chips": len(validation_ids),
        "valid_pixels": int(len(target)),
        "search": {"min_bias": args.min_bias, "max_bias": args.max_bias, "step": args.step, "passes": args.passes},
        "baseline_biases": [0.0, 0.0, 0.0, 0.0],
        "baseline_metrics": baseline,
        "biases": [float(item) for item in biases],
        "calibrated_metrics": final,
        "history": history,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "biases": report["biases"], "calibrated": final}, ensure_ascii=False))


if __name__ == "__main__":
    main()
