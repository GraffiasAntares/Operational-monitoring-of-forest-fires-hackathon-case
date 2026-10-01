"""Create one competition-ready submission from AF direct/thermal and BS models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml.bs import BurnSeverityInferenceDataset, BurnSeverityPreprocessor, FeatureStatistics
from ml.bs.baseline import AdaptiveDnbrBaseline
from ml.bs.calibration import load_biases
from ml.bs.hybrid import hybrid_prediction
from ml.bs.model import CompactUNet
from ml.bs.training import feature_indices
from ml.af_direct import AfDirectResUNet
from ml.af_residual import build_residual_features, normalize_residual_features
from ml.submission import write_submission


def af_thermal_predictions(data_dir: Path, threshold: float) -> dict[str, np.ndarray]:
    test_root = data_dir / "test"
    with (test_root / "meta.csv").open(encoding="utf-8", newline="") as stream:
        chip_ids = [row["chip_id"] for row in csv.DictReader(stream) if row.get("kind") == "af"]
    predictions: dict[str, np.ndarray] = {}
    for chip_id in chip_ids:
        path = test_root / "af" / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif"
        with rasterio.open(path) as source:
            i4, valid = source.read((4, 8))
        predictions[chip_id] = ((i4 > threshold) & (valid > 0)).astype(np.uint8)
    return predictions


def af_direct_predictions(data_dir: Path, checkpoint_path: Path, device: torch.device, threshold_override: float | None) -> tuple[dict[str, np.ndarray], float]:
    """Run the direct AF ResUNet; checkpoint contains its train-only statistics."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "af_direct_resunet":
        raise ValueError(f"{checkpoint_path} is not an AF direct ResUNet checkpoint")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict) or "feature_names" not in normalization:
        raise ValueError("AF checkpoint has no embedded normalization metadata")
    model = AfDirectResUNet(len(normalization["feature_names"]), int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    threshold = float(checkpoint["threshold"] if threshold_override is None else threshold_override)
    test_root = data_dir / "test"
    with (test_root / "meta.csv").open(encoding="utf-8", newline="") as stream:
        chip_ids = [row["chip_id"] for row in csv.DictReader(stream) if row.get("kind") == "af"]
    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for chip_id in chip_ids:
            viirs_path = test_root / "af" / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif"
            aux_path = test_root / "af" / "aux" / f"{chip_id}_AUX.tif"
            with rasterio.open(viirs_path) as source:
                viirs = source.read().astype(np.float32, copy=False)
            with rasterio.open(aux_path) as source:
                aux = source.read().astype(np.float32, copy=False)
            valid = viirs[-1] > 0
            sample = {"image": np.concatenate((viirs, aux)), "valid_mask": valid}
            image = normalize_residual_features(build_residual_features(sample), mean, std)
            tensor = torch.from_numpy(image.copy()).unsqueeze(0).to(device)
            probability = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
            predictions[chip_id] = ((probability >= threshold) & valid).astype(np.uint8)
    return predictions, threshold


def bs_predictions(data_dir: Path, checkpoint_path: Path, preprocessing_path: Path, calibration_path: Path | None, hybrid_path: Path | None, device: torch.device) -> dict[str, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CompactUNet(config["input_channels"], config["base_channels"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preprocessor = BurnSeverityPreprocessor(FeatureStatistics.load(preprocessing_path))
    biases = load_biases(calibration_path) if calibration_path is not None and calibration_path.is_file() else np.zeros(4, dtype=np.float32)
    hybrid_config = json.loads(hybrid_path.read_text(encoding="utf-8")) if hybrid_path is not None and hybrid_path.is_file() else None
    dnbr = AdaptiveDnbrBaseline({name: tuple(values) for name, values in hybrid_config["dnbr_thresholds"].items()}) if hybrid_config and hybrid_config.get("enabled") else None
    indices = np.asarray(feature_indices(config["feature_mode"]))
    dataset = BurnSeverityInferenceDataset(data_dir)
    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            layers = dataset.load_layers(index)
            image = preprocessor.transform(layers)
            tensor = torch.from_numpy(np.asarray(image)[indices].copy()).unsqueeze(0).to(device)
            logits = model(tensor).squeeze(0).cpu().numpy()
            predictions[record.chip_id] = hybrid_prediction(logits, dnbr.predict(layers) if dnbr is not None else np.zeros(logits.shape[1:], dtype=np.uint8), biases, hybrid_config)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Root of the official test archive")
    parser.add_argument("--output", required=True, help="Destination submission.csv")
    parser.add_argument("--bs-checkpoint", default="ml/artifacts/bs_unet_full_tuned_seed42/best.pt")
    parser.add_argument("--bs-preprocessing", default="ml/artifacts/bs_preprocessing_seed42.npz")
    parser.add_argument("--bs-logit-calibration", default="ml/artifacts/bs_unet_full_tuned_seed42/logit_calibration.json", help="Optional JSON from scripts/calibrate_bs_logits.py; neutral biases are used if absent")
    parser.add_argument("--bs-dnbr-hybrid", default="ml/artifacts/bs_unet_full_tuned_seed42/dnbr_hybrid.json", help="Optional train-calibrated dNBR fusion JSON; ignored when absent or disabled")
    parser.add_argument("--af-checkpoint", help="Direct AF ResUNet checkpoint. When omitted, the thermal I4 fallback is used.")
    parser.add_argument("--af-direct-threshold", type=float, help="Optional override for the threshold stored in --af-checkpoint")
    parser.add_argument("--af-threshold", type=float, default=330.0)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--masks-output", help="Optional .npz archive of AF and BS predicted masks")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_root = data_dir / "test"
    template = test_root / "sample_submission.csv"
    if not template.is_file():
        raise FileNotFoundError(f"sample submission not found: {template}")
    preprocessing_path = Path(args.bs_preprocessing)
    if not preprocessing_path.is_file():
        raise FileNotFoundError(
            f"BS preprocessing statistics not found: {preprocessing_path}. "
            "Create it with scripts/fit_bs_preprocessing.py on the training data and seed-42 split."
        )
    checkpoint_path = Path(args.bs_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BS checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)

    if args.af_checkpoint:
        af_checkpoint = Path(args.af_checkpoint)
        if not af_checkpoint.is_file():
            raise FileNotFoundError(f"AF checkpoint not found: {af_checkpoint}")
        af, af_direct_threshold = af_direct_predictions(data_dir, af_checkpoint, device, args.af_direct_threshold)
        af_mode = "direct_resunet"
    else:
        af = af_thermal_predictions(data_dir, args.af_threshold)
        af_direct_threshold = None
        af_mode = "thermal_i4"
    calibration_path = Path(args.bs_logit_calibration) if args.bs_logit_calibration else None
    hybrid_path = Path(args.bs_dnbr_hybrid) if args.bs_dnbr_hybrid else None
    bs = bs_predictions(data_dir, checkpoint_path, preprocessing_path, calibration_path, hybrid_path, device)
    predictions = af | bs
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(template, predictions, output)
    if args.masks_output:
        masks_path = Path(args.masks_output)
        masks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(masks_path, **predictions)
    summary = {
        "submission": str(output), "af_chips": len(af), "bs_chips": len(bs),
        "af_mode": af_mode, "af_threshold_kelvin": args.af_threshold if af_mode == "thermal_i4" else None,
        "af_direct_checkpoint": args.af_checkpoint if af_mode == "direct_resunet" else None,
        "af_direct_threshold": af_direct_threshold, "device": str(device),
        "bs_logit_calibration": str(calibration_path) if calibration_path is not None and calibration_path.is_file() else None,
        "bs_dnbr_hybrid": str(hybrid_path) if hybrid_path is not None and hybrid_path.is_file() else None,
        "af_positive_pixels": int(sum(mask.sum() for mask in af.values())),
        "bs_class_pixels": {str(class_id): int(sum(np.count_nonzero(mask == class_id) for mask in bs.values())) for class_id in range(4)},
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
