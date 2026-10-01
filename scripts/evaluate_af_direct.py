"""Evaluate and calibrate a direct AF ResUNet checkpoint on strict validation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.af_direct import AfDirectResUNet
from ml.data import FireChipDataset
from scripts.train_af_direct import FullChipDataset, metrics


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the NVIDIA PyTorch environment")
    prepared = Path(args.prepared_dir)
    stats = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    with (prepared / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        validation_ids = [row["chip_id"] for row in csv.DictReader(stream) if row["split"] == "validation"]
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = AfDirectResUNet(len(stats["feature_names"]), int(checkpoint["base_channels"])).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(FullChipDataset(FireChipDataset(args.data_dir, "af"), validation_ids, stats), batch_size=1, shuffle=False, pin_memory=True)
    probabilities, targets, masks = [], [], []
    for batch in loader:
        logits = model(batch["image"].cuda(non_blocking=True))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["target"].numpy()); masks.append(batch["valid"].numpy().astype(bool))
    probability, target, valid = np.concatenate(probabilities), np.concatenate(targets), np.concatenate(masks)
    f1, threshold, precision, recall = metrics(probability, target, valid)
    prediction, truth = probability[valid] >= threshold, target[valid].astype(bool)
    tp = int(np.count_nonzero(prediction & truth)); fp = int(np.count_nonzero(prediction & ~truth)); fn = int(np.count_nonzero(~prediction & truth))
    report = {"f1": f1, "iou": 0. if tp + fp + fn == 0 else tp / (tp + fp + fn), "precision": precision, "recall": recall, "threshold": threshold, "valid_pixels": int(valid.sum()), "true_fire_pixels": int(truth.sum()), "checkpoint": str(Path(args.checkpoint))}
    output = Path(args.checkpoint).with_name("strict_validation_metrics.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)); print(f"metrics={output}")


if __name__ == "__main__":
    main()
