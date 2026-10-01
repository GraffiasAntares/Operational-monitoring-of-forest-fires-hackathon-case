"""Evaluate one AF residual checkpoint and calibrate its add/remove thresholds."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.af_model import AfUNet
from ml.af_residual import BASELINE_INDEX, LOW_CANDIDATE_INDEX, build_residual_features, normalize_residual_features, residual_targets
from ml.data import FireChipDataset


class EvaluationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, source: FireChipDataset, ids: list[str], stats: dict[str, Any]) -> None:
        indexes = {record.chip_id: index for index, record in enumerate(source.records)}
        self.source, self.indices = source, [indexes[chip_id] for chip_id in ids]
        self.mean, self.std = np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.source.load(self.indices[item])
        image = build_residual_features(sample)
        valid = np.asarray(sample["valid_mask"], dtype=bool)
        baseline, candidate = image[BASELINE_INDEX] > 0, image[LOW_CANDIDATE_INDEX] > 0
        add, remove, _, _ = residual_targets(np.asarray(sample["target"]), valid, baseline, candidate)
        image = normalize_residual_features(image, self.mean, self.std)
        return {"image": torch.from_numpy(image), "add": torch.from_numpy(add), "remove": torch.from_numpy(remove), "valid": torch.from_numpy(valid)}


def f1(prediction: np.ndarray, target: np.ndarray) -> float:
    tp = int(np.count_nonzero(prediction & target)); fp = int(np.count_nonzero(prediction & ~target)); fn = int(np.count_nonzero(~prediction & target))
    return 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True); parser.add_argument("--prepared-dir", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--add-thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--remove-thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script")
    prepared = Path(args.prepared_dir)
    stats = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    with (prepared / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        validation_ids = [row["chip_id"] for row in csv.DictReader(stream) if row["split"] == "validation"]
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = AfUNet(len(stats["feature_names"]), int(checkpoint["base_channels"]), .05, out_channels=2).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(EvaluationDataset(FireChipDataset(args.data_dir, "af"), validation_ids, stats), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    bases, candidates, add_probs, remove_probs, targets = [], [], [], [], []
    for batch in loader:
        image = batch["image"].cuda(non_blocking=True); logits = model(image)
        base, candidate = image[:, BASELINE_INDEX] > 0, image[:, LOW_CANDIDATE_INDEX] > 0
        target = (base | batch["add"].cuda().bool()) & ~batch["remove"].cuda().bool(); valid = batch["valid"].cuda().bool()
        bases.append(base[valid].cpu().numpy()); candidates.append(candidate[valid].cpu().numpy())
        add_probs.append(torch.sigmoid(logits[:, 0])[valid].cpu().numpy()); remove_probs.append(torch.sigmoid(logits[:, 1])[valid].cpu().numpy())
        targets.append(target[valid].cpu().numpy())
    base, candidate = np.concatenate(bases), np.concatenate(candidates)
    add_prob, remove_prob, target = np.concatenate(add_probs), np.concatenate(remove_probs), np.concatenate(targets)
    add_thresholds = [float(item) for item in args.add_thresholds.split(",")]; remove_thresholds = [float(item) for item in args.remove_thresholds.split(",")]
    results = [{"kind": "thermal_baseline", "f1": f1(base, target)}]
    for add_threshold, remove_threshold in product(add_thresholds, remove_thresholds):
        prediction = (base | (candidate & (add_prob >= add_threshold))) & ~(base & (remove_prob >= remove_threshold))
        results.append({"kind": "residual", "add_threshold": add_threshold, "remove_threshold": remove_threshold, "f1": f1(prediction, target)})
    results.sort(key=lambda row: row["f1"], reverse=True)
    output = Path(args.checkpoint).with_name("calibration.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results[:10], ensure_ascii=False, indent=2)); print(f"full calibration={output}")


if __name__ == "__main__":
    main()
