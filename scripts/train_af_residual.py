"""Train a two-head U-Net that corrects I4 thermal AF candidates, not fire from scratch."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.af_model import AfUNet
from ml.af_residual import (
    BASELINE_INDEX,
    LOW_CANDIDATE_INDEX,
    build_residual_features,
    normalize_residual_features,
    residual_targets,
)
from ml.data import FireChipDataset


class ResidualDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, source: FireChipDataset, ids: list[str], stats: dict[str, Any], augment: bool) -> None:
        index_by_id = {record.chip_id: index for index, record in enumerate(source.records)}
        self.source, self.indices, self.augment = source, [index_by_id[item] for item in ids], augment
        self.mean, self.std = np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.source.load(self.indices[item])
        image = build_residual_features(sample)
        valid = np.asarray(sample["valid_mask"], dtype=bool)
        baseline, candidate = image[BASELINE_INDEX] > 0, image[LOW_CANDIDATE_INDEX] > 0
        add, remove, add_valid, remove_valid = residual_targets(np.asarray(sample["target"]), valid, baseline, candidate)
        image = normalize_residual_features(image, self.mean, self.std)
        labels = np.stack((add, remove, add_valid, remove_valid)).astype(np.float32)
        if self.augment:
            if random.random() < .5:
                image, labels = image[:, :, ::-1].copy(), labels[:, :, ::-1].copy()
            if random.random() < .5:
                image, labels = image[:, ::-1, :].copy(), labels[:, ::-1, :].copy()
            turns = random.randrange(4)
            if turns:
                image, labels = np.rot90(image, turns, axes=(1, 2)).copy(), np.rot90(labels, turns, axes=(1, 2)).copy()
        return {"image": torch.from_numpy(image), "target": torch.from_numpy(labels[:2]), "supervision": torch.from_numpy(labels[2:])}


def masked_focal_dice(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, alpha: float = .8, gamma: float = 2.) -> torch.Tensor:
    weights = mask.float()
    if not bool(weights.any()):
        return logits.sum() * 0
    probability = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    true_probability = probability * target + (1 - probability) * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * (1 - true_probability).pow(gamma)
    focal = (focal_weight * bce * weights).sum() / weights.sum().clamp_min(1)
    probability, target = probability * weights, target * weights
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return .7 * focal + .3 * dice


def corrected_mask(image: torch.Tensor, logits: torch.Tensor, add_threshold: float, remove_threshold: float) -> torch.Tensor:
    base = image[:, BASELINE_INDEX] > 0
    candidate = image[:, LOW_CANDIDATE_INDEX] > 0
    add_probability, remove_probability = torch.sigmoid(logits[:, 0]), torch.sigmoid(logits[:, 1])
    result = base.clone()
    result[candidate & (add_probability >= add_threshold)] = True
    result[base & (remove_probability >= remove_threshold)] = False
    return result


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, add_threshold: float, remove_threshold: float) -> tuple[float, float]:
    model.eval()
    losses, predictions, targets, masks = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target, supervision = batch["target"].to(device), batch["supervision"].to(device)
        logits = model(image)
        loss = .6 * masked_focal_dice(logits[:, :1], target[:, :1], supervision[:, :1]) + .4 * masked_focal_dice(logits[:, 1:], target[:, 1:], supervision[:, 1:])
        prediction = corrected_mask(image, logits, add_threshold, remove_threshold)
        # Original target is baseline plus residual edits. It is reconstructed
        # from the two residual labels so evaluation remains chip-level F1.
        base = image[:, BASELINE_INDEX] > 0
        fire = (base | target[:, 0].bool()) & ~target[:, 1].bool()
        valid = (supervision[:, 0].bool() | supervision[:, 1].bool() | (image[:, 7] > 0))
        losses.append(float(loss.cpu()))
        predictions.append(prediction.cpu().numpy())
        targets.append(fire.cpu().numpy())
        masks.append(valid.cpu().numpy())
    valid = np.concatenate(masks)
    prediction = np.concatenate(predictions)[valid]
    target = np.concatenate(targets)[valid]
    tp = int(np.count_nonzero(prediction & target)); fp = int(np.count_nonzero(prediction & ~target)); fn = int(np.count_nonzero(~prediction & target))
    f1 = 0. if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return float(np.mean(losses)), f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--add-threshold", type=float, default=.5)
    parser.add_argument("--remove-threshold", type=float, default=.5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    prepared = Path(args.prepared_dir)
    stats = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    with (prepared / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train_ids = [row["chip_id"] for row in rows if row["split"] == "train"]
    validation_ids = [row["chip_id"] for row in rows if row["split"] == "validation"]
    source = FireChipDataset(args.data_dir, "af")
    train_data, validation_data = ResidualDataset(source, train_ids, stats, True), ResidualDataset(source, validation_ids, stats, False)
    # Oversample chips containing true fire; correction labels are then present
    # often enough without discarding real empty-scene hard negatives.
    index_by_id = {record.chip_id: index for index, record in enumerate(source.records)}
    positives = [bool(np.any(np.asarray(source.load(index_by_id[chip_id])["target"]) == 1)) for chip_id in train_ids]
    positive_count, negative_count = sum(positives), len(positives) - sum(positives)
    weights = [(.7 / positive_count) if positive else (.3 / negative_count) for positive in positives]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ids), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    device = torch.device("cuda")
    model = AfUNet(len(stats["feature_names"]), args.base_channels, .05, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1., 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []; optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            image = batch["image"].to(device, non_blocking=True); target = batch["target"].to(device); supervision = batch["supervision"].to(device)
            with torch.autocast(device_type="cuda"):
                logits = model(image)
                loss = (.6 * masked_focal_dice(logits[:, :1], target[:, :1], supervision[:, :1]) + .4 * masked_focal_dice(logits[:, 1:], target[:, 1:], supervision[:, 1:])) / args.accumulation
            scaler.scale(loss).backward()
            if step % args.accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * args.accumulation)
        scheduler.step()
        validation_loss, f1 = evaluate(model, validation_loader, device, args.add_threshold, args.remove_threshold)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss, "validation_f1": f1}
        history.append(record); print(json.dumps(record))
        if f1 > best:
            best, stale = f1, 0
            torch.save({"model": model.state_dict(), "feature_names": stats["feature_names"], "normalization": stats, "base_channels": args.base_channels, "add_threshold": args.add_threshold, "remove_threshold": args.remove_threshold, "validation_f1": f1, "epoch": epoch}, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best validation F1={best:.4f}; checkpoint={output / 'best.pt'}")


if __name__ == "__main__":
    main()
