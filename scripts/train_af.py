"""Train the compact Active Fire U-Net on the supplied AF chips."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ml.af_model import AfUNet
from ml.af_prepare import build_af_features, normalize_af_features
from ml.data import FireChipDataset


class AfTorchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, source: FireChipDataset, ids: list[str], stats: dict[str, Any], augment: bool) -> None:
        self.source = source
        index_by_id = {record.chip_id: index for index, record in enumerate(source.records)}
        self.indices = [index_by_id[chip_id] for chip_id in ids]
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.source.load(self.indices[index])
        image = normalize_af_features(build_af_features(sample), self.mean, self.std)
        target = (np.asarray(sample["target"]) == 1).astype(np.float32)
        valid = np.asarray(sample["valid_mask"], dtype=np.bool_)
        if self.augment:
            if random.random() < .5:
                image, target, valid = image[:, :, ::-1].copy(), target[:, ::-1].copy(), valid[:, ::-1].copy()
            if random.random() < .5:
                image, target, valid = image[:, ::-1, :].copy(), target[::-1, :].copy(), valid[::-1, :].copy()
            rotations = random.randrange(4)
            if rotations:
                image = np.rot90(image, rotations, axes=(1, 2)).copy()
                target, valid = np.rot90(target, rotations).copy(), np.rot90(valid, rotations).copy()
        return {"image": torch.from_numpy(image), "target": torch.from_numpy(target)[None], "valid": torch.from_numpy(valid)[None]}


def focal_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, alpha: float = .75, gamma: float = 2.) -> torch.Tensor:
    valid_float = valid.float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probabilities = torch.sigmoid(logits)
    probability_true = probabilities * target + (1 - probabilities) * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * (1 - probability_true).pow(gamma)
    focal = (focal_weight * bce * valid_float).sum() / valid_float.sum().clamp_min(1)
    probabilities = probabilities * valid_float
    target = target * valid_float
    dice = 1 - (2 * (probabilities * target).sum() + 1) / (probabilities.sum() + target.sum() + 1)
    return .7 * focal + .3 * dice


def best_f1(probabilities: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    probabilities, target = probabilities[valid], target[valid]
    best = (0., .5)
    # Wildfire pixels are rare: calibrated probabilities can require a high
    # decision threshold.  Do not truncate the validation search at 0.80.
    for threshold in np.arange(.05, .991, .005):
        prediction = probabilities >= threshold
        tp = int(np.count_nonzero(prediction & (target == 1)))
        fp = int(np.count_nonzero(prediction & (target == 0)))
        fn = int(np.count_nonzero(~prediction & (target == 1)))
        f1 = 0. if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    probabilities, targets, masks = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(image)
            loss = focal_dice_loss(logits, target, valid)
        losses.append(float(loss.cpu()))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(target.cpu().numpy())
        masks.append(valid.cpu().numpy().astype(bool))
    f1, threshold = best_f1(np.concatenate(probabilities), np.concatenate(targets), np.concatenate(masks))
    return float(np.mean(losses)), f1, threshold


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prepared-dir", default="ml/artifacts/af_v1")
    parser.add_argument("--output-dir", default="ml/artifacts/af_v1/checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the NVIDIA-enabled PyTorch environment")
    set_seed(args.seed)
    prepared = Path(args.prepared_dir)
    stats = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    with (prepared / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    source = FireChipDataset(args.data_dir, "af")
    train_ids = [row["chip_id"] for row in manifest if row["split"] == "train"]
    validation_ids = [row["chip_id"] for row in manifest if row["split"] == "validation"]
    train_set = AfTorchDataset(source, train_ids, stats, augment=True)
    validation_set = AfTorchDataset(source, validation_ids, stats, augment=False)
    positive_by_id = {record.chip_id: float(record.metadata.get("n_fire_px") or 0) > 0 for record in source.records}
    positive_count = sum(positive_by_id[chip_id] for chip_id in train_ids)
    negative_count = len(train_ids) - positive_count
    weights = [(.7 / positive_count) if positive_by_id[chip_id] else (.3 / negative_count) for chip_id in train_ids]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ids), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda")
    model = AfUNet(in_channels=len(stats["feature_names"]), base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best, stalled = -1., 0
    history: list[dict[str, float]] = []
    print(f"device={torch.cuda.get_device_name(0)} parameters={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                loss = focal_dice_loss(model(image), target, valid) / args.accumulation
            scaler.scale(loss).backward()
            if step % args.accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()) * args.accumulation)
        scheduler.step()
        validation_loss, f1, threshold = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss, "validation_f1": f1, "threshold": threshold}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_loss={validation_loss:.4f} val_f1={f1:.4f} threshold={threshold:.2f}")
        if f1 > best:
            best, stalled = f1, 0
            torch.save({"model": model.state_dict(), "feature_names": stats["feature_names"], "normalization": stats, "base_channels": args.base_channels, "threshold": threshold, "validation_f1": f1, "epoch": epoch}, output / "best.pt")
        else:
            stalled += 1
            if stalled >= args.patience:
                print(f"early stopping after {args.patience} epochs without F1 improvement")
                break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best validation F1={best:.4f}; checkpoint={output / 'best.pt'}")


if __name__ == "__main__":
    main()
