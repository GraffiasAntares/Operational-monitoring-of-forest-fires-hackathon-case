"""Train a direct, patch-sampled residual U-Net for the AF task."""
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.af_direct import AfDirectResUNet
from ml.af_residual import build_residual_features, normalize_residual_features
from ml.data import FireChipDataset


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def choose_coordinate(mask: np.ndarray) -> tuple[int, int]:
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("cannot sample a crop centre from an empty mask")
    y, x = points[random.randrange(len(points))]
    return int(y), int(x)


def crop(image: np.ndarray, target: np.ndarray, valid: np.ndarray, centre: tuple[int, int], size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a jittered patch; reflect inputs and mark padded targets invalid."""
    height, width = target.shape
    y, x = centre
    jitter = size // 4
    y += random.randint(-jitter, jitter); x += random.randint(-jitter, jitter)
    top, left = y - size // 2, x - size // 2
    bottom, right = top + size, left + size
    pad_top, pad_bottom, pad_left, pad_right = max(0, -top), max(0, bottom - height), max(0, -left), max(0, right - width)
    if pad_top or pad_bottom or pad_left or pad_right:
        image = np.pad(image, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
        target = np.pad(target, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)
        valid = np.pad(valid, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=False)
        top += pad_top; left += pad_left
    return image[:, top:top + size, left:left + size].copy(), target[top:top + size, left:left + size].copy(), valid[top:top + size, left:left + size].copy()


class DirectPatchDataset(Dataset[dict[str, torch.Tensor]]):
    """Samples 50% fire, 25% thermal hard-negative and 25% background crops."""
    def __init__(self, source: FireChipDataset, ids: list[str], stats: dict[str, Any], patch_size: int, patches_per_epoch: int) -> None:
        by_id = {record.chip_id: index for index, record in enumerate(source.records)}
        self.source, self.indices = source, [by_id[chip_id] for chip_id in ids]
        self.mean, self.std = np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32)
        self.patch_size, self.patches_per_epoch = patch_size, patches_per_epoch
        self.fire_indices, self.hot_indices, self.background_indices = [], [], []
        for index in self.indices:
            sample = source.load(index)
            valid, target = np.asarray(sample["valid_mask"], dtype=bool), np.asarray(sample["target"]) == 1
            i4 = np.nan_to_num(np.asarray(sample["image"])[3], nan=0.0)
            if np.any(target & valid): self.fire_indices.append(index)
            if np.any((i4 > 310) & ~target & valid): self.hot_indices.append(index)
            if np.any((i4 <= 310) & valid): self.background_indices.append(index)
        if not self.fire_indices or not self.hot_indices or not self.background_indices:
            raise ValueError("one of fire / thermal-negative / background sampling pools is empty")

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        bucket = item % 4
        # 0,1 = fire (50%); 2 = warm non-fire (25%); 3 = background (25%).
        pool = self.fire_indices if bucket < 2 else self.hot_indices if bucket == 2 else self.background_indices
        sample = self.source.load(random.choice(pool))
        raw_image = np.asarray(sample["image"], dtype=np.float32)
        valid, target = np.asarray(sample["valid_mask"], dtype=bool), np.asarray(sample["target"]) == 1
        i4 = np.nan_to_num(raw_image[3], nan=0.0)
        mask = (target & valid) if bucket < 2 else ((i4 > 310) & ~target & valid) if bucket == 2 else ((i4 <= 310) & valid)
        image = normalize_residual_features(build_residual_features(sample), self.mean, self.std)
        image, target, valid = crop(image, target, valid, choose_coordinate(mask), self.patch_size)
        if random.random() < .5: image, target, valid = image[:, :, ::-1].copy(), target[:, ::-1].copy(), valid[:, ::-1].copy()
        if random.random() < .5: image, target, valid = image[:, ::-1, :].copy(), target[::-1, :].copy(), valid[::-1, :].copy()
        turns = random.randrange(4)
        if turns: image, target, valid = np.rot90(image, turns, axes=(1, 2)).copy(), np.rot90(target, turns).copy(), np.rot90(valid, turns).copy()
        return {"image": torch.from_numpy(image), "target": torch.from_numpy(target.astype(np.float32))[None], "valid": torch.from_numpy(valid)[None]}


class FullChipDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, source: FireChipDataset, ids: list[str], stats: dict[str, Any]) -> None:
        by_id = {record.chip_id: index for index, record in enumerate(source.records)}
        self.source, self.indices = source, [by_id[chip_id] for chip_id in ids]
        self.mean, self.std = np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.source.load(self.indices[item])
        image = normalize_residual_features(build_residual_features(sample), self.mean, self.std)
        target = (np.asarray(sample["target"]) == 1).astype(np.float32)
        valid = np.asarray(sample["valid_mask"], dtype=bool)
        return {"image": torch.from_numpy(image), "target": torch.from_numpy(target)[None], "valid": torch.from_numpy(valid)[None]}


def loss_function(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, positive_weight: float = 5.) -> torch.Tensor:
    mask = valid.float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * (1 + (positive_weight - 1) * target) * mask).sum() / mask.sum().clamp_min(1)
    probability = torch.sigmoid(logits) * mask; truth = target * mask
    true_positive = (probability * truth).sum()
    false_positive = (probability * (1 - truth)).sum()
    false_negative = ((1 - probability) * truth).sum()
    tversky_loss = 1 - (true_positive + 1) / (true_positive + .3 * false_positive + .7 * false_negative + 1)
    return .5 * bce + .5 * tversky_loss.pow(1.33)


def metrics(probability: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float]:
    probability, target = probability[valid], target[valid].astype(bool)
    best = (0., .5, 0., 0.)
    for threshold in np.arange(.05, .991, .005):
        prediction = probability >= threshold
        tp = int(np.count_nonzero(prediction & target)); fp = int(np.count_nonzero(prediction & ~target)); fn = int(np.count_nonzero(~prediction & target))
        f1 = 0. if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        if f1 > best[0]:
            iou = 0. if tp + fp + fn == 0 else tp / (tp + fp + fn)
            precision = 0. if tp + fp == 0 else tp / (tp + fp); recall = 0. if tp + fn == 0 else tp / (tp + fn)
            best = (f1, float(threshold), precision, recall)
    return best


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float, float, float]:
    model.eval(); losses, probabilities, targets, masks = [], [], [], []
    for batch in loader:
        image, target, valid = batch["image"].to(device), batch["target"].to(device), batch["valid"].to(device)
        with torch.autocast(device_type="cuda"):
            logits = model(image); loss = loss_function(logits, target, valid)
        losses.append(float(loss.cpu())); probabilities.append(torch.sigmoid(logits).cpu().numpy()); targets.append(target.cpu().numpy()); masks.append(valid.cpu().numpy().astype(bool))
    f1, threshold, precision, recall = metrics(np.concatenate(probabilities), np.concatenate(targets), np.concatenate(masks))
    return float(np.mean(losses)), f1, threshold, precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True); parser.add_argument("--prepared-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=256); parser.add_argument("--patches-per-epoch", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--accumulation", type=int, default=2); parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=70); parser.add_argument("--patience", type=int, default=12); parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable; activate the NVIDIA PyTorch environment")
    set_seed(args.seed)
    prepared = Path(args.prepared_dir); stats = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    with (prepared / "manifest.csv").open(encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
    train_ids = [row["chip_id"] for row in rows if row["split"] == "train"]; validation_ids = [row["chip_id"] for row in rows if row["split"] == "validation"]
    source = FireChipDataset(args.data_dir, "af")
    train_loader = DataLoader(DirectPatchDataset(source, train_ids, stats, args.patch_size, args.patches_per_epoch), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(FullChipDataset(source, validation_ids, stats), batch_size=1, shuffle=False, pin_memory=True)
    device = torch.device("cuda"); model = AfDirectResUNet(len(stats["feature_names"]), args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs); scaler = torch.amp.GradScaler("cuda")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); best, stalled, history = -1., 0, []
    print(f"device={torch.cuda.get_device_name(0)} parameters={sum(p.numel() for p in model.parameters()):,}; patches/epoch={args.patches_per_epoch}")
    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []; optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            image, target, valid = batch["image"].to(device), batch["target"].to(device), batch["valid"].to(device)
            with torch.autocast(device_type="cuda"): loss = loss_function(model(image), target, valid) / args.accumulation
            scaler.scale(loss).backward()
            if step % args.accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()) * args.accumulation)
        scheduler.step(); validation_loss, f1, threshold, precision, recall = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss, "validation_f1": f1, "threshold": threshold, "precision": precision, "recall": recall}; history.append(row)
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_loss={validation_loss:.4f} val_f1={f1:.4f} threshold={threshold:.3f} precision={precision:.3f} recall={recall:.3f}")
        if f1 > best:
            best, stalled = f1, 0
            torch.save({"model": model.state_dict(), "model_type": "af_direct_resunet", "feature_names": stats["feature_names"], "normalization": stats, "base_channels": args.base_channels, "threshold": threshold, "validation_f1": f1, "precision": precision, "recall": recall, "epoch": epoch}, output / "best.pt")
        else:
            stalled += 1
            if stalled >= args.patience: print(f"early stopping after {args.patience} epochs without F1 improvement"); break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best validation F1={best:.4f}; checkpoint={output / 'best.pt'}")


if __name__ == "__main__": main()
