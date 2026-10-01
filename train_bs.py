"""Train the compact BS U-Net with a leakage-safe event split."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.bs import BurnSeverityDataset, BurnSeverityPreprocessor, FeatureStatistics
from ml.bs.baseline import metrics_dict
from ml.bs.losses import WeightedCrossEntropyDiceLoss
from ml.bs.model import CompactUNet
from ml.bs.training import TorchBurnSeverityDataset, class_weights, feature_indices, prediction_metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_split(path: str | Path) -> tuple[set[str], set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "bs":
        raise ValueError("train_bs requires a BS split")
    return set(payload["train"]), set(payload["validation"])


def _learning_rate(initial: float, minimum: float, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Linear warm-up followed by a cosine decay, evaluated once per epoch."""
    if not 0 < minimum <= initial:
        raise ValueError("min learning rate must be positive and no greater than learning rate")
    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup epochs must be non-negative and smaller than total epochs")
    if warmup_epochs and epoch <= warmup_epochs:
        return initial * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return minimum + (initial - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def _validate(model: CompactUNet, loader: DataLoader, criterion: WeightedCrossEntropyDiceLoss, device: torch.device, epoch: int, log_every: int) -> tuple[float, object]:
    model.eval()
    losses: list[float] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for step, (image, target) in enumerate(loader, start=1):
        image, target = image.to(device), target.to(device)
        logits = model(image)
        loss = float(criterion(logits, target).item())
        losses.append(loss)
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(target.cpu())
        if log_every and (step % log_every == 0 or step == len(loader)):
            print(json.dumps({"phase": "validation", "epoch": epoch, "batch": step, "batches": len(loader), "validation_loss": round(loss, 6)}), flush=True)
    return float(np.mean(losses)), prediction_metrics(predictions, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--preprocessing", required=True, help="Fitted .npz statistics from scripts/fit_bs_preprocessing.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-mode", choices=("s2", "s2_aux", "s2_s1_aux", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save a resumable checkpoint every N epochs; set 0 to disable")
    parser.add_argument("--checkpoint-every-steps", type=int, default=500, help="Save a resumable checkpoint every N batch steps; set 0 to disable")
    parser.add_argument("--log-every", type=int, default=10, help="Print training progress every N batch steps")
    parser.add_argument("--validation-every-steps", type=int, default=1, help="Evaluate and print one validation batch every N training steps; set 0 to disable")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--dice-weight", type=float, default=0.60)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    args = parser.parse_args()
    _set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    train_ids, validation_ids = _read_split(args.split)
    dataset = BurnSeverityDataset(args.data_dir)
    train_indices = [i for i, record in enumerate(dataset.records) if record.chip_id in train_ids]
    validation_indices = [i for i, record in enumerate(dataset.records) if record.chip_id in validation_ids]
    preprocessor = BurnSeverityPreprocessor(FeatureStatistics.load(args.preprocessing))
    train_data = TorchBurnSeverityDataset(dataset, train_indices, preprocessor, args.feature_mode, augment=True, seed=args.seed)
    validation_data = TorchBurnSeverityDataset(dataset, validation_indices, preprocessor, args.feature_mode)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = CompactUNet(len(feature_indices(args.feature_mode)), args.base_channels, args.dropout).to(device)
    weights = class_weights(dataset, train_indices).to(device)
    criterion = WeightedCrossEntropyDiceLoss(weights, dice_weight=args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, object]] = []
    best_score, stale_epochs = -float("inf"), 0
    config = vars(args) | {"device": str(device), "input_channels": len(feature_indices(args.feature_mode)), "class_weights": weights.cpu().tolist()}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    validation_probe = iter(validation_loader)

    global_step = 0

    def save_checkpoint(path: Path, epoch: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
                "metrics": metrics,
            },
            path,
        )

    for epoch in range(1, args.epochs + 1):
        epoch_learning_rate = _learning_rate(
            args.learning_rate, args.min_learning_rate, epoch, args.epochs, args.warmup_epochs
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = epoch_learning_rate
        model.train()
        train_data.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        batch_losses: list[float] = []
        for step, (image, target) in enumerate(train_loader, start=1):
            global_step += 1
            image, target = image.to(device), target.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(image), target) / args.accumulate
            scaler.scale(loss).backward()
            if step % args.accumulate == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_losses.append(float(loss.item() * args.accumulate))
            if args.checkpoint_every_steps and global_step % args.checkpoint_every_steps == 0:
                save_checkpoint(checkpoint_dir / f"step_{global_step:06d}.pt", epoch, {})
            if args.log_every and (step % args.log_every == 0 or step == len(train_loader)):
                progress = {
                    "epoch": epoch,
                    "batch": step,
                    "batches": len(train_loader),
                    "global_step": global_step,
                    "train_loss": round(batch_losses[-1], 6),
                    "learning_rate": epoch_learning_rate,
                }
                if device.type == "cuda":
                    progress["vram_mib"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
                print(json.dumps(progress), flush=True)
            if args.validation_every_steps and global_step % args.validation_every_steps == 0:
                try:
                    validation_image, validation_target = next(validation_probe)
                except StopIteration:
                    validation_probe = iter(validation_loader)
                    validation_image, validation_target = next(validation_probe)
                was_training = model.training
                model.eval()
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    validation_logits = model(validation_image.to(device))
                    validation_loss = float(criterion(validation_logits, validation_target.to(device)).item())
                model.train(was_training)
                print(json.dumps({"phase": "validation", "scope": "one_batch", "epoch": epoch, "train_batch": step, "global_step": global_step, "validation_loss": round(validation_loss, 6)}), flush=True)
        validation_loss, metrics = _validate(model, validation_loader, criterion, device, epoch, args.log_every)
        record = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "train_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
            **metrics_dict(metrics),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics.local_score > best_score:
            best_score, stale_epochs = metrics.local_score, 0
            save_checkpoint(output / "best.pt", epoch, metrics_dict(metrics))
        else:
            stale_epochs += 1
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", epoch, metrics_dict(metrics))
        if stale_epochs >= args.patience:
            break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_checkpoint(output / "last.pt", history[-1]["epoch"], {key: float(history[-1][key]) for key in metrics_dict(metrics)})
    print(f"best local BS score={best_score:.5f}; checkpoint={output / 'best.pt'}")


if __name__ == "__main__":
    main()
