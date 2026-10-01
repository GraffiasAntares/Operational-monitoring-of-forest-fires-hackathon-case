"""Fit leakage-safe Burn Severity preprocessing statistics and save them as NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.bs import BurnSeverityDataset, BurnSeverityPreprocessor, grouped_split


def _train_ids(dataset: BurnSeverityDataset, split_path: str | None, validation_fraction: float, seed: int) -> list[str]:
    if split_path is None:
        return grouped_split(dataset.records, validation_fraction=validation_fraction, seed=seed)["train"]
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if payload.get("kind") not in (None, "bs"):
        raise ValueError("split file must describe the BS task")
    train_ids = payload.get("train")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("split file must contain a non-empty train list")
    known_ids = {record.chip_id for record in dataset.records}
    unknown_ids = set(train_ids).difference(known_ids)
    if unknown_ids:
        raise ValueError(f"split contains chip absent from data: {sorted(unknown_ids)[0]}")
    return train_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/bs")
    parser.add_argument("--output", required=True, help="Destination .npz statistics file")
    parser.add_argument("--split", help="Existing BS split JSON; its train portion is used exactly")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Used only without --split")
    parser.add_argument("--seed", type=int, default=42, help="Used only without --split")
    args = parser.parse_args()

    dataset = BurnSeverityDataset(args.data_dir)
    train_ids = set(_train_ids(dataset, args.split, args.validation_fraction, args.seed))
    train_indices = [index for index, record in enumerate(dataset.records) if record.chip_id in train_ids]
    preprocessor = BurnSeverityPreprocessor().fit(dataset.load_layers(index) for index in train_indices)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.statistics.save(output)
    print(f"wrote {output}: train_chips={len(train_indices)}, continuous_features={len(preprocessor.continuous_feature_names)}")


if __name__ == "__main__":
    main()
