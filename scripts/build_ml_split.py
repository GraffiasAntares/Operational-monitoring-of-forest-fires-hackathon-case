"""Create a deterministic, leakage-aware AF or BS train/validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data import FireChipDataset, grouped_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/af and train/bs")
    parser.add_argument("--kind", required=True, choices=("af", "bs"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = FireChipDataset(args.data_dir, args.kind)
    split = grouped_split(dataset.records, args.validation_fraction, args.seed)
    payload = {"kind": args.kind, "seed": args.seed, "validation_fraction": args.validation_fraction, **split}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}: train={len(split['train'])}, validation={len(split['validation'])}")


if __name__ == "__main__":
    main()
