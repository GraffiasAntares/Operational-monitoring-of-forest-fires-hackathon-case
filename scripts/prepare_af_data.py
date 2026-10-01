"""Prepare AF train/validation manifest and train-only feature normalization."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.af_prepare import prepare_af_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/af")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--strict", action="store_true", help="Use the spatial-and-seasonal holdout matching the competition shift")
    args = parser.parse_args()
    report = prepare_af_data(args.data_dir, args.output_dir, args.validation_fraction, args.seed, args.strict)
    print(f"prepared AF data: train={report['train_chip_count']}, validation={report['validation_chip_count']}")
    print(f"positive fraction: train={report['positive_fraction']['train']:.6%}, validation={report['positive_fraction']['validation']:.6%}")
    if args.strict:
        print(f"excluded to prevent leakage: {report['excluded_chip_count']}")


if __name__ == "__main__":
    main()
