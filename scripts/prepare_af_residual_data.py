"""Prepare strict-split AF features for the thermal residual correction experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.af_residual import prepare_residual_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = prepare_residual_data(args.data_dir, args.output_dir)
    print(f"prepared AF residual data: {report['chip_counts']}; corrections={report['correction_pixels']}")


if __name__ == "__main__":
    main()
