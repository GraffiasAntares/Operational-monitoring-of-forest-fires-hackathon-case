"""Prepare strict-split 26-channel data metadata for direct AF segmentation."""
from __future__ import annotations

import argparse
import json
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
    output = Path(args.output_dir)
    normalization = output / "normalization.json"
    payload = json.loads(normalization.read_text(encoding="utf-8"))
    payload["version"] = "af-direct-v1-resunet"
    payload["training"] = {
        "target": "direct_fire_mask",
        "patch_size": 256,
        "sampling": {"fire": .5, "thermal_hard_negative": .25, "background": .25},
        "validation": "full_chip_strict_spatiotemporal",
    }
    normalization.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prepared AF direct data: {report['chip_counts']}; corrections diagnostic={report['correction_pixels']}")


if __name__ == "__main__":
    main()
