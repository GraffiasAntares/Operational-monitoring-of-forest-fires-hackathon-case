"""Validate a candidate submission against the supplied sample_submission.csv."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.submission import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--template", required=True)
    args = parser.parse_args()
    validate_submission(args.submission, args.template)
    print("submission is valid")


if __name__ == "__main__":
    main()
