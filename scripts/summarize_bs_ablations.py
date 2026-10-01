"""Build a comparable CSV/Markdown table from BS baseline and training metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROWS = (
    ("A", "dNBR + adaptive land-cover thresholds"),
    ("B", "U-Net: Sentinel-2 only"),
    ("C", "U-Net: Sentinel-2 + aux"),
    ("D", "U-Net: Sentinel-2 + Sentinel-1 + aux"),
    ("E", "U-Net: D + NBR/dNBR/RdNBR, SCL and change features"),
)
METRICS = ("burn_iou", "iou_severity_1", "iou_severity_2", "iou_severity_3", "miou_severity", "local_score")


def _metrics(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = max(payload, key=lambda row: row["local_score"])
    return {name: float(payload[name]) for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="A metrics.json from scripts/run_bs_baseline.py")
    parser.add_argument("--run", action="append", default=[], metavar="LETTER=PATH", help="Best history.json or metrics.json for B, C, D or E")
    parser.add_argument("--output", required=True, help="CSV destination")
    args = parser.parse_args()
    results = {"A": _metrics(args.baseline)}
    for item in args.run:
        letter, separator, path = item.partition("=")
        if not separator or letter not in {"B", "C", "D", "E"}:
            raise ValueError("--run must use B=path, C=path, D=path or E=path")
        results[letter] = _metrics(path)
    rows = []
    for letter, name in ROWS:
        row = {"ablation": letter, "configuration": name, "status": "complete" if letter in results else "pending"}
        row.update(results.get(letter, {metric: "" for metric in METRICS}))
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ablation", "configuration", "status", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    markdown = output.with_suffix(".md")
    headings = ("Ablation", "Configuration", "Status", "IoU burn", "IoU 1", "IoU 2", "IoU 3", "mIoU", "Local score")
    lines = ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    for row in rows:
        values = [row["ablation"], row["configuration"], row["status"], *(f"{row[key]:.5f}" if isinstance(row[key], float) else "-" for key in METRICS)]
        lines.append("| " + " | ".join(values) + " |")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} and {markdown}")


if __name__ == "__main__":
    main()
