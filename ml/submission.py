"""Strict CSV/RLE helpers for the competition submission format."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


HEADER = ("chip_id", "class_id", "rle")


def encode_rle(mask: np.ndarray, class_id: int) -> str:
    """Encode a single class in row-major, one-based ``start length`` RLE."""
    flat = np.asarray(mask).reshape(-1, order="C") == class_id
    positions = np.flatnonzero(flat)
    if not len(positions):
        return ""
    starts = positions[np.r_[True, np.diff(positions) > 1]]
    ends = positions[np.r_[np.diff(positions) > 1, True]]
    values: list[str] = []
    for start, end in zip(starts, ends, strict=True):
        values.extend((str(int(start) + 1), str(int(end - start) + 1)))
    return " ".join(values)


def decode_rle(rle: str, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape[0] * shape[1], dtype=bool)
    if not rle:
        return result.reshape(shape)
    values = rle.split()
    if len(values) % 2:
        raise ValueError("RLE must contain start-length pairs")
    previous_end: int | None = None
    for start_text, length_text in zip(values[::2], values[1::2], strict=True):
        start, length = int(start_text), int(length_text)
        if start < 1 or length < 1 or (previous_end is not None and start <= previous_end + 1):
            raise ValueError("RLE runs must be positive, ordered and non-touching")
        end = start - 1 + length
        if end > result.size:
            raise ValueError("RLE run exceeds chip dimensions")
        result[start - 1:end] = True
        previous_end = end
    return result.reshape(shape)


def read_template(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HEADER:
            raise ValueError(f"expected header {HEADER}, got {reader.fieldnames}")
        rows = list(reader)
    pairs = [(row["chip_id"], row["class_id"]) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("template contains duplicate chip_id/class_id pairs")
    return rows


def write_submission(template_path: str | Path, predictions: Mapping[str, np.ndarray], output_path: str | Path) -> None:
    rows = read_template(template_path)
    expected_ids = {row["chip_id"] for row in rows}
    missing = expected_ids - predictions.keys()
    extra = predictions.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"prediction IDs differ from template; missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")
    with Path(output_path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            class_id = int(row["class_id"])
            writer.writerow({"chip_id": row["chip_id"], "class_id": class_id, "rle": encode_rle(predictions[row["chip_id"]], class_id)})


def validate_submission(submission_path: str | Path, template_path: str | Path, af_shape: tuple[int, int] = (256, 256), bs_shape: tuple[int, int] = (512, 512)) -> None:
    template = read_template(template_path)
    submitted = read_template(submission_path)
    expected_pairs = [(row["chip_id"], row["class_id"]) for row in template]
    actual_pairs = [(row["chip_id"], row["class_id"]) for row in submitted]
    if expected_pairs != actual_pairs:
        raise ValueError("submission rows must match the template chip_id/class_id pairs and order exactly")
    masks: dict[str, np.ndarray] = {}
    for row in submitted:
        is_af = row["chip_id"].lower().startswith("af_")
        shape = af_shape if is_af else bs_shape
        class_id = int(row["class_id"])
        if class_id not in ((1,) if is_af else (1, 2, 3)):
            raise ValueError(f"invalid class {class_id} for {row['chip_id']}")
        decoded = decode_rle(row["rle"], shape)
        aggregate = masks.setdefault(row["chip_id"], np.zeros(shape, dtype=np.uint8))
        if np.any(aggregate[decoded]):
            raise ValueError(f"class masks overlap for {row['chip_id']}")
        aggregate[decoded] = class_id
