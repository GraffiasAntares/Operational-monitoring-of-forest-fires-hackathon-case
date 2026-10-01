"""Leakage-aware validation splits for Active Fire chips."""
from __future__ import annotations

from ml._shared import ChipRecord, grouped_split as _grouped_split


def split_group_id(record: ChipRecord) -> str:
    if record.kind != "af":
        raise ValueError("AF splitting accepts only AF records")
    satellite = record.metadata.get("satellite", "unknown")
    acquisition = record.metadata.get("acq_datetime", "unknown")
    return f"overpass:{satellite}:{acquisition}"


def grouped_split(records: list[ChipRecord], validation_fraction: float = 0.2, seed: int = 42) -> dict[str, list[str]]:
    return _grouped_split(records, split_group_id, validation_fraction, seed)
