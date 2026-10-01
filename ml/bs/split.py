"""Leakage-aware validation splits for Burn Severity chips."""
from __future__ import annotations

from ml._shared import ChipRecord, grouped_split as _grouped_split


def split_group_id(record: ChipRecord) -> str:
    if record.kind != "bs":
        raise ValueError("BS splitting accepts only BS records")
    event_id = record.metadata.get("fire_event_id", "").strip()
    if event_id and event_id.lower() != "nan":
        return f"event:{event_id}"
    raise ValueError(f"BS chip {record.chip_id} has no fire_event_id")


def grouped_split(records: list[ChipRecord], validation_fraction: float = 0.2, seed: int = 42) -> dict[str, list[str]]:
    return _grouped_split(records, split_group_id, validation_fraction, seed)
