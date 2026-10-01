"""Compatibility imports for the explicitly separated AF and BS packages.

New code should import ``ml.af`` or ``ml.bs`` directly. This module preserves
the original entry points for scripts created before the split.
"""
from __future__ import annotations

from pathlib import Path

from ml._shared import ChipRecord, IGNORE_INDEX, Kind
from ml.af.data import ActiveFireDataset
from ml.af.split import grouped_split as grouped_af_split, split_group_id as af_split_group_id
from ml.bs.data import BurnSeverityDataset
from ml.bs.split import grouped_split as grouped_bs_split, split_group_id as bs_split_group_id


def FireChipDataset(data_dir: str | Path, kind: Kind) -> ActiveFireDataset | BurnSeverityDataset:
    """Return the task-specific dataset; prefer its explicit class in new code."""
    if kind == "af":
        return ActiveFireDataset(data_dir)
    if kind == "bs":
        return BurnSeverityDataset(data_dir)
    raise ValueError(f"unsupported dataset kind: {kind}")


def split_group_id(record: ChipRecord) -> str:
    return af_split_group_id(record) if record.kind == "af" else bs_split_group_id(record)


def grouped_split(records: list[ChipRecord], validation_fraction: float = 0.2, seed: int = 42) -> dict[str, list[str]]:
    if not records:
        raise ValueError("at least one record is required")
    kinds = {record.kind for record in records}
    if len(kinds) != 1:
        raise ValueError("AF and BS records must be split separately")
    return grouped_af_split(records, validation_fraction, seed) if records[0].kind == "af" else grouped_bs_split(records, validation_fraction, seed)
