"""AF-only data access: VIIRS, auxiliary layers and a binary fire mask."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml._shared import IGNORE_INDEX, read_aligned_rasters, read_metadata


class ActiveFireDataset:
    """Load the Active Fire task without any Burn Severity dependencies."""

    kind = "af"
    input_layers = ("viirs", "aux")

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "train" / "af"
        self.records = read_metadata(self.root / "meta.csv", self.kind)
        self._assert_complete()

    def __len__(self) -> int:
        return len(self.records)

    def _paths(self, chip_id: str) -> dict[str, Path]:
        return {
            "viirs": self.root / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif",
            "aux": self.root / "aux" / f"{chip_id}_AUX.tif",
            "target": self.root / "masks" / f"{chip_id}_MASK.tif",
        }

    def _assert_complete(self) -> None:
        missing = [str(path) for record in self.records for path in self._paths(record.chip_id).values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing AF layer; first missing path: " + missing[0])

    def load(self, index: int) -> dict[str, object]:
        record = self.records[index]
        arrays = read_aligned_rasters(self._paths(record.chip_id), record.chip_id)
        target = arrays.pop("target")[0].astype(np.uint8, copy=False)
        image = np.concatenate([arrays[name].astype(np.float32, copy=False) for name in self.input_layers])
        # VIIRS's final channel is the supplied pixel-validity layer.
        valid_mask = (target != IGNORE_INDEX) & (arrays["viirs"][-1] > 0)
        return {"chip_id": record.chip_id, "image": image, "target": target, "valid_mask": valid_mask, "metadata": record.metadata}
