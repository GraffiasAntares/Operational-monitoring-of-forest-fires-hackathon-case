"""BS-only data access: paired Sentinel imagery, auxiliary layers and severity masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml._shared import IGNORE_INDEX, read_aligned_rasters, read_metadata
from ml.bs.preprocessing import BurnSeverityPreprocessor


class BurnSeverityDataset:
    """Load the Burn Severity task without any Active Fire dependencies."""

    kind = "bs"
    input_layers = ("s2_pre", "s2_post", "s1_pre", "s1_post", "aux")

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "train" / "bs"
        self.records = read_metadata(self.root / "meta.csv", self.kind)
        self._assert_complete()

    def __len__(self) -> int:
        return len(self.records)

    def _paths(self, chip_id: str) -> dict[str, Path]:
        return {
            "s2_pre": self.root / "sentinel2_pre" / f"{chip_id}_Sentinel-2_pre.tif",
            "s2_post": self.root / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif",
            "s1_pre": self.root / "sentinel1_pre" / f"{chip_id}_Sentinel-1_pre.tif",
            "s1_post": self.root / "sentinel1_post" / f"{chip_id}_Sentinel-1_post.tif",
            "aux": self.root / "aux" / f"{chip_id}_AUX.tif",
            "target": self.root / "masks" / f"{chip_id}_MASK.tif",
        }

    def _assert_complete(self) -> None:
        missing = [str(path) for record in self.records for path in self._paths(record.chip_id).values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing BS layer; first missing path: " + missing[0])

    def load_layers(self, index: int) -> dict[str, np.ndarray]:
        """Load raw, aligned raster layers for fitting a preprocessor."""
        record = self.records[index]
        return read_aligned_rasters(self._paths(record.chip_id), record.chip_id)

    def load(self, index: int, preprocessor: BurnSeverityPreprocessor | None = None) -> dict[str, object]:
        record = self.records[index]
        arrays = self.load_layers(index)
        target = arrays.pop("target")[0].astype(np.uint8, copy=False)
        image = (
            preprocessor.transform(arrays)
            if preprocessor is not None
            else np.concatenate([arrays[name].astype(np.float32, copy=False) for name in self.input_layers])
        )
        # Cloudy BS pixels are evaluated by the competition, so retain them.
        valid_mask = target != IGNORE_INDEX
        return {"chip_id": record.chip_id, "image": image, "target": target, "valid_mask": valid_mask, "metadata": record.metadata}


class BurnSeverityInferenceDataset:
    """Load BS chips without masks for standalone prediction on test data."""

    kind = "bs"
    input_layers = BurnSeverityDataset.input_layers

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        nested_test_root = data_dir / "test" / "bs"
        global_test_meta = data_dir / "test" / "meta.csv"
        candidates = (data_dir / "bs", data_dir)
        if (nested_test_root / "meta.csv").is_file():
            self.layers_root = nested_test_root
            metadata_path = nested_test_root / "meta.csv"
        elif global_test_meta.is_file():
            # The official combined test archive has one test/meta.csv for AF
            # and BS, while rasters still live under test/bs/.
            self.layers_root = nested_test_root
            metadata_path = global_test_meta
        else:
            metadata_root = next((candidate for candidate in candidates if (candidate / "meta.csv").is_file()), None)
            self.layers_root = metadata_root / "bs" if metadata_root is not None and (metadata_root / "bs").is_dir() else metadata_root
            metadata_path = metadata_root / "meta.csv" if metadata_root is not None else None
        if self.layers_root is None or metadata_path is None:
            raise FileNotFoundError("could not find BS metadata under data-dir/test, data-dir/bs or data-dir")
        self.records = read_metadata(metadata_path, self.kind)
        if not self.records:
            raise ValueError(f"no BS rows in {metadata_path}")
        self._assert_complete()

    def __len__(self) -> int:
        return len(self.records)

    def _paths(self, chip_id: str) -> dict[str, Path]:
        return {
            "s2_pre": self.layers_root / "sentinel2_pre" / f"{chip_id}_Sentinel-2_pre.tif",
            "s2_post": self.layers_root / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif",
            "s1_pre": self.layers_root / "sentinel1_pre" / f"{chip_id}_Sentinel-1_pre.tif",
            "s1_post": self.layers_root / "sentinel1_post" / f"{chip_id}_Sentinel-1_post.tif",
            "aux": self.layers_root / "aux" / f"{chip_id}_AUX.tif",
        }

    def _assert_complete(self) -> None:
        missing = [str(path) for record in self.records for path in self._paths(record.chip_id).values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing BS inference layer; first missing path: " + missing[0])

    def load_layers(self, index: int) -> dict[str, np.ndarray]:
        record = self.records[index]
        return read_aligned_rasters(self._paths(record.chip_id), record.chip_id)

    def load(self, index: int, preprocessor: BurnSeverityPreprocessor) -> dict[str, object]:
        record = self.records[index]
        return {"chip_id": record.chip_id, "image": preprocessor.transform(self.load_layers(index)), "metadata": record.metadata}
