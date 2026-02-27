from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from itp_fabadII.utils.data_utils import (
    interpolate_series,
    load_comb_vector_lazy,
    safe_standardize,
)
from itp_fabadII.utils.scalers import load_scalers
from itp_fabadII.utils.tensor_ops import pad_to_shape


class CombDataLoader(DataLoader):
    """
    DataLoader for processed comb datasets stored as HDF5 files.

    This loader is comb-native:
    - reads `vector_index.csv`
    - filters by segment (`hatching` by default)
    - loads vectors without any `ref1/ref2` semantics
    """

    def __init__(
        self,
        dataset_path,
        metadata_csv: str = "vector_index.csv",
        fraction: float = 1.0,
        segment: str = "hatching",
        features_info: bool = False,
        neighbours_info: bool = False,
        series_info: bool = True,
        series_variants: tuple[str, ...] | list[str] | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        random_state: int = 42,
        max_L: int = 600,
        max_T: int = 1900,
    ):
        dataset = CombH5Dataset(
            dataset_path=dataset_path,
            metadata_csv=metadata_csv,
            fraction=fraction,
            segment=segment,
            features_info=features_info,
            neighbours_info=neighbours_info,
            series_info=series_info,
            series_variants=series_variants,
            random_state=random_state,
            max_L=max_L,
            max_T=max_T,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class CombH5Dataset(Dataset):
    """
    Lazy dataset for comb H5 vectors.

    Expected H5 layout per sample:
    - `/<segment>/<vector_id>/features`
    - `/<segment>/<vector_id>/meltpool_seq`
    - `/<segment>/<vector_id>/neighbours/{abs_rel_times, rel_distances, n_neighbours}`
    """

    features_scaler = None
    series_scaler = None
    scalers_dir: Path | None = None
    _SERIES_VARIANTS = {"raw", "selfstd", "scaled"}

    def __init__(
        self,
        dataset_path: Path,
        metadata_csv: Path = "vector_index.csv",
        fraction: float = 1.0,
        segment: str = "hatching",
        features_info: bool = False,
        neighbours_info: bool = False,
        series_info: bool = True,
        series_variants: tuple[str, ...] | list[str] | None = None,
        random_state: int = 42,
        max_L: int = 600,
        max_T: int = 1900,
    ):
        """
        Build a comb dataset from one split directory.

        Args:
            dataset_path: Split root containing H5 files + scaler files.
            metadata_csv: CSV index relative to `dataset_path` (or absolute path).
            fraction: Fraction of rows sampled from selected segment.
            segment: Segment to use (`hatching` or `edges`).
            features_info: Load and scale feature vectors.
            neighbours_info: Load neighbour tensors.
            series_info: Load and scale meltpool sequence.
            series_variants: Series outputs to return when `series_info=True`.
                Allowed values: `raw`, `selfstd`, `scaled`.
                Defaults to `("scaled",)`.
            random_state: Seed for deterministic row sampling.
            max_L: Target sequence length for meltpool interpolation and neighbour padding.
            max_T: Target neighbour width for 2D neighbour tensors.
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_csv = Path(metadata_csv)
        self.metadata_path = (
            self.metadata_csv if self.metadata_csv.is_absolute() else self.dataset_path / self.metadata_csv
        )

        self.fraction = fraction
        if self.fraction is not None and not (0 < self.fraction <= 1):
            raise ValueError(f"'fraction' must be between 0 and 1, got {self.fraction}")

        self.segment = str(segment)
        self.return_features = features_info
        self.return_neighbours = neighbours_info
        self.return_series = series_info
        self.series_variants = self._resolve_series_variants(series_info, series_variants)
        self.random_state = random_state
        self.max_L = int(max_L)
        self.max_T = int(max_T)

        # Build sample references from index and load required scalers once.
        self.vector_refs = self._build_vector_references()
        self._load_scalers()

    @classmethod
    def _resolve_series_variants(
        cls,
        return_series: bool,
        variants: tuple[str, ...] | list[str] | None,
    ) -> tuple[str, ...]:
        """
        Validate and normalize requested series output variants.
        """
        if not return_series:
            return tuple()

        normalized = ("scaled",) if variants is None else tuple(str(v).strip().lower() for v in variants)
        if len(normalized) == 0:
            raise ValueError("When series_info=True, `series_variants` must contain at least one value.")

        invalid = [v for v in normalized if v not in cls._SERIES_VARIANTS]
        if invalid:
            raise ValueError(
                f"Unsupported series_variants={invalid}. "
                f"Allowed: {sorted(cls._SERIES_VARIANTS)}"
            )
        return normalized

    def _build_vector_references(self) -> list[tuple[Path, str, str]]:
        """
        Build `(h5_path, segment_key, vector_key)` references from metadata CSV.
        """
        # 1) Load and validate metadata contract.
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)
        required_cols = {"segment", "vector", "file"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                f"Metadata CSV missing required columns {sorted(missing)}: {self.metadata_path}"
            )

        # 2) Keep only requested segment and optional random fraction.
        df = df[df["segment"] == self.segment].copy()
        if df.empty:
            raise ValueError(f"No rows found for segment='{self.segment}' in {self.metadata_path}.")

        if self.fraction is not None and self.fraction < 1.0:
            df = df.sample(frac=self.fraction, random_state=self.random_state)

        # 3) Resolve H5 paths from relative `file` column.
        refs: list[tuple[Path, str, str]] = []
        for row in df.itertuples(index=False):
            file_value = getattr(row, "file")
            vector_value = getattr(row, "vector")

            h5_path = Path(file_value)
            if not h5_path.is_absolute():
                h5_path = self.dataset_path / h5_path

            refs.append((h5_path, str(int(vector_value))))

        if not refs:
            raise ValueError(
                f"No vector references built from {self.metadata_path} for segment '{self.segment}'."
            )
        return refs

    def _load_scalers(self) -> None:
        """
        Load required comb scalers from dataset root.

        We only load the scalers required by active flags:
        - `feature_scaler.joblib` when `features_info=True`
        - `series_scaler.joblib` when `series_info=True`
        """
        scalers_dir = self.dataset_path.parent

        # 1) Reset class cache when switching to another split path.
        if CombH5Dataset.scalers_dir != scalers_dir:
            CombH5Dataset.features_scaler = None
            CombH5Dataset.series_scaler = None
            CombH5Dataset.scalers_dir = scalers_dir

        # 2) Build required scaler map and short-circuit if none requested.
        required_mapping: dict[str, str] = {}
        if self.return_features:
            required_mapping["feature_scaler.joblib"] = "features_scaler"
        if self.return_series and "scaled" in self.series_variants:
            required_mapping["series_scaler.joblib"] = "series_scaler"
        if not required_mapping:
            return

        # 3) Load scalers and assign to class cache.
        loaded = load_scalers(scalers_dir, file_mapping=required_mapping)
        CombH5Dataset.features_scaler = loaded.get("features_scaler")
        CombH5Dataset.series_scaler = loaded.get("series_scaler")

    def __len__(self) -> int:
        """Return number of indexed vectors for selected segment."""
        return len(self.vector_refs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Read one comb sample and return tensors ready for model input.

        Series outputs are returned according to `series_variants`:
        - `meltpool_seq_raw`
        - `meltpool_seq_selfstd`
        - `meltpool_seq_scaled`
        """
        # 1) Read selected arrays lazily from H5.
        h5_path, vector_key = self.vector_refs[idx]
        
        sample = load_comb_vector_lazy(
            path=h5_path,
            segment_key=self.segment,
            vector_key=vector_key,
            features=self.return_features,
            neighbours_info=self.return_neighbours,
            series_info=self.return_series,
        )

        out: dict[str, torch.Tensor] = {}

        # 2) Pad neighbour tensors to fixed shape expected by training code.
        if self.return_neighbours:
            out["neighs_abs_rel_times"] = pad_to_shape(
                torch.as_tensor(sample["neighs_abs_rel_times"], dtype=torch.float32),
                max_L=self.max_L,
                max_T=self.max_T,
            )
            out["neighs_rel_distances"] = pad_to_shape(
                torch.as_tensor(sample["neighs_rel_distances"], dtype=torch.float32),
                max_L=self.max_L,
                max_T=self.max_T,
            )
            out["neighs_n_neighbours"] = pad_to_shape(
                torch.as_tensor(sample["neighs_n_neighbours"], dtype=torch.int32),
                max_L=self.max_L,
            )

        # 3) Interpolate and emit requested meltpool variants (comb has no ref split).
        if self.return_series:
            raw_melt = np.asarray(sample["meltpool_seq"], dtype=np.float32)
            # SIN INTERPOLAR!!!
            #interpolated_raw = interpolate_series(raw_melt, self.max_L).astype(np.float32)

            if "raw" in self.series_variants:
                out["meltpool_seq_raw"] = pad_to_shape(
                    torch.as_tensor(raw_melt, dtype=torch.float32),
                    max_L=self.max_L)

            if "selfstd" in self.series_variants:
                selfstd = safe_standardize(raw_melt)
                out["meltpool_seq_selfstd"] = pad_to_shape(
                    torch.as_tensor(selfstd, dtype=torch.float32),
                    max_L=self.max_L)

            if "scaled" in self.series_variants:
                if CombH5Dataset.series_scaler is None:
                    self._load_scalers()

                interpolated_melt_2d = raw_melt.reshape(-1, 1)
                scaled_melt_2d = CombH5Dataset.series_scaler.transform(interpolated_melt_2d)
                scaled_melt = scaled_melt_2d.astype(np.float32).reshape(-1)
                out["meltpool_seq_scaled"] = pad_to_shape(
                    torch.as_tensor(scaled_melt, dtype=torch.float32),
                    max_L=self.max_L)

        # 4) Scale dense feature vector.
        if self.return_features:
            if CombH5Dataset.features_scaler is None:
                self._load_scalers()

            raw_feats = np.asarray(sample["features"], dtype=np.float32)
            scaled_feats_2d = CombH5Dataset.features_scaler.transform(raw_feats.reshape(1, -1))
            feats = scaled_feats_2d.astype(np.float32).reshape(-1)
            out["features"] = torch.as_tensor(feats, dtype=torch.float32)

        return out
