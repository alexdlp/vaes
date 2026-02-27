from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from itp_fabadII.utils.data_utils import (
    build_vector_references,
    interpolate_series,
    load_cube_vector_lazy,
    safe_standardize,
)
from itp_fabadII.utils.scalers import load_scalers
from itp_fabadII.utils.tensor_ops import pad_to_shape


class CubeDataLoader(DataLoader):
    """
    Custom DataLoader that directly subclasses torch.utils.data.DataLoader
    for seamless integration with Hydra configs and Fabric.

    Example:
        train_dataloader = CubeDataLoader(**cfg.data.train)
        val_dataloader = CubeDataLoader(**cfg.data.val)
    """
    def __init__(
        self,
        dataset_path,
        metadata_csv="vector_index.csv",
        fraction=1.0,
        features_info=False,
        neighbours_info=False,
        ref_info=True,
        series_variants=None,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        random_state=42,
    ):
        # Create dataset as usual
        dataset = CubeH5Dataset(
            dataset_path=dataset_path,
            metadata_csv=metadata_csv,
            fraction=fraction,
            features_info=features_info,
            neighbours_info=neighbours_info,
            ref_info=ref_info,
            series_variants=series_variants,
            random_state=random_state,
        )

        # Initialize parent DataLoader directly
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class CubeH5Dataset(Dataset):
    """
    Lazy PyTorch Dataset for cube-based HDF5 data.

    Each sample corresponds to one (cube, layer, vector, reference) entry derived
    from a metadata CSV. Data are loaded lazily from disk using HDF5, so memory
    usage remains low even for large datasets.

    The dataset supports selective loading of specific components:
    - 'features' vector
    - neighbour-related arrays
    - reference group ('ref1' or 'ref2'), containing only 'meltpool_seq'.
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
        features_info: bool = False,
        neighbours_info: bool = False,
        ref_info: bool = True,
        series_variants: tuple[str, ...] | list[str] | None = None,
        random_state: int = 42,
    ):
        """
        Initialize the CubeH5Dataset.

        Args:
            dataset_path (Path): Base directory containing the metadata CSV and .h5 files.
            metadata_csv (Path | str, optional): Relative or absolute path to the metadata CSV
                with columns ['cube', 'layer', 'vector'].
            fraction (float | None, optional): Fraction of the dataset to sample, between 0 and 1.
                If None, the full dataset is used. Defaults to None.
            features_info (bool, optional): Whether to load the 'features' vector. Defaults to False.
            neighbours_info (bool, optional): Whether to load neighbour-related arrays:
                ['neighs_abs_rel_times', 'neighs_n_neighbours', 'neighs_rel_distances'].
                Defaults to False.
            ref_info (bool, optional): Whether to load the reference group ('ref1' or 'ref2'),
                which includes only 'meltpool_seq'. Defaults to True.
            series_variants (tuple[str, ...] | list[str] | None, optional):
                Series outputs to return when `ref_info=True`.
                Allowed values: `raw`, `selfstd`, `scaled`.
                Defaults to `("scaled",)`.
            random_state (int, optional): Random seed used for sampling reproducibility. Defaults to 42.
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_csv = Path(metadata_csv)
        self.metadata_path = (
            self.metadata_csv if self.metadata_csv.is_absolute() else self.dataset_path / self.metadata_csv
        )
        self.fraction = fraction

        # Validate fraction
        if self.fraction is not None and not (0 < self.fraction <= 1):
            raise ValueError(f"'fraction' must be between 0 and 1, got {self.fraction}")

        # Flags controlling what to load
        self.return_features = features_info
        self.return_neighbours = neighbours_info
        self.return_ref = ref_info
        self.series_variants = self._resolve_series_variants(ref_info, series_variants)

        self.random_state = random_state

        # Build the list of vector references (cube_path, layer_key, vector_key, reference)
        self.vector_refs = build_vector_references(
            self.metadata_path,
            fraction=self.fraction,
            random_state=self.random_state,
        )

        self._load_scalers()

    @classmethod
    def _resolve_series_variants(
        cls,
        return_ref: bool,
        variants: tuple[str, ...] | list[str] | None,
    ) -> tuple[str, ...]:
        """
        Validate and normalize requested series output variants.
        """
        if not return_ref:
            return tuple()

        normalized = ("scaled",) if variants is None else tuple(str(v).strip().lower() for v in variants)
        if len(normalized) == 0:
            raise ValueError("When ref_info=True, `series_variants` must contain at least one value.")

        invalid = [v for v in normalized if v not in cls._SERIES_VARIANTS]
        if invalid:
            raise ValueError(
                f"Unsupported series_variants={invalid}. "
                f"Allowed: {sorted(cls._SERIES_VARIANTS)}"
            )
        return normalized

    def _load_scalers(self) -> None:
        """
        Load required scalers for cube data from dataset root.

        Notes:
        - Cubes use two scalers: features + series.
        - We load only what is needed according to active flags.
        - Scalers are cached at class level and refreshed when dataset root changes.
        """
        scalers_dir = self.dataset_path.parent

        # 1) Reset cached scalers when switching to another split path.
        if CubeH5Dataset.scalers_dir != scalers_dir:
            CubeH5Dataset.features_scaler = None
            CubeH5Dataset.series_scaler = None
            CubeH5Dataset.scalers_dir = scalers_dir

        # 2) Build mapping of required scaler files.
        required_mapping: dict[str, str] = {}
        if self.return_features:
            required_mapping["feature_scaler.joblib"] = "features_scaler"
        if self.return_ref and "scaled" in self.series_variants:
            required_mapping["series_scaler.joblib"] = "series_scaler"

        if not required_mapping:
            return

        # 3) Load scaler objects from disk and cache them at class level.
        loaded = load_scalers(scalers_dir, file_mapping=required_mapping)
        CubeH5Dataset.features_scaler = loaded.get("features_scaler")
        CubeH5Dataset.series_scaler = loaded.get("series_scaler")

    def __len__(self) -> int:
        """Return the number of available samples."""
        return len(self.vector_refs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single sample lazily from the HDF5 file.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]:
                {
                    'features': torch.Tensor,                  # if enabled
                    'neighs_abs_rel_times': torch.Tensor,      # if enabled
                    'neighs_n_neighbours': torch.Tensor,        # if enabled
                    'neighs_rel_distances': torch.Tensor,      # if enabled
                    'meltpool_seq_raw': torch.Tensor,          # if requested
                    'meltpool_seq_selfstd': torch.Tensor,      # if requested
                    'meltpool_seq_scaled': torch.Tensor        # if requested
                }
        """
        cube_path, layer_key, vector_key, reference = self.vector_refs[idx]


        # 1) Lazy-load requested tensors for one cube vector/reference.
        sample = load_cube_vector_lazy(
            cube_path,
            layer_key,
            vector_key,
            reference,
            self.return_features,
            self.return_neighbours,
            self.return_ref,
        )
        

        out: Dict[str, torch.Tensor] = {}

        # 2) Neighbour tensors: pad/truncate to fixed training shape.
        if self.return_neighbours:
            out["neighs_abs_rel_times"] = pad_to_shape(
                torch.as_tensor(sample["neighs_abs_rel_times"], dtype=torch.float32),
                max_L=600,
                max_T=1900,
            )

            out["neighs_rel_distances"] = pad_to_shape(
                torch.as_tensor(sample["neighs_rel_distances"], dtype=torch.float32),
                max_L=600,
                max_T=1900,
            )

            out["neighs_n_neighbours"] = pad_to_shape(
                torch.as_tensor(sample["neighs_n_neighbours"], dtype=torch.int32),
                max_L=600,
            )

        # 3) Meltpool series: interpolate to fixed length and return requested variants.
        if self.return_ref:
            raw_melt = np.asarray(sample["meltpool_seq"], dtype=np.float32)
            interpolated_raw = interpolate_series(raw_melt, 600).astype(np.float32)

            if "raw" in self.series_variants:
                out["meltpool_seq_raw"] = torch.tensor(interpolated_raw, dtype=torch.float32)

            if "selfstd" in self.series_variants:
                selfstd = safe_standardize(interpolated_raw)
                out["meltpool_seq_selfstd"] = torch.tensor(selfstd, dtype=torch.float32)

            if "scaled" in self.series_variants:
                if CubeH5Dataset.series_scaler is None:
                    self._load_scalers()

                interpolated_melt_2d = interpolated_raw.reshape(-1, 1)
                scaled_melt_2d = CubeH5Dataset.series_scaler.transform(interpolated_melt_2d)
                scaled_melt = scaled_melt_2d.astype(np.float32).reshape(-1)
                out["meltpool_seq_scaled"] = torch.tensor(scaled_melt, dtype=torch.float32)

        # 4) Features: scale dense feature vector.
        if self.return_features:
            raw_feats = np.asarray(sample["features"], dtype=np.float32)
            if CubeH5Dataset.features_scaler is None:
                self._load_scalers()

            feats_2d = raw_feats.reshape(1, -1)
            scaled_feats_2d = CubeH5Dataset.features_scaler.transform(feats_2d)
            feats = scaled_feats_2d.astype(np.float32).reshape(-1)
            out["features"] = torch.tensor(feats, dtype=torch.float32)

        return out
