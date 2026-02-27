from pathlib import Path
from typing import Any, Dict

import h5py
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from itp_fabadII.logger import logger

SCALER_FILES = {
    "feature_scaler.joblib": "features_scaler",
    "series_scaler.joblib": "series_scaler",
}

def load_scalers(base_dir: Path, file_mapping: Dict[str, str] = SCALER_FILES) -> Dict[str, Any]:
    """
    Load multiple scalers generically from a directory.

    Args:
        base_dir (Path):
            Directory where scaler files (.joblib) are located.
        file_mapping (dict[str, str]):
            Mapping: {filename: logical_name}
            Example:
                {
                    "feature_scaler.joblib": "features",
                    "series_scaler.joblib": "series",
                }

    Returns:
        dict[str, Any]: Dict mapping logical_name -> loaded scaler.

    Raises:
        FileNotFoundError: If any required scaler file is missing.
    """

    missing: list[str] = []
    loaded: Dict[str, Any] = {}

    for filename, logical_name in file_mapping.items():
        fpath: Path = base_dir / filename

        if not fpath.exists():
            missing.append(filename)
            continue

        loaded[logical_name] = joblib.load(fpath)

    if missing:
        raise FileNotFoundError(
            "Missing required scaler files:\n"
            + "\n".join(f"- {m}" for m in missing)
            + f"\nSearched in: {base_dir}"
        )

    return loaded



def compute_cube_scalers(train_split_dir: Path, output_dir: Path | None = None) -> None:
    """
    Computes and saves two StandardScalers:
    - One for all meltpool_seq time series from ref1 and ref2
    - One for all feature vectors

    Fit is performed using the training split only.
    Scalers are saved in the dataset root directory.

    Args:
        train_split_dir (Path): Path to processed training split directory (e.g. `<dataset>/tr`).
        output_dir (Path | None): Destination directory for scaler files.
            When omitted, defaults to `train_split_dir.parent` (dataset root).
    """
    scaler_series = StandardScaler()
    scaler_features = StandardScaler()

    train_split_dir = Path(train_split_dir)
    dest_dir = Path(output_dir) if output_dir is not None else train_split_dir.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(train_split_dir.rglob("Cube_*_Layer_*.h5"))

    for h5_path in tqdm(h5_files, desc="🔄 Fitting cube scalers (series + features)"):
        with h5py.File(h5_path, "r") as h5f:
            for layer in h5f.values():
                for vector in layer.values():

                    # Fit both time series from ref1 and ref2
                    series1 = vector["ref1"]["meltpool_seq"][:]
                    series2 = vector["ref2"]["meltpool_seq"][:]
                    scaler_series.partial_fit(series1.reshape(-1, 1))
                    scaler_series.partial_fit(series2.reshape(-1, 1))

                    # Fit feature vector
                    features = vector["features"][:]
                    scaler_features.partial_fit(features.reshape(1, -1))

    # Save both scalers
    joblib.dump(scaler_series, dest_dir / "series_scaler.joblib")
    joblib.dump(scaler_features, dest_dir / "feature_scaler.joblib")

    logger.info(f"✅ Saved series scaler to {dest_dir / 'series_scaler.joblib'}")
    logger.info(f"✅ Saved feature scaler to {dest_dir / 'feature_scaler.joblib'}")


def compute_comb_scalers(train_split_dir: Path, output_dir: Path | None = None) -> None:
    """
    Computes and saves two StandardScalers:
    - One for all meltpool_seq time series from hatching vectors
    - One for all feature vectors

    Fit is performed using the training split only.
    Scalers are saved in the dataset root directory.

    Args:
        train_split_dir (Path): Path to processed training split directory (e.g. `<dataset>/tr`).
        output_dir (Path | None): Destination directory for scaler files.
            When omitted, defaults to `train_split_dir.parent` (dataset root).
    """
    scaler_series = StandardScaler()
    scaler_features = StandardScaler()

    train_split_dir = Path(train_split_dir)
    dest_dir = Path(output_dir) if output_dir is not None else train_split_dir.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(train_split_dir.rglob("Comb_*_L*_C*.h5"))

    for h5_path in tqdm(h5_files, desc="🔄 Fitting comb scalers (series + features)"):
        with h5py.File(h5_path, "r") as h5f:
            
            if "hatching" not in h5f:
                continue
            for vector in h5f["hatching"].values():

                # Fit meltpool time series (hatching only).
                series1 = vector["meltpool_seq"][:]
                scaler_series.partial_fit(series1.reshape(-1, 1))

                if "features" not in vector:
                    raise KeyError(
                        f"Missing 'features' dataset in {h5_path} "
                        f"(segment=hatching, vector={vector.name.split('/')[-1]})."
                    )
                features = np.asarray(vector["features"][:], dtype=np.float32)
                scaler_features.partial_fit(features.reshape(1, -1))

    # Save both scalers
    joblib.dump(scaler_series, dest_dir / "series_scaler.joblib")
    joblib.dump(scaler_features, dest_dir / "feature_scaler.joblib")

    logger.info(f"✅ Saved series scaler to {dest_dir / 'series_scaler.joblib'}")
    logger.info(f"✅ Saved feature scaler to {dest_dir / 'feature_scaler.joblib'}")
