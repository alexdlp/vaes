from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import h5py
from scipy.interpolate import interp1d

def sample_vector_metadata(
    metadata_csv: Path,
    num_samples: int
) -> List[Tuple[Path, str, str]]:
    """
    Samples vector references from a CSV and builds (file_path, layer_key, vector_key) tuples.

    Args:
        metadata_csv: Path to the CSV with columns ['cube', 'layer', 'vector']
        num_samples: Number of vector references to sample.

    Returns:
        List of (Path, str, str): (cube_path, layer_key, vector_key)
    """
    metadata_csv = Path(metadata_csv) 
    df = pd.read_csv(metadata_csv)
    sampled_df = df.sample(n=num_samples, random_state=42)

    base_dir = metadata_csv.parent  # Same directory as CSV and .h5 files

    vector_refs = []
    for _, row in sampled_df.iterrows():
        cube_id = int(row["cube"])
        layer_id = int(row["layer"])
        vector_id = int(row["vector"])

        cube_path = base_dir / f"Cube_{cube_id}.h5"
        layer_key = f"layer_{layer_id:03d}"
        vector_key = f"vector_{vector_id:03d}"

        vector_refs.append((cube_path, layer_key, vector_key))

    return vector_refs



def build_vector_references(
    metadata_csv: Path,
    fraction: float = 1.0,
    random_state: int = 42
) -> List[Tuple[Path, str, str, str]]:
    """
    Builds vector references from a metadata CSV, sampling and duplicating 
    for both references ('ref1' and 'ref2').

    Args:
        metadata_csv (Path): Path to the CSV containing columns ['cube', 'layer', 'vector'].
        fraction (float, optional): Fraction of the full dataset to sample after concatenating
            both references. Must be between 0 and 1. Defaults to 1.0 (use entire dataset).
        random_state (int, optional): Random seed used for sampling. Defaults to 42.

    Returns:
        List[Tuple[Path, str, str, str]]: List of (cube_path, layer_key, vector_key, reference).
    """
    metadata_csv = Path(metadata_csv)
    df = pd.read_csv(metadata_csv)

    base_dir = metadata_csv.parent

    # Vectorized key and path creation
    df["cube_path"] = df["cube"].apply(lambda c: base_dir / f"Cube_{int(c)}.h5")
    df["layer_key"] = df["layer"].apply(lambda l: f"layer_{int(l):03d}")
    df["vector_key"] = df["vector"].apply(lambda v: f"vector_{int(v):03d}")

    # Duplicate for both references
    df_ref1 = df.assign(reference="ref1")
    df_ref2 = df.assign(reference="ref2")
    df_all = pd.concat([df_ref1, df_ref2], ignore_index=True)

    # --- Stratified sampling by (layer, reference) ---
    if fraction < 1.0:
        df_all = (
            df_all.groupby(["layer", "reference"], group_keys=False)
            .sample(frac=fraction, random_state=random_state)
        )

    # --- Return as list of tuples ---
    return list(
        df_all[["cube_path", "layer_key", "vector_key", "reference"]]
        .itertuples(index=False, name=None)
    )


def load_cube_vector_lazy(
    path: Path,
    layer_key: str,
    vector_key: str,
    reference: str,
    features: bool = True,
    neighbours_info: bool = True,
    ref_info: bool = True
) -> Dict[str, Any]:
    """
    Load one cube vector lazily from HDF5.

    This reader targets the cube structure:
    `/<layer_key>/<vector_key>/{features, neighs_*, ref1|ref2/meltpool_seq}`
    and returns only the sections requested by flags.

    Args:
        path (Path): Path to the .h5 file.
        layer_key (str): Key for the layer group (e.g. 'layer_004').
        vector_key (str): Key for the vector group (e.g. 'vector_012').
        reference (str): Either 'ref1' or 'ref2'.
        features (bool): Whether to load the 'features' dataset.
        neighbours_info (bool): Whether to load neighbour-related datasets:
            'neighs_abs_rel_times', 'neighs_n_neighbours', 'neighs_rel_distances'.
        ref_info (bool): Whether to load the selected reference group ('ref1' or 'ref2').

    Returns:
        Dict[str, Any]: Dictionary with selected content, e.g.:
            {
                'features': ndarray,
                'neighs_abs_rel_times': ndarray,
                'neighs_n_neighbours': ndarray,
                'neighs_rel_distances': ndarray,
                'ref1': {
                    'laserview_seq': ndarray,
                    'meltpool_seq': ndarray,
                    'neighs_laserviews': ndarray
                }
            }
    """
    result: Dict[str, Any] = {}

    with h5py.File(path, "r") as f:
        # 1) Validate cube hierarchy keys.
        if layer_key not in f:
            raise KeyError(f"{layer_key} not found in {path.name}")

        if vector_key not in f[layer_key]:
            raise KeyError(f"{vector_key} not found in {path.name}/{layer_key}")

        group = f[layer_key][vector_key]

        # 2) Optionally read dense feature vector.
        if features and "features" in group:
            result["features"] = group["features"][()]

        # 3) Optionally read neighbour tensors.
        if neighbours_info:
            for key in ["neighs_abs_rel_times", "neighs_n_neighbours", "neighs_rel_distances"]:
                if key in group:
                    result[key] = group[key][()]

        # 4) Optionally read selected reference meltpool sequence.
        if ref_info and reference in group and "meltpool_seq" in group[reference]:
            result["meltpool_seq"] = group[reference]["meltpool_seq"][()]

    return result


def load_comb_vector_lazy(
    path: Path,
    segment_key: str,
    vector_key: str,
    features: bool = True,
    neighbours_info: bool = True,
    series_info: bool = True,
) -> Dict[str, Any]:
    """
    Load one comb vector lazily from HDF5.

    This reader targets the comb structure:
    `/<segment_key>/<vector_key>/{features, meltpool_seq, neighbours/*}`.
    Comb vectors do not use `ref1/ref2`.

    Args:
        path (Path): Path to the comb `.h5` file.
        segment_key (str): Segment group name (typically `hatching` or `edges`).
        vector_key (str): Vector key inside the segment group.
        features (bool): Whether to read `features`.
        neighbours_info (bool): Whether to read neighbour tensors.
        series_info (bool): Whether to read `meltpool_seq`.

    Returns:
        Dict[str, Any]: Flat dictionary with selected arrays:
            `features`, `meltpool_seq`, `neighs_abs_rel_times`,
            `neighs_rel_distances`, `neighs_n_neighbours`.
    """
    result: Dict[str, Any] = {}

    with h5py.File(path, "r") as f:
        # 1) Validate comb hierarchy keys.
        if segment_key not in f:
            raise KeyError(f"{segment_key} not found in {path}")

        segment_group = f[segment_key]
        if vector_key not in segment_group:
            raise KeyError(f"{vector_key} not found in {path}/{segment_key}")

        vector_group = segment_group[vector_key]

        # 2) Optionally read dense feature vector.
        if features:
            if "features" not in vector_group:
                raise KeyError(f"Missing 'features' in {path}/{segment_key}/{vector_key}")
            result["features"] = vector_group["features"][()]

        # 3) Optionally read meltpool sequence (no ref split in comb).
        if series_info:
            if "meltpool_seq" not in vector_group:
                raise KeyError(f"Missing 'meltpool_seq' in {path}/{segment_key}/{vector_key}")
            result["meltpool_seq"] = vector_group["meltpool_seq"][()]

        # 4) Optionally read neighbour tensors from subgroup `neighbours`.
        if neighbours_info:
            if "neighbours" not in vector_group:
                raise KeyError(f"Missing 'neighbours' in {path}/{segment_key}/{vector_key}")
            neigh = vector_group["neighbours"]

            abs_key = "abs_rel_times" if "abs_rel_times" in neigh else "neighs_abs_rel_times"
            dist_key = "rel_distances" if "rel_distances" in neigh else "neighs_rel_distances"
            n_key = "n_neighbours" if "n_neighbours" in neigh else "neighs_n_neighbours"

            for expected_key in (abs_key, dist_key, n_key):
                if expected_key not in neigh:
                    raise KeyError(
                        f"Missing neighbours dataset '{expected_key}' "
                        f"in {path}/{segment_key}/{vector_key}/neighbours"
                    )

            result["neighs_abs_rel_times"] = neigh[abs_key][()]
            result["neighs_rel_distances"] = neigh[dist_key][()]
            result["neighs_n_neighbours"] = neigh[n_key][()]

    return result

def load_cube_layer_data(
    metadata_csv: Path,
    cube_index: int,
    layer_index: int,
    features: bool = False,
    neighbours_info: bool = False,
    ref1_info: bool = True,
    ref2_info: bool = False,
    coordinates: bool = False,
) -> Dict[str, Any]:
    """
    Load vector data for one cube layer from a cube HDF5 file.

    The function filters the metadata CSV for the specified cube and layer,
    and loads data from the corresponding .h5 file. The user can selectively
    enable/disable loading of different vector components.

    Args:
        metadata_csv (Path): Path to CSV with columns ['cube', 'layer', 'vector'].
        cube_index (int): Index of the cube (e.g. 12 → Cube_12.h5).
        layer_index (int): Index of the layer (e.g. 4 → layer_004).
        features (bool): Load the main 'features' vector.
        neighbours_info (bool): Load neighbour-related vectors:
            'neighs_abs_rel_times', 'neighs_n_neighbours', 'neighs_rel_distances'.
        ref1_info (bool): Load reference group 'ref1' with:
            'laserview_seq', 'meltpool_seq', 'neighs_laserviews'.
        ref2_info (bool): Load reference group 'ref2' with:
            'laserview_seq', 'meltpool_seq', 'neighs_laserviews'.
        coordinates (bool): Load 'vec_pnts' (vector coordinates).

    Returns:
        Dict[str, Any]: A dictionary structured as:
        {
            "vector_000": {
                "features": ndarray (if enabled),
                "neighs_abs_rel_times": ndarray (if enabled),
                "neighs_n_neighbours": ndarray (if enabled),
                "neighs_rel_distances": ndarray (if enabled),
                "vec_pnts": ndarray (if enabled),
                "ref1": {
                    "laserview_seq": ndarray,
                    "meltpool_seq": ndarray,
                    "neighs_laserviews": ndarray
                },
                "ref2": {
                    "laserview_seq": ndarray,
                    "meltpool_seq": ndarray,
                    "neighs_laserviews": ndarray
                }
            },
            ...
        }
    """
    # 1) Validate that requested cube/layer exists in the index CSV.
    metadata_csv = Path(metadata_csv)
    df = pd.read_csv(metadata_csv)

    filtered = df[(df["cube"] == cube_index) & (df["layer"] == layer_index)]
    if filtered.empty:
        raise ValueError(f"No data found for cube={cube_index}, layer={layer_index}")

    # 2) Resolve cube file path and target layer key.
    base_dir = metadata_csv.parent
    cube_path = base_dir / f"Cube_{cube_index}.h5"
    layer_key = f"layer_{layer_index:03d}"

    vectors: Dict[str, Any] = {}

    # 3) Iterate vectors in the layer and read requested payload sections.
    with h5py.File(cube_path, "r") as f:
        if layer_key not in f:
            raise KeyError(f"{layer_key} not found in {cube_path.name}")

        for vector_key in f[layer_key]:
            vector_group = f[layer_key][vector_key]
            vector_data: Dict[str, Any] = {}
            
            # 3.1) Core datasets (features/neighbours/coordinates).
            if features and "features" in vector_group:
                vector_data["features"] = vector_group["features"][()]

            if neighbours_info:
                for neigh_key in ["neighs_abs_rel_times", "neighs_n_neighbours", "neighs_rel_distances"]:
                    if neigh_key in vector_group:
                        vector_data[neigh_key] = vector_group[neigh_key][()]

            if coordinates and "vec_pnts" in vector_group:
                vector_data["vec_pnts"] = vector_group["vec_pnts"][()]

            # 3.2) Optional ref payloads.
            for ref_name, enabled in [("ref1", ref1_info), ("ref2", ref2_info)]:
                if enabled and ref_name in vector_group:
                    subgroup = vector_group[ref_name]
                    vector_data[ref_name] = {
                        k: subgroup[k][()]
                        for k in ["laserview_seq", "meltpool_seq", "neighs_laserviews"]
                        if k in subgroup
                    }

            vectors[vector_key] = vector_data

    # 4) Return mapping: vector_key -> loaded payload.
    return vectors

def safe_standardize(x: np.ndarray, std_eps: float = 1e-6, min_amplitude: float = 1e-3) -> np.ndarray:
    """
    Robust standardization for 1D arrays:
    - If std > std_eps → standardize.
    - If std too small → center only.
    - If flat signal → return constant array.

    Args:
        x (np.ndarray): 1D input array.
        std_eps (float): Minimum std threshold for scaling.
        min_amplitude (float): Fallback value for flat signals.

    Returns:
        np.ndarray: Standardized or centered 1D array.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("safe_standardize only supports 1D arrays.")

    if len(x) < 2:
        return np.full_like(x, min_amplitude)

    mean = np.mean(x)
    std = np.std(x)

    if std > std_eps:
        return (x - mean) / std

    # std muy pequeño → consideramos posible señal plana
    amplitude = float(np.max(x) - np.min(x))

    if amplitude < min_amplitude:
        # señal prácticamente plana → array constante
        return np.full_like(x, min_amplitude)

    # no del todo plana, pero con std numéricamente pequeña → solo centrar
    return x - mean




def interpolate_series(series:np.ndarray, target_length:int):

    series = series.flatten()


    original_length = len(series)

    # Return secuence if already correct size
    if original_length == target_length:
        return series
    
    # Ensure sequence has at least 2 points
    if original_length < 2:
        #logger.warning("Sequence has less than 2 points; returning a constant sequence.")
        return np.full(target_length, series[0])

    # Interpolate to the original length
    x_orig = np.linspace(0, 1, original_length)
    x_interp = np.linspace(0, 1, target_length)
    
    return interp1d(x_orig, series, kind='linear', fill_value="extrapolate")(x_interp)
