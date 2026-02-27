import re
from typing import Union
from pathlib import Path

from itp_fabadII.logger import logger


def extract_pattern(
    path: Union[str, Path], 
    pattern: str,
    error_message: str = "Pattern not found in path"
) -> str:
    """
    Extracts a string matching a regex pattern from a file path or name.
    
    Args:
        path: The filename or path to search in
        pattern: Regex pattern with a capturing group (e.g., r"/(\\d{5})/")
        error_message: Custom error message if pattern is not found
        
    Returns:
        The extracted string from the first capturing group
        
    Raises:
        ValueError: If the pattern is not found in the path
    """
    match = re.search(pattern, str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"{error_message}: {path}")
    return match.group(1)


### COMB
def extract_comb_laser_number(path: Union[str, Path]) -> int:
    """Extracts the numeric laser index (e.g., 'L1' → 1) from a file path or name."""
    return int(extract_pattern(path, r"L(\d+)", "Laser number not found in path"))


def extract_comb_layer_idx(path: Union[str, Path]) -> int:
    """Extracts the numeric layer index (e.g., '_C12_' or '_C12.h5' → 12) from a file path or name."""
    return int(extract_pattern(path, r"_C(\d+)(?:_|\.|$)", "Layer index not found in path"))


def extract_comb_idx(path: Union[str, Path]) -> str:
    """Extracts the numeric sample ID (e.g., 25126) from a file path."""
    return int(extract_pattern(path, r"/(\d{5})/", "Sample ID not found in path"))


### CUBE
def extract_cube_layer_idx(path: Union[str, Path]) -> int:
    """Extracts the numeric layer index (e.g., 'Layer_40' → 40) from a Cube file path."""
    return int(extract_pattern(path, r"Layer_(\d+)", "Cube layer index not found in path"))

def extract_cube_idx(path: Union[str, Path]) -> int:
    """Extracts the numeric cube ID (e.g., 'Cube_5_Layer_40.csv' → 5)."""
    return int(extract_pattern(path, r"Cube_(\d+)", "Cube ID not found in path"))


def _validate_sampling(stride: int, offset: int) -> tuple[int, int]:
    """
    Validate and normalize layer sampling parameters.

    Args:
        stride: Keep one layer every `stride` layers. Must be >= 1.
        offset: Phase of the sampling rule. Any integer is accepted.

    Returns:
        A normalized `(stride, offset)` pair cast to `int`.
    """
    stride = int(stride)
    offset = int(offset)
    if stride < 1:
        raise ValueError(f"Invalid stride={stride}. stride must be >= 1.")
    return stride, offset


def _is_selected_layer(layer_idx: int, stride: int, offset: int) -> bool:
    """
    Decide whether a layer should be selected by sampling.

    Selection rule:
      (layer_idx - offset) % stride == 0
    """
    return (int(layer_idx) - int(offset)) % int(stride) == 0


def list_comb_files(
    folder: str,
    laser: str,
    base_path: Path,
    stride: int = 1,
    offset: int = 1,
    include_first: bool = False,
) -> list[Path]:
    """
    List comb parquet files for one comb folder and one laser folder, with optional layer sampling.

    Args:
        folder: Comb folder name (e.g. "25126").
        laser: Laser folder name (e.g. "L1").
        base_path: Root directory that contains comb folders.
        stride: Keep one layer every `stride` layers.
        offset: Phase of the sampling. A layer is kept when
            `(layer_idx - offset) % stride == 0`.
        include_first: When `True`, include the first layer file in the result
            even if it does not satisfy the sampling rule.

    Returns:
        Sorted list of parquet files matching the sampling rule.
    """
    stride, offset = _validate_sampling(stride=stride, offset=offset)
    target_dir = base_path / folder / laser
    files = sorted(target_dir.glob("*.parquet"), key=lambda p: extract_comb_layer_idx(str(p)))
    selected_files = [
        file_path
        for file_path in files
        if _is_selected_layer(extract_comb_layer_idx(file_path), stride=stride, offset=offset)
    ]
    if include_first and files and files[0] not in selected_files:
        selected_files = [files[0], *selected_files]
    if files and not selected_files:
        logger.warning(
            f"No files selected in {target_dir} with stride={stride} and offset={offset}."
        )
    return selected_files


def list_cube_files(
    folder: str,
    cube: int,
    base_path: Path,
    stride: int = 1,
    offset: int = 1,
    include_first: bool = False,
) -> list[Path]:
    """
    List cube CSV files for one split folder and one cube ID, with optional layer sampling.

    Args:
        folder: Split folder name (e.g. "tr", "vl", "ts").
        cube: Cube identifier.
        base_path: Root directory that contains cube folders.
        stride: Keep one layer every `stride` layers.
        offset: Phase of the sampling. A layer is kept when
            `(layer_idx - offset) % stride == 0`.
        include_first: When `True`, include the first layer file in the result
            even if it does not satisfy the sampling rule.

    Returns:
        Sorted list of CSV files matching the sampling rule.
    """
    stride, offset = _validate_sampling(stride=stride, offset=offset)
    target_dir = base_path / folder / f"Cube_{cube}"
    files = sorted(target_dir.glob("*.csv"), key=lambda p: extract_cube_layer_idx(str(p)))
    selected_files = [
        file_path
        for file_path in files
        if _is_selected_layer(extract_cube_layer_idx(file_path), stride=stride, offset=offset)
    ]
    if include_first and files and files[0] not in selected_files:
        selected_files = [files[0], *selected_files]
    if files and not selected_files:
        logger.warning(
            f"No files selected in {target_dir} with stride={stride} and offset={offset}."
        )
    return selected_files


def resolve_versioned_output_root(output_root: Path, dataset_kind: str) -> tuple[Path, int]:
    """
    Resolve the next dataset output folder using monotonic directory versioning.

    Versioning policy:
    - cube -> `cube_dataset_v<integer>`
    - comb -> `comb_dataset_v<integer>`

    The next version is computed by scanning existing siblings in `output_root` and
    taking `max(existing_versions) + 1`. If none exist, version starts at 1.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_kind}_dataset_v"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    existing_versions: list[int] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        existing_versions.append(int(match.group(1)))

    version = (max(existing_versions) + 1) if existing_versions else 1
    dataset_output_root = output_root / f"{prefix}{version}"
    return dataset_output_root, version

