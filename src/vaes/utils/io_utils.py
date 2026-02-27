
import shutil
from pathlib import Path
from typing import List, Tuple
import argparse
from itp_fabadII.logger import logger

def clean_up(
    dirs_to_remove=("outputs", "multirun", "mlruns", "logs"),
    dry_run: bool = False
) -> None:
    """
    Removes project runtime directories such as Hydra outputs, logs, and mlruns.

    Args:
        dirs_to_remove (tuple): Directory names to remove from the project root.
        dry_run (bool): If True, only logs what would be removed.
    """
    for name in dirs_to_remove:
        for path in Path(".").glob(name):
            if path.exists() and path.is_dir():
                if dry_run:
                    logger.info(f"[clean_up] Would remove: {path}")
                else:
                    shutil.rmtree(path)
                    logger.info(f"[clean_up] Removed: {path}")
            else:
                logger.debug(f"[clean_up] Skipped (not found): {path}")


def parse_cli(argv: List[str] | None = None) -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse command-line arguments for experiment scripts.

    Recognizes predefined flags (e.g., --fast-dev-run) and returns any
    unknown arguments as Hydra configuration overrides.

    Args:
        argv (List[str] | None): Optional list of arguments. If None,
            uses sys.argv[1:].

    Returns:
        Tuple[argparse.Namespace, List[str]]:
            - Parsed known arguments as a Namespace.
            - Remaining arguments as a list of Hydra overrides.
    """
    parser = argparse.ArgumentParser(
        description="Parse CLI arguments and Hydra overrides for experiment runs."
    )

    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help=(
            "If set, performs a short development run using only one batch "
            "from the training and validation loaders, then exits early. "
            "Useful for verifying the pipeline setup."
        ),
    )

    args, hydra_overrides = parser.parse_known_args(args=argv)
    return args, hydra_overrides


import mlflow
import torch
from pathlib import Path
from typing import Any


def load_model_from_registry(
    model_name: str,
    version: int,
    map_location: str | torch.device = "cpu"
) -> Any:
    """
    Load exclusively artifacts/best_model/data/model.pth from the MLflow Model Registry.
    No fallback logic and no candidate search is performed.

    Raises an error if the file does not exist.
    """

    # Download full artifact directory for this version
    model_uri = f"models:/{model_name}/{version}"
    root = Path(mlflow.artifacts.download_artifacts(model_uri))

    target = root / "data" / "model.pth"

    if not target.exists():
        directory_listing = "\n".join(str(p) for p in root.rglob("*"))
        raise FileNotFoundError(
            f"[MLflow Loader] Expected model file not found:\n"
            f"  {target}\n\n"
            f"Root directory: {root}\n"
            f"Contents:\n{directory_listing}"
        )

    model = torch.load(target, map_location=map_location, weights_only=False)
    print(f"[MLflow Loader] Loaded model file: {target}")

    return model
