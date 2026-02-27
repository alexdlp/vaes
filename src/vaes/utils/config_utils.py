
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose, initialize
from omegaconf import DictConfig
import warnings
from typing import List, Optional
from pathlib import Path
import os
from dotenv import load_dotenv
import argparse

from types import SimpleNamespace
from collections.abc import MutableMapping

from vaes.logger import logger

class ConfigNamespace(SimpleNamespace, MutableMapping):
    """SimpleNamespace that also behaves like a dict."""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()
    
    def __repr__(self):
        return f"ConfigNamespace({self.__dict__})"

    @classmethod
    def from_dict(cls, obj):
        """Recursively build ConfigNamespace from a nested dict."""
        if not isinstance(obj, dict):
            return obj
        return cls(**{k: cls.from_dict(v) for k, v in obj.items()})

    @classmethod
    def to_builtin(cls, obj):
        """
        Recursively convert a ConfigNamespace or nested structures into
        plain Python types (dicts, lists, primitives).

        Args:
            obj: ConfigNamespace, dict, list, tuple, or primitive.

        Returns:
            A fully Python-native structure, safe for serialization or logging.
        """
        if isinstance(obj, cls):
            return {k: cls.to_builtin(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: cls.to_builtin(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [cls.to_builtin(v) for v in obj]
        else:
            return obj

def merge_model_section(cfg: DictConfig, section: str) -> DictConfig:
    """
    Promote one staged section from `cfg.model` into root config.

    Expected staged layout in selected model config files:
    - `cfg.model.model`      -> real model section (`name`, `params`, ...)
    - `cfg.model.data`       -> data section override
    - `cfg.model.callbacks`  -> callbacks section override

    Rules:
    - `section == "model"`:
      flattens `cfg.model.model` into `cfg.model` while keeping pending staged
      siblings (`data`, `callbacks`) for subsequent loop iterations.
    - Any other section:
      full replacement at root (`cfg[section] = cfg.model[section]`) to avoid
      stale keys leaking from the base config (e.g. `ref_info` in comb).

    Returns:
        The updated configuration (mutated in-place).
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected cfg to be an OmegaConf DictConfig.")
    if "model" not in cfg:
        warnings.warn("No 'model' section found in config; skipping merge.")
        return cfg
    if section not in cfg.model:
        return cfg

    if section == "model":
        if "model" not in cfg.model:
            return cfg

        model_src = cfg.model["model"]
        if not isinstance(model_src, DictConfig):
            raise TypeError("Expected cfg.model['model'] to be a DictConfig.")

        # Flatten model payload into cfg.model without dropping staged siblings.
        with open_dict(cfg.model):
            for key, value in model_src.items():
                cfg.model[key] = value
            if "model" in cfg.model:
                del cfg.model["model"]

        return cfg

    src = cfg.model[section]

    with open_dict(cfg):
        cfg[section] = OmegaConf.create(OmegaConf.to_container(src, resolve=False))

    with open_dict(cfg.model):
        if section in cfg.model:
            del cfg.model[section]

    return cfg


def load_hydra_config(hydra_overrides: List[str]) -> DictConfig:
    """
    Compose the Hydra config tree:
      - Base config: config/config.yaml (and its includes)
      - Additional CLI overrides passed through (e.g., data.* = ...)
    """
    abs_conf_dir = Path(__file__).resolve().parents[3] / "conf"
    rel_conf_dir = os.path.relpath(abs_conf_dir, start=Path(__file__).resolve().parent)

    with initialize(version_base="1.3", config_path=str(rel_conf_dir)):
        cfg = compose(config_name="config", overrides=[*hydra_overrides])

    return cfg


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flattens a nested dictionary into dot notation.
    Example:
        {"a": {"b": 1}} → {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple)):
                v = ",".join(map(str, v))
            elif not isinstance(v, (str, int, float, bool)) and v is not None:
                v = str(v)
            items.append((new_key, v))
    return dict(items)


def load_environment(args: Optional[argparse.Namespace] = None) -> None:
    """
    Loads environment variables from the project's .env file, independent of
    Hydra's working directory. Also exports external CLI arguments to environment
    variables when provided.

    - Ensures predictable .env resolution.
    - Warns if the .env file is missing.
    - Avoids silent failures by logging explicit outcomes.

    Args:
        args (Optional[argparse.Namespace]): Parsed CLI arguments to export.
    """
    env_path = Path(__file__).resolve().parents[3] / ".env"

    # --- Load .env file ---
    if env_path.is_file():
        try:
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Loaded environment variables from: {env_path}")
        except Exception as e:
            logger.warning(f"Failed to load .env file at {env_path}: {e}")
    else:
        logger.warning(f".env file not found at expected path: {env_path}")

    # --- Export external CLI args to environment ---
    if args is not None:
        try:
            export_args_to_env(args)
            logger.info("Exported CLI arguments into environment variables.")
        except Exception as e:
            logger.warning(f"Failed to export CLI args to environment: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse external CLI arguments that must remain independent from Hydra.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="CLI flags for external runtime behavior."
    )

    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Enable a minimal debugging execution path."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="MLflow run ID to resume training from."
    )

    args, unknown = parser.parse_known_args()
    return args, unknown


def export_args_to_env(args: argparse.Namespace) -> None:
    """
    Export parsed CLI arguments into environment variables so Hydra remains
    clean and unaffected while allowing the training pipeline to inspect them.

    Args:
        args (argparse.Namespace): Arguments parsed by parse_args().
    """
    os.environ["FAST_DEV_RUN"] = "1" if args.fast_dev_run else "0"

    if args.resume is not None:
        os.environ["RESUME_RUN_ID"] = args.resume
    else:
        # Ensure the variable is absent if no resume ID was provided.
        os.environ.pop("RESUME_RUN_ID", None)
