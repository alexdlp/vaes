import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from vaes.logger import logger
from vaes.pipelines import create_pipeline
from vaes.utils import merge_model_section
from vaes.utils.config_utils import export_args_to_env, load_environment, normalize_hydra_overrides, parse_args


# Resolve the config directory from this module so the console script can
# locate the Hydra tree without relying on the current working directory.
CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


@hydra.main(config_path=CONF_DIR, config_name="config", version_base="1.3")
def run_training_pipeline(cfg: DictConfig):

    for section in list(cfg.model.keys()):
        merge_model_section(cfg, section)

    pipeline = create_pipeline(cfg)

    try:
        pipeline.setup()
        pipeline.fit()
    except Exception as ex:
        logger.error(f"❌ Exception occurred: {ex}", exc_info=True)
    finally:
        pass


def main():
    # 1. Parse external CLI args and preserve Hydra runtime args such as -m.
    args, hydra_argv = parse_args()

    # 2. Export CLI args to env.
    export_args_to_env(args)

    # 3. Load .env before Hydra composes the config tree.
    load_environment()

    # 4. Normalize user-facing model overrides to the staged Hydra layout.
    hydra_argv = normalize_hydra_overrides(hydra_argv)

    # 5. Leave only Hydra runtime args on sys.argv so Hydra can parse them normally.
    sys.argv = [sys.argv[0], *hydra_argv]

    # 6. Launch Hydra pipeline.
    run_training_pipeline()
