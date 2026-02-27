from omegaconf import DictConfig
from vaes.logger import logger
from vaes.pipelines import create_pipeline
from vaes.utils import merge_model_section
from vaes.utils.config_utils import load_environment, parse_args, export_args_to_env, load_hydra_config


def run_training_pipeline(cfg: DictConfig):

    for section in list(cfg.model.keys()):
        merge_model_section(cfg, section)

    pipeline = create_pipeline(cfg)

    try:
        pipeline.setup()
        pipeline.fit()
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}", exc_info=True)


def main():
    args, hydra_overrides = parse_args()
    export_args_to_env(args)
    load_environment()
    cfg = load_hydra_config(hydra_overrides=hydra_overrides)
    run_training_pipeline(cfg=cfg)
