from omegaconf import DictConfig
from vaes.logger import logger
from vaes.pipelines import create_pipeline
from vaes.utils import merge_model_section
from vaes.utils.config_utils import load_environment, parse_args, export_args_to_env, load_hydra_config

# @hydra.main(config_path=str(Path(__file__).resolve().parents[3] / "conf"), 
#             config_name="config", version_base=None)
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
    # 1. Parse CLI args (
    args, hydra_overrides = parse_args()

    # 2. Export CLI args to env 
    export_args_to_env(args)

    # 3. Load .env environment
    load_environment()

    # 4. Launch Hydra pipeline
    cfg = load_hydra_config(hydra_overrides=hydra_overrides) 
    run_training_pipeline(cfg=cfg)
  

 

