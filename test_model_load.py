import logging
import os
import argparse
import math

import hydra
from hydra.core.config_store import ConfigStore

from src.config import BabyLMConfig

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

# A logger for this file
logger = logging.getLogger(__name__)

@record
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig):
    logger.info("Initializing model")
    model = load_base_model(cfg)

if __name__ == "__main__":
    main()