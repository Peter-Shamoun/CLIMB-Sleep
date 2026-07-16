import logging
import os
import argparse
import math
from src.config import BabyLMConfig


def main(cfg: BabyLMConfig):
    logger.info("Initializing model")
    model = load_base_model(cfg)

if __name__ == "__main__":
    main()