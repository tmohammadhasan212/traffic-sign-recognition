import argparse
import logging
from dataclasses import replace
from pathlib import Path

from traffic_signs.training.config import Config
from traffic_signs.training.engine import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--seed", type=int)
parser.add_argument("--output-dir")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    logging.getLogger(__name__).info("Best checkpoint: %s", train(config))
