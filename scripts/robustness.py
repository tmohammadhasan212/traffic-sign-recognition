"""Evaluate clean and corrupted held-out images."""

import argparse
import logging
from pathlib import Path

from traffic_signs.robustness.evaluate import evaluate, save_results
from traffic_signs.training.engine import load_checkpoint, select_device

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--split", choices=["validation", "test"], default="validation")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--output", type=Path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    device = select_device("auto")
    model, config = load_checkpoint(args.checkpoint, device)
    output = args.output or args.checkpoint.parent / f"robustness_{args.split}"
    records = evaluate(model, config, args.split, device, args.seed)
    save_results(records, output)
    logging.getLogger(__name__).info("Wrote %d conditions to %s", len(records), output)
