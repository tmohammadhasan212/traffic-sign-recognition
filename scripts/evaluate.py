"""Calibrate on validation first; test requires that saved calibration."""

import argparse
import logging
from pathlib import Path

from traffic_signs.calibration.temperature import (
    choose_threshold,
    fit_temperature,
    probabilities,
    save_calibration,
)
from traffic_signs.evaluation.report import write_report
from traffic_signs.training.engine import collect, load_checkpoint, loader, select_device

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--split", choices=["validation", "test"], required=True)
parser.add_argument("--calibration", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--target-accuracy", type=float, default=0.99)
parser.add_argument("--no-latency", action="store_true")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    device = select_device("auto")
    model, config = load_checkpoint(args.checkpoint, device)
    if args.split == "test" and config.smoke_limit is not None:
        raise ValueError("Smoke checkpoints cannot be used for final test evaluation")
    calibration = args.calibration or args.checkpoint.parent / "calibration.json"
    output = args.output or args.checkpoint.parent / args.split
    logits, targets = collect(model, loader(config, args.split), device)
    if args.split == "validation":
        temperature = fit_temperature(logits, targets)
        threshold = choose_threshold(probabilities(logits, temperature), targets.numpy(),
                                     args.target_accuracy)
        save_calibration(calibration, temperature, threshold, str(args.checkpoint.resolve()))
    else:
        if not calibration.is_file():
            raise FileNotFoundError("Run validation calibration before final test evaluation")
        import json
        parameters = json.loads(calibration.read_text())
        if Path(parameters["checkpoint"]).resolve() != args.checkpoint.resolve():
            raise ValueError("Calibration checkpoint mismatch")
        temperature, threshold = parameters["temperature"], parameters["threshold"]
    report = write_report(output, logits, targets, temperature, model, threshold,
                          not args.no_latency)
    logging.getLogger(__name__).info("%s accuracy %.4f, macro F1 %.4f, ECE %.4f", args.split,
                 report["metrics"]["accuracy"], report["metrics"]["macro_f1"],
                 report["metrics"]["ece"])
