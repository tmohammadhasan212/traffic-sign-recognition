import argparse
import json
from pathlib import Path

from traffic_signs.evaluation.analysis import error_cases

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--manifest", type=Path, default=Path("data/split_manifest.json"))
parser.add_argument("--split", choices=["validation", "test"], required=True)
parser.add_argument("--output", type=Path, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(error_cases(args.predictions, args.manifest, args.split),
                                      indent=2))
