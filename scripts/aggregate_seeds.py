import argparse
import json
from pathlib import Path

from traffic_signs.evaluation.analysis import aggregate

parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate(args.reports), indent=2))
