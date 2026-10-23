import argparse
import logging
from pathlib import Path

from traffic_signs.data.download import download
from traffic_signs.data.prepare import prepare, write_or_verify_track_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("data/raw"))
parser.add_argument("--output", type=Path, default=Path("data/split_manifest.json"))
parser.add_argument("--download", action="store_true")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--validation-fraction", type=float, default=0.2)
parser.add_argument("--allow-subset", action="store_true", help="For local fixtures only")
parser.add_argument("--skip-image-verification", action="store_true")
parser.add_argument("--track-manifest", type=Path, default=Path("configs/gtsrb_track_split.json"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    if args.download:
        download(args.root)
    result = prepare(args.root, args.output, args.seed, args.validation_fraction,
                     not args.allow_subset, not args.skip_image_verification)
    if not args.allow_subset:
        write_or_verify_track_manifest(result, args.track_manifest)
    logging.getLogger(__name__).info("Prepared %d training and %d test rows at %s", len(result["training"]),
                 len(result["test"]), args.output)
