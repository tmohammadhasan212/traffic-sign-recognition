import argparse
import json
from pathlib import Path

from traffic_signs.inference.predictor import Predictor, decode_image

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--calibration", type=Path)
parser.add_argument("--explain-output", type=Path)

if __name__ == "__main__":
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, args.calibration)
    image = decode_image(args.image.read_bytes())
    print(json.dumps(predictor.predict(image), indent=2))
    if args.explain_output:
        args.explain_output.write_bytes(predictor.explain(image))
