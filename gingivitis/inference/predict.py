"""Export image-level predictions from a user-supplied gingivitis checkpoint."""

import argparse
import json
from pathlib import Path

from core import collect_inputs, load_model, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True, help="Tooth-crop directory, not whole-jaw images")
    parser.add_argument("--output", default="outputs/gingivitis/inference/image_predictions.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--longitudinal", action="store_true", help="Preserve separate visit units")
    args = parser.parse_args()
    inputs = collect_inputs(args.input, args.longitudinal)
    model = load_model(args.weights, args.device)
    results = run_pipeline(model, inputs, args.device, args.batch_size)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing predictions: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    print(f"Saved {len(inputs)} image predictions to {output}")


if __name__ == "__main__":
    main()
