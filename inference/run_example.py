"""Validate the public 34-view input example without private diagnostic logic."""

import argparse
import json
from pathlib import Path


FDI_ORDER = [18, 17, 16, 15, 14, 13, 12, 11,
             21, 22, 23, 24, 25, 26, 27, 28,
             38, 37, 36, 35, 34, 33, 32, 31,
             41, 42, 43, 44, 45, 46, 47, 48]
ROOT = Path(__file__).resolve().parents[1]


def collect_inputs(directory):
    images = sorted(directory.glob("*.png"))
    if len(images) != 34:
        raise ValueError(f"Expected 34 PNG inputs; found {len(images)}")
    annotations = []
    for slot, image in enumerate(images, 1):
        if not image.name.startswith(f"{slot:02d}_"):
            raise ValueError(f"Invalid slot order: {image.name}")
        if slot <= 32 and not image.name.startswith(f"{slot:02d}_T{FDI_ORDER[slot - 1]}_"):
            raise ValueError(f"Invalid tooth order: {image.name}")
        if slot == 33 and "lower" not in image.stem:
            raise ValueError("Slot 33 must be the lower jaw")
        if slot == 34 and "upper" not in image.stem:
            raise ValueError("Slot 34 must be the upper jaw")
        annotation = image.with_suffix(".json")
        data = json.loads(annotation.read_text(encoding="utf-8"))
        if data.get("imagePath") != image.name or not data.get("shapes"):
            raise ValueError(f"Broken PNG/JSON pairing or empty annotations: {annotation}")
        annotations.append(annotation)
    return images, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "DATA/D0064")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs only (the default)")
    args = parser.parse_args()
    collect_inputs(args.input.resolve())
    print("Validated 34 image/JSON pairs: 32 tooth views + lower + upper", flush=True)
    print("Validation only: diagnostic aggregation and visualization are not distributed.")


if __name__ == "__main__":
    main()
