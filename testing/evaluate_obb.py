"""Evaluate a trained OBB checkpoint on a named dataset split."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", default="training/cfgs/yolov11_obb_data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("outputs/testing"))
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Missing checkpoint: {args.weights}")
    if args.device == "cpu":
        parser.error("DM checkpoints require CUDA: the archived Mamba implementation bypasses Mamba on CPU")
    from ultralytics import YOLO

    model = YOLO(str(args.weights), task="obb")
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                        batch=args.batch, device=args.device, workers=8,
                        project=str(args.output), name=args.weights.stem + "_" + args.split,
                        exist_ok=False, plots=True)
    summary = {"split": args.split, "weights": str(args.weights), "classes": []}
    for position, class_id in enumerate(metrics.box.ap_class_index):
        class_id = int(class_id)
        summary["classes"].append({
            "class_id": class_id, "class_name": model.names[class_id],
            "precision": float(metrics.box.p[position]),
            "recall": float(metrics.box.r[position]),
            "mAP50": float(metrics.box.ap50[position]),
            "mAP50_95": float(metrics.box.ap[position]),
        })
    destination = Path(metrics.save_dir) / "metrics.json"
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
