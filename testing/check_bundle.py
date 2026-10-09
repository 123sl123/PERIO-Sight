"""Check English-only text, Python syntax and the complete 34-view example."""

import ast
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")


def main():
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if CJK.search(str(path.relative_to(ROOT))):
            raise ValueError(f"Non-English filename: {path}")
        if path.is_file() and path.suffix not in {".png", ".tiff", ".pyc"}:
            content = path.read_text(encoding="utf-8-sig")
            if CJK.search(content):
                raise ValueError(f"Non-English text: {path}")
            if path.suffix == ".json" and CJK.search(str(json.loads(content))):
                raise ValueError(f"Non-English JSON value: {path}")
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    spec = importlib.util.spec_from_file_location("sample_runner", ROOT / "inference/run_example.py")
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    images, annotations = module.collect_inputs(ROOT / "DATA/D0064")
    for image, annotation in zip(images, annotations):
        width, height = struct.unpack(">II", image.read_bytes()[16:24])
        data = json.loads(annotation.read_text(encoding="utf-8"))
        if (data["imageWidth"], data["imageHeight"]) != (width, height):
            raise ValueError(f"Image dimensions disagree: {annotation}")
        if any(len(shape["points"]) < 3 for shape in data["shapes"]):
            raise ValueError(f"Invalid polygon: {annotation}")
    print("PASS: English-only filenames/text, Python syntax, 34 ordered PNG/JSON pairs and matching dimensions")


if __name__ == "__main__":
    main()
