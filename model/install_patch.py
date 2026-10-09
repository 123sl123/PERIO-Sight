"""Install the four DM-YOLO overlay files into an explicit framework checkout."""

import argparse
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True,
                        help="Path to ultralytics/ containing nn/ and __init__.py")
    args = parser.parse_args()
    target = args.package_dir.resolve()
    init = target / "__init__.py"
    if not init.is_file() or not (target / "nn/tasks.py").is_file():
        parser.error("--package-dir must point to a complete Ultralytics package")
    version = re.search(r'__version__\s*=\s*[\"\']([^\"\']+)', init.read_text())
    if version is None or version[1] != "8.4.146":
        parser.error("This snapshot requires Ultralytics 8.4.146; use a separate matching checkout")
    overlay = Path(__file__).parent / "ultralytics_patch"
    for source in sorted(overlay.rglob("*.py")):
        destination = target / source.relative_to(overlay)
        if destination.is_file() and destination.read_bytes() != source.read_bytes():
            backup = destination.with_suffix(".py.before_dm")
            if not backup.exists():
                shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(destination)


if __name__ == "__main__":
    main()
