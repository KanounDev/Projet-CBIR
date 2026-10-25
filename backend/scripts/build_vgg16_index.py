"""Deprecated compatibility entry point for the unified index builder.

Use backend/scripts/build_indexes.py for new workflows.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the existing index")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ROOT / "backend" / "scripts" / "build_indexes.py"),
        "--faiss",
        "--descriptors",
        "vgg16",
        "--metrics",
        "euclidean",
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
