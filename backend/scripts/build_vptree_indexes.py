"""Deprecated compatibility entry point for the 50k VP-Tree demonstration.

Use backend/scripts/build_indexes.py --vptree --limit 50000 for new workflows.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.indexing.metrics import VP_TREE_DEMO_SIZE

BUILD_SCRIPT = ROOT / "backend" / "scripts" / "build_indexes.py"
CONFIGURATIONS = {
    "vgg16": ["euclidean"],
    "edgehistogram": ["euclidean"],
    "homogeneoustexture": ["euclidean"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing VP-Trees")
    args = parser.parse_args()

    for descriptor, metrics in CONFIGURATIONS.items():
        command = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--vptree",
            "--limit",
            str(VP_TREE_DEMO_SIZE),
            "--descriptors",
            descriptor,
            "--metrics",
            *metrics,
        ]
        if args.force:
            command.append("--force")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
