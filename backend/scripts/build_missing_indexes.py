"""Deprecated compatibility entry point for optional MPEG-7 FAISS indexes.

Use build_indexes.py directly for new workflows. Chi-squared remains a
brute-force metric and is intentionally not built as a FAISS index.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild optional indexes even when index files already exist",
    )
    args = parser.parse_args()

    print(
        "Deprecated: use backend/scripts/build_indexes.py; "
        "building MPEG-7 cosine and L1 indexes (chi2 is brute-force only)."
    )
    command = [
        sys.executable,
        str(ROOT / "backend" / "scripts" / "build_indexes.py"),
        "--faiss",
        "--descriptors",
        "edgehistogram",
        "homogeneoustexture",
        "--metrics",
        "cosine",
        "l1",
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
