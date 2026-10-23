"""Build missing VP-Tree indexes for all descriptor/metric combinations.

VP-Tree is suitable for demonstration on moderate-sized subsets (10k-100k vectors).
This script builds VP-Trees for a 50k subset for testing purposes.

Usage:
    cd backend
    python scripts/build_vptree_indexes.py
"""
import os
import sys
from pathlib import Path
import pickle
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.vptree import VPTree

# Configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
FEATURES_DIR = DATA_DIR / "features"
INDEXES_DIR = DATA_DIR / "indexes"

# Use a subset for VP-Tree (faster to build, suitable for demonstration)
SUBSET_SIZE = 50000  # 50k images for VP-Tree demo

# Descriptor configurations
DESCRIPTORS = {
    "vgg16": {
        "file": "vgg16_gap512_1M.npy",
        "dimension": 512,
        "metrics": ["euclidean", "cosine"],
    },
    "edgehistogram": {
        "file": "edgehistogram_normalized.npy",
        "dimension": 150,
        "metrics": ["euclidean", "cosine", "chi2", "l1"],
    },
    "homogeneoustexture": {
        "file": "homogeneoustexture_normalized.npy",
        "dimension": 43,
        "metrics": ["euclidean", "cosine", "l1"],
    },
}


def build_vptree(features: np.ndarray, metric: str, subset_size: int = 50000) -> VPTree:
    """Build a VP-Tree for the given features and metric.

    Args:
        features: Feature matrix (N, d)
        metric: Distance metric
        subset_size: Number of vectors to include (VP-Tree is for demos)

    Returns:
        VPTree instance
    """
    # Use subset for VP-Tree (full 1M is too slow for demo)
    n = min(len(features), subset_size)
    print(f"  Using subset of {n} vectors for VP-Tree...")

    # Build tree
    X_subset = features[:n]
    tree = VPTree(X_subset, leaf_size=16, distance=metric)

    start = time.time()
    tree.build(progress=True)
    elapsed = time.time() - start

    print(f"  Build time: {elapsed:.1f}s")

    return tree


def main():
    print("=" * 60)
    print("Building VP-Tree Indexes (50k subset for demonstration)")
    print("=" * 60)

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    for desc_name, config in DESCRIPTORS.items():
        print(f"\n{desc_name.upper()}:")

        # Load features
        features_file = FEATURES_DIR / config["file"]
        if not features_file.exists():
            features_file = FEATURES_DIR / config["file"].replace("_normalized", "")

        if not features_file.exists():
            print(f"  SKIPPING: Feature file not found: {features_file}")
            continue

        print(f"  Loading features from {features_file.name}...")
        features = np.load(str(features_file), mmap_mode='r')
        print(f"  Shape: {features.shape}")

        for metric in config["metrics"]:
            index_file = INDEXES_DIR / f"{desc_name}_{metric}.vptree"

            if index_file.exists():
                print(f"  OK {metric}: already exists ({index_file.name})")
                continue

            print(f"  Building {metric} VP-Tree...")
            try:
                tree = build_vptree(features, metric, SUBSET_SIZE)

                # Save tree
                tree.save(str(index_file))
                print(f"  OK Saved: {index_file.name}")

            except Exception as e:
                print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("Done! VP-Tree files in:", INDEXES_DIR)
    print("=" * 60)

    # List all VP-Tree indexes
    print("\nAll VP-Tree indexes:")
    for f in sorted(INDEXES_DIR.glob("*.vptree")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")

    print("\nNote: VP-Trees are built on a 50k subset for demonstration.")
    print("For full 1M search, use FAISS indexes instead.")


if __name__ == "__main__":
    main()
