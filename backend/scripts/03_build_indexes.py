"""Step 7 - Build VP-Tree and FAISS indexes for fast nearest neighbor search.

Usage (from C:\\projet-CBIR, environment active):
    # Build indexes for test subset (1000 images)
    python backend/scripts/03_build_indexes.py --test

    # Build FAISS for full 1M dataset (recommended for production)
    python backend/scripts/03_build_indexes.py --faiss

    # Build VP-Tree for 100k subset (for demonstration)
    python backend/scripts/03_build_indexes.py --vptree --limit 100000

    # Build everything
    python backend/scripts/03_build_indexes.py --all

Output: data/indexes/<descriptor>_<metric>.{faiss,vptree}
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.vptree import VPTree
from app.indexing.faiss_index import FAISSIndex

DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
INDEXES_DIR = DATA_DIR / "indexes"

DESCRIPTORS = {
    "edgehistogram": 150,
    "homogeneoustexture": 43,
}

METRICS = ["euclidean"]  # Add more as needed: "l1", "cosine", "chi2"


def load_features(name: str, limit: int = None) -> np.ndarray:
    """Load features, optionally limited to first N vectors."""
    suffix = "_test" if limit and limit <= 10000 else ""
    path = FEATURES_DIR / f"{name}{suffix}.npy"

    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    print(f"Loading {path}...")
    X = np.load(path, mmap_mode="r")

    if limit and limit < len(X):
        print(f"  Limiting to first {limit} vectors")
        X = np.asarray(X[:limit], dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    return X


def build_vptree(name: str, X: np.ndarray, metric: str, progress: bool = True):
    """Build and save VP-Tree index."""
    n, d = X.shape
    out_path = INDEXES_DIR / f"{name}_{metric}.vptree"

    print(f"\n  Building VP-Tree for {name} ({n} vectors, {metric})...")
    t0 = time.time()

    tree = VPTree(X, leaf_size=16, distance=metric)
    tree.build(progress=progress)
    tree.save(str(out_path))

    elapsed = time.time() - t0
    print(f"    Saved {out_path} in {elapsed:.1f}s")

    # Test query
    t0 = time.time()
    ids, dists = tree.query(X[0], k=5)
    query_time = (time.time() - t0) * 1000
    print(f"    Sample query: {query_time:.2f}ms, top-5 ids: {ids[:5].tolist()}")

    return elapsed


def build_faiss(name: str, X: np.ndarray, metric: str):
    """Build and save FAISS index."""
    n, d = X.shape
    out_path = INDEXES_DIR / f"{name}_{metric}.faiss"

    print(f"\n  Building FAISS index for {name} ({n} vectors, {metric})...")
    t0 = time.time()

    idx = FAISSIndex(index_type='flat')
    idx.build(X, metric=metric)
    idx.save(str(out_path))

    elapsed = time.time() - t0
    print(f"    Saved {out_path} in {elapsed:.1f}s")

    # Test query
    t0 = time.time()
    ids, dists = idx.query(X[0], k=5)
    query_time = (time.time() - t0) * 1000
    print(f"    Sample query: {query_time:.2f}ms, top-5 ids: {ids[:5].tolist()}")

    return elapsed


def main():
    ap = argparse.ArgumentParser(description="Build indexes for CBIR")
    ap.add_argument("--test", action="store_true", help="Build on test subset (1000 images)")
    ap.add_argument("--faiss", action="store_true", help="Build FAISS indexes for full dataset")
    ap.add_argument("--vptree", action="store_true", help="Build VP-Tree indexes")
    ap.add_argument("--all", action="store_true", help="Build all index types")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of vectors")
    ap.add_argument("--descriptors", nargs="+", default=list(DESCRIPTORS.keys()),
                   help="Which descriptors to index")
    ap.add_argument("--metrics", nargs="+", default=METRICS,
                   help="Distance metrics to use")
    args = ap.parse_args()

    # Default: build FAISS if no specific option given
    if not (args.test or args.faiss or args.vptree or args.all):
        args.faiss = True

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {INDEXES_DIR}")

    total_time = 0
    stats = []

    for name in args.descriptors:
        if name not in DESCRIPTORS:
            print(f"Skipping unknown descriptor: {name}")
            continue

        print(f"\n{'='*60}")
        print(f"Descriptor: {name}")

        # Determine limit
        limit = args.limit
        if args.test:
            limit = 1000

        # Load features
        t0 = time.time()
        X = load_features(name, limit)
        load_time = time.time() - t0
        print(f"  Loaded {X.shape[0]} vectors in {load_time:.1f}s")

        for metric in args.metrics:
            # VP-Tree (for moderate-sized subsets, demonstration)
            if args.vptree or args.all:
                if len(X) <= 100000:  # VP-Tree practical limit
                    t = build_vptree(name, X, metric, progress=True)
                    stats.append(("VP-Tree", name, metric, len(X), t))
                else:
                    print(f"\n  Skipping VP-Tree (too large: {len(X)} > 100k)")

            # FAISS (for full dataset, production)
            if args.faiss or args.all:
                t = build_faiss(name, X, metric)
                stats.append(("FAISS", name, metric, len(X), t))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Index':<10} {'Descriptor':<20} {'Metric':<12} {'N':>10} {'Time (s)':>10}")
    print("-" * 60)
    for idx_type, name, metric, n, t in stats:
        print(f"{idx_type:<10} {name:<20} {metric:<12} {n:>10} {t:>10.1f}")

    total = sum(t for _, _, _, _, t in stats)
    print("-" * 60)
    print(f"Total time: {total:.1f}s")


if __name__ == "__main__":
    main()
