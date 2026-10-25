"""Step 7 - Build VP-Tree and FAISS indexes for fast nearest neighbor search.

Usage (from C:\\projet-CBIR, environment active):
    # Build a 1000-vector FAISS test index (stored separately from full indexes)
    python backend/scripts/build_indexes.py --test --faiss

    # Build FAISS for full 1M dataset (recommended for production)
    python backend/scripts/build_indexes.py --faiss

    # Build VP-Tree for 50k subset (for demonstration)
    python backend/scripts/build_indexes.py --vptree --limit 50000

    # Build everything
    python backend/scripts/build_indexes.py --all

Output: data/indexes/<descriptor>_<metric>.{faiss,vptree}
FAISS supports euclidean, cosine, and L1 here. Chi-squared remains brute-force only.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.vptree import VPTree
from app.indexing.faiss_index import FAISSIndex
from app.indexing.metrics import FAISS_METRICS, SUPPORTED_METRICS, VP_TREE_METRICS

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
FEATURES_DIR = DATA_DIR / "features"
INDEXES_DIR = DATA_DIR / "indexes"

DESCRIPTORS = {
    "vgg16": 512,
    "edgehistogram": 150,
    "homogeneoustexture": 43,
}
EXPECTED_COUNTS = {name: 1_000_000 for name in DESCRIPTORS}

METRICS = ["euclidean"]


def load_features(name: str, limit: int = None, normalized: bool = False) -> np.ndarray:
    """Load features, optionally limited to first N vectors."""
    suffix = "_test" if limit and limit <= 10000 else ""
    normalized_suffix = "_normalized" if normalized and name != "vgg16" else ""
    path = FEATURES_DIR / f"{name}{suffix}{normalized_suffix}.npy"

    if not path.is_file() and suffix:
        path = FEATURES_DIR / f"{name}{normalized_suffix}.npy"
    if not path.is_file() and normalized_suffix:
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


def build_vptree(
    name: str,
    X: np.ndarray,
    metric: str,
    progress: bool = True,
    force: bool = False,
    indexes_dir: Path = INDEXES_DIR,
    source_transform: str = "none",
):
    """Build and save VP-Tree index."""
    if name not in VP_TREE_METRICS or metric not in VP_TREE_METRICS[name]:
        raise ValueError(f"VP-Tree does not support {metric} distance for {name}")
    n, d = X.shape
    indexes_dir.mkdir(parents=True, exist_ok=True)
    out_path = indexes_dir / f"{name}_{metric}.vptree"
    manifest_path = out_path.with_suffix(out_path.suffix + ".json")
    if source_transform not in {"none", "l2_normalized"}:
        raise ValueError(f"Unsupported VP-Tree feature transform: {source_transform}")
    if out_path.is_file() and out_path.stat().st_size and not force:
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}
        if (
            existing_manifest.get("format_version") == 1
            and existing_manifest.get("descriptor") == name
            and existing_manifest.get("metric") == metric
            and existing_manifest.get("source_transform") == source_transform
            and existing_manifest.get("dimension") == d
            and existing_manifest.get("vector_count") == n
        ):
            print(f"    Keeping current VP-Tree: {out_path}")
            return 0.0
        print(f"    Rebuilding legacy or incompatible VP-Tree: {out_path}")

    print(f"\n  Building VP-Tree for {name} ({n} vectors, {metric})...")
    t0 = time.time()

    tree = VPTree(X, leaf_size=16, distance=metric)
    tree.build(progress=progress)
    temporary_path = out_path.with_name(out_path.name + ".tmp")
    tree.save(str(temporary_path))
    os.replace(temporary_path, out_path)
    manifest = {
        "format_version": 1,
        "descriptor": name,
        "metric": metric,
        "source_transform": source_transform,
        "dimension": d,
        "vector_count": n,
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)

    elapsed = time.time() - t0
    print(f"    Saved {out_path} in {elapsed:.1f}s")

    # Test query
    t0 = time.time()
    ids, dists = tree.query(X[0], k=min(5, len(X)))
    query_time = (time.time() - t0) * 1000
    print(f"    Sample query: {query_time:.2f}ms, top-5 ids: {ids[:5].tolist()}")

    return elapsed


def build_faiss(
    name: str,
    X: np.ndarray,
    metric: str,
    force: bool = False,
    indexes_dir: Path = INDEXES_DIR,
    expected_count: int = None,
):
    """Build and save FAISS index."""
    n, d = X.shape
    if expected_count is None:
        expected_count = EXPECTED_COUNTS[name]
    if n > expected_count:
        raise ValueError(f"{name} index has {n} vectors; expected at most {expected_count}")
    subset_suffix = f"_subset{n}" if n != expected_count else ""
    indexes_dir.mkdir(parents=True, exist_ok=True)
    out_path = indexes_dir / f"{name}_{metric}{subset_suffix}.faiss"
    manifest_path = out_path.with_suffix(out_path.suffix + ".json")
    if metric not in FAISS_METRICS[name]:
        raise ValueError(
            f"FAISS metric '{metric}' is unsupported for {name}; supported metrics: "
            f"{', '.join(sorted(FAISS_METRICS[name]))}"
        )
    if out_path.is_file() and out_path.stat().st_size and not force:
        if metric == "euclidean":
            print(f"    Keeping existing Euclidean index (use --force to rebuild): {out_path}")
            return 0.0
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}
        expected_transform = "l2_normalized" if metric == "cosine" else "none"
        if (
            existing_manifest.get("format_version") == 1
            and existing_manifest.get("descriptor") == name
            and existing_manifest.get("metric") == metric
            and existing_manifest.get("source_transform") == expected_transform
            and existing_manifest.get("dimension") == d
            and existing_manifest.get("vector_count") == n
        ):
            print(f"    Keeping current index: {out_path}")
            return 0.0
        print(f"    Rebuilding legacy or incompatible index: {out_path}")

    print(f"\n  Building FAISS index for {name} ({n} vectors, {metric})...")
    t0 = time.time()

    idx = FAISSIndex(index_type="flat")
    idx.build(X, metric=metric)
    temporary_path = out_path.with_name(out_path.name + ".tmp")
    idx.save(str(temporary_path))
    os.replace(temporary_path, out_path)
    manifest = {
        "format_version": 1,
        "descriptor": name,
        "metric": metric,
        "source_transform": "l2_normalized" if metric == "cosine" else "none",
        "dimension": d,
        "vector_count": n,
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)

    elapsed = time.time() - t0
    print(f"    Saved {out_path} in {elapsed:.1f}s")

    # Test query
    t0 = time.time()
    ids, dists = idx.query(X[0], k=min(5, len(X)))
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
    ap.add_argument("--descriptors", nargs="+", choices=DESCRIPTORS, default=list(DESCRIPTORS),
                   help="Which descriptors to index")
    ap.add_argument("--metrics", nargs="+", default=METRICS,
                   help="Distance metrics to use")
    ap.add_argument("--force", action="store_true", help="Replace existing index files")
    args = ap.parse_args()

    if not (args.faiss or args.vptree or args.all):
        args.faiss = True

    known_metrics = set().union(*SUPPORTED_METRICS.values())
    unknown_metrics = set(args.metrics) - known_metrics
    if unknown_metrics:
        ap.error(
            f"Unknown metrics: {', '.join(sorted(unknown_metrics))}. "
            f"Supported metrics: {', '.join(sorted(known_metrics))}."
        )

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {INDEXES_DIR}")

    total_time = 0
    stats = []

    for name in args.descriptors:
        if (args.faiss or args.all) and set(args.metrics) - FAISS_METRICS[name]:
            invalid_metrics = set(args.metrics) - FAISS_METRICS[name]
            ap.error(
                f"FAISS does not support {', '.join(sorted(invalid_metrics))} "
                f"for {name}; use brute-force for those metrics."
            )
        if (args.vptree or args.all) and set(args.metrics) - VP_TREE_METRICS[name]:
            invalid_metrics = set(args.metrics) - VP_TREE_METRICS[name]
            ap.error(
                f"VP-Tree does not support {', '.join(sorted(invalid_metrics))} "
                f"for {name}."
            )

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
                    t = build_vptree(
                        name,
                        X,
                        metric,
                        progress=True,
                        force=args.force,
                    )
                    stats.append(("VP-Tree", name, metric, len(X), t))
                else:
                    print(f"\n  Skipping VP-Tree (too large: {len(X)} > 100k)")

            # FAISS (for full dataset, production)
            if args.faiss or args.all:
                if name == "vgg16" and metric == "cosine":
                    print("  Skipping separate VGG16 cosine index; use its normalized Euclidean index.")
                    continue
                t = build_faiss(
                    name,
                    X,
                    metric,
                    force=args.force,
                    expected_count=EXPECTED_COUNTS[name],
                )
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
