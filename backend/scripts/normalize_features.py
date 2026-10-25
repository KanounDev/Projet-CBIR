"""Normalize feature vectors and rebuild indexes.

Normalization helps when:
1. Combining multiple descriptors (they need to be on the same scale)
2. Using distance metrics like cosine similarity
3. Features have different ranges/scales

Usage (from C:\\projet-CBIR, environment active):
    # Normalize and save as new files
    python backend/scripts/normalize_features.py --method l2

    # Rebuild indexes with normalized features
    python backend/scripts/normalize_features.py --method l2 --rebuild-indexes

    # Test on subset first
    python backend/scripts/normalize_features.py --method l2 --test --rebuild-indexes

Normalization methods:
    l2: Unit norm (each vector has length 1) - RECOMMENDED for CBIR
    zscore: Zero mean, unit variance per dimension
    minmax: Scale to [0, 1] range
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"

DESCRIPTORS = ["edgehistogram", "homogeneoustexture"]


def normalize_l2(X: np.ndarray, chunk_size: int = 100000) -> np.ndarray:
    """L2 normalization: make each vector unit length.

    Each row vector is divided by its L2 norm.
    Result: ||x|| = 1 for all vectors.
    """
    print("  Normalizing with L2 (unit norm)...")

    # Process in chunks for memory efficiency
    n = len(X)
    normalized = np.zeros_like(X)

    for i in range(0, n, chunk_size):
        chunk = np.asarray(X[i:i+chunk_size], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-9
        normalized[i:i+chunk_size] = chunk / norms

        if i % (chunk_size * 10) == 0:
            print(f"    Processed {i}/{n} vectors...")

    print(f"    Done. Sample norms: {np.linalg.norm(normalized[:5], axis=1)}")
    return normalized


def normalize_zscore(X: np.ndarray, chunk_size: int = 100000) -> tuple:
    """Z-score normalization: zero mean, unit variance per dimension.

    Returns: (normalized_data, mean, std)
    """
    print("  Normalizing with Z-score (zero mean, unit variance)...")

    # Compute mean and std in chunks
    n = len(X)
    mean = np.zeros(X.shape[1], dtype=np.float32)

    for i in range(0, n, chunk_size):
        chunk = np.asarray(X[i:i+chunk_size], dtype=np.float32)
        mean += chunk.sum(axis=0)
    mean /= n

    # Compute std
    var = np.zeros(X.shape[1], dtype=np.float32)
    for i in range(0, n, chunk_size):
        chunk = np.asarray(X[i:i+chunk_size], dtype=np.float32)
        var += ((chunk - mean) ** 2).sum(axis=0)
    std = np.sqrt(var / n) + 1e-9

    # Normalize
    normalized = np.zeros_like(X)
    for i in range(0, n, chunk_size):
        chunk = np.asarray(X[i:i+chunk_size], dtype=np.float32)
        normalized[i:i+chunk_size] = (chunk - mean) / std
        if i % (chunk_size * 10) == 0:
            print(f"    Processed {i}/{n} vectors...")

    print(f"    Done. Mean range: [{normalized.mean(axis=0).min():.4f}, {normalized.mean(axis=0).max():.4f}]")
    return normalized, mean, std


def normalize_minmax(X: np.ndarray, chunk_size: int = 100000) -> tuple:
    """Min-Max normalization: scale to [0, 1] range.

    Returns: (normalized_data, min_val, max_val)
    """
    print("  Normalizing with Min-Max (scale to [0, 1])...")

    # Compute min and max in chunks
    n = len(X)
    d = X.shape[1]
    min_val = np.full(d, np.inf, dtype=np.float32)
    max_val = np.full(d, -np.inf, dtype=np.float32)

    for i in range(0, n, chunk_size):
        chunk = np.asarray(X[i:i+chunk_size], dtype=np.float32)
        min_val = np.minimum(min_val, chunk.min(axis=0))
        max_val = np.maximum(max_val, chunk.max(axis=0))

    # Normalize
    normalized = np.zeros_like(X)
    for i in range(0, n, chunk_size):
        chunk = np.asarray(X[i:i+chunk_size], dtype=np.float32)
        normalized[i:i+chunk_size] = (chunk - min_val) / (max_val - min_val + 1e-9)
        if i % (chunk_size * 10) == 0:
            print(f"    Processed {i}/{n} vectors...")

    print(f"    Done. Value range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    return normalized, min_val, max_val


def main():
    ap = argparse.ArgumentParser(description="Normalize feature vectors")
    ap.add_argument("--method", choices=['l2', 'zscore', 'minmax'], default='l2',
                   help="Normalization method (default: l2)")
    ap.add_argument("--descriptors", nargs="+", default=DESCRIPTORS,
                   help="Which descriptors to normalize")
    ap.add_argument("--test", action="store_true", help="Process test subset only")
    ap.add_argument(
        "--rebuild-indexes",
        action="store_true",
        help="Rebuild the standard FAISS indexes from raw features (normalization is applied only for cosine)",
    )
    ap.add_argument("--compare", action="store_true",
                   help="Show before/after statistics")
    args = ap.parse_args()

    print(f"Normalization Settings:")
    print(f"  Method: {args.method}")
    print(f"  Descriptors: {args.descriptors}")
    print(f"  Mode: {'Test subset' if args.test else 'Full dataset'}")
    print(f"  Rebuild indexes: {args.rebuild_indexes}")

    for name in args.descriptors:
        suffix = "_test" if args.test else ""
        input_file = FEATURES_DIR / f"{name}{suffix}.npy"
        output_file = FEATURES_DIR / f"{name}{suffix}_normalized.npy"

        if not input_file.exists():
            print(f"\n⚠ Skipping {name}: {input_file} not found")
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {name}")
        print(f"{'='*70}")

        # Load features
        print(f"Loading {input_file}...")
        X = np.load(input_file, mmap_mode="r")
        X = np.asarray(X, dtype=np.float32)
        print(f"  Shape: {X.shape}")

        # Show before statistics
        if args.compare:
            print(f"\n  Before normalization:")
            print(f"    Mean: {X.mean():.4f}, Std: {X.std():.4f}")
            print(f"    Min: {X.min():.4f}, Max: {X.max():.4f}")
            print(f"    L2 norms (first 5): {np.linalg.norm(X[:5], axis=1)}")

        # Normalize
        t0 = time.time()

        if args.method == 'l2':
            normalized = normalize_l2(X)
            params = None
        elif args.method == 'zscore':
            normalized, mean, std = normalize_zscore(X)
            params = {'mean': mean, 'std': std}
        elif args.method == 'minmax':
            normalized, min_val, max_val = normalize_minmax(X)
            params = {'min': min_val, 'max': max_val}

        elapsed = time.time() - t0
        print(f"\n  Normalization completed in {elapsed:.1f}s")

        # Show after statistics
        if args.compare:
            print(f"\n  After normalization:")
            print(f"    Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
            print(f"    Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
            if args.method == 'l2':
                print(f"    L2 norms (first 5): {np.linalg.norm(normalized[:5], axis=1)}")

        # Save normalized features
        print(f"\n  Saving to {output_file}...")
        np.save(output_file, normalized)
        print(f"  ✓ Saved {output_file}")

        # Save normalization parameters for later use
        if params is not None:
            params_file = FEATURES_DIR / f"{name}{suffix}_normalized_params.npz"
            np.savez(params_file, **params)
            print(f"  ✓ Saved parameters to {params_file}")

    # Rebuild indexes if requested
    if args.rebuild_indexes:
        print(f"\n{'='*70}")
        print("Rebuilding standard indexes from the original feature arrays...")
        print(f"{'='*70}")

        import subprocess

        # Build command
        cmd = [
            sys.executable,
            str(ROOT / "backend" / "scripts" / "build_indexes.py"),
            "--faiss",
            "--descriptors", *args.descriptors,
            "--metrics", "euclidean",
            "--force",
        ]

        if args.test:
            cmd.append("--test")

        subprocess.run(cmd, cwd=str(ROOT), check=True)
        print("\nIndexes rebuilt successfully.")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("Normalized features saved to:")
    for name in args.descriptors:
        suffix = "_test" if args.test else ""
        output_file = FEATURES_DIR / f"{name}{suffix}_normalized.npy"
        if output_file.exists():
            print(f"  ✓ {output_file}")
    print("\nNext steps:")
    print("  1. Compare search results with/without normalization:")
    print("     python backend/scripts/search_by_id.py --id 42 --k 10 --save")
    print("  2. If using normalized features, update scripts to load *_normalized.npy")


if __name__ == "__main__":
    main()
