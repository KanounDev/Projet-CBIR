"""Benchmark: compare brute-force, VP-Tree, and FAISS search performance.

Generates comparison table for the report (Étape 7.4).

Usage (from C:\\projet-CBIR, environment active):
    python backend/scripts/07_benchmark_indexes.py --sizes 10000 100000 1000000
    python backend/scripts/07_benchmark_indexes.py --test  # quick test on 1000 images

Output: Prints table and saves to data/benchmark_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.distances import METRICS
from app.indexing.vptree import VPTree
from app.indexing.faiss_index import FAISSIndex

DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
INDEXES_DIR = DATA_DIR / "indexes"


def load_subset(name: str, size: int) -> np.ndarray:
    """Load first N vectors from feature file."""
    path = FEATURES_DIR / f"{name}.npy"
    X = np.load(path, mmap_mode="r")
    if size < len(X):
        X = np.asarray(X[:size], dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    return X


def benchmark_brute(X: np.ndarray, queries: np.ndarray, k: int = 10,
                    metric: str = 'euclidean', n_runs: int = 3):
    """Benchmark brute-force search."""
    times = []
    results = []

    for _ in range(n_runs):
        t0 = time.time()
        fn = METRICS[metric]
        for q in queries:
            dists = fn(X, q)
            idx = np.argpartition(dists, k)[:k]
            idx = idx[np.argsort(dists[idx])]
            results.append(idx)
        times.append(time.time() - t0)

    return {
        'avg_time_per_query_ms': (np.mean(times) / len(queries)) * 1000,
        'total_time_s': np.mean(times),
        'results': results[:len(queries)],  # Only keep one set of results
    }


def benchmark_vptree(X: np.ndarray, queries: np.ndarray, k: int = 10,
                     metric: str = 'euclidean', n_runs: int = 3):
    """Benchmark VP-Tree search."""
    times = []
    results = []

    # Build tree
    print("    Building VP-Tree...")
    t0 = time.time()
    tree = VPTree(X, leaf_size=16, distance=metric)
    tree.build(progress=False)
    build_time = time.time() - t0
    print(f"    Build time: {build_time:.1f}s")

    for _ in range(n_runs):
        t0 = time.time()
        for q in queries:
            ids, _ = tree.query(q, k=k)
            results.append(ids)
        times.append(time.time() - t0)

    return {
        'avg_time_per_query_ms': (np.mean(times) / len(queries)) * 1000,
        'total_time_s': np.mean(times),
        'build_time_s': build_time,
        'results': results[:len(queries)],
    }


def benchmark_faiss(X: np.ndarray, queries: np.ndarray, k: int = 10,
                    metric: str = 'euclidean', n_runs: int = 3):
    """Benchmark FAISS search."""
    times = []
    results = []

    # Build index
    print("    Building FAISS index...")
    t0 = time.time()
    idx = FAISSIndex(index_type='flat')
    idx.build(X, metric=metric)
    build_time = time.time() - t0
    print(f"    Build time: {build_time:.1f}s")

    for _ in range(n_runs):
        t0 = time.time()
        for q in queries:
            ids, _ = idx.query(q, k=k)
            results.append(ids)
        times.append(time.time() - t0)

    return {
        'avg_time_per_query_ms': (np.mean(times) / len(queries)) * 1000,
        'total_time_s': np.mean(times),
        'build_time_s': build_time,
        'results': results[:len(queries)],
    }


def compute_recall(brute_results, index_results):
    """Compute recall@k: fraction of true neighbors found."""
    k = len(brute_results[0])
    total = 0
    for brute_ids, idx_ids in zip(brute_results, index_results):
        total += len(set(brute_ids) & set(idx_ids)) / k
    return total / len(brute_results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[10000, 100000, 1000000],
                   help="Dataset sizes to benchmark")
    ap.add_argument("--descriptor", default="edgehistogram", help="Descriptor to use")
    ap.add_argument("--metric", default="euclidean", help="Distance metric")
    ap.add_argument("--k", type=int, default=10, help="Number of neighbors")
    ap.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    ap.add_argument("--n-runs", type=int, default=3, help="Number of runs for timing")
    ap.add_argument("--test", action="store_true", help="Quick test with 1000 vectors")
    ap.add_argument("--no-vptree", action="store_true", help="Skip VP-Tree (slow for large N)")
    args = ap.parse_args()

    if args.test:
        args.sizes = [1000]
        args.n_queries = 10

    print(f"Benchmark settings:")
    print(f"  Descriptor: {args.descriptor}")
    print(f"  Metric: {args.metric}")
    print(f"  k: {args.k}")
    print(f"  Queries: {args.n_queries}")
    print(f"  Dataset sizes: {args.sizes}")

    results = []

    for size in args.sizes:
        print(f"\n{'='*70}")
        print(f"Dataset size: {size:,} vectors")
        print(f"{'='*70}")

        # Load data
        print("  Loading features...")
        X = load_subset(args.descriptor, size)
        n, d = X.shape
        print(f"    Shape: {n} x {d}")

        # Generate queries (random from dataset)
        rng = np.random.default_rng(42)
        query_ids = rng.choice(n, args.n_queries, replace=False)
        queries = X[query_ids]
        print(f"  Generated {args.n_queries} random queries")

        row = {
            'size': size,
            'descriptor': args.descriptor,
            'metric': args.metric,
            'dimension': d,
            'n_queries': args.n_queries,
        }

        # Brute force
        print("  Benchmarking brute-force...")
        brute = benchmark_brute(X, queries, args.k, args.metric, args.n_runs)
        print(f"    Avg query time: {brute['avg_time_per_query_ms']:.2f}ms")
        row['brute'] = brute

        # VP-Tree (skip if too large)
        if not args.no_vptree and size <= 100000:
            print("  Benchmarking VP-Tree...")
            vptree = benchmark_vptree(X, queries, args.k, args.metric, args.n_runs)
            print(f"    Avg query time: {vptree['avg_time_per_query_ms']:.2f}ms")
            vptree['recall@k'] = compute_recall(brute['results'], vptree['results'])
            print(f"    Recall@{args.k}: {vptree['recall@k']:.2%}")
            row['vptree'] = vptree
        else:
            print("  Skipping VP-Tree (size > 100k)")
            row['vptree'] = None

        # FAISS
        print("  Benchmarking FAISS...")
        faiss = benchmark_faiss(X, queries, args.k, args.metric, args.n_runs)
        print(f"    Avg query time: {faiss['avg_time_per_query_ms']:.2f}ms")
        faiss['recall@k'] = compute_recall(brute['results'], faiss['results'])
        print(f"    Recall@{args.k}: {faiss['recall@k']:.2%}")
        row['faiss'] = faiss

        results.append(row)

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (for report)")
    print(f"{'='*70}\n")

    print(f"{'Size':>10} | {'Method':<12} | {'Query (ms)':>12} | {'Build (s)':>10} | {'Recall@k':>10}")
    print("-" * 70)

    for row in results:
        size = row['size']
        brute = row['brute']
        print(f"{size:>10,} | {'Brute-force':<12} | {brute['avg_time_per_query_ms']:>12.2f} | {'N/A':>10} | {'100%':>10}")

        if row['vptree']:
            vptree = row['vptree']
            print(f"{size:>10,} | {'VP-Tree':<12} | {vptree['avg_time_per_query_ms']:>12.2f} | {vptree['build_time_s']:>10.1f} | {vptree['recall@k']:>9.1%}")

        if row['faiss']:
            faiss = row['faiss']
            print(f"{size:>10,} | {'FAISS':<12} | {faiss['avg_time_per_query_ms']:>12.2f} | {faiss['build_time_s']:>10.1f} | {faiss['recall@k']:>9.1%}")

        print()

    # Save results
    out_file = DATA_DIR / "benchmark_results.json"

    # Clean up results for JSON serialization
    clean_results = []
    for row in results:
        clean_row = {
            'size': row['size'],
            'descriptor': row['descriptor'],
            'metric': row['metric'],
            'brute_avg_query_ms': row['brute']['avg_time_per_query_ms'],
        }
        if row['vptree']:
            clean_row['vptree_avg_query_ms'] = row['vptree']['avg_time_per_query_ms']
            clean_row['vptree_build_s'] = row['vptree']['build_time_s']
            clean_row['vptree_recall'] = row['vptree']['recall@k']
        if row['faiss']:
            clean_row['faiss_avg_query_ms'] = row['faiss']['avg_time_per_query_ms']
            clean_row['faiss_build_s'] = row['faiss']['build_time_s']
            clean_row['faiss_recall'] = row['faiss']['recall@k']
        clean_results.append(clean_row)

    with open(out_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
