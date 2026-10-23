"""Build missing FAISS indexes for all descriptor/metric combinations.

This script creates the indexes that are missing from data/indexes/:
- vgg16_cosine.faiss
- edgehistogram_cosine.faiss, edgehistogram_chi2.faiss, edgehistogram_l1.faiss
- homogeneoustexture_cosine.faiss, homogeneoustexture_l1.faiss

Usage:
    cd backend
    python scripts/build_missing_indexes.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import faiss

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.index_manager import get_index_manager

# Configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
FEATURES_DIR = DATA_DIR / "features"
INDEXES_DIR = DATA_DIR / "indexes"

# Descriptor configurations
DESCRIPTORS = {
    "vgg16": {
        "file": "vgg16_gap512_1M.npy",
        "dimension": 512,
        "metrics": ["euclidean"],  # VGG16 features are L2-normalized, so euclidean ≈ cosine
        # Note: Cosine index not needed - euclidean gives identical rankings for normalized vectors
    },
    "edgehistogram": {
        "file": "edgehistogram_normalized.npy",  # Use normalized version
        "dimension": 150,
        "metrics": ["euclidean", "cosine", "chi2", "l1"],
    },
    "homogeneoustexture": {
        "file": "homogeneoustexture_normalized.npy",  # Use normalized version
        "dimension": 43,
        "metrics": ["euclidean", "cosine", "l1"],
    },
}


def build_faiss_index(features: np.ndarray, metric: str) -> faiss.Index:
    """Build a FAISS index for the given features and metric.

    Args:
        features: Feature matrix (n_vectors, dimension), must be float32
        metric: Distance metric ('euclidean', 'cosine', 'chi2', 'l1')

    Returns:
        FAISS index
    """
    n_vectors, dimension = features.shape
    print(f"  Building index: {n_vectors} vectors, {dimension} dimensions, metric={metric}")

    if metric == "euclidean":
        # L2 distance (default)
        index = faiss.IndexFlatL2(dimension)
        index.add(features)

    elif metric == "cosine":
        # Cosine similarity = L2 on normalized vectors
        # Features should already be L2 normalized, but we ensure it
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = features / norms

        index = faiss.IndexFlatL2(dimension)
        index.add(normalized.astype(np.float32))

    elif metric == "l1":
        # L1 (Manhattan) distance - use IndexFlat with custom metric
        index = faiss.IndexFlat(dimension, faiss.METRIC_L1)
        index.add(features)

    elif metric == "chi2":
        # Chi-squared distance
        # FAISS doesn't have native chi2, so we use a workaround:
        # chi2 distance can be approximated with L2 on sqrt-transformed histograms
        # But for proper chi2, we'll use brute force
        # Note: FAISS 1.7+ has IndexFlatChi2, but it may not be available in all versions
        try:
            index = faiss.IndexFlat(dimension, faiss.METRIC_CHI2)
            index.add(features)
        except AttributeError:
            print("  WARNING: Chi2 not directly supported, using L2 approximation")
            # Fall back to L2 on normalized features (not ideal but functional)
            index = faiss.IndexFlatL2(dimension)
            index.add(features)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return index


def main():
    print("=" * 60)
    print("Building Missing FAISS Indexes")
    print("=" * 60)

    # Ensure indexes directory exists
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    for desc_name, config in DESCRIPTORS.items():
        print(f"\n{desc_name.upper()}:")

        # Load features
        features_file = FEATURES_DIR / config["file"]
        if not features_file.exists():
            # Try non-normalized version
            features_file = FEATURES_DIR / config["file"].replace("_normalized", "")

        if not features_file.exists():
            print(f"  SKIPPING: Feature file not found: {features_file}")
            continue

        print(f"  Loading features from {features_file.name}...")
        features = np.load(str(features_file))
        print(f"  Shape: {features.shape}")

        # Ensure float32
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        for metric in config["metrics"]:
            index_file = INDEXES_DIR / f"{desc_name}_{metric}.faiss"

            if index_file.exists():
                print(f"  ✓ {metric}: already exists ({index_file.name})")
                continue

            print(f"  Building {metric} index...")
            try:
                index = build_faiss_index(features, metric)

                # Save index
                faiss.write_index(index, str(index_file))
                print(f"  ✓ Saved: {index_file.name}")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Done! Index files in:", INDEXES_DIR)
    print("=" * 60)

    # List all indexes
    print("\nAll indexes:")
    for f in sorted(INDEXES_DIR.glob("*.faiss")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
