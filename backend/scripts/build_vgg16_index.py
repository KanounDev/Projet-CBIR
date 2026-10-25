"""Build FAISS index for VGG16 features.

Usage:
    python build_vgg16_index.py

Creates:
    data/indexes/vgg16_euclidean.faiss
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import faiss

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
FEATURES_PATH = DATA_DIR / "features" / "vgg16_gap512_1M.npy"
INDEX_PATH = DATA_DIR / "indexes" / "vgg16_euclidean.faiss"


def build_vgg16_faiss_index():
    """Build FAISS index for VGG16 features (512-dim, L2 distance)."""

    print("=" * 60)
    print("Building VGG16 FAISS Index")
    print("=" * 60)

    # Check if features exist
    if not FEATURES_PATH.exists():
        print(f"ERROR: VGG16 features not found at {FEATURES_PATH}")
        return False

    # Load features
    print(f"\n1. Loading VGG16 features from {FEATURES_PATH}...")
    features = np.load(FEATURES_PATH, mmap_mode="r")
    print(f"   Shape: {features.shape}")
    print(f"   Dtype: {features.dtype}")
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError(f"Expected VGG16 feature shape (N, 512), got {features.shape}")

    # L2 normalize (important for cosine similarity via L2 distance)
    print("\n2. L2 normalizing and adding features in chunks...")

    # Build FAISS index
    print("\n3. Building FAISS index...")
    n_vectors, dim = features.shape
    print(f"   Vectors: {n_vectors:,}")
    print(f"   Dimensions: {dim}")

    # Use IndexFlatL2 (exact search, L2 distance)
    index = faiss.IndexFlatL2(dim)

    # Add vectors
    chunk_size = 50_000
    for start in range(0, n_vectors, chunk_size):
        end = min(start + chunk_size, n_vectors)
        chunk = np.asarray(features[start:end], dtype=np.float32).copy()
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms[norms == 0] = 1
        chunk /= norms
        index.add(chunk)
        print(f"   Added {end:,}/{n_vectors:,} vectors")
    print(f"   Index total vectors: {index.ntotal:,}")

    # Save index
    print(f"\n4. Saving index to {INDEX_PATH}...")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"   Index size: {INDEX_PATH.stat().st_size / (1024**2):.2f} MB")

    # Test search
    print("\n5. Testing search...")
    query = features[0:1]  # First image
    k = 5
    distances, ids = index.search(query, k)
    print(f"   Query: image 0")
    print(f"   Top {k} IDs: {ids[0]}")
    print(f"   Distances: {distances[0]}")

    print("\n" + "=" * 60)
    print("VGG16 FAISS index built successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = build_vgg16_faiss_index()
    sys.exit(0 if success else 1)
