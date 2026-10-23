"""Build FAISS index for VGG16 features.

Usage:
    python build_vgg16_index.py

Creates:
    data/indexes/vgg16_euclidean.faiss
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import faiss

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
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
    features = np.load(FEATURES_PATH)
    print(f"   Shape: {features.shape}")
    print(f"   Dtype: {features.dtype}")
    print(f"   Size: {features.nbytes / (1024**3):.2f} GB")

    # Ensure float32
    if features.dtype != np.float32:
        print("   Converting to float32...")
        features = features.astype(np.float32)

    # L2 normalize (important for cosine similarity via L2 distance)
    print("\n2. L2 normalizing features...")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    features = features / norms
    print(f"   Done. Sample norm: {np.linalg.norm(features[0]):.6f}")

    # Build FAISS index
    print("\n3. Building FAISS index...")
    n_vectors, dim = features.shape
    print(f"   Vectors: {n_vectors:,}")
    print(f"   Dimensions: {dim}")

    # Use IndexFlatL2 (exact search, L2 distance)
    index = faiss.IndexFlatL2(dim)

    # Add vectors
    print("   Adding vectors to index...")
    index.add(features)
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
