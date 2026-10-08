"""FAISS-based nearest neighbor search for large-scale vectors.

FAISS (Facebook AI Similarity Search) is optimized for high-dimensional vectors
and millions of points. It uses quantization and GPU acceleration for speed.

For 1M vectors of dimension 150 (Edge Histogram) or 43 (Homogeneous Texture),
FAISS provides sub-millisecond search times vs seconds for brute force.

Reference: https://github.com/facebookresearch/faiss
"""
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    print("WARNING: faiss not installed. Run: pip install faiss-cpu")


class FAISSIndex:
    """FAISS index for fast nearest neighbor search.

    Supports two index types:
    - 'flat': Exact search (brute force, but optimized)
    - 'ivf': Inverted file index (approximate, faster for large N)

    For MIRFLICKR-1M (1M vectors), 'flat' is fast enough (~1ms per query)
    and guarantees exact results. Use 'ivf' for >10M vectors.
    """

    def __init__(self, index_type: str = 'flat', nlist: int = 100, nprobe: int = 10):
        """
        Args:
            index_type: 'flat' for exact, 'ivf' for approximate
            nlist: Number of clusters for IVF (ignored for flat)
            nprobe: Number of clusters to search (ignored for flat)
        """
        if faiss is None:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index: Optional[faiss.Index] = None
        self.n: int = 0
        self.d: int = 0

    def build(self, X: np.ndarray, metric: str = 'euclidean', train_size: int = 100000):
        """Build the index from data matrix.

        Args:
            X: Data matrix (N, d), float32
            metric: 'euclidean' (L2) or 'cosine'
            train_size: Number of points to use for IVF training (ignored for flat)
        """
        X = np.asarray(X, dtype=np.float32)
        self.n, self.d = X.shape

        # Normalize for cosine similarity (then use L2)
        if metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
            X = X / norms
            self._metric = 'cosine'
        else:
            self._metric = 'euclidean'

        if self.index_type == 'flat':
            # Exact search
            if metric == 'cosine':
                # Inner product on normalized vectors = cosine similarity
                # FAISS returns -inner_product, so we negate for distance
                self.index = faiss.IndexFlatIP(self.d)
            else:
                # L2 (Euclidean) distance
                self.index = faiss.IndexFlatL2(self.d)
        elif self.index_type == 'ivf':
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatL2(self.d)
            self.index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)

            # Train on subset
            train_size = min(train_size, self.n)
            rng = np.random.default_rng(42)
            train_idx = rng.choice(self.n, train_size, replace=False)
            train_data = X[train_idx] if isinstance(X, np.memmap) else X[train_idx]
            self.index.train(train_data)

        # Add vectors (in chunks for memory efficiency with memmap)
        chunk_size = 100000
        for i in range(0, self.n, chunk_size):
            chunk = X[i:i + chunk_size]
            if isinstance(chunk, np.memmap):
                chunk = np.asarray(chunk, dtype=np.float32)
            self.index.add(chunk)

        if self.index_type == 'ivf':
            self.index.nprobe = self.nprobe

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors.

        Args:
            q: Query vector (d,) or (n_queries, d)
            k: Number of neighbors

        Returns:
            (indices, distances) where indices is (k,) or (n_queries, k)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # Normalize for cosine
        if self._metric == 'cosine':
            norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
            q = q / norms

        # Search
        distances, indices = self.index.search(q, k)

        # For cosine with IP index, distance is -similarity; convert to 1 - similarity
        if self._metric == 'cosine' and self.index_type == 'flat':
            distances = 1.0 + distances  # distance = 1 - similarity

        if indices.shape[0] == 1:
            return indices[0], distances[0]
        return indices, distances

    def save(self, path: str):
        """Save index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")
        faiss.write_index(self.index, path)

    def load(self, path: str, metric: str = 'euclidean'):
        """Load index from disk."""
        self.index = faiss.read_index(path)
        self.n = self.index.ntotal
        self.d = self.index.d
        self._metric = metric


def build_faiss_index(X: np.ndarray, metric: str = 'euclidean',
                      index_type: str = 'flat') -> FAISSIndex:
    """Convenience function to build a FAISS index.

    Args:
        X: Data matrix (N, d)
        metric: 'euclidean' or 'cosine'
        index_type: 'flat' or 'ivf'

    Returns:
        FAISSIndex ready for queries
    """
    idx = FAISSIndex(index_type=index_type)
    idx.build(X, metric=metric)
    return idx
