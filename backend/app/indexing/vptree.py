"""VP-Tree (Vantage Point Tree) for nearest neighbor search.

A VP-Tree partitions the space using vantage points and distance thresholds,
enabling O(log N) search complexity instead of O(N) for brute force.

Reference: Yianilos, "Data Structures and Algorithms for Nearest Neighbor Search"
https://en.wikipedia.org/wiki/Vantage-point_tree

This implementation is pure Python/NumPy, suitable for demonstration on
moderate-sized subsets (10k-100k). For 1M vectors, use FAISS instead.
"""
import heapq
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class VPTreeNode:
    """A node in the VP-Tree."""

    def __init__(self):
        self.vantage_idx: Optional[int] = None  # index of vantage point
        self.threshold: float = 0.0  # median distance
        self.left: Optional['VPTreeNode'] = None  # points with dist <= threshold
        self.right: Optional['VPTreeNode'] = None  # points with dist > threshold
        self.indices: Optional[np.ndarray] = None  # leaf: indices of points
        self.is_leaf: bool = False


class VPTree:
    """VP-Tree for approximate nearest neighbor search.

    Attributes:
        X: Data matrix (N, d), kept as memory-mapped array
        leaf_size: Maximum number of points in a leaf node
        distance: Distance function ('euclidean', 'l1', 'cosine', 'chi2')
    """

    DISTANCES = {
        'euclidean': lambda a, b: np.sqrt(np.sum((a - b) ** 2)),
        'l1': lambda a, b: np.sum(np.abs(a - b)),
        'cosine': lambda a, b: 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9),
        'chi2': lambda a, b: 0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)),
    }

    def __init__(self, X: np.ndarray, leaf_size: int = 16, distance: str = 'euclidean'):
        """
        Args:
            X: Data matrix (N, d), can be memory-mapped
            leaf_size: Stop splitting when node has <= leaf_size points
            distance: Distance metric name
        """
        self.X = np.asarray(X) if not isinstance(X, np.memmap) else X
        self.leaf_size = leaf_size
        self.distance = distance
        self._dist_fn = self.DISTANCES[distance]
        self.root: Optional[VPTreeNode] = None
        self.n, self.d = X.shape

    def _distances_from(self, vantage_idx: int, indices: np.ndarray) -> np.ndarray:
        """Compute distances from vantage point to all points in indices."""
        v = np.asarray(self.X[vantage_idx], dtype=np.float32)
        # Vectorized distance computation
        points = np.asarray(self.X[indices], dtype=np.float32)

        if self.distance == 'euclidean':
            diff = points - v
            return np.sqrt(np.sum(diff * diff, axis=1))
        elif self.distance == 'l1':
            return np.sum(np.abs(points - v), axis=1)
        elif self.distance == 'cosine':
            norms = np.linalg.norm(points, axis=1) * np.linalg.norm(v)
            return 1.0 - (points @ v) / (norms + 1e-9)
        elif self.distance == 'chi2':
            diff = points - v
            return 0.5 * np.sum(diff * diff / (points + v + 1e-9), axis=1)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

    def build(self, indices: Optional[np.ndarray] = None, progress: bool = False) -> VPTreeNode:
        """Build the VP-Tree recursively.

        Args:
            indices: Indices of points to include (None = all)
            progress: Print progress during construction

        Returns:
            Root node of the tree
        """
        if indices is None:
            indices = np.arange(self.n, dtype=np.int64)

        self.root = self._build_recursive(indices, depth=0, progress=progress)
        return self.root

    def _build_recursive(self, indices: np.ndarray, depth: int, progress: bool) -> VPTreeNode:
        """Recursively build tree for given indices."""
        node = VPTreeNode()

        # Leaf condition
        if len(indices) <= self.leaf_size:
            node.is_leaf = True
            node.indices = indices
            return node

        # Choose vantage point (here: random)
        rng = np.random.default_rng(42 + depth)
        vp_local_idx = rng.integers(0, len(indices))
        node.vantage_idx = int(indices[vp_local_idx])

        # Compute distances from vantage point to all points
        dists = self._distances_from(node.vantage_idx, indices)

        # Median as threshold
        node.threshold = float(np.median(dists))

        # Split points
        left_mask = dists <= node.threshold
        right_mask = ~left_mask

        # Avoid degenerate splits
        if left_mask.all() or right_mask.all():
            # All points equidistant (or nearly), make leaf
            node.is_leaf = True
            node.indices = indices
            return node

        # Remove vantage point from children
        left_indices = indices[left_mask & (indices != node.vantage_idx)]
        right_indices = indices[right_mask & (indices != node.vantage_idx)]

        if progress and depth <= 3:
            print(f"  Depth {depth}: {len(left_indices)} left, {len(right_indices)} right")

        # Recurse
        if len(left_indices) > 0:
            node.left = self._build_recursive(left_indices, depth + 1, progress)
        if len(right_indices) > 0:
            node.right = self._build_recursive(right_indices, depth + 1, progress)

        return node

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors to query vector.

        Args:
            q: Query vector (d,)
            k: Number of neighbors

        Returns:
            (indices, distances) sorted by distance (nearest first)
        """
        if self.root is None:
            raise RuntimeError("Tree not built. Call build() first.")

        q = np.asarray(q, dtype=np.float32)
        heap = []  # max-heap as (-dist, idx)
        tau = np.inf  # current k-th distance

        self._search_recursive(self.root, q, k, heap, tau)

        # Extract results from heap
        results = sorted([(-d, idx) for d, idx in heap])
        indices = np.array([idx for _, idx in results[:k]], dtype=np.int64)
        distances = np.array([-d for d, _ in results[:k]], dtype=np.float32)

        return indices, distances

    def _search_recursive(self, node: VPTreeNode, q: np.ndarray, k: int,
                          heap: List, tau: float) -> float:
        """Recursive search with pruning."""
        if node.is_leaf:
            # Brute force within leaf
            for idx in node.indices:
                d = self._dist_fn(np.asarray(self.X[idx], dtype=np.float32), q)
                if len(heap) < k:
                    heapq.heappush(heap, (-d, int(idx)))
                    tau = -heap[0][0]
                elif d < -heap[0][0]:
                    heapq.heapreplace(heap, (-d, int(idx)))
                    tau = -heap[0][0]
            return tau

        # Distance to vantage point
        d = self._dist_fn(np.asarray(self.X[node.vantage_idx], dtype=np.float32), q)

        # Add vantage point to candidates
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.vantage_idx))
            tau = -heap[0][0]
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.vantage_idx))
            tau = -heap[0][0]

        # Decide which subtree(s) to search
        if d < node.threshold:
            # Query is inside the ball - search left first
            if node.left and d - tau <= node.threshold:
                tau = self._search_recursive(node.left, q, k, heap, tau)
            if node.right and d + tau >= node.threshold:
                tau = self._search_recursive(node.right, q, k, heap, tau)
        else:
            # Query is outside - search right first
            if node.right and d + tau >= node.threshold:
                tau = self._search_recursive(node.right, q, k, heap, tau)
            if node.left and d - tau <= node.threshold:
                tau = self._search_recursive(node.left, q, k, heap, tau)

        return tau

    def save(self, path: str):
        """Save tree structure (not the data matrix)."""
        data = {
            'root': self.root,
            'leaf_size': self.leaf_size,
            'distance': self.distance,
            'n': self.n,
            'd': self.d,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, X: np.ndarray) -> 'VPTree':
        """Load tree and attach data matrix."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tree = cls(X, leaf_size=data['leaf_size'], distance=data['distance'])
        tree.root = data['root']
        tree.n = data['n']
        tree.d = data['d']
        return tree
