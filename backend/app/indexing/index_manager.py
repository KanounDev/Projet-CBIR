"""Index manager: loads, caches, and dispatches to the right index.

Provides a unified interface for:
- VP-Tree (for demonstration, moderate datasets)
- FAISS (for production, large datasets)
- Brute force fallback (for correctness verification)
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.indexing.distances import METRICS
from app.indexing.vptree import VPTree
from app.indexing.faiss_index import FAISSIndex


class IndexManager:
    """Manages indexes for multiple descriptors.

    Automatically selects the best available index:
    1. FAISS if available (fastest for 1M+ vectors)
    2. VP-Tree if available (good for demonstration)
    3. Brute force fallback

    Indexes are loaded lazily and cached in memory.
    """

    def __init__(self, data_dir: Optional[Path] = None, use_faiss: bool = True):
        """
        Args:
            data_dir: Directory containing features/ and indexes/ subdirs
            use_faiss: Whether to use FAISS (set False for pure Python demo)
        """
        self.data_dir = Path(data_dir) if data_dir else Path(os.environ.get("DATA_DIR", "data"))
        self.feat_dir = self.data_dir / "features"
        self.idx_dir = self.data_dir / "indexes"
        self.idx_dir.mkdir(parents=True, exist_ok=True)

        self.use_faiss = use_faiss
        self._faiss: Dict[str, FAISSIndex] = {}
        self._vptree: Dict[str, VPTree] = {}
        self._features: Dict[str, np.ndarray] = {}  # memmap

    def _feat_path(self, name: str) -> Path:
        """Path to feature .npy file."""
        return self.feat_dir / f"{name}.npy"

    def _faiss_path(self, name: str, metric: str) -> Path:
        """Path to FAISS index file."""
        return self.idx_dir / f"{name}_{metric}.faiss"

    def _vptree_path(self, name: str, metric: str) -> Path:
        """Path to VP-Tree index file."""
        return self.idx_dir / f"{name}_{metric}.vptree"

    def load_features(self, name: str) -> np.ndarray:
        """Load features as memory-mapped array."""
        if name not in self._features:
            path = self._feat_path(name)
            if not path.exists():
                raise FileNotFoundError(f"Features not found: {path}")
            self._features[name] = np.load(path, mmap_mode="r")
        return self._features[name]

    def get_index_type(self, name: str, metric: str = 'euclidean') -> str:
        """Determine which index to use for given descriptor and metric."""
        if self.use_faiss and self._faiss_path(name, metric).exists():
            return 'faiss'
        elif self._vptree_path(name, metric).exists():
            return 'vptree'
        else:
            return 'brute'

    def search(self, name: str, q: np.ndarray, k: int = 10,
               metric: str = 'euclidean', exclude_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            name: Descriptor name (e.g., 'edgehistogram')
            q: Query vector
            k: Number of results
            metric: Distance metric
            exclude_id: Optional ID to exclude from results

        Returns:
            (indices, distances) sorted by distance
        """
        q = np.asarray(q, dtype=np.float32)
        index_type = self.get_index_type(name, metric)

        if index_type == 'faiss':
            return self._search_faiss(name, q, k, metric, exclude_id)
        elif index_type == 'vptree':
            return self._search_vptree(name, q, k, metric, exclude_id)
        else:
            return self._search_brute(name, q, k, metric, exclude_id)

    def _search_faiss(self, name: str, q: np.ndarray, k: int, metric: str,
                      exclude_id: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index."""
        key = (name, metric)
        if key not in self._faiss:
            idx = FAISSIndex()
            idx.load(str(self._faiss_path(name, metric)), metric=metric)
            self._faiss[key] = idx

        idx = self._faiss[key]
        ids, dists = idx.query(q, k=k + (1 if exclude_id is not None else 0))

        # Exclude if needed
        if exclude_id is not None:
            mask = ids != exclude_id
            ids, dists = ids[mask][:k], dists[mask][:k]

        return ids[:k], dists[:k]

    def _search_vptree(self, name: str, q: np.ndarray, k: int, metric: str,
                       exclude_id: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Search using VP-Tree index."""
        key = (name, metric)
        if key not in self._vptree:
            X = self.load_features(name)
            tree = VPTree.load(str(self._vptree_path(name, metric)), X)
            self._vptree[key] = tree

        tree = self._vptree[key]
        ids, dists = tree.query(q, k=k + (1 if exclude_id is not None else 0))

        # Exclude if needed
        if exclude_id is not None:
            mask = ids != exclude_id
            ids, dists = ids[mask][:k], dists[mask][:k]

        return ids[:k], dists[:k]

    def _search_brute(self, name: str, q: np.ndarray, k: int, metric: str,
                      exclude_id: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Brute force search (fallback)."""
        X = self.load_features(name)
        dists = METRICS[metric](X, q)

        if exclude_id is not None:
            dists[exclude_id] = np.inf

        k = min(k, len(dists) - 1)
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]

        return idx, dists[idx]

    def build_all_indexes(self, descriptors: List[str] = None,
                          metrics: List[str] = None,
                          use_vptree: bool = True,
                          use_faiss: bool = True,
                          vptree_max_size: int = 100000,
                          progress: bool = True):
        """Build indexes for all descriptors.

        Args:
            descriptors: List of descriptor names (None = all found)
            metrics: List of metrics (None = ['euclidean'])
            use_vptree: Build VP-Tree for subsets
            use_faiss: Build FAISS for full dataset
            vptree_max_size: Max vectors for VP-Tree (use FAISS for larger)
            progress: Print progress
        """
        if descriptors is None:
            descriptors = [p.stem for p in self.feat_dir.glob("*.npy")
                          if not p.stem.endswith(('_ids', '_valid', '_test'))]

        metrics = metrics or ['euclidean']

        for name in descriptors:
            X = self.load_features(name)
            n, d = X.shape

            if progress:
                print(f"\n{name}: {n} vectors, dim={d}")

            for metric in metrics:
                # FAISS for full dataset
                if use_faiss and self.use_faiss and n > vptree_max_size:
                    if progress:
                        print(f"  Building FAISS index ({metric})...")
                    t0 = time.time()
                    idx = FAISSIndex(index_type='flat')
                    idx.build(X, metric=metric)
                    idx.save(str(self._faiss_path(name, metric)))
                    if progress:
                        print(f"    Done in {time.time() - t0:.1f}s")

                # VP-Tree for subset (demonstration)
                if use_vptree and n <= vptree_max_size:
                    if progress:
                        print(f"  Building VP-Tree ({metric}) on {n} vectors...")
                    t0 = time.time()
                    X_arr = np.asarray(X, dtype=np.float32)
                    tree = VPTree(X_arr, leaf_size=16, distance=metric)
                    tree.build(progress=progress)
                    tree.save(str(self._vptree_path(name, metric)))
                    if progress:
                        print(f"    Done in {time.time() - t0:.1f}s")

    def benchmark(self, name: str, queries: np.ndarray, k: int = 10,
                  metric: str = 'euclidean', n_runs: int = 3) -> Dict:
        """Compare search methods on a set of queries.

        Returns dict with timing and recall metrics.
        """
        results = {
            'brute': {'times': [], 'results': None},
            'faiss': {'times': [], 'results': None},
            'vptree': {'times': [], 'results': None},
        }

        # Brute force baseline
        for _ in range(n_runs):
            t0 = time.time()
            brute_results = [self._search_brute(name, q, k, metric, None) for q in queries]
            results['brute']['times'].append(time.time() - t0)
        results['brute']['results'] = brute_results
        results['brute']['avg_time'] = np.mean(results['brute']['times'])

        # FAISS
        if self._faiss_path(name, metric).exists():
            for _ in range(n_runs):
                t0 = time.time()
                faiss_results = [self._search_faiss(name, q, k, metric, None) for q in queries]
                results['faiss']['times'].append(time.time() - t0)
            results['faiss']['results'] = faiss_results
            results['faiss']['avg_time'] = np.mean(results['faiss']['times'])

            # Recall@k
            correct = sum(
                1 for i in range(len(queries))
                if set(faiss_results[i][0]) == set(brute_results[i][0])
            )
            results['faiss']['recall@k'] = correct / len(queries)

        # VP-Tree
        if self._vptree_path(name, metric).exists():
            for _ in range(n_runs):
                t0 = time.time()
                vptree_results = [self._search_vptree(name, q, k, metric, None) for q in queries]
                results['vptree']['times'].append(time.time() - t0)
            results['vptree']['results'] = vptree_results
            results['vptree']['avg_time'] = np.mean(results['vptree']['times'])

            # Recall@k
            correct = sum(
                1 for i in range(len(queries))
                if set(vptree_results[i][0]) == set(brute_results[i][0])
            )
            results['vptree']['recall@k'] = correct / len(queries)

        return results
