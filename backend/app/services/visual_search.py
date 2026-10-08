"""Brute-force visual similarity search (reference version, Step 6).

Exact but slow: compares the query with all N vectors. Later indexes (VP-Tree, FAISS)
must return the same top-k.

Fusion of several descriptors: for each descriptor the distances are divided by a scale
(median distance between random vectors of that descriptor, so descriptors become
comparable), then combined with a weighted sum.
"""
import os
from pathlib import Path

import numpy as np

from app.indexing.distances import METRICS

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DESCRIPTORS = ("edgehistogram", "homogeneoustexture")


class VisualSearcher:
    def __init__(self, data_dir=DATA_DIR, descriptors=DESCRIPTORS):
        self.feat_dir = Path(data_dir) / "features"
        self.X, self.valid, self._scales = {}, {}, {}
        for name in descriptors:
            X = np.load(self.feat_dir / f"{name}.npy", mmap_mode="r")
            self.X[name] = X
            self.valid[name] = self._load_valid(name, X)
        self.n = len(next(iter(self.X.values())))

    def _load_valid(self, name, X, chunk=100_000):
        """valid[i] is False for all-zero vectors (blank images): excluded from results."""
        path = self.feat_dir / f"{name}_valid.npy"
        if path.exists():
            return np.load(path)
        valid = np.empty(len(X), dtype=bool)
        for s in range(0, len(X), chunk):
            valid[s:s + chunk] = np.asarray(X[s:s + chunk]).any(axis=1)
        np.save(path, valid)
        return valid

    def scale(self, name, metric):
        """Typical distance between two random images (median), used to normalise."""
        key = (name, metric)
        if key not in self._scales:
            rng = np.random.default_rng(0)
            pool = np.flatnonzero(self.valid[name])
            sample = np.sort(rng.choice(pool, min(20_000, len(pool)), replace=False))
            sample_mat = np.asarray(self.X[name][sample], dtype=np.float32)
            queries = rng.choice(pool, 30, replace=False)
            fn = METRICS[metric]
            vals = [fn(sample_mat, np.asarray(self.X[name][i], dtype=np.float32)) for i in queries]
            self._scales[key] = float(np.median(np.concatenate(vals))) or 1.0
        return self._scales[key]

    def search_vectors(self, queries, weights=None, metric="euclidean", k=10, exclude_id=None):
        """queries: {descriptor_name: vector}. weights: {descriptor_name: weight}.
        metric: one name for all descriptors, or {descriptor_name: metric}.
        Returns (ids, fused_distances), best first."""
        weights = weights or {name: 1.0 for name in queries}
        total_w = sum(weights.values())
        fused = np.zeros(self.n, dtype=np.float32)
        ok = np.ones(self.n, dtype=bool)
        for name, w in weights.items():
            m = metric[name] if isinstance(metric, dict) else metric
            d = METRICS[m](self.X[name], queries[name])
            fused += (w / total_w) * d / self.scale(name, m)
            ok &= self.valid[name]
        fused[~ok] = np.inf
        if exclude_id is not None:
            fused[exclude_id] = np.inf
        k = min(k, self.n - 1)
        idx = np.argpartition(fused, k)[:k]
        idx = idx[np.argsort(fused[idx])]
        return idx, fused[idx]

    def search(self, query_id, weights=None, metric="euclidean", k=10, exclude_query=False):
        """Query by an image of the database (id = file number, image = im<id+1>.jpg)."""
        weights = weights or {"edgehistogram": 1.0}
        queries = {n: np.asarray(self.X[n][query_id], dtype=np.float32) for n in weights}
        return self.search_vectors(queries, weights, metric, k, query_id if exclude_query else None)
