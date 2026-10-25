"""Index manager for FAISS and VP-Tree indexes.

Provides a unified interface to:
- Load indexes based on descriptor name and distance metric
- Search for k-nearest neighbors
- Fallback to brute-force if index not available

Supported descriptors:
- vgg16 (512-d)
- edgehistogram (150-d)
- homogeneoustexture (43-d)

Supported distances:
- euclidean (L2)
- cosine
- chi2 (for histograms)
- l1 (manhattan)

Index types:
- faiss: Fast production index for 1M+ vectors
- vptree: Demonstrative index for the 50k-image subset
- brute: Brute-force reference search
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from app.indexing.distances import METRICS
from app.indexing.metrics import (
    FAISS_METRICS,
    SUPPORTED_METRICS,
    VP_TREE_DEMO_SIZE,
    VP_TREE_METRICS,
)
from app.indexing.vptree import VPTree

# Configuration
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))

# Descriptor metadata
DESCRIPTOR_INFO = {
    "vgg16": {
        "file": "vgg16_gap512_1M.npy",
        "dim": 512,
        "valid_file": None,  # VGG16 has no valid mask
    },
    "edgehistogram": {
        "file": "edgehistogram.npy",
        "dim": 150,
        "valid_file": "edgehistogram_valid.npy",
    },
    "homogeneoustexture": {
        "file": "homogeneoustexture.npy",
        "dim": 43,
        "valid_file": "homogeneoustexture_valid.npy",
    },
}

class IndexManager:
    """Manages feature vectors and indexes for visual search."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.features_dir = self.data_dir / "features"
        self.indexes_dir = self.data_dir / "indexes"

        # Loaded features (lazy loading)
        self._features: Dict[str, np.ndarray] = {}
        self._valid: Dict[str, np.ndarray] = {}

        # Loaded indexes (lazy loading)
        self._faiss_indexes: Dict[Tuple[str, str], object] = {}
        self._vptree_indexes: Dict[Tuple[str, str], object] = {}
        self._vptree_transforms: Dict[Tuple[str, str], str] = {}

    def load_features(self, descriptor: str) -> np.ndarray:
        """Load feature vectors for a descriptor (lazy, memoized).

        Args:
            descriptor: Descriptor name (vgg16, edgehistogram, homogeneoustexture)

        Returns:
            Feature matrix (N, d) as memory-mapped array
        """
        if descriptor not in self._features:
            info = DESCRIPTOR_INFO.get(descriptor)
            if not info:
                raise ValueError(f"Unknown descriptor: {descriptor}")

            path = self.features_dir / info["file"]
            if not path.exists():
                raise FileNotFoundError(f"Feature file not found: {path}")

            self._features[descriptor] = np.load(path, mmap_mode="r")

        return self._features[descriptor]

    def load_valid_mask(self, descriptor: str) -> Optional[np.ndarray]:
        """Load validity mask for a descriptor (images with non-zero features).

        Args:
            descriptor: Descriptor name

        Returns:
            Boolean array (N,) or None if no mask available
        """
        if descriptor not in self._valid:
            info = DESCRIPTOR_INFO.get(descriptor)
            if not info or not info.get("valid_file"):
                # No valid mask for this descriptor (e.g., VGG16)
                return None

            path = self.features_dir / info["valid_file"]
            if path.exists():
                self._valid[descriptor] = np.load(path)
            else:
                return None

        return self._valid.get(descriptor)

    def get_feature_vector(self, descriptor: str, image_id: int) -> np.ndarray:
        """Get feature vector for a single image.

        Args:
            descriptor: Descriptor name
            image_id: Image ID (0 to 999999)

        Returns:
            Feature vector (d,) as float32
        """
        X = self.load_features(descriptor)
        return np.asarray(X[image_id], dtype=np.float32)

    def search_brute_force(
        self,
        descriptor: str,
        query: np.ndarray,
        metric: str = "euclidean",
        k: int = 10,
        exclude_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Brute-force k-NN search using distance functions.

        Args:
            descriptor: Descriptor name
            query: Query vector (d,)
            metric: Distance metric (euclidean, cosine, chi2, l1)
            k: Number of neighbors
            exclude_id: Optional image ID to exclude from results

        Returns:
            Tuple of (image_ids, distances)
        """
        X = self.load_features(descriptor)
        valid = self.load_valid_mask(descriptor)

        # Compute distances
        distance_fn = METRICS.get(metric)
        if not distance_fn:
            raise ValueError(f"Unknown metric: {metric}")

        distances = distance_fn(X, query)

        # Apply validity mask
        if valid is not None:
            distances[~valid] = np.inf

        # Exclude query image
        if exclude_id is not None:
            distances[exclude_id] = np.inf

        # Find k nearest
        k = min(k, len(distances) - 1)
        idx = np.argpartition(distances, k)[:k]
        idx = idx[np.argsort(distances[idx])]

        return idx, distances[idx]

    def search_faiss(
        self,
        descriptor: str,
        query: np.ndarray,
        metric: str = "euclidean",
        k: int = 10,
        exclude_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index.

        Args:
            descriptor: Descriptor name
            query: Query vector (d,)
            metric: Distance metric (euclidean, cosine, chi2, l1)
            k: Number of neighbors
            exclude_id: Optional image ID to exclude from results

        Returns:
            Tuple of (image_ids, distances)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

        if descriptor not in FAISS_METRICS or metric not in FAISS_METRICS[descriptor]:
            raise ValueError(f"FAISS does not support {metric} distance for {descriptor}")

        # VGG16 vectors are already L2-normalized, so cosine uses the Euclidean
        # index and converts squared L2 distances to cosine distances.
        actual_metric = "euclidean" if descriptor == "vgg16" and metric == "cosine" else metric
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape != (DESCRIPTOR_INFO[descriptor]["dim"],):
            raise ValueError(f"Query for {descriptor} must have {DESCRIPTOR_INFO[descriptor]['dim']} values")
        if not np.isfinite(query).all():
            raise ValueError("Query contains NaN or infinite values")
        if metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        # Load index (lazy)
        key = (descriptor, actual_metric)
        if key not in self._faiss_indexes:
            index_path = self.indexes_dir / f"{descriptor}_{actual_metric}.faiss"
            if not index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index not found: {index_path}. "
                    f"Build it with backend/scripts/build_indexes.py."
                )

            if actual_metric != "euclidean":
                manifest_path = index_path.with_suffix(index_path.suffix + ".json")
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"{index_path.name} has no valid build manifest; rebuild it "
                        "with backend/scripts/build_indexes.py --force"
                    ) from exc
                expected_transform = "l2_normalized" if actual_metric == "cosine" else "none"
                if (
                    manifest.get("format_version") != 1
                    or manifest.get("descriptor") != descriptor
                    or manifest.get("metric") != actual_metric
                    or manifest.get("source_transform") != expected_transform
                    or manifest.get("dimension") != DESCRIPTOR_INFO[descriptor]["dim"]
                ):
                    raise ValueError(
                        f"{index_path.name} was built with incompatible feature settings; "
                        "rebuild it with backend/scripts/build_indexes.py --force"
                    )

            index = faiss.read_index(str(index_path))
            metric_types = {
                "euclidean": faiss.METRIC_L2,
                "cosine": faiss.METRIC_INNER_PRODUCT,
                "l1": faiss.METRIC_L1,
            }
            if index.metric_type != metric_types[actual_metric]:
                raise ValueError(
                    f"{index_path.name} contains a different FAISS metric; "
                    "rebuild it with backend/scripts/build_indexes.py --force"
                )
            features = self.load_features(descriptor)
            if index.d != features.shape[1] or index.ntotal != len(features):
                raise ValueError(f"FAISS index does not match feature data: {index_path}")
            if actual_metric != "euclidean" and manifest.get("vector_count") != index.ntotal:
                raise ValueError(f"FAISS index manifest does not match its vectors: {index_path}")
            self._faiss_indexes[key] = index

        index = self._faiss_indexes[key]

        valid = self.load_valid_mask(descriptor)
        search_k = min(k + int(exclude_id is not None), index.ntotal)
        while True:
            distances, ids = index.search(query.reshape(1, -1), search_k)
            distances = distances[0]
            ids = ids[0]

            if metric == "cosine" and actual_metric == "cosine":
                distances = 1.0 - np.clip(distances, -1.0, 1.0)
            elif actual_metric == "euclidean":
                if metric == "cosine":
                    distances = np.maximum(distances, 0.0) / 2.0
                else:
                    distances = np.sqrt(np.maximum(distances, 0.0))

            keep = ids >= 0
            if valid is not None:
                keep &= valid[np.maximum(ids, 0)]
            if exclude_id is not None:
                keep &= ids != exclude_id
            ids = ids[keep]
            distances = distances[keep]

            if len(ids) >= k or search_k == index.ntotal:
                break
            search_k = min(index.ntotal, max(search_k + 1, search_k * 2))

        return ids[:k], distances[:k]

    def available_faiss_metrics(self, descriptor: str) -> List[str]:
        """Return metrics with an index built for this descriptor and feature contract."""
        if not FAISS_AVAILABLE or descriptor not in FAISS_METRICS:
            return []
        available = []
        for metric in sorted(FAISS_METRICS[descriptor]):
            actual_metric = "euclidean" if descriptor == "vgg16" and metric == "cosine" else metric
            index_path = self.indexes_dir / f"{descriptor}_{actual_metric}.faiss"
            if not index_path.is_file() or index_path.stat().st_size == 0:
                continue
            if actual_metric == "euclidean":
                available.append(metric)
                continue
            manifest_path = index_path.with_suffix(index_path.suffix + ".json")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            expected_transform = "l2_normalized" if actual_metric == "cosine" else "none"
            if (
                manifest.get("format_version") == 1
                and manifest.get("descriptor") == descriptor
                and manifest.get("metric") == actual_metric
                and manifest.get("source_transform") == expected_transform
                and manifest.get("dimension") == DESCRIPTOR_INFO[descriptor]["dim"]
            ):
                available.append(metric)
        return available

    def available_vptree_metrics(self, descriptor: str) -> List[str]:
        """Return VP-Tree metrics whose file and feature-transform manifest are valid."""
        if descriptor not in VP_TREE_METRICS:
            return []
        available = []
        for metric in sorted(VP_TREE_METRICS[descriptor]):
            index_path = self.indexes_dir / f"{descriptor}_{metric}.vptree"
            manifest_path = index_path.with_suffix(index_path.suffix + ".json")
            if not index_path.is_file() or index_path.stat().st_size == 0:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                manifest.get("format_version") == 1
                and manifest.get("descriptor") == descriptor
                and manifest.get("metric") == metric
                and manifest.get("dimension") == DESCRIPTOR_INFO[descriptor]["dim"]
                and manifest.get("vector_count") == VP_TREE_DEMO_SIZE
                and manifest.get("source_transform") in {"none", "l2_normalized"}
                and (
                    manifest.get("source_transform") != "l2_normalized"
                    or (self.features_dir / f"{descriptor}_normalized.npy").is_file()
                )
            ):
                available.append(metric)
        return available

    def search_vptree(
        self,
        descriptor: str,
        query: np.ndarray,
        metric: str = "euclidean",
        k: int = 10,
        exclude_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the saved VP-Tree, which covers only its indexed image subset."""
        if descriptor not in VP_TREE_METRICS or metric not in VP_TREE_METRICS[descriptor]:
            raise ValueError(f"VP-Tree does not support {metric} distance for {descriptor}")
        key = (descriptor, metric)
        if key not in self._vptree_indexes:
            index_path = self.indexes_dir / f"{descriptor}_{metric}.vptree"
            if not index_path.exists():
                raise FileNotFoundError(f"VP-Tree index not found: {index_path}")

            info = DESCRIPTOR_INFO.get(descriptor)
            if info is None:
                raise ValueError(f"Unknown descriptor: {descriptor}")

            features_path = self.features_dir / info["file"]
            normalized_path = self.features_dir / f"{descriptor}_normalized.npy"
            manifest_path = index_path.with_suffix(index_path.suffix + ".json")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{index_path.name} has no valid build manifest; rebuild it with "
                    "backend/scripts/build_vptree_indexes.py --force"
                ) from exc
            expected_transform = manifest.get("source_transform")
            if expected_transform not in {"none", "l2_normalized"}:
                raise ValueError(f"Invalid feature transform in VP-Tree manifest: {index_path}")
            if expected_transform == "l2_normalized":
                if not normalized_path.is_file():
                    raise FileNotFoundError(
                        f"Normalized features required by {index_path.name}: {normalized_path}"
                    )
                features_path = normalized_path
            if not features_path.is_file():
                raise FileNotFoundError(f"Feature file not found: {features_path}")

            features = np.load(features_path, mmap_mode="r")
            vptree = VPTree.load(str(index_path), features)
            if (
                manifest.get("format_version") != 1
                or manifest.get("descriptor") != descriptor
                or manifest.get("metric") != metric
                or manifest.get("dimension") != info["dim"]
                or manifest.get("vector_count") != vptree.n
                or vptree.n > VP_TREE_DEMO_SIZE
                or vptree.n < 1
                or vptree.n > len(features)
                or vptree.d != features.shape[1]
            ):
                raise ValueError(f"VP-Tree index does not match feature file: {index_path}")

            self._vptree_indexes[key] = vptree
            self._vptree_transforms[key] = expected_transform

        vptree = self._vptree_indexes[key]
        if self._vptree_transforms[key] == "l2_normalized":
            query = np.asarray(query, dtype=np.float32)
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
        valid = self.load_valid_mask(descriptor)
        search_k = min(k + int(exclude_id is not None), vptree.n)
        while True:
            ids, distances = vptree.query(query, k=search_k)
            keep = np.ones(len(ids), dtype=bool)
            if valid is not None:
                keep &= valid[ids]
            if exclude_id is not None:
                keep &= ids != exclude_id
            ids = ids[keep]
            distances = distances[keep]
            if len(ids) >= k or search_k == vptree.n:
                break
            search_k = min(vptree.n, max(search_k + 1, search_k * 2))
        return ids[:k], distances[:k]

    def search(
        self,
        descriptor: str,
        query: np.ndarray,
        metric: str = "euclidean",
        index_type: str = "faiss",
        k: int = 10,
        exclude_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unified search interface.

        Args:
            descriptor: Descriptor name (vgg16, edgehistogram, homogeneoustexture)
            query: Query vector (d,)
            metric: Distance metric (euclidean, cosine, chi2, l1)
            index_type: Index type (faiss, vptree, brute)
            k: Number of neighbors
            exclude_id: Optional image ID to exclude

        Returns:
            Tuple of (image_ids, distances)
        """
        if index_type == "brute":
            return self.search_brute_force(descriptor, query, metric, k, exclude_id)
        elif index_type == "faiss":
            return self.search_faiss(descriptor, query, metric, k, exclude_id)
        elif index_type == "vptree":
            return self.search_vptree(descriptor, query, metric, k, exclude_id)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def get_scale(self, descriptor: str, metric: str = "euclidean") -> float:
        """Get typical distance scale for a descriptor (for normalization in fusion).

        Uses precomputed median distances.

        Args:
            descriptor: Descriptor name
            metric: Distance metric

        Returns:
            Median distance value
        """
        # Precomputed scales (can be updated after running benchmarks)
        SCALES = {
            ("vgg16", "euclidean"): 0.5,
            ("edgehistogram", "euclidean"): 2.5,
            ("edgehistogram", "chi2"): 0.8,
            ("homogeneoustexture", "euclidean"): 1.5,
        }
        return SCALES.get((descriptor, metric), 1.0)


# Singleton instance
_index_manager: Optional[IndexManager] = None


def get_index_manager() -> IndexManager:
    """Get the singleton IndexManager instance."""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager


if __name__ == "__main__":
    # Quick test
    print("Testing IndexManager...")

    manager = get_index_manager()

    # Test feature loading
    print("\n1. Loading features:")
    for desc in ["edgehistogram", "homogeneoustexture"]:
        X = manager.load_features(desc)
        print(f"   {desc}: shape={X.shape}, dtype={X.dtype}")

    # Test brute-force search
    print("\n2. Brute-force search (edgehistogram, euclidean):")
    query = manager.get_feature_vector("edgehistogram", 0)
    ids, dists = manager.search_brute_force("edgehistogram", query, "euclidean", k=5)
    print(f"   Query image: 0")
    print(f"   Top 5: {ids}")
    print(f"   Distances: {dists}")

    # Test FAISS search
    print("\n3. FAISS search (edgehistogram, euclidean):")
    try:
        ids, dists = manager.search_faiss("edgehistogram", query, "euclidean", k=5)
        print(f"   Top 5: {ids}")
        print(f"   Distances: {dists}")
    except Exception as e:
        print(f"   Error: {e}")
