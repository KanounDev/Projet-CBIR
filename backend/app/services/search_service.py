"""Unified search service for CBIR.

Provides a single interface to search by:
- Keywords (text search with AND/OR mode)
- Visual similarity (choose descriptor, distance, index type)
- Combined: keywords + visual similarity

The user can choose:
- Descriptor(s): vgg16, edgehistogram, homogeneoustexture (one or multiple)
- Distance metric per descriptor: euclidean, cosine, chi2, l1
- Index type: faiss (production), vptree (demo), brute (reference)
- Fusion weights for combining multiple descriptors
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from app.db.mongodb import get_images_by_ids
from app.services.text_search import search_by_keywords, normalize_keywords
from app.indexing.index_manager import get_index_manager


@dataclass
class SearchResult:
    """A single search result."""
    id: int
    distance: float
    thumb: str
    keywords: List[str]
    photo_url: Optional[str] = None
    web_url: Optional[str] = None
    license: Optional[str] = None


@dataclass
class SearchRequest:
    """A search request with all parameters."""
    # Text search
    keywords: Optional[List[str]] = None
    keyword_mode: str = "and"  # "and" or "or"

    # Visual search
    query_image_id: Optional[int] = None
    descriptors: Optional[List[str]] = None  # ["vgg16", "edgehistogram", "homogeneoustexture"]
    metrics: Optional[Dict[str, str]] = None  # {"vgg16": "euclidean", "edgehistogram": "chi2"}
    weights: Optional[Dict[str, float]] = None  # {"vgg16": 1.0, "edgehistogram": 0.5}
    index_type: str = "faiss"  # "faiss", "vptree", "brute"

    # Result options
    k: int = 20
    exclude_query: bool = True


class SearchService:
    """Unified search service combining text and visual search."""

    def __init__(self):
        self.index_manager = get_index_manager()

    def search_text_only(self, request: SearchRequest) -> List[SearchResult]:
        """Search by keywords only.

        Args:
            request: Search request with keywords and mode

        Returns:
            List of search results
        """
        if not request.keywords:
            return []

        # Get matching image IDs
        image_ids = search_by_keywords(
            request.keywords,
            mode=request.keyword_mode,
            limit=request.k
        )

        # Get metadata
        images = get_images_by_ids(image_ids)

        # Build results
        results = []
        for img in images:
            results.append(SearchResult(
                id=img["_id"],
                distance=0.0,  # No distance for text-only
                thumb=img.get("thumb", ""),
                keywords=img.get("keywords", []),
                photo_url=img.get("photo_url"),
                web_url=img.get("web_url"),
                license=img.get("license"),
            ))

        return results

    def search_visual_only(self, request: SearchRequest) -> List[SearchResult]:
        """Search by visual similarity only.

        Args:
            request: Search request with query image and descriptor settings

        Returns:
            List of search results
        """
        if request.query_image_id is None:
            raise ValueError("query_image_id required for visual search")

        if not request.descriptors:
            request.descriptors = ["vgg16"]  # Default to VGG16

        # Default metrics
        if not request.metrics:
            request.metrics = {d: "euclidean" for d in request.descriptors}

        # Default weights
        if not request.weights:
            request.weights = {d: 1.0 for d in request.descriptors}

        # Multi-descriptor fusion
        if len(request.descriptors) == 1:
            # Single descriptor: simple search
            desc = request.descriptors[0]
            query_vec = self.index_manager.get_feature_vector(desc, request.query_image_id)
            metric = request.metrics.get(desc, "euclidean")

            ids, dists = self.index_manager.search(
                descriptor=desc,
                query=query_vec,
                metric=metric,
                index_type=request.index_type,
                k=request.k,
                exclude_id=request.query_image_id if request.exclude_query else None,
            )
        else:
            # Multiple descriptors: fusion
            ids, dists = self._search_with_fusion(request)

        # Get metadata
        images = get_images_by_ids(ids.tolist())

        # Build results
        results = []
        dist_dict = dict(zip(ids, dists))
        for img in images:
            results.append(SearchResult(
                id=img["_id"],
                distance=float(dist_dict.get(img["_id"], 0.0)),
                thumb=img.get("thumb", ""),
                keywords=img.get("keywords", []),
                photo_url=img.get("photo_url"),
                web_url=img.get("web_url"),
                license=img.get("license"),
            ))

        return results

    def _search_with_fusion(self, request: SearchRequest) -> Tuple[np.ndarray, np.ndarray]:
        """Search with multiple descriptors and fuse results.

        Strategy:
        1. Get distances from each descriptor
        2. Normalize by scale (median distance)
        3. Combine with weighted sum

        Args:
            request: Search request with multiple descriptors

        Returns:
            Tuple of (image_ids, fused_distances)
        """
        # Get number of images
        n = len(self.index_manager.load_features(request.descriptors[0]))

        # Accumulate weighted distances
        fused = np.zeros(n, dtype=np.float32)
        valid = np.ones(n, dtype=bool)

        total_weight = sum(request.weights.values())

        # Import distance functions for direct computation
        from app.indexing.distances import METRICS

        for desc in request.descriptors:
            weight = request.weights.get(desc, 1.0)
            metric = request.metrics.get(desc, "euclidean")

            # Get query vector and feature matrix
            query_vec = self.index_manager.get_feature_vector(desc, request.query_image_id)
            X = self.index_manager.load_features(desc)

            # Compute distances directly (full array, no sorting)
            distance_fn = METRICS[metric]
            distances = distance_fn(X, query_vec)  # Returns (N,) array

            # Normalize by scale
            scale = self.index_manager.get_scale(desc, metric)

            # Add weighted normalized distances
            fused += (weight / total_weight) * distances / scale

            # Update validity mask
            valid_mask = self.index_manager.load_valid_mask(desc)
            if valid_mask is not None:
                valid &= valid_mask

        # Exclude query image if requested
        if request.exclude_query and request.query_image_id is not None:
            valid[request.query_image_id] = False

        # Exclude invalid images
        fused[~valid] = np.inf

        # Find top-k
        k = min(request.k, n - 1)
        idx = np.argpartition(fused, k)[:k]
        idx = idx[np.argsort(fused[idx])]

        return idx, fused[idx]

    def search_combined(self, request: SearchRequest) -> List[SearchResult]:
        """Search by keywords first, then rank by visual similarity.

        Strategy:
        1. Filter images by keywords
        2. Compute visual similarity on filtered set
        3. Return top-k by visual similarity

        Args:
            request: Search request with both keywords and visual parameters

        Returns:
            List of search results
        """
        if not request.keywords:
            return self.search_visual_only(request)

        # Step 1: Get candidates by keywords
        candidate_ids = search_by_keywords(
            request.keywords,
            mode=request.keyword_mode,
            limit=10000  # Get many candidates, then rank by visual
        )

        if not candidate_ids:
            return []

        # Step 2: Compute visual similarity for candidates
        if not request.descriptors:
            request.descriptors = ["vgg16"]

        if not request.metrics:
            request.metrics = {d: "euclidean" for d in request.descriptors}

        if not request.weights:
            request.weights = {d: 1.0 for d in request.descriptors}

        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        fused_distances = np.zeros(len(candidate_ids), dtype=np.float32)
        valid = np.ones(len(candidate_ids), dtype=bool)
        total_weight = sum(request.weights.values())
        from app.indexing.distances import METRICS

        for desc in request.descriptors:
            metric = request.metrics.get(desc, "euclidean")
            weight = request.weights.get(desc, 1.0)
            query_vec = self.index_manager.get_feature_vector(
                desc, request.query_image_id
            )
            features = self.index_manager.load_features(desc)
            distances = METRICS[metric](features[candidate_ids], query_vec)
            fused_distances += (
                (weight / total_weight)
                * distances
                / self.index_manager.get_scale(desc, metric)
            )

            valid_mask = self.index_manager.load_valid_mask(desc)
            if valid_mask is not None:
                valid &= valid_mask[candidate_ids]

        if request.exclude_query and request.query_image_id is not None:
            valid &= candidate_ids != request.query_image_id
        candidate_ids = candidate_ids[valid]
        fused_distances = fused_distances[valid]
        order = np.argsort(fused_distances)[:request.k]

        # Get metadata
        ids = candidate_ids[order].tolist()
        dists = fused_distances[order]
        images = get_images_by_ids(ids)

        # Build results
        results = []
        dist_dict = dict(zip(ids, dists))
        for img in images:
            results.append(SearchResult(
                id=img["_id"],
                distance=float(dist_dict.get(img["_id"], 0.0)),
                thumb=img.get("thumb", ""),
                keywords=img.get("keywords", []),
                photo_url=img.get("photo_url"),
                web_url=img.get("web_url"),
                license=img.get("license"),
            ))

        return results

    def search(self, request: SearchRequest) -> List[SearchResult]:
        """Unified search entry point.

        Determines which search mode to use based on request parameters.

        Args:
            request: Search request

        Returns:
            List of search results
        """
        has_keywords = bool(request.keywords)
        has_image = request.query_image_id is not None

        if has_keywords and has_image:
            return self.search_combined(request)
        elif has_keywords:
            return self.search_text_only(request)
        elif has_image:
            return self.search_visual_only(request)
        else:
            raise ValueError("Either keywords or query_image_id must be provided")


# Singleton instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get the singleton SearchService instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


if __name__ == "__main__":
    # Quick test
    print("Testing SearchService...")

    service = get_search_service()

    # Test 1: Text-only search
    print("\n1. Text-only search (keywords=['sunset'], mode='and'):")
    request = SearchRequest(
        keywords=["sunset"],
        keyword_mode="and",
        k=5
    )
    results = service.search(request)
    for r in results:
        print(f"   ID={r.id}, keywords={r.keywords[:3]}...")

    # Test 2: Visual-only search (VGG16)
    print("\n2. Visual search (query_image_id=0, descriptor='vgg16'):")
    request = SearchRequest(
        query_image_id=0,
        descriptors=["vgg16"],
        metrics={"vgg16": "euclidean"},
        index_type="brute",
        k=5
    )
    results = service.search(request)
    for r in results:
        print(f"   ID={r.id}, distance={r.distance:.4f}")

    # Test 3: Visual search (Edge Histogram + chi2)
    print("\n3. Visual search (query_image_id=0, descriptor='edgehistogram', metric='chi2'):")
    request = SearchRequest(
        query_image_id=0,
        descriptors=["edgehistogram"],
        metrics={"edgehistogram": "chi2"},
        index_type="brute",
        k=5
    )
    results = service.search(request)
    for r in results:
        print(f"   ID={r.id}, distance={r.distance:.4f}")
