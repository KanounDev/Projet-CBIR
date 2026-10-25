import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from unittest.mock import patch

from app.indexing.index_manager import IndexManager
from app.indexing.vptree import VPTree
from app.schemas.search import SearchRequest
from app.services.search_service import SearchService, SearchRequest as ServiceSearchRequest


class SearchRequestTests(unittest.TestCase):
    def test_accepts_visual_search_request(self):
        request = SearchRequest(
            query_image_id=42,
            descriptors=["vgg16"],
            metrics={"vgg16": "euclidean"},
        )

        self.assertEqual(request.query_image_id, 42)
        self.assertEqual(request.index_type, "faiss")

    def test_rejects_invalid_search_options(self):
        invalid_requests = (
            {"keywords": ["sunset"], "keyword_mode": "xor"},
            {"query_image_id": 1_000_000, "descriptors": ["vgg16"]},
            {"query_image_id": 42, "descriptors": ["unknown"]},
            {"query_image_id": 42, "descriptors": ["vgg16"], "metrics": {"vgg16": "chi2"}},
            {
                "query_image_id": 42,
                "descriptors": ["vgg16", "edgehistogram"],
                "index_type": "faiss",
            },
            {"keywords": []},
        )

        for payload in invalid_requests:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    SearchRequest(**payload)


class VPTreeSearchTests(unittest.TestCase):
    def test_manager_searches_saved_subset_and_excludes_query(self):
        features = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            feature_dir = data_dir / "features"
            index_dir = data_dir / "indexes"
            feature_dir.mkdir()
            index_dir.mkdir()
            np.save(feature_dir / "edgehistogram.npy", features)

            tree = VPTree(features, leaf_size=2, distance="euclidean")
            tree.build()
            tree.save(str(index_dir / "edgehistogram_euclidean.vptree"))

            manager = IndexManager(data_dir)
            ids, distances = manager.search(
                "edgehistogram",
                features[1],
                metric="euclidean",
                index_type="vptree",
                k=2,
                exclude_id=1,
            )
            manager._vptree_indexes.clear()
            manager._features.clear()

        self.assertEqual(ids.tolist(), [0, 2])
        np.testing.assert_allclose(distances, [1.0, 1.0])

    def test_leaf_query_returns_nearest_vectors_in_order(self):
        features = np.array(
            [[0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        tree = VPTree(features, leaf_size=len(features), distance="euclidean")
        tree.build()

        ids, distances = tree.query(np.array([0.9, 0.0], dtype=np.float32), k=3)

        self.assertEqual(ids.tolist(), [2, 0, 3])
        np.testing.assert_allclose(distances, [0.1, 0.9, 1.1], atol=1e-6)


class CombinedSearchTests(unittest.TestCase):
    def test_combined_search_fuses_all_selected_descriptors(self):
        class FakeIndexManager:
            features = {
                "vgg16": np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
                "edgehistogram": np.array([[0.0], [4.0], [1.0], [3.0]], dtype=np.float32),
            }

            def get_feature_vector(self, descriptor, image_id):
                return self.features[descriptor][image_id]

            def load_features(self, descriptor):
                return self.features[descriptor]

            def get_scale(self, descriptor, metric):
                return 1.0

            def load_valid_mask(self, descriptor):
                return None

        service = SearchService.__new__(SearchService)
        service.index_manager = FakeIndexManager()
        request = ServiceSearchRequest(
            keywords=["water"],
            query_image_id=0,
            descriptors=["vgg16", "edgehistogram"],
            metrics={"vgg16": "euclidean", "edgehistogram": "euclidean"},
            weights={"vgg16": 0.5, "edgehistogram": 0.5},
            index_type="brute",
            k=2,
            exclude_query=True,
        )
        expected_images = [
            {"_id": 2, "thumb": "thumb-2.jpg", "keywords": []},
            {"_id": 1, "thumb": "thumb-1.jpg", "keywords": []},
        ]

        with (
            patch("app.services.search_service.search_by_keywords", return_value=[0, 1, 2, 3]),
            patch("app.services.search_service.get_images_by_ids", return_value=expected_images),
        ):
            results = service.search_combined(request)

        self.assertEqual([result.id for result in results], [2, 1])
        np.testing.assert_allclose([result.distance for result in results], [1.5, 2.5])


if __name__ == "__main__":
    unittest.main()
