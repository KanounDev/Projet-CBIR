import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from unittest.mock import patch

from app.indexing.index_manager import IndexManager
from app.indexing.faiss_index import FAISSIndex
from app.indexing.vptree import VPTree
from app.schemas.search import SearchRequest
from app.services.search_service import SearchService, SearchRequest as ServiceSearchRequest
from scripts.build_indexes import build_faiss, build_vptree, main as build_indexes_main


class SearchRequestTests(unittest.TestCase):
    def test_accepts_visual_search_request(self):
        request = SearchRequest(
            query_image_id=42,
            descriptors=["vgg16"],
            metrics={"vgg16": "euclidean"},
        )

        self.assertEqual(request.query_image_id, 42)
        self.assertEqual(request.index_type, "faiss")

    def test_combined_search_uses_exhaustive_metric_even_if_faiss_selected(self):
        request = SearchRequest(
            keywords=["water"],
            query_image_id=42,
            descriptors=["edgehistogram"],
            metrics={"edgehistogram": "chi2"},
            index_type="faiss",
        )

        self.assertEqual(request.metrics["edgehistogram"], "chi2")

    def test_rejects_invalid_search_options(self):
        invalid_requests = (
            {"keywords": ["sunset"], "keyword_mode": "xor"},
            {"query_image_id": 1_000_000, "descriptors": ["vgg16"]},
            {"query_image_id": 42, "descriptors": ["unknown"]},
            {"query_image_id": 42, "descriptors": ["vgg16"], "metrics": {"vgg16": "chi2"}},
            {
                "query_image_id": 42,
                "descriptors": ["edgehistogram"],
                "metrics": {"edgehistogram": "chi2"},
                "index_type": "faiss",
            },
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
        features = np.zeros((4, 150), dtype=np.float32)
        features[:, 0] = [0.0, 1.0, 2.0, 5.0]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            feature_dir = data_dir / "features"
            index_dir = data_dir / "indexes"
            feature_dir.mkdir()
            index_dir.mkdir()
            np.save(feature_dir / "edgehistogram.npy", features)
            np.save(
                feature_dir / "edgehistogram_valid.npy",
                np.array([False, True, True, True]),
            )

            build_vptree(
                "edgehistogram",
                features,
                "euclidean",
                progress=False,
                indexes_dir=index_dir,
            )

            manager = IndexManager(data_dir)
            self.assertNotIn("euclidean", manager.available_vptree_metrics("edgehistogram"))
            ids, distances = manager.search(
                "edgehistogram",
                features[1],
                metric="euclidean",
                index_type="vptree",
                k=2,
                exclude_id=1,
            )
            manager._vptree_indexes.clear()
            manager._vptree_transforms.clear()
            manager._features.clear()

        self.assertEqual(ids.tolist(), [2, 3])
        np.testing.assert_allclose(distances, [1.0, 4.0])

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


class FaissMetricTests(unittest.TestCase):
    def test_builder_defaults_to_faiss_with_euclidean_metrics(self):
        features = np.zeros((5, 150), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "sys.argv",
                    ["build_indexes.py", "--descriptors", "edgehistogram"],
                ),
                patch("scripts.build_indexes.INDEXES_DIR", Path(temp_dir)),
                patch("scripts.build_indexes.load_features", return_value=features),
                patch("scripts.build_indexes.build_faiss", return_value=0.0) as build,
                redirect_stdout(StringIO()),
            ):
                build_indexes_main()

        build.assert_called_once()

    def test_search_skips_invalid_features(self):
        features = np.zeros((3, 150), dtype=np.float32)
        features[:, 0] = [0.0, 0.1, 2.0]
        query = np.zeros(150, dtype=np.float32)
        valid = np.array([False, True, True])

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            feature_dir = data_dir / "features"
            index_dir = data_dir / "indexes"
            feature_dir.mkdir()
            index_dir.mkdir()
            np.save(feature_dir / "edgehistogram.npy", features)
            np.save(feature_dir / "edgehistogram_valid.npy", valid)
            build_faiss(
                "edgehistogram",
                features,
                "euclidean",
                indexes_dir=index_dir,
                expected_count=len(features),
            )

            manager = IndexManager(data_dir)
            actual_ids, actual_distances = manager.search(
                "edgehistogram", query, index_type="faiss", k=2
            )
            expected_ids, expected_distances = manager.search_brute_force(
                "edgehistogram", query, k=2
            )
            manager._features.clear()
            manager._valid.clear()
            manager._faiss_indexes.clear()

        self.assertEqual(actual_ids.tolist(), expected_ids.tolist())
        np.testing.assert_allclose(actual_distances, expected_distances)

    def test_cosine_and_l1_match_brute_force_on_raw_features(self):
        features = np.zeros((4, 150), dtype=np.float32)
        features[:, :2] = [[3.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 2.0]]
        query = np.zeros(150, dtype=np.float32)
        query[:2] = [1.0, 1.2]

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            feature_dir = data_dir / "features"
            index_dir = data_dir / "indexes"
            feature_dir.mkdir()
            index_dir.mkdir()
            np.save(feature_dir / "edgehistogram.npy", features)

            for metric in ("cosine", "l1"):
                build_faiss(
                    "edgehistogram",
                    features,
                    metric,
                    indexes_dir=index_dir,
                    expected_count=len(features),
                )

                manager = IndexManager(data_dir)
                self.assertIn(metric, manager.available_faiss_metrics("edgehistogram"))
                actual_ids, actual_distances = manager.search(
                    "edgehistogram", query, metric=metric, index_type="faiss", k=3
                )
                expected_ids, expected_distances = manager.search_brute_force(
                    "edgehistogram", query, metric=metric, k=3
                )

                self.assertEqual(actual_ids.tolist(), expected_ids.tolist())
                np.testing.assert_allclose(actual_distances, expected_distances, atol=1e-5)
                manager._features.clear()
                manager._faiss_indexes.clear()

    def test_chi2_is_not_a_faiss_metric(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            FAISSIndex().build(np.ones((3, 2), dtype=np.float32), "chi2")


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
