"""Comprehensive evaluation script for CBIR system.

This script evaluates:
1. Self-retrieval sanity checks
2. FAISS/brute-force top-k consistency and performance
3. Descriptor distance statistics
4. Multi-descriptor fusion timing
5. Text search performance

Outputs:
- backend/evaluation_results.json - All metrics
- backend/evaluation_report.md - Human-readable summary
- backend/evaluation_charts/ - Performance charts

Usage:
    cd backend
    python scripts/comprehensive_evaluation.py
"""
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.index_manager import get_index_manager
from app.services.search_service import get_search_service, SearchRequest

# Configuration
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Test queries - diverse set of image IDs
TEST_QUERY_IDS = list(range(0, 1000000, 50000))[:20]  # 20 queries spread across dataset

# For keyword-based evaluation
TEST_KEYWORDS = ["sunset", "beach", "flower", "car", "dog", "cat", "tree", "water", "sky", "city"]


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    query_id: int
    descriptor: str
    metric: str
    index_type: str
    k: int
    search_time_ms: float
    self_retrieval_rate_pct: float
    top_k_jaccard_pct: Optional[float]
    mean_distance: float
    std_distance: float


@dataclass
class ComparisonResult:
    """Comparison between two methods."""
    method1: str
    method2: str
    metric: str
    difference: float
    winner: str


def compute_self_retrieval_score(
    manager,
    descriptor: str,
    query_id: int,
    metric: str,
    index_type: str,
    k: int,
) -> Tuple[float, float, np.ndarray]:
    """Check if query image is retrieved at rank 0 (self-retrieval test).

    Returns self-retrieval success, elapsed time, and neighbor distances excluding the query.
    """
    query_vec = manager.get_feature_vector(descriptor, query_id)

    start = time.perf_counter()
    if index_type == "brute":
        ids, dists = manager.search_brute_force(descriptor, query_vec, metric, k=k, exclude_id=None)
    elif index_type == "faiss":
        ids, dists = manager.search_faiss(descriptor, query_vec, metric, k=k)
    else:
        raise ValueError(f"Unsupported evaluation index type: {index_type}")

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Check if query is at rank 0
    is_first = 1.0 if len(ids) > 0 and ids[0] == query_id else 0.0
    return is_first, elapsed_ms, dists[ids != query_id]


def compute_neighbor_consistency(manager, descriptor: str, query_id: int, metric: str, k: int = 10) -> float:
    """Compute consistency between brute force and FAISS results.

    Returns:
        Jaccard similarity between top-k results from both methods
    """
    query_vec = manager.get_feature_vector(descriptor, query_id)

    ids_brute, _ = manager.search_brute_force(descriptor, query_vec, metric, k=k)
    ids_faiss, _ = manager.search_faiss(descriptor, query_vec, metric, k=k)

    set_brute = set(ids_brute)
    set_faiss = set(ids_faiss)
    union = set_brute | set_faiss
    return len(set_brute & set_faiss) / len(union) if union else 0.0


def evaluate_descriptor(
    manager,
    descriptor: str,
    metrics: List[str],
    index_types: List[str],
    query_ids: List[int],
    k_values: List[int]
) -> List[EvaluationResult]:
    """Evaluate a single descriptor across all metrics and index types."""
    results = []

    for metric in metrics:
        for index_type in index_types:
            if (
                index_type == "faiss"
                and metric not in manager.available_faiss_metrics(descriptor)
            ):
                print(f"  Skipping {descriptor}/{metric}/faiss: no compatible built index")
                continue
            print(f"  {descriptor}/{metric}/{index_type}:")

            for k in k_values:
                times = []
                self_retrieval_scores = []
                consistency_scores = []
                all_distances = []  # Collect all distances for statistics

                for query_id in query_ids:
                    # Self-retrieval test
                    is_first, time_ms, dists = compute_self_retrieval_score(
                        manager, descriptor, query_id, metric, index_type, k
                    )
                    times.append(time_ms)
                    self_retrieval_scores.append(is_first)

                    # Collect all distances (excluding distance to self)
                    all_distances.extend(dists)

                    # Consistency with brute force (only for FAISS)
                    if index_type == "faiss":
                        consistency = compute_neighbor_consistency(manager, descriptor, query_id, metric, k)
                        consistency_scores.append(consistency)

                avg_time = np.mean(times)
                self_retrieval_rate = np.mean(self_retrieval_scores) * 100
                consistency_rate = np.mean(consistency_scores) * 100 if consistency_scores else None

                # Compute distance statistics
                mean_dist = np.mean(all_distances) if all_distances else 0.0
                std_dist = np.std(all_distances) if all_distances else 0.0

                consistency_label = f"{consistency_rate:.1f}%" if consistency_rate is not None else "N/A"
                print(f"    k={k}: time={avg_time:.2f}ms, self_retrieval={self_retrieval_rate:.1f}%, top-k Jaccard={consistency_label}")

                results.append(EvaluationResult(
                    query_id=-1,  # Aggregated
                    descriptor=descriptor,
                    metric=metric,
                    index_type=index_type,
                    k=k,
                    search_time_ms=avg_time,
                    self_retrieval_rate_pct=self_retrieval_rate,
                    top_k_jaccard_pct=consistency_rate,
                    mean_distance=mean_dist,
                    std_distance=std_dist
                ))

    return results


def evaluate_text_search(service, keywords_list: List[str], k_values: List[int]) -> Dict:
    """Evaluate text search performance."""
    print("\n=== Text Search Evaluation ===")

    results = {}

    for keyword in keywords_list:
        print(f"\nKeyword: '{keyword}'")

        for mode in ["and", "or"]:
            for k in k_values:
                request = SearchRequest(
                    keywords=[keyword],
                    keyword_mode=mode,
                    k=k
                )

                start = time.perf_counter()
                try:
                    search_results = service.search(request)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    count = len(search_results)
                    print(f"  {mode.upper()} k={k}: {count} results, {elapsed_ms:.2f}ms")

                    key = f"{keyword}_{mode}_{k}"
                    results[key] = {
                        "keyword": keyword,
                        "mode": mode,
                        "k": k,
                        "result_count": count,
                        "time_ms": elapsed_ms
                    }
                except Exception as e:
                    print(f"  {mode.upper()} k={k}: Error - {e}")
                    results[f"{keyword}_{mode}_{k}"] = {"error": str(e)}

    return results


def evaluate_multi_descriptor_fusion(manager, query_ids: List[int], k: int = 20) -> Dict:
    """Evaluate multi-descriptor fusion effectiveness."""
    print("\n=== Multi-Descriptor Fusion Evaluation ===")

    results = {}

    # Single descriptors
    single_results = {}
    descriptors = ["vgg16", "edgehistogram", "homogeneoustexture"]

    for desc in descriptors:
        print(f"\n{desc}:")
        times = []
        distances = []

        for query_id in query_ids:
            query_vec = manager.get_feature_vector(desc, query_id)
            start = time.perf_counter()
            try:
                ids, dists = manager.search_brute_force(desc, query_vec, "euclidean", k=k)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                distances.extend(dists)
            except Exception as e:
                print(f"  Error for query {query_id}: {e}")

        if times:
            single_results[desc] = {
                "avg_time_ms": np.mean(times),
                "avg_distance": np.mean(distances),
                "std_distance": np.std(distances)
            }
            print(f"  Avg time: {np.mean(times):.2f}ms, Avg distance: {np.mean(distances):.4f}")

    # Fusion of VGG16 + Edge Histogram
    print("\nFusion (VGG16 + EdgeHistogram):")
    service = get_search_service()
    fusion_times = []
    fusion_distances = []

    for query_id in query_ids:
        request = SearchRequest(
            query_image_id=query_id,
            descriptors=["vgg16", "edgehistogram"],
            metrics={"vgg16": "euclidean", "edgehistogram": "euclidean"},
            weights={"vgg16": 0.7, "edgehistogram": 0.3},
            index_type="brute",
            k=k
        )

        start = time.perf_counter()
        try:
            results_list = service.search(request)
            elapsed = (time.perf_counter() - start) * 1000
            fusion_times.append(elapsed)
            fusion_distances.extend([r.distance for r in results_list])
        except Exception as e:
            print(f"  Error for query {query_id}: {e}")

    if fusion_times:
        results["fusion_vgg16_edgehistogram"] = {
            "avg_time_ms": np.mean(fusion_times),
            "avg_distance": np.mean(fusion_distances),
            "std_distance": np.std(fusion_distances)
        }
        print(f"  Avg time: {np.mean(fusion_times):.2f}ms, Avg distance: {np.mean(fusion_distances):.4f}")

    results["single_descriptors"] = single_results

    return results


def compare_index_types(manager, descriptor: str, query_ids: List[int], k: int = 20) -> Dict:
    """Compare FAISS vs Brute Force performance and accuracy."""
    print(f"\n=== Index Type Comparison ({descriptor}) ===")

    results = {
        "brute": {"times": [], "consistency": []},
        "faiss": {"times": [], "consistency": []}
    }

    for query_id in query_ids:
        query_vec = manager.get_feature_vector(descriptor, query_id)

        # Brute force
        start = time.perf_counter()
        ids_brute, _ = manager.search_brute_force(descriptor, query_vec, "euclidean", k=k)
        elapsed_brute = (time.perf_counter() - start) * 1000
        results["brute"]["times"].append(elapsed_brute)

        # FAISS
        start = time.perf_counter()
        ids_faiss, _ = manager.search_faiss(descriptor, query_vec, "euclidean", k=k)
        elapsed_faiss = (time.perf_counter() - start) * 1000
        results["faiss"]["times"].append(elapsed_faiss)
        union = set(ids_brute) | set(ids_faiss)
        consistency = len(set(ids_brute) & set(ids_faiss)) / len(union) if union else 0.0
        results["faiss"]["consistency"].append(consistency)

    # Summary
    brute_avg = np.mean(results["brute"]["times"])
    faiss_avg = np.mean(results["faiss"]["times"])
    speedup = brute_avg / faiss_avg if faiss_avg > 0 else 0
    consistency_avg = np.mean(results["faiss"]["consistency"]) if results["faiss"]["consistency"] else 0

    print(f"  Brute Force: {brute_avg:.2f}ms avg")
    print(f"  FAISS: {faiss_avg:.2f}ms avg")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Consistency (Jaccard@k): {consistency_avg*100:.1f}%")

    return {
        "brute_avg_time_ms": brute_avg,
        "faiss_avg_time_ms": faiss_avg,
        "speedup": speedup,
        "consistency": consistency_avg
    }


def generate_report(all_results: Dict) -> str:
    """Generate a Markdown evaluation report."""
    report = []
    report.append("# CBIR System Evaluation Report")
    report.append(f"\n**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")

    # 1. Self-Retrieval Tests
    report.append("## 1. Self-Retrieval Sanity Checks")
    report.append(
        "\nSelf-retrieval checks whether a query's own feature vector is returned first. "
        "This is an implementation sanity check, not a measure of semantic relevance or precision. "
        "Top-k Jaccard compares FAISS results with brute force and is not applicable to brute-force rows.\n"
    )

    if "descriptor_evaluation" in all_results:
        report.append("| Descriptor | Metric | Index | k | Self-Retrieval Rate | Top-k Jaccard vs brute force | Avg Time (ms) |")
        report.append("|------------|--------|-------|---|---------------------|-----------------------------|---------------|")

        for r in all_results["descriptor_evaluation"]:
            if r.index_type in ["brute", "faiss"]:
                jaccard = f"{r.top_k_jaccard_pct:.1f}%" if r.top_k_jaccard_pct is not None else "N/A"
                report.append(
                    f"| {r.descriptor} | {r.metric} | {r.index_type} | {r.k} | "
                    f"{r.self_retrieval_rate_pct:.1f}% | {jaccard} | {r.search_time_ms:.2f} |"
                )

    # 2. Index Type Comparison
    report.append("\n## 2. Index Type Comparison (FAISS vs Brute Force)\n")

    if "index_comparison" in all_results:
        for desc, comp in all_results["index_comparison"].items():
            report.append(f"### {desc.upper()}")
            report.append(f"- Brute Force: {comp['brute_avg_time_ms']:.2f}ms avg")
            report.append(f"- FAISS: {comp['faiss_avg_time_ms']:.2f}ms avg")
            report.append(f"- **Speedup: {comp['speedup']:.1f}x**")
            report.append(f"- Top-k Jaccard vs brute force: {comp['consistency']*100:.1f}%\n")

    # 3. Multi-Descriptor Fusion
    report.append("\n## 3. Multi-Descriptor Fusion\n")

    if "fusion_evaluation" in all_results:
        fusion = all_results["fusion_evaluation"]
        if "single_descriptors" in fusion:
            report.append("| Descriptor | Avg Time (ms) | Avg Distance |")
            report.append("|------------|---------------|--------------|")
            for desc, data in fusion["single_descriptors"].items():
                report.append(f"| {desc} | {data['avg_time_ms']:.2f} | {data['avg_distance']:.4f} |")

        if "fusion_vgg16_edgehistogram" in fusion:
            f = fusion["fusion_vgg16_edgehistogram"]
            report.append(f"\n**Fusion (VGG16 70% + EdgeHistogram 30%):**")
            report.append(f"- Avg Time: {f['avg_time_ms']:.2f}ms")
            report.append(f"- Avg Distance: {f['avg_distance']:.4f}\n")

    # 4. Text Search
    report.append("\n## 4. Text Search Performance\n")

    if "text_evaluation" in all_results:
        report.append("| Keyword | Mode | k | Results | Time (ms) |")
        report.append("|---------|------|---|---------|-----------|")
        for key, data in all_results["text_evaluation"].items():
            if "error" not in data:
                report.append(f"| {data['keyword']} | {data['mode']} | {data['k']} | {data['result_count']} | {data['time_ms']:.2f} |")

    # 5. Recommendations
    report.append("\n## 5. Recommendations\n")
    report.append("Based on the evaluation results:\n")
    report.append("1. **Descriptor quality:** This evaluation has no human relevance labels, so it cannot rank semantic quality.")
    report.append("2. **Index choice:** Use FAISS for supported metrics; verify its top-k Jaccard against brute force for each descriptor and metric.")
    report.append("3. **Metric choice:** Select a distance metric based on descriptor properties and task requirements.")
    report.append("4. **Fusion:** Treat multi-descriptor fusion as experimental until evaluated with relevance judgments.")

    return "\n".join(report)


def main():
    print("=" * 60)
    print("CBIR System Comprehensive Evaluation")
    print("=" * 60)

    manager = get_index_manager()
    service = get_search_service()

    all_results = {}

    # 1. Descriptor Evaluation
    print("\n=== Descriptor Evaluation ===")

    descriptor_configs = {
        "vgg16": ["euclidean", "cosine"],
        "edgehistogram": ["euclidean", "cosine", "chi2", "l1"],
        "homogeneoustexture": ["euclidean", "cosine", "l1"],
    }

    all_results["descriptor_evaluation"] = []

    for desc, metrics in descriptor_configs.items():
        print(f"\n{desc.upper()}:")
        results = evaluate_descriptor(
            manager,
            desc,
            metrics,
            ["brute", "faiss"],
            TEST_QUERY_IDS,
            [10, 20, 50]
        )
        all_results["descriptor_evaluation"].extend(results)

    # 2. Index Type Comparison
    print("\n" + "=" * 60)
    all_results["index_comparison"] = {}

    for desc in ["vgg16", "edgehistogram"]:
        if "euclidean" not in manager.available_faiss_metrics(desc):
            print(f"Skipping {desc} index comparison: no compatible Euclidean FAISS index")
            continue
        all_results["index_comparison"][desc] = compare_index_types(
            manager, desc, TEST_QUERY_IDS
        )

    # 3. Multi-Descriptor Fusion
    print("\n" + "=" * 60)
    all_results["fusion_evaluation"] = evaluate_multi_descriptor_fusion(
        manager, TEST_QUERY_IDS[:10]
    )

    # 4. Text Search
    print("\n" + "=" * 60)
    all_results["text_evaluation"] = evaluate_text_search(service, TEST_KEYWORDS, [20])

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")

    # Convert dataclasses to dicts and numpy types to Python types
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    serializable_results = {}
    for key, value in all_results.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], EvaluationResult):
            serializable_results[key] = convert_to_serializable([asdict(r) for r in value])
        else:
            serializable_results[key] = convert_to_serializable(value)

    # JSON output
    json_path = OUTPUT_DIR / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Saved: {json_path}")

    # Markdown report
    report = generate_report(all_results)
    report_path = OUTPUT_DIR / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
