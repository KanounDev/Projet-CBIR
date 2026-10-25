"""Search for similar images using VP-Tree or FAISS indexes.

Usage (from C:\\projet-CBIR, environment active):
    # Using FAISS (recommended for 1M dataset)
    python backend/scripts/search_by_id.py --id 42 --method faiss --k 10

    # Using normalized features (L2 normalized)
    python backend/scripts/search_by_id.py --id 42 --method faiss --k 10 --normalized

    # Using VP-Tree (only works on test subset with 1k images)
    python backend/scripts/search_by_id.py --id 42 --method vptree --k 10 --test

    # Using brute-force (reference)
    python backend/scripts/search_by_id.py --id 42 --method brute --k 10

    # Save results as image montage
    python backend/scripts/search_by_id.py --id 42 --method faiss --k 10 --save

This script:
1. Loads the feature vector for the given image ID
2. Searches for k most similar images using the chosen method
3. Displays results with distances and optionally shows thumbnails
4. Optionally saves a montage image showing query + results
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.index_manager import IndexManager
from app.services.visual_search import VisualSearcher

DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
INDEXES_DIR = DATA_DIR / "indexes"
RAW_DIR = DATA_DIR / "mirflickr" / "raw"

DESCRIPTORS = {
    "edgehistogram": "Edge Histogram (MPEG-7)",
    "homogeneoustexture": "Homogeneous Texture (MPEG-7)",
}


def get_image_path(image_id: int) -> str:
    """Get thumbnail path for image ID (id = image_number - 1)."""
    # Images are numbered 1 to 1000000, IDs are 0 to 999999
    image_number = image_id + 1

    # Structure: thumbs1-33/thumbnails/<folder_num>/im<image_number>.jpg
    # Each folder contains 10000 images (1-10000 in folder 1, 10001-20000 in folder 2, etc.)
    folder_num = ((image_number - 1) // 10000) + 1  # 1 to 100

    if folder_num <= 33:
        thumbs_dir = RAW_DIR / "thumbs1-33" / "thumbnails" / str(folder_num)
    elif folder_num <= 66:
        thumbs_dir = RAW_DIR / "thumbs34-66" / "thumbnails" / str(folder_num)
    else:
        thumbs_dir = RAW_DIR / "thumbs67-100" / "thumbnails" / str(folder_num)

    # Image filename: im<number>.jpg
    filename = f"im{image_number}.jpg"
    path = thumbs_dir / filename

    if path.exists():
        return str(path)

    return None


def create_montage(query_id: int, result_ids: np.ndarray, distances: np.ndarray,
                   method: str, descriptor: str, output_path: Path,
                   thumb_size: int = 128, padding: int = 4):
    """Create a montage image showing query + k nearest neighbors.

    Layout: Query image on the left, then k results horizontally.
    Each image has a label showing rank and distance.

    Args:
        query_id: Query image ID
        result_ids: Array of result image IDs
        distances: Array of distances
        method: Search method name (for title)
        descriptor: Descriptor name (for title)
        output_path: Where to save the montage
        thumb_size: Size of each thumbnail
        padding: Padding between images
    """
    k = len(result_ids)
    total_images = k + 1  # query + results

    # Create blank canvas
    canvas_width = total_images * thumb_size + (total_images - 1) * padding + 20
    canvas_height = thumb_size + 60  # Extra space for labels
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 10)
        title_font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Add title
    title = f"Query: ID {query_id} | Method: {method.upper()} | Descriptor: {descriptor}"
    draw.text((10, 5), title, fill=(0, 0, 0), font=title_font)

    def add_image(img_id: int, x_pos: int, label: str, is_query: bool = False):
        """Add a single image to the canvas."""
        path = get_image_path(img_id)

        if path and os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            except:
                img = Image.new('RGB', (thumb_size, thumb_size), color=(200, 200, 200))
        else:
            img = Image.new('RGB', (thumb_size, thumb_size), color=(200, 200, 200))

        # Paste image
        y_pos = 25
        canvas.paste(img, (x_pos, y_pos))

        # Draw border (thicker for query)
        border_color = (255, 0, 0) if is_query else (100, 100, 100)
        border_width = 3 if is_query else 1
        draw.rectangle([x_pos-border_width, y_pos-border_width,
                       x_pos+thumb_size+border_width, y_pos+thumb_size+border_width],
                      outline=border_color, width=border_width)

        # Add label
        draw.text((x_pos, y_pos + thumb_size + 5), label, fill=(0, 0, 0), font=font)

    # Add query image
    add_image(query_id, 10, f"QUERY\nID {query_id}", is_query=True)

    # Add result images
    for rank, (img_id, dist) in enumerate(zip(result_ids, distances), 1):
        x_pos = 10 + rank * (thumb_size + padding)
        label = f"#{rank}\nID {img_id}\n{dist:.4f}"
        add_image(img_id, x_pos, label, is_query=False)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f"\n  ✓ Saved montage: {output_path}")
    return output_path


def display_results(query_id: int, results: dict, show_paths: bool = False,
                   save_montage: bool = False, descriptor: str = 'edgehistogram'):
    """Display search results in a formatted table and optionally save montage."""
    print(f"\n{'='*70}")
    print(f"Query Image: ID {query_id} (im{query_id+1}.jpg)")
    print(f"{'='*70}")

    if show_paths:
        query_path = get_image_path(query_id)
        print(f"Thumbnail: {query_path}")

    for method_name, (ids, distances, query_time) in results.items():
        print(f"\n{method_name.upper()} Results ({query_time:.2f}ms):")
        print(f"{'Rank':<6} {'ID':<10} {'Image':<15} {'Distance':<12}")
        print("-" * 50)

        for rank, (img_id, dist) in enumerate(zip(ids, distances), 1):
            image_name = f"im{img_id+1}.jpg"
            print(f"{rank:<6} {img_id:<10} {image_name:<15} {dist:<12.4f}")

            if show_paths and rank <= 3:  # Show paths for top 3
                path = get_image_path(img_id)
                if path:
                    print(f"       Path: {path}")

        # Save montage if requested
        if save_montage:
            output_dir = DATA_DIR / "search_results"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"search_{query_id}_{method_name}_{descriptor}_{timestamp}.jpg"
            output_path = output_dir / filename

            create_montage(
                query_id=query_id,
                result_ids=ids,
                distances=distances,
                method=method_name,
                descriptor=descriptor,
                output_path=output_path
            )


def search_with_index_manager(query_id: int, method: str, descriptor: str,
                               metric: str, k: int, use_test: bool, use_normalized: bool):
    """Search using IndexManager (chooses best available index)."""
    manager = IndexManager(data_dir=DATA_DIR, use_faiss=(method in ['faiss', 'both']))

    # Load the query vector
    suffix = "_test" if use_test else ""
    norm_suffix = "_normalized" if use_normalized else ""
    feature_file = FEATURES_DIR / f"{descriptor}{suffix}{norm_suffix}.npy"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    X = np.load(feature_file, mmap_mode="r")

    if query_id >= len(X):
        raise ValueError(f"Query ID {query_id} out of range (max: {len(X)-1})")

    q = np.asarray(X[query_id], dtype=np.float32)

    # Search
    t0 = time.time()
    ids, dists = manager.search(descriptor, q, k=k, metric=metric, exclude_id=query_id)
    query_time = (time.time() - t0) * 1000

    return ids, dists, query_time


def search_brute_force(query_id: int, descriptor: str, metric: str, k: int,
                       use_test: bool, use_normalized: bool):
    """Search using brute force (VisualSearcher)."""
    suffix = "_test" if use_test else ""
    norm_suffix = "_normalized" if use_normalized else ""

    # Create searcher with only the needed descriptor
    searcher = VisualSearcher(
        data_dir=DATA_DIR,
        descriptors=(descriptor,)
    )

    # Load to ensure we're using the right subset
    feature_file = FEATURES_DIR / f"{descriptor}{suffix}{norm_suffix}.npy"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    searcher.X = {descriptor: np.load(feature_file, mmap_mode="r")}
    searcher.n = len(searcher.X[descriptor])

    t0 = time.time()
    ids, dists = searcher.search(query_id, weights={descriptor: 1.0}, metric=metric, k=k+1)
    query_time = (time.time() - t0) * 1000

    # Exclude the query itself from results
    mask = ids != query_id
    ids, dists = ids[mask][:k], dists[mask][:k]

    return ids, dists, query_time


def main():
    ap = argparse.ArgumentParser(description="Search for similar images by ID")
    ap.add_argument("--id", type=int, required=True, help="Query image ID (0-999999)")
    ap.add_argument("--method", choices=['faiss', 'vptree', 'brute', 'both'],
                   default='faiss', help="Search method")
    ap.add_argument("--descriptor", choices=list(DESCRIPTORS.keys()),
                   default='edgehistogram', help="Feature descriptor")
    ap.add_argument("--metric", default='euclidean', help="Distance metric")
    ap.add_argument("--k", type=int, default=10, help="Number of results")
    ap.add_argument("--test", action="store_true", help="Use test subset (1k images)")
    ap.add_argument("--show-paths", action="store_true", help="Show image paths")
    ap.add_argument("--compare", action="store_true", help="Compare all methods")
    ap.add_argument("--save", action="store_true", help="Save montage image to data/search_results/")
    ap.add_argument("--normalized", action="store_true", help="Use L2-normalized features")
    args = ap.parse_args()

    print(f"Search Settings:")
    print(f"  Query ID: {args.id} (image im{args.id+1}.jpg)")
    print(f"  Method: {args.method}")
    print(f"  Descriptor: {DESCRIPTORS[args.descriptor]}")
    print(f"  Metric: {args.metric}")
    print(f"  k: {args.k}")
    print(f"  Dataset: {'Test (1k)' if args.test else 'Full (1M)'}")
    print(f"  Features: {'Normalized (L2)' if args.normalized else 'Original'}")

    results = {}

    # Determine which methods to run
    methods = []
    if args.compare or args.method == 'both':
        methods = ['brute', 'faiss']
        if args.test:  # VP-Tree only available on small subsets
            methods.insert(1, 'vptree')
    else:
        methods = [args.method]

    # Run searches
    for method in methods:
        try:
            if method == 'brute':
                ids, dists, query_time = search_brute_force(
                    args.id, args.descriptor, args.metric, args.k,
                    args.test, args.normalized
                )
            else:
                ids, dists, query_time = search_with_index_manager(
                    args.id, method, args.descriptor, args.metric, args.k,
                    args.test, args.normalized
                )

            results[method] = (ids, dists, query_time)
            print(f"  ✓ {method.upper()} search completed in {query_time:.2f}ms")

        except FileNotFoundError as e:
            print(f"  ✗ {method.upper()} index not found: {e}")
            if args.normalized:
                print(f"     (Normalized features might not exist. Run: python backend/scripts/normalize_features.py)")
        except Exception as e:
            print(f"  ✗ {method.upper()} error: {type(e).__name__}: {e}")

    # Display results
    if results:
        display_results(args.id, results, show_paths=args.show_paths,
                       save_montage=args.save, descriptor=args.descriptor)

        # Compare methods if multiple were run
        if len(results) > 1:
            print(f"\n{'='*70}")
            print("METHOD COMPARISON:")
            print(f"{'='*70}")
            print(f"{'Method':<15} {'Query Time (ms)':<20} {'Top-1 ID':<12} {'Top-1 Dist':<12}")
            print("-" * 70)
            for method, (ids, dists, query_time) in results.items():
                print(f"{method.upper():<15} {query_time:<20.2f} {ids[0]:<12} {dists[0]:<12.4f}")

            # Check if results match
            if 'brute' in results and 'faiss' in results:
                brute_ids = set(results['brute'][0])
                faiss_ids = set(results['faiss'][0])
                overlap = len(brute_ids & faiss_ids)
                print(f"\nRecall@{args.k}: {overlap}/{args.k} = {100*overlap/args.k:.1f}%")
    else:
        print("\nNo results obtained. Check that indexes exist.")
        print("Run: python backend/scripts/build_indexes.py --faiss")


if __name__ == "__main__":
    main()
