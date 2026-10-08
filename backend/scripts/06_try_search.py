"""Step 6 - try the brute-force search and look at the results.

Usage (from C:\\projet-CBIR, environment active):
    python backend\\scripts\\06_try_search.py                 # random query image
    python backend\\scripts\\06_try_search.py --id 339 --k 10

Prints the top results for several settings, with timings, and saves a picture
data\\search_demo_<id>.png (query on the left, then the 10 closest thumbnails) for each setting.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.visual_search import VisualSearcher  # noqa: E402

RAW = ROOT / "data" / "mirflickr" / "raw"
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

CONFIGS = [
    ("EH euclidean", {"edgehistogram": 1.0}, "euclidean"),
    ("EH L1", {"edgehistogram": 1.0}, "l1"),
    ("EH chi2", {"edgehistogram": 1.0}, "chi2"),
    ("HT euclidean", {"homogeneoustexture": 1.0}, "euclidean"),
    ("EH+HT fusion 50/50 (L1)", {"edgehistogram": 0.5, "homogeneoustexture": 0.5}, "l1"),
]


def thumb_paths(ids):
    try:
        from pymongo import MongoClient
        col = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)["cbir"]["images"]
        docs = {d["_id"]: d.get("thumb") for d in col.find({"_id": {"$in": [int(i) for i in ids]}}, {"thumb": 1})}
        return docs
    except Exception as e:  # MongoDB not running, etc.
        print("  (no picture: MongoDB not reachable:", type(e).__name__, ")")
        return {}


def save_montage(query_id, ids, label, out_dir):
    from PIL import Image
    paths = thumb_paths([query_id, *ids])
    if not paths:
        return
    size, tiles = 128, []
    for i in [query_id, *ids]:
        p = paths.get(int(i))
        tiles.append(Image.open(RAW / p).convert("RGB").resize((size, size)) if p else Image.new("RGB", (size, size)))
    sheet = Image.new("RGB", (size * len(tiles) + 12, size), "white")
    for n, t in enumerate(tiles):
        sheet.paste(t, (n * size + (12 if n else 0), 0))
    slug = re.sub(r"\W+", "_", label).strip("_")
    out = out_dir / f"search_demo_{query_id}_{slug}.png"
    sheet.save(out)
    print("  picture:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-picture", action="store_true")
    args = ap.parse_args()

    t = time.time()
    searcher = VisualSearcher()
    print(f"Loaded {searcher.n} vectors in {time.time() - t:.1f}s")
    qid = args.id if args.id is not None else int(np.random.default_rng().integers(0, searcher.n))
    print(f"Query: id {qid} (image im{qid + 1}.jpg)")

    for label, weights, metric in CONFIGS:
        for name in weights:  # warm-up: compute the normalisation scale outside the timing
            searcher.scale(name, metric)
        t = time.time()
        ids, dist = searcher.search(qid, weights, metric, k=args.k)
        dt = time.time() - t
        self_ok = "OK" if ids[0] == qid and dist[0] < 1e-5 else "CHECK: query not first"
        print(f"\n{label}: {dt:.2f}s  [{self_ok}]")
        print("  ids:", ids.tolist())
        print("  dist:", np.round(dist, 3).tolist())
        if not args.no_picture:
            save_montage(qid, ids, label, ROOT / "data")


if __name__ == "__main__":
    main()
