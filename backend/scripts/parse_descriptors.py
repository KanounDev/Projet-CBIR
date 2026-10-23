"""Step 3 - convert the MPEG-7 text files (one file per image) into one .npy matrix.

Usage (from C:\\projet-CBIR, environment active):
    python backend/scripts/03_parse_descriptors.py edgehistogram --limit 1000    # quick test
    python backend/scripts/03_parse_descriptors.py edgehistogram
    python backend/scripts/03_parse_descriptors.py homogeneoustexture

Output: data/features/<name>.npy   shape (1000000, dim), float32
Row i = file i.txt = image number i+1 (im<i+1>.jpg).
"""
import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "mirflickr" / "raw"
OUT = ROOT / "data" / "features"
DIMS = {"edgehistogram": 150, "homogeneoustexture": 43}
NAME_RE = re.compile(r"^(\d+)\.txt$")  # ignores .DS_Store and feature_names.txt


def list_files(folder: Path):
    found = {}
    for root, _, files in os.walk(folder):
        for f in files:
            m = NAME_RE.match(f)
            if m:
                found[int(m.group(1))] = os.path.join(root, f)
    return found


def read_vector(path, dim):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        values = fh.read().split()
    return np.asarray(values, dtype=np.float32) if len(values) == dim else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=DIMS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids (test)")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    dim = DIMS[args.name]

    folder = RAW / f"features_{args.name}"
    print("Scanning", folder, "...")
    files = list_files(folder)
    ids = sorted(files)
    if args.limit:
        ids = ids[: args.limit]
    n = len(ids)
    print(f"{n} numbered .txt files, ids {ids[0]}..{ids[-1]}")
    if args.limit is None and ids != list(range(n)):
        print("WARNING: ids are not exactly 0..N-1, check for missing files")

    mat = np.zeros((n, dim), dtype=np.float32)
    bad = []
    t0 = time.time()
    with ThreadPoolExecutor(args.threads) as pool:
        for row, vec in enumerate(pool.map(lambda i: read_vector(files[i], dim), ids, chunksize=256)):
            if vec is None:
                bad.append(ids[row])
            else:
                mat[row] = vec
            if row % 50000 == 0 and row:
                print(f"  {row}/{n}  ({time.time() - t0:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / (args.name + ("_test" if args.limit else "") + ".npy")
    np.save(out_file, mat)
    np.save(OUT / (args.name + ("_test" if args.limit else "") + "_ids.npy"), np.asarray(ids, dtype=np.int32))
    print(f"Saved {out_file} shape={mat.shape} in {time.time() - t0:.0f}s")
    print(f"Files with a wrong number of values: {len(bad)}", bad[:10])
    print("min/max/mean:", mat.min(), mat.max(), mat.mean())


if __name__ == "__main__":
    main()