"""Step 4b - load data/metadata/metadata.jsonl into MongoDB (database cbir, collection images).

Usage (from C:\\projet-CBIR, environment active, MongoDB running):
    python backend/scripts/04b_load_mongo.py --test     # loads metadata_test.jsonl (1000 images)
    python backend/scripts/04b_load_mongo.py            # loads the full metadata.jsonl

The collection is emptied before each load, so the script can be re-run safely.
"""
import argparse
import json
import time
from pathlib import Path

from pymongo import ASCENDING, MongoClient

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "metadata"
BATCH = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="use metadata_test.jsonl")
    ap.add_argument("--uri", default="mongodb://localhost:27017")
    ap.add_argument("--db", default="cbir")
    args = ap.parse_args()

    src = META / ("metadata_test.jsonl" if args.test else "metadata.jsonl")
    client = MongoClient(args.uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # fails fast if MongoDB is not running
    col = client[args.db]["images"]
    col.drop()

    t0, n, batch = time.time(), 0, []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            rec["_id"] = rec["id"]  # id = _id, so lookups by id are free
            batch.append(rec)
            if len(batch) == BATCH:
                col.insert_many(batch, ordered=False)
                n += len(batch)
                batch = []
                print(f"  {n} inserted ({time.time() - t0:.0f}s)")
    if batch:
        col.insert_many(batch, ordered=False)
        n += len(batch)

    col.create_index([("keywords", ASCENDING)])  # multikey: $all = AND, $in = OR
    col.create_index([("license", ASCENDING)])
    print(f"Loaded {n} documents from {src.name} in {time.time() - t0:.0f}s; count in DB = {col.count_documents({})}")
    print("indexes:", [i["name"] for i in col.list_indexes()])
    print("example:", col.find_one({"keywords": {"$exists": True, "$ne": []}}, {"_id": 1, "keywords": 1, "thumb": 1}))


if __name__ == "__main__":
    main()