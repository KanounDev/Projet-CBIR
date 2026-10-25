"""Step 4b - load data/metadata/metadata.jsonl into MongoDB (database cbir, collection images).

Usage with Docker Compose (loads into the MongoDB used by the API):
    docker-compose exec -T backend python backend/scripts/load_mongo.py

Usage with a local backend/MongoDB:
    python backend/scripts/load_mongo.py --test     # loads metadata_test.jsonl (1000 images)
    python backend/scripts/load_mongo.py            # loads the full metadata.jsonl

By default the collection is replaced. Use --if-empty for idempotent startup.
"""
import argparse
import json
import os
import time
from pathlib import Path

from pymongo import ASCENDING, MongoClient

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "metadata"
BATCH = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="use metadata_test.jsonl")
    ap.add_argument(
        "--if-empty",
        action="store_true",
        help="skip loading when the images collection already contains documents",
    )
    ap.add_argument("--expected-count", type=int, default=None)
    ap.add_argument("--uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db", default=os.environ.get("MONGO_DB", "cbir"))
    args = ap.parse_args()

    src = META / ("metadata_test.jsonl" if args.test else "metadata.jsonl")
    if not src.is_file():
        raise FileNotFoundError(f"Metadata file not found: {src}")
    client = MongoClient(args.uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # fails fast if MongoDB is not running
    col = client[args.db]["images"]
    existing_count = col.estimated_document_count()
    if args.if_empty and existing_count:
        if args.expected_count is not None and existing_count != args.expected_count:
            raise RuntimeError(
                f"{args.db}.images contains {existing_count} documents; "
                f"expected {args.expected_count}. Run without --if-empty to reload it."
            )
        print(f"Skipping import; {args.db}.images already contains {existing_count} documents.")
        return
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