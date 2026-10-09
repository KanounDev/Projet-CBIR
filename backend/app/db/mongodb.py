"""MongoDB connection and queries for CBIR.

Provides:
- Connection management (singleton pattern)
- Image lookup by ID
- Batch image lookup by IDs
- Keyword search (ET/OU modes)
"""
import os
from pathlib import Path
from typing import Optional

from pymongo import MongoClient, ASCENDING

# Configuration
DEFAULT_URI = "mongodb://localhost:27017"
DEFAULT_DB = "cbir"

_client: Optional[MongoClient] = None


def get_db(uri: str = None, db_name: str = None):
    """Get MongoDB database connection (singleton).

    Args:
        uri: MongoDB connection URI (default: from env or localhost)
        db_name: Database name (default: from env or 'cbir')

    Returns:
        MongoDB database object
    """
    global _client
    if _client is None:
        uri = uri or os.environ.get("MONGO_URI", DEFAULT_URI)
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db_name = db_name or os.environ.get("MONGO_DB", DEFAULT_DB)
    return _client[db_name]


def get_collection(collection: str = "images"):
    """Get a MongoDB collection.

    Args:
        collection: Collection name (default: 'images')

    Returns:
        MongoDB collection object
    """
    return get_db()[collection]


def get_image_by_id(image_id: int):
    """Get a single image document by its ID.

    Args:
        image_id: Image ID (0 to 999999 for MIRFLICKR)

    Returns:
        Image document or None if not found
    """
    return get_collection().find_one({"_id": image_id})


def get_images_by_ids(image_ids: list):
    """Get multiple image documents by their IDs.

    Args:
        image_ids: List of image IDs

    Returns:
        List of image documents (in the order of image_ids)
    """
    if not image_ids:
        return []

    # Query all images with these IDs
    cursor = get_collection().find({"_id": {"$in": image_ids}})

    # Build a dict for fast lookup
    images_by_id = {img["_id"]: img for img in cursor}

    # Return in the requested order
    return [images_by_id.get(i) for i in image_ids if i in images_by_id]


def check_connection():
    """Check if MongoDB connection is working.

    Returns:
        True if connected, raises exception otherwise
    """
    get_db().command("ping")
    return True


def get_stats():
    """Get database statistics.

    Returns:
        Dict with collection stats
    """
    col = get_collection()
    return {
        "total_images": col.count_documents({}),
        "images_with_keywords": col.count_documents({"keywords": {"$exists": True, "$ne": []}}),
        "indexes": [i["name"] for i in col.list_indexes()],
    }


if __name__ == "__main__":
    # Quick test
    print("Testing MongoDB connection...")
    try:
        check_connection()
        stats = get_stats()
        print(f"✅ Connected to MongoDB")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Images with keywords: {stats['images_with_keywords']}")
        print(f"   Indexes: {stats['indexes']}")

        # Test retrieval
        img = get_image_by_id(0)
        if img:
            print(f"\n   Example image (id=0):")
            print(f"   - thumb: {img.get('thumb')}")
            print(f"   - keywords: {img.get('keywords', [])[:5]}...")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
