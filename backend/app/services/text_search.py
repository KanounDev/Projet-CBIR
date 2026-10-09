"""Text-based search using MongoDB keyword queries.

Provides:
- Keyword search with ET (AND) mode: all keywords must match
- Keyword search with OU (OR) mode: at least one keyword must match
- Normalization: lowercase, strip whitespace
"""
import re
from typing import List, Optional

from app.db.mongodb import get_collection


def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword for search.

    - Lowercase
    - Strip whitespace
    - Remove special characters (keep alphanumeric and hyphens)

    Args:
        keyword: Raw keyword string

    Returns:
        Normalized keyword
    """
    if not keyword:
        return ""
    # Lowercase and strip
    kw = keyword.lower().strip()
    # Keep only alphanumeric, hyphens, underscores
    kw = re.sub(r"[^\w\-]", "", kw)
    return kw


def normalize_keywords(keywords: List[str]) -> List[str]:
    """Normalize a list of keywords.

    Args:
        keywords: List of raw keywords

    Returns:
        List of normalized keywords (empty strings removed)
    """
    return [kw for kw in (normalize_keyword(k) for k in keywords) if kw]


def search_by_keywords(keywords: List[str], mode: str = "and", limit: int = 1000):
    """Search images by keywords using MongoDB.

    Args:
        keywords: List of keywords to search for
        mode: "and" (all keywords must match) or "or" (at least one must match)
        limit: Maximum number of results to return

    Returns:
        List of image IDs matching the query
    """
    keywords = normalize_keywords(keywords)
    if not keywords:
        # No keywords = return all images (limited)
        cursor = get_collection().find({}, {"_id": 1}).limit(limit)
        return [doc["_id"] for doc in cursor]

    # Build MongoDB query
    if mode == "and":
        # $all: all keywords must be present
        query = {"keywords": {"$all": keywords}}
    else:  # mode == "or"
        # $in: at least one keyword must be present
        query = {"keywords": {"$in": keywords}}

    cursor = get_collection().find(query, {"_id": 1}).limit(limit)
    return [doc["_id"] for doc in cursor]


def search_by_keywords_with_metadata(keywords: List[str], mode: str = "and", limit: int = 1000):
    """Search images by keywords and return full metadata.

    Args:
        keywords: List of keywords to search for
        mode: "and" or "or"
        limit: Maximum number of results

    Returns:
        List of image documents matching the query
    """
    keywords = normalize_keywords(keywords)
    if not keywords:
        cursor = get_collection().find({}).limit(limit)
        return list(cursor)

    if mode == "and":
        query = {"keywords": {"$all": keywords}}
    else:
        query = {"keywords": {"$in": keywords}}

    cursor = get_collection().find(query).limit(limit)
    return list(cursor)


def get_keywords_for_image(image_id: int) -> List[str]:
    """Get keywords for a specific image.

    Args:
        image_id: Image ID

    Returns:
        List of keywords (or empty list if not found)
    """
    doc = get_collection().find_one({"_id": image_id}, {"keywords": 1})
    return doc.get("keywords", []) if doc else []


if __name__ == "__main__":
    # Quick test
    print("Testing text search...")

    # Test normalization
    print("\n1. Normalization test:")
    test_kws = ["  SUNSET  ", "Beach!", "ocean@"]
    normalized = normalize_keywords(test_kws)
    print(f"   Input: {test_kws}")
    print(f"   Output: {normalized}")

    # Test search
    print("\n2. Search test (AND mode):")
    results = search_by_keywords(["sunset", "beach"], mode="and", limit=5)
    print(f"   'sunset' AND 'beach': {len(results)} results, first 5: {results}")

    print("\n3. Search test (OR mode):")
    results = search_by_keywords(["sunset", "beach"], mode="or", limit=5)
    print(f"   'sunset' OR 'beach': {len(results)} results, first 5: {results}")

    # Test with metadata
    print("\n4. Search with metadata:")
    docs = search_by_keywords_with_metadata(["sunset"], mode="and", limit=2)
    for doc in docs[:2]:
        print(f"   Image {doc['_id']}: {doc.get('keywords', [])[:5]}...")
