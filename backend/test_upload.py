"""Test script for image upload endpoint.

Usage:
    python test_upload.py <image_path>

Example:
    python test_upload.py test_image.jpg
"""
import sys
import requests

API_BASE = "http://localhost:8000"


def test_upload(image_path: str):
    """Test the upload endpoint with an image file."""

    # Check server health
    print("1. Checking server health...")
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=5)
        print(f"   Status: {resp.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    # Test upload
    print(f"\n2. Uploading image: {image_path}")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            data = {
                "descriptors": "vgg16",
                "metric": "euclidean",
                "index_type": "faiss",
                "k": "5"
            }
            resp = requests.post(f"{API_BASE}/api/search/upload", files=files, data=data, timeout=60)

        if resp.status_code != 200:
            print(f"   ERROR: Status {resp.status_code}")
            print(f"   Response: {resp.text}")
            return

        result = resp.json()
        print(f"   Search mode: {result['search_mode']}")
        print(f"   Total results: {result['total']}")
        print(f"\n   Top 5 results:")
        for i, r in enumerate(result['results'][:5], 1):
            print(f"   {i}. ID={r['id']}, distance={r['distance']:.4f}, keywords={r['keywords'][:3]}")

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_upload.py <image_path>")
        sys.exit(1)

    test_upload(sys.argv[1])
