"""Image upload endpoint for external query images.

Allows users to upload their own image for visual search.
The server extracts features (VGG16, Edge Histogram, etc.) and returns
similar images from the database.
"""
import io
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.schemas.search import SearchResponse, SearchResultItem
from app.services.search_service import get_search_service, SearchRequest as InternalSearchRequest
from app.indexing.index_manager import get_index_manager

router = APIRouter()

# Configuration
# Note: Uploaded images are processed in memory, NOT saved to disk

# Descriptors that can be used for uploaded images.
# Edge Histogram and Homogeneous Texture are only simplified approximations
# (the texture one even uses random placeholders), so their vectors are NOT in the
# same space as the pre-extracted database features. Add them back here only once
# the extractors really match the offline ones.
ALLOWED_UPLOAD_DESCRIPTORS = {"vgg16"}

# Metrics supported per descriptor (mirrors /descriptors in routes.py)
ALLOWED_METRICS = {
    "vgg16": {"euclidean", "cosine"},
    "edgehistogram": {"euclidean", "chi2", "l1", "cosine"},
    "homogeneoustexture": {"euclidean", "cosine", "l1"},
}

# Cached VGG16 extractor (loaded once, on first use)
_vgg16_extractor = None


def resize_image(image: Image.Image, max_size: int = 224) -> Image.Image:
    """Resize image to max dimension while preserving aspect ratio.

    VGG16 expects 224x224, so we resize to that for deep learning.
    For MPEG-7 descriptors, we use the original size (they handle different sizes).

    Args:
        image: PIL Image
        max_size: Maximum dimension (width or height)

    Returns:
        Resized PIL Image (RGB)
    """
    # Convert to RGB if necessary (handles PNG with alpha, grayscale, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Calculate new size preserving aspect ratio
    width, height = image.size
    if max(width, height) > max_size:
        if width >= height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def extract_features_from_image(image: Image.Image, descriptors: List[str]) -> dict:
    """Extract features from an uploaded image.

    Args:
        image: PIL Image (already resized)
        descriptors: List of descriptors to extract

    Returns:
        Dict mapping descriptor name to feature vector
    """
    features = {}

    for desc in descriptors:
        if desc == "vgg16":
            # VGG16 extraction
            features[desc] = extract_vgg16_features(image)
        elif desc == "edgehistogram":
            # Edge Histogram extraction (MPEG-7)
            features[desc] = extract_edge_histogram(image)
        elif desc == "homogeneoustexture":
            # Homogeneous Texture extraction (MPEG-7)
            features[desc] = extract_homogeneous_texture(image)
        else:
            raise ValueError(f"Unknown descriptor: {desc}")

    return features


def extract_vgg16_features(image: Image.Image) -> np.ndarray:
    """Extract VGG16 features from an image.

    Uses the SAME method as the pre-extracted database features:
    - Resize to 64x64 (like MIRFLICKR thumbnails)
    - Upscale to 224x224
    - VGG16 conv features + Global Average Pooling -> 512-dim
    - L2 normalized

    Args:
        image: PIL Image

    Returns:
        Feature vector (512,) as float32
    """
    from app.descriptors.vgg import pil_to_uint8, build_extractor, embed

    # Load extractor (cached globally)
    global _vgg16_extractor
    if _vgg16_extractor is None:
        _vgg16_extractor = build_extractor(device="cpu")

    # Convert PIL image to uint8 tensor (3, 64, 64)
    # This resizes to 64x64 to match MIRFLICKR thumbnails
    x_uint8 = pil_to_uint8(image).unsqueeze(0)  # (1, 3, 64, 64)

    # Extract features (L2 normalized)
    features = embed(_vgg16_extractor, x_uint8, device="cpu")

    return features[0]  # (512,)


def extract_edge_histogram(image: Image.Image) -> np.ndarray:
    """Extract Edge Histogram (MPEG-7) from an image.

    Simplified implementation - returns a feature vector compatible with MIRFLICKR.

    Note: This is a simplified version. For full MPEG-7 compatibility,
    use a proper implementation like LIRE or implement the full standard.

    Args:
        image: PIL Image

    Returns:
        Feature vector (150,) as float32
    """
    import cv2

    # Convert to grayscale
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Resize to 64x64 (thumbnail size for consistency)
    gray = cv2.resize(gray, (64, 64))

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Divide into 4x4 blocks
    block_size = 16
    histograms = []

    for i in range(4):
        for j in range(4):
            block = edges[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            # Simplified: count edge pixels in each block
            hist = np.histogram(block, bins=5, range=(0, 255))[0]
            histograms.append(hist)

    # Global histogram
    global_hist = np.histogram(edges, bins=5, range=(0, 255))[0]

    # Semi-global histograms (4 vertical, 4 horizontal)
    semi_global = []
    for i in range(4):
        semi_global.append(np.histogram(edges[:, i*16:(i+1)*16], bins=5, range=(0, 255))[0])
    for i in range(4):
        semi_global.append(np.histogram(edges[i*16:(i+1)*16, :], bins=5, range=(0, 255))[0])

    # Combine: 80 local + 5 global + 40 semi-global (simplified)
    # For full 150-dim, we'd need proper edge direction quantization
    feature = np.concatenate([np.array(h).flatten() for h in histograms] + [global_hist] + semi_global)

    # Pad or truncate to 150 dimensions
    if len(feature) < 150:
        feature = np.pad(feature, (0, 150 - len(feature)))
    else:
        feature = feature[:150]

    return feature.astype(np.float32)


def extract_homogeneous_texture(image: Image.Image) -> np.ndarray:
    """Extract Homogeneous Texture (MPEG-7) from an image.

    Simplified implementation. WARNING: part of the vector is random placeholder
    data, so results are not meaningful. Not enabled for uploads (see
    ALLOWED_UPLOAD_DESCRIPTORS).

    Args:
        image: PIL Image

    Returns:
        Feature vector (43,) as float32
    """
    import cv2

    # Convert to grayscale
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Resize to 64x64
    gray = cv2.resize(gray, (64, 64))

    # Compute mean and std (DC and SD)
    mean_val = np.mean(gray)
    std_val = np.std(gray)

    # Gabor filters for texture analysis (simplified)
    # In full MPEG-7, this uses 30 Gabor filters (5 scales x 6 orientations)
    feature = [mean_val / 255.0, std_val / 255.0]

    # Add some texture statistics
    for scale in [2, 4, 8, 16, 32]:
        kernel = np.ones((scale, scale)) / (scale * scale)
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        feature.append(np.std(filtered) / 255.0)

    for angle in [0, 30, 60, 90, 120, 150]:
        # Simplified directional features
        feature.append(np.random.rand())  # Placeholder

    # Pad to 43 dimensions
    while len(feature) < 43:
        feature.append(np.random.rand() * 0.1)

    return np.array(feature[:43], dtype=np.float32)


@router.post("/search/upload", response_model=SearchResponse, tags=["search"])
async def search_by_uploaded_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG, etc.)"),
    # NOTE: these are multipart FORM fields (the frontend sends them in FormData),
    # not query parameters. Without Form(...) FastAPI would ignore them and use defaults.
    descriptors: str = Form("vgg16", description="Comma-separated: 'vgg16,edgehistogram'"),
    metric: str = Form("euclidean"),
    index_type: str = Form("faiss"),
    k: int = Form(20),
):
    """Search for similar images using an uploaded image.

    The server extracts features from the uploaded image and searches
    the database for visually similar images.

    Args:
        file: Uploaded image file
        descriptors: Comma-separated list of descriptors (default: vgg16)
        metric: Distance metric (default: euclidean)
        index_type: Index type (default: faiss)
        k: Number of results (default: 20)

    Returns:
        Search results with similar images, sorted by distance (ascending)
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Parse and validate descriptors
    desc_list = [d.strip() for d in descriptors.split(",") if d.strip()]
    if not desc_list:
        desc_list = ["vgg16"]

    unsupported = [d for d in desc_list if d not in ALLOWED_UPLOAD_DESCRIPTORS]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Descriptor(s) not supported for uploaded images: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(ALLOWED_UPLOAD_DESCRIPTORS))}"
            ),
        )

    if len(desc_list) > 1:
        raise HTTPException(status_code=501, detail="Multi-descriptor search for uploaded images not yet implemented")

    desc = desc_list[0]

    # Validate metric for this descriptor
    allowed = ALLOWED_METRICS.get(desc, set())
    if metric not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Metric '{metric}' is not supported for {desc}. Supported: {', '.join(sorted(allowed))}",
        )

    k = max(1, min(200, k))

    # Read and process image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = resize_image(image, max_size=224)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    # Extract features
    try:
        features = extract_features_from_image(image, desc_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    # Search using extracted features
    try:
        manager = get_index_manager()
        query_vec = features[desc]

        ids, dists = manager.search(
            descriptor=desc,
            query=query_vec,
            metric=metric,
            index_type=index_type,
            k=k,
        )

        # Keep the ranking returned by the index (FAISS pads with -1 if < k results)
        ids = [int(i) for i in np.asarray(ids).ravel()]
        dists = [float(d) for d in np.asarray(dists).ravel()]
        dist_dict = {i: d for i, d in zip(ids, dists) if i >= 0}

        # Get metadata (MongoDB does NOT return documents in the order of the ids)
        from app.db.mongodb import get_images_by_ids
        images = get_images_by_ids(list(dist_dict.keys()))

        results_list = []
        for img in images:
            thumb = img.get("thumb", "")
            results_list.append(SearchResultItem(
                id=img["_id"],
                distance=dist_dict.get(int(img["_id"]), 0.0),
                thumb=thumb,
                thumb_url=f"/static/thumbs/{thumb}" if thumb else "",
                keywords=img.get("keywords", []),
                photo_url=img.get("photo_url"),
                web_url=img.get("web_url"),
                license=img.get("license"),
            ))

        # Restore the distance ranking
        results_list.sort(key=lambda r: r.distance)

    except HTTPException:
        # Let deliberate HTTP errors (e.g. 501) pass through unchanged
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Feature or index not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return SearchResponse(
        query_image=None,  # No ID for uploaded images
        results=results_list,
        total=len(results_list),
        search_mode="visual",
    )