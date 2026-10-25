"""FastAPI routes for CBIR search API.

Endpoints:
- GET /health - Health check
- GET /descriptors - List available descriptors
- GET /images - Browse/paginate images
- POST /search - Search by keywords, image, or both
"""
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    DescriptorInfo,
    ImageInfo,
    ImageListResponse,
    HealthResponse,
)
from app.db.mongodb import get_collection, get_image_by_id, get_images_by_ids, check_connection, get_stats
from app.services.text_search import normalize_keywords
from app.services.search_service import get_search_service, SearchRequest as InternalSearchRequest
from app.indexing.index_manager import get_index_manager, FAISS_AVAILABLE
from app.indexing.metrics import SUPPORTED_METRICS

# Configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[3] / "data"))

# Create router
router = APIRouter()


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Check service health and dependencies.

    Returns:
        Health status including MongoDB connection and loaded features
    """
    # Check MongoDB
    try:
        check_connection()
        mongodb_status = "connected"
    except Exception as e:
        mongodb_status = f"error: {str(e)[:50]}"

    # Check loaded features
    manager = get_index_manager()
    features_loaded = []
    indexes_available = []

    for desc in ["vgg16", "edgehistogram", "homogeneoustexture"]:
        info = {
            "vgg16": "vgg16_gap512_1M.npy",
            "edgehistogram": "edgehistogram.npy",
            "homogeneoustexture": "homogeneoustexture.npy",
        }
        if (manager.features_dir / info[desc]).is_file():
            features_loaded.append(desc)

    indexes_available = sorted(
        path.name
        for pattern in ("*.faiss", "*.vptree")
        for path in manager.indexes_dir.glob(pattern)
        if path.is_file()
    )

    return HealthResponse(
        status="ok" if mongodb_status == "connected" else "degraded",
        mongodb=mongodb_status,
        features_loaded=features_loaded,
        indexes_available=indexes_available[:20],
    )


# ============================================================================
# Descriptors
# ============================================================================

@router.get("/descriptors", response_model=List[DescriptorInfo], tags=["search"])
async def list_descriptors():
    """List available image descriptors.

    Returns:
        List of available descriptors with their properties
    """
    manager = get_index_manager()
    descriptors = [
        DescriptorInfo(
            name="vgg16",
            dimension=512,
            description="VGG16 deep learning features (conv + GAP), best for semantic similarity",
            available_metrics=["euclidean", "cosine"],
            available_indexes=_available_indexes(manager, "vgg16"),
            available_metrics_by_index=_available_metrics_by_index(manager, "vgg16"),
        ),
        DescriptorInfo(
            name="edgehistogram",
            dimension=150,
            description="MPEG-7 Edge Histogram, captures edge distribution",
            available_metrics=["euclidean", "chi2", "l1", "cosine"],
            available_indexes=_available_indexes(manager, "edgehistogram"),
            available_metrics_by_index=_available_metrics_by_index(manager, "edgehistogram"),
        ),
        DescriptorInfo(
            name="homogeneoustexture",
            dimension=43,
            description="MPEG-7 Homogeneous Texture, captures texture patterns",
            available_metrics=["euclidean", "cosine", "l1"],
            available_indexes=_available_indexes(manager, "homogeneoustexture"),
            available_metrics_by_index=_available_metrics_by_index(manager, "homogeneoustexture"),
        ),
    ]

    return descriptors


def _available_indexes(manager, descriptor: str) -> List[str]:
    """Report index implementations that have files available for this descriptor."""
    metrics_by_index = _available_metrics_by_index(manager, descriptor)
    available = ["brute"]
    if metrics_by_index["faiss"]:
        available.append("faiss")
    if metrics_by_index["vptree"]:
        available.append("vptree")
    return available


def _available_metrics_by_index(manager, descriptor: str):
    """Report only metric/index combinations that runtime can execute correctly."""
    return {
        "brute": sorted(SUPPORTED_METRICS[descriptor]),
        "faiss": manager.available_faiss_metrics(descriptor),
        "vptree": manager.available_vptree_metrics(descriptor),
    }


# ============================================================================
# Images
# ============================================================================

@router.get("/images", response_model=ImageListResponse, tags=["browse"])
async def list_images(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Images per page"),
):
    """Browse images with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of images per page

    Returns:
        List of images with metadata
    """
    collection = get_collection()

    # Get total count
    total = collection.count_documents({})

    # Paginate
    skip = (page - 1) * page_size
    cursor = collection.find({}).skip(skip).limit(page_size)

    images = []
    for doc in cursor:
        thumb_path = doc.get("thumb", "")
        images.append(ImageInfo(
            id=doc["_id"],
            thumb_url=f"/static/thumbs/{thumb_path}" if thumb_path else "",
            keywords=doc.get("keywords", []),
            photo_url=doc.get("photo_url"),
            web_url=doc.get("web_url"),
        ))

    return ImageListResponse(
        images=images,
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/images/{image_id}", response_model=ImageInfo, tags=["browse"])
async def get_image(image_id: int):
    """Get information about a specific image.

    Args:
        image_id: Image ID (0-999999)

    Returns:
        Image metadata
    """
    doc = get_image_by_id(image_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

    thumb_path = doc.get("thumb", "")
    return ImageInfo(
        id=doc["_id"],
        thumb_url=f"/static/thumbs/{thumb_path}" if thumb_path else "",
        keywords=doc.get("keywords", []),
        photo_url=doc.get("photo_url"),
        web_url=doc.get("web_url"),
    )


# ============================================================================
# Search
# ============================================================================

@router.post("/search", response_model=SearchResponse, tags=["search"])
def search(request: SearchRequest):
    """Search images by keywords, visual similarity, or both.

    ## Search Modes:

    1. **Text-only**: Provide `keywords` (no `query_image_id`)
       - Returns images matching keywords
       - Use `keyword_mode` to control AND/OR logic

    2. **Visual-only**: Provide `query_image_id` (no `keywords`)
       - Returns visually similar images
       - Specify `descriptors`, `metrics`, `index_type`

    3. **Combined**: Provide both `keywords` and `query_image_id`
       - Filters by keywords, ranks by visual similarity

    ## Parameters:

    - `descriptors`: Which feature(s) to use (vgg16 recommended)
    - `metrics`: Distance metric per descriptor
    - `weights`: Weight for each descriptor when combining
    - `index_type`: faiss (fast), vptree (demo), brute (reference)
    - `k`: Number of results

    Returns:
        Search results sorted by distance (ascending)
    """
    if request.query_image_id is not None and not request.keywords:
        manager = get_index_manager()
        descriptors = request.descriptors or ["vgg16"]
        for descriptor in descriptors:
            metric = (request.metrics or {}).get(descriptor, "euclidean")
            available = _available_metrics_by_index(manager, descriptor).get(
                request.index_type, []
            )
            if metric not in available:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Metric '{metric}' is not available for {descriptor} "
                        f"with index_type '{request.index_type}'. Available metrics: "
                        f"{', '.join(available) or 'none'}."
                    ),
                )

    # Build internal request
    internal_request = InternalSearchRequest(
        keywords=normalize_keywords(request.keywords) if request.keywords else None,
        keyword_mode=request.keyword_mode,
        query_image_id=request.query_image_id,
        descriptors=request.descriptors,
        metrics=request.metrics,
        weights=request.weights,
        index_type=request.index_type,
        k=request.k,
        exclude_query=request.exclude_query,
    )

    query_doc = None
    if request.query_image_id is not None:
        query_doc = get_image_by_id(request.query_image_id)
        if query_doc is None:
            raise HTTPException(
                status_code=404,
                detail=f"Image {request.query_image_id} not found",
            )

    # Execute search
    try:
        service = get_search_service()
        results = service.search(internal_request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Feature or index not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Build response
    result_items = []
    for r in results:
        thumb_url = f"/static/thumbs/{r.thumb}" if r.thumb else ""
        result_items.append(SearchResultItem(
            id=r.id,
            distance=r.distance,
            thumb=r.thumb,
            thumb_url=thumb_url,
            keywords=r.keywords,
            photo_url=r.photo_url,
            web_url=r.web_url,
            license=r.license,
        ))

    # Determine search mode
    has_keywords = bool(request.keywords)
    has_image = request.query_image_id is not None
    if has_keywords and has_image:
        search_mode = "combined"
    elif has_keywords:
        search_mode = "text"
    else:
        search_mode = "visual"

    # Get query image info if visual search
    query_image = None
    if query_doc is not None:
        thumb_path = query_doc.get("thumb", "")
        query_image = SearchResultItem(
            id=query_doc["_id"],
            distance=0.0,
            thumb=thumb_path,
            thumb_url=f"/static/thumbs/{thumb_path}" if thumb_path else "",
            keywords=query_doc.get("keywords", []),
            photo_url=query_doc.get("photo_url"),
            web_url=query_doc.get("web_url"),
            license=query_doc.get("license"),
        )

    return SearchResponse(
        query_image=query_image,
        results=result_items,
        total=len(result_items),
        search_mode=search_mode,
    )
