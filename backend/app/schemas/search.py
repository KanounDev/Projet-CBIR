"""Pydantic schemas for search API.

Defines request and response models for:
- /search: Combined text and visual search
- /descriptors: List available descriptors
- /images: Browse/paginate images
- /health: Health check
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Search Request
# ============================================================================

class SearchRequest(BaseModel):
    """Request model for /search endpoint.

    Supports:
    - Text search (keywords with AND/OR mode)
    - Visual search (query image + descriptor + distance + index)
    - Combined search (keywords + visual)
    """
    # Text search
    keywords: Optional[List[str]] = Field(
        default=None,
        description="List of keywords for text search",
        example=["sunset", "beach"]
    )
    keyword_mode: str = Field(
        default="and",
        description="Keyword combination mode: 'and' (all must match) or 'or' (any match)"
    )

    # Visual search
    query_image_id: Optional[int] = Field(
        default=None,
        description="Image ID to use as visual query (0-999999)",
        example=42
    )
    descriptors: Optional[List[str]] = Field(
        default=None,
        description="Descriptor(s) to use: vgg16, edgehistogram, homogeneoustexture",
        example=["vgg16"]
    )
    metrics: Optional[Dict[str, str]] = Field(
        default=None,
        description="Distance metric per descriptor: euclidean, cosine, chi2, l1",
        example={"vgg16": "euclidean"}
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weight per descriptor for fusion (sum should be 1.0)",
        example={"vgg16": 1.0}
    )
    index_type: str = Field(
        default="faiss",
        description="Index type: faiss (production), vptree (demo), brute (reference)"
    )

    # Result options
    k: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Number of results to return"
    )
    exclude_query: bool = Field(
        default=True,
        description="Exclude query image from results"
    )


# ============================================================================
# Search Response
# ============================================================================

class SearchResultItem(BaseModel):
    """A single search result."""
    id: int = Field(description="Image ID")
    distance: float = Field(description="Distance score (lower = more similar)")
    thumb: str = Field(description="Thumbnail path (relative)")
    thumb_url: str = Field(description="Full thumbnail URL")
    keywords: List[str] = Field(default_factory=list, description="Image keywords")
    photo_url: Optional[str] = Field(default=None, description="Flickr photo URL (if available)")
    web_url: Optional[str] = Field(default=None, description="Flickr page URL (if available)")
    license: Optional[str] = Field(default=None, description="License info")


class SearchResponse(BaseModel):
    """Response model for /search endpoint."""
    query_image: Optional[SearchResultItem] = Field(
        default=None,
        description="Query image info (if visual search)"
    )
    results: List[SearchResultItem] = Field(
        default_factory=list,
        description="Search results sorted by distance (ascending)"
    )
    total: int = Field(description="Total number of results returned")
    search_mode: str = Field(description="Search mode used: text, visual, or combined")


# ============================================================================
# Descriptor Info
# ============================================================================

class DescriptorInfo(BaseModel):
    """Information about an available descriptor."""
    name: str = Field(description="Descriptor name")
    dimension: int = Field(description="Vector dimension")
    description: str = Field(description="Human-readable description")
    available_metrics: List[str] = Field(description="Available distance metrics")
    available_indexes: List[str] = Field(description="Available index types")


# ============================================================================
# Image Info
# ============================================================================

class ImageInfo(BaseModel):
    """Information about an image in the database."""
    id: int = Field(description="Image ID")
    thumb_url: str = Field(description="Thumbnail URL")
    keywords: List[str] = Field(default_factory=list, description="Image keywords")
    photo_url: Optional[str] = Field(default=None, description="Flickr photo URL")
    web_url: Optional[str] = Field(default=None, description="Flickr page URL")


class ImageListResponse(BaseModel):
    """Response for /images endpoint."""
    images: List[ImageInfo] = Field(description="List of images")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of images per page")
    total: int = Field(description="Total number of images in database")


# ============================================================================
# Health Check
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status: ok or error")
    mongodb: str = Field(description="MongoDB connection status")
    features_loaded: List[str] = Field(description="Loaded feature descriptors")
    indexes_available: List[str] = Field(description="Available index files")
