"""Pydantic schemas for search API.

Defines request and response models for:
- /search: Combined text and visual search
- /descriptors: List available descriptors
- /images: Browse/paginate images
- /health: Health check
"""
import math
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


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
    keyword_mode: Literal["and", "or"] = Field(
        default="and",
        description="Keyword combination mode: 'and' (all must match) or 'or' (any match)"
    )

    # Visual search
    query_image_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=999999,
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
    index_type: Literal["faiss", "vptree", "brute"] = Field(
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

    @model_validator(mode="after")
    def validate_search_parameters(self):
        if not self.keywords and self.query_image_id is None:
            raise ValueError("Provide at least one keyword or a query_image_id")

        supported_metrics = {
            "vgg16": {"euclidean", "cosine"},
            "edgehistogram": {"euclidean", "cosine", "chi2", "l1"},
            "homogeneoustexture": {"euclidean", "cosine", "l1"},
        }
        if self.descriptors:
            unknown = set(self.descriptors) - supported_metrics.keys()
            if unknown:
                raise ValueError(f"Unknown descriptor(s): {', '.join(sorted(unknown))}")
            if len(set(self.descriptors)) != len(self.descriptors):
                raise ValueError("Descriptors must not contain duplicates")
            if any(not descriptor for descriptor in self.descriptors):
                raise ValueError("Descriptors must not be empty strings")
            for descriptor, metric in (self.metrics or {}).items():
                if descriptor not in self.descriptors:
                    raise ValueError(f"Metric provided for unselected descriptor: {descriptor}")
                if metric not in supported_metrics[descriptor]:
                    raise ValueError(
                        f"Metric '{metric}' is not supported for {descriptor}"
                    )
            if len(self.descriptors) > 1 and self.index_type != "brute":
                raise ValueError(
                    "Multi-descriptor fusion uses exhaustive search; set index_type to 'brute'"
                )
        elif self.metrics:
            raise ValueError("Metrics require at least one selected descriptor")

        if self.weights:
            if not self.descriptors or set(self.weights) - set(self.descriptors):
                raise ValueError("Weights must correspond to selected descriptors")
            if any(
                not math.isfinite(weight) or weight <= 0
                for weight in self.weights.values()
            ):
                raise ValueError("Descriptor weights must be finite and positive")

        return self


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
