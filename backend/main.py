"""FastAPI application for CBIR search engine.

Main entry point that:
- Creates the FastAPI app
- Mounts static files for thumbnails
- Includes API routes
- Configures CORS
- Loads indexes at startup

Environment variables:
- MONGO_URI: MongoDB connection string (default: mongodb://localhost:27017)
- DATA_DIR: Path to data directory (default: ./data)
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.upload import router as upload_router
from app.db.mongodb import check_connection, get_stats
from app.indexing.index_manager import get_index_manager

# Configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events.

    - Check MongoDB connection
    - Pre-load feature indexes
    """
    # Startup
    print("=" * 60)
    print("CBIR Search Engine - Starting up")
    print("=" * 60)

    # Check MongoDB
    print("\n1. Checking MongoDB connection...")
    try:
        check_connection()
        stats = get_stats()
        print(f"   [OK] Connected to MongoDB")
        print(f"   - Total images: {stats['total_images']}")
        print(f"   - Images with keywords: {stats['images_with_keywords']}")
    except Exception as e:
        print(f"   [WARN] MongoDB connection failed: {e}")
        print("   Search will work but metadata may be unavailable")

    # Pre-load indexes
    print("\n2. Loading feature indexes...")
    manager = get_index_manager()

    for desc in ["vgg16", "edgehistogram", "homogeneoustexture"]:
        try:
            manager.load_features(desc)
            print(f"   [OK] {desc}: loaded")
        except Exception as e:
            print(f"   [WARN] {desc}: not available ({e})")

    print("\n" + "=" * 60)
    print("Server ready! Access the API at http://localhost:8000/docs")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    print("\nShutting down...")


# Create app
app = FastAPI(
    title="CBIR Search Engine",
    description="""
Content-Based Image Retrieval (CBIR) system for MIRFLICKR-1M dataset.

## Features

- **Text search**: Find images by keywords (AND/OR modes)
- **Visual search**: Find similar images using deep learning (VGG16) or MPEG-7 descriptors
- **Combined search**: Filter by keywords, rank by visual similarity

## Descriptors

| Name | Dimensions | Best for |
|------|-----------|----------|
| `vgg16` | 512 | Semantic similarity (recommended) |
| `edgehistogram` | 150 | Edge patterns (MPEG-7) |
| `homogeneoustexture` | 43 | Texture patterns (MPEG-7) |

## Distance Metrics

- `euclidean`: L2 distance (default)
- `cosine`: Cosine similarity
- `chi2`: Chi-squared (for histograms)
- `l1`: Manhattan distance

## Index Types

- `faiss`: Fast production index (recommended for 1M images)
- `vptree`: Demonstrative index (for small datasets)
- `brute`: Brute-force reference
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for thumbnails
thumbs_dir = DATA_DIR / "mirflickr" / "raw"
if thumbs_dir.exists():
    # Mount each thumbs directory
    for thumbs_subdir in ["thumbs1-33", "thumbs34-66", "thumbs67-100"]:
        subdir_path = thumbs_dir / thumbs_subdir
        if subdir_path.exists():
            app.mount(
                f"/static/thumbs/{thumbs_subdir}",
                StaticFiles(directory=str(subdir_path)),
                name=f"thumbs_{thumbs_subdir}",
            )
    print(f"Mounted thumbnail directories from {thumbs_dir}")
else:
    print(f"Warning: Thumbnail directory not found at {thumbs_dir}")

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(upload_router, prefix="/api")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CBIR Search Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "descriptors": "/api/descriptors",
            "images": "/api/images",
            "search": "/api/search",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
