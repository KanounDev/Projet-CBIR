"""Database layer for CBIR application."""
from .mongodb import get_db, get_collection, get_image_by_id, get_images_by_ids

__all__ = ["get_db", "get_collection", "get_image_by_id", "get_images_by_ids"]
