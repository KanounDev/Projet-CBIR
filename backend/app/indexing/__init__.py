"""Indexing module for CBIR.

Provides fast nearest neighbor search using:
- VP-Tree: Euclidean demonstration on a 50k-image subset
- FAISS: for production on large datasets (1M+)
- Brute-force fallback: for correctness verification
"""
from app.indexing.distances import METRICS, euclidean, l1, cosine, chi2
from app.indexing.vptree import VPTree, VPTreeNode
from app.indexing.faiss_index import FAISSIndex, build_faiss_index
from app.indexing.index_manager import IndexManager

__all__ = [
    # Distances
    'METRICS', 'euclidean', 'l1', 'cosine', 'chi2',
    # VP-Tree
    'VPTree', 'VPTreeNode',
    # FAISS
    'FAISSIndex', 'build_faiss_index',
    # Manager
    'IndexManager',
]
