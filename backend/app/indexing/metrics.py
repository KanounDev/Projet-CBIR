"""Supported distance metrics for each CBIR descriptor and index family."""

VP_TREE_DEMO_SIZE = 50_000

SUPPORTED_METRICS = {
    "vgg16": {"euclidean", "cosine"},
    "edgehistogram": {"euclidean", "cosine", "chi2", "l1"},
    "homogeneoustexture": {"euclidean", "cosine", "l1"},
}

FAISS_METRICS = {
    "vgg16": {"euclidean", "cosine"},
    "edgehistogram": {"euclidean", "cosine", "l1"},
    "homogeneoustexture": {"euclidean", "cosine", "l1"},
}

VP_TREE_METRICS = {
    "vgg16": {"euclidean"},
    "edgehistogram": {"euclidean"},
    "homogeneoustexture": {"euclidean"},
}
