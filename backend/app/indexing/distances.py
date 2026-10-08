"""Distances between ONE query vector q (d,) and a matrix X (N, d) -> array (N,).

Computed in chunks so that a 1M x 150 matrix never needs a full-size temporary array.
All functions return float32 values; smaller = more similar.
"""
import numpy as np

EPS = 1e-9
CHUNK = 100_000


def _chunked(fn):
    def wrapper(X, q, chunk=CHUNK):
        q = np.asarray(q, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)
        for s in range(0, len(X), chunk):
            out[s:s + chunk] = fn(np.asarray(X[s:s + chunk], dtype=np.float32), q)
        return out
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@_chunked
def euclidean(Xc, q):
    """L2 distance."""
    d = Xc - q
    return np.sqrt(np.einsum("ij,ij->i", d, d))


@_chunked
def l1(Xc, q):
    """Manhattan (L1) distance."""
    return np.abs(Xc - q).sum(axis=1)


@_chunked
def cosine(Xc, q):
    """1 - cosine similarity."""
    norms = np.linalg.norm(Xc, axis=1) * np.linalg.norm(q)
    return 1.0 - (Xc @ q) / (norms + EPS)


@_chunked
def chi2(Xc, q):
    """Chi-squared distance (for non-negative histograms)."""
    d = Xc - q
    return 0.5 * np.sum(d * d / (Xc + q + EPS), axis=1)


METRICS = {"euclidean": euclidean, "l1": l1, "cosine": cosine, "chi2": chi2}
