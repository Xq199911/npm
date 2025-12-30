"""
Energy / reconstruction loss utilities for MP-KVM.
Provides functions to compute reconstruction loss and assignments.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def compute_energy_loss(original_vectors: np.ndarray, centroid_vectors: np.ndarray, assignment: np.ndarray, lambda_diversity: float = 0.0) -> float:
    """
    Compute SSE reconstruction loss given explicit assignments.
    - original_vectors: (N, d)
    - centroid_vectors: (M, d)
    - assignment: (N,) integer indices into centroid_vectors
    """
    if original_vectors.size == 0:
        return 0.0
    dif = original_vectors - centroid_vectors[assignment]
    sse = float(np.sum(dif * dif))
    diversity = 0.0
    if lambda_diversity > 0.0 and centroid_vectors.shape[0] > 1:
        # pairwise centroid diversity (sum pairwise distances)
        pd = np.linalg.norm(centroid_vectors[:, None] - centroid_vectors[None, :], axis=-1)
        diversity = float(np.sum(np.triu(pd, k=1)))
    return float(sse + lambda_diversity * diversity)


def reconstruction_loss(keys: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
    """
    Compute sum_i ||k_i - centroid_assignment(i)||^2
    - keys: (N, D)
    - assignments: (N,) integer indices into centroids
    - centroids: (C, D)
    """
    if keys.shape[0] == 0:
        return 0.0
    rec = keys - centroids[assignments]
    return float((rec ** 2).sum())


def assign_by_cosine(keys: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each key to the nearest centroid by cosine similarity.
    Returns an array of shape (N,) with centroid indices.
    """
    if centroids.shape[0] == 0 or keys.shape[0] == 0:
        return np.zeros((keys.shape[0],), dtype=int)
    kn = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12)
    cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = np.dot(kn, cn.T)
    return np.argmax(sims, axis=1)


__all__ = ["compute_energy_loss", "reconstruction_loss", "assign_by_cosine"]


