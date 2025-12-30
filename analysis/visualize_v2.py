"""
Fallback-safe visualization utilities for KV manifolds.
Provides `visualize_kv_and_centroids` used by experiments/export scripts.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def visualize_kv_and_centroids(keys: np.ndarray, centroids: Optional[np.ndarray] = None, save_path: Optional[str] = None, title: str = "KV manifold", random_state: int = 42):
    """
    Visualize key vectors and optional centroids in 2D using UMAP if available, otherwise PCA.
    - keys: (N, D)
    - centroids: (C, D) or None
    """
    if keys is None and (centroids is None or centroids.size == 0):
        raise ValueError("empty inputs")

    X = keys
    if centroids is not None and centroids.shape[0] > 0:
        X = np.concatenate([keys, centroids], axis=0)

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        Z = reducer.fit_transform(X)
    except ImportError:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)

    n = keys.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Z[:n, 0], Z[:n, 1], s=6, alpha=0.6, label="tokens", c="C0")
    if centroids is not None and centroids.shape[0] > 0:
        C = centroids.shape[0]
        ax.scatter(Z[n:n + C, 0], Z[n:n + C, 1], s=60, alpha=0.9, label="centroids", c="C3", marker="X")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig, ax


__all__ = ["visualize_kv_and_centroids"]


