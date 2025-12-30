"""
Visualization utilities for KV manifold.
Produces UMAP projections and basic matplotlib figures.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


def visualize_kv_and_centroids(keys: np.ndarray, centroids: np.ndarray, save_path: Optional[str] = None, title: str = "KV manifold"):
    """
    keys: (N, D)
    centroids: (C, D)
    """
    # prefer UMAP but fall back to PCA if not available
    import matplotlib.pyplot as plt
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=42)
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)

    if keys.shape[0] == 0 and centroids.shape[0] == 0:
        raise ValueError("empty inputs")

    X = np.concatenate([keys, centroids], axis=0)
    Z = reducer.fit_transform(X)
    n = keys.shape[0]
    C = centroids.shape[0]
    zk = Z[:n]
    zc = Z[n:]

    plt.figure(figsize=(8, 6))
    plt.scatter(zk[:, 0], zk[:, 1], s=6, alpha=0.6, label="tokens", c="C0")
    if C > 0:
        plt.scatter(zc[:, 0], zc[:, 1], s=60, alpha=0.9, label="centroids", c="C3", marker="X")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()


def create_topic_transition_visualization(tokens: np.ndarray, centroids: np.ndarray,
                                          topic_boundaries: list, topic_names: list,
                                          save_path: Optional[str] = None):
    """
    Create visualization showing how MP-KVM adapts to topic transitions.

    Args:
        tokens: (N, D) token embeddings
        centroids: (C, D) centroid embeddings
        topic_boundaries: list of indices where topics change
        topic_names: list of topic names
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt

    # Use UMAP for dimensionality reduction
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)

    # Reduce dimensionality
    if centroids is not None and centroids.shape[0] > 0:
        X = np.concatenate([tokens, centroids], axis=0)
        Z = reducer.fit_transform(X)
        n_tokens = tokens.shape[0]
        z_tokens = Z[:n_tokens]
        z_centroids = Z[n_tokens:]
    else:
        z_tokens = reducer.fit_transform(tokens)
        z_centroids = None

    # Create color map for topics
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    topic_colors = colors[:len(topic_names)]

    plt.figure(figsize=(12, 8))

    # Plot tokens by topic
    start_idx = 0
    for i, (boundary, topic_name, color) in enumerate(zip(topic_boundaries + [len(tokens)], topic_names, topic_colors)):
        end_idx = boundary
        plt.scatter(z_tokens[start_idx:end_idx, 0], z_tokens[start_idx:end_idx, 1],
                   s=8, alpha=0.7, label=f"{topic_name} tokens", c=color)
        start_idx = end_idx

    # Plot centroids
    if z_centroids is not None and z_centroids.shape[0] > 0:
        plt.scatter(z_centroids[:, 0], z_centroids[:, 1], s=100, alpha=0.9,
                   label="MP-KVM Centroids", c='red', marker="X", edgecolors='black', linewidth=2)

        # Add centroid labels
        for i, (x, y) in enumerate(z_centroids):
            plt.annotate(f'C{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    plt.title("MP-KVM Manifold Partitioning: Topic Transitions and Centroid Adaptation\n"
             "Demonstrates semantic clustering across context shifts", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


__all__ = ["visualize_kv_and_centroids", "create_topic_transition_visualization"]


