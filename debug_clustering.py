#!/usr/bin/env python3
"""
Debug script to analyze MP-KVM clustering issues.

This script reproduces the clustering problem and analyzes potential causes.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

# Copy the OnlineManifoldClustering class locally to avoid torch import issues
def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

class SimpleOnlineManifoldClustering:
    """Simplified version for debugging without torch dependencies."""

    def __init__(self, dim: int, similarity_threshold: float = 0.8, max_centroids: int = 128):
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self.max_centroids = max_centroids
        self.centroids = []
        self.centroid_counts = []

    def add(self, keys: np.ndarray, values: np.ndarray, weights: np.ndarray = None):
        """Add keys to clustering."""
        if weights is None:
            weights = np.ones(len(keys))

        for i, key in enumerate(keys):
            self._add_single(key, values[i], weights[i])

    def _add_single(self, key: np.ndarray, value: np.ndarray, weight: float):
        """Add a single key-value pair."""
        if len(self.centroids) == 0:
            # First centroid
            self.centroids.append(key.copy())
            self.centroid_counts.append(1)
            return

        # Find best matching centroid
        centroids_array = np.stack(self.centroids, axis=0)
        k_norm = _normalize_rows(key.reshape(1, -1))
        c_norm = _normalize_rows(centroids_array)
        sims = np.dot(k_norm, c_norm.T)[0]

        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        if best_sim >= self.similarity_threshold:
            # Merge into existing centroid (simple average for debugging)
            old_count = self.centroid_counts[best_idx]
            new_count = old_count + 1
            self.centroids[best_idx] = (self.centroids[best_idx] * old_count + key) / new_count
            self.centroid_counts[best_idx] = new_count
        else:
            # Create new centroid
            if len(self.centroids) < self.max_centroids:
                self.centroids.append(key.copy())
                self.centroid_counts.append(1)

    def get_centroids(self):
        """Return centroids array."""
        if len(self.centroids) == 0:
            return np.zeros((0, self.dim), dtype=np.float32), np.array([], dtype=int)
        return np.stack(self.centroids, axis=0), np.array(self.centroid_counts, dtype=int)

def assign_by_cosine(keys: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign keys to nearest centroids by cosine similarity."""
    if centroids.shape[0] == 0:
        return np.zeros(len(keys), dtype=int)

    k_norm = _normalize_rows(keys)
    c_norm = _normalize_rows(centroids)
    sims = np.dot(k_norm, c_norm.T)
    return np.argmax(sims, axis=1)

def debug_clustering_issue():
    """Debug the clustering similarity issue."""

    # Reproduce the data generation from run_benchmark.py
    seed = 0
    n_tokens = 2048
    dim = 128

    rng = np.random.RandomState(seed)

    # Create synthetic data with semantic clusters
    n_clusters = 6
    cluster_centers = rng.normal(0, 2, (n_clusters, dim)).astype(np.float32)

    # Normalize centers to unit sphere
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    keys = []
    cluster_assignments = []

    # Generate tokens from clusters with temporal structure
    tokens_per_cluster = n_tokens // n_clusters

    for cluster_idx in range(n_clusters):
        # Create cluster points with controlled spread
        cluster_points = cluster_centers[cluster_idx] + rng.normal(0, 0.3, (tokens_per_cluster, dim))
        # Normalize to improve clustering quality
        cluster_points = cluster_points / np.linalg.norm(cluster_points, axis=1, keepdims=True)
        keys.append(cluster_points.astype(np.float32))
        cluster_assignments.extend([cluster_idx] * tokens_per_cluster)

    keys = np.concatenate(keys, axis=0)
    vals = keys.copy()

    print(f"Generated {len(keys)} vectors of dimension {dim}")
    print(f"Ground truth clusters: {n_clusters}")
    print(f"Vectors per cluster: {tokens_per_cluster}")

    # Test different similarity thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]

    for thresh in thresholds:
        print(f"\n=== Testing similarity_threshold = {thresh} ===")

        # Create clusterer with streaming parameters like original code
        clusterer = SimpleOnlineManifoldClustering(
            dim=dim,
            similarity_threshold=thresh,
            max_centroids=128
        )

        # Use streaming batch processing like original code (compress_batch = 512)
        compress_batch = 512
        for i in range(0, len(keys), compress_batch):
            k_batch = keys[i:i+compress_batch]
            v_batch = vals[i:i+compress_batch]
            w_batch = np.ones(len(k_batch))
            clusterer.add(k_batch, v_batch, w_batch)

        centroids, counts = clusterer.get_centroids()

        print(f"Generated centroids: {centroids.shape[0]}")
        print(f"Total counts: {counts.sum()}")
        print(f"Max count: {counts.max()}, Min count: {counts.min()}")

        if centroids.shape[0] > 0:
            # Compute reconstruction quality
            assignments = assign_by_cosine(keys, centroids)
            reconstructed = centroids[assignments]

            # Compute cosine similarity
            original_norm = keys / np.linalg.norm(keys, axis=1, keepdims=True)
            reconstructed_norm = reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)
            cosine_similarities = np.sum(original_norm * reconstructed_norm, axis=1)
            avg_cosine_similarity = np.mean(cosine_similarities)

            print(".3f")

            # Analyze cluster assignments vs ground truth
            cluster_purity = []
            for c_idx in range(centroids.shape[0]):
                assigned_points = keys[assignments == c_idx]
                if len(assigned_points) > 0:
                    # Find which ground truth cluster this centroid represents
                    similarities_to_gt = []
                    for gt_idx in range(n_clusters):
                        gt_center = cluster_centers[gt_idx]
                        sims = np.dot(assigned_points, gt_center)
                        similarities_to_gt.append(np.mean(sims))
                    best_gt = np.argmax(similarities_to_gt)
                    purity = similarities_to_gt[best_gt]
                    cluster_purity.append(purity)

            if cluster_purity:
                print(".3f")
        else:
            print("No centroids generated!")

        # Check if centroids are too similar to each other
        if centroids.shape[0] > 1:
            c_norm = _normalize_rows(centroids)
            centroid_sims = np.dot(c_norm, c_norm.T)
            np.fill_diagonal(centroid_sims, 0)  # Ignore self-similarity
            max_inter_centroid_sim = np.max(centroid_sims)
            print(".3f")

if __name__ == "__main__":
    debug_clustering_issue()
