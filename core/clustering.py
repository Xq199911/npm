"""Online manifold-partitioned clustering operator for MP-KVM.

Features:
- Online updates for streaming KV vectors
- Support for 'cosine' and 'euclidean' distance metrics
- Sliding window + persistent centroids
- Centroid counts and weighted updates
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

"""
Online manifold-partitioned clustering operator for MP-KVM.

Features:
- Online updates for streaming KV vectors
- Support for 'cosine' and 'euclidean' distance metrics
- Sliding window + persistent centroids
- Centroid counts and weighted updates
"""
import numpy as np
from typing import List, Optional, Tuple


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class OnlineManifoldClustering:
    """
    A lightweight online clustering operator tailored for KV cache compression.

    API:
    - add(k_vecs, v_vecs, weights=None): ingest new KV vectors (np.ndarray)
    - get_centroids(): return centroids and metadata
    - partition_and_compress(): run a local clustering pass and synthesize centroids
    """

    def __init__(
        self,
        dim: int,
        max_memory_size: int = 65536,
        window_size: int = 4096,
        max_centroids: int = 1024,
        metric: str = "cosine",
        similarity_threshold: float = 0.8,
        adaptive_threshold: bool = False,
        threshold_quantile: float = 0.9,
        min_merge_similarity: Optional[float] = None,
        init_preserve_first_n: Optional[int] = None,
    ):
        self.dim = dim
        self.metric = metric
        self.similarity_threshold = float(similarity_threshold)
        # adaptive thresholding (per-instance / per-layer)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.threshold_quantile = float(threshold_quantile)
        # when trimming centroids, only merge closest pairs if their similarity exceeds this value.
        # If None, fallback to the original behavior.
        self.min_merge_similarity = float(min_merge_similarity) if min_merge_similarity is not None else None
        # when initializing from the first compressed batch, optionally preserve the first N items
        # as separate centroids to avoid early global merging (helps maintain diversity).
        self.init_preserve_first_n = int(init_preserve_first_n) if init_preserve_first_n is not None else None
        # recent similarity windows used to compute adaptive threshold quantiles
        self._recent_best_sims: List[np.ndarray] = []

        # Raw buffers for streaming tokens (kept as ring buffer slice semantics)
        self.keys_buffer: List[np.ndarray] = []
        self.values_buffer: List[np.ndarray] = []
        self.weights_buffer: List[float] = []

        # Persistent centroids: list of centroid vectors, counts, and accumulated weights
        self.centroids: List[np.ndarray] = []
        self.value_centroids: List[np.ndarray] = []  # Separate centroids for values
        self.centroid_counts: List[int] = []
        self.centroid_weights: List[float] = []

        self.max_memory_size = int(max_memory_size)
        self.window_size = int(window_size)
        self.max_centroids = int(max_centroids)

    # ----------------------
    # Ingestion / buffering
    # ----------------------
    def add(self, keys: np.ndarray, values: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Add a batch of KV vectors.
        - keys: (N, D)
        - values: (N, Dv) (kept but not used for clustering distance)
        - weights: (N,) optional importance (e.g., attention scores)
        """
        assert keys.ndim == 2 and keys.shape[1] == self.dim
        n = keys.shape[0]
        if weights is None:
            weights = np.ones((n,), dtype=float)
        for i in range(n):
            self.keys_buffer.append(keys[i].astype(np.float32))
            self.values_buffer.append(values[i].astype(np.float32))
            self.weights_buffer.append(float(weights[i]))

        # enforce sliding window trimming
        if len(self.keys_buffer) > self.window_size:
            excess = len(self.keys_buffer) - self.window_size
            # drop oldest into centroids via local compression
            self._compress_oldest_batch(excess)

        # enforce global memory cap by merging into persistent centroids
        total_items = len(self.keys_buffer) + sum(self.centroid_counts)
        if total_items > self.max_memory_size:
            self._prune_to_budget()

    # ----------------------
    # Core local compression
    # ----------------------
    def _compress_oldest_batch(self, batch_size: int):
        """
        Compress the oldest `batch_size` items in the sliding buffer into centroids.
        This is a cheap local clustering pass (greedy agglomeration).
        """
        if batch_size <= 0:
            return
        # take slice
        keys = np.stack(self.keys_buffer[:batch_size], axis=0)
        vals = np.stack(self.values_buffer[:batch_size], axis=0)
        w = np.array(self.weights_buffer[:batch_size], dtype=float)
        # remove them from buffers
        del self.keys_buffer[:batch_size]
        del self.values_buffer[:batch_size]
        del self.weights_buffer[:batch_size]

        # greedy assign to existing centroids if sufficiently similar
        if len(self.centroids) == 0:
            # Optionally preserve first N items as separate centroids to avoid immediate collapse
            if self.init_preserve_first_n is not None and self.init_preserve_first_n > 0:
                to_preserve = min(self.init_preserve_first_n, batch_size, self.max_centroids)
                for i in range(to_preserve):
                    self.centroids.append(keys[i].copy())
                    self.value_centroids.append(vals[i].copy())  
                    self.centroid_counts.append(1)
                    self.centroid_weights.append(float(w[i]))
                # for any remaining items, process them normally (assign/merge)
                start_idx = to_preserve
                if start_idx >= batch_size:
                    return
                keys = keys[start_idx:]
                w = w[start_idx:]
                batch_size = keys.shape[0]
            else:
                # initialize centroids directly using weighted average groups
                centroid = self._weighted_mean(keys, w)
                self.centroids.append(centroid)
                self.centroid_counts.append(batch_size)
                self.centroid_weights.append(float(w.sum()))
                return

        # compute similarities
        if self.metric == "cosine":
            k_norm = _normalize_rows(keys)
            c_stack = np.stack(self.centroids, axis=0)
            c_norm = _normalize_rows(c_stack)
            sims = np.dot(k_norm, c_norm.T)  # (n, C)
            best_idx = np.argmax(sims, axis=1)
            best_sim = sims[np.arange(len(keys)), best_idx]
            # collect recent best similarities for adaptive thresholding
            if self.adaptive_threshold:
                self._recent_best_sims.append(best_sim.copy())
                # keep bounded history length
                if len(self._recent_best_sims) > 64:
                    self._recent_best_sims.pop(0)
                all_sims = np.concatenate(self._recent_best_sims)
                # compute quantile-based threshold and update similarity threshold
                new_thresh = float(np.quantile(all_sims, self.threshold_quantile))
                # only update if numeric and finite
                if np.isfinite(new_thresh):
                    self.similarity_threshold = float(new_thresh)
        else:
            # euclidean: smaller distance -> larger negative sim
            c_stack = np.stack(self.centroids, axis=0)
            dists = np.linalg.norm(keys[:, None, :] - c_stack[None, :, :], axis=2)
            best_idx = np.argmin(dists, axis=1)
            best_sim = -dists[np.arange(len(keys)), best_idx]

        # assign or create new centroid
        for i, sim_score in enumerate(best_sim):
            idx = int(best_idx[i])
            if (self.metric == "cosine" and sim_score >= self.similarity_threshold) or (
                self.metric != "cosine" and -sim_score <= (1.0 - self.similarity_threshold)
            ):
                # merge into centroid idx (weighted)
                self._merge_into_centroid(idx, keys[i], vals[i], w[i])
            else:
                # create new centroid
                self.centroids.append(keys[i].copy())
                self.value_centroids.append(vals[i].copy())
                self.centroid_counts.append(1)
                self.centroid_weights.append(float(w[i]))

        # cap number of centroids
        self._trim_centroids_if_needed()

    def _weighted_mean(self, xs: np.ndarray, ws: np.ndarray) -> np.ndarray:
        ws_sum = float(ws.sum()) if ws.sum() != 0 else 1.0
        return (xs * ws[:, None]).sum(axis=0) / ws_sum

    def _merge_into_centroid(self, idx: int, key_vec: np.ndarray, value_vec: np.ndarray, weight: float):
        prev_w = float(self.centroid_weights[idx])
        prev_count = int(self.centroid_counts[idx])
        new_w = prev_w + float(weight)
        # weighted incremental update of key centroid
        updated_key = (self.centroids[idx] * prev_w + key_vec * float(weight)) / new_w
        self.centroids[idx] = updated_key
        # weighted incremental update of value centroid
        if idx < len(self.value_centroids):
            updated_value = (self.value_centroids[idx] * prev_w + value_vec * float(weight)) / new_w
            self.value_centroids[idx] = updated_value
        else:
            # Initialize value centroid if not exists
            self.value_centroids.append(value_vec.copy())
        self.centroid_weights[idx] = new_w
        self.centroid_counts[idx] = prev_count + 1

    def _trim_centroids_if_needed(self):
        # If too many centroids, greedily merge closest pairs until under budget
        while len(self.centroids) > self.max_centroids:
            c = np.stack(self.centroids, axis=0)
            C = len(self.centroids)

            if self.metric == "cosine":
                # Vectorized cosine similarity computation
                cn = _normalize_rows(c)
                sim_mat = np.dot(cn, cn.T)
                # Mask diagonal and upper triangle (since similarity is symmetric)
                sim_mat = np.triu(sim_mat, k=1)
                # Find the maximum similarity pair
                if sim_mat.max() > 0:  # Only merge if there are similar pairs
                    max_idx = np.argmax(sim_mat)
                    a, b = np.unravel_index(max_idx, sim_mat.shape)
                    best_score = sim_mat[a, b]
                else:
                    # No similar pairs, merge least important centroids
                    a, b = 0, 1
                    best_score = sim_mat[a, b]
            else:
                # Euclidean distance - find closest pair
                dist_mat = np.zeros((C, C))
                for i in range(C):
                    for j in range(i+1, C):
                        dist_mat[i, j] = np.linalg.norm(c[i] - c[j])
                        dist_mat[j, i] = dist_mat[i, j]  # Make symmetric

                # Create mask to exclude diagonal and lower triangle
                # Only search upper triangle (above diagonal) for minimum distance
                mask = np.tril(np.ones((C, C), dtype=bool))  # True for lower triangle + diagonal
                search_mat = dist_mat.copy()
                search_mat[mask] = np.inf  # Set lower triangle + diagonal to infinity
                min_idx = np.unravel_index(np.argmin(search_mat), dist_mat.shape)
                a, b = min_idx
                best_score = -dist_mat[a, b]  # Convert distance to similarity score
            # If a minimum merge similarity is configured and the best pair doesn't meet it,
            # prefer merging two smallest-weight centroids (less impact) instead of the closest pair.
            if self.min_merge_similarity is not None and self.metric == "cosine" and best_score < float(self.min_merge_similarity):
                # find two smallest-weight centroids
                idx_sorted = np.argsort(self.centroid_weights)
                a = int(idx_sorted[0])
                b = int(idx_sorted[1]) if len(idx_sorted) > 1 else int(idx_sorted[0])

            # merge b into a using weighted combination
            wa = self.centroid_weights[a]
            wb = self.centroid_weights[b]
            merged_key = (self.centroids[a] * wa + self.centroids[b] * wb) / (wa + wb + 1e-12)
            self.centroids[a] = merged_key
            # Also merge value centroids if they exist
            if len(self.value_centroids) > max(a, b):
                merged_value = (self.value_centroids[a] * wa + self.value_centroids[b] * wb) / (wa + wb + 1e-12)
                self.value_centroids[a] = merged_value
                del self.value_centroids[b]
            self.centroid_weights[a] = wa + wb
            self.centroid_counts[a] = self.centroid_counts[a] + self.centroid_counts[b]
            # remove b
            del self.centroids[b]
            del self.centroid_weights[b]
            del self.centroid_counts[b]

    # ----------------------
    # Budget pruning
    # ----------------------
    def _prune_to_budget(self):
        """
        If we exceed max_memory_size, aggressively merge oldest centroids
        or least-weighted centroids until under budget.
        """
        total = len(self.keys_buffer) + sum(self.centroid_counts)
        if total <= self.max_memory_size:
            return
        # keep merging smallest-weight centroids
        while total > self.max_memory_size and len(self.centroids) > 1:
            # find two smallest-weight centroids and merge
            idx_sorted = np.argsort(self.centroid_weights)
            a = int(idx_sorted[0])
            b = int(idx_sorted[1])
            wa = self.centroid_weights[a]
            wb = self.centroid_weights[b]
            merged_key = (self.centroids[a] * wa + self.centroids[b] * wb) / (wa + wb + 1e-12)
            self.centroids[a] = merged_key
            # Also merge value centroids if they exist
            if len(self.value_centroids) > max(a, b):
                merged_value = (self.value_centroids[a] * wa + self.value_centroids[b] * wb) / (wa + wb + 1e-12)
                self.value_centroids[a] = merged_value
                del self.value_centroids[b]
            self.centroid_weights[a] = wa + wb
            self.centroid_counts[a] = self.centroid_counts[a] + self.centroid_counts[b]
            # remove b
            del self.centroids[b]
            del self.centroid_weights[b]
            del self.centroid_counts[b]
            total = len(self.keys_buffer) + sum(self.centroid_counts)

    # ----------------------
    # Public inspection
    # ----------------------
    def get_centroids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (centroids: (C, D), counts: (C,), weights: (C,))
        """
        if len(self.centroids) == 0:
            return np.zeros((0, self.dim), dtype=np.float32), np.array([], dtype=int), np.array([], dtype=float)
        c = np.stack(self.centroids, axis=0)
        return c, np.array(self.centroid_counts, dtype=int), np.array(self.centroid_weights, dtype=float)

    def get_key_value_centroids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (key_centroids: (C, D), value_centroids: (C, D), counts: (C,), weights: (C,))
        """
        if len(self.centroids) == 0:
            empty = np.zeros((0, self.dim), dtype=np.float32)
            return empty, empty, np.array([], dtype=int), np.array([], dtype=float)
        k = np.stack(self.centroids, axis=0)
        if len(self.value_centroids) == len(self.centroids):
            v = np.stack(self.value_centroids, axis=0)
        else:
            # Fallback: use key centroids as values (should not happen in properly implemented code)
            v = k.copy()
        return k, v, np.array(self.centroid_counts, dtype=int), np.array(self.centroid_weights, dtype=float)

    def snapshot_buffer(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current sliding window buffers as arrays (keys, values, weights)."""
        if len(self.keys_buffer) == 0:
            return (np.zeros((0, self.dim), dtype=np.float32), np.zeros((0, self.dim), dtype=np.float32), np.zeros((0,), dtype=float))
        k = np.stack(self.keys_buffer, axis=0)
        v = np.stack(self.values_buffer, axis=0)
        w = np.array(self.weights_buffer, dtype=float)
        return k, v, w


__all__ = ["OnlineManifoldClustering"]


