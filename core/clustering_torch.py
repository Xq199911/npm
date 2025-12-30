"""
Torch-based online manifold clustering for GPU-side aggregation.

API mirrors `core.clustering.OnlineManifoldCluster` but operates on torch tensors.
Centroids and counts are kept as torch tensors on the same device as inputs.
Conversion to numpy happens only when `get_centroids()` is called.
"""
from __future__ import annotations
import torch
from typing import Optional, Tuple, List


def _normalize_rows_t(x: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(x, p=2, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), norms)
    return x / norms


class TorchOnlineManifoldCluster:
    def __init__(
        self,
        dim: int,
        max_centroids: int = 1024,
        distance: str = "cosine",
        sliding_window_size: Optional[int] = None,
        persistence_decay: float = 1.0,
        min_count_threshold: float = 1e-3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        min_merge_similarity: Optional[float] = None,
        init_preserve_first_n: Optional[int] = None,
    ):
        assert distance in ("cosine", "euclidean")
        self.dim = int(dim)
        self.max_centroids = int(max_centroids)
        self.distance = distance
        self.sliding_window_size = int(sliding_window_size) if sliding_window_size is not None else None
        self.persistence_decay = float(persistence_decay)
        self.min_count_threshold = float(min_count_threshold)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.min_merge_similarity = float(min_merge_similarity) if min_merge_similarity is not None else None
        self.init_preserve_first_n = int(init_preserve_first_n) if init_preserve_first_n is not None else None

        self._sums: List[torch.Tensor] = []
        self._counts: List[float] = []
        self._history: List[torch.Tensor] = [] if self.sliding_window_size is not None else None
        self._step = 0

    def _pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (n,d), y: (m,d) -> (n,m)
        if x.numel() == 0 or y.numel() == 0:
            return torch.zeros((x.shape[0], y.shape[0]), device=self.device, dtype=self.dtype)
        if self.distance == "cosine":
            xn = _normalize_rows_t(x)
            yn = _normalize_rows_t(y)
            return 1.0 - torch.matmul(xn, yn.T)
        else:
            # euclidean
            x2 = (x * x).sum(dim=1).unsqueeze(1)
            y2 = (y * y).sum(dim=1).unsqueeze(0)
            xy = torch.matmul(x, y.T)
            d2 = x2 + y2 - 2.0 * xy
            d2 = torch.clamp(d2, min=0.0)
            return torch.sqrt(d2)

    def _current_centroids(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._sums:
            return torch.empty((0, self.dim), device=self.device, dtype=self.dtype), torch.empty((0,), device=self.device, dtype=self.dtype)
        sums_stack = torch.stack(self._sums, dim=0)
        counts = torch.tensor(self._counts, device=self.device, dtype=self.dtype)
        # Use smoother numerical stabilization: add small epsilon to denominator
        # This is more numerically stable than clamping counts to a minimum value
        centroids = sums_stack / (counts.unsqueeze(1) + 1e-9)
        return centroids, counts

    def add(self, vectors: torch.Tensor, weights: Optional[torch.Tensor] = None, similarity_threshold: float = 0.1):
        """
        vectors: (n, d) torch tensor on desired device
        weights: (n,) optional torch tensor
        """
        assert vectors.dim() == 2 and vectors.shape[1] == self.dim
        vectors = vectors.to(device=self.device, dtype=self.dtype)
        n = vectors.shape[0]
        if weights is None:
            weights = torch.ones((n,), device=self.device, dtype=self.dtype)
        else:
            weights = weights.to(device=self.device, dtype=self.dtype)

        if self._history is not None:
            for i in range(n):
                self._history.append(vectors[i].detach().clone())
            while len(self._history) > self.sliding_window_size:
                self._history.pop(0)

        centroids, counts = self._current_centroids()
        if centroids.shape[0] == 0:
            # optionally preserve first N items as separate centroids to avoid early collapse
            if self.init_preserve_first_n is not None and self.init_preserve_first_n > 0:
                to_preserve = min(self.init_preserve_first_n, n, self.max_centroids)
                for i in range(to_preserve):
                    self._sums.append((vectors[i] * weights[i]).detach().clone())
                    self._counts.append(float(weights[i].item()))
                if n > to_preserve:
                    self.add(vectors[to_preserve:], weights=weights[to_preserve:], similarity_threshold=similarity_threshold)
                return
            to_take = min(self.max_centroids, n)
            for i in range(to_take):
                self._sums.append((vectors[i] * weights[i]).detach().clone())
                self._counts.append(float(weights[i].item()))
            if n > to_take:
                self.add(vectors[to_take:], weights=weights[to_take:], similarity_threshold=similarity_threshold)
            return

        dists = self._pairwise_distance(vectors, centroids)  # (n, m)
        nearest = torch.argmin(dists, dim=1)
        nearest_dist = dists[torch.arange(n, device=self.device), nearest]

        for i in range(n):
            d = float(nearest_dist[i].item())
            idx = int(nearest[i].item())
            w = float(weights[i].item())
            v = vectors[i].detach().clone()
            if d <= float(similarity_threshold):
                # update sums and counts in-place on GPU
                self._sums[idx] = (self._sums[idx] + v * w).detach()
                self._counts[idx] = self._counts[idx] + w
            else:
                self._sums.append((v * w).detach())
                self._counts.append(w)

        if self.persistence_decay < 1.0:
            self._counts = [c * self.persistence_decay for c in self._counts]

        self._step += 1
        self._prune_low_count_centroids()
        self._compress_if_needed()

    def _prune_low_count_centroids(self):
        keep = [i for i, c in enumerate(self._counts) if c >= self.min_count_threshold]
        if len(keep) == len(self._counts):
            return
        self._sums = [self._sums[i] for i in keep]
        self._counts = [self._counts[i] for i in keep]

    def _compress_if_needed(self):
        while len(self._sums) > self.max_centroids:
            centroids, counts = self._current_centroids()
            m = centroids.shape[0]
            if m <= 1:
                break
            dists = self._pairwise_distance(centroids, centroids)
            inf = torch.tensor(float("inf"), device=self.device, dtype=self.dtype)
            dists.fill_diagonal_(inf)
            # find closest pair (minimum distance)
            idx = int(torch.argmin(dists).item())
            i, j = divmod(idx, dists.shape[1])
            # If configured, compute cosine similarity and if below threshold, instead merge two smallest-weight centroids
            if self.min_merge_similarity is not None and self.distance == "cosine":
                # compute similarity matrix
                cn = _normalize_rows_t(centroids)
                sim_mat = torch.matmul(cn, cn.T)
                sim_mat.fill_diagonal_(-float("inf"))
                max_sim = float(torch.max(sim_mat).item())
                if max_sim < float(self.min_merge_similarity):
                    # merge two smallest-weight centroids
                    counts_tensor = torch.tensor(self._counts, device=self.device, dtype=self.dtype)
                    sorted_idx = torch.argsort(counts_tensor)
                    if sorted_idx.numel() >= 2:
                        i = int(sorted_idx[0].item())
                        j = int(sorted_idx[1].item())

            # weighted merge: sums and counts
            self._sums[i] = (self._sums[i] + self._sums[j]).detach()
            self._counts[i] = self._counts[i] + self._counts[j]
            # remove the j-th (merged data is now in i, so remove j regardless of i/j order)
            # Always remove the higher index first to avoid shifting issues
            if j > i:
                self._sums.pop(j)
                self._counts.pop(j)
            else:
                # j < i, so remove j first, then i becomes j's old position
                self._sums.pop(j)
                self._counts.pop(j)

    def get_centroids(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (centroids (C,d) torch, counts (C,) torch) on device"""
        return self._current_centroids()

    def get_centroids_numpy(self) -> Tuple:
        """Convert centroids and counts to numpy (CPU) - call only when needed."""
        centroids, counts = self._current_centroids()
        return centroids.detach().cpu().numpy(), counts.detach().cpu().numpy()

    def energy_loss(self, lambda_diversity: float = 0.0) -> float:
        # approximate using history if present (move to cpu for computation)
        if self._history is None or len(self._history) == 0:
            return 0.0
        hist = torch.stack(self._history, dim=0).to(self.device, dtype=self.dtype)
        centroids, _ = self._current_centroids()
        dists = self._pairwise_distance(hist, centroids)
        nearest = torch.argmin(dists, dim=1)
        sse = 0.0
        for idx in range(centroids.shape[0]):
            sel = hist[nearest == idx]
            if sel.numel() == 0:
                continue
            dif = sel - centroids[idx : idx + 1]
            sse += float((dif * dif).sum().item())
        # diversity on device
        diversity = 0.0
        if lambda_diversity > 0.0 and centroids.shape[0] > 1:
            pd = self._pairwise_distance(centroids, centroids)
            diversity = float(pd.triu(1).sum().item())
        return float(sse + lambda_diversity * diversity)


__all__ = ["TorchOnlineManifoldCluster"]


