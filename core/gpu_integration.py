"""
GPU-side MPKVM manager that uses TorchOnlineManifoldCluster for in-GPU aggregation.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from .clustering_torch import TorchOnlineManifoldCluster


class TorchMPKVMManager:
    """
    Per-layer torch-based manager. Accepts torch tensors directly and performs
    clustering/aggregation on device to avoid frequent CPU-GPU transfers.
    """

    def __init__(self, dim: int, num_layers: int = 32, cluster_kwargs: Optional[dict] = None, device: Optional[torch.device] = None):
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        cluster_kwargs = cluster_kwargs or {}
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._layers: Dict[int, TorchOnlineManifoldCluster] = {
            i: TorchOnlineManifoldCluster(dim=dim, device=self.device, **cluster_kwargs) for i in range(self.num_layers)
        }

    def add_kv_tensor(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, weights: Optional[torch.Tensor] = None, similarity_threshold: float = 0.1):
        """
        k, v: torch tensors on some device (will be moved to manager.device)
        Shapes expected: (B, S, H, D_head) or (B*S*H, D)
        This function flattens to (N, D) before adding.
        """
        if layer_idx not in self._layers:
            self._layers[layer_idx] = TorchOnlineManifoldCluster(dim=self.dim, device=self.device)

        kn = k.to(device=self.device)
        vn = v.to(device=self.device)
        # flatten last dim as vector dims
        if kn.dim() >= 3:
            kn_flat = kn.reshape(-1, kn.shape[-1]).to(device=self.device)
            vn_flat = vn.reshape(-1, vn.shape[-1]).to(device=self.device)
        else:
            kn_flat = kn.to(device=self.device)
            vn_flat = vn.to(device=self.device)

        if weights is not None:
            w = weights.to(device=self.device).reshape(-1)
        else:
            w = None
        self._layers[layer_idx].add(kn_flat, weights=w, similarity_threshold=similarity_threshold)

    def get_layer_centroids_numpy(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if layer_idx not in self._layers:
            return np.zeros((0, self.dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return self._layers[layer_idx].get_centroids_numpy()

    def energy_loss(self, lambda_diversity: float = 0.0) -> Dict[int, float]:
        return {i: c.energy_loss(lambda_diversity=lambda_diversity) for i, c in self._layers.items()}


__all__ = ["TorchMPKVMManager"]


