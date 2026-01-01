"""
GPU-side lightweight KV aggregator for MP-KVM.

Design:
- Maintain per-layer GPU centroids as summed key/value tensors and total weights (torch tensors on device).
- When the number of GPU centroids for a layer exceeds a threshold, flush aggregated centroids to a CPU MPKVMManager
  by converting to numpy and calling its `add_kv`.
- This minimizes frequent CPU-GPU copies by performing local merging on-device.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import numpy as np

# Debug flag for GPU aggregator
_DEBUG = os.getenv("MPKVM_DEBUG", "1") == "1"


class MPKVMGPUAggregator:
    def __init__(
        self,
        cpu_manager: Any,
        dim: int,
        device: Optional[str] = None,
        max_gpu_centroids_per_layer: int = 512,
        similarity_threshold: float = 0.1,
        head_mean: bool = False,
        sample_stride: int = 1,
        flush_threshold: int = 256,  # Trigger flush when GPU centroids exceed this
        flush_interval: int = 100,   # Flush every N additions to hide latency
    ):
        """
        cpu_manager: an instance of core.integration.MPKVMManager (expects numpy add_kv)
        dim: hidden dimension
        device: torch device string (e.g., 'cuda:0') or None (auto)
        flush_threshold: Trigger CPU flush when GPU centroids exceed this number
        flush_interval: Force flush every N additions to prevent unbounded growth
        """
        self.cpu_manager = cpu_manager
        self.dim = int(dim)
        self.max_gpu_centroids_per_layer = int(max_gpu_centroids_per_layer)
        self.similarity_threshold = float(similarity_threshold)
        self.head_mean = bool(head_mean)
        self.sample_stride = int(sample_stride)
        self.device = device

        # Trigger-based flushing parameters
        self.flush_threshold = int(flush_threshold)
        self.flush_interval = int(flush_interval)
        self._addition_counter = 0  # Track additions since last flush

        # per-layer storage: maps layer_idx -> list of sum_k (torch), sum_v (torch), count (float)
        self._layers: Dict[int, Dict[str, List]] = {}

    def _ensure_tensor_storage(self, layer_idx: int) -> Dict[str, Any]:
        """
        Ensure tensor storage format for the given layer.
        Convert legacy list storage to tensor storage if needed.
        This prevents the V-centroid risk by guaranteeing tensor existence.
        """
        storage = self._layers[layer_idx]
        import torch

        # Check if we need to convert from list to tensor storage
        if (storage.get("centroid_k_tensor", None) is None and
            storage.get("sum_k", None) is not None and
            len(storage["sum_k"]) > 0):

            # Convert list storage to tensor storage
            sum_k_list = storage["sum_k"]
            sum_v_list = storage["sum_v"]
            count_list = storage["count"]

            # Stack and move to device
            device = sum_k_list[0].device if hasattr(sum_k_list[0], 'device') else self.device
            if self.device is not None:
                device = torch.device(self.device)

            centroid_k = torch.stack(sum_k_list).to(device)
            centroid_v = torch.stack(sum_v_list).to(device)
            counts = torch.tensor(count_list, dtype=torch.float32, device=device)

            # Store as tensors
            storage["centroid_k_tensor"] = centroid_k
            storage["centroid_v_tensor"] = centroid_v
            storage["counts_tensor"] = counts

            # Clear list storage to avoid confusion
            storage.pop("sum_k", None)
            storage.pop("sum_v", None)
            storage.pop("count", None)

            if _DEBUG:
                print(f"[MPKVM][GPU] Converted list storage to tensor storage for layer {layer_idx}")

        return storage

    def _ensure_layer(self, layer_idx: int):
        if layer_idx not in self._layers:
            self._layers[layer_idx] = {"sum_k": [], "sum_v": [], "count": []}

    def _get_current_gpu_centroids_count(self, layer_idx: int) -> int:
        """Get current number of GPU centroids for a layer."""
        if layer_idx not in self._layers:
            return 0
        storage = self._layers[layer_idx]
        # Check optimized tensor storage first
        if storage.get("centroid_k_tensor", None) is not None:
            return storage["centroid_k_tensor"].shape[0]
        # Fallback to legacy list storage
        return len(storage["sum_k"])

    def add_kv_torch(self, layer_idx: int, k_tensor, v_tensor, weights: Optional[Any] = None):
        """
        Accept torch tensors for K and V and perform on-device reduction.
        Expected k_tensor shape: (B, S, H, D_head) or (B, S, D)
        This function will perform head-averaging if configured and flatten tokens.
        """
        import torch
        # move to configured device if needed
        dev = torch.device(self.device) if self.device is not None else k_tensor.device
        if k_tensor.device != dev:
            k_tensor = k_tensor.to(dev)
        if v_tensor.device != dev:
            v_tensor = v_tensor.to(dev)
        # convert shape
        if k_tensor.ndim == 4:
            # (B, S, H, D)
            if self.head_mean:
                k_proc = k_tensor.mean(dim=2)  # (B, S, D)
                v_proc = v_tensor.mean(dim=2)
            else:
                # reshape to (B*S*H, D)
                k_proc = k_tensor.reshape(-1, k_tensor.shape[-1])
                v_proc = v_tensor.reshape(-1, v_tensor.shape[-1])
        elif k_tensor.ndim == 3:
            k_proc = k_tensor.reshape(-1, k_tensor.shape[-1])
            v_proc = v_tensor.reshape(-1, v_tensor.shape[-1])
        else:
            k_proc = k_tensor.reshape(-1, k_tensor.shape[-1])
            v_proc = v_tensor.reshape(-1, v_tensor.shape[-1])

        # optional subsampling
        if self.sample_stride is not None and self.sample_stride > 1:
            k_proc = k_proc[:: self.sample_stride]
            v_proc = v_proc[:: self.sample_stride]

        self._ensure_layer(layer_idx)
        layer_storage = self._layers[layer_idx]
        sum_k_list = layer_storage["sum_k"]
        sum_v_list = layer_storage["sum_v"]
        count_list = layer_storage["count"]

        # greedy on-device merge: for each vector, either merge to nearest centroid or append
        if len(sum_k_list) == 0:
            # initialize from batch by promoting up to capacity vectors as initial centroids
            n_init = min(k_proc.shape[0], self.max_gpu_centroids_per_layer)
            # VECTORIZED INITIALIZATION: Use tensor slicing instead of Python loops
            init_k = k_proc[:n_init].clone()  # (n_init, D)
            init_v = v_proc[:n_init].clone()  # (n_init, D)
            init_counts = torch.ones(n_init, dtype=torch.float32, device=k_proc.device)

            # Convert to lists for storage (still needed for legacy compatibility)
            sum_k_list.extend(init_k)
            sum_v_list.extend(init_v)
            count_list.extend(init_counts.tolist())

            if k_proc.shape[0] > n_init:
                rem = k_proc[n_init:]
                rem_v = v_proc[n_init:]
            else:
                rem = None
                rem_v = None
        else:
            # stack existing centroids for distance computation
            centroid_k = torch.stack(sum_k_list, dim=0)  # (C, D) but sums not normalized
            centroid_count = torch.tensor(count_list, device=centroid_k.device, dtype=centroid_k.dtype)
            centroid_mean = centroid_k / centroid_count[:, None]
            # normalize for cosine
            if self.similarity_threshold >= 0.0:
                k_norm = k_proc / (k_proc.norm(dim=1, keepdim=True) + 1e-12)
                c_norm = centroid_mean / (centroid_mean.norm(dim=1, keepdim=True) + 1e-12)
                sims = torch.matmul(k_norm, c_norm.T)  # (N, C)
                best_sim, best_idx = sims.max(dim=1)
                # VECTORIZED ASSIGNMENT: Use PyTorch operations instead of Python loops
                # This eliminates the performance bottleneck and provides fair comparison

                # Create masks for merge vs create decisions
                should_merge = best_sim >= float(self.similarity_threshold)

                if should_merge.any():
                    # For vectors that should merge: accumulate into existing centroids
                    merge_indices = best_idx[should_merge]
                    merge_vectors_k = k_proc[should_merge]
                    merge_vectors_v = v_proc[should_merge]

                    # Use scatter_add for efficient accumulation
                    # Convert sum_k_list to tensor for vectorized operations
                    sum_k_tensor = torch.stack(sum_k_list, dim=0)  # (C, D)
                    sum_v_tensor = torch.stack(sum_v_list, dim=0)  # (C, D)
                    count_tensor = torch.tensor(count_list, device=sum_k_tensor.device, dtype=sum_k_tensor.dtype)

                    # Accumulate using scatter_add
                    sum_k_tensor.scatter_add_(0, merge_indices.unsqueeze(-1).expand(-1, sum_k_tensor.shape[-1]), merge_vectors_k)
                    sum_v_tensor.scatter_add_(0, merge_indices.unsqueeze(-1).expand(-1, sum_v_tensor.shape[-1]), merge_vectors_v)
                    count_tensor.scatter_add_(0, merge_indices, torch.ones_like(merge_indices, dtype=count_tensor.dtype))

                    # Update the lists with modified tensors
                    sum_k_list[:] = [sum_k_tensor[i] for i in range(len(sum_k_list))]
                    sum_v_list[:] = [sum_v_tensor[i] for i in range(len(sum_v_list))]
                    count_list[:] = count_tensor.tolist()

                # For vectors that should create new centroids: append them
                should_create = ~should_merge
                if should_create.any():
                    create_vectors_k = k_proc[should_create]
                    create_vectors_v = v_proc[should_create]

                    # VECTORIZED APPEND: Extend lists with tensor data
                    sum_k_list.extend(create_vectors_k)
                    sum_v_list.extend(create_vectors_v)
                    count_list.extend([1.0] * create_vectors_k.shape[0])
            else:
                # fallback: just append (vectorized)
                sum_k_list.extend(k_proc.clone())
                sum_v_list.extend(v_proc.clone())
                count_list.extend([1.0] * k_proc.shape[0])

        # flush to CPU manager if too many GPU centroids
        if len(sum_k_list) >= self.max_gpu_centroids_per_layer:
            self.flush_layer_to_cpu(layer_idx)

    def flush_layer_to_cpu(self, layer_idx: int):
        """Convert GPU aggregated centroids to numpy and call cpu_manager.add_kv"""
        import torch
        if layer_idx not in self._layers:
            return
        # Ensure tensor storage format before flushing
        storage = self._ensure_tensor_storage(layer_idx)
        # support two storage formats: list-based (legacy) or tensor-based (optimized)
        sum_k_list = storage.get("sum_k", None)
        sum_v_list = storage.get("sum_v", None)
        count_list = storage.get("count", None)
        cent_k = None
        cent_v = None

        # Ensure tensor storage format before accessing
        storage = self._ensure_tensor_storage(layer_idx)

        if storage.get("centroid_k_tensor", None) is not None and storage.get("counts_tensor", None) is not None:
            cent_k = storage["centroid_k_tensor"]
            # CRITICAL: Value centroids MUST exist separately - no fallback to Key
            if storage.get("centroid_v_tensor", None) is None:
                raise RuntimeError("[MPKVM][ERROR] Missing centroid_v_tensor in GPU storage. "
                                 "Value vectors were not properly aggregated. "
                                 "MP-KVM requires separate K and V centroid computation for manifold partitioning.")
            cent_v = storage["centroid_v_tensor"]
            counts = storage["counts_tensor"]
            # ensure tensors detached
            cent_k = cent_k.detach()
            cent_v = cent_v.detach()
            counts = counts.detach()
        else:
            if sum_k_list is None or len(sum_k_list) == 0:
                return
            # compute centroids as sums / counts
            ks = torch.stack(sum_k_list, dim=0)
            vs = torch.stack(sum_v_list, dim=0)
            # ensure counts are float32 for division
            counts = torch.tensor(count_list, device=ks.device, dtype=torch.float32)
            cent_k = ks / counts[:, None]
            cent_v = vs / counts[:, None]

        # move to cpu numpy and call cpu_manager.add_kv
        cent_k_cpu_tensor = cent_k.detach()
        # CRITICAL: cent_v must exist - no fallback
        if cent_v is None:
            raise RuntimeError("[MPKVM][ERROR] cent_v is None in flush_layer_to_cpu. "
                             "Value centroids were not computed properly.")
        cent_v_cpu_tensor = cent_v.detach()
        # cast half precision / bfloat16 to float32 before numpy conversion
        if cent_k_cpu_tensor.dtype in (torch.bfloat16, torch.float16):
            cent_k_cpu_tensor = cent_k_cpu_tensor.to(dtype=torch.float32)
        if cent_v_cpu_tensor.dtype in (torch.bfloat16, torch.float16):
            cent_v_cpu_tensor = cent_v_cpu_tensor.to(dtype=torch.float32)
        cent_k_cpu = cent_k_cpu_tensor.cpu().numpy().astype(np.float32)
        cent_v_cpu = cent_v_cpu_tensor.cpu().numpy().astype(np.float32)
        # Extract counts as weights for proper density preservation
        counts_cpu = counts.detach().cpu().numpy().astype(np.float32)
        # call cpu manager in one batch with weights
        try:
            self.cpu_manager.add_kv(layer_idx, cent_k_cpu, cent_v_cpu, weights=counts_cpu)
        except Exception as e:
            # best-effort, ignore errors during flush
            print(f"Warning: Failed to flush layer {layer_idx} to CPU: {e}")
          

        # clear gpu storage for this layer
        self._layers[layer_idx] = {"sum_k": [], "sum_v": [], "count": []}

    def flush_all_to_cpu(self):
        for layer_idx in list(self._layers.keys()):
            self.flush_layer_to_cpu(layer_idx)

    def record_attention(self, layer_idx: int, attn_array: "np.ndarray"):
        """
        Receive attention weights (numpy array) and forward to the CPU manager if it
        implements `record_attention`. If not available, attempt a best-effort save
        to disk under an `_attn_out_dir` if configured on this aggregator or cpu_manager.
        This allows adapters to always call `manager.record_attention(...)` without
        crashing when a GPU-side aggregator is active.
        """
        # Prefer forwarding to cpu_manager.record_attention if available
        if hasattr(self.cpu_manager, "record_attention") and callable(getattr(self.cpu_manager, "record_attention")):
            self.cpu_manager.record_attention(layer_idx, attn_array)
            return

        # Best-effort: save to configured attn output directory if present
        out_base = getattr(self, "_attn_out_dir", None) or getattr(self.cpu_manager, "_attn_out_dir", None)
        if out_base:
            import os
            import time
            import numpy as _np

            out_layer_dir = os.path.join(out_base, f"layer_{layer_idx}")
            os.makedirs(out_layer_dir, exist_ok=True)
            fname = os.path.join(out_layer_dir, f"attn_gpu_{int(time.time() * 1000)}.npy")
            # ensure numpy array type
            _np.save(fname, _np.asarray(attn_array))
       
    def get_gpu_centroids(self, layer_idx: int):
        """
        Return centroids on-device as (centroids_tensor, counts_tensor) if available.
        Centroids are computed as sums / counts and kept on the device of the stored tensors.
        Returns (None, None) if layer has no GPU aggregated centroids.
        """
        import torch
        if layer_idx not in self._layers:
            return None, None
        storage = self._layers[layer_idx]
        # support optimized tensor-backed storage
        if storage.get("centroid_k_tensor", None) is not None and storage.get("counts_tensor", None) is not None:
            centroids = storage["centroid_k_tensor"]
            counts = storage["counts_tensor"]
            if centroids.numel() == 0:
                return None, None
            return centroids, counts
        # fallback to legacy list storage
        sum_k_list = storage.get("sum_k", [])
        sum_v_list = storage.get("sum_v", [])
        count_list = storage.get("count", [])
        if len(sum_k_list) == 0:
            return None, None
        ks = torch.stack(sum_k_list, dim=0)
        counts = torch.tensor(count_list, device=ks.device, dtype=torch.float32)
        centroids = ks / counts[:, None]

        # Also create V centroids and store as tensor-backed storage
        if len(sum_v_list) > 0:
            vs = torch.stack(sum_v_list, dim=0)
            centroid_v = vs / counts[:, None]
            storage["centroid_k_tensor"] = centroids.detach()
            storage["centroid_v_tensor"] = centroid_v.detach()
            storage["counts_tensor"] = counts.detach()

        return centroids, counts



__all__ = ["MPKVMGPUAggregator"]


class MPKVMGPUAggregatorOptimized(MPKVMGPUAggregator):
    """
    Optimized GPU-side aggregator that vectorizes assignment and merging operations
    to reduce Python loop overhead. Maintains per-layer centroids as tensors on the
    configured device and updates sums/counts in batched operations.
    """

    def add_kv_torch(self, layer_idx: int, k_tensor, v_tensor, weights: Optional[Any] = None):
        import torch
        # flatten incoming keys to (N, D) on target device
        dev = torch.device(self.device) if self.device is not None else k_tensor.device
        if k_tensor.device != dev:
            k_tensor = k_tensor.to(dev)
        if v_tensor.device != dev:
            v_tensor = v_tensor.to(dev)

        if k_tensor.ndim >= 2:
            k_flat = k_tensor.reshape(-1, k_tensor.shape[-1]).to(device=dev)
            v_flat = v_tensor.reshape(-1, v_tensor.shape[-1]).to(device=dev)
        else:
            k_flat = k_tensor.reshape(-1, self.dim).to(device=dev)
            v_flat = v_tensor.reshape(-1, self.dim).to(device=dev)

        if self.sample_stride is not None and self.sample_stride > 1:
            k_flat = k_flat[:: self.sample_stride]
            v_flat = v_flat[:: self.sample_stride]

        self._ensure_layer(layer_idx)
        storage = self._layers[layer_idx]
        sum_k_list = storage["sum_k"]
        sum_v_list = storage["sum_v"]
        count_list = storage["count"]

        # If no existing centroids (check both list and tensor storage)
        has_existing_centroids = (len(sum_k_list) > 0 or
                                storage.get("centroid_k_tensor", None) is not None)
        if not has_existing_centroids:
            n_init = min(k_flat.shape[0], self.max_gpu_centroids_per_layer)
            # VECTORIZED INITIALIZATION: Use tensor slicing instead of loops
            init_k = k_flat[:n_init].clone()
            init_v = v_flat[:n_init].clone()
            sum_k_list.extend(init_k)
            sum_v_list.extend(init_v)
            count_list.extend([1.0] * n_init)
            if k_flat.shape[0] <= n_init:
                return
            rem_k = k_flat[n_init:]
            rem_v = v_flat[n_init:]
        else:
            rem_k = k_flat
            rem_v = v_flat

        if rem_k.numel() == 0:
            return

        # Stack existing centroids and compute mean vectors
        centroid_k = torch.stack(sum_k_list, dim=0)  # (C, D)
        centroid_v = torch.stack(sum_v_list, dim=0)  # (C, D) - Also stack V centroids
        counts = torch.tensor(count_list, device=centroid_k.device, dtype=centroid_k.dtype)
        centroid_mean = centroid_k / counts[:, None]

        # normalize routine
        def normalize(t):
            n = t.norm(dim=-1, keepdim=True)
            n[n == 0] = 1.0
            return t / n

        c_norm = normalize(centroid_mean)
        k_norm = normalize(rem_k)

        # compute similarities in one matmul: (N_rem, C)
        sims = torch.matmul(k_norm, c_norm.T)
        best_sim, best_idx = sims.max(dim=1)

        # mask for matches vs new
        match_mask = best_sim >= float(self.similarity_threshold)
        new_mask = ~match_mask

        # process matches: accumulate sums and counts per centroid in vectorized way
        if match_mask.any():
            matched_indices = best_idx[match_mask]  # indices into centroids
            matched_keys = rem_k[match_mask]
            matched_vals = rem_v[match_mask]
            # aggregate per-centroid by scatter_add
            C = centroid_k.shape[0]
            D = centroid_k.shape[1]
            delta_k = torch.zeros((C, D), device=centroid_k.device, dtype=centroid_k.dtype)
            delta_v = torch.zeros((C, D), device=centroid_k.device, dtype=centroid_k.dtype)
            delta_count = torch.zeros((C,), device=centroid_k.device, dtype=centroid_k.dtype)
            idx_expand = matched_indices.unsqueeze(-1).expand(-1, D)
            delta_k.scatter_add_(0, idx_expand, matched_keys)
            delta_v.scatter_add_(0, idx_expand, matched_vals)
            # counts
            one_vec = torch.ones((matched_indices.shape[0],), device=centroid_k.device, dtype=centroid_k.dtype)
            delta_count.scatter_add_(0, matched_indices, one_vec)
            # apply updates (vectorized) and persist as tensor-backed storage
            centroid_k = centroid_k + delta_k
            centroid_v = centroid_v + delta_v  # Also update V centroids
            counts = counts + delta_count
            centroid_mean = centroid_k / counts[:, None]
            # Persist updated centroids to storage
            storage["centroid_k_tensor"] = centroid_k.detach()
            storage["centroid_v_tensor"] = centroid_v.detach()
            storage["counts_tensor"] = counts.detach()
            # refresh normalized centroids
            c_norm = normalize(centroid_mean)

        # Always ensure tensor storage is initialized after processing matches or when centroids exist
        if len(sum_k_list) > 0:
            # If we have centroids but no tensor storage, initialize it now
            if storage.get("centroid_k_tensor", None) is None:
                storage["centroid_k_tensor"] = centroid_k.detach()
            if storage.get("centroid_v_tensor", None) is None:
                storage["centroid_v_tensor"] = centroid_v.detach()
            if storage.get("counts_tensor", None) is None:
                storage["counts_tensor"] = counts.detach()
            # clear legacy list storage to avoid duplication
            storage["sum_k"] = []
            storage["sum_v"] = []
            storage["count"] = []

        # process new vectors: append in batch up to capacity
        if new_mask.any():
            new_keys = rem_k[new_mask]
            new_vals = rem_v[new_mask]
            # append new vectors in batch up to capacity using tensor-backed storage
            # ensure we operate on tensors centroid_k and counts where available
            if storage.get("centroid_k_tensor", None) is not None and storage.get("counts_tensor", None) is not None:
                centk = storage["centroid_k_tensor"]
                cnts = storage["counts_tensor"]
            else:
                centk = torch.stack(sum_k_list, dim=0)
                cnts = torch.tensor(count_list, device=centk.device, dtype=centk.dtype)

            space_left = self.max_gpu_centroids_per_layer - centk.shape[0]
            if space_left > 0:
                to_take = min(space_left, new_keys.shape[0])
                if to_take > 0:
                    # CRITICAL FIX: Concatenate both K and V centroids together
                    centk = torch.cat([centk, new_keys[:to_take]], dim=0)

                    # Get existing V centroids (must exist since we have K centroids)
                    if storage.get("centroid_v_tensor", None) is not None:
                        centv = storage["centroid_v_tensor"]
                        centv = torch.cat([centv, new_vals[:to_take]], dim=0)
                        storage["centroid_v_tensor"] = centv
                    elif len(sum_v_list) > 0:
                        # Convert list storage to tensor and concatenate
                        centv = torch.stack(sum_v_list, dim=0)
                        centv = torch.cat([centv, new_vals[:to_take]], dim=0)
                        storage["centroid_v_tensor"] = centv
                        storage["sum_v"] = []  # Clear list storage
                    else:
                        # Initialize V centroids tensor directly
                        centv = new_vals[:to_take].clone()
                        storage["centroid_v_tensor"] = centv

                    cnts = torch.cat([cnts, torch.ones((to_take,), device=centk.device, dtype=cnts.dtype)], dim=0)
            # for remaining new vectors beyond capacity, merge into best existing centroid
            rem_start = min(new_keys.shape[0], max(0, space_left))
            if rem_start < new_keys.shape[0]:
                remaining = new_keys[rem_start:]
                remaining_vals = new_vals[rem_start:]
                # compute sims to existing centroids in batch
                rn = normalize(remaining)
                cnorm = normalize(centk / (cnts[:, None] + 1e-12))
                sims_rem = torch.matmul(rn, cnorm.T)
                best_sim_rem, best_idx_rem = sims_rem.max(dim=1)
                # Also need V centroids for merging remaining vectors
                if storage.get("centroid_v_tensor", None) is not None:
                    centv_existing = storage["centroid_v_tensor"]
                else:
                    # CRITICAL: Do NOT fallback to centk.clone() - this corrupts Value centroids!
                    # If we don't have V centroids, we cannot properly merge remaining vectors.
                    # This should be an error condition, not a silent fallback.
                    raise RuntimeError("[MPKVM][ERROR] Missing centroid_v_tensor during GPU merging. "
                                     "Value centroids must be properly initialized before merging operations. "
                                     "This indicates a bug in the GPU aggregation initialization.")
                for j in range(remaining.shape[0]):
                    idx2 = int(best_idx_rem[j].item())
                    centk[idx2] = centk[idx2] + remaining[j].detach()
                    centv_existing[idx2] = centv_existing[idx2] + remaining_vals[j].detach()
                    cnts[idx2] = cnts[idx2] + 1.0
                centv = centv_existing
            else:
                # Get V centroids from existing storage or create from list
                if storage.get("centroid_v_tensor", None) is not None:
                    centv = storage["centroid_v_tensor"]
                else:
                    # CRITICAL: Do NOT fallback to centk.clone() - this corrupts Value centroids!
                    # If we have V centroids in list form, use them; otherwise this is an error.
                    if len(sum_v_list) > 0:
                        centv = torch.stack(sum_v_list, dim=0)
                    else:
                        raise RuntimeError("[MPKVM][ERROR] No V centroids available for GPU storage. "
                                         "Value centroids must be properly maintained alongside Key centroids. "
                                         "This indicates a bug in GPU centroid initialization.")
            storage["centroid_k_tensor"] = centk.detach()
            storage["centroid_v_tensor"] = centv.detach()
            storage["counts_tensor"] = cnts.detach()
            storage["sum_k"] = []
            storage["sum_v"] = []
            storage["count"] = []

        # if we still exceed capacity (rare), perform greedy merge of most similar centroid pairs
        # Check both list-based and tensor-based storage
        def _get_current_centroid_count(storage):
            if storage.get("centroid_k_tensor", None) is not None:
                return storage["centroid_k_tensor"].shape[0]
            return len(sum_k_list)

        while _get_current_centroid_count(storage) > self.max_gpu_centroids_per_layer:
            # Handle both list-based and tensor-based storage
            if storage.get("centroid_k_tensor", None) is not None:
                # Tensor-based storage: operate on tensors directly
                cent_k = storage["centroid_k_tensor"]
                cent_v = storage["centroid_v_tensor"]
                counts = storage["counts_tensor"]
                Cc = cent_k.shape[0]

                # Compute centroids (means)
                centroid_mean = cent_k / counts[:, None]
                cn = normalize(centroid_mean)
                sim_mat = torch.matmul(cn, cn.T)
                sim_mat.fill_diagonal_(-float("inf"))
                # find pair to merge
                maxval, maxidx = torch.max(sim_mat.view(-1), dim=0)
                idx_i = int((maxidx // sim_mat.shape[1]).item())
                idx_j = int((maxidx % sim_mat.shape[1]).item())

                # weighted merge j into i
                wa = counts[idx_i]
                wb = counts[idx_j]
                cent_k = cent_k.clone()
                cent_v = cent_v.clone()
                counts = counts.clone()
                cent_k[idx_i] = (cent_k[idx_i] * wa + cent_k[idx_j] * wb) / (wa + wb + 1e-12)
                cent_v[idx_i] = (cent_v[idx_i] * wa + cent_v[idx_j] * wb) / (wa + wb + 1e-12)
                counts[idx_i] = wa + wb

                # remove j (tensor slicing)
                keep_mask = torch.arange(Cc, device=cent_k.device) != idx_j
                cent_k = cent_k[keep_mask]
                cent_v = cent_v[keep_mask]
                counts = counts[keep_mask]

                # Update storage
                storage["centroid_k_tensor"] = cent_k.detach()
                storage["centroid_v_tensor"] = cent_v.detach()
                storage["counts_tensor"] = counts.detach()
            else:
                # List-based storage (fallback)
                Cc = len(sum_k_list)
                cent_stack = torch.stack(sum_k_list, dim=0)
                cn = normalize(cent_stack / (torch.tensor(count_list, device=cent_stack.device, dtype=cent_stack.dtype)[:, None]))
                sim_mat = torch.matmul(cn, cn.T)
                sim_mat.fill_diagonal_(-float("inf"))
                # find pair to merge
                maxval, maxidx = torch.max(sim_mat.view(-1), dim=0)
                idx_i = int((maxidx // sim_mat.shape[1]).item())
                idx_j = int((maxidx % sim_mat.shape[1]).item())
                # weighted merge j into i
                wa = count_list[idx_i]
                wb = count_list[idx_j]
                sum_k_list[idx_i] = (sum_k_list[idx_i] * wa + sum_k_list[idx_j] * wb) / (wa + wb + 1e-12)
                # CRITICAL: Also merge V centroids with proper weighting
                sum_v_list[idx_i] = (sum_v_list[idx_i] * wa + sum_v_list[idx_j] * wb) / (wa + wb + 1e-12)
                count_list[idx_i] = float(wa + wb)
                # remove j
                del sum_k_list[idx_j]
                del sum_v_list[idx_j]
                del count_list[idx_j]

        # Trigger-based flushing: hide communication latency by flushing at optimal times
        current_gpu_centroids = self._get_current_gpu_centroids_count(layer_idx)
        self._addition_counter += 1

        should_flush = (
            current_gpu_centroids >= self.flush_threshold or  # Capacity threshold
            self._addition_counter >= self.flush_interval or   # Time-based interval
            current_gpu_centroids >= self.max_gpu_centroids_per_layer  # Hard limit
        )

        if should_flush:
            # Asynchronous flush to hide latency
            self.flush_layer_to_cpu(layer_idx)
            self._addition_counter = 0


