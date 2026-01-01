"""Clean MP-KVM integration layer: manager + monkey-patch helpers.

This module provides a lightweight MPKVMManager and utilities to
monkey-patch an attention module's forward to route KV storage
through the MP-KVM operator.
"""
from __future__ import annotations
import os
from typing import Any, Optional, Tuple, Dict
import numpy as np
from .clustering import OnlineManifoldClustering


class MPKVMManager:
    """
    Manager that holds per-layer clustering operators and exposes
    an API for attention layers to add KV vectors and retrieve centroids.
    """

    def __init__(self, dim: int, num_layers: int = 32, num_heads: int = 32, per_head_clustering: bool = False, **cluster_kwargs):
        """
        Manager that holds per-layer (and optionally per-head) clustering operators.
        Accepts `num_layers` for compatibility with older callers (also previously used `layers`).
        """
        self.layers = {}
        self.dim = dim
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.per_head_clustering = bool(per_head_clustering)

        # support callers passing a single `cluster_kwargs` dict (e.g., MPKVMManager(..., cluster_kwargs={...}))
        if "cluster_kwargs" in cluster_kwargs and isinstance(cluster_kwargs["cluster_kwargs"], dict):
            cluster_kwargs = cluster_kwargs["cluster_kwargs"]

        # normalize common naming differences to the clustering constructor
        # e.g., sliding_window_size -> window_size, max_centroids_per_layer -> max_centroids
        if "sliding_window_size" in cluster_kwargs:
            cluster_kwargs["window_size"] = cluster_kwargs.pop("sliding_window_size")
        if "max_centroids_per_layer" in cluster_kwargs:
            cluster_kwargs["max_centroids"] = cluster_kwargs.pop("max_centroids_per_layer")

        # Store normalized cluster kwargs for later use in dimension resets
        self._cluster_kwargs = cluster_kwargs.copy()

        # Initialize clustering operators
        for l in range(self.num_layers):
            if self.per_head_clustering:
                # Per-head clustering: each layer has multiple clusterers (one per head)
                self.layers[l] = [OnlineManifoldClustering(dim=dim, **cluster_kwargs) for _ in range(self.num_heads)]
            else:
                # Standard per-layer clustering
                self.layers[l] = OnlineManifoldClustering(dim=dim, **cluster_kwargs)
        # container to store traced attention weights per-layer (list of numpy arrays)
        self._attn_weights = {l: [] for l in range(self.num_layers)}
        # optional output directory for immediate attention dumps
        self._attn_out_dir: Optional[str] = None
        # per-layer counters for file naming
        self._attn_counters: Dict[int, int] = {}
        # CRITICAL FIX: Track previously processed sequence lengths per layer
        # This prevents the severe bug of feeding full historical cache repeatedly
        self._processed_lengths: Dict[int, int] = {}

    def set_attn_out_dir(self, path: str) -> None:
        """
        Configure a directory where attention numpy arrays will be written immediately.
        Creates per-layer subdirectories to avoid races later.
        """
        self._attn_out_dir = str(path)
        os.makedirs(self._attn_out_dir, exist_ok=True)
        for l in range(self.num_layers):
            os.makedirs(os.path.join(self._attn_out_dir, f"layer_{l}"), exist_ok=True)
        # initialize counters if not present
        for l in range(self.num_layers):
            self._attn_counters.setdefault(l, 0)

    def add_kv(self, layer_idx: int, keys: np.ndarray, values: np.ndarray, weights: Optional[np.ndarray] = None, head_idx: Optional[int] = None):
        # If keys provided, infer dimensionality and ensure layer cluster matches it.
        if keys is not None and hasattr(keys, "shape") and keys.ndim == 2:
            key_dim = int(keys.shape[1])
        else:
            key_dim = self.dim

        if layer_idx not in self.layers:
            # lazily create cluster operator with inferred key dim
            if self.per_head_clustering:
                self.layers[layer_idx] = [OnlineManifoldClustering(dim=key_dim, **self._cluster_kwargs) for _ in range(self.num_heads)]
            else:
                self.layers[layer_idx] = OnlineManifoldClustering(dim=key_dim, **self._cluster_kwargs)

        # Get the appropriate cluster operator
        if self.per_head_clustering:
            if head_idx is None:
                raise ValueError(f"Per-head clustering enabled but head_idx not provided for layer {layer_idx}")
            if not isinstance(self.layers[layer_idx], list):
                # Convert single cluster to list for backward compatibility
                old_cluster = self.layers[layer_idx]
                self.layers[layer_idx] = [old_cluster] + [OnlineManifoldClustering(dim=key_dim, **self._cluster_kwargs) for _ in range(self.num_heads - 1)]

            cluster = self.layers[layer_idx][head_idx]
            # Check dimension compatibility
            if int(cluster.dim) != key_dim:
                print(f"[MPKVM][WARNING] Layer {layer_idx} Head {head_idx} dimension mismatch: expected {cluster.dim}, got {key_dim}. "
                      f"Resetting cluster - accumulated compression knowledge will be lost!")
                self.layers[layer_idx][head_idx] = OnlineManifoldClustering(dim=key_dim, **self._cluster_kwargs)
                cluster = self.layers[layer_idx][head_idx]
        else:
            cluster = self.layers[layer_idx]
            # if existing cluster dim mismatches incoming keys, reinit that layer's cluster to match keys
            existing_dim = int(cluster.dim)
            if existing_dim != key_dim:
                # WARNING: Dimension mismatch detected - this will reset accumulated compression knowledge
                print(f"[MPKVM][WARNING] Layer {layer_idx} dimension mismatch: expected {existing_dim}, got {key_dim}. "
                      f"Resetting cluster - accumulated compression knowledge will be lost!")
                # replace with a new cluster matching the incoming key dimensionality
                self.layers[layer_idx] = OnlineManifoldClustering(dim=key_dim, **self._cluster_kwargs)
                cluster = self.layers[layer_idx]

        # add to the cluster (weights may be None)
        cluster.add(keys, values, weights)

    def add_kv_incremental(self, layer_idx: int, keys: np.ndarray, values: np.ndarray,
                          current_seq_len: int, weights: Optional[np.ndarray] = None, head_idx: Optional[int] = None):
        """
        CRITICAL FIX: Add only the incremental KV vectors since last processing.

        This prevents the severe bug where full historical cache was repeatedly fed
        to the clustering operator during generation, causing exponential memory growth
        and incorrect weighting.

        Args:
            layer_idx: Layer index
            keys: Full current KV cache keys (shape: [seq_len, dim])
            values: Full current KV cache values (shape: [seq_len, dim])
            current_seq_len: Current total sequence length
            weights: Optional weights for each token
            head_idx: Optional head index for per-head clustering
        """
        if keys is None or keys.shape[0] == 0:
            return

        # Get previously processed length for this layer (and head if per-head)
        length_key = (layer_idx, head_idx) if self.per_head_clustering and head_idx is not None else layer_idx
        prev_len = self._processed_lengths.get(length_key, 0)

        # Calculate how many new tokens to process
        new_tokens = current_seq_len - prev_len
        if new_tokens <= 0:
            # No new tokens to process
            return

        # Extract only the new tokens from the end of the sequence
        start_idx = prev_len
        end_idx = min(current_seq_len, keys.shape[0])

        if start_idx >= end_idx:
            return

        new_keys = keys[start_idx:end_idx]
        new_values = values[start_idx:end_idx]
        new_weights = weights[start_idx:end_idx] if weights is not None else None

        # Process the new tokens
        self.add_kv(layer_idx, new_keys, new_values, new_weights, head_idx)

        # Update the processed length
        self._processed_lengths[length_key] = end_idx

    def get_layer_centroids(self, layer_idx: int, head_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if layer_idx not in self.layers:
            empty = np.zeros((0, self.dim), dtype=np.float32)
            return empty, empty, np.array([], dtype=int), np.array([], dtype=float)

        if self.per_head_clustering:
            if head_idx is None:
                # Return centroids from first head as default, or merge across heads
                if isinstance(self.layers[layer_idx], list) and len(self.layers[layer_idx]) > 0:
                    return self.layers[layer_idx][0].get_key_value_centroids()
                else:
                    empty = np.zeros((0, self.dim), dtype=np.float32)
                    return empty, empty, np.array([], dtype=int), np.array([], dtype=float)
            else:
                if isinstance(self.layers[layer_idx], list) and head_idx < len(self.layers[layer_idx]):
                    return self.layers[layer_idx][head_idx].get_key_value_centroids()
                else:
                    empty = np.zeros((0, self.dim), dtype=np.float32)
                    return empty, empty, np.array([], dtype=int), np.array([], dtype=float)
        else:
            return self.layers[layer_idx].get_key_value_centroids()

    def record_attention(self, layer_idx: int, attn_weights_np: np.ndarray) -> None:
        """
        Record attention weights (numpy) for the given layer.
        attn_weights_np expected shape: (..., seq_q, seq_k) or (seq_q, seq_k)
        """
        # Debug entry log to help trace why recordings may be missing.
        print(f"[MPKVM][layer {layer_idx}] record_attention called; out_dir={getattr(self, '_attn_out_dir', None)}")
        # Note: self._attn_weights is pre-initialized in __init__, so layer_idx is always present
        arr = np.asarray(attn_weights_np)
        while arr.ndim > 2:
            arr = arr.mean(axis=0)
        arr = arr.astype(np.float32)
        self._attn_weights[layer_idx].append(arr)
        # if output dir configured, write file immediately for ON/OFF pairing
        if getattr(self, "_attn_out_dir", None):
            out_dir = os.path.join(self._attn_out_dir, f"layer_{layer_idx}")
            os.makedirs(out_dir, exist_ok=True)
            cnt = self._attn_counters.get(layer_idx, 0)
            fname = os.path.join(out_dir, f"attn_{cnt:06d}.npy")
            np.save(fname, arr)
            self._attn_counters[layer_idx] = cnt + 1

    def get_recorded_attention(self, layer_idx: int):
        """Return list of recorded attention numpy arrays for layer or empty list."""
        return list(self._attn_weights.get(layer_idx, []))


def monkey_patch_attention_forward(attn_module: Any, manager: MPKVMManager, layer_idx: int):
    """
    Monkey-patches a huggingface-style attention module instance.
    The attn_module is expected to have a `forward` method with signature
    (hidden_states, past_key_value=None, attention_mask=None, *args, **kwargs)
    and to produce `key`, `value` tensors or expose them as attributes.

    This function wraps the forward to intercept the produced KV and pass to manager.add_kv.
    It's intentionally conservative and will not modify attention math itself.
    """

    original_forward = getattr(attn_module, "forward")

    def _patched_forward(*args, **kwargs):
        # call original forward, capture outputs
        outputs = original_forward(*args, **kwargs)
        # best-effort extraction of k/v from module attributes or outputs
        # huggingface often stores k/v as attn_module.k_proj(...), attn_module.v_proj(...)
        k = getattr(attn_module, "last_key", None)
        v = getattr(attn_module, "last_value", None)

        # If not available on module, try to extract from outputs (tuple)
        if k is None or v is None:
            if isinstance(outputs, tuple) and len(outputs) > 0:
                # heuristic: outputs[1] could be present_key_value
                pv = outputs[1] if len(outputs) > 1 else None
                if pv is not None and isinstance(pv, (list, tuple)) and len(pv) >= 2:
                    k, v = pv[0], pv[1]

        # Convert to numpy if tensors (avoid torch import at top-level)
        import torch
        def to_np(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return np.asarray(t)
        if k is None or v is None:
            raise RuntimeError(f"[MPKVM][ERROR] Failed to extract K and V tensors for layer {layer_idx}. "
                             f"MP-KVM requires proper KV tensor extraction to function. "
                             f"K: {type(k)}, V: {type(v)}")
        kn = to_np(k.reshape(-1, k.shape[-1]))
        vn = to_np(v.reshape(-1, v.shape[-1]))
        manager.add_kv(layer_idx, kn, vn)
        return outputs
    setattr(attn_module, "forward", _patched_forward)
    return attn_module

def patch_llama_attention(attn_module: Any, manager: MPKVMManager, layer_idx: int):
    """
    Compatibility wrapper (keeps previous API) that calls the generic monkey-patch.
    """
    return monkey_patch_attention_forward(attn_module, manager, layer_idx)


__all__ = ["MPKVMManager", "monkey_patch_attention_forward", "patch_llama_attention"]


