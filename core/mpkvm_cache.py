"""
MPKVM Cache: Robust KV Cache Implementation with Manifold Partitioning

This module provides a stable MPKVMCache class that properly inherits from
transformers.Cache instead of using fragile monkey-patching.

The cache maintains compressed KV representations through online manifold clustering,
while providing the standard Cache interface for seamless integration.
"""
from __future__ import annotations
import torch
from typing import Optional, Dict, Any, Tuple, List
from transformers.cache_utils import Cache
from .clustering_torch import TorchOnlineManifoldCluster


class MPKVMCache(Cache):
    """
    MP-KVM Cache implementation that properly integrates with transformers.Cache.

    This cache maintains compressed KV representations through online manifold clustering
    while providing standard Cache interface methods for seamless model integration.

    Key features:
    - Maintains compressed centroid representations instead of full KV cache
    - Uses proper RoPE alignment for similarity computation
    - Implements EMA updates to prevent centroid freezing
    - Provides standard transformers.Cache interface
    """

    def __init__(
        self,
        config,
        max_batch_size: int = 1,
        max_cache_len: int = 8192,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cluster_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.config = config
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        # Extract model dimensions
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Initialize clustering with robust defaults
        self.cluster_kwargs = cluster_kwargs or {
            'ema_decay': 0.99,  # Exponential Moving Average to prevent centroid freezing
            'use_rotary_alignment': True,  # Use proper RoPE alignment instead of discarding phase
            'max_centroids': 1024,
            'distance': 'cosine',
            'min_merge_similarity': 0.8,  # Minimum similarity for centroid merging
        }

        # Initialize per-layer clusterers
        self.clusterers = {}
        for layer_idx in range(self.num_layers):
            self.clusterers[layer_idx] = TorchOnlineManifoldCluster(
                dim=self.head_dim,
                device=self.device,
                dtype=self.dtype,
                **self.cluster_kwargs
            )

        # Track sequence lengths per layer
        self._seq_lengths = [0] * self.num_layers

        # Store centroids for reconstruction during attention
        self.centroid_keys: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self.centroid_values: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self.centroid_weights: List[Optional[torch.Tensor]] = [None] * self.num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new KV states and return augmented states for attention.

        This method:
        1. Adds new KV states to the manifold clusterer
        2. Retrieves current centroids
        3. Returns augmented KV states (original + centroids) for attention

        Args:
            key_states: (batch_size, num_heads, seq_len, head_dim) - New key states
            value_states: (batch_size, num_heads, seq_len, head_dim) - New value states
            layer_idx: Current layer index
            cache_kwargs: Additional cache arguments (unused)

        Returns:
            Tuple of augmented (key_states, value_states) ready for attention
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Get clusterer for this layer
        clusterer = self.clusterers[layer_idx]

        # Flatten batch and heads for clustering: (batch_size * num_heads, seq_len, head_dim)
        flat_keys = key_states.view(-1, seq_len, head_dim)
        flat_values = value_states.view(-1, seq_len, head_dim)

        # Add new tokens to clusterer (this updates centroids internally)
        for seq_idx in range(flat_keys.shape[0]):
            k_seq = flat_keys[seq_idx]  # (seq_len, head_dim)
            v_seq = flat_values[seq_idx]  # (seq_len, head_dim)
            clusterer.add(k_seq, weights=None)

        # Get current centroids for attention augmentation
        centroids_k, centroid_weights = clusterer._current_centroids()

        # Update cached centroids for this layer
        self.centroid_keys[layer_idx] = centroids_k
        self.centroid_weights[layer_idx] = centroid_weights

        # For values, use centroids as approximation (can be improved with proper value clustering)
        if centroids_k is not None and centroids_k.shape[0] > 0:
            self.centroid_values[layer_idx] = centroids_k.clone()

        # Augment KV states with centroids for attention
        if centroids_k is not None and centroids_k.shape[0] > 0:
            # Expand centroids to match batch/heads dimensions
            # centroids_k: (n_centroids, head_dim) -> (1, 1, n_centroids, head_dim)
            centroids_k_expanded = centroids_k.unsqueeze(0).unsqueeze(0)
            centroids_v_expanded = centroids_k_expanded.clone()  # Use keys as values for now

            # Concatenate original KV with centroids
            # key_states: (batch_size, num_heads, seq_len, head_dim)
            # centroids_k_expanded: (1, 1, n_centroids, head_dim)
            aug_keys = torch.cat([key_states, centroids_k_expanded.expand(batch_size, num_heads, -1, -1)], dim=2)
            aug_values = torch.cat([value_states, centroids_v_expanded.expand(batch_size, num_heads, -1, -1)], dim=2)
        else:
            # No centroids yet, return original states
            aug_keys = key_states
            aug_values = value_states

        # Update sequence length tracking
        self._seq_lengths[layer_idx] = seq_len

        return aug_keys, aug_values

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get sequence length for layer (required by Cache interface)"""
        return self._seq_lengths[layer_idx or 0]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """
        Convert to legacy cache format for compatibility.

        MP-KVM doesn't maintain traditional KV cache, so we return centroids
        in a format that can be used for reconstruction.
        """
        # Convert centroids to legacy format
        legacy_keys = []
        legacy_values = []

        for layer_idx in range(self.num_layers):
            if self.centroid_keys[layer_idx] is not None:
                # Expand centroids to match expected shape: (batch_size, num_heads, seq_len, head_dim)
                # For legacy compatibility, we treat centroids as a "sequence"
                centroids_k = self.centroid_keys[layer_idx]  # (n_centroids, head_dim)
                centroids_v = self.centroid_values[layer_idx]  # (n_centroids, head_dim)

                # Reshape to (1, num_heads, n_centroids, head_dim) assuming single batch/head for legacy
                batch_size = 1
                n_centroids = centroids_k.shape[0]
                legacy_k = centroids_k.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_key_value_heads, n_centroids, -1)
                legacy_v = centroids_v.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_key_value_heads, n_centroids, -1)

                legacy_keys.append(legacy_k)
                legacy_values.append(legacy_v)
            else:
                # No centroids, create empty tensors
                empty_k = torch.empty((1, self.num_key_value_heads, 0, self.head_dim), device=self.device, dtype=self.dtype)
                empty_v = torch.empty((1, self.num_key_value_heads, 0, self.head_dim), device=self.device, dtype=self.dtype)
                legacy_keys.append(empty_k)
                legacy_values.append(empty_v)

        return tuple(legacy_keys), tuple(legacy_values)

    def get_max_length(self) -> Optional[int]:
        """Get maximum cache length"""
        return self.max_cache_len

    def reset(self):
        """Reset cache state"""
        self._seq_lengths = [0] * self.num_layers
        self.centroid_keys = [None] * self.num_layers
        self.centroid_values = [None] * self.num_layers
        self.centroid_weights = [None] * self.num_layers

        # Reset clusterers
        for clusterer in self.clusterers.values():
            # Clear internal state
            clusterer._sums.clear()
            clusterer._counts.clear()
            clusterer._step = 0
            if clusterer._history is not None:
                clusterer._history.clear()

    def get_centroids(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get centroids for a specific layer (for debugging/analysis)

        Returns:
            Tuple of (centroid_keys, centroid_values, centroid_weights)
        """
        return (
            self.centroid_keys[layer_idx],
            self.centroid_values[layer_idx],
            self.centroid_weights[layer_idx]
        )
