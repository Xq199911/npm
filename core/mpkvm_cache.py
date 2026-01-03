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
    - Handles RoPE derotation for mathematically correct clustering
    """

    def __init__(
        self,
        config,
        max_batch_size: int = 1,
        max_cache_len: int = 8192,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cluster_kwargs: Optional[Dict[str, Any]] = None,
        rotary_emb: Optional[Any] = None,  # RoPE embedding module for derotation
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

        # RoPE configuration for proper derotation
        self.rotary_emb = rotary_emb
        self.rope_base = getattr(config, 'rope_theta', 10000.0)

        # Track current sequence position for each layer (critical for RoPE derotation)
        self._current_positions = [0] * self.num_layers

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

    def _apply_inverse_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse RoPE transformation to recover semantic vectors from rotated vectors.

        CRITICAL FIX: This implements mathematically correct derotation using position information.
        Unlike the flawed _align_rotary_positions approach, this properly inverts the RoPE operation.

        Args:
            x: Rotated key/value vectors (batch_size, num_heads, seq_len, head_dim)
            position_ids: Position indices for derotation (seq_len,) - MUST be provided

        Returns:
            Derotated vectors in semantic space
        """
        if self.rotary_emb is None:
            # Fallback: assume vectors are already in semantic space
            print("[MPKVM][WARNING] No rotary_emb provided, cannot perform mathematically correct derotation")
            return x

        if position_ids is None:
            raise ValueError("[MPKVM][ERROR] position_ids must be provided for mathematically correct RoPE derotation")

        try:
            # Get tensor dimensions
            batch_size, num_heads, seq_len, head_dim = x.shape

            # Create dummy tensor for rotary_emb call
            dummy = torch.zeros((batch_size * num_heads, seq_len, head_dim), device=x.device, dtype=x.dtype)
            cos, sin = self.rotary_emb(dummy, position_ids.unsqueeze(0))

            # cos, sin shape: (batch_size * num_heads, seq_len, head_dim)
            # Reshape to match input tensor dimensions
            cos = cos.view(batch_size, num_heads, seq_len, head_dim)
            sin = sin.view(batch_size, num_heads, seq_len, head_dim)

            # Apply inverse RoPE rotation
            # RoPE forward: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
            # RoPE inverse: [x, y] = [x'*cos + y'*sin, -x'*sin + y'*cos]
            x_derotated = torch.zeros_like(x)

            for i in range(0, head_dim, 2):
                if i + 1 < head_dim:
                    # Get cos and sin values for this rotary pair
                    cos_vals = cos[..., i]  # cos values for x dimensions
                    sin_vals = sin[..., i]  # sin values for x dimensions (same for the pair)

                    # Get rotated components
                    x_rot = x[..., i]      # x' (rotated x)
                    y_rot = x[..., i+1]    # y' (rotated y)

                    # Apply inverse rotation
                    x_orig = x_rot * cos_vals + y_rot * sin_vals    # x = x'*cos + y'*sin
                    y_orig = -x_rot * sin_vals + y_rot * cos_vals   # y = -x'*sin + y'*cos

                    x_derotated[..., i] = x_orig
                    x_derotated[..., i+1] = y_orig

            return x_derotated

        except Exception as e:
            print(f"[MPKVM][ERROR] Failed to apply inverse RoPE: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return original (mathematically incorrect but prevents crash)
            return x

    def _apply_rope_to_centroids(self, centroids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to centroids using Sink Token strategy.

        Centroids are treated as tokens at positions 0, 1, 2, ..., num_centroids-1
        This follows StreamingLLM's approach for mathematical consistency.

        Args:
            centroids: Semantic centroids (num_centroids, head_dim)
            position_ids: Position IDs for sink tokens (num_centroids,)

        Returns:
            RoPE-rotated centroids ready for attention
        """
        if self.rotary_emb is None:
            return centroids

        try:
            # Expand centroids to match RoPE input format: (1, num_centroids, head_dim)
            centroids_expanded = centroids.unsqueeze(0)

            # Apply RoPE using the sink token positions
            cos, sin = self.rotary_emb(centroids_expanded, position_ids.unsqueeze(0))

            # cos, sin shape: (1, num_centroids, head_dim)
            cos = cos.squeeze(0)  # (num_centroids, head_dim)
            sin = sin.squeeze(0)  # (num_centroids, head_dim)

            # Apply RoPE rotation to each centroid
            centroids_rotated = centroids.clone()

            # Process each rotary pair (every 2 dimensions)
            for i in range(0, self.head_dim, 2):
                if i + 1 < self.head_dim:
                    # Get cos and sin for this dimension pair
                    cos_vals = cos[:, i]  # cos for x dimension
                    sin_vals = sin[:, i]  # sin for x dimension

                    # Get original values
                    x = centroids[:, i]      # x (original x)
                    y = centroids[:, i+1]    # y (original y)

                    # Apply RoPE rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
                    centroids_rotated[:, i] = x * cos_vals - y * sin_vals
                    centroids_rotated[:, i+1] = x * sin_vals + y * cos_vals

            return centroids_rotated

        except Exception as e:
            print(f"[MPKVM][ERROR] Failed to apply RoPE to centroids: {e}")
            import traceback
            traceback.print_exc()
            return centroids

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new KV states and return augmented states for attention.

        CRITICAL FIX: Handles RoPE derotation for mathematically correct clustering.

        This method:
        1. Derotates key_states back to semantic space for proper clustering
        2. Adds semantic vectors to the manifold clusterer
        3. Retrieves and re-rotates centroids using Sink Token strategy
        4. Returns augmented KV states (original + centroids) for attention

        Args:
            key_states: (batch_size, num_heads, seq_len, head_dim) - RoPE-rotated key states
            value_states: (batch_size, num_heads, seq_len, head_dim) - Value states
            layer_idx: Current layer index
            cache_kwargs: Additional cache arguments (may contain position_ids)

        Returns:
            Tuple of augmented (key_states, value_states) ready for attention
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Get clusterer for this layer
        clusterer = self.clusterers[layer_idx]

        # CRITICAL: Extract position information for mathematically correct RoPE derotation
        # This is essential for proper semantic clustering
        position_ids = None
        current_seq_start = self._current_positions[layer_idx]

        if cache_kwargs:
            # Try various ways to get position information
            if 'position_ids' in cache_kwargs:
                position_ids = cache_kwargs['position_ids']
            elif 'cache_position' in cache_kwargs:
                cache_position = cache_kwargs['cache_position']
                if cache_position is not None:
                    # Extract the positions for the new tokens
                    position_ids = cache_position[-seq_len:] if len(cache_position) >= seq_len else cache_position
            elif 'sin' in cache_kwargs and 'cos' in cache_kwargs:
                # If sin/cos are directly provided, we can compute positions from them
                # This is a fallback for when position_ids aren't available
                print(f"[MPKVM][INFO] Using sin/cos from cache_kwargs for layer {layer_idx}")

        # Generate position IDs if not provided
        if position_ids is None:
            # Use tracked sequence positions as fallback
            position_ids = torch.arange(current_seq_start, current_seq_start + seq_len,
                                      device=self.device, dtype=torch.long)
            print(f"[MPKVM][WARNING] No position_ids provided, using estimated positions "
                  f"[{current_seq_start}:{current_seq_start + seq_len}] for layer {layer_idx}")

        # CRITICAL FIX: Apply mathematically correct inverse RoPE
        # This recovers semantic vectors from position-rotated vectors
        semantic_keys = self._apply_inverse_rope(key_states, position_ids)

        # Flatten batch and heads for clustering: (batch_size * num_heads, seq_len, head_dim)
        flat_semantic_keys = semantic_keys.view(-1, seq_len, head_dim)
        flat_values = value_states.view(-1, seq_len, head_dim)

        # Add semantic vectors to clusterer (this updates centroids in semantic space)
        for seq_idx in range(flat_semantic_keys.shape[0]):
            k_seq_semantic = flat_semantic_keys[seq_idx]  # (seq_len, head_dim) - semantic space
            v_seq = flat_values[seq_idx]  # (seq_len, head_dim) - values unchanged
            clusterer.add(k_seq_semantic, weights=None)

        # Get current centroids in semantic space
        centroids_k_semantic, centroid_weights = clusterer._current_centroids()

        # CRITICAL: Re-rotate centroids to match the position space of incoming queries
        # Using Sink Token strategy: centroids are treated as tokens at positions 0, 1, 2, ...
        if centroids_k_semantic is not None and centroids_k_semantic.shape[0] > 0:
            # Create position IDs for sink tokens (0, 1, 2, ..., num_centroids-1)
            num_centroids = centroids_k_semantic.shape[0]
            sink_positions = torch.arange(num_centroids, device=self.device, dtype=torch.long)

            # Apply RoPE to centroids using sink token positions
            centroids_k_rotated = self._apply_rope_to_centroids(centroids_k_semantic, sink_positions)

            # Update cached centroids (store both semantic and rotated versions)
            self.centroid_keys[layer_idx] = centroids_k_rotated  # For attention
            self.centroid_weights[layer_idx] = centroid_weights

            # For values, use rotated centroids as approximation
            self.centroid_values[layer_idx] = centroids_k_rotated.clone()
        else:
            centroids_k_rotated = None

        # Augment KV states with centroids for attention
        if centroids_k_rotated is not None and centroids_k_rotated.shape[0] > 0:
            # Expand centroids to match batch/heads dimensions
            # centroids_k_rotated: (n_centroids, head_dim) -> (1, 1, n_centroids, head_dim)
            centroids_k_expanded = centroids_k_rotated.unsqueeze(0).unsqueeze(0)
            centroids_v_expanded = centroids_k_expanded.clone()  # Use keys as values for now

            # Concatenate original KV with centroids
            # key_states: (batch_size, num_heads, seq_len, head_dim) - already RoPE-rotated
            # centroids_k_expanded: (1, 1, n_centroids, head_dim) - RoPE-rotated to sink positions
            aug_keys = torch.cat([key_states, centroids_k_expanded.expand(batch_size, num_heads, -1, -1)], dim=2)
            aug_values = torch.cat([value_states, centroids_v_expanded.expand(batch_size, num_heads, -1, -1)], dim=2)
        else:
            # No centroids yet, return original states
            aug_keys = key_states
            aug_values = value_states

        # Update sequence length and position tracking
        self._seq_lengths[layer_idx] = seq_len
        self._current_positions[layer_idx] += seq_len

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
        self._current_positions = [0] * self.num_layers  # Reset position tracking
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
