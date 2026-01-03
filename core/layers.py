"""
Reconstructed attention utilities for MP-KVM.
Provides a helper attention implementation that can concatenate centroid tokens
into the key/value matrices so that queries can attend to both recent tokens and
abstracted centroid tokens.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import torch


def _ensure_torch():
    import torch
    return torch


def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: Optional[np.ndarray] = None, score_bias: Optional[np.ndarray] = None):
    """
    q: (tq, d)
    k: (tk, d)
    v: (tk, d_v)
    returns: (tq, d_v)
    """
    d = q.shape[-1]
    scores = np.dot(q, k.T) / np.sqrt(float(d))
    if score_bias is not None:
        sb = np.asarray(score_bias, dtype=float)
        # broadcast to (tq, tk)
        if sb.ndim == 1:
            scores = scores + sb[None, :]
        else:
            scores = scores + sb
    if mask is not None:
        # mask is expected to be broadcastable boolean where True means keep
        scores = np.where(mask, scores, -1e9)
    # stable softmax over last axis
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-12)
    return np.dot(weights, v)


def reconstruct_with_centroids(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    centroids_k: Optional[np.ndarray] = None,
    centroids_v: Optional[np.ndarray] = None,
    centroid_weighting: Optional[np.ndarray] = None,
    positionless: bool = False,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Concatenate centroid tokens to the end of K and V and perform attention.
    centroid_weighting currently not used directly here but provided for future use.
    """
    # prepare centroids (optionally convert into positionless representation)
    if centroids_k is not None and centroids_k.shape[0] > 0:
        if positionless:
            centroids_k_proc = _make_positionless_numpy(centroids_k)
        else:
            centroids_k_proc = centroids_k
        k_aug = np.vstack([k, centroids_k_proc])
        if centroids_v is not None:
            v_aug = np.vstack([v, centroids_v])
        else:
            v_aug = np.vstack([v, centroids_k_proc])
        # construct optional score bias: zeros for original keys, log(count) for centroids
        score_bias = None
        if centroid_weighting is not None:
            c_w = np.asarray(centroid_weighting, dtype=float)
            bias_centroids = np.log(c_w + 1e-12)
            score_bias = np.concatenate([np.zeros((k.shape[0],), dtype=float), bias_centroids], axis=0)
    else:
        k_aug = k
        v_aug = v
        score_bias = None

    # compute attention with optional per-key bias by manually implementing stable softmax
    d = q.shape[-1]
    scores = np.dot(q, k_aug.T) / np.sqrt(float(d))
    if score_bias is not None:
        scores = scores + score_bias[None, :]
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-12)
    return np.dot(weights, v_aug)


def _make_positionless_numpy(keys: np.ndarray) -> np.ndarray:
    """
    DEPRECATED: This function discards crucial phase information and causes semantic loss.
    Use _align_rotary_positions_numpy instead for proper RoPE handling.

    Create a 'positionless' version of key vectors by collapsing rotary pairs'
    phase information into a magnitude on the first component of each pair and
    zeroing the second. This removes position-dependent rotation (RoPE) phase.
    Operates on last dimension pairs: (0,1), (2,3), ...
    """
    print("[MPKVM][WARN] _make_positionless_numpy is deprecated. Use _align_rotary_positions_numpy for proper RoPE handling.")
    k = keys.copy().astype(np.float32)
    D = k.shape[-1]
    # vectorized across pairs for better numerical stability and speed
    # handle even and odd D gracefully (leave last dim unchanged if odd)
    even_D = D - (D % 2)
    if even_D > 0:
        x = k[..., :even_D:2]
        y = k[..., 1:even_D:2]
        r = np.sqrt(x * x + y * y)
        k[..., :even_D:2] = r
        k[..., 1:even_D:2] = 0.0
    return k


def _align_rotary_positions_numpy(vec1: np.ndarray, vec2: np.ndarray, base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two RoPE-rotated vectors to the same virtual position for similarity computation.

    This preserves phase information while allowing meaningful similarity comparison.
    Rotates vec2 to align with vec1's virtual position.

    Args:
        vec1, vec2: (D,) vectors that have been RoPE-rotated
        base: RoPE base frequency (should match model's RoPE base)

    Returns:
        Tuple of aligned vectors at the same virtual position
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have same shape for alignment")

    D = vec1.shape[-1]
    aligned_vec1 = vec1.copy().astype(np.float32)
    aligned_vec2 = vec2.copy().astype(np.float32)

    # Process each rotary pair (x,y)
    for i in range(0, D-1, 2):
        x1, y1 = vec1[i], vec1[i+1]
        x2, y2 = vec2[i], vec2[i+1]

        # Compute magnitudes and phases
        r1 = np.sqrt(x1*x1 + y1*y1)
        r2 = np.sqrt(x2*x2 + y2*y2)

        if r1 == 0 or r2 == 0:
            continue

        theta1 = np.arctan2(y1, x1)
        theta2 = np.arctan2(y2, x2)

        # Align vec2 to vec1's position by rotating vec2 by (theta1 - theta2)
        delta_theta = theta1 - theta2

        cos_dt = np.cos(delta_theta)
        sin_dt = np.sin(delta_theta)

        # Rotate vec2: (x2*cos + y2*sin, -x2*sin + y2*cos) for inverse rotation
        aligned_vec2[i] = x2 * cos_dt + y2 * sin_dt
        aligned_vec2[i+1] = -x2 * sin_dt + y2 * cos_dt

    return aligned_vec1, aligned_vec2


def compute_rotary_aligned_similarity(keys: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise similarities between RoPE-rotated vectors by aligning them to common positions.

    This preserves semantic information while allowing proper clustering.

    Args:
        keys: (N, D) array of RoPE-rotated key vectors
        metric: similarity metric ("cosine" or "euclidean")

    Returns:
        (N, N) similarity matrix
    """
    N, D = keys.shape
    similarities = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i, N):  # Only compute upper triangle
            vec1, vec2 = _align_rotary_positions_numpy(keys[i], keys[j])

            if metric == "cosine":
                # Cosine similarity on aligned vectors
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(vec1, vec2) / (norm1 * norm2)
                else:
                    sim = 0.0
            else:  # euclidean
                sim = -np.linalg.norm(vec1 - vec2)  # Negative distance for consistency

            similarities[i, j] = sim
            similarities[j, i] = sim  # Symmetric

    return similarities


class ReconstructedAttention:
    """
    Small helper wrapper that uses reconstructed attention for inference-time hooking.
    """

    def __init__(self):
        pass

    def __call__(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, centroids_k: Optional[np.ndarray] = None, centroids_v: Optional[np.ndarray] = None):
        return reconstruct_with_centroids(query, key, value, centroids_k=centroids_k, centroids_v=centroids_v)


__all__ = ["reconstruct_with_centroids", "ReconstructedAttention"]

# -----------------------
# Torch / GPU helpers
# -----------------------

def scaled_dot_product_attention_torch(q, k, v, mask: Optional[any] = None, score_bias=None):
    """
    Torch-compatible scaled dot-product attention supporting common tensor shapes.
    q,k,v expected shapes: (..., seq_q, d), (..., seq_k, d) or (batch, heads, seq, d)
    """
    torch = _ensure_torch()
    # compute scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(q.shape[-1]))
    if mask is not None:
        # mask expected to be broadcastable boolean mask where True -> keep
        scores = scores.masked_fill(~mask, float("-1e9"))
    # optional per-key bias (e.g., log-counts for centroid tokens)
    if score_bias is not None:
        # score_bias expected shape (..., seq_k) or (seq_k,)
        sb = score_bias
        if sb.dim() == 1:
            # broadcast to scores shape: (..., seq_k)
            shape = [1] * (scores.dim() - 1) + [-1]
            scores = scores + sb.view(*shape)
        else:
            scores = scores + sb
            pass
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def augment_kv_with_centroids_torch(k, v, centroids_k=None, centroids_v=None):
    """
    Concatenate centroid tokens to k and v on-device.
    Supports these k/v shapes:
      - (batch, heads, seq, d)
      - (batch, seq, d)
      - (seq, d)
    centroids_k expected shape: (C, d)
    """
    torch = _ensure_torch()
    if centroids_k is None:
        return k, v
    if centroids_k.numel() == 0:
        return k, v

    # ensure centroids on same device/dtype
    device = k.device
    centroids_k = centroids_k.to(device=device)
    if centroids_v is not None:
        centroids_v = centroids_v.to(device=device)
    else:
        centroids_v = centroids_k

    # Branch by k dimensionality
    if k.ndim == 4:
        # (B, H, S, D)
        B, H, S, D = k.shape
        C = centroids_k.shape[0]
        # expand centroids to (1,1,C,D) then repeat to (B,H,C,D)
        ck = centroids_k.view(1, 1, C, D).expand(B, H, C, D)
        cv = centroids_v.view(1, 1, C, D).expand(B, H, C, D)
        # reshape to (B, H, S+C, D)
        ck = ck.reshape(B, H, C, D)
        cv = cv.reshape(B, H, C, D)
        k_aug = torch.cat([k, ck], dim=2)
        v_aug = torch.cat([v, cv], dim=2)
        return k_aug, v_aug
    elif k.ndim == 3:
        # (B, S, D)
        B, S, D = k.shape
        C = centroids_k.shape[0]
        ck = centroids_k.view(1, C, D).expand(B, C, D)
        cv = centroids_v.view(1, C, D).expand(B, C, D)
        k_aug = torch.cat([k, ck], dim=1)
        v_aug = torch.cat([v, cv], dim=1)
        return k_aug, v_aug
    elif k.ndim == 2:
        # (S, D) treat like (1, S, D)
        S, D = k.shape
        ck = centroids_k
        cv = centroids_v
        k_aug = torch.cat([k, ck], dim=0)
        v_aug = torch.cat([v, cv], dim=0)
        return k_aug, v_aug
    else:
        # fallback: attempt to flatten last dim join
        k_flat = k.reshape(-1, k.shape[-1])
        v_flat = v.reshape(-1, v.shape[-1])
        k_aug = torch.cat([k_flat, centroids_k], dim=0)
        v_aug = torch.cat([v_flat, centroids_v], dim=0)
        return k_aug, v_aug


class ReconstructedAttentionTorch:
    """
    Torch-side reconstructed attention that can accept GPU centroids and run entirely on device.
    """

    def __init__(self):
        pass

    def __call__(self, query, key, value, centroids_k=None, centroids_v=None, mask: Optional[any] = None, centroid_weighting=None, positionless: bool = False):
        """
        query/key/value : torch tensors
        centroids_k/centroids_v: torch tensors on same device (C, D)
        """
        torch = _ensure_torch()
        # optionally make centroids positionless (undo RoPE phase) before augment
        if centroids_k is not None and positionless:
            centroids_k = _make_positionless_torch(centroids_k)
        # augment k/v
        k_aug, v_aug = augment_kv_with_centroids_torch(key, value, centroids_k=centroids_k, centroids_v=centroids_v)
        # prepare optional score bias from centroid_weighting (log-count)
        score_bias = None
        if centroid_weighting is not None and centroids_k is not None:
            cw = centroid_weighting.to(device=centroids_k.device, dtype=centroids_k.dtype)
            bias_centroids = torch.log(cw + 1e-12)
            # zeros for original key positions
            orig_len = key.shape[-2] if key.ndim >= 2 else key.shape[0]
            zeros = torch.zeros((orig_len,), dtype=bias_centroids.dtype, device=bias_centroids.device)
            score_bias = torch.cat([zeros, bias_centroids], dim=0)
        return scaled_dot_product_attention_torch(query, k_aug, v_aug, mask=mask, score_bias=score_bias)


def _make_positionless_torch(centroids: "torch.Tensor"):
    """
    DEPRECATED: This function discards crucial phase information and causes semantic loss.
    Use _align_rotary_positions_torch instead for proper RoPE handling.

    Torch equivalent of `_make_positionless_numpy`.
    Collapse rotary pair phase information into magnitude on first component of each pair.
    Vectorized implementation for GPU performance.
    """
    print("[MPKVM][WARN] _make_positionless_torch is deprecated. Use _align_rotary_positions_torch for proper RoPE handling.")
    torch = _ensure_torch()
    c = centroids.clone()
    if c.ndim != 2:
        # expect (C, D)
        c = c.view(-1, c.shape[-1])
    D = c.shape[-1]
    # Vectorized processing for GPU performance (much faster than loop)
    if D >= 2:
        # Handle even dimensions in pairs: (0,1), (2,3), ...
        even_D = D - (D % 2)
        x = c[:, 0:even_D:2]  # Even indices: 0, 2, 4, ...
        y = c[:, 1:even_D:2]  # Odd indices: 1, 3, 5, ...
        r = torch.sqrt(x * x + y * y)
        c[:, 0:even_D:2] = r
        c[:, 1:even_D:2] = torch.zeros_like(r)
    return c


def _align_rotary_positions_torch(vec1: torch.Tensor, vec2: torch.Tensor, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DEPRECATED: This function is mathematically incorrect and erases semantic differences.

    It attempts to align RoPE-rotated vectors but actually forces different semantic
    content (like "King" and "Apple") to have the same angle, making clustering meaningless.

    DO NOT USE. Use MPKVMCache._apply_inverse_rope for proper RoPE handling.
    """
    raise DeprecationWarning("_align_rotary_positions_torch is mathematically incorrect. "
                           "Use MPKVMCache._apply_inverse_rope for proper RoPE handling.")


def compute_rotary_aligned_similarity_torch(keys: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """
    DEPRECATED: Uses the mathematically incorrect alignment function.

    DO NOT USE. RoPE alignment should be handled by proper inverse rotation in MPKVMCache.
    """
    raise DeprecationWarning("compute_rotary_aligned_similarity_torch is mathematically incorrect. "
                           "Use MPKVMCache for proper RoPE-aware clustering.")

"""
Reconstructed attention layer utilities that can query MP-KVM centroids.

This module provides helper wrappers that merge centroids with current keys/values
so that attention can attend to both recent tokens and abstracted centroids.
"""



def merge_with_centroids(keys: np.ndarray, values: np.ndarray, centroids: np.ndarray, centroid_weights: np.ndarray, centroids_v: Optional[np.ndarray] = None, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge current sliding-window KV (keys: (N, D), values: (N, Dv))
    with centroids (C, D) to produce augmented KV for attention.

    Simple weighted concatenation: centroids are treated as pseudo-tokens.
    centroid_weights: (C,) used to scale centroid activations.
    centroids_v: (C, Dv) value centroids for each cluster. If None, uses global mean as fallback.
    """
    if centroids is None or centroids.shape[0] == 0:
        return keys, values

    C = centroids.shape[0]
    if top_k is not None and top_k < C:
        # pick top_k centroids by weight
        idx = np.argsort(-centroid_weights)[:top_k]
        centroids = centroids[idx]
        centroid_weights = centroid_weights[idx]
        if centroids_v is not None:
            centroids_v = centroids_v[idx]

    # Handle centroid values
    if centroids_v is not None:
        # Use provided centroid values (proper manifold partitioning)
        centroid_values = centroids_v
    elif values is not None and values.shape[0] > 0:
        # Fallback: use per-cluster weighted average if centroids_v not provided
        # This is still better than global mean - at least it's cluster-specific
        dv = values.shape[1]
        centroid_values = np.zeros((C, dv), dtype=np.float32)

        # Vectorized distance computation (much faster than nested loops)
        # centroids: (C, D), values: (N, Dv) - Note: we use centroids for distance, not values
        # Actually, we need to use keys for distance computation, not centroids directly
        # But since we don't have the original keys here, use centroids as approximation
        diff = centroids[:, None, :] - centroids[None, :, :]  # (C, C, D)
        centroid_distances = np.linalg.norm(diff, axis=2)  # (C, C)

        # ERROR: centroids_v should always be provided for proper manifold partitioning
        # Using keys as values violates the architectural separation between keys and values
        raise ValueError("[MPKVM][ERROR] centroids_v must be provided to merge_with_centroids. "
                        "Key and value vectors serve different purposes in attention mechanisms. "
                        "Please ensure your clustering implementation properly maintains separate K and V centroids.")
    else:
        # Final fallback: zeros (though this should rarely happen)
        dv = centroids.shape[1]
        centroid_values = np.zeros((centroids.shape[0], dv), dtype=np.float32)

    # scale centroids by sqrt(weight) (heuristic for attention)
    scaled_centroids = centroids * (np.sqrt(centroid_weights)[:, None] + 1e-12)

    new_keys = np.concatenate([keys, scaled_centroids], axis=0)
    new_values = np.concatenate([values, centroid_values], axis=0)
    return new_keys, new_values


__all__ = ["merge_with_centroids","reconstruct_with_centroids", "ReconstructedAttention", "ReconstructedAttentionTorch", "augment_kv_with_centroids_torch"]


