"""
Adapter utilities to attach MP-KVM to HuggingFace-style Llama attention modules.

Usage:
    from core import MPKVMManager
    from adapters.llama_adapter import attach_mpkvm_to_hf_llama

    manager = MPKVMManager(dim=hidden_size, num_layers=model.config.num_hidden_layers, cluster_kwargs={...})
    attach_mpkvm_to_hf_llama(model, manager, head_mean=False, sample_stride=1)

This file provides a conservative wrapper that:
 - wraps per-layer attention `forward` methods,
 - extracts produced key/value tensors (best-effort heuristics),
 - optionally averages across heads or subsamples token dimension to reduce CPU-GPU copies,
 - converts tensors to numpy and calls manager.add_kv(layer_idx, keys, values).
"""
from __future__ import annotations
from typing import Any, Optional, Iterable, Set
import os
import numpy as np


from core.clustering import OnlineManifoldClustering
from core.integration_clean import MPKVMManager
from core.layers import ReconstructedAttentionTorch, augment_kv_with_centroids_torch


def _to_numpy(tensor):
    import torch

    if isinstance(tensor, torch.Tensor):
        t = tensor.detach()
        # move to CPU
        t = t.cpu()
        # cast half / bfloat16 to float32 for numpy conversion
        if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
            t = t.to(dtype=torch.float32)
        return t.numpy()
    return np.asarray(tensor)


_DEBUG = os.getenv("MPKVM_DEBUG", "1") == "1"


def _process_kv_direct(manager: MPKVMManager, layer_idx: int, k_tensor, v_tensor, seq_len: int,
                      head_mean: bool = False, sample_stride: int = 1, pre_rope_key=None, head_idx: Optional[int] = None):
    """
    Process KV tensors directly (already sliced to contain only new tokens).
    This is called after pre-slicing to avoid the flattened slicing bug.
    """
    # normalize to numpy where possible but keep original tensors for possible GPU path
    kn = _to_numpy(k_tensor)
    vn = _to_numpy(v_tensor)

    # Validate tensor shapes before processing
    if kn.shape != vn.shape:
        print(f"[MPKVM][WARN] K and V shapes don't match: K{kn.shape} vs V{vn.shape}")

    # expected shapes: (batch, seq, n_heads, head_dim) or (batch, seq, head_dim) or already flattened (N, D)
    def _reshape_proc(arr):
        if arr.ndim == 4:
            # (B, S, H, D_head)
            if head_mean:
                return arr.mean(axis=2).reshape((-1, arr.shape[-1]))
            return arr.reshape((-1, arr.shape[-1]))
        elif arr.ndim == 3:
            # (B, S, D) or (B, S, H) after head_mean handled above
            return arr.reshape((-1, arr.shape[-1]))
        elif arr.ndim == 2:
            return arr
        else:
            return arr.reshape((-1, arr.shape[-1]))

    kn_proc = _reshape_proc(kn)
    vn_proc = _reshape_proc(vn)

    # CRITICAL FIX: Use Pre-RoPE Keys for clustering if available
    # This preserves semantic similarity by avoiding RoPE rotation effects
    if pre_rope_key is not None:
        # Use Pre-RoPE keys for clustering (semantic similarity preservation)
        k_clustering = _to_numpy(pre_rope_key)
        k_clustering_proc = _reshape_proc(k_clustering)

        if sample_stride is not None and sample_stride > 1:
            k_clustering_proc = k_clustering_proc[::sample_stride]

        # No need for positionless transformation - Pre-RoPE keys are already position-agnostic
        kn_proc = k_clustering_proc.astype(np.float32)
    else:
        # Fallback: Use Post-RoPE keys with positionless transformation
        if sample_stride is not None and sample_stride > 1:
            kn_proc = kn_proc[::sample_stride]

        # CRITICAL FIX: Apply positionless transformation to remove RoPE rotation
        # This ensures semantic clustering is not affected by token position
        kn_proc = _make_positionless_numpy(kn_proc)
        kn_proc = kn_proc.astype(np.float32)

    if sample_stride is not None and sample_stride > 1:
        vn_proc = vn_proc[::sample_stride]

    vn_proc = vn_proc.astype(np.float32)

    # Use direct add_kv (not incremental, since we already sliced)
    manager.add_kv(layer_idx, kn_proc, vn_proc, head_idx=head_idx)


def _process_kv_incremental(manager: MPKVMManager, layer_idx: int, k_tensor, v_tensor, seq_len: int,
                          head_mean: bool = False, sample_stride: int = 1, pre_rope_key=None, per_head_clustering: bool = False):
    """
    CRITICAL FIX: Process only incremental KV tensors since last call.

    This prevents the severe bug where full historical cache was repeatedly fed
    to the clustering operator, causing memory explosion and incorrect weighting.
    """
    # normalize to numpy where possible but keep original tensors for possible GPU path
    kn = _to_numpy(k_tensor)
    vn = _to_numpy(v_tensor)

    # Validate tensor shapes before processing
    if kn.shape != vn.shape:
        print(f"[MPKVM][WARN] K and V shapes don't match: K{kn.shape} vs V{vn.shape}")

    if per_head_clustering and kn.ndim == 4:
        # Per-head clustering: process each head separately
        B, S, H, D = kn.shape
        for h in range(H):
            # Extract data for this head
            k_head = kn[:, :, h, :]  # (B, S, D)
            v_head = vn[:, :, h, :]  # (B, S, D)
            pre_rope_h = pre_rope_key[:, :, h, :] if pre_rope_key is not None else None

            # Flatten for processing: (B*S, D)
            k_head_flat = k_head.reshape(-1, D)
            v_head_flat = v_head.reshape(-1, D)

            # Process pre-rope keys if available
            if pre_rope_h is not None:
                k_clustering = pre_rope_h.reshape(-1, D)  # (B*S, D)
                if sample_stride is not None and sample_stride > 1:
                    k_clustering = k_clustering[::sample_stride]
                # No need for positionless transformation - Pre-RoPE keys are already position-agnostic
                k_final = k_clustering.astype(np.float32)
            else:
                k_final = k_head_flat
                if sample_stride is not None and sample_stride > 1:
                    k_final = k_final[::sample_stride]
                # Apply positionless transformation to remove RoPE rotation
                k_final = _make_positionless_numpy(k_final)

            # Process values
            v_final = v_head_flat
            if sample_stride is not None and sample_stride > 1:
                v_final = v_final[::sample_stride]
            v_final = v_final.astype(np.float32)

            # Add to per-head cluster
            manager.add_kv_incremental(layer_idx, k_final, v_final, seq_len, head_idx=h)
    else:
        # Standard processing (all heads together or per-layer)
        # expected shapes: (batch, seq, n_heads, head_dim) or (batch, seq, head_dim) or already flattened (N, D)
        def _reshape_proc(arr):
            if arr.ndim == 4:
                # (B, S, H, D_head)
                if head_mean:
                    return arr.mean(axis=2).reshape((-1, arr.shape[-1]))
                return arr.reshape((-1, arr.shape[-1]))
            elif arr.ndim == 3:
                # (B, S, D) or (B, S, H) after head_mean handled above
                return arr.reshape((-1, arr.shape[-1]))
            elif arr.ndim == 2:
                return arr
            else:
                return arr.reshape((-1, arr.shape[-1]))

        kn_proc = _reshape_proc(kn)
        vn_proc = _reshape_proc(vn)

        # CRITICAL FIX: Use Pre-RoPE Keys for clustering if available
        # This preserves semantic similarity by avoiding RoPE rotation effects
        if pre_rope_key is not None:
            # Use Pre-RoPE keys for clustering (semantic similarity preservation)
            k_clustering = _to_numpy(pre_rope_key)
            k_clustering_proc = _reshape_proc(k_clustering)

            if sample_stride is not None and sample_stride > 1:
                k_clustering_proc = k_clustering_proc[::sample_stride]

            # No need for positionless transformation - Pre-RoPE keys are already position-agnostic
            kn_proc = k_clustering_proc.astype(np.float32)
        else:
            # Fallback: Use Post-RoPE keys with positionless transformation
            if sample_stride is not None and sample_stride > 1:
                kn_proc = kn_proc[::sample_stride]

            # This ensures semantic clustering is not affected by token position
            kn_proc = _make_positionless_numpy(kn_proc)
            kn_proc = kn_proc.astype(np.float32)

        if sample_stride is not None and sample_stride > 1:
            vn_proc = vn_proc[::sample_stride]

        vn_proc = vn_proc.astype(np.float32)

        # Use incremental addition to manager
        manager.add_kv_incremental(layer_idx, kn_proc, vn_proc, seq_len)


def _extract_pre_rope_key(hidden_states, attn_module, head_mean: bool = False):
    """
    Extract Pre-RoPE Key vectors from hidden states for semantic clustering.
    This bypasses RoPE rotation to preserve semantic similarity.
    """
    import torch

    # Get the projection layer
    if not hasattr(attn_module, "k_proj"):
        return None

    # Project hidden states to key space (Pre-RoPE)
    k_pre_rope = attn_module.k_proj(hidden_states)

    # Apply head reshaping if needed
    if k_pre_rope.ndim == 3 and head_mean:
        # (B, S, D) -> average across heads dimension if it exists
        # For models like Llama, k_proj output is already (B, S, head_dim)
        pass  # No head averaging needed for pre-RoPE keys
    elif k_pre_rope.ndim == 4:
        # (B, H, S, D_head) -> handle head dimension
        if head_mean:
            k_pre_rope = k_pre_rope.mean(dim=1)  # (B, S, D_head)
        else:
            k_pre_rope = k_pre_rope.transpose(1, 2).reshape(k_pre_rope.shape[0], -1, k_pre_rope.shape[-1])

    return k_pre_rope


def _process_kv_and_add(manager: MPKVMManager, layer_idx: int, k_tensor, v_tensor, head_mean: bool = False, sample_stride: int = 1, pre_rope_key=None):
    # normalize to numpy where possible but keep original tensors for possible GPU path
    kn = _to_numpy(k_tensor)
    vn = _to_numpy(v_tensor)

    # Validate tensor shapes before processing
    if kn.shape != vn.shape:
        print(f"[MPKVM][WARN] K and V shapes don't match: K{kn.shape} vs V{vn.shape}")

    # expected shapes: (batch, seq, n_heads, head_dim) or (batch, seq, head_dim) or already flattened (N, D)
    def _reshape_proc(arr):
        if arr.ndim == 4:
            # (B, S, H, D_head)
            if head_mean:
                return arr.mean(axis=2).reshape((-1, arr.shape[-1]))
            return arr.reshape((-1, arr.shape[-1]))
        elif arr.ndim == 3:
            # (B, S, D) or (B, S, H) after head_mean handled above
            return arr.reshape((-1, arr.shape[-1]))
        elif arr.ndim == 2:
            return arr
        else:
            return arr.reshape((-1, arr.shape[-1]))

    kn_proc = _reshape_proc(kn)
    vn_proc = _reshape_proc(vn)

    # CRITICAL FIX: Use Pre-RoPE Keys for clustering if available
    # This preserves semantic similarity by avoiding RoPE rotation effects
    if pre_rope_key is not None:
        # Use Pre-RoPE keys for clustering (semantic similarity preservation)
        k_clustering = _to_numpy(pre_rope_key)
        k_clustering_proc = _reshape_proc(k_clustering)

        if sample_stride is not None and sample_stride > 1:
            k_clustering_proc = k_clustering_proc[::sample_stride]

        # No need for positionless transformation - Pre-RoPE keys are already position-agnostic
        kn_proc = k_clustering_proc.astype(np.float32)
    else:
        # Fallback: Use Post-RoPE keys with positionless transformation
        if sample_stride is not None and sample_stride > 1:
            kn_proc = kn_proc[::sample_stride]

        # CRITICAL FIX: Apply positionless transformation to remove RoPE rotation
        # This ensures semantic clustering is not affected by token position
        kn_proc = _make_positionless_numpy(kn_proc)
        kn_proc = kn_proc.astype(np.float32)

    if sample_stride is not None and sample_stride > 1:
        vn_proc = vn_proc[::sample_stride]

    vn_proc = vn_proc.astype(np.float32)

    # prefer GPU aggregator if available and original tensors look like torch tensors
    if hasattr(manager, "add_kv_torch"):
        # pass through original tensors first to avoid CPU<->GPU copies
        manager.add_kv_torch(layer_idx, k_tensor, v_tensor)
        if _DEBUG:
            print(f"[MPKVM][layer {layer_idx}] add_kv_torch succeeded")
        return

    # CPU path
    if _DEBUG:
        print(f"[MPKVM][layer {layer_idx}] add_kv (CPU) called with shapes k_proc={kn_proc.shape} v_proc={vn_proc.shape}")
    manager.add_kv(layer_idx, kn_proc, vn_proc)


def attach_mpkvm_to_hf_llama(
    model: Any,
    manager: MPKVMManager,
    head_mean: bool = False,
    sample_stride: int = 1,
    enable_injection: Optional[bool] = None,
    max_injected_centroids: int = 256,
    per_layer_injection: Optional[Iterable[int]] = None,
    pass_centroid_weighting: bool = True,
    per_head_clustering: bool = False,
    positionless_injection: bool = True,  # Default to True since we now use Pre-RoPE centroids
    sliding_window_size: Optional[int] = None,
    cluster_kwargs: Optional[dict] = None,
    strict_mode: bool = False
):
    """
    Attach MP-KVM to HuggingFace Llama model with incremental KV processing.

    CRITICAL FIX: This adapter now tracks previously processed sequence lengths
    per layer to ensure only NEW tokens are sent to the clustering operator.
    This prevents the severe bug of repeatedly feeding full historical cache.
    """
    """
    Walk the model and attach wrappers to attention modules.
    This is heuristic and tries common HF Llama model layouts.

    Args:
        sliding_window_size: If set, enables true compression by maintaining only the most
            recent N tokens plus centroids. This prevents linear memory growth during generation.
            When enabled, cache size stays constant at sliding_window_size + num_centroids.
            Recommended: 512-2048 for long context tasks.
    """
    # resolve env vars / defaults
    if enable_injection is None:
        env_val = os.getenv("MPKVM_ENABLE_INJECTION", "1")
        enable_injection = False if env_val in ("0", "false", "False") else True
    max_injected_centroids = int(os.getenv("MPKVM_MAX_INJECTED_CENTROIDS", str(max_injected_centroids)))
    per_layer_set: Optional[Set[int]] = None
    if per_layer_injection is not None:
        per_layer_set = set(int(x) for x in per_layer_injection)
    else:
        # env var like "0,1,2"
        env_layers = os.getenv("MPKVM_PER_LAYER_INJECTION", None)
        if env_layers:
            per_layer_set = set(int(x.strip()) for x in env_layers.split(",") if x.strip() != "")
    # locate the module that holds transformer layers
    candidates = []
    if hasattr(model, "model"):
        candidates.append(model.model)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    # fallback to model itself
    candidates.append(model)

    layers_container = None
    for cand in candidates:
        if hasattr(cand, "layers"):
            layers_container = cand
            break
        if hasattr(cand, "decoder") and hasattr(cand.decoder, "layers"):
            layers_container = cand.decoder
            break

    if layers_container is None:
        raise RuntimeError("Could not find transformer layers container in provided model.")

    # iterate layers and attach wrapper to common attention attribute names
    for idx, layer in enumerate(getattr(layers_container, "layers")):
        # common attr names
        attn_attr_names = ["self_attn", "attention", "attn", "qkv"]
        attn_module = None
        for name in attn_attr_names:
            if hasattr(layer, name):
                attn_module = getattr(layer, name)
                break
        if attn_module is None:
            # sometimes the attention module is nested; search attributes
            for attr_name in dir(layer):
                attr = getattr(layer, attr_name)
                if hasattr(attr, "forward") and "attn" in attr_name.lower():
                    attn_module = attr
                    break
        if attn_module is None:
            # skip if not found
            continue
        # debug: print module summary once (not per-forward) to avoid noise
        if _DEBUG:
            attrs = [a for a in dir(attn_module) if "key" in a.lower() or "value" in a.lower() or "proj" in a.lower() or "attn" in a.lower() or "present" in a.lower() or "last" in a.lower()]
            print(f"[MPKVM][layer {idx}] attaching to {type(attn_module).__name__}, candidate attrs: {attrs}")

        # wrap forward with more robust KV extraction heuristics
        orig_forward = getattr(attn_module, "forward")

        def make_wrapped(orig_forward, attn_module, layer_idx, enable_injection=enable_injection, max_injected_centroids=max_injected_centroids, per_layer_set=per_layer_set, pass_centroid_weighting=pass_centroid_weighting, positionless_injection=positionless_injection, sliding_window_size=sliding_window_size):
            def _compute_and_record_attention(qf, kf, layer_idx, debug_label="attention"):
                """
                Private helper function to compute attention weights and record them.
                Handles GQA reshaping and shape compatibility checks.
                """
                import torch
                if qf is None or kf is None:
                    return

                # Handle potential shape mismatches (GQA, etc.)
                if qf.shape[0] == kf.shape[0] and qf.shape[-1] == kf.shape[-1]:
                    # perfectly aligned already
                    pass
                else:
                    # attempt reshaping qf into (Bn, Sq, H, Dh) as previous logic
                    Bn, Sq, Dq = qf.shape
                    Dh = int(kf.shape[-1])

                    # More robust GQA detection: check if dimensions are compatible
                    # GQA pattern: q_head_dim should be multiple of k_head_dim
                    handled = False
                    if kf.shape[-1] > 0 and qf.shape[-1] > kf.shape[-1] and (qf.shape[-1] % kf.shape[-1]) == 0:
                        # Calculate number of query heads per key/value head
                        H = int(qf.shape[-1] // kf.shape[-1])
                        # Additional validation: H should be reasonable (typically 1-32)
                        if 1 <= H <= 32:
                            handled = True
                        else:
                            if _DEBUG:
                                print(f"[MPKVM][layer {layer_idx}] suspicious head ratio H={H}, skipping GQA reshape")
                    elif kf.shape[-1] > 0 and qf.shape[-1] == kf.shape[-1]:
                        # No GQA, dimensions already match
                        H = 1
                        handled = True
                    if handled:
                        qf = qf.reshape(Bn, Sq, H, Dh).permute(0, 2, 1, 3).reshape(-1, Sq, Dh)
                    else:
                        # incompatible shapes; skip recording to avoid expensive errors
                        if _DEBUG:
                            print(f"[MPKVM][layer {layer_idx}] incompatible q/k head dims q={qf.shape} k={kf.shape}")
                        return

                # Now compute attention scores and weights
                dq = float(qf.shape[-1])
                scores = torch.matmul(qf, kf.transpose(-2, -1)) / (dq ** 0.5)
                weights = torch.softmax(scores, dim=-1)

                # record as numpy (best-effort with guarded fallback)
                manager.record_attention(layer_idx, weights.detach().cpu().to(torch.float32).numpy())
                if _DEBUG:
                    print(f"[MPKVM][layer {layer_idx}] {debug_label} recorded attention weights shape={getattr(weights,'shape',None)}")

            def _flatten_for_attn(t):
                """
                Unified helper function to flatten tensors for attention computation.
                Handles various tensor shapes: (B,H,S,D) -> (B*H,S,D), etc.
                """
                if t.ndim == 4:
                    # (B, H, S, D) -> (B*H, S, D)
                    return t.reshape(-1, t.shape[-2], t.shape[-1])
                if t.ndim == 3:
                    # (B, S, D) -> (B, S, D)
                    return t
                if t.ndim == 2:
                    # (S, D) or (N, D) -> (1, S, D) or (N, S, D)
                    return t.unsqueeze(0)
                # fallback: try to reshape to (..., S, D)
                return t.reshape(-1, t.shape[-2], t.shape[-1])

            def wrapped(*args, **kwargs):
                outputs = orig_forward(*args, **kwargs)

                # 1) Try module attributes (common names)
                candidates = ["last_key", "last_value", "key", "value", "present_key", "present_value", "present_key_value_states", "present"]
                k = None
                v = None
                for name in candidates:
                    attr = getattr(attn_module, name, None)
                    if attr is not None:
                            # some attrs may be tuples/lists (k,v)
                            if isinstance(attr, (list, tuple)) and len(attr) >= 2:
                                k, v = attr[0], attr[1]
                                break
                            # else try to infer by name
                            if "key" in name and k is None:
                                k = attr
                            if "value" in name and v is None:
                                v = attr
                            if k is not None and v is not None:
                                break

                # 2) Try outputs (many HF models return present_key_value_states or tuple)
                if (k is None or v is None) and isinstance(outputs, tuple) and len(outputs) > 1:
                    pv = outputs[1]
                    # present might be (k, v) or list of layers; handle several shapes
                    if isinstance(pv, (list, tuple)) and len(pv) >= 2:
                        # pv could be ((k_layer,...),(v_layer,...)) or (k,v)
                        first = pv[0]
                        second = pv[1]
                        # if first/second are tensors or arrays, assign
                        k = k or first
                        v = v or second
                    elif hasattr(pv, "present_key_values") or hasattr(pv, "past_key_values"):
                        # model-specific container; attempt attribute access
                        p = getattr(pv, "present_key_values", None) or getattr(pv, "past_key_values", None)
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            k = k or p[0]
                            v = v or p[1]

                # 3) Try kwargs like past_key_values
                if (k is None or v is None) and "past_key_values" in kwargs:
                    p = kwargs.get("past_key_values", None)
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        k = k or p[0]
                        v = v or p[1]

                # 4) Conservative probe: use projection layers if exposed
                if (k is None or v is None):
                    query = None
                    if len(args) > 0:
                        query = args[0]
                    elif "hidden_states" in kwargs:
                        query = kwargs["hidden_states"]
                    if query is not None:
                        if hasattr(attn_module, "k_proj") and hasattr(attn_module, "v_proj"):
                            k_try = getattr(attn_module, "k_proj")(query)
                            v_try = getattr(attn_module, "v_proj")(query)
                            if k_try is not None and v_try is not None:
                                k = k or k_try
                                v = v or v_try
                        if (k is None or v is None) and hasattr(attn_module, "q_proj") and hasattr(attn_module, "k_proj") and hasattr(attn_module, "v_proj"):
                            q_try = attn_module.q_proj(query)
                            k_try = attn_module.k_proj(query)
                            v_try = attn_module.v_proj(query)
                            k = k or k_try
                            v = v or v_try

                # send KV to manager (prefer GPU path if available)
                if k is not None and v is not None:
                    # Validate KV tensor shapes - CRITICAL for MP-KVM correctness
                    k_shape = getattr(k, "shape", None)
                    v_shape = getattr(v, "shape", None)

                    if k_shape is not None and v_shape is not None:
                        # STRICT validation - no compromises for core innovation
                        if len(k_shape) != len(v_shape):
                            raise ValueError(f"[MPKVM][ERROR] K and V have incompatible dimensions: K{k_shape} vs V{v_shape}. "
                                           f"MP-KVM requires proper KV tensor shapes for manifold partitioning.")
                        if len(k_shape) >= 2 and len(v_shape) >= 2:
                            # Last dimension must match (head_dim) - no exceptions
                            if k_shape[-1] != v_shape[-1]:
                                raise ValueError(f"[MPKVM][ERROR] K and V have mismatched head dimensions: K{k_shape} vs V{v_shape}. "
                                               f"Cannot perform manifold partitioning with incompatible dimensions.")

                    if _DEBUG:
                        print(f"[MPKVM][layer {layer_idx}] extracted k type={type(k).__name__ if k is not None else None} shape={k_shape}  v type={type(v).__name__ if v is not None else None} shape={v_shape}")

                    # CRITICAL: Process KV tensors - this MUST succeed for MP-KVM to work
                    if k is None or v is None:
                        raise RuntimeError(f"[MPKVM][ERROR] Failed to extract both K and V tensors for layer {layer_idx}. "
                                         f"MP-KVM cannot function without proper KV tensor extraction. "
                                         f"This indicates the adapter is not compatible with the current model architecture.")

                    # CRITICAL FIX: Extract Pre-RoPE Keys for semantic clustering
                    # This preserves semantic similarity by avoiding RoPE rotation effects
                    pre_rope_key = None
                    if hasattr(attn_module, "k_proj"):
                        # Try to get hidden states from inputs
                        hidden_states = None
                        if len(args) > 0 and isinstance(args[0], torch.Tensor):
                            hidden_states = args[0]
                        elif "hidden_states" in kwargs:
                            hidden_states = kwargs["hidden_states"]

                        if hidden_states is not None:
                            try:
                                pre_rope_key = _extract_pre_rope_key(hidden_states, attn_module, head_mean=head_mean)
                                if _DEBUG and pre_rope_key is not None:
                                    print(f"[MPKVM][layer {layer_idx}] Extracted Pre-RoPE keys for semantic clustering")
                            except Exception as e:
                                if _DEBUG:
                                    print(f"[MPKVM][layer {layer_idx}] Failed to extract Pre-RoPE keys: {e}")

                    # CRITICAL FIX: Prioritize GPU path over CPU incremental path
                    # The GPU aggregator provides better performance but was unreachable due to
                    # the incremental path always being taken first
                    if hasattr(manager, 'add_kv_torch') and hasattr(k, 'device') and hasattr(v, 'device'):
                        # GPU path available - use it directly with full tensors
                        # GPU aggregator handles incremental logic internally
                        import torch
                        if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                            manager.add_kv_torch(layer_idx, k, v)
                            if _DEBUG:
                                print(f"[MPKVM][GPU] Layer {layer_idx}: Used GPU aggregator for {k.shape}")
                        else:
                            # Fallback to CPU processing if tensors are not torch
                            _process_kv_and_add(manager, layer_idx, k, v, head_mean=head_mean, sample_stride=sample_stride, pre_rope_key=pre_rope_key)
                    elif hasattr(manager, 'add_kv_incremental'):
                        # CPU incremental path
                        # Try to get current sequence length from various sources
                        seq_len = None
                        if hasattr(k, 'shape'):
                            seq_len = k.shape[-2] if k.ndim >= 2 else k.shape[0]
                        elif hasattr(k, '__len__'):
                            seq_len = len(k)

                        if seq_len is not None:
                            # CRITICAL FIX: Extract only NEW tokens before flattening to avoid index bug
                            # The bug was: flattening [B,S,H,D] -> [B*S*H,D] then slicing [start:end]
                            # would only get a tiny fraction of new tokens

                            # Get previously processed length for this layer (and head if per-head)
                            length_key = (layer_idx, None) if per_head_clustering and k.ndim == 4 else layer_idx
                            prev_len = getattr(manager, '_processed_lengths', {}).get(length_key, 0)

                            # Calculate how many new tokens to process
                            new_tokens = seq_len - prev_len
                            if new_tokens > 0:
                                # Extract only the new tokens from the end of each sequence
                                # Handle different tensor shapes: [B,H,S,D] or [B,S,H,D] or [S,D]
                                if k.ndim == 4:
                                    # Shape: [B, H/kv, S, D] or [B, S, H, D]
                                    # Assume last dimension is S (sequence), extract last new_tokens
                                    k_new = k[..., -new_tokens:, :]  # Keep all dims except last, take last new_tokens
                                    v_new = v[..., -new_tokens:, :]
                                    pre_rope_new = pre_rope_key[..., -new_tokens:, :] if pre_rope_key is not None else None
                                elif k.ndim == 3:
                                    # Shape: [B, S, D] - already flattened heads
                                    k_new = k[:, -new_tokens:, :]  # Take last new_tokens sequences
                                    v_new = v[:, -new_tokens:, :]
                                    pre_rope_new = pre_rope_key[:, -new_tokens:, :] if pre_rope_key is not None else None
                                elif k.ndim == 2:
                                    # Shape: [S, D] - single sequence
                                    k_new = k[-new_tokens:, :]  # Take last new_tokens tokens
                                    v_new = v[-new_tokens:, :]
                                    pre_rope_new = pre_rope_key[-new_tokens:, :] if pre_rope_key is not None else None
                                else:
                                    # Fallback: assume sequence is last dimension
                                    k_new = k[..., -new_tokens:, :]
                                    v_new = v[..., -new_tokens:, :]
                                    pre_rope_new = pre_rope_key[..., -new_tokens:, :] if pre_rope_key is not None else None

                                # Now process the extracted new tokens
                                if per_head_clustering and k_new.ndim == 4:
                                    # Per-head processing: process each head separately
                                    B = k_new.shape[0]
                                    # Determine which dimension is heads
                                    if k_new.shape[1] == k.shape[1] and k.shape[1] != B:  # Same as original, not batch
                                        H = k_new.shape[1]  # [B, H, S, D]
                                        head_dim = 1
                                    else:
                                        H = k_new.shape[2] if k_new.shape[1] == B else k_new.shape[1]  # [B, S, H, D] or similar
                                        head_dim = 2 if k_new.shape[1] == B else 1

                                    for h in range(H):
                                        if head_dim == 1:
                                            k_head = k_new[:, h:h+1, :, :]  # [B,1,S_new,D]
                                            v_head = v_new[:, h:h+1, :, :]
                                            pre_rope_h = pre_rope_new[:, h:h+1, :, :] if pre_rope_new is not None else None
                                        else:  # head_dim == 2
                                            k_head = k_new[:, :, h:h+1, :]  # [B,S_new,1,D]
                                            v_head = v_new[:, :, h:h+1, :]
                                            pre_rope_h = pre_rope_new[:, :, h:h+1, :] if pre_rope_new is not None else None

                                        _process_kv_direct(manager, layer_idx, k_head, v_head, new_tokens,
                                                         head_mean=head_mean, sample_stride=sample_stride,
                                                         pre_rope_key=pre_rope_h, head_idx=h)
                                else:
                                    # Standard processing (all heads together or per-layer)
                                    _process_kv_direct(manager, layer_idx, k_new, v_new, new_tokens,
                                                     head_mean=head_mean, sample_stride=sample_stride, pre_rope_key=pre_rope_new)

                                # Update the processed length
                                if hasattr(manager, '_processed_lengths'):
                                    manager._processed_lengths[length_key] = seq_len
                            # If no new tokens, skip processing
                        else:
                            # Fallback to original processing if seq_len cannot be determined
                            _process_kv_and_add(manager, layer_idx, k, v, head_mean=head_mean, sample_stride=sample_stride, pre_rope_key=pre_rope_key)
                    else:
                        # Fallback for managers without incremental support
                        _process_kv_and_add(manager, layer_idx, k, v, head_mean=head_mean, sample_stride=sample_stride, pre_rope_key=pre_rope_key)
                else:
                    raise RuntimeError(f"[MPKVM][ERROR] Layer {layer_idx}: Could not extract both K and V tensors. "
                                     f"MP-KVM requires successful KV extraction to perform manifold partitioning. "
                                     f"This is a critical failure - check model compatibility.")
                    # The following code would only execute if we had fallback mechanisms, but MP-KVM should fail hard
                    import torch
                    q_tensor = None
                    # prefer explicit query variable if present
                    if 'query' in locals() and isinstance(query, torch.Tensor):
                        q_tensor = query
                    # fallback to last_query attr
                    if q_tensor is None:
                        q_tensor = getattr(attn_module, "last_query", None)
                    # if still None, try to extract from args/kwargs (hidden_states)
                    if q_tensor is None:
                        if len(args) > 0 and isinstance(args[0], torch.Tensor):
                            q_tensor = args[0]
                        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                            q_tensor = kwargs["hidden_states"]
                    # if q_proj exists, try projecting a query to get proper shape
                    if q_tensor is not None and not hasattr(q_tensor, "shape") and hasattr(attn_module, "q_proj"):
                        q_tensor = attn_module.q_proj(q_tensor)
                    # ensure q and k are torch tensors and compute attention weights
                    if q_tensor is not None and isinstance(q_tensor, torch.Tensor) and isinstance(k, torch.Tensor):
                            # Normalize q/k shapes to (N, S, D) where N can be B*H if needed.
                            # Before flattening, handle GQA: if q_tensor and k have differing head counts,
                            # expand k along head dim so flattened batch dims align.
                            if hasattr(q_tensor, "ndim") and hasattr(k, "ndim") and q_tensor.ndim == 4 and k.ndim == 4:
                                # shapes: (B, Hq, S, Dq) and (B, Hk, S, Dk)
                                Hq = int(q_tensor.shape[1])
                                Hk = int(k.shape[1])
                                if Hq != Hk:
                                    if Hq % Hk == 0:
                                        ratio = Hq // Hk
                                        k = k.repeat_interleave(ratio, dim=1)
                                        if _DEBUG:
                                            print(f"[MPKVM][layer {layer_idx}] expanded k via repeat_interleave: Hk->{Hk*ratio}")
                                    else:
                                        if _DEBUG:
                                            print(f"[MPKVM][layer {layer_idx}] GQA head mismatch not divisible: Hq={Hq} Hk={Hk}")

                            qf = _flatten_for_attn(q_tensor)
                            kf = _flatten_for_attn(k)

                            if qf is None or kf is None:
                                raise RuntimeError("could not normalize q/k tensors for attention recording")

                            # Use the unified attention recording function (handles GQA automatically)
                            _compute_and_record_attention(qf, kf, layer_idx, "recorded")

                    if _DEBUG:
                        print(f"[MPKVM][layer {layer_idx}] _process_kv_and_add succeeded")

                # Forced-projection fallback: if attention weights were not recorded above,
                # try to compute q and k via projection layers (q_proj/k_proj) from hidden_states
                # and record the resulting attention. This helps when module attributes like
                # last_query/last_key are not populated.
                    if _DEBUG:
                        print(f"[MPKVM][layer {layer_idx}] attempting forced projection fallback for attention recording")
                    import torch
                    # obtain a candidate query source
                    query_src = None
                    if len(args) > 0:
                        query_src = args[0]
                    elif "hidden_states" in kwargs:
                        query_src = kwargs["hidden_states"]

                    q_proj = getattr(attn_module, "q_proj", None)
                    k_proj = getattr(attn_module, "k_proj", None)

                    qf = None
                    kf = None
                    if query_src is not None:
                            qf = None
                            kf = None

                    # fallback to existing tensors if projection didn't yield tensors
                    if kf is None and isinstance(k, torch.Tensor):
                        kf = k
                    if qf is None:
                        qf = getattr(attn_module, "last_query", None)

                    if isinstance(qf, torch.Tensor) and isinstance(kf, torch.Tensor):
                        qff = _flatten_for_attn(qf)
                        kff = _flatten_for_attn(kf)
                        # Use the unified attention recording function
                        _compute_and_record_attention(qff, kff, layer_idx, "forced-proj")

                # Forced projection fallback: if previous recording attempts failed, try to
                # compute q/k via q_proj/k_proj from hidden_states (safe, guarded) and record.
                    if _DEBUG:
                        print(f"[MPKVM][layer {layer_idx}] attempting forced projection fallback for attention recording")
                    import torch

                    # attempt to locate a query source
                    query_src = None
                    if len(args) > 0:
                        query_src = args[0]
                    elif "hidden_states" in kwargs:
                        query_src = kwargs["hidden_states"]

                    qf = None
                    kf = None
                    # If projection modules exist, try to apply them to query_src
                    q_proj = getattr(attn_module, "q_proj", None)
                    k_proj = getattr(attn_module, "k_proj", None)
                    if query_src is not None:
                        qf = q_proj(query_src) if q_proj is not None else None
                        kf = k_proj(query_src) if k_proj is not None else None

                    # fallback to available tensors
                    if kf is None and isinstance(k, torch.Tensor):
                        kf = k
                    # q_tensor may have been set above; try to reuse
                    q_tensor_local = None
                    if qf is None and isinstance(q_tensor_local, torch.Tensor):
                        qf = q_tensor_local

                    # compute attention if we have tensors
                    if isinstance(qf, torch.Tensor) and isinstance(kf, torch.Tensor):
                        qff = _flatten_for_attn(qf)
                        kff = _flatten_for_attn(kf)
                        # Use the unified attention recording function
                        _compute_and_record_attention(qff, kff, layer_idx, "forced-proj")

                # Attempt GPU-side centroid injection if enabled and allowed for this layer
                if enable_injection and (per_layer_set is None or layer_idx in per_layer_set):
                    try:
                        # Get centroids for this layer
                        if per_head_clustering and hasattr(k, 'shape') and k.ndim == 4:
                            # Per-head clustering: collect centroids from all heads and merge
                            _, _, H, _ = k.shape
                            all_centroids_k = []
                            all_centroids_v = []
                            all_counts = []
                            all_weights = []

                            for h in range(H):
                                ck, cv, cc, cw = manager.get_layer_centroids(layer_idx, head_idx=h)
                                if ck.shape[0] > 0:
                                    all_centroids_k.append(ck)
                                    all_centroids_v.append(cv)
                                    all_counts.append(cc)
                                    all_weights.append(cw)

                            if all_centroids_k:
                                # Merge centroids from all heads
                                centroids_k_np = np.concatenate(all_centroids_k, axis=0)
                                centroids_v_np = np.concatenate(all_centroids_v, axis=0)
                                centroid_counts_np = np.concatenate(all_counts, axis=0)
                                centroid_weights_np = np.concatenate(all_weights, axis=0)
                            else:
                                centroids_k_np = np.zeros((0, manager.dim), dtype=np.float32)
                                centroids_v_np = np.zeros((0, manager.dim), dtype=np.float32)
                                centroid_counts_np = np.array([], dtype=int)
                                centroid_weights_np = np.array([], dtype=float)
                        else:
                            # Standard per-layer clustering
                            centroids_k_np, centroids_v_np, centroid_counts_np, centroid_weights_np = manager.get_layer_centroids(layer_idx)

                        if centroids_k_np.shape[0] > 0 and centroids_k_np.shape[0] <= max_injected_centroids:
                            import torch

                            # Convert centroids to torch tensors on the same device as k/v
                            device = k.device if hasattr(k, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            centroids_k = torch.from_numpy(centroids_k_np).to(device=device, dtype=k.dtype)
                            centroids_v = torch.from_numpy(centroids_v_np).to(device=device, dtype=v.dtype)

                            # Handle RoPE positioning for centroids (critical for injection mode)
                            # Centroids are aggregates of multiple tokens and don't have a single position
                            # Option 1: positionless injection (current default) - centroids act as global memory
                            # Option 2: positioned injection - centroids get position encodings (experimental)
                            if positionless_injection:
                                centroids_k = _make_positionless_torch(centroids_k)

                            # Augment current KV with centroids
                            k_aug, v_aug = augment_kv_with_centroids_torch(k, v, centroids_k=centroids_k, centroids_v=centroids_v)

                            # CRITICAL: Modify function outputs to inject centroids into the actual KV cache
                            # This is the key to making injection work in generation mode
                            if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                                outputs_list = list(outputs)
                                present_kv = outputs_list[1]  # Usually index 1 contains the KV cache

                                # Handle different cache formats
                                if isinstance(present_kv, (list, tuple)) and len(present_kv) > layer_idx:
                                    # Legacy tuple format: (key_layer_0, value_layer_0, key_layer_1, ...)
                                    # or per-layer tuple: ((key_0, value_0), (key_1, value_1), ...)
                                    if isinstance(present_kv[layer_idx], (list, tuple)) and len(present_kv[layer_idx]) >= 2:
                                        # Per-layer tuple format
                                        outputs_list[1] = list(outputs_list[1]) if isinstance(outputs_list[1], tuple) else outputs_list[1]
                                        outputs_list[1][layer_idx] = (k_aug, v_aug)
                                    else:
                                        # Legacy flat format (alternating keys and values)
                                        kv_idx = 2 * layer_idx
                                        if kv_idx + 1 < len(present_kv):
                                            outputs_list[1] = list(outputs_list[1]) if isinstance(outputs_list[1], tuple) else outputs_list[1]
                                            outputs_list[1][kv_idx] = k_aug      # key at even indices
                                            outputs_list[1][kv_idx + 1] = v_aug  # value at odd indices

                                elif hasattr(present_kv, "key_cache") and hasattr(present_kv, "value_cache"):
                                    # Modern DynamicCache format (transformers >= 4.36)
                                    if layer_idx < len(present_kv.key_cache):
                                        current_seq_len = present_kv.key_cache[layer_idx].shape[-2]

                                        if sliding_window_size is not None and sliding_window_size > 0:
                                            # TRUE COMPRESSION: Sliding Window + Centroids strategy
                                            # Keep recent sliding_window_size tokens + centroids
                                            # Replace earlier tokens with centroids to prevent linear growth

                                            # Step 1: Determine how many original tokens to keep
                                            keep_tokens = min(sliding_window_size, current_seq_len)

                                            # Step 2: StreamingLLM-style compression - centroids as sink tokens at the beginning
                                            if centroids_k.shape[0] > 0:
                                                # Determine cache layout: [sink_tokens|sliding_window]
                                                num_sink = centroids_k.shape[0]
                                                max_window = sliding_window_size - num_sink  # Reserve space for sink tokens

                                                if current_seq_len <= max_window:
                                                    # Not enough tokens yet, just accumulate
                                                    # Keep all tokens as sliding window
                                                    window_tokens = min(current_seq_len, max_window)
                                                    if current_seq_len > window_tokens:
                                                        present_kv.key_cache[layer_idx] = present_kv.key_cache[layer_idx][:, :, -window_tokens:, :]
                                                        present_kv.value_cache[layer_idx] = present_kv.value_cache[layer_idx][:, :, -window_tokens:, :]
                                                else:
                                                    # Enough tokens: construct [centroids|recent_window]
                                                    window_tokens = max_window

                                                    # Extract recent window tokens
                                                    recent_k = present_kv.key_cache[layer_idx][:, :, -window_tokens:, :]
                                                    recent_v = present_kv.value_cache[layer_idx][:, :, -window_tokens:, :]

                                                    # Construct new cache: [centroids|recent_window]
                                                    # Ensure centroids have correct dimensions for GQA compatibility
                                                    cache_shape = recent_k.shape  # (B, H_kv, S_window, D)
                                                    B, H_kv, S_window, D = cache_shape

                                                    # Expand centroids to match KV cache dimensions: (B, H_kv, C, D)
                                                    # In GQA models, H_kv may be different from H_q, but KV cache uses H_kv
                                                    # Each KV head gets the same centroids (shared compression representation)
                                                    centroids_k_expanded = centroids_k.unsqueeze(0).unsqueeze(0).expand(B, H_kv, -1, -1)
                                                    centroids_v_expanded = centroids_v.unsqueeze(0).unsqueeze(0).expand(B, H_kv, -1, -1)

                                                    # GQA compatibility check: ensure expansion is valid
                                                    if centroids_k_expanded.shape[-1] != D:
                                                        print(f"[MPKVM][WARNING] Centroid dim {centroids_k_expanded.shape[-1]} != KV dim {D}, skipping injection")
                                                        return outputs  # Skip injection for this layer

                                                    present_kv.key_cache[layer_idx] = torch.cat([
                                                        centroids_k_expanded,  # Sink tokens at beginning
                                                        recent_k
                                                    ], dim=-2)
                                                    present_kv.value_cache[layer_idx] = torch.cat([
                                                        centroids_v_expanded,  # Sink tokens at beginning
                                                        recent_v
                                                    ], dim=-2)

                                                    actual_cache_size = present_kv.key_cache[layer_idx].shape[-2]
                                                    target_size = num_sink + window_tokens

                                                    if _DEBUG:
                                                        print(f"[MPKVM][Compression] Layer {layer_idx}: {current_seq_len} -> {actual_cache_size} "
                                                              f"(target: {target_size}, sink: {num_sink}, window: {window_tokens}, "
                                                              f"compression: {current_seq_len/actual_cache_size:.1f}x)")
                                            else:
                                                # No centroids available, just apply sliding window
                                                if current_seq_len > keep_tokens:
                                                    present_kv.key_cache[layer_idx] = present_kv.key_cache[layer_idx][:, :, -keep_tokens:, :]
                                                    present_kv.value_cache[layer_idx] = present_kv.value_cache[layer_idx][:, :, -keep_tokens:, :]
                                        else:
                                            # Legacy behavior: simple append (will cause linear growth)
                                            # WARNING: This will cause the centroid mixing bug!
                                            # Only use when sliding_window_size is not set
                                            present_kv.key_cache[layer_idx] = torch.cat([
                                                present_kv.key_cache[layer_idx],
                                                centroids_k.unsqueeze(0).unsqueeze(0)
                                            ], dim=-2)
                                            present_kv.value_cache[layer_idx] = torch.cat([
                                                present_kv.value_cache[layer_idx],
                                                centroids_v.unsqueeze(0).unsqueeze(0)
                                            ], dim=-2)

                                            if _DEBUG and current_seq_len > 1000:  # Warn about potential memory issues
                                                print(f"[MPKVM][WARNING] Layer {layer_idx}: Cache growing linearly to {present_kv.key_cache[layer_idx].shape[-2]} tokens. "
                                                      f"This will cause CENTROID MIXING BUG! Set sliding_window_size to enable proper compression.")

                                elif hasattr(present_kv, "past_key_values"):
                                    # Some custom cache formats
                                    if layer_idx < len(present_kv.past_key_values):
                                        layer_kv = present_kv.past_key_values[layer_idx]
                                        if isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2:
                                            # Update the layer cache
                                            present_kv.past_key_values[layer_idx] = (k_aug, v_aug)

                                # Update outputs
                                outputs = tuple(outputs_list)

                                if _DEBUG:
                                    print(f"[MPKVM][Injection] Layer {layer_idx} cache updated with {centroids_k.shape[0]} centroids. "
                                          f"New KV shape: {k_aug.shape}")

                            # Also update kwargs for backward compatibility (less critical now that outputs are modified)
                            past_key_values = kwargs.get('past_key_values', None)
                            if past_key_values is not None and layer_idx < len(past_key_values):
                                if isinstance(past_key_values[layer_idx], (list, tuple)) and len(past_key_values[layer_idx]) >= 2:
                                    if isinstance(past_key_values, tuple):
                                        past_key_values = list(past_key_values)
                                    layer_cache = list(past_key_values[layer_idx])
                                    layer_cache[0] = k_aug
                                    layer_cache[1] = v_aug
                                    past_key_values[layer_idx] = tuple(layer_cache)
                                    kwargs['past_key_values'] = tuple(past_key_values) if isinstance(past_key_values, list) else past_key_values

                            # Fallback: modify attn_module attributes (least reliable)
                            else:
                                if hasattr(attn_module, 'last_key'):
                                    attn_module.last_key = k_aug
                                if hasattr(attn_module, 'last_value'):
                                    attn_module.last_value = v_aug

                                if _DEBUG:
                                    print(f"[MPKVM][layer {layer_idx}] fallback injection: modified attn_module attributes")

                    except Exception as e:
                        if _DEBUG:
                            print(f"[MPKVM][layer {layer_idx}] centroid injection failed: {e}")

                return outputs

            return wrapped

        # If cluster_kwargs provided and manager has per-layer storage, attempt to (re)initialize the layer's cluster operator.
            pass
        setattr(attn_module, "forward", make_wrapped(orig_forward, attn_module, idx))
        # mark module as wrapped to avoid double-wrapping by other tools/scripts
        setattr(attn_module, "_mpkvm_wrapped", True)
    return model


