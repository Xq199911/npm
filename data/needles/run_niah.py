"""
Synthetic "Needles-in-a-haystack" experiment for MP-KVM.

This script creates a long stream of KV vectors with a small fraction of
rare "needle" vectors and measures how often those needles are preserved
by the online clustering operator (i.e., whether centroids recover them).

Usage:
    python -m data.needles.run_niah
"""
from __future__ import annotations
import argparse
import os
import json
import numpy as np

from core.clustering import OnlineManifoldClustering


def evaluate_recall(centroids: np.ndarray, needles: np.ndarray, threshold: float = 0.85):
    """
    For each needle vector, compute max cosine similarity with centroids.
    Count as recovered if max_sim >= threshold.
    """
    if centroids is None or centroids.shape[0] == 0:
        return 0.0

    # normalize
    def _norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    c_n = _norm(centroids)
    needles_n = _norm(needles)
    sims = np.dot(needles_n, c_n.T)
    max_sims = sims.max(axis=1)
    recovered = (max_sims >= threshold).sum()
    return float(recovered) / float(needles.shape[0])


def run_experiment(args):
    # Check if we should use real model data
    if hasattr(args, 'use_real_model') and args.use_real_model:
        print("Using real model KV data...")
        from run_real_model_experiment import setup_model_and_tokenizer, RealModelKVExtractor, create_long_context_text

        model_path = getattr(args, 'model_path', 'model/Llama-3.1-8B-Instruct')
        model, tokenizer = setup_model_and_tokenizer(model_path, device="cpu")  # Use CPU for KV extraction
        extractor = RealModelKVExtractor(model, tokenizer, device="cpu")

        # Create long context and extract KV vectors
        context_text = create_long_context_text()
        keys, values = extractor.extract_kv_from_text(context_text, max_length=min(args.total_tokens, 2048))

        # Create needles from EARLY PART of the real KV data to ensure fair baseline comparison
        # Needles should be placed in the first 30% of the sequence to avoid being trivially
        # preserved by sliding window methods or random sampling
        total_available = keys.shape[0]
        early_portion = int(total_available * 0.3)  # First 30% of sequence
        early_portion = max(early_portion, min(256, total_available))  # At least 256, or all if smaller

        # Limit needle density to 2-5% of early portion
        max_needles = int(early_portion * 0.05)  # 5% density
        n_needles = min(args.n_needles, max_needles, early_portion)

        if n_needles > 0:
            needle_indices = np.random.RandomState(args.seed).choice(early_portion, size=n_needles, replace=False)
            needles = keys[needle_indices].copy()
            print(f"Placed {n_needles} needles in first {early_portion} positions (of {total_available} total, {n_needles/early_portion:.1%} density)")
        else:
            # Fallback if no early positions available
            needles = keys[:1].copy()
            print(f"Warning: Using fallback needle from position 0")

        # Adjust dimensions if needed
        target_dim = args.dim
        if keys.shape[1] != target_dim:
            print(f"Adjusting dimension from {keys.shape[1]} to {target_dim}")
            # Simple projection for dimension adjustment
            if keys.shape[1] > target_dim:
                keys = keys[:, :target_dim]
                values = values[:, :target_dim]
                needles = needles[:, :target_dim]
            else:
                # Pad with zeros
                pad_size = target_dim - keys.shape[1]
                keys = np.pad(keys, ((0, 0), (0, pad_size)), mode='constant')
                values = np.pad(values, ((0, 0), (0, pad_size)), mode='constant')
                needles = np.pad(needles, ((0, 0), (0, pad_size)), mode='constant')

        print(f"Using real model data: {keys.shape[0]} KV pairs, {n_needles} needles")

        # Cleanup
        extractor.cleanup()
        del model, tokenizer
    # Note: Synthetic data generation has been removed. Always use real model data.

    # create online clustering operator with sliding window to emulate KV cache
    cluster = OnlineManifoldClustering(dim=args.dim, window_size=args.window_size, max_memory_size=args.max_memory_size,
                                       max_centroids=args.max_centroids, metric="cosine",
                                       similarity_threshold=args.similarity_threshold)

    # ingest stream in small batches to mimic token-by-token arrival
    batch_size = args.batch_size
    n = keys.shape[0]
    for i in range(0, n, batch_size):
        k_batch = keys[i: i + batch_size]
        v_batch = values[i: i + batch_size]
        cluster.add(k_batch, v_batch)

    centroids, counts, weights = cluster.get_centroids()

    # Debug information
    print(f"Debug: Generated {centroids.shape[0] if centroids is not None else 0} centroids")
    print(f"Debug: Total tokens processed: {keys.shape[0]}")
    print(f"Debug: Max centroids allowed: {args.max_centroids}")
    print(f"Debug: Similarity threshold: {args.similarity_threshold}")

    if centroids is not None and centroids.shape[0] > 0:
        print(f"Debug: Centroid counts range: {counts.min()} - {counts.max()}")
        print(f"Debug: Centroid weights range: {weights.min():.3f} - {weights.max():.3f}")

    recall = evaluate_recall(centroids, needles, threshold=args.recall_threshold)

    # Debug needle recovery
    if centroids is not None and centroids.shape[0] > 0:
        # normalize function (same as in evaluate_recall)
        def _norm(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return x / n

        # Compute similarities between needles and centroids
        c_n = _norm(centroids)
        needles_n = _norm(needles)
        sims = np.dot(needles_n, c_n.T)
        max_sims = sims.max(axis=1)
        print(f"Debug: Needle-centroid similarities: min={max_sims.min():.3f}, max={max_sims.max():.3f}, mean={max_sims.mean():.3f}")
        print(f"Debug: Recall threshold: {args.recall_threshold}")
        recovered_count = (max_sims >= args.recall_threshold).sum()
        print(f"Debug: Recovered needles: {recovered_count}/{len(needles)}")
        # Show top 5 similarities for debugging
        top_indices = np.argsort(max_sims)[-5:][::-1]
        print("Debug: Top 5 needle similarities:")
        for i, idx in enumerate(top_indices):
            print(".3f")

    out = {
        "params": vars(args),
        "num_centroids": int(centroids.shape[0]) if centroids is not None else 0,
        "recall": float(recall),
    }

    # Always print results
    print(json.dumps(out, indent=2))

    # Save results if output directory is specified
    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "needles_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        # Save centroids and metadata for visualization/export
        if getattr(args, "save_centroids", True):
            if centroids is not None:
                cent_path = os.path.join(args.out, "centroids.npy")
                np.save(cent_path, centroids.astype(np.float32))

            counts_path = os.path.join(args.out, "centroid_counts.npy")
            np.save(counts_path, counts.astype(np.float32) if counts is not None else np.array([], dtype=np.float32))

            weights_path = os.path.join(args.out, "centroid_weights.npy")
            np.save(weights_path, weights.astype(np.float32) if weights is not None else np.array([], dtype=np.float32))

            print(f"Wrote centroids to {cent_path} and counts to {counts_path}")

        print(f"Wrote results to {out_path}")

    # Always return the result for programmatic use
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--total-tokens", type=int, default=20000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--cluster-std", type=float, default=0.5)
    p.add_argument("--n-needles", type=int, default=50)
    p.add_argument("--window-size", type=int, default=4096)
    p.add_argument("--max-memory-size", type=int, default=65536)
    p.add_argument("--max-centroids", type=int, default=1024)
    p.add_argument("--similarity-threshold", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--recall-threshold", type=float, default=0.85)
    p.add_argument("--needle-near", dest="needle_near", action="store_true",
                   help="Sample needles near existing cluster centers (easier to recover).")
    p.add_argument("--needle-near-scale", type=float, default=0.5,
                   help="Scale for needle proximity when --needle-near is used.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
