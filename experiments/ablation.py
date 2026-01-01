"""
Ablation script: vary similarity_threshold and max_centroids and measure needles recall.
Saves CSV and plots to output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.clustering import OnlineManifoldClustering
from data.needles.run_niah import evaluate_recall
from run_real_model_experiment import setup_model_and_tokenizer, create_long_context_text, RealModelKVExtractor
import math
import torch
import numpy as np


def compute_ppl_with_compression(model, tokenizer, device, eval_text: str,
                               max_length: int = 512, use_compression: bool = False,
                               compression_config: Optional[Dict] = None) -> float:
    """
    Compute perplexity with or without MP-KVM compression.

    Args:
        model: The language model
        tokenizer: Tokenizer
        device: Device to run on
        eval_text: Text to evaluate
        max_length: Maximum sequence length
        use_compression: Whether to apply MP-KVM compression
        compression_config: Configuration for compression (centroids, etc.)
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(eval_text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(device)

        if use_compression and compression_config is not None:
            # Apply MP-KVM compression during inference
            from adapters.llama_adapter import attach_mpkvm_to_hf_llama
            from core.integration_clean import MPKVMManager

            # Create MP-KVM manager with provided centroids
            manager = MPKVMManager(
                dim=compression_config.get('dim', 4096),  # Default Llama hidden size
                num_layers=model.config.num_hidden_layers,
                cluster_kwargs=compression_config.get('cluster_kwargs', {})
            )

            # Load pre-computed centroids if available
            if 'centroids' in compression_config:
                # This would need to be implemented to load centroids into manager
                # For now, we'll use the adapter which will handle centroid injection
                pass

            # Attach MP-KVM to model for compressed inference
            attach_mpkvm_to_hf_llama(
                model, manager,
                enable_injection=True,
                max_injected_centroids=compression_config.get('max_centroids', 256)
            )

            # Run inference with compression
            outputs = model(input_ids, labels=input_ids)

            # Note: In a full implementation, we would need to ensure that the
            # centroids are properly loaded into the manager before inference
            loss = outputs.loss.item()

        else:
            # Standard inference without compression
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

    try:
        ppl = float(math.exp(loss))
    except OverflowError:
        ppl = float("inf")
    return ppl


def compute_ppl_on_model(model, tokenizer, device, eval_text: str, max_length: int = 512) -> float:
    """Legacy function for backward compatibility - computes PPL without compression."""
    return compute_ppl_with_compression(model, tokenizer, device, eval_text, max_length,
                                       use_compression=False)


def run_single(keys, values, needles, similarity_threshold: float, max_centroids: int, args, model=None, tokenizer=None, device=None):
    cluster = OnlineManifoldClustering(
        dim=keys.shape[1],
        window_size=args.window_size,
        max_memory_size=args.max_memory_size,
        max_centroids=max_centroids,
        metric="cosine",
        similarity_threshold=similarity_threshold,
    )
    n = keys.shape[0]
    bs = args.batch_size
    for i in range(0, n, bs):
        cluster.add(keys[i : i + bs], values[i : i + bs])
    centroids, counts, weights = cluster.get_centroids()
    recall = evaluate_recall(centroids, needles, threshold=args.recall_threshold)
    result = {"similarity_threshold": similarity_threshold,
              "max_centroids": max_centroids,
              "num_centroids": int(centroids.shape[0]) if centroids is not None else 0,
              "recall": float(recall)}
    # If model provided, compute PPL with and without compression
    if model is not None and tokenizer is not None and device is not None:
        try:
            eval_text = create_long_context_text(length=512)

            # Compute PPL without compression (baseline)
            ppl_no_compression = compute_ppl_with_compression(
                model, tokenizer, device, eval_text, max_length=512,
                use_compression=False
            )
            result["ppl_no_compression"] = float(ppl_no_compression)

            # Compute PPL with compression (if centroids are available)
            if centroids is not None and centroids.shape[0] > 0:
                compression_config = {
                    'dim': keys.shape[1],
                    'max_centroids': max_centroids,
                    'centroids': centroids,
                    'cluster_kwargs': {
                        'similarity_threshold': similarity_threshold,
                        'max_centroids': max_centroids
                    }
                }

                ppl_with_compression = compute_ppl_with_compression(
                    model, tokenizer, device, eval_text, max_length=512,
                    use_compression=True, compression_config=compression_config
                )
                result["ppl_with_compression"] = float(ppl_with_compression)
                result["ppl_ratio"] = float(ppl_with_compression / ppl_no_compression) if ppl_no_compression > 0 else float('inf')
            else:
                result["ppl_with_compression"] = None
                result["ppl_ratio"] = None

        except Exception as e:
            result["ppl_error"] = str(e)
            result["ppl_no_compression"] = None
            result["ppl_with_compression"] = None
            result["ppl_ratio"] = None
    return result


def run_naive_vs_weighted_ablation(args):
    """Ablation experiment: Naive Mean vs Weighted Mean clustering."""
    print("Running Naive Mean vs Weighted Mean ablation...")

    # Use real model data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model_and_tokenizer(getattr(args, "model_path", "model/Llama-3.1-8B-Instruct"), device)
    extractor = RealModelKVExtractor(model, tokenizer, device=device)

    # Create long context and extract KV vectors
    context_text = create_long_context_text()
    keys, values = extractor.extract_kv_from_text(context_text, max_length=min(args.total_tokens, 2048))

    # Create needles from EARLY PART of the real KV data
    total_available = keys.shape[0]
    early_portion = int(total_available * 0.3)  # First 30% of sequence
    early_portion = max(early_portion, min(256, total_available))  # At least 256, or all if smaller

    # Limit needle density
    max_needles = int(early_portion * 0.05)  # 5% density
    n_needles = min(args.n_needles, max_needles)

    # Sample needles from early portion
    needle_indices = np.random.RandomState(args.seed).choice(early_portion, size=n_needles, replace=False)
    needles = keys[needle_indices].copy()

    print(f"Real model KV data: {keys.shape[0]} tokens, {keys.shape[1]} dimensions, {n_needles} needles")

    results = []

    # Test different similarity thresholds
    for sim_thresh in [0.6, 0.8, 0.9]:
        # Weighted Mean (MP-KVM standard)
        cluster_weighted = OnlineManifoldClustering(
            dim=keys.shape[1],  # Use actual dimension from extracted KV vectors
            window_size=args.window_size,
            max_memory_size=args.max_memory_size,
            max_centroids=args.max_centroids_list[0],  # Use first max_centroids value
            metric="cosine",
            similarity_threshold=sim_thresh,
        )

        # Add data with weights (this enables weighted averaging)
        for i in range(0, len(keys), args.batch_size):
            k_batch = keys[i : i + args.batch_size]
            v_batch = values[i : i + args.batch_size]
            # Use attention-like weights (simulate importance)
            w_batch = np.random.exponential(1.0, len(k_batch)).astype(np.float32)
            cluster_weighted.add(k_batch, v_batch, w_batch)

        centroids_weighted, counts_weighted, weights_weighted = cluster_weighted.get_centroids()
        recall_weighted = evaluate_recall(centroids_weighted, needles, threshold=args.recall_threshold)

        # Naive Mean (uniform weights)
        cluster_naive = OnlineManifoldClustering(
            dim=keys.shape[1],  # Use actual dimension from extracted KV vectors
            window_size=args.window_size,
            max_memory_size=args.max_memory_size,
            max_centroids=args.max_centroids_list[0],
            metric="cosine",
            similarity_threshold=sim_thresh,
        )

        # Add data without weights (uniform weighting)
        for i in range(0, len(keys), args.batch_size):
            k_batch = keys[i : i + args.batch_size]
            v_batch = values[i : i + args.batch_size]
            cluster_naive.add(k_batch, v_batch)  # No weights = uniform weighting

        centroids_naive, counts_naive, weights_naive = cluster_naive.get_centroids()
        recall_naive = evaluate_recall(centroids_naive, needles, threshold=args.recall_threshold)

        results.append({
            "similarity_threshold": sim_thresh,
            "max_centroids": args.max_centroids_list[0],  # Use first max_centroids value
            "method": "Weighted Mean (MP-KVM)",
            "recall": float(recall_weighted),
            "num_centroids": centroids_weighted.shape[0] if centroids_weighted is not None else 0,
        })

        results.append({
            "similarity_threshold": sim_thresh,
            "max_centroids": args.max_centroids_list[0],  # Use first max_centroids value
            "method": "Naive Mean (Uniform)",
            "recall": float(recall_naive),
            "num_centroids": centroids_naive.shape[0] if centroids_naive is not None else 0,
        })

    return results


def run_grid(args):
    # Always use real model data (synthetic data removed)
    print("Using REAL MODEL KV data for ablation study")
    # Load real model and extract KV data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model_and_tokenizer(getattr(args, "model_path", "model/Llama-3.1-8B-Instruct"), device)
    extractor = RealModelKVExtractor(model, tokenizer, device=device)

    # Create long context and extract KV vectors
    context_text = create_long_context_text()
    keys, values = extractor.extract_kv_from_text(context_text, max_length=min(args.total_tokens, 2048))

    # Create needles from EARLY PART of the real KV data
    total_available = keys.shape[0]
    early_portion = int(total_available * 0.3)  # First 30% of sequence
    early_portion = max(early_portion, min(256, total_available))  # At least 256, or all if smaller

    # Limit needle density
    max_needles = int(early_portion * 0.05)  # 5% density
    n_needles = min(args.n_needles, max_needles)
    print(f"Using {n_needles} needles from first {early_portion} tokens of {total_available} total tokens")

    # Sample needles from early portion
    needle_indices = np.random.RandomState(args.seed).choice(early_portion, size=n_needles, replace=False)
    needles = keys[needle_indices].copy()

    print(f"Real model KV data: {keys.shape[0]} tokens, {keys.shape[1]} dimensions, {n_needles} needles")

    results = []
    # Load model for PPL computation if requested
    model = None
    tokenizer = None
    device = None
    if getattr(args, "use_real_model", False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = setup_model_and_tokenizer(getattr(args, "model_path", "model/Llama-3.1-8B-Instruct"), device)

    for mc in args.max_centroids_list:
        for th in args.sim_thresholds:
            print(f"Running mc={mc} th={th}")
            r = run_single(keys, values, needles, similarity_threshold=th, max_centroids=mc, args=args, model=model, tokenizer=tokenizer, device=device)
            results.append(r)

    out = {"params": vars(args), "results": results}
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote ablation results to", out_path)
    return results


def plot_results(results, out_dir: str):
    # results: list of dicts with keys similarity_threshold, max_centroids, recall
    os.makedirs(out_dir, exist_ok=True)
    # group by max_centroids
    by_mc = {}
    for r in results:
        mc = r["max_centroids"]
        by_mc.setdefault(mc, []).append(r)
    plt.figure(figsize=(8, 5))
    for mc, items in sorted(by_mc.items()):
        items_sorted = sorted(items, key=lambda x: x["similarity_threshold"])
        xs = [it["similarity_threshold"] for it in items_sorted]
        ys = [it["recall"] for it in items_sorted]
        plt.plot(xs, ys, marker="o", label=f"max_centroids={mc}")
    plt.xlabel("similarity_threshold")
    plt.ylabel("recall")
    plt.title("Ablation: recall vs similarity_threshold")
    plt.legend()
    out_png = os.path.join(out_dir, "ablation_recall_vs_threshold.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved plot to", out_png)
    return out_png


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results/ablation")
    p.add_argument("--total-tokens", type=int, default=8000)  # Reduced to ensure more frequent compression
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--cluster-std", type=float, default=0.5)
    p.add_argument("--n-needles", type=int, default=40)  # Reduced needle count
    p.add_argument("--batch-size", type=int, default=64)  # Increased batch size for better clustering
    p.add_argument("--window-size", type=int, default=2048)  # Reduced window size to trigger more frequent compression
    p.add_argument("--max-memory-size", type=int, default=32768)  # Reduced memory size
    p.add_argument("--recall-threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])  # Lower thresholds for better clustering
    p.add_argument("--max-centroids-list", type=int, nargs="+", default=[32, 64, 128, 256])  # More reasonable centroid limits
    p.add_argument("--needle-near", dest="needle_near", action="store_true", help="Sample needles near cluster centers (easier to recover).")
    p.add_argument("--needle-near-scale", type=float, default=0.5, help="Scale for needle proximity when --needle-near is used.")
    p.add_argument("--use-real-model", action="store_true", help="Use a real Llama model to compute PPL for each ablation configuration")
    p.add_argument("--model-path", type=str, default="model/Llama-3.1-8B-Instruct", help="Path to real model for PPL computation")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # normalize lists
    args.sim_thresholds = list(args.sim_thresholds)
    args.max_centroids_list = list(args.max_centroids_list)

    # Run standard grid search
    results = run_grid(args)

    # Run naive vs weighted ablation
    naive_weighted_results = run_naive_vs_weighted_ablation(args)
    results.extend(naive_weighted_results)

    plot_results(results, args.out)


if __name__ == "__main__":
    main()


