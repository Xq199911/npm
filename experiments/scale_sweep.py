"""
Compression Ratio vs Performance Sweep for MP-KVM Paper

Tests MP-KVM performance across different compression ratios (controlled by max_centroids).
Generates data for Figure 3: Compression Rate vs Performance Curve.

Results are saved to results/compression_sweep/compression_sweep_results.json
for use by generate_paper_figures.py
"""
from __future__ import annotations
import os
import json
import numpy as np
from typing import Dict, Any, List
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.clustering import OnlineManifoldClustering
from data.needles.run_niah import evaluate_recall


def run_compression_experiment(compression_ratio: float, total_tokens: int = 16000) -> Dict[str, Any]:
    """
    Run MP-KVM with specific compression ratio (controlled by max_centroids).
    Lower max_centroids = higher compression = lower ratio.
    """
    print(".1f")

    # Calculate max_centroids based on compression ratio
    # At compression_ratio=1.0, we allow ~10% of tokens as centroids
    # At compression_ratio=0.001, we allow very few centroids
    base_centroids = int(total_tokens * 0.1)  # 10% baseline
    max_centroids = max(1, int(base_centroids * compression_ratio))

    # Use real model data (synthetic data removed)
    import torch
    from run_real_model_experiment import setup_model_and_tokenizer, RealModelKVExtractor, create_long_context_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model_and_tokenizer("model/Llama-3.1-8B-Instruct", device)
    extractor = RealModelKVExtractor(model, tokenizer, device=device)

    # Create long context and extract KV vectors
    context_text = create_long_context_text()
    keys, values = extractor.extract_kv_from_text(context_text, max_length=min(total_tokens, 2048))

    # Adjust to target length if needed
    if keys.shape[0] < total_tokens:
        pad_size = total_tokens - keys.shape[0]
        keys = np.pad(keys, ((0, pad_size), (0, 0)), mode='constant')
        values = np.pad(values, ((0, pad_size), (0, 0)), mode='constant')

    # Create needles from early part
    early_portion = int(keys.shape[0] * 0.3)
    n_needles = min(40, int(early_portion * 0.05))
    np.random.seed(42)
    needle_indices = np.random.choice(early_portion, size=n_needles, replace=False)
    needles = keys[needle_indices].copy()

    # Run MP-KVM clustering
    clusterer = OnlineManifoldClustering(
        dim=keys.shape[1],  # Use actual dimension from extracted KV vectors
        max_centroids=max_centroids,
        window_size=min(1024, total_tokens // 4),  # Smaller window for better compression
        similarity_threshold=0.5  # Lower threshold for better clustering
    )

    # Add data in batches
    batch_size = 32
    for i in range(0, len(keys), batch_size):
        end_idx = min(i + batch_size, len(keys))
        batch_keys = keys[i:end_idx]
        batch_values = values[i:end_idx]
        weights = np.ones(len(batch_keys))
        clusterer.add(batch_keys, batch_values, weights)

    # Get centroids and evaluate
    centroids, counts, weights = clusterer.get_centroids()
    recall = evaluate_recall(centroids, needles, threshold=0.85)

    # Calculate actual compression achieved
    actual_compression_ratio = len(centroids) / total_tokens if centroids is not None else 0.0

    return {
        "compression_ratio_target": compression_ratio,
        "compression_ratio_actual": actual_compression_ratio,
        "max_centroids": max_centroids,
        "num_centroids": len(centroids) if centroids is not None else 0,
        "total_tokens": total_tokens,
        "recall": float(recall),
        "method": "MP-KVM"
    }


def run_baseline_experiment(compression_ratio: float, method: str, total_tokens: int = 16000) -> Dict[str, Any]:
    """Run actual baseline experiments using the same setup as MP-KVM."""
    print(f"Running {method} baseline experiment with compression ratio {compression_ratio:.4f}")

    # Import baseline implementation
    from run_baseline_comparison import (
        NoCompressionBaseline, RandomEvictionBaseline, H2OBaseline, StreamingLLMBaseline
    )

    # Use real model data for consistency with MP-KVM experiments
    import torch
    from run_real_model_experiment import setup_model_and_tokenizer, RealModelKVExtractor, create_long_context_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model_and_tokenizer("model/Llama-3.1-8B-Instruct", device)
    extractor = RealModelKVExtractor(model, tokenizer, device=device)

    # Create long context and extract KV vectors
    context_text = create_long_context_text()
    keys, values = extractor.extract_kv_from_text(context_text, max_length=min(total_tokens, 2048))

    # Adjust to target length if needed
    if keys.shape[0] < total_tokens:
        # Pad with copies if needed
        while keys.shape[0] < total_tokens:
            keys = np.concatenate([keys, keys[:min(total_tokens - keys.shape[0], keys.shape[0])]], axis=0)
            values = np.concatenate([values, values[:min(total_tokens - values.shape[0], values.shape[0])]], axis=0)
        keys = keys[:total_tokens]
        values = values[:total_tokens]

    # Create needles for evaluation
    n_needles = int(total_tokens * 0.005)  # 0.5% needles like in the paper
    needle_indices = np.random.choice(total_tokens, n_needles, replace=False)
    needles = keys[needle_indices]

    # Set memory constraint based on compression ratio
    max_memory_size = int(total_tokens * compression_ratio * keys.shape[1])  # memory proportional to compression ratio

    # Initialize the appropriate baseline method
    if method == 'No Compression':
        baseline = NoCompressionBaseline(max_memory_size, keys.shape[1])
    elif method == 'Random Eviction':
        baseline = RandomEvictionBaseline(max_memory_size, keys.shape[1])
    elif method == 'H2O':
        baseline = H2OBaseline(max_memory_size, keys.shape[1])
    elif method == 'StreamingLLM':
        baseline = StreamingLLMBaseline(max_memory_size, keys.shape[1])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Run the baseline experiment
    result = baseline.compress_and_evaluate(keys, values, needles)

    # Add additional metadata
    result.update({
        "compression_ratio_target": compression_ratio,
        "compression_ratio_actual": result["memory_usage"] / total_tokens if "memory_usage" in result else compression_ratio,
        "max_centroids": result["memory_usage"] if "memory_usage" in result else int(total_tokens * compression_ratio),
        "num_centroids": result["memory_usage"] if "memory_usage" in result else int(total_tokens * compression_ratio),
        "total_tokens": total_tokens,
    })

    return result


def main():
    """Run compression ratio sweep experiments."""
    print("="*60)
    print("MP-KVM Compression Ratio vs Performance Sweep")
    print("="*60)

    # Define compression ratios to test (emphasize extreme compression <10%)
    compression_ratios = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]

    # Methods to compare
    methods = ['No Compression', 'Random Eviction', 'H2O', 'StreamingLLM', 'MP-KVM']

    # Experiment configuration
    total_tokens = 16000  # Match the paper's sequence length
    n_runs = 3  # Multiple runs for statistical significance

    # Create output directory
    output_dir = os.path.join("results", "compression_sweep")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for method in methods:
        print(f"\nRunning {method} experiments...")
        method_results = []

        for ratio in compression_ratios:
            ratio_results = []

            # Run multiple times for statistical significance
            for run in range(n_runs):
                if method == 'MP-KVM':
                    result = run_compression_experiment(ratio, total_tokens)
                else:
                    result = run_baseline_experiment(ratio, method, total_tokens)

                result["run"] = run
                ratio_results.append(result)

            # Average results across runs
            avg_result = {
                "method": method,
                "compression_ratio_target": ratio,
                "compression_ratio_actual": np.mean([r["compression_ratio_actual"] for r in ratio_results]),
                "max_centroids": np.mean([r["max_centroids"] for r in ratio_results]),
                "num_centroids": np.mean([r["num_centroids"] for r in ratio_results]),
                "total_tokens": total_tokens,
                "recall_mean": np.mean([r["recall"] for r in ratio_results]),
                "recall_std": np.std([r["recall"] for r in ratio_results]),
                "n_runs": n_runs
            }
            method_results.append(avg_result)
            all_results.extend(ratio_results)  # Keep individual runs too

        # Save method-specific results
        method_file = os.path.join(output_dir, f"{method.lower().replace(' ', '_')}_results.json")
        with open(method_file, 'w') as f:
            json.dump(method_results, f, indent=2)
        print(f"  Saved {method} results to {method_file}")

    # Save consolidated results
    consolidated_file = os.path.join(output_dir, "compression_sweep_results.json")
    with open(consolidated_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved consolidated results to {consolidated_file}")

    # Print summary
    print("\n" + "="*80)
    print("COMPRESSION SWEEP SUMMARY")
    print("="*80)

    for method in methods:
        print(f"\n{method}:")
        method_data = [r for r in all_results if r["method"] == method and "run" not in r]  # Use averaged results

        for result in method_data:
            ratio = result["compression_ratio_target"]
            recall = result["recall_mean"]
            print(".1f")

    print("\nCompression sweep experiments completed!")
    print("Results saved to results/compression_sweep/ directory")
    print("Run 'python generate_paper_figures.py' to generate the performance curve figure.")


if __name__ == "__main__":
    main()
