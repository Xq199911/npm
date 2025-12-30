"""
Attention Analysis for MP-KVM Paper

Demonstrates how score_bias = torch.log(cw) corrects attention dilution.
Analyzes attention weight distributions before and after applying log-count compensation.

Results are saved to results/attention_analysis/attention_spectrum_data.json
for use by generate_paper_figures.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import torch
from typing import Dict, Any, List
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layers import reconstruct_with_centroids
from core.clustering import OnlineManifoldClustering
# Synthetic data generation removed


def analyze_attention_weights(query: np.ndarray, keys: np.ndarray, centroids: np.ndarray,
                            centroid_weights: np.ndarray, apply_bias: bool = True) -> Dict[str, Any]:
    """
    Analyze attention weights for a query against original keys and centroids.

    Returns attention weights and energy bias analysis.
    """
    # Create augmented key matrix with centroids
    if centroids is not None and centroids.shape[0] > 0:
        # Augment keys with centroids
        k_aug = np.concatenate([keys, centroids], axis=0)

        # Create score bias if requested
        score_bias = None
        if apply_bias and centroid_weights is not None:
            # score_bias = log(count) for centroids, 0 for original keys
            bias_centroids = np.log(centroid_weights + 1e-12)
            bias_original = np.zeros((keys.shape[0],), dtype=float)
            score_bias = np.concatenate([bias_original, bias_centroids], axis=0)
    else:
        k_aug = keys
        score_bias = None

    # Compute attention weights using scaled dot-product attention
    d = query.shape[-1]
    scores = np.dot(query, k_aug.T) / np.sqrt(float(d))

    if score_bias is not None:
        scores = scores + score_bias[None, :]  # Add bias to scores

    # Stable softmax
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-12)

    # Separate weights for original tokens vs centroids
    n_original = keys.shape[0]
    original_weights = weights[:, :n_original]
    centroid_weights_out = weights[:, n_original:] if centroids is not None else np.array([])

    # Compute statistics
    result = {
        "attention_weights_original": original_weights.flatten().tolist(),
        "attention_weights_centroids": centroid_weights_out.flatten().tolist() if centroid_weights_out.size > 0 else [],
        "max_weight_original": float(np.max(original_weights)),
        "max_weight_centroids": float(np.max(centroid_weights_out)) if centroid_weights_out.size > 0 else 0.0,
        "mean_weight_original": float(np.mean(original_weights)),
        "mean_weight_centroids": float(np.mean(centroid_weights_out)) if centroid_weights_out.size > 0 else 0.0,
        "centroid_count": int(centroids.shape[0]) if centroids is not None else 0,
        "applied_bias": apply_bias
    }

    # Add centroid weight analysis if available
    if centroid_weights is not None and len(centroid_weights) > 0:
        result["log_centroid_weights"] = np.log(centroid_weights + 1e-12).tolist()
        result["raw_centroid_weights"] = centroid_weights.tolist()

    return result


def run_attention_analysis() -> Dict[str, Any]:
    """Run comprehensive attention analysis for MP-KVM paper"""

    print("="*80)
    print("MP-KVM Attention Analysis")
    print("="*80)

    # Use real model data (synthetic data removed)
    import torch
    from run_real_model_experiment import setup_model_and_tokenizer, RealModelKVExtractor, create_long_context_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model_and_tokenizer("model/Llama-3.1-8B-Instruct", device)
    extractor = RealModelKVExtractor(model, tokenizer, device=device)

    # Create long context and extract KV vectors
    context_text = create_long_context_text()
    keys, values = extractor.extract_kv_from_text(context_text, max_length=2048)

    # Create needles from early part of the sequence
    total_available = keys.shape[0]
    early_portion = int(total_available * 0.3)
    early_portion = max(early_portion, min(256, total_available))

    n_needles = min(20, int(early_portion * 0.05))
    np.random.seed(42)
    needle_indices = np.random.choice(early_portion, size=n_needles, replace=False)
    needles = keys[needle_indices].copy()

    dim = keys.shape[1]
    seq_length = keys.shape[0]

    print(f"Using real model data: {seq_length} tokens, {dim} dimensions, {n_needles} needles")

    # Run MP-KVM clustering
    clusterer = OnlineManifoldClustering(
        dim=dim,
        max_centroids=256,
        window_size=2048,
        similarity_threshold=0.8
    )

    # Process data in batches
    batch_size = 64
    for i in range(0, len(keys), batch_size):
        end_idx = min(i + batch_size, len(keys))
        batch_keys = keys[i:end_idx]
        batch_values = values[i:end_idx]
        clusterer.add(batch_keys, batch_values)

    # Get centroids
    centroids, counts, weights = clusterer.get_centroids()

    print(f"Generated {centroids.shape[0]} centroids from {seq_length} tokens")

    # Analyze attention for multiple queries (simulate different positions)
    query_positions = [100, 500, 1000, 2000, 3000]  # Different positions in sequence
    analysis_results = []

    for pos in query_positions:
        if pos >= len(keys):
            continue

        query = keys[pos:pos+1]  # Single query token

        print(f"Analyzing attention for query at position {pos}...")

        # Analyze without bias (standard attention)
        result_no_bias = analyze_attention_weights(
            query, keys, centroids, weights, apply_bias=False
        )
        result_no_bias["query_position"] = pos
        result_no_bias["analysis_type"] = "no_bias"

        # Analyze with bias (MP-KVM energy compensation)
        result_with_bias = analyze_attention_weights(
            query, keys, centroids, weights, apply_bias=True
        )
        result_with_bias["query_position"] = pos
        result_with_bias["analysis_type"] = "with_bias"

        analysis_results.extend([result_no_bias, result_with_bias])

    # Compute aggregate statistics for the paper
    no_bias_results = [r for r in analysis_results if r["analysis_type"] == "no_bias"]
    with_bias_results = [r for r in analysis_results if r["analysis_type"] == "with_bias"]

    # Aggregate attention weight distributions
    all_weights_no_bias = []
    all_weights_with_bias = []

    for result in no_bias_results:
        all_weights_no_bias.extend(result["attention_weights_original"])
        all_weights_no_bias.extend(result["attention_weights_centroids"])

    for result in with_bias_results:
        all_weights_with_bias.extend(result["attention_weights_original"])
        all_weights_with_bias.extend(result["attention_weights_centroids"])

    # Compute energy bias effects
    bias_effects = []
    for nb, wb in zip(no_bias_results, with_bias_results):
        if nb["centroid_count"] > 0 and wb["centroid_count"] > 0:
            # Compare max centroid weights
            max_centroid_nb = nb["max_weight_centroids"]
            max_centroid_wb = wb["max_weight_centroids"]

            bias_effects.append({
                "query_position": nb["query_position"],
                "max_centroid_weight_no_bias": max_centroid_nb,
                "max_centroid_weight_with_bias": max_centroid_wb,
                "improvement_ratio": max_centroid_wb / (max_centroid_nb + 1e-12)
            })

    # Build per-centroid attention matrices across queries for binning analysis
    C = int(centroids.shape[0]) if centroids is not None else 0
    cent_no = np.zeros((len(query_positions), C), dtype=float) if C > 0 else np.zeros((0, 0))
    cent_with = np.zeros((len(query_positions), C), dtype=float) if C > 0 else np.zeros((0, 0))
    qi_idx = 0
    for i in range(0, len(analysis_results), 2):
        # pair: no_bias then with_bias
        nb = analysis_results[i]
        wb = analysis_results[i+1] if (i+1) < len(analysis_results) else None
        if C > 0 and len(nb.get("attention_weights_centroids", [])) >= C:
            cent_no[qi_idx, :] = np.array(nb["attention_weights_centroids"])[:C]
        if wb is not None and C > 0 and len(wb.get("attention_weights_centroids", [])) >= C:
            cent_with[qi_idx, :] = np.array(wb["attention_weights_centroids"])[:C]
        qi_idx += 1

    # Compute per-centroid means across queries
    per_centroid_mean_no = cent_no.mean(axis=0).tolist() if C > 0 else []
    per_centroid_mean_with = cent_with.mean(axis=0).tolist() if C > 0 else []

    # Binning by centroid size (counts)
    centroid_counts = counts if 'counts' in locals() else np.zeros((C,), dtype=int)
    bins = [1,2,4,8,16,32,64,128,256,512,1024, 1<<20]
    bin_labels = []
    bin_stats = {}
    for i in range(len(bins)-1):
        lo = bins[i]
        hi = bins[i+1]
        label = f"{lo}-{hi-1}"
        bin_labels.append(label)
        idxs = [j for j,c in enumerate(centroid_counts) if c >= lo and c < hi]
        if idxs:
            mean_no = float(np.mean([per_centroid_mean_no[j] for j in idxs]))
            mean_with = float(np.mean([per_centroid_mean_with[j] for j in idxs]))
            bin_stats[label] = {"count": len(idxs), "mean_weight_no_bias": mean_no, "mean_weight_with_bias": mean_with, "improvement": mean_with/(mean_no+1e-12)}
        else:
            bin_stats[label] = {"count": 0, "mean_weight_no_bias": 0.0, "mean_weight_with_bias": 0.0, "improvement": 0.0}

    # Prepare final results
    final_results = {
        "experiment": "MP-KVM Attention Energy Spectrum Analysis",
        "description": "Demonstrates how score_bias = torch.log(cw) corrects attention dilution",
        "sequence_length": seq_length,
        "n_centroids": int(centroids.shape[0]),
        "query_positions_analyzed": query_positions,
        "attention_distributions": {
            "no_bias_weights": all_weights_no_bias,
            "with_bias_weights": all_weights_with_bias
        },
        "bias_effects": bias_effects,
        "detailed_results": analysis_results,
        "centroid_weights": weights.tolist() if weights is not None else [],
        "log_centroid_weights": np.log(weights + 1e-12).tolist() if weights is not None else [],
        "centroid_counts": counts.tolist() if counts is not None else [],
        "per_centroid_mean_no_bias": per_centroid_mean_no,
        "per_centroid_mean_with_bias": per_centroid_mean_with,
        "centroid_size_bins": bin_stats
    }

    return final_results


def main():
    """Main function to run attention analysis and save results"""

    # Create output directory
    output_dir = "results/attention_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Run analysis
    results = run_attention_analysis()

    # Save results
    output_file = os.path.join(output_dir, "attention_spectrum_data.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved attention analysis results to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*80)

    bias_effects = results["bias_effects"]
    if bias_effects:
        avg_improvement = np.mean([e["improvement_ratio"] for e in bias_effects])
        print(".2f")
        print("This demonstrates how log-count compensation prevents centroid attention dilution.")

    print(f"\nAnalyzed {len(results['query_positions_analyzed'])} query positions")
    print(f"Generated {results['n_centroids']} centroids from {results['sequence_length']} tokens")
    print("\nRun 'python generate_paper_figures.py' to generate the attention spectrum figure.")


if __name__ == "__main__":
    main()
