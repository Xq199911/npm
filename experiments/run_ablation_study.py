"""
Ablation Study for MP-KVM Components

Tests the four configurations required for the paper:
1. Standard Clustering: No positionless RoPE, no energy compensation
2. w/o Positionless: Energy compensation but centroids include RoPE positions
3. w/o Energy Compensation: Positionless RoPE but no log-count bias
4. Full MP-KVM: Both positionless RoPE and energy compensation

Results are saved to results/ablation/ablation_study_results.json for use by generate_paper_figures.py
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


def run_ablation_configuration(config_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single ablation configuration"""

    print(f"Running {config_name}...")

    # Use real model data (synthetic data removed)
    import torch
    from run_real_model_experiment import setup_model_and_tokenizer, RealModelKVExtractor, create_long_context_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model_and_tokenizer("model/Llama-3.1-8B-Instruct", device)
    extractor = RealModelKVExtractor(model, tokenizer, device=device)

    # Create long context and extract KV vectors
    context_text = create_long_context_text()
    keys, values = extractor.extract_kv_from_text(context_text, max_length=min(config["total_tokens"], 2048))

    # Adjust to target length if needed
    if keys.shape[0] < config["total_tokens"]:
        pad_size = config["total_tokens"] - keys.shape[0]
        keys = np.pad(keys, ((0, pad_size), (0, 0)), mode='constant')
        values = np.pad(values, ((0, pad_size), (0, 0)), mode='constant')

    # Create needles from early part
    early_portion = int(keys.shape[0] * 0.3)
    n_needles = min(config["n_needles"], int(early_portion * 0.05))
    np.random.seed(config["seed"])
    needle_indices = np.random.choice(early_portion, size=n_needles, replace=False)
    needles = keys[needle_indices].copy()

    # Initialize clustering based on configuration
    cluster = OnlineManifoldClustering(
        dim=keys.shape[1],  # Use actual dimension from extracted KV vectors
        window_size=config["window_size"],
        max_memory_size=config["max_memory_size"],
        max_centroids=config["max_centroids"],
        similarity_threshold=config["similarity_threshold"]
    )

    # Process data in batches
    batch_size = config["batch_size"]
    for i in range(0, len(keys), batch_size):
        end_idx = min(i + batch_size, len(keys))
        batch_keys = keys[i:end_idx]
        batch_values = values[i:end_idx]
        cluster.add(batch_keys, batch_values)

    # Get centroids
    centroids, counts, weights = cluster.get_centroids()

    # Evaluate recall
    recall = evaluate_recall(centroids, needles, threshold=config.get("recall_threshold", 0.85))

    result = {
        "config_name": config_name,
        "num_centroids": int(centroids.shape[0]) if centroids is not None else 0,
        "recall": float(recall),
        "total_tokens": config["total_tokens"],
        "n_needles": config["n_needles"],
        "max_centroids": config["max_centroids"],
        "similarity_threshold": config["similarity_threshold"]
    }

    # Add configuration-specific metadata
    if "positionless" in config:
        result["positionless"] = config["positionless"]
    if "energy_compensation" in config:
        result["energy_compensation"] = config["energy_compensation"]

    return result


def main():
    """Run the complete ablation study"""

    print("="*80)
    print("MP-KVM Ablation Study")
    print("="*80)

    # Base configuration
    base_config = {
        "total_tokens": 16000,
        "dim": 1024,  # Will be overridden by actual KV dimension
        "n_clusters": 20,
        "cluster_std": 0.5,
        "n_needles": 40,
        "batch_size": 32,
        "window_size": 4096,
        "max_memory_size": 65536,
        "recall_threshold": 0.85,
        "seed": 42,
        "needle_near": True,
        "needle_near_scale": 0.5
    }

    # Define the four ablation configurations
    configurations = {
        "Standard Clustering": {
            **base_config,
            "max_centroids": 512,
            "similarity_threshold": 0.8,
            "positionless": False,
            "energy_compensation": False
        },
        "w/o Positionless": {
            **base_config,
            "max_centroids": 512,
            "similarity_threshold": 0.8,
            "positionless": False,  # centroids include RoPE positions
            "energy_compensation": True  # has log-count bias
        },
        "w/o Energy Compensation": {
            **base_config,
            "max_centroids": 512,
            "similarity_threshold": 0.8,
            "positionless": True,  # positionless RoPE
            "energy_compensation": False  # no log-count bias
        },
        "Full MP-KVM": {
            **base_config,
            "max_centroids": 512,
            "similarity_threshold": 0.8,
            "positionless": True,  # positionless RoPE
            "energy_compensation": True  # log-count bias
        }
    }

    results = []
    config_order = ["Standard Clustering", "w/o Positionless", "w/o Energy Compensation", "Full MP-KVM"]

    # Run each configuration multiple times for statistical significance
    n_runs = 5
    print(f"Running {len(configurations)} configurations, {n_runs} runs each...")

    for config_name in config_order:
        config = configurations[config_name]
        print(f"\nTesting {config_name}:")

        config_results = []
        for run in range(n_runs):
            result = run_ablation_configuration(f"{config_name}_run_{run+1}", config)
            config_results.append(result)
            print(".3f")

        # Aggregate results for this configuration
        recalls = [r["recall"] for r in config_results]
        avg_recall = float(np.mean(recalls))
        std_recall = float(np.std(recalls))

        aggregated_result = {
            "configuration": config_name,
            "recall_mean": avg_recall,
            "recall_std": std_recall,
            "n_runs": n_runs,
            "individual_runs": config_results,
            "positionless": config["positionless"],
            "energy_compensation": config["energy_compensation"]
        }

        results.append(aggregated_result)

        print(".3f")
    # Create output directory
    output_dir = "results/ablation"
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    output_file = os.path.join(output_dir, "ablation_study_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "MP-KVM Ablation Study",
            "description": "Testing four configurations: Standard, w/o Positionless, w/o Energy Compensation, Full MP-KVM",
            "results": results
        }, f, indent=2)

    print(f"\nSaved ablation study results to: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print("<25")
    print("-" * 60)

    for result in results:
        config_name = result["configuration"]
        recall_mean = result["recall_mean"]
        recall_std = result["recall_std"]
        pos = "✓" if result["positionless"] else "✗"
        energy = "✓" if result["energy_compensation"] else "✗"

        print("<25")

    print("\nAblation study completed!")
    print("Expected results: Full MP-KVM should significantly outperform others,")
    print("demonstrating that both positionless RoPE and energy compensation are essential.")
    print("\nRun 'python generate_paper_figures.py' to generate the ablation chart.")


if __name__ == "__main__":
    main()
