"""
Minimal experiment runner for MP-KVM PoC.

This script creates synthetic KV vectors, feeds them into MPKVMManager and
produces a UMAP plot and energy loss numbers for inspection.
"""
from __future__ import annotations
import argparse
import numpy as np
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import MPKVMManager
from core.clustering import OnlineManifoldClustering
from analysis.energy_loss import reconstruction_loss, assign_by_cosine
from analysis.manifold_viz import visualize_kv_and_centroids


def run_synthetic(cfg_path: str, out_dir: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dim = int(cfg["model"]["dim"])
    maxc = int(cfg["run"]["max_centroids_per_layer"])
    sim_thresh = float(cfg["run"]["similarity_threshold"])
    rng = np.random.RandomState(int(cfg["run"]["seed"]))

    # synthetic: create N points from several gaussian clusters to emulate semantic clusters
    centers = rng.normal(size=(8, dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    points = []
    for c in centers:
        points.append(c + 0.01 * rng.normal(size=(200, dim)))
    points = np.vstack(points).astype(np.float32)

    manager = MPKVMManager(dim=dim, num_layers=1, max_centroids=maxc, metric="cosine", window_size=500)
    # add in small batches to simulate streaming
    batch = 64
    for i in range(0, points.shape[0], batch):
        b = points[i : i + batch]
        manager.add_kv(0, b, b)

    centroids, counts, weights = manager.get_layer_centroids(0)
    os.makedirs(out_dir, exist_ok=True)
    vis_path = os.path.join(out_dir, "manifold.png")
    visualize_kv_and_centroids(points, centroids, save_path=vis_path)
    print(f"Saved manifold visualization to {vis_path}")
    # Note: energy_loss calculation skipped as it's not implemented in MPKVMManager
    # losses = manager.energy_loss(lambda_diversity=float(cfg["run"].get("lambda_diversity", 0.0)))
    # for k, v in losses.items():
    #     print(f"Layer {k} energy loss: {v:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/configs/default.yaml")
    p.add_argument("--out", default="results/synthetic")
    args = p.parse_args()
    run_synthetic(args.config, args.out)


if __name__ == "__main__":
    main()

def poc_run(seed: int = 0, n_tokens: int = 2048, dim: int = 128, compress_batch: int = 512):
    rng = np.random.RandomState(seed)

    # Create synthetic data with semantic clusters that are actually clusterable
    # Use multiple Gaussian clusters with reasonable separation
    n_clusters = 6
    cluster_centers = rng.normal(0, 2, (n_clusters, dim)).astype(np.float32)

    # Normalize centers to unit sphere for better cosine similarity
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    keys = []
    cluster_assignments = []

    # Generate tokens from clusters with some temporal structure
    tokens_per_cluster = n_tokens // n_clusters

    for cluster_idx in range(n_clusters):
        # Create cluster points with controlled spread
        cluster_points = cluster_centers[cluster_idx] + rng.normal(0, 0.3, (tokens_per_cluster, dim))
        # Normalize to improve clustering quality
        cluster_points = cluster_points / np.linalg.norm(cluster_points, axis=1, keepdims=True)
        keys.append(cluster_points.astype(np.float32))
        cluster_assignments.extend([cluster_idx] * tokens_per_cluster)

    # Handle remainder tokens
    remainder = n_tokens % n_clusters
    if remainder > 0:
        extra_points = cluster_centers[0] + rng.normal(0, 0.3, (remainder, dim))
        extra_points = extra_points / np.linalg.norm(extra_points, axis=1, keepdims=True)
        keys.append(extra_points.astype(np.float32))
        cluster_assignments.extend([0] * remainder)

    keys = np.concatenate(keys, axis=0)
    vals = keys.copy()  # Values follow keys

    # Create attention weights that favor recent tokens (simulating real attention patterns)
    attn_scores = np.ones(n_tokens, dtype=float)
    # Recent tokens get higher attention (exponential decay)
    for i in range(n_tokens):
        attn_scores[i] = np.exp(-(n_tokens - i - 1) / (n_tokens / 4))
    attn_scores /= attn_scores.sum()

    # Track centroids evolution over time
    m = OnlineManifoldClustering(dim=dim, window_size=1024, max_centroids=128, metric="cosine", similarity_threshold=0.7)

    centroids_history = []
    # Feed in chunks and track centroid evolution
    snapshot_points = [0, n_tokens//4, n_tokens//2, 3*n_tokens//4, n_tokens - compress_batch]

    for i in range(0, n_tokens, compress_batch):
        k_batch = keys[i : i + compress_batch]
        v_batch = vals[i : i + compress_batch]
        w_batch = attn_scores[i : i + compress_batch]
        m.add(k_batch, v_batch, w_batch)

        # Snapshot centroids at key points
        if any(abs(i - point) < compress_batch for point in snapshot_points):
            centroids, counts, weights = m.get_centroids()
            centroids_history.append({
                "step": i,
                "centroids": centroids.copy() if centroids is not None else None,
                "theme": f"cluster_{i // (n_tokens // n_clusters)}"
            })

    centroids, counts, weights = m.get_centroids()

    # Enhanced visualization data
    cluster_size = n_tokens // n_clusters
    phase_boundaries = [i * cluster_size for i in range(n_clusters + 1)]
    themes = [f"cluster_{i}" for i in range(n_clusters)]

    vis_data = {
        "keys": keys,
        "centroids": centroids,
        "phase_boundaries": phase_boundaries,
        "themes": themes,
        "centroids_history": centroids_history
    }

    # Compute assignments and reconstruction loss
    assignments = assign_by_cosine(keys, centroids) if centroids.shape[0] > 0 else np.zeros((keys.shape[0],), dtype=int)
    loss_centroid = reconstruction_loss(keys, assignments, centroids) if centroids.shape[0] > 0 else float("inf")

    # Baseline: random sampling of same number as centroids
    if centroids.shape[0] > 0:
        rng_idx = rng.choice(np.arange(keys.shape[0]), size=centroids.shape[0], replace=False)
        random_centroids = keys[rng_idx]
        assignments_rand = assign_by_cosine(keys, random_centroids)
        loss_rand = reconstruction_loss(keys, assignments_rand, random_centroids)
    else:
        loss_rand = float("inf")

    # Sanity check: verify reconstruction quality
    if centroids.shape[0] > 0:
        # Reconstruct original vectors using centroids
        reconstructed = centroids[assignments]
        # Compute cosine similarity between original and reconstructed
        original_norm = keys / np.linalg.norm(keys, axis=1, keepdims=True)
        reconstructed_norm = reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)
        cosine_similarities = np.sum(original_norm * reconstructed_norm, axis=1)
        avg_cosine_similarity = np.mean(cosine_similarities)
        # CRITICAL: Check if reconstruction is reasonable (cosine similarity > 0.8)
        # This ensures MP-KVM maintains high-fidelity token representations
        if avg_cosine_similarity < 0.8:
            raise RuntimeError(
                f"[MPKVM][ERROR] Sanity check FAILED: Average cosine similarity between original and "
                f"reconstructed vectors is only {avg_cosine_similarity:.3f} (required > 0.8). "
                f"This indicates fundamental issues with the manifold clustering or reconstruction logic. "
                f"MP-KVM compression is not working correctly."
            )
        print(f" Sanity check passed: Average reconstruction similarity = {avg_cosine_similarity:.3f}")
    else:
        raise RuntimeError("[MPKVM][ERROR] No centroids generated - this indicates a fundamental failure in the clustering algorithm!")

    return {
        "n_tokens": n_tokens,
        "dim": dim,
        "n_centroids": centroids.shape[0],
        "loss_centroid": loss_centroid,
        "loss_random": loss_rand,
        "centroids": centroids,
        "keys": keys,
        "sanity_check_cosine_sim": avg_cosine_similarity if centroids.shape[0] > 0 else 0.0,
        "visualization_data": vis_data,  # Enhanced data for theme-transition visualization
    }


def poc_main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_tokens", type=int, default=2048)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=str, default="poc_out.png")
    args = p.parse_args()

    res = poc_run(seed=args.seed, n_tokens=args.n_tokens, dim=args.dim)
    print("Results:")
    print(f"  tokens: {res['n_tokens']}  dim: {res['dim']}  centroids: {res['n_centroids']}")
    print(f"  loss (centroid): {res['loss_centroid']:.4f}")
    print(f"  loss (random baseline): {res['loss_random']:.4f}")

    if args.plot:
        os.makedirs("figures", exist_ok=True)
        visualize_kv_and_centroids(res["keys"][:1024], res["centroids"], save_path=os.path.join("figures", args.out))
        print(f"Saved manifold plot to figures/{args.out}")


if __name__ == "__main__":
    main()