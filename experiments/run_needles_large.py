"""
Run larger Needles experiments (total_tokens=20000) on selected configs and save results.
"""
from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List
import torch
from core.integration_clean import MPKVMManager
from core.integration_gpu import MPKVMGPUAggregatorOptimized
from data.needles.run_niah import evaluate_recall


def run_cfg(cfg: Dict[str, Any]):
    # Use real model data (synthetic data removed)
    import torch
    from run_real_model_experiment import setup_model_and_tokenizer, RealModelKVExtractor, create_long_context_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model_and_tokenizer("model/Llama-3.1-8B-Instruct", device)
    extractor = RealModelKVExtractor(model, tokenizer, device=device)

    # Create long context and extract KV vectors
    context_text = create_long_context_text()
    keys, values = extractor.extract_kv_from_text(context_text, max_length=min(cfg["total_tokens"], 2048))

    # Adjust to target length if needed
    if keys.shape[0] < cfg["total_tokens"]:
        pad_size = cfg["total_tokens"] - keys.shape[0]
        keys = np.pad(keys, ((0, pad_size), (0, 0)), mode='constant')
        values = np.pad(values, ((0, pad_size), (0, 0)), mode='constant')

    # Create needles from early part
    early_portion = int(keys.shape[0] * 0.3)
    n_needles = min(cfg["n_needles"], int(early_portion * 0.05))
    np.random.seed(cfg.get("seed", 0))
    needle_indices = np.random.choice(early_portion, size=n_needles, replace=False)
    needles = keys[needle_indices].copy()
    cpu_mgr = MPKVMManager(dim=cfg["dim"], num_layers=1, **{"cluster_kwargs": cfg["cluster_kwargs"]})
    agg = MPKVMGPUAggregatorOptimized(cpu_mgr, dim=cfg["dim"], device=cfg["device"], max_gpu_centroids_per_layer=cfg.get("max_gpu_centroids_per_layer", 512), similarity_threshold=cfg.get("agg_similarity", 0.7))
    start = time.perf_counter()
    bs = cfg["batch_size"]
    for i in range(0, keys.shape[0], bs):
        kt = torch.from_numpy(keys[i:i+bs]).to(cfg["device"])
        vt = torch.from_numpy(values[i:i+bs]).to(cfg["device"])
        agg.add_kv_torch(0, kt, vt)
        if ((i // bs) + 1) % (cfg["flush_interval"] // bs) == 0:
            agg.flush_all_to_cpu()
    agg.flush_all_to_cpu()
    total_time = time.perf_counter() - start
    centroids, counts, weights = cpu_mgr.get_layer_centroids(0)
    recall = evaluate_recall(centroids, needles, threshold=cfg.get("recall_threshold", 0.85))
    return {"cfg": cfg, "total_time_s": total_time, "num_centroids": int(centroids.shape[0]) if centroids is not None else 0, "recall": float(recall)}


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = {"total_tokens": 20000, "dim": 64, "n_clusters": 20, "cluster_std": 0.5, "n_needles": 50, "batch_size": 32, "flush_interval": 256, "device": device, "agg_similarity": 0.7, "recall_threshold": 0.85, "max_gpu_centroids_per_layer": 512}
    configs: List[Dict[str, Any]] = []
    configs.append({**base, "cluster_kwargs": {"init_preserve_first_n": None, "similarity_threshold": 0.4, "min_merge_similarity": None, "window_size": 50}})
    configs.append({**base, "cluster_kwargs": {"init_preserve_first_n": 2, "similarity_threshold": 0.4, "min_merge_similarity": None, "window_size": 50}})
    configs.append({**base, "cluster_kwargs": {"init_preserve_first_n": 10, "similarity_threshold": 0.4, "min_merge_similarity": None, "window_size": 50}})

    out = []
    out_dir = os.path.join("experiments", "needles_large_out")
    os.makedirs(out_dir, exist_ok=True)
    for cfg in configs:
        print("Running large:", cfg["cluster_kwargs"])
        res = run_cfg(cfg)
        out.append(res)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()


