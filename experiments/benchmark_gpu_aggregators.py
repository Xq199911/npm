"""
Benchmark baseline vs optimized GPU aggregators on synthetic batches.
Saves results to results/gpu_benchmark/benchmark_results.json and CSV.
"""
from __future__ import annotations
import csv
import json
import os
import time
from typing import List, Dict, Any


def run_benchmark(batch_sizes: List[int], dims: List[int], repeats: int = 3):
    import torch
    import sys
    import os

    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core.integration_clean import MPKVMManager
    from core.integration_gpu import MPKVMGPUAggregator, MPKVMGPUAggregatorOptimized

    out_dir = os.path.join("results", "gpu_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    # choose device: prefer CUDA if available
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda:0"

    for D in dims:
        for N in batch_sizes:
            for rep in range(repeats):
                seed = 42 + rep
                torch.manual_seed(seed)
                # create tensors on chosen device
                k = torch.randn((N, D), dtype=torch.float32, device=device_str)
                v = k.clone()

                # baseline
                cpu_mgr1 = MPKVMManager(dim=D, num_layers=1)
                agg1 = MPKVMGPUAggregator(cpu_mgr1, dim=D, device=device_str, max_gpu_centroids_per_layer=256, similarity_threshold=0.5)
                t0 = time.perf_counter()
                agg1.add_kv_torch(0, k, v)
                t1 = time.perf_counter()
                cent1, counts1 = agg1.get_gpu_centroids(0)
                c1 = 0 if cent1 is None else int(cent1.shape[0])
                time_baseline = t1 - t0
                agg1.flush_all_to_cpu()

                # optimized
                cpu_mgr2 = MPKVMManager(dim=D, num_layers=1)
                agg2 = MPKVMGPUAggregatorOptimized(cpu_mgr2, dim=D, device=device_str, max_gpu_centroids_per_layer=256, similarity_threshold=0.5)
                t0 = time.perf_counter()
                agg2.add_kv_torch(0, k, v)
                t1 = time.perf_counter()
                cent2, counts2 = agg2.get_gpu_centroids(0)
                c2 = 0 if cent2 is None else int(cent2.shape[0])
                time_opt = t1 - t0
                agg2.flush_all_to_cpu()

                entry = {
                    "dim": D,
                    "batch_size": N,
                    "rep": rep,
                    "time_baseline_s": time_baseline,
                    "centroids_baseline": c1,
                    "time_optimized_s": time_opt,
                    "centroids_optimized": c2,
                    "ratio_time_opt_over_base": time_opt / (time_baseline + 1e-12),
                }
                results.append(entry)
                print(f"D={D} N={N} rep={rep} baseline={time_baseline:.4f}s opt={time_opt:.4f}s c_base={c1} c_opt={c2}")

    json_path = os.path.join(out_dir, "benchmark_results.json")
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["dim", "batch_size", "rep", "time_baseline_s", "centroids_baseline", "time_optimized_s", "centroids_optimized", "ratio_time_opt_over_base"])
        for r in results:
            writer.writerow([r["dim"], r["batch_size"], r["rep"], r["time_baseline_s"], r["centroids_baseline"], r["time_optimized_s"], r["centroids_optimized"], r["ratio_time_opt_over_base"]])

    print(f"Saved benchmark results to {json_path} and {csv_path}")
    return results


if __name__ == "__main__":
    batches = [128, 1024, 4096]
    dims = [64, 256]
    run_benchmark(batches, dims, repeats=3)


