def test_mpkvm_gpu_aggregator_optimized_basic():
    import torch

    from core.integration_clean import MPKVMManager
    from core.integration_gpu import MPKVMGPUAggregatorOptimized

    cpu_mgr = MPKVMManager(dim=4, num_layers=2)
    agg = MPKVMGPUAggregatorOptimized(cpu_mgr, dim=4, device="cpu", max_gpu_centroids_per_layer=4, similarity_threshold=0.5)

    k = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    v = k.clone()
    agg.add_kv_torch(0, k, v)
    centroids, counts = agg.get_gpu_centroids(0)
    assert centroids is not None and counts is not None
    assert centroids.shape[0] <= agg.max_gpu_centroids_per_layer
    agg.flush_all_to_cpu()


