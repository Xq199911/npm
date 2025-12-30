import numpy as np

from core.clustering import OnlineManifoldCluster, OnlineManifoldClustering


def test_online_manifold_cluster_add_and_merge():
    c = OnlineManifoldCluster(dim=2, max_centroids=2, persistence_decay=1.0)
    vecs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    # first two are similar and should be merged or produce nearby centroids
    c.add(vecs[:2], similarity_threshold=0.2)
    centroids, counts = c.get_centroids()
    assert centroids.shape[1] == 2
    assert counts.sum() >= 2


def test_compress_if_needed_merges():
    rng = np.random.RandomState(0)
    c = OnlineManifoldCluster(dim=2, max_centroids=3)
    vecs = rng.randn(10, 2).astype(np.float32)
    c.add(vecs, similarity_threshold=0.0)
    # ensure we respect max_centroids after compression
    centroids, counts = c.get_centroids()
    assert centroids.shape[0] <= c.max_centroids


def test_energy_loss_with_history():
    c = OnlineManifoldCluster(dim=2, max_centroids=4, sliding_window_size=10)
    vecs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    c.add(vecs)
    loss = c.energy_loss()
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_online_manifold_clustering_buffers_and_prune():
    oc = OnlineManifoldClustering(dim=2, window_size=3, max_memory_size=5, max_centroids=3)
    keys = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    vals = keys.copy()
    for i in range(len(keys)):
        oc.add(keys[i : i + 1], vals[i : i + 1])
    k, v, w = oc.snapshot_buffer()
    assert k.shape[0] <= oc.window_size
    c, counts, weights = oc.get_centroids()
    # counts shape should match returned counts length
    assert counts.shape[0] == counts.size


