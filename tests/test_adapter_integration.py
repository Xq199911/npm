import numpy as np
from types import SimpleNamespace

from adapters.llama_adapter import attach_mpkvm_to_hf_llama
from core.integration_clean import MPKVMManager
from core.clustering import OnlineManifoldClustering


class MockAttn:
    def forward(self, hidden_states, *args, **kwargs):
        # produce present_key_value_states as (k, v) where k/v shape -> (B, S, H, D)
        B, S, D = hidden_states.shape
        k = np.tile(hidden_states[..., None, :], (1, 1, 1, 1)).astype(np.float32)
        v = k.copy()
        return (hidden_states, (k, v))


class Layer:
    def __init__(self):
        self.attn = MockAttn()


class DummyModel:
    def __init__(self):
        self.model = SimpleNamespace(layers=[Layer()])


def test_adapter_attaches_and_captures_kv():
    model = DummyModel()
    # manager with one layer
    manager = MPKVMManager(dim=16, num_layers=1, cluster_kwargs={})

    cluster_kwargs = {"init_preserve_first_n": 2, "max_centroids": 10, "window_size": 1000}
    # attach adapter and pass cluster kwargs
    attach_mpkvm_to_hf_llama(model, manager, cluster_kwargs=cluster_kwargs)

    # call forward with synthetic hidden states
    hs = np.random.randn(1, 4, 16).astype(np.float32)
    out = model.model.layers[0].attn.forward(hs)

    # after forward, manager should have per-layer cluster instance initialized with our kwargs
    assert 0 in manager.layers
    layer_cluster = manager.layers[0]
    assert isinstance(layer_cluster, OnlineManifoldClustering)
    # init_preserve_first_n should be reflected
    assert getattr(layer_cluster, "init_preserve_first_n", None) == 2
    # keys_buffer should have received entries
    kb, vb, wb = layer_cluster.snapshot_buffer()
    assert kb.shape[0] > 0


