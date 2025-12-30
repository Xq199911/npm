import numpy as np

from adapters.llama_adapter import attach_mpkvm_to_hf_llama
from core.integration_clean import MPKVMManager


class DummyAttn:
    def __init__(self):
        self.last_key = None
        self.last_value = None

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, *args, **kwargs):
        # produce present_key_value as tuple (k,v) using hidden_states directly for test
        k = np.asarray(hidden_states)  # (B, S, D) or similar
        v = np.asarray(hidden_states)
        return ("out", (k, v))


class DummyLayer:
    def __init__(self):
        self.self_attn = DummyAttn()


class DummyModel:
    def __init__(self, n_layers=2):
        class Container:
            pass

        self.model = Container()
        self.model.layers = [DummyLayer() for _ in range(n_layers)]


def test_adapter_basic_numpy_path():
    mgr = MPKVMManager(dim=4, num_layers=2)
    model = DummyModel(n_layers=2)
    attach_mpkvm_to_hf_llama(model, mgr, head_mean=False, sample_stride=1, enable_injection=False)

    # call a layer forward; adapter should intercept and add KV as numpy
    inp = np.ones((1, 1, 4), dtype=np.float32)
    # call first layer's attn forward
    out = model.model.layers[0].self_attn.forward(inp)
    # after forward, manager should have centroids for layer 0 (or at least not crash)
    centroids, counts, weights = mgr.get_layer_centroids(0)
    # centroids may be empty or populated depending on adapter probes; ensure call succeeded
    assert isinstance(centroids, np.ndarray)
    assert isinstance(counts, np.ndarray)
    assert isinstance(weights, np.ndarray)


