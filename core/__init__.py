# Core package for MP-KVM
from .clustering import OnlineManifoldClustering
# use clean integration implementation
from .integration_clean import MPKVMManager, monkey_patch_attention_forward, patch_llama_attention

# Try to import layers, but handle torch import failures gracefully
try:
    from .layers import ReconstructedAttention
except ImportError as e:
    if "torch" in str(e).lower():
        # PyTorch not available, skip layers import
        print(f"Warning: Skipping layers import due to PyTorch unavailability: {e}")
        ReconstructedAttention = None
    else:
        raise

"""
Core MP-KVM modules package.
"""


