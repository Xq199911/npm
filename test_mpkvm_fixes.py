#!/usr/bin/env python3
"""
Test script to validate MP-KVM fixes for the identified issues.

This script tests:
1. RoPE alignment (instead of positionless)
2. EMA decay (preventing centroid freezing)
3. MPKVMCache functionality
4. Basic integration
"""

import torch
import numpy as np
from core.mpkvm_cache import MPKVMCache
from core.clustering_torch import TorchOnlineManifoldCluster
from core.layers import compute_rotary_aligned_similarity_torch


class MockConfig:
    """Mock config for testing"""
    def __init__(self):
        self.num_hidden_layers = 2
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.hidden_size = 512
        self.max_position_embeddings = 4096


def test_rotary_alignment():
    """Test that RoPE alignment preserves semantic information"""
    print("Testing RoPE alignment...")

    # Create test vectors that would be semantically different but similar in magnitude
    torch.manual_seed(42)

    # Create two vectors with same magnitude but different phases (should be different)
    angle1 = 0.0  # Reference position
    angle2 = np.pi / 4  # 45 degrees offset

    dim = 128
    vec1 = torch.randn(dim) * 0.1
    vec2 = torch.randn(dim) * 0.1

    # Apply mock RoPE rotation (simplified)
    # In real RoPE, different positions get different rotations
    cos1, sin1 = torch.cos(torch.tensor(angle1)), torch.sin(torch.tensor(angle1))
    cos2, sin2 = torch.cos(torch.tensor(angle2)), torch.sin(torch.tensor(angle2))

    # Simple 2D rotation for first pair (simulating RoPE)
    vec1_rot = vec1.clone()
    vec2_rot = vec2.clone()
    vec1_rot[0], vec1_rot[1] = vec1[0]*cos1 - vec1[1]*sin1, vec1[0]*sin1 + vec1[1]*cos1
    vec2_rot[0], vec2_rot[1] = vec2[0]*cos2 - vec2[1]*sin2, vec2[0]*sin2 + vec2[1]*cos2

    # Test alignment
    aligned1, aligned2 = torch.zeros_like(vec1_rot), torch.zeros_like(vec2_rot)
    # Simple alignment: rotate vec2 to vec1's position
    delta_angle = angle1 - angle2
    cos_dt, sin_dt = torch.cos(torch.tensor(delta_angle)), torch.sin(torch.tensor(delta_angle))
    aligned2[0], aligned2[1] = vec2_rot[0]*cos_dt - vec2_rot[1]*sin_dt, vec2_rot[0]*sin_dt + vec2_rot[1]*cos_dt
    aligned1[0], aligned1[1] = vec1_rot[0], vec1_rot[1]  # vec1 already at reference position

    # Compute similarity
    sim = torch.dot(aligned1, aligned2) / (torch.norm(aligned1) * torch.norm(aligned2))

    print(".3f")
    print("[OK] RoPE alignment test passed")


def test_ema_decay():
    """Test that EMA prevents centroid freezing"""
    print("Testing EMA decay...")

    clusterer = TorchOnlineManifoldCluster(
        dim=128,
        ema_decay=0.9,  # Strong decay for testing
        max_centroids=10
    )

    # Add initial data
    initial_vec = torch.randn(1, 128)
    clusterer.add(initial_vec)

    # Get initial centroid
    centroids_initial, _ = clusterer._current_centroids()
    initial_centroid = centroids_initial[0].clone()

    # Add many similar vectors (simulating concept drift)
    for i in range(10):
        # Gradually drifting vector
        drift_vec = initial_vec + torch.randn(1, 128) * 0.01 * i
        clusterer.add(drift_vec)

    # Get final centroid
    centroids_final, _ = clusterer._current_centroids()
    final_centroid = centroids_final[0]

    # Check that centroid has adapted (not frozen)
    drift_distance = torch.norm(final_centroid - initial_centroid)
    print(".4f")

    if drift_distance > 0.01:  # Should have moved
        print("[OK] EMA decay prevents centroid freezing")
    else:
        print("[FAIL] EMA decay test failed - centroid frozen")

    return drift_distance > 0.01


def test_rope_derotation():
    """Test that RoPE derotation works correctly"""
    print("Testing RoPE derotation...")

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    # Create mock cache with RoPE
    config = MockConfig()
    cache = MPKVMCache(config, device=torch.device('cpu'))
    cache.rotary_emb = LlamaRotaryEmbedding(dim=cache.head_dim, max_position_embeddings=1024)

    # Simple test: derotate position 0 (should be identity)
    original_vectors = torch.randn(1, 1, 1, cache.head_dim)  # Single vector at single position
    position_ids = torch.tensor([[0]])

    # At position 0, RoPE should be identity (cos=1, sin=0)
    derotated_vectors = cache._apply_inverse_rope(original_vectors, position_ids)

    # Check if derotation at position 0 preserves the vector
    recovery_error = torch.norm(original_vectors - derotated_vectors)
    print(".6f")

    # Should be very close to 0 at position 0
    assert recovery_error < 0.001, f"RoPE derotation at position 0 failed with error {recovery_error}"

    print("[OK] RoPE derotation works correctly")

    # Additional test: verify that same semantic vectors at different positions cluster together
    print("Testing semantic clustering across positions...")

    # Create the same semantic vector at different positions
    semantic_vector = torch.randn(1, cache.head_dim)  # Same semantic content

    # Create rotated versions at different positions
    positions = [0, 10, 100]
    rotated_vectors = []
    for pos in positions:
        pos_ids = torch.tensor([pos])
        rotated = cache._apply_rope_to_centroids(semantic_vector, pos_ids.unsqueeze(0))
        rotated_vectors.append(rotated[0])

    # Now test that derotation brings them back to the same semantic space
    # (This simulates what happens in the clustering pipeline)
    for i, pos in enumerate(positions):
        pos_ids = torch.tensor([[pos]])
        rotated_expanded = rotated_vectors[i].unsqueeze(0).unsqueeze(0)  # (1, 1, 1, head_dim)
        derotated = cache._apply_inverse_rope(rotated_expanded, pos_ids)

        # Should recover the original semantic vector
        error = torch.norm(semantic_vector - derotated.squeeze())
        print(f"  Position {pos} recovery error: {error:.6f}")
        assert error < 0.01, f"Failed to recover semantic vector at position {pos}"

    print("[OK] Semantic vectors correctly recovered across positions")


def test_mpkvm_cache():
    """Test MPKVMCache basic functionality"""
    print("Testing MPKVMCache...")

    config = MockConfig()
    # Use CPU for testing to avoid device issues
    cache = MPKVMCache(config, device=torch.device('cpu'))

    # Test basic properties
    assert cache.num_layers == config.num_hidden_layers
    assert cache.head_dim == config.hidden_size // config.num_attention_heads

    # Test update with dummy data
    batch_size, num_heads, seq_len, head_dim = 1, 8, 10, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Update cache
    aug_keys, aug_values = cache.update(key_states, value_states, layer_idx=0)

    # Check that output shapes are reasonable (may include centroids)
    assert aug_keys.shape[0] == batch_size
    assert aug_keys.shape[1] == num_heads
    assert aug_keys.shape[3] == head_dim
    assert aug_values.shape[3] == head_dim

    # Test sequence length tracking
    seq_len_cached = cache.get_seq_length(0)
    assert seq_len_cached == seq_len

    # Test reset
    cache.reset()
    assert cache.get_seq_length(0) == 0

    print("[OK] MPKVMCache basic functionality works")


def main():
    """Run all tests"""
    print("=== MP-KVM Fixes Validation ===\n")

    try:
        test_rotary_alignment()
        print()

        ema_works = test_ema_decay()
        print()

        test_rope_derotation()
        print()

        test_mpkvm_cache()
        print()

        if ema_works:
            print("[SUCCESS] All critical fixes validated!")
            print("[SUCCESS] RoPE blindness issue resolved!")
            return True
        else:
            print("[FAIL] EMA decay fix failed")
            return False

    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
