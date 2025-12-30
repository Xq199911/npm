import numpy as np

from core.layers import _make_positionless_numpy, scaled_dot_product_attention, reconstruct_with_centroids


def test_make_positionless_numpy_pairs():
    # build a simple key with pair entries (x,y) where magnitude r = sqrt(x^2+y^2)
    keys = np.array([[3.0, 4.0, 1.0, 0.0]], dtype=np.float32)  # two pairs: (3,4)->5 and (1,0)->1
    pl = _make_positionless_numpy(keys)
    # first pair magnitude should be 5, second pair magnitude 1, second components zero
    assert np.allclose(pl[0, 0], 5.0, atol=1e-6)
    assert np.allclose(pl[0, 1], 0.0, atol=1e-6)
    assert np.allclose(pl[0, 2], 1.0, atol=1e-6)
    assert np.allclose(pl[0, 3], 0.0, atol=1e-6)


def test_score_bias_changes_attention():
    q = np.array([[1.0, 0.0]], dtype=np.float32)  # (1,2)
    k = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # (2,2)
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # without bias, query dot keys -> scores [1,0]
    out_no_bias = scaled_dot_product_attention(q, k, v)
    # with strong positive bias on second key, attention should shift
    bias = np.array([0.0, 10.0], dtype=float)
    out_with_bias = scaled_dot_product_attention(q, k, v, score_bias=bias)
    assert not np.allclose(out_no_bias, out_with_bias)


def test_reconstruct_with_centroids_bias_effect():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    k = np.array([[1.0, 0.0]], dtype=np.float32)
    v = np.array([[1.0, 0.0]], dtype=np.float32)
    cent_k = np.array([[0.0, 1.0]], dtype=np.float32)
    cent_v = np.array([[0.0, 1.0]], dtype=np.float32)
    # with centroid weighting None, should be neutral
    out_default = reconstruct_with_centroids(q, k, v, centroids_k=cent_k, centroids_v=cent_v, centroid_weighting=None)
    # with large centroid weight, output should shift towards centroid value
    out_weighted = reconstruct_with_centroids(q, k, v, centroids_k=cent_k, centroids_v=cent_v, centroid_weighting=np.array([100.0], dtype=float))
    assert not np.allclose(out_default, out_weighted)


