import numpy as np
import tensorflow as tf
from src.models.embedding_head import build_embedding_head

def test_embedding_head_output_shape():
    head = build_embedding_head(input_dim=1280, embedding_dim=256)
    dummy_input = np.random.rand(4, 1280).astype(np.float32)
    output = head(dummy_input, training=False)
    assert output.shape == (4, 256)

def test_embedding_head_l2_normalized():
    head = build_embedding_head(input_dim=1280, embedding_dim=256)
    dummy_input = np.random.rand(4, 1280).astype(np.float32)
    output = head(dummy_input, training=False)
    norms = np.linalg.norm(output.numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
