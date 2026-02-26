import numpy as np
from src.inference.embedder import Embedder
from src.inference.matcher import compute_similarity, is_same_pet


def test_embedder_output_shape():
    embedder = Embedder(embedding_dim=256)
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    embedding = embedder.embed(dummy_img)
    assert embedding.shape == (256,)


def test_embedder_output_normalized():
    embedder = Embedder(embedding_dim=256)
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    embedding = embedder.embed(dummy_img)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-5


def test_compute_similarity_same_vector():
    v = np.random.rand(256).astype(np.float32)
    v = v / np.linalg.norm(v)
    assert abs(compute_similarity(v, v) - 1.0) < 1e-5


def test_is_same_pet_threshold():
    v = np.ones(256, dtype=np.float32) / np.sqrt(256)
    assert is_same_pet(v, v, threshold=0.8) == True
    assert is_same_pet(v, -v, threshold=0.8) == False
