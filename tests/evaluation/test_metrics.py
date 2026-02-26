import numpy as np
from src.evaluation.metrics import compute_eer, compute_roc

def test_compute_eer_perfect_separation():
    similarities = np.array([0.9, 0.95, 0.1, 0.05])
    labels = np.array([1, 1, 0, 0])  # 1=same, 0=different
    eer, threshold = compute_eer(similarities, labels)
    assert eer < 0.1

def test_compute_eer_returns_tuple():
    similarities = np.random.rand(100)
    labels = (np.random.rand(100) > 0.5).astype(int)
    result = compute_eer(similarities, labels)
    assert len(result) == 2

def test_compute_roc_returns_arrays():
    similarities = np.random.rand(50)
    labels = (np.random.rand(50) > 0.5).astype(int)
    fpr, tpr, thresholds = compute_roc(similarities, labels)
    assert len(fpr) == len(tpr)
    assert np.all(fpr >= 0) and np.all(fpr <= 1)
