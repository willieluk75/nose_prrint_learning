import numpy as np
from src.data.augmentor import augment_image

def test_augment_output_shape():
    img = np.random.rand(224, 224, 3).astype(np.float32)
    result = augment_image(img)
    assert result.shape == (224, 224, 3)

def test_augment_output_range():
    img = np.random.rand(224, 224, 3).astype(np.float32)
    result = augment_image(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

def test_augment_produces_variation():
    img = np.ones((224, 224, 3), dtype=np.float32) * 0.5
    results = [augment_image(img) for _ in range(10)]
    all_same = all(np.allclose(results[0], r) for r in results[1:])
    assert not all_same
