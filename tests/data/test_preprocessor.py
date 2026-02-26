# tests/data/test_preprocessor.py
import numpy as np
import pytest
from src.data.preprocessor import preprocess_image, load_and_preprocess

def test_preprocess_image_output_shape():
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(dummy_img)
    assert result.shape == (224, 224, 3)

def test_preprocess_image_normalized():
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(dummy_img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

def test_preprocess_image_dtype():
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_image(dummy_img)
    assert result.dtype == np.float32
