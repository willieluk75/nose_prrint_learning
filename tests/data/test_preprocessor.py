# tests/data/test_preprocessor.py
import numpy as np
import pytest
from PIL import Image
from pillow_heif import register_heif_opener
from src.data.preprocessor import load_and_preprocess, preprocess_image

register_heif_opener()

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

def test_jpeg_image(tmp_path):
    """Test that JPEG images still work"""
    # Create a simple JPEG image
    img = Image.new('RGB', (100, 100), color='red')
    jpeg_path = tmp_path / "test.jpg"
    img.save(jpeg_path, format='JPEG')

    result = load_and_preprocess(str(jpeg_path))
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0

def test_heic_image(tmp_path):
    """Test that HEIC images are supported"""
    # Create a HEIF image
    img = Image.new('RGB', (100, 100), color='blue')
    heif_path = tmp_path / "test.heif"
    img.save(heif_path, format='HEIF')

    result = load_and_preprocess(str(heif_path))
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32

def test_file_not_found():
    """Test that missing file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_and_preprocess("nonexistent.jpg")

def test_invalid_image(tmp_path):
    """Test that invalid image raises ValueError"""
    # Create a text file instead of image
    invalid_path = tmp_path / "not_an_image.txt"
    invalid_path.write_text("not an image")

    with pytest.raises(ValueError, match="Cannot read image"):
        load_and_preprocess(str(invalid_path))
