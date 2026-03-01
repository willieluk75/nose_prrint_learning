import pytest
import io
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from api.services.embedding_service import EmbeddingService

register_heif_opener()


@pytest.fixture
def embedder():
    return EmbeddingService(weights_path=None, embedding_dim=256)


def test_jpeg_conversion(embedder):
    """Test that JPEG images still work"""
    # Create a simple RGB image as JPEG
    img_pil = Image.new('RGB', (100, 100), color='red')
    jpeg_bytes = io.BytesIO()
    img_pil.save(jpeg_bytes, format='JPEG')
    result = embedder.image_bytes_to_embedding(jpeg_bytes.getvalue())
    assert result.shape == (256,)
    assert np.all(np.isfinite(result))


def test_heic_conversion(embedder):
    """Test that HEIC images are auto-converted"""
    # Create a simple RGB image and save as HEIF
    img_pil = Image.new('RGB', (100, 100), color='blue')
    heif_bytes = io.BytesIO()
    img_pil.save(heif_bytes, format='HEIF')

    result = embedder.image_bytes_to_embedding(heif_bytes.getvalue())
    assert result.shape == (256,)
    assert np.all(np.isfinite(result))


def test_invalid_image(embedder):
    """Test that invalid data raises ValueError"""
    with pytest.raises(ValueError, match="Cannot decode"):
        embedder.image_bytes_to_embedding(b"not-an-image")
