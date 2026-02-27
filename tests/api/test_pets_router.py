import pytest
import io
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_router_globals():
    import api.routers.pets as pets_module
    pets_module._supabase = None
    pets_module._embedder = None
    yield
    pets_module._supabase = None
    pets_module._embedder = None


@pytest.fixture
def client():
    with patch("api.routers.pets.SupabaseService") as mock_sb, \
         patch("api.routers.pets.EmbeddingService") as mock_emb:

        mock_emb_instance = MagicMock()
        mock_emb_instance.image_bytes_to_embedding.return_value = (
            np.ones(256, dtype=np.float32) / np.sqrt(256)
        )
        mock_emb.return_value = mock_emb_instance

        mock_sb_instance = MagicMock()
        mock_sb_instance.register_pet.return_value = "pet-uuid-123"
        mock_sb_instance.save_embedding.return_value = "emb-uuid-456"
        mock_sb_instance.upload_image.return_value = "raw/pet-uuid-123/001.jpg"
        mock_sb_instance.find_matching_pet.return_value = {
            "pet_id": "pet-uuid-123",
            "embedding_id": "emb-uuid-456",
            "similarity": 0.94,
        }
        mock_sb_instance.get_pet.return_value = {
            "id": "pet-uuid-123",
            "name": "小白",
            "species": "dog",
        }
        mock_sb.return_value = mock_sb_instance

        from api.main import app
        yield TestClient(app)


def test_register_pet(client):
    image_bytes = b"fake-image-data"
    response = client.post(
        "/api/v1/pets/register",
        data={"name": "小白", "species": "dog"},
        files={"image": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "pet_id" in data
    assert data["name"] == "小白"


def test_verify_pet(client):
    image_bytes = b"fake-image-data"
    response = client.post(
        "/api/v1/pets/verify",
        files={"image": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "matched" in data
    assert "similarity" in data


def test_get_pet(client):
    response = client.get("/api/v1/pets/pet-uuid-123")
    assert response.status_code == 200
