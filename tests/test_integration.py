# tests/test_integration.py
"""
端對端整合測試：ML pipeline + API layer（全部 mock Supabase）
"""
import numpy as np
import pytest
import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_router_globals():
    import api.routers.pets as pets_module
    pets_module._supabase = None
    pets_module._embedder = None
    yield
    pets_module._supabase = None
    pets_module._embedder = None


@pytest.fixture
def full_client():
    """完整 mock：Supabase + ML model（避免下載 weights）"""
    with patch("api.routers.pets.get_supabase") as mock_get_sb, \
         patch("api.routers.pets.get_embedder") as mock_get_emb:

        mock_emb = MagicMock()
        mock_emb.image_bytes_to_embedding.return_value = (
            np.ones(256, dtype=np.float32) / np.sqrt(256)
        )
        mock_get_emb.return_value = mock_emb

        mock_sb = MagicMock()
        mock_sb.register_pet.return_value = "integrated-pet-uuid"
        mock_sb.save_embedding.return_value = "integrated-emb-uuid"
        mock_sb.upload_image.return_value = "raw/integrated-pet-uuid/001.jpg"
        mock_sb.find_matching_pet.return_value = {
            "pet_id": "integrated-pet-uuid",
            "embedding_id": "integrated-emb-uuid",
            "similarity": 0.96,
        }
        mock_sb.get_pet.return_value = {
            "id": "integrated-pet-uuid",
            "name": "小白",
            "species": "dog",
        }
        mock_get_sb.return_value = mock_sb

        from api.main import app
        yield TestClient(app)


def test_register_then_verify(full_client):
    """完整流程：登記 → 驗證"""
    # 1. 登記新寵物
    image_bytes = b"fake-nose-image"
    register_response = full_client.post(
        "/api/v1/pets/register",
        data={"name": "小白", "species": "dog"},
        files={"image": ("nose.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert register_response.status_code == 200
    pet_id = register_response.json()["pet_id"]
    assert pet_id == "integrated-pet-uuid"

    # 2. 驗證身份
    verify_response = full_client.post(
        "/api/v1/pets/verify",
        files={"image": ("nose2.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert verify_response.status_code == 200
    verify_data = verify_response.json()
    assert verify_data["matched"] == True
    assert verify_data["similarity"] > 0.85
    assert verify_data["pet_name"] == "小白"


def test_health_endpoint(full_client):
    response = full_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_add_embedding(full_client):
    """為已登記寵物新增鼻紋樣本"""
    image_bytes = b"another-nose-image"
    response = full_client.post(
        "/api/v1/pets/integrated-pet-uuid/embeddings",
        files={"image": ("nose3.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    assert "embedding_id" in response.json()
