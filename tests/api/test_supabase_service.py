import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from api.services.supabase_service import SupabaseService


@pytest.fixture
def mock_supabase_service():
    with patch("api.services.supabase_service.create_client") as mock_client:
        mock_client.return_value = MagicMock()
        service = SupabaseService(url="https://mock.supabase.co", key="mock-key")
        service.client = MagicMock()
        yield service


def test_register_pet_returns_pet_id(mock_supabase_service):
    mock_supabase_service.client.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "test-uuid-123"}
    ]
    pet_id = mock_supabase_service.register_pet(
        name="小白",
        species="dog",
        owner_id="owner-uuid",
    )
    assert pet_id == "test-uuid-123"


def test_save_embedding_calls_insert(mock_supabase_service):
    mock_supabase_service.client.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "emb-uuid"}
    ]
    embedding = np.random.rand(256).astype(np.float32)
    emb_id = mock_supabase_service.save_embedding(
        pet_id="pet-uuid",
        embedding=embedding,
        image_url="pet-nose-images/raw/dog_001/001.jpg",
    )
    assert emb_id == "emb-uuid"


def test_find_matching_pet_returns_result(mock_supabase_service):
    mock_supabase_service.client.rpc.return_value.execute.return_value.data = [
        {"pet_id": "matched-uuid", "embedding_id": "emb-uuid", "similarity": 0.92}
    ]
    embedding = np.random.rand(256).astype(np.float32)
    result = mock_supabase_service.find_matching_pet(embedding, threshold=0.85)
    assert result is not None
    assert result["pet_id"] == "matched-uuid"
    assert result["similarity"] == 0.92


def test_find_matching_pet_no_match(mock_supabase_service):
    mock_supabase_service.client.rpc.return_value.execute.return_value.data = []
    embedding = np.random.rand(256).astype(np.float32)
    result = mock_supabase_service.find_matching_pet(embedding)
    assert result is None
