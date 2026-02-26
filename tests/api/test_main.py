import pytest
from unittest.mock import patch, MagicMock


def test_health_check():
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_docs_available():
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200
