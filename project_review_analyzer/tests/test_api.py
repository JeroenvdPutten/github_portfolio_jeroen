"""Contract tests use a fake predictor: fast, offline, and independent of model accuracy."""
import pytest
from fastapi.testclient import TestClient

from app.api import create_app


class RecordingPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(texts)
        return [{"label": "positive" if "great" in text else "negative", "score": 0.9, "truncated": False} for text in texts]


@pytest.fixture
def service():
    model = RecordingPredictor()
    with TestClient(create_app(model)) as client:
        yield client, model


def test_health_reports_ready_and_limits(service):
    client, _ = service
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["max_batch_size"] == 100


def test_single_review_is_trimmed_and_forwarded(service):
    client, model = service
    response = client.post("/predict", json={"text": "  great headphones  "})
    assert response.status_code == 200
    assert response.json() == {"label": "positive", "score": 0.9, "truncated": False}
    assert model.calls == [["great headphones"]]


def test_batch_preserves_input_order(service):
    client, model = service
    response = client.post("/predict/batch", json={"texts": ["broken", "great", "broken"]})
    assert response.status_code == 200
    assert [item["label"] for item in response.json()["predictions"]] == ["negative", "positive", "negative"]
    assert model.calls == [["broken", "great", "broken"]]


@pytest.mark.parametrize("payload", [{}, {"text": " "}, {"text": "x" * 5001}, {"text": 123}, {"text": None}, {"text": "fine", "unexpected": 1}])
def test_invalid_review_never_reaches_model(service, payload):
    client, model = service
    assert client.post("/predict", json=payload).status_code == 422
    assert model.calls == []


@pytest.mark.parametrize("texts", [[], ["ok"] * 101, ["ok", " "], [42], ["x" * 5001]])
def test_invalid_batch_never_reaches_model(service, texts):
    client, model = service
    assert client.post("/predict/batch", json={"texts": texts}).status_code == 422
    assert model.calls == []


def test_maximum_batch_is_accepted(service):
    client, _ = service
    response = client.post("/predict/batch", json={"texts": ["great"] * 100})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 100
