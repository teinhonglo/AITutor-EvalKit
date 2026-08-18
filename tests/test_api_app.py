"""CPU-only functional tests for the FastAPI lifecycle and endpoints."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app

DIMENSIONS = (
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
)
SAMPLE_PATH = Path(__file__).with_name("sample_request.json")


class FakeInvalidPredictionError(RuntimeError):
    pass


class FakeEvaluator:
    initializations = 0
    evaluations = 0

    def __init__(self):
        type(self).initializations += 1

    def evaluate(self, conversation_history, tutor_response):
        type(self).evaluations += 1
        assert conversation_history
        assert tutor_response
        return {task: "Yes" for task in DIMENSIONS}


FakeEvaluator.invalid_prediction_error = FakeInvalidPredictionError


def sample_request():
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def test_health_evaluate_and_single_startup_initialization():
    FakeEvaluator.initializations = 0
    FakeEvaluator.evaluations = 0
    with TestClient(create_app(FakeEvaluator)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model_ready": True,
            "model": "google/gemma-2-2b-it",
            "evaluator": "LoMTL",
        }

        first = client.post("/evaluate", json=sample_request())
        second = client.post("/evaluate", json=sample_request())
        assert first.status_code == second.status_code == 200
        assert tuple(first.json()) == DIMENSIONS
        assert set(first.json().values()) == {"Yes"}
        assert FakeEvaluator.initializations == 1
        assert FakeEvaluator.evaluations == 2


def test_request_schema_rejects_empty_and_extra_fields():
    with TestClient(create_app(FakeEvaluator)) as client:
        assert client.post(
            "/evaluate", json={"conversation_history": "", "tutor_response": "response"}
        ).status_code == 422
        payload = sample_request() | {"model": "caller-must-not-select-this"}
        assert client.post("/evaluate", json=payload).status_code == 422


def test_invalid_generation_is_reported_as_bad_gateway():
    class InvalidEvaluator(FakeEvaluator):
        def evaluate(self, conversation_history, tutor_response):
            raise FakeInvalidPredictionError("invalid generated label")

    InvalidEvaluator.invalid_prediction_error = FakeInvalidPredictionError
    with TestClient(create_app(InvalidEvaluator)) as client:
        response = client.post("/evaluate", json=sample_request())
        assert response.status_code == 502
        assert "invalid generated label" in response.json()["detail"]


def test_unexpected_inference_error_is_reported_as_server_error():
    class BrokenEvaluator(FakeEvaluator):
        def evaluate(self, conversation_history, tutor_response):
            raise ValueError("internal detail must not leak")

    # Use a distinct type so ValueError follows the generic failure branch.
    BrokenEvaluator.invalid_prediction_error = FakeInvalidPredictionError
    with TestClient(create_app(BrokenEvaluator)) as client:
        response = client.post("/evaluate", json=sample_request())
        assert response.status_code == 500
        assert response.json() == {"detail": "LoMTL inference failed"}
