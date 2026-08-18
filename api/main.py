"""FastAPI application for the paper-best LoMTL evaluator."""

import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, status

from .schemas import EvaluationRequest, EvaluationResponse, HealthResponse

logger = logging.getLogger(__name__)
BASE_MODEL = "google/gemma-2-2b-it"


def create_app(evaluator_factory: Callable | None = None) -> FastAPI:
    """Build the API, optionally injecting a lightweight evaluator for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Delay ML imports until startup, keeping schema/app imports lightweight.
        if evaluator_factory is None:
            from .evaluator import AutoEvaluator, InvalidPredictionError

            factory = AutoEvaluator
            invalid_prediction_error = InvalidPredictionError
        else:
            factory = evaluator_factory
            invalid_prediction_error = getattr(
                evaluator_factory, "invalid_prediction_error", RuntimeError
            )
        # This runs exactly once per Uvicorn process before accepting requests.
        app.state.evaluator = factory()
        app.state.invalid_prediction_error = invalid_prediction_error
        app.state.model_ready = True
        yield
        app.state.model_ready = False

    application = FastAPI(
        title="AITutor-EvalKit LoMTL API",
        description="Paper-best LoMTL automated tutor evaluator.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request):
        if not getattr(request.app.state, "model_ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready",
            )
        return {
            "status": "ok",
            "model_ready": True,
            "model": BASE_MODEL,
            "evaluator": "LoMTL",
        }

    @application.post("/evaluate", response_model=EvaluationResponse)
    def evaluate(payload: EvaluationRequest, request: Request):
        evaluator = getattr(request.app.state, "evaluator", None)
        if evaluator is None or not getattr(request.app.state, "model_ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready",
            )
        try:
            return evaluator.evaluate(payload.conversation_history, payload.tutor_response)
        except request.app.state.invalid_prediction_error as exc:
            logger.error("Invalid model generation: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("LoMTL inference failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LoMTL inference failed",
            ) from exc

    return application


app = create_app()
