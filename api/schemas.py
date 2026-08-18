"""Strict request and response schemas for the LoMTL API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["Yes", "To some extent", "No"]


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    conversation_history: str = Field(min_length=1)
    tutor_response: str = Field(min_length=1)


class EvaluationResponse(BaseModel):
    Mistake_Identification: Label
    Mistake_Location: Label
    Providing_Guidance: Label
    Actionability: Label


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_ready: Literal[True]
    model: Literal["google/gemma-2-2b-it"]
    evaluator: Literal["LoMTL"]
