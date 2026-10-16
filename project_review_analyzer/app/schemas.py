from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.settings import MAX_BATCH_SIZE, MAX_TEXT_CHARS

ReviewText = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)
]


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: ReviewText


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[ReviewText] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    label: Literal["positive", "negative"]
    score: float = Field(ge=0, le=1)
    truncated: bool


class BatchResponse(BaseModel):
    predictions: list[Prediction]
