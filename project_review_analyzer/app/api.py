"""HTTP layer: validate requests, call the model, serialize responses."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model import SentimentModel
from app.schemas import BatchRequest, BatchResponse, Prediction, ReviewRequest
from app.settings import MAX_BATCH_SIZE, MAX_TEXT_CHARS, MAX_TOKENS, MODEL_ID, MODEL_REVISION


def create_app(predictor=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        # Startup fails visibly if the model cannot download/load. Never fake predictions.
        application.state.predictor = predictor if predictor is not None else SentimentModel()
        yield

    application = FastAPI(
        title="Customer Review Analyzer",
        description="English positive/negative sentiment. Scores are model outputs, not measured reliability.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health():
        return {
            "status": "ready",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "max_batch_size": MAX_BATCH_SIZE,
            "max_text_chars": MAX_TEXT_CHARS,
            "max_tokens": MAX_TOKENS,
        }

    @application.post("/predict", response_model=Prediction)
    def predict(request: ReviewRequest):
        return application.state.predictor.predict([request.text])[0]

    @application.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest):
        return {"predictions": application.state.predictor.predict(request.texts)}

    return application


app = create_app()
