from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .constants import DEFAULT_MODEL_PATH
from .transformers import SentinelToNaN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_state: dict[str, object] = {}

def _register_legacy_pickling_shim() -> None:
    """Allow loading older joblib artifacts that reference __main__.SentinelToNaN."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "SentinelToNaN"):
        setattr(main_module, "SentinelToNaN", SentinelToNaN)


class ArticleFeatures(BaseModel):
    id: int = Field(..., description="Article identifier")
    year: Annotated[int, Field(ge=2000, le=2100)]
    word_count: Annotated[int, Field(ge=1)]
    caps_ratio: Annotated[float, Field(ge=0.0, le=1.0)]
    exclamation_count: Annotated[int, Field(ge=0)]
    source_credibility_rank: Annotated[int, Field(ge=1, le=100)]


class PredictionResponse(BaseModel):
    id: int
    prediction: int
    label: str
    prob_fake: float
    prob_real: float


class BatchRequest(BaseModel):
    articles: list[ArticleFeatures]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


def _to_frame(articles: list[ArticleFeatures]) -> pd.DataFrame:
    return pd.DataFrame([item.model_dump() for item in articles])


def _predict(df: pd.DataFrame) -> list[PredictionResponse]:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)

    result: list[PredictionResponse] = []
    for idx, row_id in enumerate(df["id"]):
        pred = int(predictions[idx])
        prob_real = float(round(probabilities[idx][0], 4))
        prob_fake = float(round(probabilities[idx][1], 4))
        result.append(
            PredictionResponse(
                id=int(row_id),
                prediction=pred,
                label="Fake" if pred == 1 else "Real",
                prob_fake=prob_fake,
                prob_real=prob_real,
            )
        )

    return result


def create_app(model_path: Path = DEFAULT_MODEL_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_file = Path(model_path)
        if not model_file.exists():
            raise RuntimeError(
                f"Model file not found at {model_file}. Run training first."
            )

        _register_legacy_pickling_shim()
        logger.info("Loading model from %s", model_file)
        _state["pipeline"] = joblib.load(model_file)
        logger.info("Model loaded")
        yield
        _state.clear()
        logger.info("Model state cleared")

    app = FastAPI(
        title="Fake News Detector API",
        version="1.0.0",
        description="Inference API for fake-news classification pipeline",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
 
    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(FRONTEND_DIR / "index.html")
 
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        if "pipeline" not in _state:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ok", "model": str(model_path)}

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(payload: ArticleFeatures) -> PredictionResponse:
        frame = _to_frame([payload])
        return _predict(frame)[0]

    @app.post("/predict_batch", response_model=BatchResponse, tags=["inference"])
    def predict_batch(payload: BatchRequest) -> BatchResponse:
        if not payload.articles:
            raise HTTPException(status_code=422, detail="Batch must not be empty")
        if len(payload.articles) > 1000:
            raise HTTPException(status_code=422, detail="Maximum batch size is 1000")

        frame = _to_frame(payload.articles)
        return BatchResponse(predictions=_predict(frame))

    return app


app = create_app()
