"""KServe v1-style REST prediction endpoint for CONCH, TITAN, or Virchow2.

MODEL_NAME selects the model; each serving pod runs exactly one. Bare-bones
first iteration: no request-size or image-size validation.
"""
import base64
import io
import json
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from model import LOADERS, MODELS


def create_app(loader=None):
    model_name = os.getenv("MODEL_NAME", "conch")
    if model_name not in MODELS:
        raise SystemExit(f"MODEL_NAME must be one of {', '.join(MODELS)}")
    weights_dir = os.getenv("WEIGHTS_DIR", "/mnt/models")
    if loader is None:
        loader = lambda: LOADERS[model_name](weights_dir)
    gate = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        # Fail startup if loading or a real warm-up inference fails.
        app.state.model = await run_in_threadpool(loader)
        await run_in_threadpool(
            app.state.model.predict, app.state.model.warmup_payload()
        )
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.ready = False

    @app.get("/health/live")
    def live():
        return {"live": True}

    @app.get("/health/ready")
    def ready():
        if not app.state.ready:
            raise HTTPException(503, "Model not ready")
        return {"ready": True}

    def infer(encoded):
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "Model busy; retry later")
        try:
            model = app.state.model
            try:
                raw = base64.b64decode(encoded, validate=True)
                payload = raw
                if model.payload_kind == "image":
                    payload = Image.open(io.BytesIO(raw)).convert("RGB")
                return {"predictions": [model.predict(payload)]}
            except HTTPException:
                raise
            except Exception:
                logging.exception("Model inference failed")
                raise HTTPException(400, "Invalid payload or model inference failed")
        finally:
            gate.release()

    @app.post("/v1/models/{model_name}:predict")
    async def predict(model_name: str, request: Request):
        if not app.state.ready:
            raise HTTPException(503, "Model not ready")
        body = await request.body()
        try:
            encoded = json.loads(body)["instances"][0]["image"]["b64"]
            if not isinstance(encoded, str):
                raise ValueError()
        except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
            raise HTTPException(400, 'Expected one instances item with image.b64')
        return await run_in_threadpool(infer, encoded)

    @app.post("/v1/models/{model_name}:text")
    async def text(model_name: str, request: Request):
        """CONCH text embeddings: {"texts": ["...", "..."]} -> {"embeddings": [[...], ...]}."""
        if not app.state.ready:
            raise HTTPException(503, "Model not ready")
        model = app.state.model
        if not hasattr(model, "predict_text"):
            raise HTTPException(404, f"{model_name} has no text encoder")
        body = await request.body()
        try:
            texts = json.loads(body)["texts"]
            if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t for t in texts):
                raise ValueError()
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise HTTPException(400, 'Expected {"texts": ["...", ...]} with non-empty strings')
        return await run_in_threadpool(infer_text, texts)

    def infer_text(texts):
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "Model busy; retry later")
        try:
            try:
                return {"embeddings": app.state.model.predict_text(texts)}
            except Exception:
                logging.exception("Model text inference failed")
                raise HTTPException(400, "Text inference failed")
        finally:
            gate.release()

    return app


app = create_app()
