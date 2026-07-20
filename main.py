"""
main.py
=======
FastAPI service for Indian Currency Fake/Real Detection.

Run with:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /predict   — upload an image, get fake/real verdict
    GET  /health    — quick health check
    GET  /          — API info
"""

import os
import sys
import shutil
import tempfile

# Make sure the inference module is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "inference"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from currency_predictor import predict_image  # noqa: E402

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Currency Fraud Detector",
    description="Classifies Indian currency notes as real or fake using MobileNetV2.",
    version="1.0.0",
)

# ── CORS — allow all origins so the frontend can call this freely ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="API Info")
def root():
    return {
        "service": "Currency Fraud Detector",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload a currency note image to classify it as real or fake",
            "GET  /health":  "Health check",
        },
    }


@app.get("/health", summary="Health Check")
def health():
    return {"status": "ok"}


@app.post("/predict", summary="Classify a currency note image")
async def predict(file: UploadFile = File(...)):
    """
    Upload a JPG/PNG image of a currency note.

    Returns:
        - verdict    : "real" or "fake"
        - confidence : how confident the model is (0.0 – 1.0)
        - raw_score  : raw sigmoid output (closer to 1.0 = more likely real)
    """
    # Validate file type
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[-1].lower() if file.filename else ".jpg"
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )

    # Save to a temp file, run prediction, then clean up
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_image(tmp_path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)
