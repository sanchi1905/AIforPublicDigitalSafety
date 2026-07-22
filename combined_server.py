"""
combined_server.py
==================
Single FastAPI server that exposes:
  POST /predict        — currency fake/real detection  (main.py)
  POST /predict/scam   — scam message detection        (api/main.py)
  GET  /               — serves frontend/index.html
  GET  /health         — health check

Run with:
    uvicorn combined_server:app --port 8000 --reload

Then tunnel with:
    python start_ngrok.py
"""

import os
import sys
import shutil
import tempfile

# ── path setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, "inference"))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from currency_predictor import predict_image          # inference/
from heuristics.extract import extract_heuristics
from heuristics.merge import merge_verdict, heuristic_only_fallback
from llm.classify import classify_with_llm, LLMClassificationError
from api.schemas import ScamInput, ScamPrediction

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Digital Public Safety — Detection Console",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ── frontend ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health Check")
def health():
    return {"status": "ok", "services": ["currency", "scam"]}


# ── currency endpoint ─────────────────────────────────────────────────────────
@app.post("/predict", summary="Classify a currency note image")
async def predict_currency(file: UploadFile = File(...)):
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[-1].lower() if file.filename else ".jpg"
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        return predict_image(tmp_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── scam endpoint ─────────────────────────────────────────────────────────────
@app.post("/predict/scam", response_model=ScamPrediction, summary="Classify a message for scam")
async def predict_scam(payload: ScamInput):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=422, detail="text field must not be empty")

    heuristics = extract_heuristics(payload.text, payload.urls)

    try:
        llm_result = classify_with_llm(payload.text, heuristics)
        final = merge_verdict(heuristics, llm_result)
        llm_status = "ok"
    except LLMClassificationError as e:
        final = heuristic_only_fallback(heuristics)
        llm_result = None
        llm_status = f"fallback: {e}"

    return ScamPrediction(
        module="scam",
        verdict=final["verdict"],
        confidence=final["confidence"],
        explanation=final["explanation"],
        raw_output={
            "heuristics": heuristics,
            "llm": llm_result,
            "llm_status": llm_status,
            "key_signals": final["key_signals"],
        },
    )
