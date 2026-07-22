import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from heuristics.extract import extract_heuristics
from heuristics.merge import merge_verdict, heuristic_only_fallback
from llm.classify import classify_with_llm, LLMClassificationError
from llm.transcribe import transcribe_audio, TranscriptionError
from api.schemas import ScamInput, ScamPrediction

app = FastAPI(title="Scam Detection Service")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/scams", StaticFiles(directory=os.path.join(FRONTEND_DIR, "scams")), name="scams")


@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Allow the local frontend (opened as a file or served separately) to call this API.
# For the hackathon demo this is wide open; tighten allow_origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/scam", response_model=ScamPrediction)
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


@app.post("/predict/scam-audio", response_model=ScamPrediction)
async def predict_scam_audio(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Audio file must not be empty")
        
    try:
        transcript = transcribe_audio(file_bytes, file.filename)
    except TranscriptionError as e:
        raise HTTPException(status_code=502, detail=str(e))
        
    heuristics = extract_heuristics(transcript)
    
    try:
        llm_result = classify_with_llm(transcript, heuristics)
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
            "transcript": transcript,
            "heuristics": heuristics,
            "llm": llm_result,
            "llm_status": llm_status,
            "key_signals": final["key_signals"],
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
