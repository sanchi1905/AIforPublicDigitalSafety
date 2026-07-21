import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from heuristics.extract import extract_heuristics
from heuristics.merge import merge_verdict, heuristic_only_fallback
from llm.classify import classify_with_llm, LLMClassificationError
from api.schemas import ScamInput, ScamPrediction

app = FastAPI(title="Scam Detection Service")

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


@app.get("/health")
async def health():
    return {"status": "ok"}
