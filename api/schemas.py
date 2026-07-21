from typing import List, Optional
from pydantic import BaseModel, Field


class ScamInput(BaseModel):
    text: str = Field(..., description="Message text to analyze")
    urls: Optional[List[str]] = Field(default=None, description="Optional extra URLs to check")


class ScamPrediction(BaseModel):
    module: str = "scam"
    verdict: str  # "scam" | "suspicious" | "safe"
    confidence: float
    explanation: str
    raw_output: dict
