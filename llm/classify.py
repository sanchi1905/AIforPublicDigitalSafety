"""
LLM-based scam classification via Groq.
Takes raw text + pre-extracted heuristic features, returns structured verdict.
"""
import os
import json
import httpx
from dotenv import load_dotenv

# Load .env file (if present) so GROQ_API_KEY is available via os.environ
load_dotenv()


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a scam detection assistant analyzing messages sent \
to users in India (SMS, WhatsApp, email). You will receive the raw message \
text and a list of heuristic risk factors already detected by a rules engine.

Weigh the heuristic factors but use your own judgment on the full context - \
heuristics can misfire (e.g. a real bank OTP message will also mention "OTP").

Return ONLY valid JSON, no markdown fences, no preamble, matching exactly:
{
  "verdict": "scam" | "suspicious" | "safe",
  "confidence": <float 0.0-1.0>,
  "explanation": "<one plain-language sentence a non-technical user can understand>",
  "key_signals": ["<short phrase>", ...]
}"""


class LLMClassificationError(Exception):
    pass


def classify_with_llm(text: str, heuristics: dict, timeout: float = 10.0) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMClassificationError("GROQ_API_KEY environment variable not set")

    user_content = json.dumps({
        "message_text": text,
        "heuristic_risk_score": heuristics["risk_score"],
        "heuristic_risk_factors": heuristics["risk_factors"],
    })

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = httpx.post(GROQ_API_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
    except httpx.TimeoutException as e:
        raise LLMClassificationError(f"Groq API timeout: {e}") from e
    except httpx.HTTPStatusError as e:
        raise LLMClassificationError(f"Groq API error: {e.response.status_code} {e.response.text}") from e
    except (KeyError, json.JSONDecodeError) as e:
        raise LLMClassificationError(f"Malformed LLM response: {e}") from e

    # basic shape validation
    required = {"verdict", "confidence", "explanation", "key_signals"}
    if not required.issubset(parsed):
        raise LLMClassificationError(f"LLM response missing fields: {parsed}")

    return parsed
