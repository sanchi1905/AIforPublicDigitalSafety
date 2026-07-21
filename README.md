# Scam Detection Module

Hybrid heuristic + LLM pipeline that classifies SMS/WhatsApp-style messages
(and any URLs they contain) as `scam`, `suspicious`, or `safe`. Built as the
scam-detection half of the AI for Public Digital Safety hackathon project —
pairs with the [currency fraud detector](https://github.com/sanchi1905/AIforPublicDigitalSafety)
behind a shared gateway.

## How it works

1. **Heuristic pass** (`heuristics/extract.py`) — fast, free, explainable checks:
   URL shorteners, IP-based URLs, suspicious TLDs, lookalike domains against
   known Indian brands, urgency language, OTP/PIN/UPI requests, prize/lottery bait.
2. **LLM pass** (`llm/classify.py`) — sends the message + heuristic flags to
   Groq (Llama 3.3 70B) for contextual judgment and a human-readable explanation.
3. **Merge** (`heuristics/merge.py`) — if heuristics find 2+ strong signals but
   the LLM says "safe", the verdict is bumped to "suspicious" rather than
   silently trusting the LLM. If the LLM call fails, a heuristic-only fallback
   keeps the endpoint responsive instead of erroring out.

## Setup

```
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"   # https://console.groq.com
```

## Run

```
uvicorn api.main:app --reload --port 8001
```

## Test

```
python tests/test_samples.py       # heuristics only, no API key needed
```

## Example request

```
POST /predict/scam
{
  "text": "Your SBI account will be BLOCKED. Verify now: http://sbi-kyc.xyz/verify"
}
```

```json
{
  "module": "scam",
  "verdict": "scam",
  "confidence": 0.93,
  "explanation": "This message uses urgency and a suspicious lookalike link to try to steal your bank login.",
  "raw_output": { "...": "..." }
}
```
