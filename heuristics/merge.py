"""
Combines heuristic signals with the LLM verdict into a final result.
Also provides a heuristic-only fallback if the LLM call fails,
so a flaky API never crashes the demo.
"""

# If heuristics find this many distinct risk categories, we don't let
# the LLM fully override with a low-confidence "safe" verdict.
STRONG_SIGNAL_THRESHOLD = 2


def merge_verdict(heuristics: dict, llm_result: dict) -> dict:
    verdict = llm_result["verdict"]
    confidence = float(llm_result["confidence"])
    explanation = llm_result["explanation"]
    signals = list(llm_result.get("key_signals", []))

    if heuristics["risk_score"] >= STRONG_SIGNAL_THRESHOLD and verdict == "safe":
        verdict = "suspicious"
        confidence = max(confidence, 0.6)
        explanation = (
            f"{explanation} (Note: multiple risk patterns were also detected: "
            f"{', '.join(heuristics['risk_factors'][:3])})"
        )

    signals.extend(f"heuristic:{f}" for f in heuristics["risk_factors"] if f not in signals)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "explanation": explanation,
        "key_signals": signals,
    }


def heuristic_only_fallback(heuristics: dict) -> dict:
    """Used when the LLM call fails - degrade gracefully instead of erroring out."""
    score = heuristics["risk_score"]
    if score >= 3:
        verdict, confidence = "suspicious", 0.6
    elif score >= 1:
        verdict, confidence = "suspicious", 0.4
    else:
        verdict, confidence = "safe", 0.3  # low confidence - we have no LLM read

    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": (
            "LLM classification unavailable - verdict based on rule-based signals only."
        ),
        "key_signals": [f"heuristic:{f}" for f in heuristics["risk_factors"]],
    }
