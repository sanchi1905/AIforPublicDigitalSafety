"""
LLM integration test — calls Groq API with the same samples used in test_samples.py.
Requires GROQ_API_KEY to be set in the environment.
Run: python tests/test_llm.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heuristics.extract import extract_heuristics
from llm.classify import classify_with_llm, LLMClassificationError

SAMPLES = [
    {
        "label": "scam - fake KYC update",
        "text": (
            "Dear Customer, your SBI account will be BLOCKED within 24 hours. "
            "Verify your account immediately: http://sbi-kyc-update.xyz/verify"
        ),
    },
    {
        "label": "scam - lottery win",
        "text": (
            "Congratulations! You've won Rs 25,00,000 in the Amazon lucky draw. "
            "Claim your prize now: http://bit.ly/amzn-claim9"
        ),
    },
    {
        "label": "scam - OTP phishing",
        "text": "URGENT: Share your OTP and UPI PIN to receive your refund of Rs 4999 immediately.",
    },
    {
        "label": "legit - real bank OTP",
        "text": (
            "Your OTP for transaction of Rs 500 at Flipkart is 483920. "
            "Valid for 10 minutes. Do not share this OTP with anyone."
        ),
    },
    {
        "label": "legit - delivery update",
        "text": (
            "Your Flipkart order #FK102938 has been shipped and will arrive by Friday. "
            "Track at flipkart.com/track"
        ),
    },
]

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"

def run():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set. Exiting.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  LLM Integration Test  (model: llama-3.3-70b-versatile)")
    print(f"{'='*60}\n")

    errors = 0
    for sample in SAMPLES:
        label = sample["label"]
        text  = sample["text"]
        try:
            heuristics = extract_heuristics(text)
            result     = classify_with_llm(text, heuristics)

            verdict     = result["verdict"]
            confidence  = result["confidence"]
            explanation = result["explanation"]
            signals     = result["key_signals"]

            # simple sanity: scam samples should not be "safe", legit should not be "scam"
            expected_safe  = label.startswith("legit")
            wrong = (expected_safe and verdict == "scam") or \
                    (not expected_safe and verdict == "safe")

            status = FAIL if wrong else PASS
            if wrong:
                errors += 1

            print(f"[{status}] {label}")
            print(f"         verdict     : {verdict}  (confidence={confidence:.2f})")
            print(f"         explanation : {explanation}")
            print(f"         key_signals : {signals}")
            print(f"         heuristics  : score={heuristics['risk_score']} "
                  f"factors={heuristics['risk_factors']}\n")

        except LLMClassificationError as e:
            errors += 1
            print(f"[{FAIL}] {label}")
            print(f"         ERROR: {e}\n")

    print(f"{'='*60}")
    print(f"  Results: {len(SAMPLES) - errors}/{len(SAMPLES)} passed")
    print(f"{'='*60}\n")
    sys.exit(0 if errors == 0 else 1)

if __name__ == "__main__":
    run()
