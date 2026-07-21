"""
Quick smoke test for the heuristic layer (no API key required).
Run: python tests/test_samples.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heuristics.extract import extract_heuristics

SAMPLES = [
    {
        "label": "scam - fake KYC update",
        "text": "Dear Customer, your SBI account will be BLOCKED within 24 hours. "
                "Verify your account immediately: http://sbi-kyc-update.xyz/verify",
    },
    {
        "label": "scam - lottery win",
        "text": "Congratulations! You've won Rs 25,00,000 in the Amazon lucky draw. "
                "Claim your prize now: http://bit.ly/amzn-claim9",
    },
    {
        "label": "scam - OTP phishing",
        "text": "URGENT: Share your OTP and UPI PIN to receive your refund of Rs 4999 immediately.",
    },
    {
        "label": "legit - real bank OTP",
        "text": "Your OTP for transaction of Rs 500 at Flipkart is 483920. Valid for 10 minutes. "
                "Do not share this OTP with anyone.",
    },
    {
        "label": "legit - delivery update",
        "text": "Your Flipkart order #FK102938 has been shipped and will arrive by Friday. "
                "Track at flipkart.com/track",
    },
]

if __name__ == "__main__":
    for sample in SAMPLES:
        result = extract_heuristics(sample["text"])
        print(f"\n--- {sample['label']} ---")
        print(f"risk_score: {result['risk_score']}")
        print(f"risk_factors: {result['risk_factors']}")
